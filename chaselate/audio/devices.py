"""Device enumeration across the two viable Windows loopback backends.

Windows will not let an ordinary process record the speaker mix through the normal input
API, so the macOS original needed a virtual cable (BlackHole). On Windows the supported
route is WASAPI loopback, which both backends here expose:

* ``soundcard`` -- pure-Python/cffi, no build step, uniform API for mic and loopback.
* ``pyaudiowpatch`` -- a PyAudio fork with explicit loopback support; the fallback for
  machines where soundcard's COM enumeration trips over a driver.

Both are optional imports. Whichever is installed gets used; if both are, ``soundcard``
wins because it needs no native toolchain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional

log = logging.getLogger(__name__)

BACKEND_SOUNDCARD = "soundcard"
BACKEND_PYAUDIO = "pyaudiowpatch"
#: Preference order when the config says "auto".
BACKEND_PRIORITY = (BACKEND_SOUNDCARD, BACKEND_PYAUDIO)

KIND_LOOPBACK = "loopback"
KIND_MIC = "mic"

DEFAULT_RATE = 48000


@dataclass(frozen=True)
class DeviceInfo:
    """One capture endpoint, backend-agnostic."""

    name: str
    kind: str
    backend: str
    channels: int
    rate: int
    is_default: bool
    #: Backend handle: the soundcard device id, or the PyAudio device index as a string.
    raw_id: str

    @property
    def label(self) -> str:
        tag = "system audio" if self.kind == KIND_LOOPBACK else "microphone"
        suffix = " - default" if self.is_default else ""
        return f"{self.name} [{tag}{suffix}]"

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.label


def _import_soundcard():
    try:
        import soundcard  # type: ignore

        return soundcard
    except Exception as exc:  # noqa: BLE001 - any import problem means "unavailable"
        log.debug("soundcard unavailable: %s", exc)
        return None


def _import_pyaudio():
    try:
        import pyaudiowpatch  # type: ignore

        return pyaudiowpatch
    except Exception as exc:  # noqa: BLE001
        log.debug("pyaudiowpatch unavailable: %s", exc)
        return None


def available_backends() -> List[str]:
    """Backends importable in this interpreter, in preference order."""
    found = []
    if _import_soundcard() is not None:
        found.append(BACKEND_SOUNDCARD)
    if _import_pyaudio() is not None:
        found.append(BACKEND_PYAUDIO)
    return found


def pick_backend(requested: str = "auto") -> str:
    """Resolve a backend name, raising only when nothing at all is installed."""
    found = available_backends()
    if not found:
        raise RuntimeError(
            "No audio backend available. Install one of: "
            "pip install soundcard   (or)   pip install PyAudioWPatch"
        )
    if requested and requested != "auto":
        if requested in found:
            return requested
        log.warning("audio backend %r not available; falling back to %s", requested, found[0])
    for name in BACKEND_PRIORITY:
        if name in found:
            return name
    return found[0]


# -- soundcard ------------------------------------------------------------------


def _list_soundcard() -> List[DeviceInfo]:
    sc = _import_soundcard()
    if sc is None:
        return []
    devices: List[DeviceInfo] = []

    default_speaker_name = ""
    default_mic_name = ""
    try:
        default_speaker_name = sc.default_speaker().name
    except Exception as exc:  # noqa: BLE001
        log.debug("no default speaker: %s", exc)
    try:
        default_mic_name = sc.default_microphone().name
    except Exception as exc:  # noqa: BLE001
        log.debug("no default microphone: %s", exc)

    try:
        mics = sc.all_microphones(include_loopback=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("soundcard enumeration failed: %s", exc)
        return []

    for mic in mics:
        try:
            loopback = bool(getattr(mic, "isloopback", False))
            kind = KIND_LOOPBACK if loopback else KIND_MIC
            reference = default_speaker_name if loopback else default_mic_name
            devices.append(
                DeviceInfo(
                    name=mic.name,
                    kind=kind,
                    backend=BACKEND_SOUNDCARD,
                    channels=max(1, int(getattr(mic, "channels", 2) or 2)),
                    rate=DEFAULT_RATE,
                    is_default=bool(reference) and mic.name == reference,
                    raw_id=str(mic.id),
                )
            )
        except Exception as exc:  # noqa: BLE001 - skip a single bad endpoint
            log.debug("skipping soundcard device: %s", exc)
    return devices


# -- pyaudiowpatch ---------------------------------------------------------------


def _list_pyaudio() -> List[DeviceInfo]:
    pa = _import_pyaudio()
    if pa is None:
        return []
    devices: List[DeviceInfo] = []
    handle = None
    try:
        handle = pa.PyAudio()
        try:
            wasapi = handle.get_host_api_info_by_type(pa.paWASAPI)
        except Exception as exc:  # noqa: BLE001
            log.warning("WASAPI host API not present: %s", exc)
            return []

        default_out_index = wasapi.get("defaultOutputDevice", -1)
        default_in_index = wasapi.get("defaultInputDevice", -1)
        default_out_name = ""
        if default_out_index is not None and default_out_index >= 0:
            try:
                default_out_name = handle.get_device_info_by_index(default_out_index)["name"]
            except Exception:  # noqa: BLE001
                pass

        seen = set()
        try:
            for info in handle.get_loopback_device_info_generator():
                idx = int(info["index"])
                if idx in seen:
                    continue
                seen.add(idx)
                raw_name = str(info["name"])
                # PyAudioWPatch appends "[Loopback]" to the endpoint name.
                clean = raw_name.replace(" [Loopback]", "").strip()
                devices.append(
                    DeviceInfo(
                        name=clean,
                        kind=KIND_LOOPBACK,
                        backend=BACKEND_PYAUDIO,
                        channels=max(1, int(info.get("maxInputChannels") or 2)),
                        rate=int(info.get("defaultSampleRate") or DEFAULT_RATE),
                        is_default=bool(default_out_name) and clean == default_out_name.strip(),
                        raw_id=str(idx),
                    )
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("loopback enumeration failed: %s", exc)

        for i in range(int(wasapi.get("deviceCount") or 0)):
            try:
                info = handle.get_device_info_by_host_api_device_index(wasapi["index"], i)
            except Exception:  # noqa: BLE001
                continue
            if int(info.get("maxInputChannels") or 0) <= 0:
                continue
            if info.get("isLoopbackDevice"):
                continue
            idx = int(info["index"])
            if idx in seen:
                continue
            seen.add(idx)
            devices.append(
                DeviceInfo(
                    name=str(info["name"]),
                    kind=KIND_MIC,
                    backend=BACKEND_PYAUDIO,
                    channels=max(1, int(info.get("maxInputChannels") or 1)),
                    rate=int(info.get("defaultSampleRate") or DEFAULT_RATE),
                    is_default=idx == default_in_index,
                    raw_id=str(idx),
                )
            )
    except Exception as exc:  # noqa: BLE001
        log.warning("pyaudiowpatch enumeration failed: %s", exc)
    finally:
        if handle is not None:
            try:
                handle.terminate()
            except Exception:  # noqa: BLE001
                pass
    return devices


# -- public API ------------------------------------------------------------------


def list_devices(backend: str = "auto", kind: Optional[str] = None) -> List[DeviceInfo]:
    """Enumerate capture endpoints, loopback first then microphones.

    Never raises for a missing backend; an empty list means nothing is usable.
    """
    try:
        chosen = pick_backend(backend)
    except RuntimeError as exc:
        log.warning("%s", exc)
        return []

    devices = _list_soundcard() if chosen == BACKEND_SOUNDCARD else _list_pyaudio()
    if not devices and chosen == BACKEND_SOUNDCARD:
        log.info("soundcard returned no devices; trying pyaudiowpatch")
        devices = _list_pyaudio()
    elif not devices and chosen == BACKEND_PYAUDIO:
        log.info("pyaudiowpatch returned no devices; trying soundcard")
        devices = _list_soundcard()

    if kind:
        devices = [d for d in devices if d.kind == kind]
    # Default first, loopback before mic, then alphabetical - matches how the combo box
    # should read.
    devices.sort(key=lambda d: (not d.is_default, d.kind != KIND_LOOPBACK, d.name.casefold()))
    return devices


def default_device(backend: str = "auto", source: str = KIND_LOOPBACK) -> Optional[DeviceInfo]:
    """The endpoint to use when the user has not chosen one."""
    devices = list_devices(backend, kind=source)
    if not devices:
        return None
    for dev in devices:
        if dev.is_default:
            return dev
    return devices[0]


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.casefold(), b.casefold()).ratio()


def resolve_device(
    name: str = "",
    source: str = KIND_LOOPBACK,
    backend: str = "auto",
) -> Optional[DeviceInfo]:
    """Find the configured device, degrading gracefully rather than failing.

    Device names change when Windows renames an endpoint or the user swaps headsets, so an
    exact-match-only lookup would strand the app on a device that no longer exists. Exact
    match wins, then a close fuzzy match, then the system default.
    """
    devices = list_devices(backend, kind=source)
    if not devices:
        return None
    if not name:
        return default_device(backend, source)

    for dev in devices:
        if dev.name == name or dev.raw_id == name:
            return dev
    for dev in devices:
        if dev.name.casefold() == name.casefold():
            return dev

    scored = sorted(devices, key=lambda d: _similarity(d.name, name), reverse=True)
    best = scored[0]
    score = _similarity(best.name, name)
    if score >= 0.7:
        log.info("device %r not found; using closest match %r (%.2f)", name, best.name, score)
        return best

    fallback = default_device(backend, source)
    log.warning(
        "device %r not found and no close match; falling back to %s",
        name,
        fallback.name if fallback else "nothing",
    )
    return fallback
