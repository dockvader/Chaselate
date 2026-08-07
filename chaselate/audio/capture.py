"""The capture thread: WASAPI endpoint -> 16 kHz mono float32 blocks.

One thread owns the audio device for its whole lifetime. That is not just tidiness: WASAPI
clients are COM objects with thread affinity, so opening a stream on one thread and reading
it from another is undefined behaviour. Everything the backend touches therefore happens
inside :meth:`AudioCapture._run`, and the only thing crossing the thread boundary is a
numpy array handed to the ``on_audio`` callback.

The callback runs on the capture thread and must not block -- anything slow (VAD, ASR)
belongs behind a queue.
"""

from __future__ import annotations

import logging
import threading
import time
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

from ..config import AudioConfig
from .devices import (
    BACKEND_PYAUDIO,
    BACKEND_SOUNDCARD,
    KIND_LOOPBACK,
    DeviceInfo,
    resolve_device,
)
from .resample import TARGET_RATE, StreamResampler, downmix_mono

log = logging.getLogger(__name__)

#: Seconds of audio requested from the device per read. Small enough that captions feel
#: live, large enough that we are not paying COM overhead thousands of times a second.
READ_SECONDS = 0.032

class CaptureError(RuntimeError):
    """Raised when the audio device cannot be opened or dies mid-stream."""


# Deliberately *not* calling CoInitializeEx here.
#
# The obvious defensive move is to join the capture thread to a COM apartment before touching
# WASAPI. Doing so breaks soundcard: its own CoInitializeEx then returns S_FALSE ("already
# initialised on this thread"), which soundcard treats as an error because it only tolerates
# RPC_E_CHANGED_MODE. Its device enumeration raises, its COM wrapper is left half-constructed,
# and the visible result is a stream of AttributeErrors from a __del__ plus a silent demotion
# to the fallback backend.
#
# Both backends initialise COM themselves on whatever thread they are used from, so the
# correct thing to do is stay out of the way.


class _Stream(ABC):
    """Uniform read interface over the backend-specific capture objects."""

    rate: int = TARGET_RATE
    channels: int = 1
    #: Times the device reported dropping samples because we read too late.
    discontinuities: int = 0

    @abstractmethod
    def read(self, frames: int) -> np.ndarray:
        """Block until ``frames`` frames are available; return ``(frames, channels)``."""

    @abstractmethod
    def close(self) -> None: ...


class _SoundcardStream(_Stream):
    def __init__(self, device: DeviceInfo, rate: int, blocksize: int):
        import soundcard as sc

        mic = None
        try:
            mic = sc.get_microphone(id=device.raw_id, include_loopback=True)
        except Exception as exc:  # noqa: BLE001 - ids can go stale; fall back to name
            log.debug("soundcard id lookup failed (%s); matching by name", exc)
        if mic is None:
            for candidate in sc.all_microphones(include_loopback=True):
                if candidate.name == device.name:
                    mic = candidate
                    break
        if mic is None:
            raise CaptureError(f"audio device not found: {device.name}")

        self.channels = max(1, min(int(device.channels or 2), 8))
        self.rate = int(rate)
        self._ctx = mic.recorder(
            samplerate=self.rate, channels=self.channels, blocksize=blocksize
        )
        try:
            self._rec = self._ctx.__enter__()
        except Exception as exc:  # noqa: BLE001
            raise CaptureError(f"could not open {device.name}: {exc}") from exc

    #: Incremented whenever WASAPI reports it had to drop samples before we collected them.
    discontinuities = 0

    def read(self, frames: int) -> np.ndarray:
        # soundcard raises a Python warning per glitch, which floods stderr and says nothing
        # actionable on its own. Count them instead: a rising count means the consumer is too
        # slow, which the metrics display can show alongside the queue depths.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            data = self._rec.record(numframes=frames)
        for entry in caught:
            if "discontinuity" in str(entry.message):
                self.discontinuities += 1
            else:
                warnings.warn_explicit(
                    entry.message, entry.category, entry.filename, entry.lineno
                )
        return np.asarray(data, dtype=np.float32)

    def close(self) -> None:
        try:
            self._ctx.__exit__(None, None, None)
        except Exception as exc:  # noqa: BLE001 - closing must never raise
            log.debug("error closing soundcard stream: %s", exc)


class _PyAudioStream(_Stream):
    def __init__(self, device: DeviceInfo, rate: int, blocksize: int):
        import pyaudiowpatch as pyaudio

        self._pa = pyaudio.PyAudio()
        self.channels = max(1, min(int(device.channels or 2), 8))
        self.rate = int(rate)
        self._blocksize = blocksize
        try:
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.rate,
                input=True,
                input_device_index=int(device.raw_id),
                frames_per_buffer=blocksize,
            )
        except Exception as exc:  # noqa: BLE001
            try:
                self._pa.terminate()
            except Exception:  # noqa: BLE001
                pass
            raise CaptureError(f"could not open {device.name}: {exc}") from exc

    def read(self, frames: int) -> np.ndarray:
        raw = self._stream.read(frames, exception_on_overflow=False)
        flat = np.frombuffer(raw, dtype=np.float32)
        if self.channels > 1:
            usable = (flat.size // self.channels) * self.channels
            return flat[:usable].reshape(-1, self.channels)
        return flat.reshape(-1, 1)

    def close(self) -> None:
        for closer in (
            lambda: self._stream.stop_stream(),
            lambda: self._stream.close(),
            lambda: self._pa.terminate(),
        ):
            try:
                closer()
            except Exception as exc:  # noqa: BLE001
                log.debug("error closing pyaudio stream: %s", exc)


def _open_stream(device: DeviceInfo, rate: int, blocksize: int) -> _Stream:
    if device.backend == BACKEND_SOUNDCARD:
        return _SoundcardStream(device, rate, blocksize)
    if device.backend == BACKEND_PYAUDIO:
        return _PyAudioStream(device, rate, blocksize)
    raise CaptureError(f"unknown audio backend: {device.backend}")


class AudioCapture:
    """Background capture of system audio (or a mic) as 16 kHz mono float32.

    ``on_audio`` is called on the capture thread with contiguous blocks of exactly
    ``config.block_frames`` samples, in order, with no gaps -- silence included, because
    the VAD needs to see silence to know an utterance ended.
    """

    #: Consecutive read failures tolerated before giving up on the device.
    MAX_CONSECUTIVE_ERRORS = 5

    def __init__(
        self,
        config: AudioConfig,
        on_audio: Callable[[np.ndarray], None],
        on_error: Optional[Callable[[str], None]] = None,
        on_level: Optional[Callable[[float], None]] = None,
        on_started: Optional[Callable[[DeviceInfo], None]] = None,
    ):
        self._config = config
        self._on_audio = on_audio
        self._on_error = on_error
        self._on_level = on_level
        self._on_started = on_started

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._device: Optional[DeviceInfo] = None
        self._lock = threading.Lock()
        self._stream: Optional[_Stream] = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive() and not self._stop.is_set())

    @property
    def device(self) -> Optional[DeviceInfo]:
        return self._device

    @property
    def discontinuities(self) -> int:
        """Times the device dropped samples because we collected them too late.

        Non-zero means audio was genuinely lost, usually because something on the consumer
        side (or the GIL) held the capture thread up. Worth showing the user rather than
        hiding, since it degrades recognition.
        """
        stream = self._stream
        return stream.discontinuities if stream is not None else 0

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name="chaselate-capture", daemon=True
            )
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout)
        self._thread = None

    # -- worker ------------------------------------------------------------

    def _fail(self, message: str) -> None:
        log.error("%s", message)
        if self._on_error:
            try:
                self._on_error(message)
            except Exception:  # noqa: BLE001 - a bad callback must not kill the thread
                log.exception("on_error callback raised")

    def _run(self) -> None:
        stream: Optional[_Stream] = None
        try:
            device = resolve_device(
                self._config.device_name, self._config.source, self._config.backend
            )
            if device is None:
                kind = (
                    "system audio (loopback)"
                    if self._config.source == KIND_LOOPBACK
                    else "microphone"
                )
                self._fail(
                    f"No {kind} device found. Check that an output device is enabled in "
                    "Windows sound settings."
                )
                return
            self._device = device

            rate = int(self._config.capture_rate) or int(device.rate)
            read_frames = max(64, int(rate * READ_SECONDS))
            try:
                stream = _open_stream(device, rate, read_frames)
            except CaptureError as exc:
                self._fail(str(exc))
                return
            self._stream = stream

            log.info(
                "capturing %s via %s at %d Hz, %d ch",
                device.name,
                device.backend,
                stream.rate,
                stream.channels,
            )
            if self._on_started:
                try:
                    self._on_started(device)
                except Exception:  # noqa: BLE001
                    log.exception("on_started callback raised")

            self._pump(stream, read_frames)
        except Exception as exc:  # noqa: BLE001 - last resort so the thread reports why
            self._fail(f"Audio capture stopped: {exc}")
            log.exception("capture thread crashed")
        finally:
            if stream is not None:
                stream.close()

    def _pump(self, stream: _Stream, read_frames: int) -> None:
        resampler = StreamResampler(stream.rate, TARGET_RATE)
        block_frames = max(160, int(self._config.block_frames))
        gain = float(self._config.gain)
        # Carry samples that do not fill a whole block over to the next iteration so the
        # consumer always sees the same block size.
        carry = np.zeros(0, dtype=np.float32)
        errors = 0

        while not self._stop.is_set():
            try:
                raw = stream.read(read_frames)
                errors = 0
            except Exception as exc:  # noqa: BLE001
                errors += 1
                log.warning("audio read failed (%d/%d): %s", errors, self.MAX_CONSECUTIVE_ERRORS, exc)
                if errors >= self.MAX_CONSECUTIVE_ERRORS:
                    self._fail(f"Audio device stopped responding: {exc}")
                    return
                time.sleep(0.05)
                continue

            if raw is None or raw.size == 0:
                continue

            mono = downmix_mono(raw)
            if gain != 1.0:
                mono = mono * gain
            # Clip only after gain; WASAPI float samples can legitimately exceed 1.0.
            np.clip(mono, -4.0, 4.0, out=mono)

            resampled = resampler.process(mono)
            if resampled.size == 0:
                continue

            if self._on_level is not None:
                try:
                    self._on_level(float(np.sqrt(np.mean(resampled**2))))
                except Exception:  # noqa: BLE001
                    log.exception("on_level callback raised")

            carry = np.concatenate((carry, resampled)) if carry.size else resampled
            count = carry.size // block_frames
            if count:
                usable = count * block_frames
                blocks = carry[:usable].reshape(count, block_frames)
                carry = carry[usable:].copy()
                for block in blocks:
                    if self._stop.is_set():
                        return
                    try:
                        self._on_audio(np.ascontiguousarray(block))
                    except Exception:  # noqa: BLE001
                        log.exception("on_audio callback raised")
