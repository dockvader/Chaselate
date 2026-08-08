"""Speech recognition via faster-whisper (CTranslate2).

The macOS original used MLX Whisper, which only exists for Apple silicon. faster-whisper is
the Windows equivalent: same weights, CTranslate2 backend, CUDA or CPU.

Two Windows-specific problems are solved here.

**Finding cuBLAS.** CTranslate2 loads ``cublas64_12.dll`` and cuDNN with a plain
``LoadLibrary`` from inside its own native DLL. When those libraries come from the
``nvidia-*-cu12`` pip wheels rather than a system CUDA install, that call fails with
"Library cublas64_12.dll is not found" -- and ``os.add_dll_directory`` does *not* fix it,
because the Windows loader only consults those directories for modules it resolves itself.
Prepending the wheel directories to ``PATH`` does work, since plain ``LoadLibrary`` searches
``PATH``. :func:`ensure_cuda_libraries` must therefore run before the first GPU operation.

**First-call latency.** On a GPU newer than the CUDA kernels shipped in the wheel (an
RTX 50-series board, for instance), the driver JIT-compiles kernels on the first inference:
one call can take ~15 s before the cache is warm, then drops to a fraction of realtime.
:meth:`WhisperEngine.warmup` gets that out of the way while the UI can still say so.
"""

from __future__ import annotations

import glob
import importlib.util
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import AsrConfig
from .languages import whisper_code
from .textutils import collapse_repeats, is_hallucination, normalize_ws

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_cuda_libs_ready: Optional[bool] = None
_cuda_lock = threading.Lock()


def ensure_cuda_libraries() -> bool:
    """Put the ``nvidia-*-cu12`` wheel DLL directories on ``PATH``. Idempotent.

    Returns True when at least one directory containing ``cublas64_*.dll`` was found (or
    the DLL is already reachable). A False result is not fatal -- it just means the CPU
    backend is the only option.
    """
    global _cuda_libs_ready
    with _cuda_lock:
        if _cuda_libs_ready is not None:
            return _cuda_libs_ready

        found_cublas = False
        added: List[str] = []
        spec = importlib.util.find_spec("nvidia")
        roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []

        # A frozen build (PyInstaller) has no "nvidia" pip package for find_spec to locate --
        # find_spec looks for an installed Python package, and a frozen app's CUDA support is
        # just DLLs the installer dropped next to the executable (see packaging/installer.nsi's
        # optional CUDA component), not a package. Check the same nvidia/<lib>/bin layout next
        # to the executable so ensure_cuda_libraries works the same way whether the process is
        # a normal venv run or a frozen one.
        if getattr(sys, "frozen", False):
            frozen_nvidia_dir = os.path.join(os.path.dirname(sys.executable), "nvidia")
            if os.path.isdir(frozen_nvidia_dir):
                roots.append(frozen_nvidia_dir)

        for root in roots:
            for pattern in ("*/bin", "*/lib"):
                for path in glob.glob(os.path.join(root, pattern)):
                    if not os.path.isdir(path):
                        continue
                    dlls = glob.glob(os.path.join(path, "*.dll"))
                    if not dlls:
                        continue
                    added.append(path)
                    if any(
                        os.path.basename(d).lower().startswith("cublas64") for d in dlls
                    ):
                        found_cublas = True
                    # add_dll_directory does not help CTranslate2 itself, but it does help
                    # any Python extension that loads these libraries properly.
                    try:
                        os.add_dll_directory(path)
                    except (OSError, AttributeError):
                        pass

        if added:
            existing = os.environ.get("PATH", "")
            # Only prepend directories not already present, so repeated calls in one
            # process cannot grow PATH without bound.
            current = {p.rstrip("\\/").casefold() for p in existing.split(os.pathsep) if p}
            fresh = [p for p in added if p.rstrip("\\/").casefold() not in current]
            if fresh:
                os.environ["PATH"] = os.pathsep.join(fresh) + os.pathsep + existing
            log.info("added %d CUDA library directories to PATH", len(fresh))

        if not found_cublas:
            # A system CUDA toolkit satisfies the same requirement.
            for path in os.environ.get("PATH", "").split(os.pathsep):
                if path and glob.glob(os.path.join(path, "cublas64_*.dll")):
                    found_cublas = True
                    break

        _cuda_libs_ready = found_cublas
        return found_cublas


_preloaded: Optional[bool] = None


def preload_native_libraries() -> bool:
    """Import the native ML extensions **before PyQt5 is imported**. Returns success.

    This is load-bearing, not an optimisation. PyQt5 ships its own copies of the Visual C++
    runtime in ``PyQt5/Qt5/bin`` (``msvcp140.dll``, ``vcruntime140.dll``, ``concrt140.dll``)
    and puts that directory on the DLL search path when it is imported. They are older than
    the system copies. CTranslate2 bundles Intel OpenMP (``libiomp5md.dll``), which links
    against that runtime -- so if PyQt5 goes first, CTranslate2's OpenMP binds to PyQt5's
    older ``msvcp140.dll`` and the process dies with an access violation the moment a model
    is constructed. It is not thread- or CUDA-specific: importing PyQt5 first and then using
    CTranslate2 at all is enough to crash, on CPU as well as GPU, on the main thread as well
    as a worker.

    Importing CTranslate2 first makes Windows resolve ``msvcp140.dll`` from the system
    directory, and because the loader matches already-loaded modules by name, PyQt5's copy is
    then never loaded. Setting ``KMP_DUPLICATE_LIB_OK`` does *not* help; the order is the fix.

    Call this once, early, from a module that has not imported PyQt5. Idempotent, so callers
    that cannot see each other can both insist on it.
    """
    global _preloaded
    if _preloaded is not None:
        return _preloaded

    if "PyQt5.QtCore" in sys.modules:
        # PyQt5 going first is only dangerous if its bundled C++ runtime got loaded, which
        # chaselate._runtime prevents by pinning the System32 copies on package import. Only
        # complain when that did not happen, otherwise every normal startup logs a scary
        # error about a problem that is already handled.
        from ._runtime import was_pinned

        if not was_pinned():
            log.error(
                "PyQt5 was imported before the system C++ runtime could be pinned. "
                "CTranslate2 may now crash with an access violation. Import chaselate before "
                "PyQt5."
            )
        else:
            log.debug("PyQt5 imported first, but the system C++ runtime was already pinned")

    ensure_cuda_libraries()
    try:
        import ctranslate2  # noqa: F401
        import faster_whisper  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        log.error("could not import the speech recognition backend: %s", exc)
        _preloaded = False
        return False

    # onnxruntime (for Silero) belongs in the same batch: its DLLs have the same dependency.
    try:
        import onnxruntime  # noqa: F401

        from faster_whisper.vad import get_vad_model

        # Cached by faster-whisper, so the worker thread later gets this instance for free.
        get_vad_model()
    except Exception as exc:  # noqa: BLE001
        log.warning("voice activity detector unavailable: %s", exc)
    _preloaded = True
    return True


def cuda_available() -> bool:
    """True when CTranslate2 sees a GPU *and* the CUDA support libraries are reachable."""
    try:
        import ctranslate2
    except Exception as exc:  # noqa: BLE001
        log.debug("ctranslate2 import failed: %s", exc)
        return False
    try:
        if ctranslate2.get_cuda_device_count() <= 0:
            return False
    except Exception as exc:  # noqa: BLE001
        log.debug("get_cuda_device_count failed: %s", exc)
        return False
    return ensure_cuda_libraries()


def resolve_device(requested: str) -> str:
    """Map ``auto``/``cuda``/``cpu`` onto a device this machine can actually use."""
    requested = (requested or "auto").strip().lower()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if cuda_available():
            return "cuda"
        log.warning("CUDA requested but unavailable; using CPU")
        return "cpu"
    return "cuda" if cuda_available() else "cpu"


def resolve_compute_type(requested: str, device: str) -> str:
    """Pick a compute type the device supports, defaulting to the fastest sane option."""
    requested = (requested or "auto").strip().lower()
    if requested != "auto":
        return requested
    if device != "cuda":
        return "int8"
    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types("cuda")
    except Exception:  # noqa: BLE001
        return "float16"
    for candidate in ("float16", "bfloat16", "int8_float16", "float32"):
        if candidate in supported:
            return candidate
    return "float32"


@dataclass
class Transcript:
    """Result of transcribing one segment."""

    text: str
    language: str = ""
    language_probability: float = 0.0
    avg_logprob: float = 0.0
    no_speech_prob: float = 0.0
    #: Audio duration in seconds.
    duration: float = 0.0
    #: Wall-clock seconds spent transcribing.
    elapsed: float = 0.0
    #: Segments dropped by the confidence / hallucination filters.
    dropped: int = 0

    @property
    def empty(self) -> bool:
        return not self.text

    @property
    def rtf(self) -> float:
        """Realtime factor: below 1.0 means faster than the audio plays."""
        return self.elapsed / self.duration if self.duration > 0 else 0.0


class AsrError(RuntimeError):
    """Model could not be loaded or transcription failed unrecoverably."""


class WhisperEngine:
    """Lazily-loaded faster-whisper wrapper.

    Not thread-safe by contract of the underlying model, so :meth:`transcribe` holds a lock.
    Intended to be driven by a single worker thread anyway.
    """

    def __init__(self, config: Optional[AsrConfig] = None):
        self.config = config or AsrConfig()
        self._model = None
        self._lock = threading.RLock()
        self._device = ""
        self._compute_type = ""
        self._loaded_key: Tuple = ()

    # -- introspection -----------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    @property
    def compute_type(self) -> str:
        return self._compute_type

    @property
    def description(self) -> str:
        if not self.loaded:
            return f"{self.config.model} (not loaded)"
        return f"{self.config.model} / {self._device} / {self._compute_type}"

    def _key(self) -> Tuple:
        cfg = self.config
        return (cfg.model, cfg.device, cfg.compute_type, cfg.cpu_threads)

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load the model, downloading it on first use. Safe to call repeatedly."""
        with self._lock:
            if self._model is not None and self._loaded_key == self._key():
                return
            if self._model is not None:
                self.unload()

            device = resolve_device(self.config.device)
            compute_type = resolve_compute_type(self.config.compute_type, device)
            from faster_whisper import WhisperModel

            kwargs = {"device": device, "compute_type": compute_type}
            if device == "cpu" and self.config.cpu_threads > 0:
                kwargs["cpu_threads"] = int(self.config.cpu_threads)

            log.info("loading Whisper %s on %s/%s", self.config.model, device, compute_type)
            try:
                self._model = WhisperModel(self.config.model, **kwargs)
            except Exception as exc:  # noqa: BLE001
                if device == "cuda":
                    # A GPU that reports itself present can still fail to run, e.g. when
                    # cuDNN is missing. CPU is slower but always works.
                    log.warning("GPU model load failed (%s); retrying on CPU", exc)
                    compute_type = resolve_compute_type(self.config.compute_type, "cpu")
                    try:
                        self._model = WhisperModel(
                            self.config.model, device="cpu", compute_type=compute_type
                        )
                        device = "cpu"
                    except Exception as cpu_exc:  # noqa: BLE001
                        raise AsrError(f"could not load Whisper model: {cpu_exc}") from cpu_exc
                else:
                    raise AsrError(f"could not load Whisper model: {exc}") from exc

            self._device = device
            self._compute_type = compute_type
            self._loaded_key = self._key()
            log.info("Whisper ready: %s", self.description)

    def unload(self) -> None:
        with self._lock:
            self._model = None
            self._loaded_key = ()

    def warmup(self) -> float:
        """Run one throwaway inference so the first real utterance is not the slow one.

        Returns the seconds it took. On a GPU whose kernels need JIT compilation this can
        be 10-20 s; afterwards the driver cache makes it negligible.
        """
        self.load()
        started = time.time()
        silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
        try:
            with self._lock:
                segments, _ = self._model.transcribe(
                    silence, language="en", beam_size=1, vad_filter=False
                )
                list(segments)
        except Exception as exc:  # noqa: BLE001 - warmup failing is not fatal
            log.warning("warmup failed: %s", exc)
        elapsed = time.time() - started
        log.info("warmup took %.2fs", elapsed)
        return elapsed

    # -- transcription -----------------------------------------------------

    def transcribe(self, audio: np.ndarray, language: Optional[str] = None) -> Transcript:
        """Transcribe 16 kHz mono float32 audio.

        ``language`` overrides the configured source language; ``None``/``auto`` lets
        Whisper detect. Confidence filtering and hallucination rejection are applied, so an
        empty ``text`` means "nothing worth showing", not necessarily an error.
        """
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.size / float(SAMPLE_RATE)
        if audio.size == 0:
            return Transcript(text="", duration=0.0)

        self.load()
        cfg = self.config
        lang_code = whisper_code(language if language is not None else cfg.source_lang)

        started = time.time()
        try:
            with self._lock:
                segments, info = self._model.transcribe(
                    audio,
                    language=lang_code,
                    beam_size=max(1, int(cfg.beam_size)),
                    vad_filter=bool(cfg.vad_filter),
                    condition_on_previous_text=bool(cfg.condition_on_previous_text),
                    initial_prompt=cfg.initial_prompt or None,
                    word_timestamps=False,
                )
                collected = list(segments)
        except Exception as exc:  # noqa: BLE001
            raise AsrError(f"transcription failed: {exc}") from exc
        elapsed = time.time() - started

        kept: List[str] = []
        logprobs: List[float] = []
        no_speech: List[float] = []
        dropped = 0
        detected = getattr(info, "language", "") or (lang_code or "")

        for seg in collected:
            text = normalize_ws(seg.text)
            if not text:
                continue
            avg_logprob = float(getattr(seg, "avg_logprob", 0.0) or 0.0)
            nsp = float(getattr(seg, "no_speech_prob", 0.0) or 0.0)
            if avg_logprob < cfg.min_logprob or nsp > cfg.max_no_speech_prob:
                log.debug(
                    "dropping low-confidence segment (logprob %.2f, no_speech %.2f): %r",
                    avg_logprob, nsp, text,
                )
                dropped += 1
                continue
            if cfg.filter_hallucinations and is_hallucination(text, detected):
                log.debug("dropping hallucination: %r", text)
                dropped += 1
                continue
            kept.append(text)
            logprobs.append(avg_logprob)
            no_speech.append(nsp)

        joined = normalize_ws(" ".join(kept))
        if joined:
            joined = collapse_repeats(joined, detected)
            # collapse_repeats can turn "thank you thank you thank you" into a phrase the
            # filter recognises, so re-check.
            if cfg.filter_hallucinations and is_hallucination(joined, detected):
                dropped += 1
                joined = ""

        return Transcript(
            text=joined,
            language=detected,
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            avg_logprob=float(np.mean(logprobs)) if logprobs else 0.0,
            no_speech_prob=float(np.mean(no_speech)) if no_speech else 0.0,
            duration=duration,
            elapsed=elapsed,
            dropped=dropped,
        )
