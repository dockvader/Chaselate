"""Speech recognition via Voxtral Realtime (Mistral, Apache 2.0), through a locally-built
voxtral.cpp subprocess -- see CLAUDE.md for the full story of why this exists and how the
build was produced.

**Experimental. Not wired into the shipped app by default** (``AsrConfig.engine`` defaults to
``"whisper"``). This exists to let ``experiment/voxtral-asr`` actually run end to end with the
new engine, not as a production alternative -- see the CLAUDE.md landmine entry before
touching this file.

Unlike faster-whisper, voxtral.cpp has no Python bindings and no in-process library call
available from here. It is driven as a long-lived subprocess in ``--stdin`` interactive mode
(a small patch on top of the upstream CLI -- see CLAUDE.md): the model loads once, then each
utterance is transcribed by writing its audio to a temp WAV file and sending the path over the
subprocess's stdin, one path per line, reading back the transcribed text terminated by a
``__VOXTRAL_END__`` sentinel line. This avoids paying the ~1.3-1.5s model load cost per
utterance, the same reason :class:`~chaselate.asr.WhisperEngine` keeps its model warm between
calls.
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import threading
import time
import wave
from typing import Optional

import numpy as np

from .asr import AsrError, Transcript
from .config import AsrConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
#: How long to wait for the model to load and the __VOXTRAL_READY__ sentinel.
READY_TIMEOUT_S = 60.0
#: How long a single transcription may take before it is treated as a hang, not just a slow
#: GPU pass -- generous, since the CPU fallback path (or a cold JIT-compile on an unfamiliar
#: GPU architecture) can legitimately take tens of seconds. See CLAUDE.md for measured RTF.
TRANSCRIBE_TIMEOUT_S = 90.0

_READY_SENTINEL = "__VOXTRAL_READY__"
_END_SENTINEL = "__VOXTRAL_END__"


class VoxtralEngine:
    """Drives a persistent ``voxtral.exe --stdin`` subprocess.

    Same public shape as :class:`~chaselate.asr.WhisperEngine` (``load``/``unload``/
    ``warmup``/``transcribe``/``description``/``loaded``) so it drops into
    :class:`~chaselate.pipeline.Pipeline` in place of it. Not thread-safe by contract of the
    underlying subprocess's stdin/stdout protocol (one request must finish before the next is
    written), so :meth:`transcribe` holds a lock -- intended to be driven by a single worker
    thread anyway, same as WhisperEngine.
    """

    #: Read by Pipeline._transcribe to pick the cheap single-pass path (see
    #: pipeline.CHEAP_MAX_SILENCE_HOLDS) instead of the LocalAgreement hold buffer built for
    #: faster-whisper. That machinery re-transcribes its whole growing audio buffer on every
    #: new VAD segment to catch Whisper's specific "confident wrong terminator on truncated
    #: audio" failure mode; a live session showed it costs O(segments^2), which faster-whisper
    #: (10x+ Voxtral's throughput here) can absorb and Voxtral cannot -- captions fell further
    #: and further behind in testing. Also unneeded: Voxtral was never observed inventing a
    #: premature terminator in testing (if anything it undershoots -- see CLAUDE.md), so the
    #: expensive cross-pass verification is not buying anything for this engine.
    NEEDS_AUDIO_REVERIFY = False

    def __init__(self, config: Optional[AsrConfig] = None):
        self.config = config or AsrConfig()
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()

    # -- introspection -------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    @property
    def device(self) -> str:
        return self.config.voxtral_gpu

    @property
    def compute_type(self) -> str:
        return os.path.basename(self.config.voxtral_model)

    @property
    def description(self) -> str:
        if not self.loaded:
            return f"voxtral-realtime {self.compute_type} (not loaded)"
        return f"voxtral-realtime {self.compute_type} / {self.device}"

    # -- lifecycle -------------------------------------------------------------

    def load(self) -> None:
        with self._lock:
            if self.loaded:
                return
            cfg = self.config
            if not os.path.isfile(cfg.voxtral_exe):
                raise AsrError(f"voxtral.exe not found at {cfg.voxtral_exe}")
            if not os.path.isfile(cfg.voxtral_model):
                raise AsrError(f"voxtral model not found at {cfg.voxtral_model}")

            env = dict(os.environ)
            if cfg.voxtral_cuda_bin and os.path.isdir(cfg.voxtral_cuda_bin):
                # CUDA 13.x moved the runtime DLLs (cudart64_13.dll, cublas64_13.dll) into
                # bin\x64, not bin -- without this on PATH, voxtral.exe fails to even start
                # with STATUS_DLL_NOT_FOUND. See CLAUDE.md for how this was found.
                env["PATH"] = cfg.voxtral_cuda_bin + os.pathsep + env.get("PATH", "")

            log.info("starting voxtral subprocess: %s", cfg.voxtral_exe)
            try:
                self._proc = subprocess.Popen(
                    [
                        cfg.voxtral_exe,
                        "--model", cfg.voxtral_model,
                        "--gpu", cfg.voxtral_gpu,
                        "--log-level", "warn",
                        "--stdin",
                    ],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )
            except OSError as exc:
                raise AsrError(f"failed to start voxtral.exe: {exc}") from exc

            deadline = time.time() + READY_TIMEOUT_S
            while time.time() < deadline:
                line = self._proc.stdout.readline()
                if not line:
                    stderr = self._proc.stderr.read()
                    self._proc = None
                    raise AsrError(f"voxtral subprocess exited during startup: {stderr[-2000:]}")
                if line.strip() == _READY_SENTINEL:
                    log.info("voxtral ready")
                    return
            self.unload()
            raise AsrError(f"voxtral did not become ready within {READY_TIMEOUT_S}s")

    def unload(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None:
                return
            try:
                if proc.stdin:
                    proc.stdin.close()
                proc.wait(timeout=5.0)
            except Exception:  # noqa: BLE001
                proc.kill()

    def warmup(self) -> float:
        started = time.time()
        self.load()
        return time.time() - started

    # -- transcription -----------------------------------------------------

    def transcribe(
        self, audio: np.ndarray, language: Optional[str] = None, prompt_hint: str = ""
    ) -> Transcript:
        """Transcribe 16 kHz mono float32 audio.

        ``prompt_hint`` is accepted for interface compatibility with
        :meth:`~chaselate.asr.WhisperEngine.transcribe` but ignored -- voxtral.cpp's
        ``--prompt`` is documented as "currently ignored for realtime mode" upstream.
        """
        audio = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
        duration = audio.size / float(SAMPLE_RATE)
        if audio.size == 0:
            return Transcript(text="", duration=0.0)

        with self._lock:
            self.load()
            proc = self._proc
            if proc is None:
                raise AsrError("voxtral subprocess not running")

            tmp_path = self._write_wav(audio)
            started = time.time()
            try:
                text = self._request(proc, tmp_path)
            finally:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
            elapsed = time.time() - started

        return Transcript(text=text, language=language or "", duration=duration, elapsed=elapsed)

    def _write_wav(self, audio: np.ndarray) -> str:
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="chaselate_voxtral_")
        os.close(fd)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm.tobytes())
        return path

    def _request(self, proc: subprocess.Popen, wav_path: str) -> str:
        try:
            proc.stdin.write(wav_path + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            self.unload()
            raise AsrError(f"voxtral subprocess died: {exc}") from exc

        lines = []
        deadline = time.time() + TRANSCRIBE_TIMEOUT_S
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                stderr = proc.stderr.read() if proc.stderr else ""
                self.unload()
                raise AsrError(f"voxtral subprocess died mid-transcription: {stderr[-2000:]}")
            line = line.rstrip("\n")
            if line == _END_SENTINEL:
                text = "\n".join(lines).strip()
                if text.startswith("[error]") or text == "[no-transcript]":
                    return ""
                return text
            lines.append(line)

        self.unload()
        raise AsrError(f"voxtral transcription timed out after {TRANSCRIBE_TIMEOUT_S}s")
