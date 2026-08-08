"""Persistent settings.

One nested dataclass tree, serialised to ``%APPDATA%\\Chaselate\\config.json``. Loading
is deliberately forgiving: unknown keys are dropped and missing keys keep their default,
so a config written by an older build never blocks startup.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

APP_DIR_NAME = "Chaselate"
CONFIG_FILENAME = "config.json"

WHISPER_MODELS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
]

ASR_DEVICES = ["auto", "cuda", "cpu"]
COMPUTE_TYPES = ["auto", "float16", "int8_float16", "bfloat16", "float32", "int8"]
AUDIO_BACKENDS = ["auto", "soundcard", "pyaudiowpatch"]
LAYOUTS = ["side", "stacked"]


def config_dir() -> Path:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return Path(base) / APP_DIR_NAME


def config_path() -> Path:
    return config_dir() / CONFIG_FILENAME


@dataclass
class AudioConfig:
    backend: str = "auto"
    #: Empty means "system default output". Matched loosely against device names so a
    #: renamed or reconnected device degrades to the default instead of failing.
    device_name: str = ""
    #: "loopback" captures what the speakers play; "mic" captures an input device.
    source: str = "loopback"
    #: 0 = ask the backend for the device's native rate.
    capture_rate: int = 0
    #: Frames of 16 kHz mono audio handed to the segmenter at a time (512 = 32 ms).
    block_frames: int = 512
    #: Drop audio rather than grow the queue without bound when ASR falls behind.
    queue_seconds: float = 30.0
    gain: float = 1.0


@dataclass
class VadConfig:
    #: Silero speech probability above which a frame counts as speech.
    threshold: float = 0.5
    min_speech_ms: int = 250
    #: Trailing silence that closes a segment. Lower = snappier but choppier -- and every
    #: closure force-flushes whatever text is pending as a "sentence" even without a
    #: terminator (see pipeline.py's _process_asr_result), so too low a value here reads as
    #: fragmented, disconnected captions rather than just choppy timing. 300ms undercuts the
    #: length of an ordinary clause-boundary pause in continuous speech (Japanese especially,
    #: with its short pauses after は/けど/、); 500ms leaves more room for a real sentence to
    #: finish before being cut.
    min_silence_ms: int = 500
    #: Audio kept either side of detected speech so consonants are not clipped.
    speech_pad_ms: int = 200
    #: Hard cut for someone who never pauses, so captions keep flowing.
    max_segment_s: float = 14.0
    #: Below this RMS the block is treated as silence without invoking Silero.
    silence_rms: float = 0.0009


@dataclass
class AsrConfig:
    model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    #: "auto" lets Whisper detect per segment; pinning a code is far more stable.
    source_lang: str = "auto"
    beam_size: int = 1
    #: Whisper's own VAD pass, on top of our segmenter. Cheap insurance.
    vad_filter: bool = True
    condition_on_previous_text: bool = False
    #: Segments whose mean logprob is below this are discarded as garbage.
    min_logprob: float = -1.0
    #: Segments with a higher no-speech probability are discarded.
    max_no_speech_prob: float = 0.7
    #: Reject transcriptions Whisper is known to invent over music or silence.
    filter_hallucinations: bool = True
    cpu_threads: int = 0
    initial_prompt: str = ""

    # -- experimental: Voxtral Realtime engine (see chaselate/voxtral_engine.py) -----------
    # Not a shipped feature -- an experiment/voxtral-asr branch spike to try Voxtral Realtime
    # (Mistral, Apache 2.0, native streaming architecture) as a drop-in replacement for
    # faster-whisper, evaluated after finding a real CUDA crash bug in the upstream project
    # (voxtral.cpp issue #4) and hand-patching it locally. See CLAUDE.md for the full story.
    #: "whisper" (default, production) or "voxtral" (experimental).
    engine: str = "whisper"
    #: Path to the patched voxtral.exe build (see CLAUDE.md for how it was built/patched).
    voxtral_exe: str = r"C:\voxtral\bin\voxtral.exe"
    voxtral_model: str = r"C:\voxtral\models\Q4_K_M.gguf"
    #: CUDA Toolkit's runtime DLL directory -- voxtral.exe needs this on PATH to find
    #: cudart64_13.dll/cublas64_13.dll (CUDA 13.x moved these into bin\x64, not bin).
    voxtral_cuda_bin: str = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
    voxtral_gpu: str = "cuda"


@dataclass
class TranslateConfig:
    enabled: bool = True
    base_url: str = "http://127.0.0.1:11434"
    model: str = "translategemma:latest"
    target_lang: str = "zh-TW"
    #: Previous sentence pairs given to the model as context. 0 disables context. Higher
    #: helps languages that drop pronouns/subjects (Japanese in particular) read as a
    #: continuous conversation instead of isolated, disconnected lines.
    context_sentences: int = 5
    temperature: float = 0.2
    num_predict: int = 512
    #: Passed to Ollama's keep_alive so the model is not evicted between utterances.
    keep_alive: str = "10m"
    #: Socket read timeout. Cancellation is only checked between streamed chunks, so this also
    #: bounds how long a wedged request can delay shutdown. A sentence that has produced no
    #: tokens for this long is not going to be useful as a live caption anyway.
    request_timeout: float = 45.0
    #: Render tokens as they arrive instead of waiting for the full translation.
    stream: bool = True
    #: Optional extra instruction appended to the system prompt (tone, glossary...).
    extra_instructions: str = ""


@dataclass
class UiConfig:
    opacity: float = 0.92
    font_size: int = 20
    original_font_size: int = 15
    always_on_top: bool = True
    show_original: bool = True
    layout: str = "side"
    #: Click-through: mouse events pass to the window underneath.
    mouse_transparent: bool = False
    theme: str = "dark"
    max_history: int = 200
    #: Saved as [x, y, w, h]; empty means "centre on the primary screen".
    geometry: list = field(default_factory=list)
    show_status_bar: bool = True


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    #: Start capturing as soon as the window opens.
    autostart: bool = False
    log_level: str = "INFO"

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "AppConfig":
        cfg = cls()
        if isinstance(data, dict):
            _apply(cfg, data)
        return cfg

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AppConfig":
        path = path or config_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            log.warning("ignoring unreadable config %s: %s", path, exc)
            return cls()
        try:
            return cls.from_dict(raw)
        except Exception as exc:  # noqa: BLE001
            # A hand-edited config must never be able to stop the app from starting; running
            # with defaults and saying so is always better than refusing to launch.
            log.warning("ignoring malformed config %s: %s", path, exc)
            return cls()

    def save(self, path: Optional[Path] = None) -> None:
        path = path or config_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write then replace so a crash mid-write cannot truncate a good config.
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(
                json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            os.replace(tmp, path)
        except OSError as exc:
            log.warning("could not save config to %s: %s", path, exc)

    def copy(self) -> "AppConfig":
        return AppConfig.from_dict(self.to_dict())


def _apply(target: Any, data: Dict[str, Any]) -> None:
    """Recursively copy known keys from ``data`` onto the dataclass ``target``."""
    valid = {f.name: f for f in fields(target)}
    for key, value in data.items():
        spec = valid.get(key)
        if spec is None:
            continue
        current = getattr(target, key)
        if is_dataclass(current):
            # A section must be an object. Assigning a stray scalar here would "load"
            # successfully and then fail much later with an AttributeError from whatever first
            # reads a field off it, a long way from the actual cause.
            if isinstance(value, dict):
                _apply(current, value)
            else:
                log.warning("ignoring config key %r: expected an object, got %s", key, type(value).__name__)
            continue
        coerced = _coerce(value, current)
        if coerced is not None:
            setattr(target, key, coerced)


#: int()/float() raise OverflowError, not ValueError, on an infinite input -- and Python's json
#: module happily parses both ``Infinity`` and ``1e400``. Catching only ValueError let a hand-
#: edited config abort startup.
_CAST_ERRORS = (TypeError, ValueError, OverflowError)


def _coerce(value: Any, current: Any) -> Any:
    """Best-effort cast of a JSON value to the default's type; ``None`` = reject."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        if isinstance(value, (int, float)):
            return bool(value)
        return None
    if isinstance(current, int) and not isinstance(current, bool):
        try:
            return int(value)
        except _CAST_ERRORS:
            return None
    if isinstance(current, float):
        try:
            result = float(value)
        except _CAST_ERRORS:
            return None
        # A NaN or infinite setting is never useful and propagates into layout arithmetic.
        return result if math.isfinite(result) else None
    if isinstance(current, str):
        return value if isinstance(value, str) else None
    if isinstance(current, list):
        return list(value) if isinstance(value, (list, tuple)) else None
    return value
