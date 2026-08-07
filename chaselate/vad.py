"""Voice activity detection and utterance segmentation.

Whisper wants whole utterances. Chopping the stream on a fixed timer splits words in half
and produces garbage at every boundary, so instead we watch for pauses and cut there --
the "lossless segmentation" idea from the original project.

Two details make the difference between usable and unusable captions:

* **Padding.** Silero marks the frame where energy crosses the threshold, which is already
  inside the first phoneme. Segments are therefore widened by ``speech_pad_ms`` on both
  sides, otherwise leading consonants are clipped and Whisper guesses.
* **Overlap on forced cuts.** Someone reading aloud may not pause for a minute. When
  ``max_segment_s`` forces a cut mid-sentence, the next segment restarts *before* the cut so
  no audio is lost; the duplicated words are removed downstream by
  :func:`chaselate.textutils.dedupe_overlap`.

The Silero model ships inside faster-whisper, so no extra download or torch dependency.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import VadConfig

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
#: Silero v5/v6 operate on fixed 512-sample frames at 16 kHz (32 ms).
FRAME_SAMPLES = 512
FRAME_MS = FRAME_SAMPLES * 1000 // SAMPLE_RATE

#: Frames per ONNX call. Larger is cheaper but delays the pause decision, so this is the
#: floor on end-of-utterance latency (8 frames = 256 ms).
CHUNK_FRAMES = 8
#: Frames of already-seen audio prepended to each call purely to warm the LSTM state,
#: which the model resets on every invocation. Their outputs are discarded.
WARMUP_FRAMES = 2

#: Never let the working buffer grow past this, whatever the config says.
MAX_BUFFER_SECONDS = 60.0


@dataclass
class Segment:
    """One utterance of 16 kHz mono audio, ready for transcription."""

    audio: np.ndarray
    #: Seconds since capture started, for the padded audio actually handed over.
    start: float
    end: float
    #: Why the segment closed: ``silence``, ``maxlen`` or ``flush``.
    reason: str = "silence"
    #: True when the previous segment was cut mid-speech, so text may overlap this one.
    continues_previous: bool = False

    @property
    def duration(self) -> float:
        return self.audio.size / float(SAMPLE_RATE)


class SileroVad:
    """Thin wrapper over the Silero ONNX model bundled with faster-whisper.

    Call :meth:`load` up front rather than relying on lazy loading. Importing onnxruntime
    while another thread is pulling the CUDA libraries into the process can fail on the
    Windows loader lock, and a lazy load would hit exactly that window -- the first audio
    arrives while the Whisper model is still loading. The observable symptom was a silent
    downgrade to the energy gate for the whole session.
    """

    #: Transient failures get another chance; a genuinely broken onnxruntime does not need
    #: to be retried on every audio chunk.
    MAX_LOAD_ATTEMPTS = 3

    def __init__(self) -> None:
        self._model = None
        self._attempts = 0
        self._gave_up = False

    @property
    def available(self) -> bool:
        """Whether the model is loaded *right now*. Does not trigger a load."""
        return self._model is not None

    def load(self) -> bool:
        """Load the model if needed. Returns whether it is usable afterwards."""
        if self._model is not None:
            return True
        if self._gave_up:
            return False
        self._attempts += 1
        try:
            from faster_whisper.vad import get_vad_model

            self._model = get_vad_model()
            log.info("Silero VAD loaded")
            return True
        except Exception as exc:  # noqa: BLE001
            if self._attempts >= self.MAX_LOAD_ATTEMPTS:
                self._gave_up = True
                log.warning(
                    "Silero VAD unavailable after %d attempts (%s); falling back to the "
                    "energy gate, which segments more crudely",
                    self._attempts, exc,
                )
            else:
                log.debug("Silero VAD load attempt %d failed: %s", self._attempts, exc)
            return False

    def probabilities(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Speech probability per 512-sample frame, or ``None`` if the model is missing.

        ``audio`` length must be a multiple of :data:`FRAME_SAMPLES`.
        """
        model = self._model
        if model is None:
            return None
        if audio.size == 0 or audio.size % FRAME_SAMPLES:
            raise ValueError(
                f"audio length {audio.size} is not a multiple of {FRAME_SAMPLES}"
            )
        try:
            out = model(np.ascontiguousarray(audio, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            log.warning("Silero inference failed (%s); disabling VAD model", exc)
            self._gave_up = True
            self._model = None
            return None
        return np.asarray(out, dtype=np.float32).reshape(-1)


@dataclass
class _State:
    in_speech: bool = False
    #: Absolute sample index where the current utterance's speech began.
    speech_start: int = 0
    #: Absolute sample index just past the last frame judged to be speech.
    speech_end: int = 0
    silence_frames: int = 0
    carry_overlap: bool = False


class Segmenter:
    """Turns a continuous 16 kHz stream into padded utterances.

    Feed blocks of any length; :meth:`feed` returns the segments that closed as a result.
    Silence must be fed too -- it is what tells the segmenter an utterance ended.
    """

    def __init__(self, config: Optional[VadConfig] = None, vad: Optional[SileroVad] = None):
        self.config = config or VadConfig()
        self.vad = vad if vad is not None else SileroVad()
        # Audio held for the current utterance plus enough history for the pre-roll pad.
        self._buf = np.zeros(0, dtype=np.float32)
        #: Absolute sample index of ``_buf[0]``.
        self._origin = 0
        #: Absolute sample index just past everything appended so far.
        self._written = 0
        #: Frames accepted into _buf but not yet scored by the VAD.
        self._unscored = np.zeros(0, dtype=np.float32)
        self._unscored_start = 0
        #: Tail of already-scored audio kept only to warm the LSTM.
        self._warmup = np.zeros(0, dtype=np.float32)
        self._state = _State()
        self._speech_active = False

    # -- helpers -----------------------------------------------------------

    @property
    def speech_active(self) -> bool:
        """True while the segmenter believes someone is mid-utterance."""
        return self._speech_active

    def _pad_samples(self) -> int:
        return max(0, int(self.config.speech_pad_ms * SAMPLE_RATE // 1000))

    def _min_speech_samples(self) -> int:
        return max(0, int(self.config.min_speech_ms * SAMPLE_RATE // 1000))

    def _silence_frames_needed(self) -> int:
        return max(1, int(round(self.config.min_silence_ms / FRAME_MS)))

    def _max_segment_samples(self) -> int:
        return max(FRAME_SAMPLES, int(self.config.max_segment_s * SAMPLE_RATE))

    def reset(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._origin = 0
        self._written = 0
        self._unscored = np.zeros(0, dtype=np.float32)
        self._unscored_start = 0
        self._warmup = np.zeros(0, dtype=np.float32)
        self._state = _State()
        self._speech_active = False

    def _slice(self, start: int, end: int) -> np.ndarray:
        """Absolute sample range out of the working buffer, clamped to what we still hold."""
        start = max(start, self._origin)
        end = min(end, self._origin + self._buf.size)
        if end <= start:
            return np.zeros(0, dtype=np.float32)
        return self._buf[start - self._origin : end - self._origin].copy()

    def _trim(self, keep_from: int) -> None:
        """Drop buffered audio before ``keep_from`` (absolute index)."""
        keep_from = min(keep_from, self._written)
        if keep_from <= self._origin:
            # Even when idle, cap the buffer so a long silence cannot grow it forever.
            limit = int(MAX_BUFFER_SECONDS * SAMPLE_RATE)
            if self._buf.size > limit:
                drop = self._buf.size - limit
                self._buf = self._buf[drop:].copy()
                self._origin += drop
            return
        drop = keep_from - self._origin
        if drop > 0:
            self._buf = self._buf[drop:].copy()
            self._origin = keep_from

    # -- main loop ---------------------------------------------------------

    def feed(self, block: np.ndarray) -> List[Segment]:
        """Consume one block of 16 kHz mono audio; return any segments that closed."""
        block = np.ascontiguousarray(block, dtype=np.float32).reshape(-1)
        if block.size == 0:
            return []

        self._buf = np.concatenate((self._buf, block)) if self._buf.size else block.copy()
        if self._unscored.size == 0:
            self._unscored_start = self._written
        self._unscored = (
            np.concatenate((self._unscored, block)) if self._unscored.size else block.copy()
        )
        self._written += block.size

        segments: List[Segment] = []
        # Score whole CHUNK_FRAMES groups; anything shorter waits for the next block.
        chunk_samples = CHUNK_FRAMES * FRAME_SAMPLES
        while self._unscored.size >= chunk_samples:
            chunk = self._unscored[:chunk_samples]
            chunk_start = self._unscored_start
            self._unscored = self._unscored[chunk_samples:].copy()
            self._unscored_start += chunk_samples
            segments.extend(self._score_chunk(chunk, chunk_start))
        return segments

    def _score_chunk(self, chunk: np.ndarray, chunk_start: int) -> List[Segment]:
        n_frames = chunk.size // FRAME_SAMPLES

        # Energy gate first: cheap, and it catches the digital-silence case where Silero
        # sometimes still reports low-confidence speech.
        frames = chunk.reshape(n_frames, FRAME_SAMPLES)
        rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
        loud = rms >= float(self.config.silence_rms)

        probs: Optional[np.ndarray] = None
        if self.vad.available:
            warm = self._warmup
            padded = np.concatenate((warm, chunk)) if warm.size else chunk
            try:
                scored = self.vad.probabilities(padded)
            except ValueError:
                scored = None
            if scored is not None:
                # Drop the warm-up frames' outputs.
                offset = warm.size // FRAME_SAMPLES
                probs = scored[offset : offset + n_frames]
                if probs.size != n_frames:
                    probs = None

        keep = WARMUP_FRAMES * FRAME_SAMPLES
        self._warmup = chunk[-keep:].copy() if chunk.size >= keep else chunk.copy()

        if probs is None:
            # No model: treat "loud enough" as speech. Cruder, but it still segments.
            speech = loud
        else:
            speech = (probs >= float(self.config.threshold)) & loud

        segments: List[Segment] = []
        for i in range(n_frames):
            frame_start = chunk_start + i * FRAME_SAMPLES
            frame_end = frame_start + FRAME_SAMPLES
            segments.extend(self._advance(bool(speech[i]), frame_start, frame_end))

        self._speech_active = self._state.in_speech
        if not self._state.in_speech:
            # Keep only enough history to pad the start of the next utterance. This must be
            # measured from the end of *scored* audio, not from self._written: a single large
            # feed() is scored in several chunks, and trimming against the write position
            # would discard audio that a later chunk of the same call still needs.
            scored_end = chunk_start + chunk.size
            self._trim(scored_end - self._pad_samples() - FRAME_SAMPLES)
        return segments

    def _advance(self, is_speech: bool, frame_start: int, frame_end: int) -> List[Segment]:
        state = self._state
        out: List[Segment] = []

        if not state.in_speech:
            if is_speech:
                state.in_speech = True
                state.speech_start = frame_start
                state.speech_end = frame_end
                state.silence_frames = 0
            return out

        if is_speech:
            state.speech_end = frame_end
            state.silence_frames = 0
        else:
            state.silence_frames += 1

        if state.silence_frames >= self._silence_frames_needed():
            seg = self._close(state.speech_start, state.speech_end, "silence")
            if seg is not None:
                out.append(seg)
            state.in_speech = False
            state.silence_frames = 0
            state.carry_overlap = False
            return out

        if frame_end - state.speech_start >= self._max_segment_samples():
            # Cut here but rewind the next segment by the pad so the boundary audio is in
            # both segments; the text overlap is removed downstream.
            seg = self._close(state.speech_start, frame_end, "maxlen")
            if seg is not None:
                out.append(seg)
            overlap = self._pad_samples()
            state.speech_start = max(state.speech_start, frame_end - overlap)
            state.speech_end = frame_end
            state.silence_frames = 0
            state.carry_overlap = True
        return out

    def _close(self, speech_start: int, speech_end: int, reason: str) -> Optional[Segment]:
        if speech_end <= speech_start:
            return None
        if speech_end - speech_start < self._min_speech_samples():
            log.debug(
                "dropping %.0f ms segment (below min_speech_ms=%d)",
                (speech_end - speech_start) / SAMPLE_RATE * 1000,
                self.config.min_speech_ms,
            )
            return None

        pad = self._pad_samples()
        start = max(0, speech_start - pad)
        end = min(self._written, speech_end + pad)
        audio = self._slice(start, end)
        if audio.size == 0:
            return None
        segment = Segment(
            audio=audio,
            start=start / float(SAMPLE_RATE),
            end=end / float(SAMPLE_RATE),
            reason=reason,
            continues_previous=self._state.carry_overlap,
        )
        # Everything before this segment's speech is no longer needed.
        self._trim(max(self._origin, speech_end - pad))
        return segment

    def flush(self) -> List[Segment]:
        """Close whatever is in flight, e.g. when the user presses stop."""
        out: List[Segment] = []
        state = self._state
        if state.in_speech and state.speech_end > state.speech_start:
            seg = self._close(state.speech_start, state.speech_end, "flush")
            if seg is not None:
                out.append(seg)
        state.in_speech = False
        state.silence_frames = 0
        state.carry_overlap = False
        self._speech_active = False
        self._unscored = np.zeros(0, dtype=np.float32)
        return out
