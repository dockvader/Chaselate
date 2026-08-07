"""Tests for chaselate.vad: the segmentation state machine.

Silero itself is stubbed out for determinism and speed. A synthetic amplitude signal
(1.0 = "speech", 0.0 = "silence") drives both the stub VAD's probability output and the
energy gate, so tests describe behavior in terms of speech/silence spans rather than
opaque numbers.
"""

import numpy as np
import pytest

from chaselate.config import VadConfig
from chaselate.vad import FRAME_SAMPLES, Segmenter, SileroVad


class StubVad:
    """Deterministic VAD: frame is 'speech' when its mean abs amplitude exceeds 0.5.

    Mirrors the real SileroVad interface (``available`` + ``probabilities``) but derives
    its answer from the audio content itself, so re-scoring duplicated warm-up frames
    yields the same probability both times -- exactly what a real model would do too.
    """

    available = True

    def probabilities(self, audio: np.ndarray) -> np.ndarray:
        assert audio.size % FRAME_SAMPLES == 0
        frames = audio.reshape(-1, FRAME_SAMPLES)
        mean_abs = np.abs(frames).mean(axis=1)
        return (mean_abs > 0.5).astype(np.float32)


def _speech(n_samples: int, amplitude: float = 0.8) -> np.ndarray:
    return np.full(n_samples, amplitude, dtype=np.float32)


def _silence(n_samples: int) -> np.ndarray:
    return np.zeros(n_samples, dtype=np.float32)


def _make_segmenter(**overrides) -> Segmenter:
    cfg = VadConfig(
        threshold=0.5,
        min_speech_ms=overrides.pop("min_speech_ms", 100),
        min_silence_ms=overrides.pop("min_silence_ms", 300),
        speech_pad_ms=overrides.pop("speech_pad_ms", 100),
        max_segment_s=overrides.pop("max_segment_s", 999.0),
        silence_rms=overrides.pop("silence_rms", 0.0009),
    )
    return Segmenter(cfg, StubVad())


# -- basic state machine ---------------------------------------------------

def test_silence_only_produces_no_segments():
    seg = _make_segmenter()
    block = _silence(4096 * 4)
    segments = seg.feed(block)
    assert segments == []
    assert seg.flush() == []


def test_speech_followed_by_silence_produces_one_segment():
    seg = _make_segmenter()
    speech = _speech(8192)
    silence = _silence(8192)
    segments = seg.feed(np.concatenate([speech, silence]))
    assert len(segments) == 1
    assert segments[0].reason == "silence"
    assert segments[0].audio.size > 0


def test_speech_shorter_than_min_speech_ms_is_dropped():
    seg = _make_segmenter(min_speech_ms=2000)  # 2s minimum, well above what we send
    speech = _speech(4096)  # ~256ms of speech
    silence = _silence(8192)
    segments = seg.feed(np.concatenate([speech, silence]))
    assert segments == []


def test_max_segment_s_forces_cut_and_marks_continuation():
    seg = _make_segmenter(max_segment_s=0.25, min_silence_ms=300)
    # Long continuous speech, well past the max segment length, followed by silence.
    speech = _speech(4096 * 4)
    silence = _silence(4096 * 2)
    segments = seg.feed(np.concatenate([speech, silence]))

    assert len(segments) >= 2
    assert segments[0].reason == "maxlen"
    assert segments[0].continues_previous is False
    # Every segment after the first forced cut carries the continuation flag.
    assert all(s.continues_previous for s in segments[1:-1] if s.reason == "maxlen")
    assert segments[1].continues_previous is True


def test_flush_emits_in_progress_segment():
    seg = _make_segmenter()
    speech = _speech(4096 * 2)
    segments = seg.feed(speech)
    assert segments == []  # nothing closed yet, no trailing silence
    assert seg.speech_active is True

    flushed = seg.flush()
    assert len(flushed) == 1
    assert flushed[0].reason == "flush"
    assert seg.speech_active is False


def test_speech_active_reflects_current_state():
    seg = _make_segmenter()
    assert seg.speech_active is False

    seg.feed(_speech(4096))
    assert seg.speech_active is True

    seg.feed(_silence(4096 * 3))
    assert seg.speech_active is False


# -- padding ---------------------------------------------------------------

def test_segment_padding_matches_speech_pad_ms():
    # Fed in small (512-sample) increments to match how the real capture pipeline calls
    # feed() -- see test_feed_of_large_single_block_can_lose_audio below for why a single
    # giant block is not equivalent.
    pad_ms = 100
    seg = _make_segmenter(speech_pad_ms=pad_ms, min_silence_ms=300)
    pad_samples = seg._pad_samples()

    lead_silence = _silence(pad_samples + 4096)
    speech_len = 4096 * 2
    speech = _speech(speech_len)
    trail_silence = _silence(4096 * 2)
    signal = np.concatenate([lead_silence, speech, trail_silence])

    segments = []
    for i in range(0, signal.size, FRAME_SAMPLES):
        segments.extend(seg.feed(signal[i : i + FRAME_SAMPLES]))

    assert len(segments) == 1
    expected = speech_len + 2 * pad_samples
    # Frame-boundary rounding (32ms frames) accounts for the small tolerance.
    assert abs(segments[0].audio.size - expected) <= FRAME_SAMPLES * 2


def test_feed_of_large_single_block_yields_the_segment():
    # Regression: the history trim measured against the write position rather than the
    # scored position, so one big feed() discarded audio a later chunk of the same call
    # still needed, and the segment vanished.
    pad_ms = 100
    seg = _make_segmenter(speech_pad_ms=pad_ms, min_silence_ms=300)
    pad_samples = seg._pad_samples()

    lead_silence = _silence(pad_samples + 4096)
    speech = _speech(4096 * 2)
    trail_silence = _silence(4096 * 2)

    # Everything handed over in one call -- triggers the premature trim.
    segments = seg.feed(np.concatenate([lead_silence, speech, trail_silence]))
    assert len(segments) == 1


# -- reason field ------------------------------------------------------------

def test_reason_is_silence_when_closed_by_pause():
    seg = _make_segmenter()
    segments = seg.feed(np.concatenate([_speech(4096), _silence(4096 * 2)]))
    assert len(segments) == 1
    assert segments[0].reason == "silence"


def test_reason_is_maxlen_when_forced_cut():
    seg = _make_segmenter(max_segment_s=0.25)
    segments = seg.feed(_speech(4096 * 3))
    assert segments
    assert segments[0].reason == "maxlen"


def test_reason_is_flush_on_manual_flush():
    seg = _make_segmenter()
    seg.feed(_speech(4096))
    flushed = seg.flush()
    assert flushed[0].reason == "flush"


# -- reset -----------------------------------------------------------------

def test_reset_clears_in_progress_state():
    seg = _make_segmenter()
    seg.feed(_speech(4096))
    assert seg.speech_active is True
    seg.reset()
    assert seg.speech_active is False
    # After reset, feeding fresh silence should not resurrect the old utterance.
    segments = seg.feed(_silence(4096 * 2))
    assert segments == []


# -- non-frame-aligned feed sizes --------------------------------------------

def test_feed_accepts_non_512_multiple_chunk_sizes():
    seg = _make_segmenter()
    signal = np.concatenate([_speech(8192), _silence(8192)])
    segments = []
    # Feed in irregular, non-512-aligned pieces; the segmenter buffers internally.
    for start in range(0, len(signal), 777):
        segments.extend(seg.feed(signal[start : start + 777]))
    assert len(segments) == 1
    assert segments[0].reason == "silence"


# -- real Silero model (optional, skipped if unavailable) --------------------

def _silero_available() -> bool:
    # load(), not available: available is a pure "is it loaded right now" check and never
    # triggers a load, so probing with it would skip this test unconditionally.
    try:
        return SileroVad().load()
    except Exception:
        return False


@pytest.mark.skipif(not _silero_available(), reason="Silero VAD model not available locally")
def test_real_silero_reports_no_speech_on_digital_silence():
    vad = SileroVad()
    assert vad.load()
    n_frames = (16000 * 2) // FRAME_SAMPLES  # ~2s of digital silence, frame-aligned
    audio = np.zeros(n_frames * FRAME_SAMPLES, dtype=np.float32)
    probs = vad.probabilities(audio)
    assert probs is not None
    assert float(probs.max()) < 0.5


@pytest.mark.skipif(not _silero_available(), reason="Silero VAD model not available locally")
def test_real_silero_reports_speech_on_a_voiced_tone_burst():
    # Not real speech, but a harmonic-rich burst is enough to move Silero off the floor and
    # confirms the model is actually running rather than returning a constant.
    vad = SileroVad()
    assert vad.load()
    n_frames = (16000 * 2) // FRAME_SAMPLES
    t = np.arange(n_frames * FRAME_SAMPLES, dtype=np.float32) / 16000.0
    tone = sum(np.sin(2 * np.pi * f * t) / (i + 1) for i, f in enumerate((140, 280, 420, 560)))
    audio = (0.3 * tone).astype(np.float32)
    probs = vad.probabilities(audio)
    assert probs is not None
    assert probs.size == n_frames
    silence_probs = vad.probabilities(np.zeros(n_frames * FRAME_SAMPLES, dtype=np.float32))
    assert float(probs.max()) > float(silence_probs.max())


def test_available_does_not_trigger_a_load():
    vad = SileroVad()
    assert vad.available is False
