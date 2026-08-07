"""Tests for chaselate.audio.resample: downmixing and streaming rate conversion."""

import numpy as np
import pytest

from chaselate.audio.resample import StreamResampler, downmix_mono, resample


# -- downmix_mono ---------------------------------------------------------

def test_downmix_mono_1d_input_passthrough():
    block = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = downmix_mono(block)
    np.testing.assert_array_equal(result, block)


def test_downmix_mono_single_channel_2d():
    block = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
    result = downmix_mono(block)
    np.testing.assert_allclose(result, [0.1, 0.2, 0.3])


def test_downmix_mono_stereo_averages_channels():
    left = np.array([1.0, 0.0, 0.5], dtype=np.float32)
    right = np.array([0.0, 1.0, 0.5], dtype=np.float32)
    block = np.stack([left, right], axis=1)
    result = downmix_mono(block)
    np.testing.assert_allclose(result, [0.5, 0.5, 0.5])


def test_downmix_mono_returns_float32():
    block = np.array([[1, 2], [3, 4]], dtype=np.float64)
    result = downmix_mono(block)
    assert result.dtype == np.float32


def test_downmix_mono_averages_not_takes_first_channel():
    # If it took channel 0 this would be 1.0, not the average 0.6.
    left = np.array([1.0], dtype=np.float32)
    right = np.array([0.2], dtype=np.float32)
    block = np.stack([left, right], axis=1)
    result = downmix_mono(block)
    np.testing.assert_allclose(result, [0.6])


# -- StreamResampler: chunk invariance -------------------------------------

def _chunked_feed(resampler: StreamResampler, signal: np.ndarray, chunk_sizes):
    out = []
    i = 0
    sizes = list(chunk_sizes)
    j = 0
    n = len(signal)
    while i < n:
        size = sizes[j % len(sizes)]
        j += 1
        block = signal[i : i + size]
        i += size
        out.append(resampler.process(block))
    out.append(resampler.flush())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@pytest.mark.parametrize("src_rate", [48000, 44100])
def test_stream_resampler_chunk_invariance(src_rate):
    rng = np.random.default_rng(42)
    n = 20000
    signal = rng.uniform(-0.5, 0.5, n).astype(np.float32)

    whole = StreamResampler(src_rate, 16000)
    out_whole = np.concatenate([whole.process(signal), whole.flush()])

    chunked = StreamResampler(src_rate, 16000)
    out_chunked = _chunked_feed(chunked, signal, [1024, 512, 4096, 333, 2048])

    assert len(out_whole) == len(out_chunked)
    np.testing.assert_allclose(out_whole, out_chunked, atol=1e-5)


def test_stream_resampler_output_length_matches_ratio():
    n = 48000
    signal = np.zeros(n, dtype=np.float32)
    r = StreamResampler(48000, 16000)
    out = np.concatenate([r.process(signal), r.flush()])
    expected = n / r.ratio
    assert abs(len(out) - expected) <= 16


def test_stream_resampler_attenuates_high_frequency_above_nyquist():
    src_rate = 48000
    dst_rate = 16000
    duration = 0.5
    t = np.arange(int(src_rate * duration)) / src_rate

    low_freq_signal = (0.8 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    high_freq_signal = (0.8 * np.sin(2 * np.pi * 10000 * t)).astype(np.float32)

    low_out = resample(low_freq_signal, src_rate, dst_rate)
    high_out = resample(high_freq_signal, src_rate, dst_rate)

    # Skip initial filter-settling samples.
    low_rms = np.sqrt(np.mean(low_out[500:] ** 2))
    high_rms = np.sqrt(np.mean(high_out[500:] ** 2))

    assert low_rms > 0.6 * 0.8 * 0.9  # roughly preserved (allow some margin)
    assert high_rms < 0.05


def test_stream_resampler_passthrough_when_rates_equal():
    signal = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    r = StreamResampler(16000, 16000)
    assert r.passthrough is True
    out = np.concatenate([r.process(signal), r.flush()])
    np.testing.assert_allclose(out, signal)


def test_stream_resampler_empty_array_input_does_not_crash():
    r = StreamResampler(48000, 16000)
    out = r.process(np.zeros(0, dtype=np.float32))
    assert out.size == 0


def test_stream_resampler_reset_matches_fresh_instance():
    rng = np.random.default_rng(7)
    signal = rng.uniform(-0.5, 0.5, 5000).astype(np.float32)

    r = StreamResampler(48000, 16000)
    r.process(signal)  # warm up state
    r.reset()
    out_after_reset = np.concatenate([r.process(signal), r.flush()])

    fresh = StreamResampler(48000, 16000)
    out_fresh = np.concatenate([fresh.process(signal), fresh.flush()])

    np.testing.assert_allclose(out_after_reset, out_fresh, atol=1e-5)


def test_stream_resampler_rejects_non_positive_src_rate():
    with pytest.raises(ValueError):
        StreamResampler(0, 16000)
    with pytest.raises(ValueError):
        StreamResampler(-100, 16000)


def test_stream_resampler_upsampling_does_not_crash():
    signal = np.zeros(1600, dtype=np.float32)
    r = StreamResampler(16000, 48000)
    out = np.concatenate([r.process(signal), r.flush()])
    assert out.size > 0
