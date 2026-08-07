"""Rate conversion from whatever WASAPI hands us down to Whisper's 16 kHz.

Sound cards run at 44.1 or 48 kHz; Whisper wants 16 kHz mono. Doing that per block is
not just ``array[::3]`` -- decimating without an anti-alias filter folds everything above
8 kHz back into the speech band, and restarting the filter on every block puts a click at
every boundary. :class:`StreamResampler` therefore carries both the filter history and the
fractional sample phase across calls, so feeding it N blocks gives bit-identical output to
feeding it the concatenation of those blocks.

numpy only, on purpose: scipy would be a 40 MB dependency for one FIR.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

TARGET_RATE = 16000

# Kaiser beta 8.6 puts the stopband around -90 dB, well below the noise floor of any
# real capture path.
_KAISER_BETA = 8.6
# Cut a little below Nyquist of the *lower* rate to leave the filter room to roll off.
_CUTOFF_FRACTION = 0.45


def downmix_mono(block: np.ndarray) -> np.ndarray:
    """Average interleaved channels into a 1-D float32 mono signal.

    Averaging rather than taking channel 0: a stereo mix often carries dialogue centred,
    and dropping a channel throws away half the energy (and all of it for hard-panned
    content).
    """
    if block.ndim == 1:
        mono = block
    elif block.shape[1] == 1:
        mono = block[:, 0]
    else:
        mono = block.mean(axis=1)
    return np.ascontiguousarray(mono, dtype=np.float32)


def _design_lowpass(cutoff_norm: float, num_taps: int) -> np.ndarray:
    """Windowed-sinc lowpass, unity DC gain. ``cutoff_norm`` is cycles/sample."""
    if num_taps % 2 == 0:
        num_taps += 1
    n = np.arange(num_taps, dtype=np.float64) - (num_taps - 1) / 2.0
    h = 2.0 * cutoff_norm * np.sinc(2.0 * cutoff_norm * n)
    h *= np.kaiser(num_taps, _KAISER_BETA)
    total = h.sum()
    if total != 0:
        h /= total
    return h.astype(np.float32)


class StreamResampler:
    """Stateful mono resampler, ``src_rate`` -> ``dst_rate``.

    Anti-alias FIR followed by linear interpolation. Linear interpolation is adequate here
    because the signal has already been band-limited well below the output Nyquist, and
    Whisper's mel front-end is insensitive to the residual error.
    """

    def __init__(self, src_rate: int, dst_rate: int = TARGET_RATE, taps_per_lobe: int = 8):
        if src_rate <= 0:
            raise ValueError(f"src_rate must be positive, got {src_rate}")
        if dst_rate <= 0:
            raise ValueError(f"dst_rate must be positive, got {dst_rate}")
        self.src_rate = int(src_rate)
        self.dst_rate = int(dst_rate)
        self.ratio = self.src_rate / float(self.dst_rate)
        self.passthrough = self.src_rate == self.dst_rate

        if self.passthrough:
            self._h: Optional[np.ndarray] = None
            self._hist = np.zeros(0, dtype=np.float32)
        else:
            # Only downsampling needs the guard; when upsampling, src Nyquist is already
            # the binding limit, so filter at src's own band edge.
            cutoff_rate = min(self.dst_rate, self.src_rate) * _CUTOFF_FRACTION
            cutoff_norm = cutoff_rate / self.src_rate
            num_taps = 2 * int(round(taps_per_lobe * max(self.ratio, 1.0))) + 1
            num_taps = int(np.clip(num_taps, 17, 513))
            self._h = _design_lowpass(cutoff_norm, num_taps)
            self._hist = np.zeros(len(self._h) - 1, dtype=np.float32)

        self._pending = np.zeros(0, dtype=np.float32)
        self._phase = 0.0

    @property
    def latency_samples(self) -> int:
        """Constant group delay the FIR adds, in input samples."""
        return 0 if self._h is None else (len(self._h) - 1) // 2

    def reset(self) -> None:
        """Forget all history. Call when the stream restarts, not between blocks."""
        if self._h is not None:
            self._hist = np.zeros(len(self._h) - 1, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)
        self._phase = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        """Feed one mono block, get however many output samples are ready.

        The returned length varies by +/-1 sample around ``len(block) / ratio``; that is
        expected and is what keeps the phase exact over long runs.
        """
        block = np.ascontiguousarray(downmix_mono(block), dtype=np.float32)
        if block.size == 0:
            return np.zeros(0, dtype=np.float32)
        if self.passthrough:
            return block.copy()

        assert self._h is not None
        x = np.concatenate((self._hist, block))
        # 'valid' with a history of exactly ntaps-1 yields one output per input sample.
        filtered = np.convolve(x, self._h, mode="valid").astype(np.float32, copy=False)
        self._hist = x[x.size - (self._h.size - 1) :]

        buf = np.concatenate((self._pending, filtered)) if self._pending.size else filtered
        if buf.size < 2:
            self._pending = buf
            return np.zeros(0, dtype=np.float32)

        # How many output samples land at or before the last interpolable input sample.
        span = (buf.size - 1) - self._phase
        if span < 0:
            self._pending = buf
            return np.zeros(0, dtype=np.float32)
        n_out = int(np.floor(span / self.ratio)) + 1

        idx = self._phase + np.arange(n_out, dtype=np.float64) * self.ratio
        i0 = np.floor(idx).astype(np.int64)
        frac = (idx - i0).astype(np.float32)
        # An output position landing exactly on the last sample (frac == 0, which happens
        # on every block for integer ratios like 48k->16k) would index one past the end.
        # Clamping is exact there because the right-hand term is weighted by frac == 0.
        i1 = np.minimum(i0 + 1, buf.size - 1)
        out = buf[i0] * (1.0 - frac) + buf[i1] * frac

        # Keep from the last used sample onward so the next block can interpolate across
        # the boundary, and rebase the phase onto the trimmed buffer.
        consumed = int(i0[-1])
        self._pending = buf[consumed:].copy()
        self._phase = float(idx[-1] + self.ratio - consumed)
        return out.astype(np.float32, copy=False)

    def flush(self) -> np.ndarray:
        """Drain the tail at end of stream by feeding in silence to push the FIR through."""
        if self.passthrough or self._h is None:
            tail, self._pending = self._pending, np.zeros(0, dtype=np.float32)
            return tail
        return self.process(np.zeros(self.latency_samples, dtype=np.float32))


def resample(signal: np.ndarray, src_rate: int, dst_rate: int = TARGET_RATE) -> np.ndarray:
    """One-shot convenience wrapper for offline audio (tests, file input)."""
    r = StreamResampler(src_rate, dst_rate)
    head = r.process(signal)
    tail = r.flush()
    return np.concatenate((head, tail)) if tail.size else head
