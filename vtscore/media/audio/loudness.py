"""Level measurement and gain on a WAV clip's integer samples.

:class:`~vtscore.media.audio.cleaner.AudioLoudnessNormalizeCleaner` needs three
things from a clip: how loud its loudest sample is, how loud the clip is
overall, and a copy of it scaled by some gain.  :class:`PcmAudio` answers all
three on the clip's stored integer PCM samples.

That domain is deliberate.  :mod:`vtscore.media.audio.decode` would hand back a
float waveform, but writing one back out means choosing a new sample format and
trusting a float->int conversion; here a scaled clip keeps the source's channel
count, sample rate and sample width, and the same input and gain always produce
the same bytes.  A reference (thin) import relies on that to replay the cleaner
instead of storing its output.  Reading goes through
:func:`~vtscore.media.audio.wav.open_wav`, so anything the silence trim can
slice (a ``WAVE_FORMAT_EXTENSIBLE`` WAV, or a FLAC / MP3 that ``soundfile``
re-encodes to 16-bit PCM) this can measure and scale too, and nothing else.
"""

from __future__ import annotations

import io
import math
import wave
from dataclasses import dataclass

import numpy as np

from vtscore.media.audio.wav import open_wav

__all__ = ["GATE_DB", "WINDOW_SECONDS", "PcmAudio"]

#: Length (seconds) of the windows :meth:`PcmAudio.gated_rms_dbfs` measures.
WINDOW_SECONDS = 0.05

#: Windows more than this many dB below a clip's loudest window are left out of
#: its gated RMS.  The silence trim only removes audio at least its ``top_db``
#: below the clip's loudest frame, so at any ``top_db`` of 20 or more (its
#: default is 40) everything it can remove is already gated out here: trimming
#: a clip does not move its gated RMS, and the two cleaners commute.
GATE_DB = 20.0


def _dbfs(ratio: float) -> float:
    """``ratio`` of full scale in dB, ``-inf`` for zero (no ``log10(0)``)."""
    return 20.0 * math.log10(ratio) if ratio > 0 else -math.inf


def _unpack(frames: bytes, sampwidth: int, n_channels: int) -> np.ndarray:
    """Decode little-endian PCM frames to signed ints, shape ``(frames, channels)``."""
    if sampwidth == 1:
        # 8-bit WAV is the one unsigned width: silence sits at 128.
        flat = np.frombuffer(frames, dtype=np.uint8).astype(np.int64) - 128
    elif sampwidth == 3:
        b = np.frombuffer(frames, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
        flat = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        flat = np.where(flat >= 1 << 23, flat - (1 << 24), flat)
    else:
        flat = np.frombuffer(frames, dtype=f"<i{sampwidth}").astype(np.int64)
    return flat.reshape(-1, n_channels)


def _pack(samples: np.ndarray, sampwidth: int) -> bytes:
    """Inverse of :func:`_unpack`."""
    flat = samples.reshape(-1)
    if sampwidth == 1:
        return (flat + 128).astype(np.uint8).tobytes()
    if sampwidth == 3:
        u = flat & 0xFFFFFF  # 24-bit two's complement
        return np.stack([u & 0xFF, (u >> 8) & 0xFF, u >> 16], axis=1).astype(np.uint8).tobytes()
    return flat.astype(f"<i{sampwidth}").tobytes()


@dataclass(frozen=True, eq=False)
class PcmAudio:
    """A clip's integer PCM samples, shape ``(frames, channels)``."""

    samples: np.ndarray
    sample_rate: int
    sampwidth: int

    @classmethod
    def from_wav(cls, wav_bytes: bytes) -> PcmAudio:
        """Read *wav_bytes*, raising :class:`~vtscore.media.audio.wav.AudioDecodeError` if it can't."""
        with open_wav(wav_bytes) as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        return cls(_unpack(frames, sampwidth, n_channels), sample_rate, sampwidth)

    def to_wav(self) -> bytes:
        """Encode as a PCM WAV in this clip's own channel count, rate and width."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as out:
            out.setnchannels(self.samples.shape[1])
            out.setsampwidth(self.sampwidth)
            out.setframerate(self.sample_rate)
            out.writeframes(_pack(self.samples, self.sampwidth))
        return buf.getvalue()

    @property
    def full_scale(self) -> int:
        """Magnitude of the most negative sample: the 0 dBFS reference."""
        return 1 << (8 * self.sampwidth - 1)

    @property
    def peak_dbfs(self) -> float:
        """Level of the loudest sample on any channel; ``-inf`` for digital silence."""
        if self.samples.size == 0:
            return -math.inf
        return _dbfs(int(np.abs(self.samples).max()) / self.full_scale)

    def gated_rms_dbfs(self, gate_db: float = GATE_DB) -> float:
        """RMS level of the clip's audible material; ``-inf`` for digital silence.

        Every :data:`WINDOW_SECONDS` window of the clip, one starting at each
        sample (the whole clip, if it is shorter than a window), is measured,
        and only windows within *gate_db* of the loudest count.  A window at
        every offset, rather than back-to-back blocks, is what makes the result
        independent of where the clip starts: trim its head and the same
        windows are measured, where fixed blocks would straddle the sound's
        edges differently and move the level by tenths of a dB.

        Gating relative to the loudest window, rather than to an absolute level
        or to the clip's ungated mean, keeps the measure about the sound itself:
        the same bark reads the same whether it fills its clip or sits in a few
        seconds of quiet, and a gain moves every window by the same amount, so
        it never changes which windows are kept.
        """
        if self.samples.size == 0:
            return -math.inf
        x = self.samples / self.full_scale
        power = np.mean(np.square(x), axis=1)  # per frame, averaged over channels
        window = min(len(power), max(1, round(self.sample_rate * WINDOW_SECONDS)))
        sums = np.concatenate(([0.0], np.cumsum(power)))
        windows = (sums[window:] - sums[:-window]) / window  # mean power of the window starting at each frame
        loudest = float(windows.max())
        if loudest <= 0:
            return -math.inf
        kept = windows[windows >= loudest * 10.0 ** (-gate_db / 10.0)]
        return 10.0 * math.log10(float(kept.mean()))

    def with_gain(self, gain_db: float) -> PcmAudio:
        """Return a copy scaled by *gain_db*, rounded and clamped to the sample range."""
        scaled = np.rint(self.samples * 10.0 ** (gain_db / 20.0))
        clamped = np.clip(scaled, -self.full_scale, self.full_scale - 1).astype(np.int64)
        return PcmAudio(clamped, self.sample_rate, self.sampwidth)
