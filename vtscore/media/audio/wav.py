"""WAV byte-level helpers: duration and slicing, without decoding to a waveform.

These are the low-level primitives the audio clipper, the silence cleaner and
the two clip-replay paths (:mod:`vtscore.media.lazy_clip`,
:mod:`vtscore.detectors.resolver`) all need: read a WAV byte string's duration,
and cut a sub-range out of it *as WAV bytes*.  They live here rather than in the
clipper because three modules outside that clipper — two of them in other
packages — were reaching past the underscore to import them.

This is deliberately **not** :mod:`vtscore.media.audio.decode`, whose job is the
opposite: decode any container (via ``soundfile`` or ``ffmpeg``) down to a
float32 waveform array.  Nothing here decodes samples; a slice goes WAV bytes in,
WAV bytes out, so a clip keeps the source's exact sample data.  The one place the
two meet is :func:`to_pcm_wav`, which re-encodes a non-PCM WAV so the stdlib
``wave`` module can parse it.
"""

from __future__ import annotations

import io
import wave

__all__ = [
    "AudioDecodeError",
    "open_wav",
    "to_pcm_wav",
    "wav_duration",
    "wav_slice",
]


class AudioDecodeError(Exception):
    """Raised when WAV bytes can't be decoded by stdlib ``wave`` or ``soundfile``.

    Signals a corrupt or unsupported audio payload (e.g. GTZAN's truncated
    ``jazz.00054.wav``, or a non-audio blob).  Clippers catch this and fall
    back to returning the media unchanged rather than aborting the whole load.

    Distinct from :class:`vtscore.media.audio.decode.AudioDecodeError`, which
    reports a failure of the waveform decoder; catching one does not catch the
    other.
    """


def to_pcm_wav(wav_bytes: bytes) -> bytes:
    """Re-encode WAV bytes into plain PCM that stdlib ``wave`` can parse.

    The stdlib ``wave`` module only understands ``WAVE_FORMAT_PCM`` (tag 1);
    it raises ``wave.Error`` on ``WAVE_FORMAT_EXTENSIBLE`` (tag 0xFFFE, 65534),
    which is what UrbanSound8K and many other real-world WAVs use.  ``soundfile``
    reads those fine, so decode with it and re-write as 16-bit PCM.
    """
    import soundfile as sf  # noqa: PLC0415

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="int16", always_2d=False)
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _checked_wav(wf: wave.Wave_read) -> wave.Wave_read:
    """Return *wf* if its header is usable, else close it and raise.

    stdlib ``wave`` happily accepts a fmt chunk declaring a sample rate of 0,
    so a corrupt or truncated header can open cleanly and then blow up with a
    ``ZeroDivisionError`` in :func:`wav_duration` (or a ``wave.Error`` from
    ``setframerate`` in :func:`wav_slice`).  Reject it up front as a decode
    failure so callers hit the same graceful-degradation path as any other
    undecodable payload.
    """
    if wf.getframerate() <= 0:
        wf.close()
        raise AudioDecodeError("WAV header declares a non-positive sample rate")
    return wf


def open_wav(wav_bytes: bytes) -> wave.Wave_read:
    """Open WAV bytes for reading, tolerating non-PCM formats.

    Falls back to re-encoding via :func:`to_pcm_wav` when stdlib ``wave``
    can't parse the container (e.g. ``WAVE_FORMAT_EXTENSIBLE``) or parses it
    into an unusable header.  Raises :class:`AudioDecodeError` when neither
    path can decode the bytes (corrupt or unsupported payload) so callers can
    degrade gracefully.
    """
    try:
        return _checked_wav(wave.open(io.BytesIO(wav_bytes), "rb"))
    except (wave.Error, EOFError, AudioDecodeError):
        pass
    try:
        return _checked_wav(wave.open(io.BytesIO(to_pcm_wav(wav_bytes)), "rb"))
    except Exception as exc:  # soundfile LibsndfileError, wave.Error, EOFError, ...
        raise AudioDecodeError("could not decode audio bytes") from exc


def wav_duration(wav_bytes: bytes) -> float:
    """Return the duration in seconds of a WAV byte string."""
    with open_wav(wav_bytes) as wf:
        return wf.getnframes() / wf.getframerate()


def wav_slice(wav_bytes: bytes, start: float, end: float) -> bytes:
    """Extract a [start, end) slice from a WAV byte string.

    Returns a new WAV byte string containing only the requested segment.
    """
    with open_wav(wav_bytes) as wf:
        sr = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        total_frames = wf.getnframes()
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(start_frame, min(end_frame, total_frames))
        wf.setpos(start_frame)
        frames = wf.readframes(end_frame - start_frame)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(sr)
        out.writeframes(frames)
    return buf.getvalue()
