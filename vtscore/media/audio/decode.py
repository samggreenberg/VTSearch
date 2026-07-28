"""Decode audio to a float32 waveform without librosa's ``audioread`` fallback.

``librosa.load`` reaches ``audioread`` whenever ``libsndfile`` can't parse the
container - which is *always* for AAC/M4A/MP4, the codecs a lot of real audio
datasets ship in.  That fallback is deprecated as of librosa 0.10 and removed in
librosa 1.0, so every AAC/M4A decode in the codebase is riding a dependency that
is scheduled to disappear (loudly, via two ``FutureWarning``/``UserWarning``
lines per decode, and then silently, once it's gone).

:func:`decode_audio` replaces it with the two decoders we already depend on:

1. ``soundfile`` (libsndfile) for WAV/FLAC/OGG/MP3 - the fast native path,
   decoding straight out of an in-memory buffer.
2. ``ffmpeg`` for everything else, resolved by
   :func:`~vtscore.media.audio.ffmpeg.get_ffmpeg_exe` (system ``$PATH``, else
   the static binary bundled by ``imageio-ffmpeg``).  Buffers are piped in via
   ``-i pipe:0``, so archive members decode with no filesystem round-trip.

The return contract deliberately mirrors ``librosa.load``: a C-contiguous
``float32`` array normalized to [-1, 1], mono-downmixed by averaging channels,
``sr=None`` meaning "keep the native rate", and ``offset``/``duration`` measured
in seconds and applied *before* resampling.
"""

from __future__ import annotations

import io
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from vtscore.media.audio.ffmpeg import get_ffmpeg_exe

#: Seconds to wait for an ffmpeg decode before giving up.  Generous enough for
#: a long podcast-length file, bounded so a wedged subprocess can't hang a
#: worker forever.
FFMPEG_TIMEOUT = 300.0

#: Placeholder chunk size ffmpeg's WAV muxer writes when its output isn't
#: seekable (a pipe), in place of the real byte count.
_UNKNOWN_CHUNK_SIZE = 0xFFFFFFFF


class AudioDecodeError(RuntimeError):
    """Raised when neither ``soundfile`` nor ``ffmpeg`` can decode the source."""


def decode_audio(
    source: Any,
    *,
    sr: int | float | None = None,
    mono: bool = True,
    offset: float = 0.0,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """Decode *source* to ``(samples, sample_rate)``.

    *source* may be a filesystem path (``str``/``Path``), raw ``bytes``, or a
    file-like object (e.g. ``io.BytesIO``).

    ``sr`` is the target sample rate; ``None`` keeps the source's native rate.
    ``mono=True`` returns a 1-D array averaged across channels; ``mono=False``
    returns a channel-major ``(channels, samples)`` array.  ``offset`` and
    ``duration`` (seconds) select a window of the source before any resampling.

    Raises :class:`AudioDecodeError` if the audio can't be decoded.
    """
    path, raw = _normalize_source(source)
    if raw is not None and not raw:
        raise AudioDecodeError("audio source is empty")

    try:
        return _decode_soundfile(path, raw, sr=sr, mono=mono, offset=offset, duration=duration)
    except Exception as sf_error:
        soundfile_error = sf_error

    try:
        return _decode_ffmpeg(path, raw, sr=sr, mono=mono, offset=offset, duration=duration)
    except AudioDecodeError:
        raise
    except FileNotFoundError as exc:
        raise AudioDecodeError(
            f"soundfile could not decode this audio ({soundfile_error}) and ffmpeg is not available. "
            "Install ffmpeg via your OS package manager or 'pip install imageio-ffmpeg'."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AudioDecodeError(f"ffmpeg timed out after {FFMPEG_TIMEOUT:g}s decoding audio") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioDecodeError(f"ffmpeg failed to decode audio: {_stderr_text(exc)}") from exc
    except Exception as exc:
        raise AudioDecodeError(f"failed to decode audio: {exc}") from exc


# ---------------------------------------------------------------------------
# Source normalization
# ---------------------------------------------------------------------------


def _normalize_source(source: Any) -> tuple[str | None, bytes | None]:
    """Return ``(path, raw_bytes)`` for *source*; exactly one is non-``None``."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return None, bytes(source)
    if isinstance(source, (str, Path)):
        return str(source), None
    read = getattr(source, "read", None)
    if callable(read):
        seek = getattr(source, "seek", None)
        if callable(seek):
            try:
                seek(0)
            except (OSError, ValueError):
                pass
        blob = read()
        if not isinstance(blob, (bytes, bytearray, memoryview)):
            raise AudioDecodeError("file-like audio source did not yield bytes")
        return None, bytes(blob)
    raise AudioDecodeError(f"unsupported audio source type: {type(source).__name__}")


def _stderr_text(exc: subprocess.CalledProcessError) -> str:
    stderr = exc.stderr
    if isinstance(stderr, bytes):
        return stderr[:500].decode(errors="replace").strip()
    return (stderr or "")[:500].strip()


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------


def _finalize(frames: np.ndarray, native_sr: int, *, sr: int | float | None, mono: bool) -> tuple[np.ndarray, int]:
    """Downmix *frames* ``(samples, channels)``, resample, and return the pair."""
    if mono:
        y = frames[:, 0] if frames.shape[1] == 1 else frames.mean(axis=1)
    else:
        y = frames.T
    y = np.ascontiguousarray(y, dtype=np.float32)

    if sr is None or int(sr) == native_sr:
        return y, native_sr
    return _resample(y, native_sr, int(sr)), int(sr)


def _resample(y: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample with ``soxr`` at HQ quality - librosa's own default backend."""
    import soxr  # noqa: PLC0415

    if y.ndim == 1:
        out = soxr.resample(y, orig_sr, target_sr, quality="HQ")
    else:
        # soxr wants frame-major input; our non-mono arrays are channel-major.
        out = soxr.resample(y.T, orig_sr, target_sr, quality="HQ").T
    return np.ascontiguousarray(out, dtype=np.float32)


# ---------------------------------------------------------------------------
# soundfile (libsndfile) path
# ---------------------------------------------------------------------------


def _decode_soundfile(
    path: str | None,
    raw: bytes | None,
    *,
    sr: int | float | None,
    mono: bool,
    offset: float,
    duration: float | None,
) -> tuple[np.ndarray, int]:
    """Decode via ``soundfile``.  Raises for containers libsndfile can't parse."""
    import soundfile as sf  # noqa: PLC0415

    target: Any = path if path is not None else io.BytesIO(raw or b"")
    with sf.SoundFile(target) as handle:
        native_sr = int(handle.samplerate)
        if offset:
            handle.seek(int(offset * native_sr))
        frame_count = -1 if duration is None else int(duration * native_sr)
        frames = handle.read(frames=frame_count, dtype="float32", always_2d=True)

    if frames.size == 0:
        raise AudioDecodeError("decoded audio is empty")
    return _finalize(frames, native_sr, sr=sr, mono=mono)


# ---------------------------------------------------------------------------
# ffmpeg path
# ---------------------------------------------------------------------------


def _ffmpeg_command(input_arg: str, *, sr: int | float | None, offset: float, duration: float | None) -> list[str]:
    """Build the ffmpeg argv that writes float32 WAVE to ``stdout``."""
    cmd = [get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", input_arg]
    # Output-side seek/trim: sample-accurate, and works on a non-seekable pipe.
    if offset:
        cmd += ["-ss", f"{float(offset):.9g}"]
    if duration is not None:
        cmd += ["-t", f"{float(duration):.9g}"]
    # Take the first audio stream only (skipping cover art, which rides along as
    # a video stream) and keep every one of its channels: the mono downmix
    # happens in numpy below, so it matches librosa's plain channel mean.
    cmd += ["-map", "0:a:0", "-c:a", "pcm_f32le"]
    if sr is not None:
        cmd += ["-ar", str(int(sr))]
    cmd += ["-f", "wav", "pipe:1"]
    return cmd


def _decode_ffmpeg(
    path: str | None,
    raw: bytes | None,
    *,
    sr: int | float | None,
    mono: bool,
    offset: float,
    duration: float | None,
) -> tuple[np.ndarray, int]:
    """Decode via an ffmpeg subprocess, piping in-memory sources over ``stdin``."""
    if path is not None:
        blob = _run_ffmpeg(path, None, sr=sr, offset=offset, duration=duration)
    else:
        try:
            blob = _run_ffmpeg("pipe:0", raw, sr=sr, offset=offset, duration=duration)
        except subprocess.CalledProcessError:
            # Some containers (MP4/M4A with a trailing ``moov`` atom) need a
            # seekable input; those are the one case worth the temp-file spill.
            blob = _run_ffmpeg_via_tempfile(raw or b"", sr=sr, offset=offset, duration=duration)

    samples, native_sr, channels = _parse_wave(blob)
    if samples.size == 0:
        raise AudioDecodeError("decoded audio is empty")
    return _finalize(samples.reshape(-1, channels), native_sr, sr=sr, mono=mono)


def _run_ffmpeg(
    input_arg: str, stdin_bytes: bytes | None, *, sr: int | float | None, offset: float, duration: float | None
) -> bytes:
    result = subprocess.run(
        _ffmpeg_command(input_arg, sr=sr, offset=offset, duration=duration),
        # Always hand ffmpeg a stdin pipe (empty for the path case, where
        # ``-nostdin`` keeps it from being read) so the child never inherits
        # - and blocks on - the parent's terminal.
        input=stdin_bytes if stdin_bytes is not None else b"",
        capture_output=True,
        timeout=FFMPEG_TIMEOUT,
        check=True,
    )
    return result.stdout


def _run_ffmpeg_via_tempfile(raw: bytes, *, sr: int | float | None, offset: float, duration: float | None) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        tmp.write(raw)
        tmp.flush()
        return _run_ffmpeg(tmp.name, None, sr=sr, offset=offset, duration=duration)


def _parse_wave(blob: bytes) -> tuple[np.ndarray, int, int]:
    """Parse a RIFF/WAVE blob of float32 PCM into ``(samples, sr, channels)``.

    ffmpeg's WAV muxer can't backfill chunk sizes when writing to a pipe, so it
    emits placeholder lengths; the ``data`` chunk is therefore sized from what
    actually arrived rather than from its header field.
    """
    if len(blob) < 12 or blob[:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise AudioDecodeError("ffmpeg produced no WAVE output")

    pos = 12
    sample_rate = 0
    channels = 0
    while pos + 8 <= len(blob):
        chunk_id = blob[pos : pos + 4]
        (chunk_size,) = struct.unpack("<I", blob[pos + 4 : pos + 8])
        body = pos + 8
        if chunk_id == b"fmt ":
            if body + 16 > len(blob):
                raise AudioDecodeError("truncated WAVE fmt chunk")
            channels, sample_rate = struct.unpack("<HI", blob[body + 2 : body + 8])
        elif chunk_id == b"data":
            if not sample_rate or not channels:
                raise AudioDecodeError("WAVE data chunk precedes its fmt chunk")
            available = len(blob) - body
            size = chunk_size if 0 < chunk_size <= available else available
            size -= size % (4 * channels)
            # ``bytearray`` (rather than the raw ``bytes`` slice) so the array
            # owns writable memory: callers hand these samples to
            # ``torch.from_numpy``, which warns loudly on a read-only buffer.
            samples = np.frombuffer(bytearray(blob[body : body + size]), dtype="<f4")
            return samples, int(sample_rate), int(channels)
        if chunk_size >= _UNKNOWN_CHUNK_SIZE:
            break
        pos = body + chunk_size + (chunk_size & 1)

    raise AudioDecodeError("no WAVE data chunk in ffmpeg output")
