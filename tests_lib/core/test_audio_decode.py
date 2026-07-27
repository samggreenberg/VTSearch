"""Tests for :mod:`vtscore.media.audio.decode`.

The helper replaces ``librosa.load`` everywhere in the audio stack, so its
contract has to match librosa's exactly: float32 in [-1, 1], mono downmix by
channel mean, ``sr=None`` meaning "native rate", and ``offset``/``duration``
in seconds applied before any resampling.  A subtly-wrong helper wouldn't
crash — it would silently shift every audio embedding in the system — so the
soundfile path is asserted against ``librosa.load`` itself, sample for sample.
"""

import io
import subprocess

import numpy as np
import pytest

from vtscore.media.audio.audio_generator import GENERATOR_SAMPLE_RATE, generate_wav
from vtscore.media.audio.decode import AudioDecodeError, decode_audio


def _librosa_load(*args, **kwargs):
    """``librosa.load`` with its audioread deprecation warnings muted."""
    import warnings

    import librosa

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return librosa.load(*args, **kwargs)


class TestSourceTypes:
    def test_accepts_bytes_path_and_filelike(self, tmp_path):
        wav = generate_wav(440.0, 0.5)
        path = tmp_path / "tone.wav"
        path.write_bytes(wav)

        from_bytes, sr_bytes = decode_audio(wav)
        from_path, sr_path = decode_audio(path)
        from_str, sr_str = decode_audio(str(path))
        from_buf, sr_buf = decode_audio(io.BytesIO(wav))

        assert sr_bytes == sr_path == sr_str == sr_buf == GENERATOR_SAMPLE_RATE
        for other in (from_path, from_str, from_buf):
            assert np.array_equal(from_bytes, other)

    def test_rejects_unsupported_source(self):
        with pytest.raises(AudioDecodeError):
            decode_audio(object())

    def test_rewinds_a_consumed_buffer(self):
        wav = generate_wav(440.0, 0.5)
        buf = io.BytesIO(wav)
        buf.read()  # exhaust it — decode_audio must seek back to 0
        samples, _sr = decode_audio(buf)
        assert samples.size > 0


class TestLibrosaParity:
    """The soundfile path must be bit-identical to what librosa.load returned."""

    def test_native_rate_matches_librosa(self):
        wav = generate_wav(440.0, 1.0)
        ours, sr = decode_audio(wav, sr=None, mono=True)
        theirs, their_sr = _librosa_load(io.BytesIO(wav), sr=None, mono=True)
        assert sr == their_sr
        assert np.array_equal(ours, theirs)

    def test_resampled_matches_librosa(self):
        wav = generate_wav(440.0, 1.0)
        ours, sr = decode_audio(wav, sr=16000, mono=True)
        theirs, their_sr = _librosa_load(io.BytesIO(wav), sr=16000, mono=True)
        assert sr == their_sr == 16000
        assert np.array_equal(ours, theirs)

    def test_offset_and_duration_match_librosa(self):
        wav = generate_wav(440.0, 2.0)
        ours, _sr = decode_audio(wav, sr=None, offset=0.5, duration=0.75)
        theirs, _their_sr = _librosa_load(io.BytesIO(wav), sr=None, offset=0.5, duration=0.75)
        assert np.array_equal(ours, theirs)


class TestReturnContract:
    def test_returns_contiguous_float32_in_range(self):
        samples, sr = decode_audio(generate_wav(440.0, 0.5))
        assert samples.dtype == np.float32
        assert samples.ndim == 1
        assert samples.flags["C_CONTIGUOUS"]
        assert np.abs(samples).max() <= 1.0
        assert sr == GENERATOR_SAMPLE_RATE

    def test_sr_none_keeps_the_native_rate(self):
        _samples, sr = decode_audio(generate_wav(440.0, 0.25), sr=None)
        assert sr == GENERATOR_SAMPLE_RATE

    def test_duration_truncates_before_resampling(self):
        samples, sr = decode_audio(generate_wav(440.0, 2.0), sr=8000, duration=0.5)
        assert sr == 8000
        assert samples.size == pytest.approx(4000, abs=2)

    def test_offset_drops_the_leading_window(self):
        full, sr = decode_audio(generate_wav(440.0, 1.0))
        tail, _sr = decode_audio(generate_wav(440.0, 1.0), offset=0.25)
        assert tail.size == full.size - int(0.25 * sr)

    def test_mono_false_returns_channel_major(self):
        samples, _sr = decode_audio(generate_wav(440.0, 0.25), mono=False)
        assert samples.ndim == 2
        assert samples.shape[0] == 1  # the generator writes mono WAVs

    def test_samples_are_writable(self):
        """Callers pass these straight to ``torch.from_numpy``, which warns on a
        read-only buffer — so the array must own writable memory."""
        samples, _sr = decode_audio(generate_wav(440.0, 0.25))
        assert samples.flags["WRITEABLE"]


class TestFailures:
    def test_empty_source_raises(self):
        with pytest.raises(AudioDecodeError):
            decode_audio(b"")

    def test_garbage_source_raises(self):
        with pytest.raises(AudioDecodeError):
            decode_audio(b"not audio data" * 64)

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(AudioDecodeError):
            decode_audio(tmp_path / "nope.wav")


class TestFfmpegPath:
    """AAC/M4A: libsndfile can't parse it, so these all run through ffmpeg."""

    def test_decodes_aac_from_memory_without_a_tempfile(self, aac_bytes, monkeypatch):
        """The whole point of piping to ``-i pipe:0``: no filesystem round-trip.

        ``_decode_audio_file_bytes``'s temp-file spill existed only because
        librosa's audioread fallback demanded a real path.  Booby-trap
        ``NamedTemporaryFile`` to prove the common path never reaches for one.
        """
        import tempfile

        def _explode(*_args, **_kwargs):
            raise AssertionError("AAC decode must not spill to a temp file")

        monkeypatch.setattr(tempfile, "NamedTemporaryFile", _explode)

        samples, sr = decode_audio(aac_bytes, sr=None, mono=True)
        assert samples.dtype == np.float32
        assert sr == GENERATOR_SAMPLE_RATE
        # ~1 s of tone, modulo the encoder's priming samples.
        assert samples.size == pytest.approx(GENERATOR_SAMPLE_RATE, rel=0.05)
        assert np.abs(samples).max() > 0.1

    def test_path_and_bytes_agree(self, aac_bytes, tmp_path):
        path = tmp_path / "tone.m4a"
        path.write_bytes(aac_bytes)
        from_bytes, sr_bytes = decode_audio(aac_bytes)
        from_path, sr_path = decode_audio(path)
        assert sr_bytes == sr_path
        assert np.array_equal(from_bytes, from_path)

    def test_samples_are_writable(self, aac_bytes):
        samples, _sr = decode_audio(aac_bytes)
        assert samples.flags["WRITEABLE"]

    def test_resamples_via_ffmpeg(self, aac_bytes):
        samples, sr = decode_audio(aac_bytes, sr=16000)
        assert sr == 16000
        assert samples.size == pytest.approx(16000, rel=0.05)

    def test_duration_limits_the_ffmpeg_decode(self, aac_bytes):
        samples, sr = decode_audio(aac_bytes, duration=0.25)
        assert samples.size == pytest.approx(0.25 * sr, rel=0.05)

    def test_spills_to_a_tempfile_when_the_pipe_decode_fails(self, aac_bytes, monkeypatch):
        """Containers needing a seekable input (a trailing ``moov`` atom) retry
        through a temp file rather than failing outright."""
        from vtscore.media.audio import decode as decode_mod

        real_run = decode_mod._run_ffmpeg
        seen = []

        def fake_run(input_arg, stdin_bytes, **kwargs):
            seen.append(input_arg)
            if input_arg == "pipe:0":
                raise subprocess.CalledProcessError(1, "ffmpeg", stderr=b"moov atom not found")
            return real_run(input_arg, stdin_bytes, **kwargs)

        monkeypatch.setattr(decode_mod, "_run_ffmpeg", fake_run)

        samples, _sr = decode_audio(aac_bytes)
        assert samples.size > 0
        assert seen[0] == "pipe:0"  # tried the pipe first
        assert len(seen) == 2 and seen[1] != "pipe:0"  # then the temp file


class TestWaveParsing:
    def test_rejects_non_wave_output(self):
        from vtscore.media.audio.decode import _parse_wave

        with pytest.raises(AudioDecodeError):
            _parse_wave(b"RIFF\x00\x00\x00\x00NOPEfmt ")

    def test_sizes_the_data_chunk_from_what_arrived(self):
        """ffmpeg can't backfill chunk sizes on a pipe, so it writes a
        placeholder; the parser must trust the byte count, not the header."""
        import struct

        from vtscore.media.audio.decode import _parse_wave

        pcm = np.array([0.25, -0.5, 0.75], dtype="<f4").tobytes()
        fmt = struct.pack("<HHIIHH", 3, 1, 44100, 44100 * 4, 4, 32)
        blob = (
            b"RIFF"
            + struct.pack("<I", 0xFFFFFFFF)
            + b"WAVE"
            + b"fmt "
            + struct.pack("<I", len(fmt))
            + fmt
            + b"data"
            + struct.pack("<I", 0xFFFFFFFF)
            + pcm
        )
        samples, sr, channels = _parse_wave(blob)
        assert sr == 44100
        assert channels == 1
        assert np.allclose(samples, [0.25, -0.5, 0.75])
