import io
import wave

from vtscore.media.audio.audio_generator import GENERATOR_SAMPLE_RATE, generate_wav
from vtscore.media.audio.media_type import generate_waveform_thumbnail


class TestGenerateWaveformThumbnail:
    def test_returns_png_bytes(self):
        wav_bytes = generate_wav(440.0, 1.0)
        result = generate_waveform_thumbnail(wav_bytes)
        assert result is not None
        assert isinstance(result, bytes)
        # PNG magic number
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_returns_square_image(self):
        from PIL import Image

        wav_bytes = generate_wav(440.0, 1.0)
        result = generate_waveform_thumbnail(wav_bytes)
        assert result is not None
        img = Image.open(io.BytesIO(result))
        assert img.size == (128, 128)

    def test_is_theme_agnostic_alpha_mask(self):
        """The thumbnail bakes in no colour: transparent background, opaque wave.

        This is the contract the frontend relies on to tint the waveform to the
        live theme (issue #2369) — an RGBA image whose corners are fully
        transparent and whose alpha channel actually varies (the wave is drawn).
        """
        from PIL import Image

        wav_bytes = generate_wav(440.0, 1.0)
        result = generate_waveform_thumbnail(wav_bytes)
        assert result is not None
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGBA"
        # Corners are background → fully transparent, and *white* RGB (matching
        # the wave) so downscaling can't fringe the wave with dark pixels.
        w, h = img.size
        for xy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
            assert img.getpixel(xy) == (255, 255, 255, 0)
        # The wave itself is drawn → some pixels are opaque.
        alphas = img.getchannel("A").getextrema()
        assert alphas == (0, 255)
        # Colour is carried entirely by alpha: every pixel is white RGB.
        rgb = img.convert("RGB")
        assert rgb.getextrema() == ((255, 255), (255, 255), (255, 255))

    def test_custom_size(self):
        from PIL import Image

        wav_bytes = generate_wav(440.0, 1.0)
        result = generate_waveform_thumbnail(wav_bytes, size=64)
        assert result is not None
        img = Image.open(io.BytesIO(result))
        assert img.size == (64, 64)

    def test_deterministic_output(self):
        """Same audio should produce the exact same thumbnail."""
        wav_bytes = generate_wav(440.0, 1.0)
        thumb_a = generate_waveform_thumbnail(wav_bytes)
        thumb_b = generate_waveform_thumbnail(wav_bytes)
        assert thumb_a == thumb_b

    def test_returns_none_for_invalid_audio(self):
        result = generate_waveform_thumbnail(b"not audio data")
        assert result is None

    def test_returns_none_for_empty_bytes(self):
        result = generate_waveform_thumbnail(b"")
        assert result is None

    def test_short_audio(self):
        """Very short audio should still produce a thumbnail."""
        wav_bytes = generate_wav(440.0, 0.01)
        result = generate_waveform_thumbnail(wav_bytes)
        assert result is not None
        assert result[:8] == b"\x89PNG\r\n\x1a\n"

    def test_falls_back_to_tempfile_when_buffer_decode_fails(self, monkeypatch):
        """AAC/M4A can't decode from a BytesIO (librosa only reaches ffmpeg via a
        real path), so a buffer-decode failure must retry through a temp file
        rather than silently returning None."""
        import librosa
        import numpy as np

        calls = {"buffer": 0, "path": 0}

        def fake_load(path, **kwargs):
            if isinstance(path, io.BytesIO):
                calls["buffer"] += 1
                raise RuntimeError("libsndfile cannot parse AAC-in-MP4 from a buffer")
            calls["path"] += 1
            rng = np.random.default_rng(0)
            return rng.standard_normal(4000).astype(np.float32), 44100

        monkeypatch.setattr(librosa, "load", fake_load)

        result = generate_waveform_thumbnail(b"fake aac bytes", filename="clip.m4a")
        assert result is not None
        assert result[:8] == b"\x89PNG\r\n\x1a\n"
        assert calls["buffer"] == 1  # tried the buffer first
        assert calls["path"] == 1  # then fell back to the temp file


class TestGenerateWav:
    def test_returns_valid_wav(self):
        data = generate_wav(440.0, 1.0)
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == GENERATOR_SAMPLE_RATE

    def test_duration_determines_frame_count(self):
        for dur in (0.5, 1.0, 2.0):
            data = generate_wav(440.0, dur)
            buf = io.BytesIO(data)
            with wave.open(buf, "rb") as wf:
                expected = int(GENERATOR_SAMPLE_RATE * dur)
                assert wf.getnframes() == expected

    def test_different_frequencies_produce_different_output(self):
        wav_a = generate_wav(200.0, 0.5)
        wav_b = generate_wav(800.0, 0.5)
        assert wav_a != wav_b

    def test_zero_duration_produces_empty_frames(self):
        data = generate_wav(440.0, 0.0)
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0
