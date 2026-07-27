import io
import wave

from vtscore.media.audio.audio_generator import GENERATOR_SAMPLE_RATE, generate_wav
from vtscore.media.audio.media_type import _render_waveform, generate_waveform_thumbnail


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

    def test_loud_audio_is_not_a_solid_block(self):
        """Loud, dense broadband audio must still render as a waveform, not a
        filled rectangle.

        The old min/max envelope pinned every column top-to-bottom on loud
        real-world clips (a single peak sample per column), so the alpha mask was
        fully opaque and tinted to a solid rectangle (issue #2555).  The RMS
        envelope leaves a transparent margin above and below the wave, so the
        thumbnail reads as a waveform.
        """
        import numpy as np
        from PIL import Image

        rng = np.random.default_rng(0)
        # Loud broadband noise — the ESC-50 "rain / fire / insects" case that
        # saturated the peak envelope.
        loud = np.clip(rng.standard_normal(220_500).astype(np.float32) * 0.5, -1.0, 1.0)
        result = _render_waveform(loud)
        assert result is not None
        img = Image.open(io.BytesIO(result))
        alpha = np.array(img.getchannel("A"))
        # The wave doesn't reach the frame edges: the top and bottom rows are
        # fully transparent (no opaque pixel), which a solid block would violate.
        assert alpha[0, :].max() == 0
        assert alpha[-1, :].max() == 0
        # And well under all the pixels are opaque — nowhere near a filled block.
        assert (alpha > 0).mean() < 0.9

    def test_decodes_ffmpeg_only_codec_from_memory(self, aac_bytes):
        """AAC/M4A - which libsndfile can't parse at all - still thumbnails.

        ffmpeg reads the buffer off ``stdin``, so the codec that used to force
        librosa's (now-removed) audioread fallback plus a temp-file spill now
        renders straight from memory.
        """
        result = generate_waveform_thumbnail(aac_bytes)
        assert result is not None
        assert result[:8] == b"\x89PNG\r\n\x1a\n"


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
