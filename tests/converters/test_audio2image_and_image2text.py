"""Tests for the audio→image (spectrogram) and image→text (OCR) converters."""

from __future__ import annotations

import io
import wave
from unittest.mock import patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sine_wav(duration_s: float = 0.5, sample_rate: int = 16000, freq: float = 440.0) -> bytes:
    """Build a tiny mono 16-bit PCM WAV containing a sine wave."""
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, duration_s, int(duration_s * sample_rate), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)
    # Sprinkle a touch of noise so the spectrogram has structure beyond a single bin
    signal = signal + 0.01 * rng.standard_normal(len(signal))
    samples = (signal * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_png_bytes(width: int = 16, height: int = 16, color: tuple[int, int, int] = (200, 200, 200)) -> bytes:
    from PIL import Image

    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fake_paddleocr_module(fake_cls: type) -> object:
    """Build a stand-in module object that exposes ``PaddleOCR`` for patch.dict."""
    import types

    mod = types.ModuleType("paddleocr")
    mod.PaddleOCR = fake_cls  # pyright: ignore[reportAttributeAccessIssue]
    return mod


# ===========================================================================
# Audio2ImageMediaConverter
# ===========================================================================


class TestAudio2ImageMediaConverter:
    def test_source_and_target_types(self):
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        c = Audio2ImageMediaConverter()
        assert c.source_type == "audio"
        assert c.target_type == "image"

    def test_name_and_display(self):
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        c = Audio2ImageMediaConverter()
        assert c.name == "audio2image"
        assert "Spectrogram" in c.display_name or "spectrogram" in c.display_name

    def test_fields_have_expected_keys(self):
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        c = Audio2ImageMediaConverter()
        keys = {f.key for f in c.fields}
        assert {"spectrogram_type", "n_mels", "time_window_s", "colormap"} <= keys

    def test_field_defaults(self):
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        c = Audio2ImageMediaConverter()
        assert c.get_param({}, "spectrogram_type") == "mel"
        assert c.get_param({}, "n_mels") == "128"
        assert c.get_param({}, "time_window_s") == "30"
        assert c.get_param({}, "colormap") == "magma"

    def test_convert_no_data(self):
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        c = Audio2ImageMediaConverter()
        assert c.convert({"filename": "x.wav"}) == []

    def test_convert_empty_bytes(self):
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        c = Audio2ImageMediaConverter()
        assert c.convert({"filename": "x.wav", "media_bytes": b""}) == []

    def test_convert_mel_spectrogram(self):
        pytest.importorskip("librosa")
        pytest.importorskip("matplotlib")
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        wav_bytes = _make_sine_wav(duration_s=0.5)
        media = {"filename": "tone.wav", "media_bytes": wav_bytes}
        c = Audio2ImageMediaConverter()
        results = c.convert(media)

        assert len(results) == 1
        r = results[0]
        assert r["filename"] == "tone_spec_mel.png"
        assert isinstance(r["media_bytes"], bytes)
        assert r["media_bytes"][:4] == b"\x89PNG"
        assert r["duration"] == 0
        assert r["width"] is None or r["width"] > 0
        assert r["height"] is None or r["height"] > 0

    def test_convert_cqt_spectrogram(self):
        pytest.importorskip("librosa")
        pytest.importorskip("matplotlib")
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        wav_bytes = _make_sine_wav(duration_s=0.5)
        media = {"filename": "tone.wav", "media_bytes": wav_bytes}
        c = Audio2ImageMediaConverter()
        results = c.convert(media, {"spectrogram_type": "cqt"})

        assert len(results) == 1
        r = results[0]
        assert r["filename"] == "tone_spec_cqt.png"
        assert r["media_bytes"][:4] == b"\x89PNG"

    def test_convert_custom_colormap(self):
        pytest.importorskip("librosa")
        pytest.importorskip("matplotlib")
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        wav_bytes = _make_sine_wav(duration_s=0.3)
        media = {"filename": "tone.wav", "media_bytes": wav_bytes}
        c = Audio2ImageMediaConverter()
        results = c.convert(media, {"colormap": "viridis"})
        assert len(results) == 1
        assert results[0]["media_bytes"][:4] == b"\x89PNG"

    def test_convert_time_window_truncates(self):
        """A small time_window_s caps how much audio gets rendered."""
        pytest.importorskip("librosa")
        pytest.importorskip("matplotlib")
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        wav_bytes = _make_sine_wav(duration_s=1.0)
        media = {"filename": "tone.wav", "media_bytes": wav_bytes}
        c = Audio2ImageMediaConverter()
        results = c.convert(media, {"time_window_s": "0.1"})
        assert len(results) == 1
        assert results[0]["media_bytes"][:4] == b"\x89PNG"

    def test_convert_from_path(self, tmp_path):
        pytest.importorskip("librosa")
        pytest.importorskip("matplotlib")
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        wav_bytes = _make_sine_wav(duration_s=0.3)
        wav_path = tmp_path / "tone.wav"
        wav_path.write_bytes(wav_bytes)
        media = {"filename": "tone.wav", "media_bytes": None, "media_path": str(wav_path)}
        c = Audio2ImageMediaConverter()
        results = c.convert(media)
        assert len(results) == 1
        assert results[0]["media_bytes"][:4] == b"\x89PNG"

    def test_convert_bad_audio_returns_empty(self):
        pytest.importorskip("librosa")
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        # Random bytes are not a decodable audio container
        media = {"filename": "bad.wav", "media_bytes": b"not really audio data" * 32}
        c = Audio2ImageMediaConverter()
        assert c.convert(media) == []

    def test_invalid_spectrogram_type_falls_back_to_mel(self):
        pytest.importorskip("librosa")
        pytest.importorskip("matplotlib")
        from vtsearch.converters.audio2image import Audio2ImageMediaConverter

        wav_bytes = _make_sine_wav(duration_s=0.3)
        media = {"filename": "tone.wav", "media_bytes": wav_bytes}
        c = Audio2ImageMediaConverter()
        results = c.convert(media, {"spectrogram_type": "garbage"})
        assert len(results) == 1
        assert results[0]["filename"].endswith("_spec_mel.png")


# ===========================================================================
# Image2TextMediaConverter
# ===========================================================================


class TestImage2TextMediaConverter:
    def test_source_and_target_types(self):
        from vtsearch.converters.image2text import Image2TextMediaConverter

        c = Image2TextMediaConverter()
        assert c.source_type == "image"
        assert c.target_type == "text"

    def test_name_and_display(self):
        from vtsearch.converters.image2text import Image2TextMediaConverter

        c = Image2TextMediaConverter()
        assert c.name == "image2text"
        assert "OCR" in c.display_name

    def test_fields_have_expected_keys(self):
        from vtsearch.converters.image2text import Image2TextMediaConverter

        c = Image2TextMediaConverter()
        keys = {f.key for f in c.fields}
        assert {"language", "threshold"} <= keys

    def test_field_defaults(self):
        from vtsearch.converters.image2text import Image2TextMediaConverter

        c = Image2TextMediaConverter()
        assert c.get_param({}, "language") == "en"
        assert c.get_param({}, "threshold") == "0.5"

    def test_convert_no_data(self):
        from vtsearch.converters.image2text import Image2TextMediaConverter

        c = Image2TextMediaConverter()
        assert c.convert({"filename": "x.png"}) == []

    def test_convert_empty_bytes(self):
        from vtsearch.converters.image2text import Image2TextMediaConverter

        c = Image2TextMediaConverter()
        assert c.convert({"filename": "x.png", "media_bytes": b""}) == []

    def test_convert_no_paddleocr_returns_empty(self):
        """When PaddleOCR is not importable, convert returns []."""
        from vtsearch.converters.image2text import Image2TextMediaConverter

        png_bytes = _make_png_bytes()
        media = {"filename": "x.png", "media_bytes": png_bytes}
        c = Image2TextMediaConverter()

        # Setting the module to None in sys.modules makes `import paddleocr` raise.
        with patch.dict("sys.modules", {"paddleocr": None}):
            assert c.convert(media) == []

    def test_convert_with_mocked_paddleocr(self):
        """Mock PaddleOCR to verify result-flattening and threshold filtering."""
        from vtsearch.converters.image2text import Image2TextMediaConverter

        png_bytes = _make_png_bytes()
        media = {"filename": "screenshot.png", "media_bytes": png_bytes}
        c = Image2TextMediaConverter()

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                pass

            def ocr(self, _img, cls=True):
                return [
                    [
                        ([[0, 0], [10, 0], [10, 5], [0, 5]], ("Hello", 0.99)),
                        ([[0, 10], [10, 10], [10, 15], [0, 15]], ("World", 0.95)),
                        ([[0, 20], [10, 20], [10, 25], [0, 25]], ("low conf", 0.2)),
                    ],
                ]

        with patch.dict("sys.modules", {"paddleocr": _fake_paddleocr_module(FakePaddleOCR)}):
            results = c.convert(media, {"threshold": "0.5"})

        assert len(results) == 1
        r = results[0]
        assert r["filename"] == "screenshot.txt"
        text = r["media_string"]
        assert "Hello" in text
        assert "World" in text
        assert "low conf" not in text
        assert r["word_count"] >= 2
        assert r["character_count"] > 0
        assert r["duration"] == 0

    def test_convert_with_mocked_paddleocr_no_hits(self):
        """When OCR finds no text above threshold, returns []."""
        from vtsearch.converters.image2text import Image2TextMediaConverter

        png_bytes = _make_png_bytes()
        media = {"filename": "blank.png", "media_bytes": png_bytes}
        c = Image2TextMediaConverter()

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                pass

            def ocr(self, _img, cls=True):
                return [[]]

        with patch.dict("sys.modules", {"paddleocr": _fake_paddleocr_module(FakePaddleOCR)}):
            assert c.convert(media) == []

    def test_convert_passes_language_to_paddleocr(self):
        """The language param flows into the PaddleOCR constructor."""
        from vtsearch.converters.image2text import Image2TextMediaConverter

        png_bytes = _make_png_bytes()
        media = {"filename": "x.png", "media_bytes": png_bytes}
        c = Image2TextMediaConverter()

        captured: dict = {}

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def ocr(self, _img, cls=True):
                return [[]]

        with patch.dict("sys.modules", {"paddleocr": _fake_paddleocr_module(FakePaddleOCR)}):
            c.convert(media, {"language": "fr"})

        assert captured.get("lang") == "fr"

    def test_convert_from_path(self, tmp_path):
        from vtsearch.converters.image2text import Image2TextMediaConverter

        png_bytes = _make_png_bytes()
        png_path = tmp_path / "x.png"
        png_path.write_bytes(png_bytes)
        media = {"filename": "x.png", "media_bytes": None, "media_path": str(png_path)}
        c = Image2TextMediaConverter()

        class FakePaddleOCR:
            def __init__(self, **kwargs):
                pass

            def ocr(self, _img, cls=True):
                return [[([[0, 0], [10, 0], [10, 5], [0, 5]], ("Path text", 0.9))]]

        with patch.dict("sys.modules", {"paddleocr": _fake_paddleocr_module(FakePaddleOCR)}):
            results = c.convert(media)

        assert len(results) == 1
        assert "Path text" in results[0]["media_string"]


# ===========================================================================
# Registry / package integration
# ===========================================================================


class TestNewConverterRegistryIntegration:
    def test_audio2image_in_registry(self):
        from vtsearch.converters import get_converter, list_converters

        assert get_converter("audio2image") is not None
        names = [c.name for c in list_converters()]
        assert "audio2image" in names

    def test_image2text_in_registry(self):
        from vtsearch.converters import get_converter, list_converters

        assert get_converter("image2text") is not None
        names = [c.name for c in list_converters()]
        assert "image2text" in names

    def test_audio2image_listed_for_image_target(self):
        from vtsearch.converters import list_converters_for_target

        names = [c.name for c in list_converters_for_target("image")]
        assert "audio2image" in names

    def test_audio2image_listed_for_audio_source(self):
        from vtsearch.converters import list_converters_for_source

        names = [c.name for c in list_converters_for_source("audio")]
        assert "audio2image" in names

    def test_image2text_listed_for_text_target(self):
        from vtsearch.converters import list_converters_for_target

        names = [c.name for c in list_converters_for_target("text")]
        assert "image2text" in names

    def test_image2text_listed_for_image_source(self):
        from vtsearch.converters import list_converters_for_source

        names = [c.name for c in list_converters_for_source("image")]
        assert "image2text" in names

    def test_api_lists_new_converters(self, client):
        resp = client.get("/api/converters")
        assert resp.status_code == 200
        names = [c["name"] for c in resp.get_json()["converters"]]
        assert "audio2image" in names
        assert "image2text" in names
