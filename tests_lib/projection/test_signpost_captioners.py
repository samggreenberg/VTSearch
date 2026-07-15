"""Library-tier tests for the generative captioner signpost providers.

Covers ``vtscore.projection.signpost_captioners``: the media-decode helpers
(run for real against tiny in-memory fixtures) and the provider ``build_texts``
batching / filtering / early-exit logic (with the heavy model seams stubbed, so
no multi-GB download).  No toponymy, no real VLM.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from vtscore.projection import signpost_captioners as sc


def _png_bytes(color=(200, 30, 30), size=(8, 8)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _wav_bytes(seconds=0.5, sr=16000) -> bytes:
    import soundfile as sf

    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(int(seconds * sr)) * 0.01).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue()


class TestImageDecode:
    def test_decodes_media_bytes_to_rgb(self):
        img = sc._load_image({"media_bytes": _png_bytes(size=(8, 6))})
        assert img is not None
        assert img.mode == "RGB"
        assert img.size == (8, 6)

    def test_decodes_media_path(self, tmp_path):
        p = tmp_path / "x.png"
        p.write_bytes(_png_bytes())
        img = sc._load_image({"media_path": str(p)})
        assert img is not None and img.mode == "RGB"

    def test_missing_source_returns_none(self):
        assert sc._load_image({}) is None

    def test_undecodable_bytes_returns_none(self):
        assert sc._load_image({"media_bytes": b"not an image"}) is None


class TestAudioDecode:
    def test_decodes_media_bytes_to_mono_float32(self):
        wav = sc._load_audio({"media_bytes": _wav_bytes()})
        assert wav is not None
        assert wav.dtype == np.float32
        assert wav.ndim == 1

    def test_missing_source_returns_none(self):
        assert sc._load_audio({}) is None

    def test_undecodable_bytes_returns_none(self):
        assert sc._load_audio({"media_bytes": b"not audio"}) is None


class TestImageCaptionProvider:
    def _stub_model(self, monkeypatch, gen):
        loaded = {"n": 0}

        def _load(model_id, on_progress=None):
            loaded["n"] += 1
            return ("model", "processor", "cpu")

        monkeypatch.setattr(sc, "_load_image_model", _load)
        monkeypatch.setattr(sc, "_generate_image_captions", gen)
        return loaded

    def test_batches_and_stamps_decodable_only(self, monkeypatch):
        batches: list[int] = []

        def gen(model, processor, device, images, prompt, max_new_tokens):
            batches.append(len(images))
            return [f"a photo {i}" for i in range(len(images))]

        self._stub_model(monkeypatch, gen)
        provider = sc.ImageCaptionProvider(batch_size=2)
        png = _png_bytes()
        medias = {1: {"media_bytes": png}, 2: {"media_bytes": png}, 3: {"media_bytes": png}, 4: {}}
        texts = provider.build_texts([1, 2, 3, 4], medias, np.zeros((4, 4), dtype=np.float32), None)

        assert set(texts) == {1, 2, 3}  # id 4 has no decodable image
        assert batches == [2, 1]  # batch_size=2 over 3 images

    def test_empty_captions_are_dropped(self, monkeypatch):
        self._stub_model(monkeypatch, lambda *a, **k: ["", "  "])
        provider = sc.ImageCaptionProvider(batch_size=8)
        png = _png_bytes()
        texts = provider.build_texts([1, 2], {1: {"media_bytes": png}, 2: {"media_bytes": png}}, np.zeros((2, 4)), None)
        assert texts == {}

    def test_no_decodable_media_skips_model_load(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("model must not load when nothing decodes")

        monkeypatch.setattr(sc, "_load_image_model", _boom)
        provider = sc.ImageCaptionProvider()
        assert provider.build_texts([1, 2], {1: {}, 2: {}}, np.zeros((2, 4)), None) == {}

    def test_signature_is_model_id_not_embedder(self):
        provider = sc.ImageCaptionProvider()

        class E:
            name = "siglip"

        assert provider.signature(E()) == "caption:qwen2.5-vl-3b"


class TestAudioCaptionProvider:
    def test_batches_and_stamps_decodable_only(self, monkeypatch):
        batches: list[int] = []

        def gen(model, tokenizer, fe, device, style_ids, waveforms):
            batches.append(len(waveforms))
            return [f"a sound {i}" for i in range(len(waveforms))]

        monkeypatch.setattr(sc, "_load_audio_model", lambda mid, on_progress=None: ("m", "t", "fe", "cpu", "s"))
        monkeypatch.setattr(sc, "_generate_audio_captions", gen)
        provider = sc.AudioCaptionProvider(batch_size=2)
        wav = _wav_bytes()
        medias = {1: {"media_bytes": wav}, 2: {"media_bytes": wav}, 3: {}}
        texts = provider.build_texts([1, 2, 3], medias, np.zeros((3, 4)), None)

        assert set(texts) == {1, 2}
        assert batches == [2]

    def test_no_decodable_media_skips_model_load(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("model must not load when nothing decodes")

        monkeypatch.setattr(sc, "_load_audio_model", _boom)
        provider = sc.AudioCaptionProvider()
        assert provider.build_texts([1], {1: {}}, np.zeros((1, 4)), None) == {}
