"""CPU unit tests for embedder wrapper pre/post-processing.

Every real embedder model is stubbed session-wide (see
``tests_lib/conftest.py``) and the GPU suite loads CLAP/CLIP/X-CLIP/E5
straight from ``transformers``, bypassing VTSearch's own wrapper code.  That
left the *plumbing* around each forward pass - input prep, device movement,
pooling, CLS extraction, L2-normalisation, batching, and the try/except
fallbacks - almost entirely uncovered.

These tests inject a **fake tiny model** (and processor / tokenizer / feature
extractor) into a *fresh* embedder instance, then call the wrapper methods
directly.  No weights are downloaded and everything runs on CPU in
milliseconds.  Full model loads stay behind the ``gpu``/``slow`` markers; here
we only exercise the deterministic Python glue that turns a media dict into a
normalised vector.

The fresh instances are deliberately *not* the registered singletons, so the
``_stub_embedding_models`` session fixture (which patches ``embed_media`` /
``embed_text`` / ``load_models`` on the singletons) does not touch them - the
real wrapper code runs.
"""

from __future__ import annotations

import io
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed_torch():
    """Seed torch so the few tests that build real ``nn`` layers are deterministic."""
    torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


class _FakeBatch(dict):
    """A ``dict`` that also allows attribute access.

    Mimics ``transformers.BatchFeature``: the wrappers reach processor output
    both ways - ``{k: v.to(device) for k, v in inputs.items()}`` (dict) and
    ``inputs.input_features`` / ``inputs.input_values`` (attribute).
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


class _FakeProcessor:
    """Callable returning a fixed ``_FakeBatch`` of CPU tensors.

    Records every ``__call__`` so tests can assert on the kwargs the wrapper
    passed (prefixes, padding/truncation flags, sampling rate, …).
    """

    def __init__(self, keys=("pixel_values",), shape=(1, 2, 3)):
        self.keys = keys
        self.shape = shape
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _FakeBatch({k: torch.zeros(*self.shape) for k in self.keys})


def _cpu_param():
    """A single CPU parameter tensor so ``next(model.parameters()).device`` works."""
    return torch.zeros(1)


def _fake_librosa_load(monkeypatch, samples=1000):
    """Patch ``librosa.load`` to return a fixed-length silent mono clip."""
    import librosa

    def _load(source, sr=None, mono=True):
        return np.zeros(samples, dtype=np.float32), sr

    monkeypatch.setattr(librosa, "load", _load)


def _png_bytes(size=(8, 8), color=(120, 30, 200)) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color=color).save(buf, format="PNG")
    return buf.getvalue()


# ===========================================================================
# CLAP (_clap_shared.py, via AudioClapEmbedder)
# ===========================================================================


class _FakeClapModel:
    def __init__(self, dim=512):
        self.dim = dim
        self._p = _cpu_param()
        self.audio_calls: list[dict] = []
        self.text_calls: list[dict] = []

    def parameters(self):
        return iter([self._p])

    def audio_model(self, **kwargs):
        self.audio_calls.append(kwargs)
        return SimpleNamespace(pooler_output=torch.ones(1, 4))

    def audio_projection(self, x):
        return torch.arange(self.dim, dtype=torch.float32).reshape(1, self.dim)

    def text_model(self, **kwargs):
        self.text_calls.append(kwargs)
        return SimpleNamespace(pooler_output=torch.ones(1, 4))

    def text_projection(self, x):
        return (torch.arange(self.dim, dtype=torch.float32) + 1.0).reshape(1, self.dim)


def _make_clap(dim=512):
    from vtscore.media.audio.embedder_clap import AudioClapEmbedder

    emb = AudioClapEmbedder()
    emb._model = _FakeClapModel(dim)
    emb._processor = _FakeProcessor(keys=("input_features",))
    return emb


class TestClapWrapper:
    def test_embed_media_from_path(self, monkeypatch):
        _fake_librosa_load(monkeypatch, samples=1000)
        emb = _make_clap(dim=512)
        vec = emb.embed_media({"media_path": "/fake.wav"})
        assert vec is not None
        assert vec.shape == (512,)
        # embed_media() L2-normalises the raw projection output.
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_forwards_audio_kwarg(self, monkeypatch):
        _fake_librosa_load(monkeypatch, samples=1000)
        emb = _make_clap()
        emb.embed_media({"media_path": "/fake.wav"})
        # The processor gets the deterministic-truncation settings.
        _args, kwargs = emb._processor.calls[0]
        assert kwargs["truncation"] == "rand_trunc"
        assert kwargs["padding"] == "max_length"
        assert "sampling_rate" in kwargs
        assert "max_length" in kwargs

    def test_embed_media_truncates_long_audio(self, monkeypatch):
        from vtscore.media.audio._clap_shared import CLAP_MAX_SAMPLES

        _fake_librosa_load(monkeypatch, samples=CLAP_MAX_SAMPLES + 5000)
        emb = _make_clap()
        vec = emb.embed_media({"media_path": "/long.wav"})
        assert vec is not None
        # The audio handed to the processor is capped at CLAP_MAX_SAMPLES.
        _args, kwargs = emb._processor.calls[0]
        assert kwargs["audio"].shape[0] == CLAP_MAX_SAMPLES

    def test_embed_media_from_bytes(self, monkeypatch):
        _fake_librosa_load(monkeypatch, samples=500)
        emb = _make_clap()
        vec = emb.embed_media({"media_bytes": b"RIFFfake"})
        assert vec is not None
        assert vec.shape == (512,)

    def test_embed_media_no_source_returns_none(self):
        emb = _make_clap()
        assert emb.embed_media({}) is None

    def test_embed_media_swallows_errors(self, monkeypatch):
        import librosa

        def _boom(*a, **k):
            raise RuntimeError("decode failed")

        monkeypatch.setattr(librosa, "load", _boom)
        emb = _make_clap()
        assert emb.embed_media({"media_path": "/fake.wav"}) is None

    def test_embed_text(self):
        emb = _make_clap(dim=512)
        vec = emb.embed_text("a dog barking")
        assert vec is not None
        assert vec.shape == (512,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)
        assert emb._model.text_calls, "text_model was never called"

    def test_embed_media_returns_none_when_processor_missing(self):
        emb = _make_clap()
        emb._processor = None
        assert emb.embed_media({"media_path": "/fake.wav"}) is None


# ===========================================================================
# AST (embedder_ast.py)
# ===========================================================================


class _FakeASTModel:
    def __init__(self, dim=768):
        self.dim = dim
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def __call__(self, **kwargs):
        return SimpleNamespace(pooler_output=torch.arange(self.dim, dtype=torch.float32).reshape(1, self.dim))


class TestASTWrapper:
    def _make(self, dim=768):
        from vtscore.media.audio.embedder_ast import AudioASTEmbedder

        emb = AudioASTEmbedder()
        emb._model = _FakeASTModel(dim)
        emb._processor = _FakeProcessor(keys=("input_values",))
        return emb

    def test_embed_media(self, monkeypatch):
        _fake_librosa_load(monkeypatch, samples=2000)
        emb = self._make(dim=768)
        vec = emb.embed_media({"media_path": "/fake.wav"})
        assert vec is not None
        assert vec.shape == (768,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_from_bytes(self, monkeypatch):
        _fake_librosa_load(monkeypatch, samples=2000)
        emb = self._make()
        vec = emb.embed_media({"media_bytes": b"RIFFfake"})
        assert vec is not None

    def test_embed_media_no_source(self):
        assert self._make().embed_media({}) is None

    def test_embed_media_swallows_errors(self, monkeypatch):
        import librosa

        monkeypatch.setattr(librosa, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert self._make().embed_media({"media_path": "/fake.wav"}) is None

    def test_text_unsupported(self):
        assert self._make().embed_text("anything") is None


# ===========================================================================
# Whisper encoder (embedder_whisper.py)
# ===========================================================================


class _FakeWhisperEncoder:
    def __init__(self, dim=512, seq=5):
        self.dim = dim
        self.seq = seq
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def __call__(self, input_features):
        # Row r is filled with value r so the time-mean is deterministic.
        base = torch.arange(self.seq, dtype=torch.float32).reshape(1, self.seq, 1)
        lhs = base.expand(1, self.seq, self.dim).contiguous()
        return SimpleNamespace(last_hidden_state=lhs)


class TestWhisperWrapper:
    def _make(self, dim=512, seq=5):
        from vtscore.media.audio.embedder_whisper import AudioWhisperEncoderEmbedder

        emb = AudioWhisperEncoderEmbedder()
        emb._model = _FakeWhisperEncoder(dim, seq)
        emb._processor = _FakeProcessor(keys=("input_features",), shape=(1, 4, 5))
        return emb

    def test_embed_media_mean_pools_time_axis(self, monkeypatch):
        _fake_librosa_load(monkeypatch, samples=1600)
        emb = self._make(dim=512, seq=5)
        vec = emb.embed_media({"media_path": "/fake.wav"})
        assert vec is not None
        assert vec.shape == (512,)
        # Pre-normalisation every element equals mean(0..4) = 2.0; L2-normalised
        # that is a uniform unit vector, so the vector stays uniform.
        assert np.allclose(vec, vec[0])
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_from_bytes(self, monkeypatch):
        _fake_librosa_load(monkeypatch, samples=1600)
        vec = self._make().embed_media({"media_bytes": b"RIFF"})
        assert vec is not None

    def test_embed_media_no_source(self):
        assert self._make().embed_media({}) is None

    def test_embed_media_swallows_errors(self, monkeypatch):
        import librosa

        monkeypatch.setattr(librosa, "load", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert self._make().embed_media({"media_path": "/fake.wav"}) is None

    def test_text_unsupported(self):
        assert self._make().embed_text("anything") is None


# ===========================================================================
# SigLIP 2 (embedder_siglip2.py)
# ===========================================================================


class _FakeVisionTextModel:
    """Fake CLIP-style model exposing get_image_features / get_text_features."""

    def __init__(self, dim=768):
        self.dim = dim
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def get_image_features(self, **kwargs):
        return torch.arange(self.dim, dtype=torch.float32).reshape(1, self.dim)

    def get_text_features(self, **kwargs):
        return (torch.arange(self.dim, dtype=torch.float32) + 1.0).reshape(1, self.dim)


class TestSiglip2Wrapper:
    def _make(self, dim=768):
        from vtscore.media.image.embedder_siglip2 import ImageSiglip2Embedder

        emb = ImageSiglip2Embedder()
        emb._model = _FakeVisionTextModel(dim)
        emb._processor = _FakeProcessor(keys=("pixel_values",))
        return emb

    def test_embed_pil_image(self):
        from PIL import Image

        emb = self._make(dim=768)
        vec = emb.embed_pil_image(Image.new("RGB", (16, 16)))
        assert vec is not None
        assert vec.shape == (768,)

    def test_embed_media_from_bytes(self):
        emb = self._make(dim=768)
        vec = emb.embed_media({"media_bytes": _png_bytes()})
        assert vec is not None
        assert vec.shape == (768,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_bad_source(self):
        assert self._make().embed_media({}) is None

    def test_embed_text_uses_padding_and_truncation(self):
        emb = self._make(dim=768)
        vec = emb.embed_text("a photo of a cat")
        assert vec is not None
        assert vec.shape == (768,)
        _args, kwargs = emb._processor.calls[0]
        assert kwargs["padding"] == "max_length"
        assert kwargs["truncation"] is True


# ===========================================================================
# DINOv2 / DINOv3 (shared CLS-extraction path)
# ===========================================================================


class _FakeViTModel:
    """Fake ViT whose last_hidden_state[:, 0] is a known CLS vector."""

    def __init__(self, dim=768, tokens=5):
        self.dim = dim
        self.tokens = tokens
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def __call__(self, **kwargs):
        # Token 0 (CLS) is all-ones * 3; other tokens differ so a bug that
        # grabbed the wrong token would change the result.
        lhs = torch.zeros(1, self.tokens, self.dim)
        lhs[:, 0] = 3.0
        return SimpleNamespace(last_hidden_state=lhs)


@pytest.mark.parametrize(
    "module_path, class_name",
    [
        ("vtscore.media.image.embedder_dinov2_single", "ImageDinov2SingleEmbedder"),
        ("vtscore.media.image.embedder_dinov3_single", "ImageDinov3SingleEmbedder"),
    ],
)
class TestDinoWrapper:
    def _make(self, module_path, class_name, dim=768):
        import importlib

        cls = getattr(importlib.import_module(module_path), class_name)
        emb = cls()
        emb._model = _FakeViTModel(dim)
        emb._processor = _FakeProcessor(keys=("pixel_values",))
        return emb

    def test_embed_pil_image_extracts_cls(self, module_path, class_name):
        from PIL import Image

        emb = self._make(module_path, class_name, dim=768)
        vec = emb.embed_pil_image(Image.new("RGB", (16, 16)))
        assert vec is not None
        assert vec.shape == (768,)
        # embed_pil_image returns the RAW CLS token (index 0), un-normalised;
        # our fake fills it with 3.0 everywhere, so a bug grabbing the wrong
        # token (filled with 0.0) would change the result.
        np.testing.assert_allclose(vec, np.full(768, 3.0, dtype=np.float32), atol=1e-5)

    def test_embed_media_from_bytes(self, module_path, class_name):
        emb = self._make(module_path, class_name)
        vec = emb.embed_media({"media_bytes": _png_bytes()})
        assert vec is not None
        # The public embed_media path L2-normalises the CLS vector.
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_bad_source(self, module_path, class_name):
        assert self._make(module_path, class_name).embed_media({}) is None

    def test_text_unsupported(self, module_path, class_name):
        assert self._make(module_path, class_name).embed_text("anything") is None


# ===========================================================================
# EUPE (shared CLS-extraction path via forward_features)
# ===========================================================================


class _FakeEupeModel:
    def __init__(self, dim=768):
        self.dim = dim
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def forward_features(self, batch):
        n = batch.shape[0]
        return {"x_norm_clstoken": torch.arange(n * self.dim, dtype=torch.float32).reshape(n, self.dim) + 1.0}


class TestEupeWrapper:
    def _make(self, dim=768):
        from vtscore.media.image.embedder_eupe_single import ImageEupeSingleEmbedder

        emb = ImageEupeSingleEmbedder()
        emb._model = _FakeEupeModel(dim)
        # EUPE's preprocess is a torchvision Compose returning a (C, H, W)
        # tensor; the fake just returns a fixed small tensor per image.
        emb._preprocess = lambda img: torch.zeros(3, 4, 4)
        return emb

    def test_embed_media_from_bytes_normalises_cls(self):
        emb = self._make(dim=768)
        vec = emb.embed_media({"media_bytes": _png_bytes()})
        assert vec is not None
        assert vec.shape == (768,)
        assert vec.dtype == np.float32
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_bad_source(self):
        assert self._make().embed_media({}) is None

    def test_forward_pil_batch(self):
        from PIL import Image

        emb = self._make(dim=768)
        arr = emb._forward_pil_batch([Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))])
        assert arr.shape == (2, 768)
        # Every row is L2-normalised.
        np.testing.assert_allclose(np.linalg.norm(arr, axis=1), [1.0, 1.0], atol=1e-5)

    def test_license_notice_present(self):
        emb = self._make()
        assert emb.license_notice is not None
        assert "Noncommercial" in emb.license_notice


# ===========================================================================
# VideoMAE (_pool_features helper + wrapper)
# ===========================================================================


class TestVideoMAEPoolFeatures:
    def test_pools_raw_2d_tensor_unchanged(self):
        from vtscore.media.video.embedder_videomae import _pool_features

        t = torch.randn(2, 8)
        out = _pool_features(t)
        assert torch.equal(out, t)

    def test_pools_raw_3d_tensor_over_time(self):
        from vtscore.media.video.embedder_videomae import _pool_features

        t = torch.ones(1, 5, 8)
        out = _pool_features(t)
        assert out.shape == (1, 8)

    def test_prefers_pooler_output(self):
        from vtscore.media.video.embedder_videomae import _pool_features

        out = _pool_features(SimpleNamespace(pooler_output=torch.ones(1, 8), last_hidden_state=torch.zeros(1, 5, 8)))
        assert out.shape == (1, 8)
        assert torch.all(out == 1.0)

    def test_mean_pools_last_hidden_state(self):
        from vtscore.media.video.embedder_videomae import _pool_features

        out = _pool_features(SimpleNamespace(pooler_output=None, last_hidden_state=torch.ones(1, 6, 8)))
        assert out.shape == (1, 8)

    def test_tuple_fallback(self):
        from vtscore.media.video.embedder_videomae import _pool_features

        out = _pool_features((torch.ones(1, 4, 8),))
        assert out.shape == (1, 8)


class _FakeVideoMAEModel:
    def __init__(self, dim=768):
        self.dim = dim
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def __call__(self, pixel_values):
        return SimpleNamespace(last_hidden_state=torch.ones(1, 5, self.dim))


class TestVideoMAEWrapper:
    def _make(self, monkeypatch, frames=8, dim=768):
        from PIL import Image

        from vtscore.media.video import embedder_videomae

        emb = embedder_videomae.VideoVideoMAEEmbedder()
        emb._model = _FakeVideoMAEModel(dim)
        pil_frames = [Image.new("RGB", (64, 48)) for _ in range(frames)]
        monkeypatch.setattr(embedder_videomae, "sample_video_frames", lambda media, n: pil_frames)
        return emb

    def test_embed_media_normalises(self, monkeypatch):
        emb = self._make(monkeypatch, dim=768)
        vec = emb.embed_media({"media_path": "/fake.mp4"})
        assert vec is not None
        assert vec.shape == (768,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_no_frames(self, monkeypatch):
        from vtscore.media.video import embedder_videomae

        emb = embedder_videomae.VideoVideoMAEEmbedder()
        emb._model = _FakeVideoMAEModel()
        monkeypatch.setattr(embedder_videomae, "sample_video_frames", lambda media, n: [])
        assert emb.embed_media({"media_path": "/fake.mp4"}) is None

    def test_text_unsupported(self, monkeypatch):
        assert self._make(monkeypatch).embed_text("anything") is None


# ===========================================================================
# LanguageBind (video + text encoders)
# ===========================================================================


class _FakeLanguageBindModel:
    def __init__(self, dim=768):
        self.dim = dim
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def get_image_features(self, pixel_values=None):
        return torch.arange(self.dim, dtype=torch.float32).reshape(1, self.dim) + 1.0

    def get_text_features(self, **kwargs):
        return (torch.arange(self.dim, dtype=torch.float32) + 2.0).reshape(1, self.dim)


class TestLanguageBindWrapper:
    def _make(self, monkeypatch, frames=8, dim=768):
        from PIL import Image

        from vtscore.media.video import embedder_languagebind

        emb = embedder_languagebind.VideoLanguageBindEmbedder()
        emb._model = _FakeLanguageBindModel(dim)
        emb._tokenizer = _FakeProcessor(keys=("input_ids", "attention_mask"), shape=(1, 3))
        pil_frames = [Image.new("RGB", (64, 48)) for _ in range(frames)]
        monkeypatch.setattr(embedder_languagebind, "sample_video_frames", lambda media, n: pil_frames)
        return emb

    def test_embed_media_normalises(self, monkeypatch):
        emb = self._make(monkeypatch, dim=768)
        vec = emb.embed_media({"media_path": "/fake.mp4"})
        assert vec is not None
        assert vec.shape == (768,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_no_frames(self, monkeypatch):
        from vtscore.media.video import embedder_languagebind

        emb = embedder_languagebind.VideoLanguageBindEmbedder()
        emb._model = _FakeLanguageBindModel()
        emb._tokenizer = _FakeProcessor(keys=("input_ids",))
        monkeypatch.setattr(embedder_languagebind, "sample_video_frames", lambda media, n: [])
        assert emb.embed_media({"media_path": "/fake.mp4"}) is None

    def test_embed_text(self, monkeypatch):
        emb = self._make(monkeypatch, dim=768)
        vec = emb.embed_text("a cat playing")
        assert vec is not None
        assert vec.shape == (768,)
        _args, kwargs = cast(Any, emb._tokenizer).calls[0]
        assert kwargs["max_length"] == 77
        assert kwargs["padding"] == "max_length"


# ===========================================================================
# X-CLIP (video + text encoders)
# ===========================================================================


class _FakeXClipModel:
    def __init__(self, dim=512):
        self.dim = dim
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def get_video_features(self, **kwargs):
        return torch.arange(self.dim, dtype=torch.float32).reshape(1, self.dim) + 1.0

    def get_text_features(self, **kwargs):
        return (torch.arange(self.dim, dtype=torch.float32) + 2.0).reshape(1, self.dim)


class TestXClipWrapper:
    def _make(self, monkeypatch, frames=8, dim=512):
        from PIL import Image

        from vtscore.media.video import embedder_xclip

        emb = embedder_xclip.VideoXClipEmbedder()
        emb._model = _FakeXClipModel(dim)
        emb._processor = _FakeProcessor(keys=("pixel_values",))
        pil_frames = [Image.new("RGB", (64, 48)) for _ in range(frames)]
        monkeypatch.setattr(embedder_xclip, "sample_video_frames", lambda media, n: pil_frames)
        return emb

    def test_embed_media(self, monkeypatch):
        emb = self._make(monkeypatch, dim=512)
        vec = emb.embed_media({"media_path": "/fake.mp4"})
        assert vec is not None
        assert vec.shape == (512,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_no_frames(self, monkeypatch):
        from vtscore.media.video import embedder_xclip

        emb = embedder_xclip.VideoXClipEmbedder()
        emb._model = _FakeXClipModel()
        emb._processor = _FakeProcessor()
        monkeypatch.setattr(embedder_xclip, "sample_video_frames", lambda media, n: [])
        assert emb.embed_media({"media_path": "/fake.mp4"}) is None

    def test_embed_text(self, monkeypatch):
        emb = self._make(monkeypatch, dim=512)
        vec = emb.embed_text("a person walking")
        assert vec is not None
        assert vec.shape == (512,)


# ===========================================================================
# E5 text embedder (encode prefixes + bulk batching)
# ===========================================================================


class _FakeSentenceTransformer:
    def __init__(self, dim=768):
        self.dim = dim
        self.encode_calls: list[tuple[object, dict]] = []

    def encode(self, sentences, normalize_embeddings=False, batch_size=None):
        self.encode_calls.append((sentences, {"normalize_embeddings": normalize_embeddings, "batch_size": batch_size}))
        if isinstance(sentences, list):
            return np.ones((len(sentences), self.dim), dtype=np.float32)
        return np.ones(self.dim, dtype=np.float32)


class TestE5Wrapper:
    def _make(self, dim=768):
        from vtscore.media.text.embedder_e5 import TextE5Embedder

        emb = TextE5Embedder()
        emb._model = cast(Any, _FakeSentenceTransformer(dim))
        return emb

    @staticmethod
    def _encode_calls(emb):
        return cast(Any, emb._model).encode_calls

    def test_read_text_prefers_media_string(self):
        from vtscore.media.text.embedder_e5 import _read_text

        assert _read_text({"media_string": "  hi  "}) == "hi"

    def test_read_text_reads_file(self, tmp_path):
        from vtscore.media.text.embedder_e5 import _read_text

        f = tmp_path / "a.txt"
        f.write_text("file body\n", encoding="utf-8")
        assert _read_text({"media_path": str(f)}) == "file body"

    def test_read_text_missing(self):
        from vtscore.media.text.embedder_e5 import _read_text

        assert _read_text({}) is None

    def test_embed_media_uses_passage_prefix(self):
        emb = self._make()
        vec = emb.embed_media({"media_string": "hello world"})
        assert vec is not None
        assert vec.shape == (768,)
        sentences, kwargs = self._encode_calls(emb)[0]
        assert sentences == "passage: hello world"
        assert kwargs["normalize_embeddings"] is True

    def test_embed_media_empty_text_returns_none(self):
        emb = self._make()
        assert emb.embed_media({"media_string": "   "}) is None

    def test_embed_text_uses_query_prefix(self):
        emb = self._make()
        emb.embed_text("what is a dog")
        sentences, _kwargs = self._encode_calls(emb)[0]
        assert sentences == "query: what is a dog"

    def test_embed_text_passage_prefix(self):
        emb = self._make()
        emb.embed_text_passage("some passage")
        sentences, _kwargs = self._encode_calls(emb)[0]
        assert sentences == "passage: some passage"

    def test_bulk_skips_empty_and_batches_ready(self):
        emb = self._make(dim=768)
        medias = [
            {"media_string": "first"},
            {"media_string": "   "},  # empty → None slot
            {"media_string": "third"},
        ]
        out = emb._embed_media_bulk_impl(medias)
        assert len(out) == 3
        assert out[1] is None
        assert out[0] is not None and out[2] is not None
        # Only the two ready passages were sent to encode, prefixed.
        sentences, kwargs = self._encode_calls(emb)[0]
        assert sentences == ["passage: first", "passage: third"]
        assert kwargs["normalize_embeddings"] is True

    def test_bulk_all_empty_returns_all_none(self):
        emb = self._make()
        out = emb._embed_media_bulk_impl([{"media_string": ""}, {}])
        assert out == [None, None]
        assert self._encode_calls(emb) == []


# ===========================================================================
# ParaSpeechCLAP embedder wrapper (embedder_paraspeechclap.py)
# ===========================================================================


class _FakeParaModel:
    def __init__(self, dim=768):
        self.dim = dim
        self._p = _cpu_param()

    def parameters(self):
        return iter([self._p])

    def get_audio_embedding(self, input_values, normalize=True):
        return torch.arange(self.dim, dtype=torch.float32).reshape(1, self.dim) + 1.0

    def get_text_embedding(self, inputs, normalize=True):
        return (torch.arange(self.dim, dtype=torch.float32) + 2.0).reshape(1, self.dim)


class TestParaSpeechClapWrapper:
    def _make(self, dim=768):
        from vtscore.media.audio.embedder_paraspeechclap import AudioParaSpeechClapEmbedder

        emb = AudioParaSpeechClapEmbedder()
        emb._model = _FakeParaModel(dim)
        emb._feature_extractor = _FakeProcessor(keys=("input_values",))
        emb._tokenizer = _FakeProcessor(keys=("input_ids", "attention_mask"), shape=(1, 3))
        return emb

    def test_embed_media(self, monkeypatch):
        _fake_librosa_load(monkeypatch, samples=1600)
        emb = self._make(dim=768)
        vec = emb.embed_media({"media_path": "/fake.wav"})
        assert vec is not None
        assert vec.shape == (768,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_media_truncates(self, monkeypatch):
        from vtscore.config import PARASPEECHCLAP_MAX_SAMPLES

        _fake_librosa_load(monkeypatch, samples=PARASPEECHCLAP_MAX_SAMPLES + 1000)
        emb = self._make()
        vec = emb.embed_media({"media_path": "/long.wav"})
        assert vec is not None
        _args, kwargs = emb._feature_extractor.calls[0]
        assert kwargs["sampling_rate"]  # forwarded
        # The clip fed to the extractor is capped.
        assert emb._feature_extractor.calls[0][0][0].shape[0] == PARASPEECHCLAP_MAX_SAMPLES

    def test_embed_media_no_source(self):
        assert self._make().embed_media({}) is None

    def test_embed_text(self):
        emb = self._make(dim=768)
        vec = emb.embed_text("a deep, raspy voice")
        assert vec is not None
        assert vec.shape == (768,)
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)


# ===========================================================================
# Vendored ParaSpeechCLAP architecture (_paraspeechclap_model.py)
# ===========================================================================


class _FakeBaseModule(torch.nn.Module):
    """Tiny stand-in for a WavLM / Granite base encoder."""

    def __init__(self, hidden):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self._hidden = hidden

    def forward(self, x=None, attention_mask=None, return_dict=True, **kwargs):
        if x is not None and isinstance(x, torch.Tensor):
            b = x.shape[0]
        else:
            ids = kwargs.get("input_ids")
            b = ids.shape[0] if isinstance(ids, torch.Tensor) else 1
        lhs = torch.ones(b, 5, self._hidden)
        if return_dict is False:
            return (lhs,)
        return SimpleNamespace(last_hidden_state=lhs)


@pytest.fixture
def _patched_transformers(monkeypatch):
    """Swap AutoConfig/AutoModel in the vendored module for tiny fakes."""
    import vtscore.media.audio._paraspeechclap_model as psm

    class _FakeAutoConfig:
        @staticmethod
        def from_pretrained(name, **kwargs):
            return SimpleNamespace(layerdrop=0.9, hidden_size=6)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(name, config=None, **kwargs):
            hidden = config.hidden_size if config is not None else 6
            return _FakeBaseModule(hidden)

    monkeypatch.setattr(psm, "AutoConfig", _FakeAutoConfig)
    monkeypatch.setattr(psm, "AutoModel", _FakeAutoModel)
    return psm


class TestParaSpeechClapModel:
    def test_projection_shape_and_residual(self):
        from vtscore.media.audio._paraspeechclap_model import Projection

        proj = Projection(4, 3, p=0.0).eval()
        out = proj(torch.randn(2, 4))
        assert out.shape == (2, 3)
        assert torch.isfinite(out).all()

    def test_speech_encoder_disables_layerdrop_and_mean_pools(self, _patched_transformers):
        from vtscore.media.audio._paraspeechclap_model import SpeechEncoder

        enc = SpeechEncoder("microsoft/wavlm-large").eval()
        assert enc.is_wavlm is True
        assert enc.hidden_size == 6
        out = enc(torch.randn(2, 100))
        # Mean over the seq axis → (B, hidden).
        assert out.shape == (2, 6)

    def test_speech_encoder_non_wavlm_branch(self, _patched_transformers):
        from vtscore.media.audio._paraspeechclap_model import SpeechEncoder

        enc = SpeechEncoder("facebook/other-speech").eval()
        assert enc.is_wavlm is False
        out = enc(torch.randn(1, 50))
        assert out.shape == (1, 6)

    def test_text_encoder_uses_cls_token(self, _patched_transformers):
        from vtscore.media.audio._paraspeechclap_model import TextEncoder

        enc = TextEncoder("ibm-granite/granite").eval()
        out = enc({"input_ids": torch.zeros(3, 4, dtype=torch.long)})
        assert out.shape == (3, 6)

    def test_clap_audio_embedding_normalises(self, _patched_transformers):
        from vtscore.media.audio._paraspeechclap_model import CLAP

        model = CLAP("microsoft/wavlm-large", "ibm-granite/granite", embedding_dim=5).eval()
        emb = model.get_audio_embedding(torch.randn(2, 80), normalize=True)
        assert emb.shape == (2, 5)
        np.testing.assert_allclose(emb.norm(dim=-1).detach().numpy(), [1.0, 1.0], atol=1e-5)

    def test_clap_audio_embedding_unnormalised(self, _patched_transformers):
        from vtscore.media.audio._paraspeechclap_model import CLAP

        model = CLAP("microsoft/wavlm-large", "ibm-granite/granite", embedding_dim=5).eval()
        emb = model.get_audio_embedding(torch.randn(1, 80), normalize=False)
        assert emb.shape == (1, 5)

    def test_clap_text_embedding(self, _patched_transformers):
        from vtscore.media.audio._paraspeechclap_model import CLAP

        model = CLAP("microsoft/wavlm-large", "ibm-granite/granite", embedding_dim=5).eval()
        emb = model.get_text_embedding({"input_ids": torch.zeros(2, 4, dtype=torch.long)}, normalize=True)
        assert emb.shape == (2, 5)
        np.testing.assert_allclose(emb.norm(dim=-1).detach().numpy(), [1.0, 1.0], atol=1e-5)
