"""Unit tests for ``MediaEmbedder.loaded_backbone`` and the loader getters.

``loaded_backbone`` is the supported replacement for the private
``_get_model_and_processor()`` / ``_get_model()`` helpers the
``vtscore.embedding.loader`` getters used to reach into (issue #3395).  The
point of the accessor is that it is defined on the ABC, so an embedder that
holds its backbone somewhere unusual overrides one documented method instead
of silently breaking the getters - and one that has no backbone at all fails
loudly rather than handing back ``None``.

No weights are downloaded: every embedder here is a hand-rolled subclass whose
``_load_models_impl`` assigns plain sentinels.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pytest

from vtscore.media.embedder import MediaEmbedder


class _StubEmbedder(MediaEmbedder):
    """Minimal embedder following the ``_model`` / ``_processor`` convention."""

    _model: Any = None
    _processor: Any = None

    @property
    def name(self) -> str:
        return "stub"

    @property
    def media_type_id(self) -> str:
        return "text"

    def _load_models_impl(self) -> None:
        self._model = "MODEL"
        self._processor = "PROCESSOR"

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        return np.zeros(4, dtype=np.float32)


class _NoProcessorEmbedder(_StubEmbedder):
    """An embedder whose backbone needs no companion processor (e.g. E5/BGE)."""

    def _load_models_impl(self) -> None:
        self._model = "MODEL"


class _BacklessEmbedder(_StubEmbedder):
    """A broken embedder: ``load_models()`` returns without setting ``_model``."""

    def _load_models_impl(self) -> None:
        return None


class _ElsewhereEmbedder(_StubEmbedder):
    """An embedder holding its backbone off the conventional attributes."""

    def _load_models_impl(self) -> None:
        self._model = "MODEL"
        self._tokenizer = "TOKENIZER"

    def loaded_backbone(self):
        self.load_models()
        return self._model, self._tokenizer


class TestLoadedBackbone:
    def test_returns_model_and_processor(self):
        emb = _StubEmbedder()
        assert emb.loaded_backbone() == ("MODEL", "PROCESSOR")

    def test_loads_lazily(self):
        emb = _StubEmbedder()
        assert emb._model is None
        emb.loaded_backbone()
        assert emb._model == "MODEL"

    def test_processor_is_none_when_absent(self):
        model, processor = _NoProcessorEmbedder().loaded_backbone()
        assert model == "MODEL"
        assert processor is None

    def test_missing_backbone_raises_rather_than_returning_none(self):
        """A silent ``None`` would surface as an unrelated crash much later."""
        with pytest.raises(RuntimeError, match="no backbone"):
            _BacklessEmbedder().loaded_backbone()

    def test_override_wins(self):
        assert _ElsewhereEmbedder().loaded_backbone() == ("MODEL", "TOKENIZER")


class TestLoaderGetters:
    """The three public getters must keep their names, shapes, and behaviour."""

    @pytest.mark.parametrize(
        ("getter", "embedder_name"),
        [("get_clap_model", "clap"), ("get_xclip_model", "xclip")],
    )
    def test_pair_getters_delegate_to_loaded_backbone(self, monkeypatch, getter, embedder_name):
        import vtscore.media as media_mod
        from vtscore import embedding as emb_pkg

        seen: list[str] = []
        stub = _StubEmbedder()

        def _fake_get_embedder(name: str):
            seen.append(name)
            return stub

        monkeypatch.setattr(media_mod, "get_embedder", _fake_get_embedder)
        assert getattr(emb_pkg, getter)() == ("MODEL", "PROCESSOR")
        assert seen == [embedder_name]

    def test_e5_getter_returns_the_model_alone(self, monkeypatch):
        import vtscore.media as media_mod
        from vtscore import embedding as emb_pkg

        monkeypatch.setattr(media_mod, "get_embedder", lambda name: _NoProcessorEmbedder())
        assert emb_pkg.get_e5_model() == "MODEL"

    def test_getters_survive_an_embedder_without_the_private_helpers(self, monkeypatch):
        """The regression this replaced: a reimplemented embedder used to break these."""
        import vtscore.media as media_mod
        from vtscore import embedding as emb_pkg

        stub = _StubEmbedder()
        assert not hasattr(stub, "_get_model_and_processor")
        monkeypatch.setattr(media_mod, "get_embedder", lambda name: stub)
        assert emb_pkg.get_clap_model() == ("MODEL", "PROCESSOR")


class TestShippedEmbeddersExposeABackbone:
    """Every in-tree embedder must satisfy the accessor's contract once loaded."""

    def test_languagebind_returns_its_tokenizer(self):
        from vtscore.media.video.embedder_languagebind import VideoLanguageBindEmbedder

        emb = VideoLanguageBindEmbedder()
        emb._model = "MODEL"
        emb._tokenizer = "TOKENIZER"
        assert emb.loaded_backbone() == ("MODEL", "TOKENIZER")
