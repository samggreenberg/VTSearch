"""Library-tier tests for the signpost prep orchestration.

Covers ``vtscore.projection.signpost_prep`` end-to-end with the toponymy fit
stubbed at the ``_fit_topic_layers`` seam: prerequisite gating, matrix/text
alignment, context-slot assignment (full vs subset), container persistence +
signature stamping, and the ingest-time ``ensure_texts_for_dataset`` stage.
No Flask; the "embedder" is a local fake resolved through a patched registry.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.projection import signpost_build as sb
from vtscore.projection import signpost_prep as sp
from vtscore.projection import signpost_texts as st
from vtscore.projection.umap_projection import Projection
from vtscore.state import DatasetContext

_N = 60
_DIM = 8
_EMBEDDER = "fake_clap"


class FakeEmbedder:
    name = _EMBEDDER
    supports_text = True

    def embed_text(self, text: str):
        import hashlib

        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 2**32
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(_DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)


@pytest.fixture
def ctx(monkeypatch):
    """A context of ``_N`` audio medias with vectors under the fake embedder.

    The fit itself is stubbed, so these tests run whether or not toponymy is
    installed — availability is patched to True and re-patched off by the
    tests that cover the unavailable path.
    """
    monkeypatch.setattr(sp, "signposting_available", lambda: True)
    rng = np.random.default_rng(7)
    context = DatasetContext("signpost-test")
    for mid in range(1, _N + 1):
        vec = rng.standard_normal(_DIM).astype(np.float32)
        context.medias[mid] = {
            "id": mid,
            "media_type": "audio",
            "embedder": _EMBEDDER,
            "embeddings": {_EMBEDDER: vec / np.linalg.norm(vec)},
        }
    monkeypatch.setattr(DatasetContext, "routed_embedder", lambda self, role: _EMBEDDER)

    import vtscore.media as media_registry

    monkeypatch.setattr(media_registry, "get_embedder", lambda name: FakeEmbedder())
    return context


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Fast deterministic stand-ins for the UMAP + toponymy stages."""
    monkeypatch.setattr(st, "_load_vocab", lambda asset: ["dog", "rain", "car"])
    monkeypatch.setattr(sb, "_clusterable_vectors", lambda m, p=None: np.asarray(m[:, :2], np.float32))

    def fake_fit(texts, embedding_vectors, clusterable_vectors, text_encoder, **kwargs):
        n = len(texts)
        return [(["half-a", "half-b"], np.array([0, 1] * (n // 2))), (["everything"], np.zeros(n, dtype=int))]

    monkeypatch.setattr(sb, "_fit_topic_layers", fake_fit)


def _proj(ctx, ids=None, projection_id="prep-proj"):
    ids = sorted(ctx.medias) if ids is None else sorted(ids)
    rng = np.random.default_rng(3)
    coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
    return Projection(projection_id, ids, coords, "pca")


class TestPrepSignposts:
    def test_full_build_assigns_and_labels(self, ctx, stub_pipeline):
        proj = _proj(ctx)
        label_set = sp.prep_signposts(ctx, proj, subset=False)

        assert label_set is not None
        assert ctx._region_labels is label_set
        assert ctx._subset_region_labels is None
        assert label_set.projection_id == proj.projection_id
        assert {lab.text for lab in label_set.labels} == {"everything", "half-a", "half-b"}
        # Texts were stamped onto the media dicts for reuse (e.g. Find→Browse).
        assert all(st.TEXT_FIELD in m for m in ctx.medias.values())

    def test_subset_build_uses_subset_slot_only(self, ctx, stub_pipeline):
        ids = sorted(ctx.medias)[: _N // 2 + 10]
        proj = _proj(ctx, ids=ids, projection_id="subset-proj")
        label_set = sp.prep_signposts(ctx, proj, subset=True)

        assert label_set is not None
        assert ctx._subset_region_labels is label_set
        assert ctx._region_labels is None

    def test_persists_full_set_with_signature(self, ctx, stub_pipeline, tmp_path, monkeypatch):
        pkl = tmp_path / "container.pkl"
        import zipfile

        with zipfile.ZipFile(pkl, "w") as zf:
            zf.writestr("placeholder", b"")
        import vtscore.datasets.registry as registry

        monkeypatch.setattr(registry, "get_dataset", lambda dataset_id: {"pkl_path": str(pkl)})

        proj = _proj(ctx)
        label_set = sp.prep_signposts(ctx, proj, subset=False)
        assert label_set is not None

        from vtscore.datasets.container import read_region_labels

        loaded = read_region_labels(pkl)
        assert loaded is not None
        stored_set, stored_signature = loaded
        assert stored_set.projection_id == proj.projection_id
        assert stored_set.labels == label_set.labels
        assert stored_signature == sp.labeler_signature(ctx)

    def test_subset_never_persists(self, ctx, stub_pipeline, monkeypatch):
        import vtscore.datasets.registry as registry

        def boom(dataset_id):
            raise AssertionError("subset prep must not touch the registry")

        monkeypatch.setattr(registry, "get_dataset", boom)
        assert sp.prep_signposts(ctx, _proj(ctx), subset=True) is not None

    def test_empty_fit_leaves_slot_untouched(self, ctx, stub_pipeline, monkeypatch):
        # An empty result must not pin an empty set — that would suppress the
        # lazy ground-truth fallback for hierarchical-category datasets.
        monkeypatch.setattr(sb, "_fit_topic_layers", lambda *a, **k: [])
        assert sp.prep_signposts(ctx, _proj(ctx), subset=False) is None
        assert ctx._region_labels is None

    def test_no_text_embedder_bails(self, ctx, stub_pipeline, monkeypatch):
        monkeypatch.setattr(DatasetContext, "routed_embedder", lambda self, role: None)
        assert sp.prep_signposts(ctx, _proj(ctx), subset=False) is None

    def test_unavailable_toponymy_bails(self, ctx, stub_pipeline, monkeypatch):
        monkeypatch.setattr(sp, "signposting_available", lambda: False)
        assert sp.prep_signposts(ctx, _proj(ctx), subset=False) is None

    def test_tiny_layout_bails_before_texts(self, ctx, stub_pipeline, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("texts must not be computed for tiny layouts")

        monkeypatch.setattr(sp, "ensure_signpost_texts", boom)
        proj = _proj(ctx, ids=sorted(ctx.medias)[: sb._MIN_POINTS - 1])
        assert sp.prep_signposts(ctx, proj, subset=False) is None

    def test_media_id_mismatch_bails(self, ctx, stub_pipeline):
        # A layout fit on ids that are no longer (all) in the dataset must not
        # be lettered — row order vs proj.coords could no longer be trusted.
        proj = _proj(ctx)
        del ctx.medias[sorted(ctx.medias)[-1]]
        assert sp.prep_signposts(ctx, proj, subset=False) is None

    def test_unsupported_media_type_bails(self, ctx, stub_pipeline):
        for media in ctx.medias.values():
            media["media_type"] = "no_provider_type"
        assert sp.prep_signposts(ctx, _proj(ctx), subset=False) is None


class TestLabelerSignature:
    def test_signature_shape(self, ctx, stub_pipeline):
        signature = sp.labeler_signature(ctx)
        assert signature is not None
        assert signature.startswith("keyphrase|tags:audioset527:fake_clap|toponymy=")

    def test_none_when_unavailable(self, ctx, monkeypatch):
        monkeypatch.setattr(sp, "signposting_available", lambda: False)
        assert sp.labeler_signature(ctx) is None


class TestEnsureTextsForDataset:
    def test_stamps_all_medias(self, ctx, stub_pipeline):
        sp.ensure_texts_for_dataset(ctx)
        assert all(st.TEXT_FIELD in m and m[st.TEXT_FIELD] for m in ctx.medias.values())
        source = next(iter(ctx.medias.values()))[st.SOURCE_FIELD]
        assert source == "tags:audioset527:fake_clap"

    def test_noop_without_text_embedder(self, ctx, stub_pipeline, monkeypatch):
        monkeypatch.setattr(DatasetContext, "routed_embedder", lambda self, role: None)
        sp.ensure_texts_for_dataset(ctx)
        assert not any(st.TEXT_FIELD in m for m in ctx.medias.values())
