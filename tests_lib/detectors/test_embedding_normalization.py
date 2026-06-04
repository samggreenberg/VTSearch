"""App-wide L2-normalization-at-ingest invariant (VTSBrowse prerequisite).

Every embedding that enters the system is L2-normalized exactly once, so
downstream similarity is a plain dot product.  These tests pin the four
guarantees that make that safe, all model-free:

* :func:`vtscore.embedding.normalize.l2_normalize` produces unit vectors,
  passes zero / non-finite norms through untouched, and is idempotent.
* :class:`vtscore.media.embedder.MediaEmbedder`'s public ``embed_media`` /
  ``embed_media_bulk`` / ``embed_text`` wrappers normalize whatever the
  subclass ``_*_impl`` returns, so no subclass has to normalize itself.
* :mod:`vtscore.training.region_similarity` scores with a dot product (no
  per-comparison normalization) and still guards the zero-query case.

See ``docs/plans/vtsbrowse.md`` §Prerequisite.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from vtscore.embedding.normalize import l2_normalize
from vtscore.media.embedder import MediaEmbedder


# ---------------------------------------------------------------------------
# l2_normalize
# ---------------------------------------------------------------------------


class TestL2Normalize:
    def test_returns_unit_vector(self):
        out = l2_normalize([3.0, 4.0])
        np.testing.assert_allclose(np.linalg.norm(out), 1.0, atol=1e-6)
        np.testing.assert_allclose(out, [0.6, 0.8], atol=1e-6)

    def test_output_is_float32(self):
        assert l2_normalize([3.0, 4.0]).dtype == np.float32
        assert l2_normalize(np.array([1, 2, 2], dtype=np.float64)).dtype == np.float32

    def test_zero_vector_passes_through_unchanged(self):
        out = l2_normalize([0.0, 0.0, 0.0])
        # No divide-by-zero -> no NaN/inf; the zero vector is preserved.
        assert np.all(out == 0.0)
        assert np.all(np.isfinite(out))

    def test_non_finite_norm_passes_through(self):
        out = l2_normalize([np.inf, 1.0])
        # norm is inf -> we must not divide (inf/inf == nan); pass through.
        assert out[1] == 1.0

    def test_idempotent_on_unit_input(self):
        once = l2_normalize([5.0, -2.0, 1.0])
        twice = l2_normalize(once)
        np.testing.assert_allclose(once, twice, atol=1e-6)


# ---------------------------------------------------------------------------
# MediaEmbedder wrappers normalize subclass output
# ---------------------------------------------------------------------------


class _FakeEmbedder(MediaEmbedder):
    """Minimal embedder returning deliberately non-unit vectors."""

    @property
    def name(self) -> str:
        return "fake"

    @property
    def media_type_id(self) -> str:
        return "audio"

    def _load_models_impl(self) -> None:  # pragma: no cover - never loaded
        pass

    def _embed_media_impl(self, media: dict) -> Optional[np.ndarray]:
        if media.get("fail"):
            return None
        return np.array([3.0, 4.0], dtype=np.float32)  # norm 5

    def _embed_text_impl(self, text: str) -> Optional[np.ndarray]:
        if text == "":
            return None
        return np.array([0.0, 6.0, 8.0], dtype=np.float32)  # norm 10


class TestEmbedderNormalizesOutput:
    def test_embed_media_is_unit_norm(self):
        vec = _FakeEmbedder().embed_media({})
        assert vec is not None
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_embed_media_none_passes_through(self):
        assert _FakeEmbedder().embed_media({"fail": True}) is None

    def test_embed_media_bulk_each_unit_norm(self):
        out = _FakeEmbedder().embed_media_bulk([{}, {"fail": True}, {}])
        assert out[1] is None
        for v in (out[0], out[2]):
            assert v is not None
            np.testing.assert_allclose(np.linalg.norm(v), 1.0, atol=1e-6)

    def test_embed_text_is_unit_norm(self):
        vec = _FakeEmbedder().embed_text("hello")
        assert vec is not None
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-6)

    def test_embed_text_none_passes_through(self):
        assert _FakeEmbedder().embed_text("") is None

    def test_base_embed_text_default_is_none(self):
        # A subclass that doesn't override _embed_text_impl gets None.
        class _NoText(_FakeEmbedder):
            _embed_text_impl = MediaEmbedder._embed_text_impl

        assert _NoText().embed_text("hi") is None


# ---------------------------------------------------------------------------
# region_similarity scores via dot product (no per-comparison normalization)
# ---------------------------------------------------------------------------


class TestRegionSimilarityDotProduct:
    def test_single_vector_cosine_equals_dot_for_unit_inputs(self):
        from vtscore.training.region_similarity import score_against_query

        media = {"embedding": np.array([0.6, 0.8], dtype=np.float32)}
        q = np.array([0.6, 0.8], dtype=np.float32)
        score, box = score_against_query(media, q)
        np.testing.assert_allclose(score, 1.0, atol=1e-6)
        assert box == (0.0, 0.0, 1.0, 1.0)

    def test_orthogonal_unit_vectors_score_zero(self):
        from vtscore.training.region_similarity import score_against_query

        media = {"embedding": np.array([1.0, 0.0], dtype=np.float32)}
        q = np.array([0.0, 1.0], dtype=np.float32)
        score, _ = score_against_query(media, q)
        np.testing.assert_allclose(score, 0.0, atol=1e-6)

    def test_zero_query_returns_zero(self):
        from vtscore.training.region_similarity import score_against_query

        media = {"embedding": np.array([1.0, 0.0], dtype=np.float32)}
        score, box = score_against_query(media, np.zeros(2, dtype=np.float32))
        assert score == 0.0
        assert box is None

    def test_no_per_comparison_normalization(self):
        # Contract change: the scorer assumes unit-norm inputs and does NOT
        # renormalize.  A deliberately non-unit query therefore yields the
        # raw dot product, not a cosine - proving the divide was removed.
        from vtscore.training.region_similarity import score_against_query

        media = {"embedding": np.array([1.0, 0.0], dtype=np.float32)}
        q = np.array([5.0, 0.0], dtype=np.float32)  # not unit-norm
        score, _ = score_against_query(media, q)
        np.testing.assert_allclose(score, 5.0, atol=1e-6)
