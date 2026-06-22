"""Tests for the v1 structural embedder (SIFT/VLAD instance matching).

Covers the library-tier matcher (``vtscore.media.structural``) and the
``sift_vlad`` embedder: geometric verification on synthetic correspondences,
VLAD aggregation, the compact storage round-trip, the match-statistic feature
vector, the shipped codebook asset, and the capability flag.  No model weights
are downloaded - SIFT + the local codebook asset are all that's needed.
"""

from __future__ import annotations

import pickle

import cv2
import numpy as np
import pytest

from vtscore.media.structural import (
    DEFAULT_VLAD_CENTROIDS,
    MATCH_STAT_DIM,
    SIFT_DESCRIPTOR_DIM,
    MatchStats,
    SiftMatcher,
    StructuralFeatures,
    StructuralMatcher,
    aggregate_vlad,
    load_vlad_codebook,
    match_stats_to_features,
    rootsift,
)


def _textured_image(seed: int = 0, size: int = 200) -> np.ndarray:
    """A deterministic, SIFT-friendly grayscale image (noise + drawn shapes).

    The drawn shapes are placed at *seed-dependent* positions so two different
    seeds are genuinely unrelated images (no shared structure to spuriously
    geometrically verify), while a given seed always reproduces the same image.
    """
    rng = np.random.default_rng(seed)
    img = (rng.random((size, size)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    lo, hi = size // 6, size - size // 6
    x0, y0, x1, y1 = sorted(rng.integers(lo, hi, size=2)) + sorted(rng.integers(lo, hi, size=2))
    cv2.rectangle(img, (int(x0), int(x1)), (int(y0), int(y1)), 255, 3)
    cx, cy = rng.integers(lo, hi, size=2)
    cv2.circle(img, (int(cx), int(cy)), int(rng.integers(15, 35)), 0, 2)
    p = rng.integers(10, size - 10, size=4)
    cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), 200, 2)
    return img


def _similarity_matrix(angle_deg: float, scale: float, tx: float, ty: float) -> np.ndarray:
    theta = np.deg2rad(angle_deg)
    return np.array(
        [
            [scale * np.cos(theta), -scale * np.sin(theta), tx],
            [scale * np.sin(theta), scale * np.cos(theta), ty],
        ],
        dtype=np.float32,
    )


class TestSiftMatcher:
    def test_detect_and_describe_shapes_and_normalisation(self):
        img = _textured_image()
        feat = SiftMatcher().detect_and_describe(img, max_features=300)
        assert feat.count > 0
        assert feat.keypoints.shape[1] == 4
        assert feat.descriptors.shape == (feat.count, SIFT_DESCRIPTOR_DIM)
        # x/y are normalised into [0, 1].
        assert feat.keypoints[:, 0].min() >= 0.0
        assert feat.keypoints[:, 0].max() <= 1.0
        assert feat.keypoints[:, 1].min() >= 0.0
        assert feat.keypoints[:, 1].max() <= 1.0

    def test_detect_caps_features(self):
        img = _textured_image()
        feat = SiftMatcher().detect_and_describe(img, max_features=50)
        assert feat.count <= 50

    def test_verify_recovers_planted_similarity(self):
        img = _textured_image()
        m = SiftMatcher()
        template = m.detect_and_describe(img, max_features=400)
        warped = cv2.warpAffine(img, _similarity_matrix(20.0, 1.2, 15.0, 10.0), (200, 200))
        candidate = m.detect_and_describe(warped, max_features=400)

        stats = m.verify(template, candidate)
        assert stats.model_ok is True
        assert stats.inlier_count >= 20
        assert stats.reflection is False
        # The fitted uniform scale recovers the planted 1.2 within tolerance.
        assert stats.scale == pytest.approx(1.2, abs=0.1)
        assert 0.0 <= stats.inlier_ratio <= 1.0
        assert stats.mean_reproj_error < 0.05
        assert stats.inlier_box is not None

    def test_verify_rejects_non_match(self):
        m = SiftMatcher()
        template = m.detect_and_describe(_textured_image(seed=1), max_features=400)
        other = m.detect_and_describe(_textured_image(seed=999), max_features=400)
        stats = m.verify(template, other)
        # Unrelated images should not geometrically verify.
        assert stats.model_ok is False
        assert stats.inlier_count < 10

    def test_verify_handles_too_few_descriptors(self):
        m = SiftMatcher()
        template = m.detect_and_describe(_textured_image(), max_features=400)
        empty = StructuralFeatures(
            keypoints=np.zeros((1, 4), dtype=np.float32),
            descriptors=np.zeros((1, SIFT_DESCRIPTOR_DIM), dtype=np.float32),
        )
        stats = m.verify(template, empty)
        assert stats.model_ok is False
        assert stats.inlier_count == 0

    def test_verify_handles_non_finite_model(self, monkeypatch):
        """A non-finite RANSAC model is treated as no fit (no NaN stats / warnings).

        ``estimateAffinePartial2D`` can return a degenerate model with NaN/inf
        entries; feeding that into the scale/determinant maths raises numpy
        "invalid value" warnings and yields garbage stats.  The matcher must
        detect it and fall back to a clean no-match result.
        """
        m = SiftMatcher()
        # Same image both sides → plenty of tentative matches, so the code path
        # actually reaches estimateAffinePartial2D (and thus the NaN guard).
        template = m.detect_and_describe(_textured_image(seed=1), max_features=400)
        candidate = m.detect_and_describe(_textured_image(seed=1), max_features=400)

        nan_model = np.full((2, 3), np.nan, dtype=np.float64)
        mask = np.ones((template.count, 1), dtype=np.uint8)
        monkeypatch.setattr(cv2, "estimateAffinePartial2D", lambda *a, **k: (nan_model, mask))

        with np.errstate(invalid="raise"):
            stats = m.verify(template, candidate)
        assert stats.model_ok is False
        assert stats.inlier_count == 0
        assert stats.inlier_box is None

    def test_sift_matcher_satisfies_protocol(self):
        assert isinstance(SiftMatcher(), StructuralMatcher)


class TestVladAggregation:
    def _codebook(self, k: int = 16, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.random((k, SIFT_DESCRIPTOR_DIM), dtype=np.float64).astype(np.float32) * 255.0

    def test_shape_and_unit_norm(self):
        cb = self._codebook(k=16)
        rng = np.random.default_rng(3)
        desc = (rng.random((120, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        v = aggregate_vlad(desc, cb)
        assert v.shape == (16 * SIFT_DESCRIPTOR_DIM,)
        assert v.dtype == np.float32
        assert float(np.linalg.norm(v)) == pytest.approx(1.0, abs=1e-5)

    def test_empty_descriptors_give_zero_vector(self):
        cb = self._codebook(k=8)
        v = aggregate_vlad(np.zeros((0, SIFT_DESCRIPTOR_DIM), dtype=np.float32), cb)
        assert v.shape == (8 * SIFT_DESCRIPTOR_DIM,)
        assert float(np.linalg.norm(v)) == 0.0

    def test_deterministic_under_seeded_codebook(self):
        cb = self._codebook(k=16, seed=7)
        rng = np.random.default_rng(11)
        desc = (rng.random((90, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        v1 = aggregate_vlad(desc, cb)
        v2 = aggregate_vlad(desc.copy(), cb.copy())
        np.testing.assert_array_equal(v1, v2)

    def test_wrong_descriptor_dim_raises(self):
        cb = self._codebook(k=8)
        with pytest.raises(ValueError):
            aggregate_vlad(np.zeros((5, 64), dtype=np.float32), cb)

    def test_rootsift_is_l1_then_sqrt(self):
        desc = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=np.float32)
        out = rootsift(desc)
        # L1-normalised to 0.25 each, then sqrt -> 0.5.
        np.testing.assert_allclose(out, np.full((1, 4), 0.5), atol=1e-6)


class TestStructuralFeaturesStorage:
    def test_compact_dtypes_and_count(self):
        rng = np.random.default_rng(0)
        kp = rng.random((10, 4)).astype(np.float32)
        desc = (rng.random((10, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        feat = StructuralFeatures(keypoints=kp, descriptors=desc)
        compact = feat.compact()
        assert compact.keypoints.dtype == np.float16
        assert compact.descriptors.dtype == np.uint8
        assert compact.count == 10
        # Cast-back helpers return float32 regardless of stored dtype.
        assert compact.keypoints_f32().dtype == np.float32
        assert compact.descriptors_f32().dtype == np.float32

    def test_compact_descriptors_near_lossless(self):
        # SIFT descriptors are integer-valued in [0, 255]; uint8 round-trips them.
        desc = np.array([[0.0, 127.0, 255.0, 42.0]], dtype=np.float32)
        feat = StructuralFeatures(keypoints=np.zeros((1, 4), dtype=np.float32), descriptors=desc)
        back = feat.compact().descriptors_f32()
        np.testing.assert_array_equal(back, desc)

    def test_pickle_round_trip(self):
        # local_features live only in the dataset pickle (the sanctioned snapshot
        # store) - they must survive a pickle round-trip as the compact form.
        rng = np.random.default_rng(5)
        feat = StructuralFeatures(
            keypoints=rng.random((7, 4)).astype(np.float32),
            descriptors=(rng.random((7, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32),
        ).compact()
        restored = pickle.loads(pickle.dumps(feat))
        assert restored.count == 7
        np.testing.assert_array_equal(restored.keypoints, feat.keypoints)
        np.testing.assert_array_equal(restored.descriptors, feat.descriptors)


class TestMatchStatsFeatures:
    def test_feature_vector_shape_and_dtype(self):
        stats = MatchStats(
            inlier_count=30,
            inlier_ratio=0.6,
            tentative_count=50,
            mean_reproj_error=0.01,
            median_reproj_error=0.008,
            scale=1.1,
            reflection=False,
            inlier_spread=0.2,
            model_ok=True,
        )
        vec = match_stats_to_features(stats)
        assert vec.shape == (MATCH_STAT_DIM,)
        assert vec.dtype == np.float32

    def test_is_match_gate(self):
        good = MatchStats(inlier_count=20, model_ok=True)
        assert good.is_match(min_inliers=8) is True
        weak = MatchStats(inlier_count=3, model_ok=True)
        assert weak.is_match(min_inliers=8) is False
        no_model = MatchStats(inlier_count=50, model_ok=False)
        assert no_model.is_match(min_inliers=8) is False


class TestCodebookAsset:
    def test_shipped_codebook_shape(self):
        cb = load_vlad_codebook()
        assert cb.shape == (DEFAULT_VLAD_CENTROIDS, SIFT_DESCRIPTOR_DIM)
        assert cb.dtype == np.float32

    def test_codebook_is_cached(self):
        assert load_vlad_codebook() is load_vlad_codebook()


class TestSiftVladEmbedder:
    def _fresh_embedder(self):
        # Construct a fresh instance so the session-wide _stub_embedding_models
        # fixture (which patches the *registered* singletons) doesn't apply.
        from vtscore.media.image.embedder_sift_vlad import ImageSiftVladEmbedder

        return ImageSiftVladEmbedder()

    def test_capability_flags(self):
        emb = self._fresh_embedder()
        assert emb.name == "sift_vlad"
        assert emb.media_type_id == "image"
        assert emb.supports_geometric_verification is True
        assert emb.supports_text is False
        assert emb.is_default is False

    def test_to_dict_surfaces_geometric_flag(self):
        emb = self._fresh_embedder()
        d = emb.to_dict()
        assert d["supports_geometric_verification"] is True
        assert d["supports_patch_regions"] is False
        assert d["supports_text"] is False

    def test_load_models_sets_matcher_and_codebook(self):
        emb = self._fresh_embedder()
        emb.load_models()
        assert emb._matcher is not None
        assert emb._codebook is not None

    def test_embed_media_returns_unit_vlad_vector(self, tmp_path):
        from PIL import Image

        emb = self._fresh_embedder()
        path = tmp_path / "img.png"
        Image.fromarray(_textured_image(), mode="L").save(path)

        vec = emb.embed_media({"media_path": str(path)})
        assert vec is not None
        assert vec.shape == (DEFAULT_VLAD_CENTROIDS * SIFT_DESCRIPTOR_DIM,)
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-4)

    def test_local_features_forward_produces_keypoints(self, tmp_path):
        from PIL import Image

        emb = self._fresh_embedder()
        path = tmp_path / "img.png"
        Image.fromarray(_textured_image(seed=2), mode="L").save(path)

        feats = emb.local_features_forward({"media_path": str(path)})
        assert isinstance(feats, StructuralFeatures)
        assert feats.count > 0
        assert feats.descriptors.shape[1] == SIFT_DESCRIPTOR_DIM

    def test_embed_returns_none_for_missing_source(self):
        emb = self._fresh_embedder()
        emb.load_models()
        assert emb.embed_media({}) is None
        assert emb.local_features_forward({}) is None


class TestTwoStageBuildingBlock:
    """The geometric re-rank promotes a true instance match.

    The Stage-1->Stage-2 wiring into the sort path is deferred to a follow-up;
    this exercises the matcher-level invariant the re-rank relies on: a warped
    copy of the template verifies strongly while an unrelated image does not,
    so a candidate the global VLAD vector ranks only moderately can still be
    promoted by geometric support.
    """

    def test_warped_copy_outscores_unrelated_image(self):
        m = SiftMatcher()
        img = _textured_image(seed=4)
        template = m.detect_and_describe(img, max_features=400)

        warped = cv2.warpAffine(img, _similarity_matrix(-12.0, 0.9, -8.0, 5.0), (200, 200))
        true_match = m.detect_and_describe(warped, max_features=400)
        unrelated = m.detect_and_describe(_textured_image(seed=555), max_features=400)

        good = m.verify(template, true_match)
        bad = m.verify(template, unrelated)
        assert good.inlier_count > bad.inlier_count
        assert good.model_ok is True
        assert bad.model_ok is False
