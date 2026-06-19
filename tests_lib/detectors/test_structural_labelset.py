"""Tests for the structural Stage-2 re-rank on the saved-detector (labelset) path.

Exercises ``vtscore.detectors.labelset_training``'s structural helpers: the
cross-dataset local-feature cache projection (:func:`_labelset_feature_snapshot`),
the no-op gating of :func:`populate_label_local_features`, and the end-to-end
:func:`maybe_labelset_structural_rerank` that lets a saved structural detector
geometrically verify against a freshly loaded dataset.

No model weights are downloaded - ``SiftMatcher`` supplies the geometry and the
labelset's features are pre-seeded into the cache so the resolver/registry/disk
path stays out of these unit tests (the same isolation
``test_score_sanitization`` uses for the embedding cache).
"""

from __future__ import annotations

import cv2
import numpy as np

from vtscore.datasets.labelset import LabeledElement, LabelSet
from vtscore.detectors import labelset_training
from vtscore.detectors.labelset_elements import stable_element_id
from vtscore.media.structural import SiftMatcher, StructuralFeatures
from vtscore.state.core import DetectorContext
from vtscore.training.structural_similarity import STRUCTURAL_DECISION_THRESHOLD


def _textured_image(seed: int = 0, size: int = 200) -> np.ndarray:
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


def _warp(img: np.ndarray, angle: float, scale: float, tx: float, ty: float) -> np.ndarray:
    h, w = img.shape
    return cv2.warpAffine(img, _similarity_matrix(angle, scale, tx, ty), (w, h))


def _feats(img: np.ndarray, *, matcher: SiftMatcher, max_features: int = 400) -> StructuralFeatures:
    return matcher.detect_and_describe(img, max_features=max_features)


def _element(name: str, label: str, *, region_box=None) -> LabeledElement:
    return LabeledElement(
        origin={"importer": "test", "params": {}},
        origin_name=name,
        label=label,
        md5=(name * 32)[:32],
        region_box=region_box,
    )


class TestLabelsetFeatureSnapshot:
    def test_splits_good_bad_and_carries_region_boxes(self):
        m = SiftMatcher()
        det_ctx = DetectorContext(detector_id="d", media_type="image")
        good = _element("g", "good", region_box=(0.1, 0.1, 0.5, 0.5))
        good_no_box = _element("h", "good")
        bad = _element("b", "bad")
        unresolved = _element("u", "good")  # no cached features -> dropped
        skipped = _element("s", "")  # not a good/bad vote -> ignored
        labelset = LabelSet(elements=[good, good_no_box, bad, unresolved, skipped])

        det_ctx.label_local_features[stable_element_id(good)] = _feats(_textured_image(1), matcher=m)
        det_ctx.label_local_features[stable_element_id(good_no_box)] = _feats(_textured_image(2), matcher=m)
        det_ctx.label_local_features[stable_element_id(bad)] = _feats(_textured_image(3), matcher=m)

        feat_snap, good_votes, bad_votes, region_boxes = labelset_training._labelset_feature_snapshot(det_ctx, labelset)

        assert set(good_votes) == {stable_element_id(good), stable_element_id(good_no_box)}
        assert set(bad_votes) == {stable_element_id(bad)}
        # The region box rides through only for the element that had one.
        assert region_boxes == {stable_element_id(good): (0.1, 0.1, 0.5, 0.5)}
        # Every voted+cached element appears in the synthetic snapshot.
        assert set(feat_snap) == set(good_votes) | set(bad_votes)
        for entry in feat_snap.values():
            assert isinstance(entry["local_features"], StructuralFeatures)


class TestPopulateLabelLocalFeatures:
    def test_noop_without_active_embedder(self):
        """No active dataset (empty snap) -> no embedder -> nothing cached."""
        det_ctx = DetectorContext(detector_id="d", media_type="image")
        labelset = LabelSet(elements=[_element("g", "good")])
        cached = labelset_training.populate_label_local_features(det_ctx, labelset, snap={})
        assert cached == 0
        assert dict(det_ctx.label_local_features) == {}


class TestMaybeLabelsetStructuralRerank:
    def test_noop_for_non_structural_dataset(self):
        det_ctx = DetectorContext(detector_id="d", media_type="image")
        labelset = LabelSet(elements=[_element("g", "good")])
        snap = {1: {"embedding": np.zeros(8, dtype=np.float32)}}  # no local_features
        results = [{"id": 1, "score": 0.7}]
        out, thresh = labelset_training.maybe_labelset_structural_rerank(det_ctx, labelset, results, 0.33, snap)
        assert out == results
        assert thresh == 0.33

    def test_reranks_active_dataset_against_cross_dataset_template(self, monkeypatch):
        """A saved structural detector verifies the freshly loaded dataset
        against templates re-derived from its (cross-dataset) labelset."""
        m = SiftMatcher()
        base = _textured_image(40)

        det_ctx = DetectorContext(detector_id="d", media_type="image")
        good = _element("g", "good")
        bad = _element("b", "bad")
        labelset = LabelSet(elements=[good, bad])
        # Pre-seed the cross-dataset feature cache (the labelset's media are NOT
        # in the loaded dataset) and skip the resolver-driven population pass.
        det_ctx.label_local_features[stable_element_id(good)] = _feats(base, matcher=m)
        det_ctx.label_local_features[stable_element_id(bad)] = _feats(_textured_image(41), matcher=m)
        monkeypatch.setattr(labelset_training, "populate_label_local_features", lambda *a, **kw: 0)
        monkeypatch.setattr(
            "vtscore.training.structural_similarity._resolve_matcher",
            lambda _snap: m,
        )

        # Active dataset: a high-VLAD unrelated item and a mid-VLAD warp of the
        # good template.
        snap = {
            10: {"local_features": _feats(_textured_image(42), matcher=m), "embedder": "sift_vlad"},
            20: {"local_features": _feats(_warp(base, 9.0, 1.05, 6.0, -4.0), matcher=m), "embedder": "sift_vlad"},
        }
        results = [
            {"id": 10, "score": 0.94},  # high VLAD, unrelated
            {"id": 20, "score": 0.35},  # warp of the cross-dataset template
        ]
        out, thresh = labelset_training.maybe_labelset_structural_rerank(det_ctx, labelset, results, 0.5, snap)

        assert thresh == STRUCTURAL_DECISION_THRESHOLD
        assert out[0]["id"] == 20, "geometrically-verified item must lead"
        assert out[0]["score"] >= STRUCTURAL_DECISION_THRESHOLD
        assert "best_region" in out[0]
