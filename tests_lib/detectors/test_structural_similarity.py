"""Tests for the Stage-2 geometric re-rank + verification classifier.

Exercises ``vtscore.training.structural_similarity``: RegionYes-as-template
filtering, max-over-templates verification, the match-statistic classifier
(plus its cold-start fallback), and the Stage-1->Stage-2 re-rank chokepoint
(the "two-stage flow" the design doc calls for - an item the VLAD coarse stage
ranks mid-pack but that geometrically verifies is promoted above a high-VLAD
item with no geometric support).

No model weights are downloaded - SIFT (via ``SiftMatcher``) is all that's
needed; the verification classifier trains on tiny synthetic match-stat vectors.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from vtscore.media.structural import (
    DEFAULT_MIN_INLIERS,
    SIFT_DESCRIPTOR_DIM,
    MatchStats,
    SiftMatcher,
    StructuralFeatures,
)
from vtscore.training.structural_similarity import (
    STRUCTURAL_DECISION_THRESHOLD,
    VerificationScorer,
    best_match_stats,
    build_templates,
    filter_features_to_box,
    maybe_structural_rerank,
    maybe_structural_rerank_example,
    snapshot_is_structural,
    structural_rerank,
    train_verification_classifier,
)


# --------------------------------------------------------------------------
# Synthetic image / feature helpers (shared shape with test_structural_embedder)
# --------------------------------------------------------------------------


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


# --------------------------------------------------------------------------
# RegionYes-as-template
# --------------------------------------------------------------------------


class TestFilterFeaturesToBox:
    def _feat(self, n: int = 20) -> StructuralFeatures:
        rng = np.random.default_rng(0)
        kp = rng.random((n, 4)).astype(np.float32)  # x, y in [0, 1]
        desc = (rng.random((n, SIFT_DESCRIPTOR_DIM)) * 255).astype(np.float32)
        return StructuralFeatures(keypoints=kp, descriptors=desc)

    def test_none_box_returns_unchanged(self):
        feat = self._feat()
        assert filter_features_to_box(feat, None) is feat

    def test_box_keeps_only_inside_keypoints(self):
        feat = self._feat(n=50)
        box = (0.0, 0.0, 0.5, 0.5)
        out = filter_features_to_box(feat, box)
        assert out.count <= feat.count
        # Every surviving keypoint is inside the box.
        assert (out.keypoints[:, 0] <= 0.5).all()
        assert (out.keypoints[:, 1] <= 0.5).all()
        # Descriptors are filtered in lockstep with keypoints.
        assert out.descriptors.shape[0] == out.count

    def test_unordered_box_is_normalised(self):
        feat = self._feat(n=40)
        ordered = filter_features_to_box(feat, (0.1, 0.2, 0.6, 0.7))
        flipped = filter_features_to_box(feat, (0.6, 0.7, 0.1, 0.2))
        np.testing.assert_array_equal(ordered.keypoints, flipped.keypoints)

    def test_empty_box_falls_back_to_full_features(self):
        feat = self._feat(n=30)
        # A box in a corner no keypoint lands in -> fall back to all features
        # (an empty template can never verify, so the full set is more useful).
        out = filter_features_to_box(feat, (0.999, 0.999, 1.0, 1.0))
        assert out is feat


class TestBuildTemplates:
    def test_one_template_per_good_vote_with_features(self):
        m = SiftMatcher()
        snap = {
            1: {"local_features": _feats(_textured_image(1), matcher=m)},
            2: {"local_features": _feats(_textured_image(2), matcher=m)},
            3: {},  # no features -> skipped
        }
        templates = build_templates({1: None, 2: None, 3: None}, snap, {})
        ids = {cid for cid, _ in templates}
        assert ids == {1, 2}

    def test_region_box_restricts_template(self):
        m = SiftMatcher()
        feat = _feats(_textured_image(5), matcher=m)
        snap = {1: {"local_features": feat}}
        templates = build_templates({1: None}, snap, {1: (0.0, 0.0, 0.5, 0.5)})
        (_, tpl) = templates[0]
        assert tpl.count <= feat.count
        assert (tpl.keypoints[:, 0] <= 0.5).all()


# --------------------------------------------------------------------------
# Max-over-templates verification
# --------------------------------------------------------------------------


class TestBestMatchStats:
    def test_picks_the_matching_template(self):
        m = SiftMatcher()
        base = _textured_image(7)
        t_match = _feats(base, matcher=m)
        t_other = _feats(_textured_image(8), matcher=m)
        candidate = _feats(_warp(base, 10.0, 1.1, 6.0, -4.0), matcher=m)

        # Order shouldn't matter - the strongest fit wins either way.
        s1 = best_match_stats([t_other, t_match], candidate, m)
        s2 = best_match_stats([t_match, t_other], candidate, m)
        assert s1.model_ok is True
        assert s2.model_ok is True
        assert s1.inlier_count == s2.inlier_count

    def test_no_templates_returns_empty_stats(self):
        m = SiftMatcher()
        candidate = _feats(_textured_image(9), matcher=m)
        stats = best_match_stats([], candidate, m)
        assert stats.model_ok is False
        assert stats.inlier_count == 0


# --------------------------------------------------------------------------
# Verification scorer (cold-start + classifier)
# --------------------------------------------------------------------------


class TestVerificationScorerColdStart:
    def test_model_not_ok_scores_zero(self):
        scorer = VerificationScorer(model=None)
        assert scorer.score(MatchStats(inlier_count=50, model_ok=False)) == 0.0

    def test_threshold_crossing_at_min_inliers(self):
        scorer = VerificationScorer(model=None, min_inliers=DEFAULT_MIN_INLIERS)
        at_gate = scorer.score(MatchStats(inlier_count=DEFAULT_MIN_INLIERS, model_ok=True))
        assert at_gate == pytest.approx(0.5)
        # Below the gate is below threshold; well above saturates at 1.0.
        assert scorer.score(MatchStats(inlier_count=2, model_ok=True)) < STRUCTURAL_DECISION_THRESHOLD
        assert scorer.score(MatchStats(inlier_count=100, model_ok=True)) == 1.0


class TestVerificationClassifier:
    def _shared_instance_snapshot(self):
        """Good votes are warps of one base image (a shared instance);
        bad votes are unrelated images."""
        m = SiftMatcher()
        base = _textured_image(101)
        snap = {
            1: {"local_features": _feats(base, matcher=m)},
            2: {"local_features": _feats(_warp(base, 12.0, 1.1, 8.0, -5.0), matcher=m)},
            3: {"local_features": _feats(_warp(base, -8.0, 0.9, -6.0, 4.0), matcher=m)},
            4: {"local_features": _feats(_textured_image(202), matcher=m)},
            5: {"local_features": _feats(_textured_image(303), matcher=m)},
        }
        good = {1: None, 2: None, 3: None}
        bad = {4: None, 5: None}
        return m, snap, good, bad

    def test_cold_start_below_min_votes_returns_none(self):
        m = SiftMatcher()
        snap = {
            1: {"local_features": _feats(_textured_image(1), matcher=m)},
            2: {"local_features": _feats(_textured_image(2), matcher=m)},
        }
        templates = build_templates({1: None}, snap, {})
        # Only 2 votes total (< MIN_VERIFICATION_VOTES) -> cold-start.
        clf = train_verification_classifier(templates, {1: None}, {2: None}, snap, m)
        assert clf is None

    def test_classifier_separates_match_from_non_match(self):
        m, snap, good, bad = self._shared_instance_snapshot()
        templates = build_templates(good, snap, {})
        clf = train_verification_classifier(templates, good, bad, snap, m)
        assert clf is not None

        scorer = VerificationScorer(model=clf)
        tpl_feats = [tpl for _, tpl in templates]

        # A held-out warp of the shared instance should score higher than an
        # unrelated image.
        base = _textured_image(101)
        held_out = _feats(_warp(base, 5.0, 1.05, 3.0, 2.0), matcher=m)
        unrelated = _feats(_textured_image(909), matcher=m)
        match_score = scorer.score(best_match_stats(tpl_feats, held_out, m))
        non_match_score = scorer.score(best_match_stats(tpl_feats, unrelated, m))
        assert match_score > non_match_score


# --------------------------------------------------------------------------
# Stage-1 -> Stage-2 chokepoint
# --------------------------------------------------------------------------


class TestStructuralRerank:
    def test_geometric_match_promoted_over_high_vlad_non_match(self):
        """The two-stage invariant: an item VLAD ranks mid-pack but that
        geometrically verifies is promoted above a high-VLAD non-match."""
        m = SiftMatcher()
        base = _textured_image(404)
        template = _feats(base, matcher=m)
        warped = _feats(_warp(base, -10.0, 0.95, -7.0, 5.0), matcher=m)
        unrelated = _feats(_textured_image(808), matcher=m)

        snap = {
            10: {"local_features": unrelated},  # high VLAD, no geometric support
            20: {"local_features": warped},  # mid VLAD, verifies strongly
        }
        # Stage-1 ranks the unrelated item first.
        results = [
            {"id": 10, "score": 0.91},
            {"id": 20, "score": 0.42},
        ]
        scorer = VerificationScorer(model=None)
        out = structural_rerank(results, snap, [template], scorer, m, top_k=50)

        assert out[0]["id"] == 20, "geometrically-verified item must be promoted"
        assert out[0]["score"] >= STRUCTURAL_DECISION_THRESHOLD
        # The verified item carries an inlier-box overlay.
        assert "best_region" in out[0]
        assert len(out[0]["best_region"]) == 4
        # The non-match falls below threshold and drops its (absent) overlay.
        non_match = next(e for e in out if e["id"] == 10)
        assert non_match["score"] < STRUCTURAL_DECISION_THRESHOLD
        assert "best_region" not in non_match

    def test_items_beyond_shortlist_are_zeroed_and_kept_in_order(self):
        m = SiftMatcher()
        base = _textured_image(11)
        template = _feats(base, matcher=m)
        snap = {
            1: {"local_features": _feats(_warp(base, 6.0, 1.0, 4.0, 0.0), matcher=m)},
            2: {"local_features": _feats(_textured_image(22), matcher=m)},
            3: {"local_features": _feats(_textured_image(33), matcher=m)},
        }
        results = [
            {"id": 1, "score": 0.9},
            {"id": 2, "score": 0.8},
            {"id": 3, "score": 0.7},
        ]
        scorer = VerificationScorer(model=None)
        # top_k=1 -> only id 1 is verified; ids 2,3 are the tail.
        out = structural_rerank(results, snap, [template], scorer, m, top_k=1)
        tail = out[1:]
        assert [e["id"] for e in tail] == [2, 3]  # Stage-1 order preserved
        assert all(e["score"] == 0.0 for e in tail)

    def test_no_templates_returns_input_unchanged(self):
        m = SiftMatcher()
        results = [{"id": 1, "score": 0.5}]
        out = structural_rerank(results, {}, [], VerificationScorer(), m)
        assert out == results


# --------------------------------------------------------------------------
# App-facing glue (gating)
# --------------------------------------------------------------------------


class TestMaybeStructuralRerank:
    def test_noop_for_non_structural_snapshot(self):
        snap = {1: {"embedding": np.zeros(8, dtype=np.float32)}}  # no local_features
        results = [{"id": 1, "score": 0.7}]
        out, thresh = maybe_structural_rerank(results, 0.33, snap, {1: None}, {}, {})
        assert out == results
        assert thresh == 0.33
        assert snapshot_is_structural(snap) is False

    def test_structural_snapshot_reranks_and_sets_threshold(self, monkeypatch):
        m = SiftMatcher()
        base = _textured_image(55)
        snap = {
            1: {"local_features": _feats(base, matcher=m), "embedder": "sift_vlad"},
            2: {"local_features": _feats(_warp(base, 9.0, 1.05, 5.0, -3.0), matcher=m), "embedder": "sift_vlad"},
            3: {"local_features": _feats(_textured_image(66), matcher=m), "embedder": "sift_vlad"},
        }
        # Resolve the matcher to our local SIFT instance (the registry embedder
        # is stubbed in the library test tier).
        monkeypatch.setattr(
            "vtscore.training.structural_similarity._resolve_matcher",
            lambda _snap: m,
        )
        results = [
            {"id": 3, "score": 0.95},  # high VLAD, unrelated
            {"id": 2, "score": 0.40},  # warp of the good-vote instance
            {"id": 1, "score": 0.30},  # the good-vote instance itself
        ]
        good = {1: None}
        bad = {3: None}
        out, thresh = maybe_structural_rerank(results, 0.77, snap, good, bad, {})
        assert thresh == STRUCTURAL_DECISION_THRESHOLD
        # id 2 (warp of the template) is geometrically verified and leads.
        assert out[0]["id"] == 2
        assert out[0]["score"] >= STRUCTURAL_DECISION_THRESHOLD

    def test_feature_snap_sources_templates_from_a_separate_snapshot(self, monkeypatch):
        """The labelset path supplies templates/classifier features from a
        synthetic ``feature_snap`` (re-derived cross-dataset features) while the
        re-rank runs over the active dataset's own ``snap``."""
        m = SiftMatcher()
        base = _textured_image(71)
        # Active dataset: an unrelated high-VLAD item and a warp of the template.
        snap = {
            10: {"local_features": _feats(_textured_image(72), matcher=m), "embedder": "sift_vlad"},
            20: {"local_features": _feats(_warp(base, 8.0, 1.05, 5.0, -3.0), matcher=m), "embedder": "sift_vlad"},
        }
        # The good vote lives only in the (cross-dataset) feature snapshot, NOT
        # in the active ``snap``: its id doesn't index into ``snap`` at all.
        feature_snap = {"good-elem": {"local_features": _feats(base, matcher=m)}}
        monkeypatch.setattr(
            "vtscore.training.structural_similarity._resolve_matcher",
            lambda _snap: m,
        )
        results = [
            {"id": 10, "score": 0.93},  # high VLAD, unrelated
            {"id": 20, "score": 0.31},  # warp of the cross-dataset template
        ]
        out, thresh = maybe_structural_rerank(
            results, 0.5, snap, {"good-elem": None}, {}, {}, feature_snap=feature_snap
        )
        assert thresh == STRUCTURAL_DECISION_THRESHOLD
        assert out[0]["id"] == 20, "the item matching the cross-dataset template must lead"
        assert out[0]["score"] >= STRUCTURAL_DECISION_THRESHOLD


class TestMaybeStructuralRerankExample:
    def test_noop_for_non_structural_snapshot(self):
        m = SiftMatcher()
        snap = {1: {"embedding": np.zeros(8, dtype=np.float32)}}  # no local_features
        results = [{"id": 1, "similarity": 0.7}]
        example = _feats(_textured_image(1), matcher=m)
        out, thresh = maybe_structural_rerank_example(results, 0.33, snap, example, score_key="similarity")
        assert out == results
        assert thresh == 0.33

    def test_noop_when_example_has_no_features(self):
        m = SiftMatcher()
        snap = {1: {"local_features": _feats(_textured_image(1), matcher=m), "embedder": "sift_vlad"}}
        results = [{"id": 1, "similarity": 0.7}]
        empty = StructuralFeatures(
            keypoints=np.zeros((0, 4), dtype=np.float32),
            descriptors=np.zeros((0, SIFT_DESCRIPTOR_DIM), dtype=np.float32),
        )
        out, thresh = maybe_structural_rerank_example(results, 0.33, snap, empty, score_key="similarity")
        assert out == results
        assert thresh == 0.33

    def test_example_template_promotes_geometric_match(self, monkeypatch):
        """The uploaded example is the template; an item that geometrically
        verifies against it is promoted above a high-cosine non-match, scored by
        the cold-start gate (example-sort has no votes)."""
        m = SiftMatcher()
        base = _textured_image(81)
        snap = {
            10: {"local_features": _feats(_textured_image(82), matcher=m), "embedder": "sift_vlad"},
            20: {"local_features": _feats(_warp(base, -9.0, 0.95, -6.0, 4.0), matcher=m), "embedder": "sift_vlad"},
        }
        monkeypatch.setattr(
            "vtscore.training.structural_similarity._resolve_matcher",
            lambda _snap: m,
        )
        example = _feats(base, matcher=m)
        results = [
            {"id": 10, "similarity": 0.95},  # high cosine, unrelated
            {"id": 20, "similarity": 0.40},  # warp of the example
        ]
        out, thresh = maybe_structural_rerank_example(results, 0.9, snap, example, score_key="similarity")
        assert thresh == STRUCTURAL_DECISION_THRESHOLD
        assert out[0]["id"] == 20
        assert out[0]["similarity"] >= STRUCTURAL_DECISION_THRESHOLD
        assert "best_region" in out[0]
        assert len(out[0]["best_region"]) == 4
