"""A re-cut at another inclusion reads the *shipped* estimator, faithfully.

Inclusion is no longer a user preference (#4269), but it is still the unit the
threshold machinery measures cuts in: Autopilot's acquisition cut sits four
inclusion steps stricter than the line, and Smart prices every model at its own
Inclusion 0 cut.  Both re-cut through
:func:`vtscore.state.core.recut_detector_threshold`, which reads the
fold-anchored population estimator parked on
``DetectorContext.anchored_cut_cache`` - no refit, no re-scoring.

These tests pin that the re-cut is the shipped estimator's (not the raw
cross-calibration quantile), that it answers the inclusion monotonically, that
a detector with no estimator falls back to the conformal rule over its cached
fold orderings, and that a cut the folds never fitted is left alone.
"""

from __future__ import annotations

import numpy as np

from vtscore.detectors.training import train_and_score
from vtscore.state.core import DetectorContext, recut_detector_threshold
from vtscore.training.thresholds import threshold_from_fold_orderings

DIM = 8


def _clips(rng: np.random.Generator, cids: range) -> dict[int, dict]:
    return {
        cid: {
            "embeddings": {"test": rng.standard_normal(DIM).astype(np.float32)},
            "embedder": "test",
            "media_type": "audio",
            "md5": f"m{cid:031d}",
        }
        for cid in cids
    }


def _trained(seed: int, cids: range, n_good: int = 4, n_bad: int = 4, detector_id: str = "det-recut"):
    rng = np.random.default_rng(seed)
    clips = _clips(rng, cids)
    ids = list(cids)
    good = {cid: None for cid in ids[:n_good]}
    bad = {cid: None for cid in ids[n_good : n_good + n_bad]}
    det_ctx = DetectorContext(detector_id=detector_id, media_type="audio")
    results, threshold, model = train_and_score(clips, good, bad, det_ctx=det_ctx)
    assert model is not None
    det_ctx.threshold = threshold
    return det_ctx, results, threshold


class TestARecutReadsTheAnchoredEstimator:
    def test_the_recut_is_the_estimators_not_the_raw_cross_calibration(self):
        det_ctx, _results, _threshold = _trained(7, range(500, 520), detector_id="det-recut-anchored")
        assert det_ctx.anchored_cut_cache is not None, "safe-on training must park the fitted estimator"

        recut = recut_detector_threshold(det_ctx, 4)

        assert recut == det_ctx.anchored_cut_cache.threshold_at(4)
        cache = det_ctx.calibration_cache
        assert cache is not None and cache[1].fallback is None
        raw_xcal = threshold_from_fold_orderings(cache[1].orderings, 4)
        assert recut is not None and abs(recut - raw_xcal) > 1e-9

    def test_the_recut_at_inclusion_zero_is_the_trained_line(self):
        """With no floor the trained line is the Inclusion 0 cut, so a re-cut there reproduces it."""
        det_ctx, _results, threshold = _trained(7, range(500, 520), detector_id="det-recut-zero")
        assert det_ctx.min_precision is None
        assert recut_detector_threshold(det_ctx, 0) == threshold

    def test_the_recut_is_monotone_and_moves_the_admitted_set(self):
        """Nested inclusion sets: the re-cut threshold never rises with k."""
        det_ctx, results, _threshold = _trained(9, range(600, 620), detector_id="det-recut-monotone")

        seen: list[float] = []
        for k in range(-10, 11):
            recut = recut_detector_threshold(det_ctx, k)
            assert recut is not None, k
            seen.append(recut)
        assert all(b <= a + 1e-12 for a, b in zip(seen, seen[1:], strict=False)), seen
        # The cut actually moves, not merely fails to rise: a constant satisfies
        # monotonicity, which is how the inclusion-blind midpoint cut slipped
        # past this test; the shipped ``mid_tilt`` rule answers the inclusion
        # (issue #2865).
        assert seen[0] > seen[-1], seen
        # Moving the number is necessary but not sufficient: the cut is carried
        # to the final model as a quantile, so pin the observable - how many
        # distinct sets the inclusions admit.
        scores = [float(r["score"]) for r in results]
        admitted = {sum(1 for s in scores if s >= thr) for thr in seen}
        assert len(admitted) > 1, f"the re-cut moved the number but never the admitted set: {sorted(admitted)}"

    def test_without_an_estimator_the_recut_falls_back_to_the_conformal_rule(self):
        """A detector whose population fit degenerated re-thresholds its cached fold orderings."""
        det_ctx, _results, _threshold = _trained(11, range(700, 720), detector_id="det-recut-no-estimator")
        # Stand in for the degenerate-fit case, where the training pass parks
        # no estimator and the threshold came from the fallback blend.
        det_ctx.anchored_cut_cache = None

        cache = det_ctx.calibration_cache
        assert cache is not None
        assert recut_detector_threshold(det_ctx, -3) == threshold_from_fold_orderings(cache[1].orderings, -3)


class TestARecutLeavesUnfittedCutsAlone:
    def test_too_few_votes_has_nothing_to_recut(self):
        """Folds that never split carry a sentinel, not a cut an inclusion can move.

        With three votes the calibration folds fall back to 0.5 and training
        stores the schedule blend over the haystack.  Re-cutting used to write
        the bare sentinel over it; ``None`` tells the caller to keep the line.
        """
        det_ctx, _results, threshold = _trained(3, range(800, 840), n_good=2, n_bad=1, detector_id="det-recut-fallback")
        cache = det_ctx.calibration_cache
        assert cache is not None and cache[1].fallback is not None, "the fixture must be below the fold floor"
        assert det_ctx.anchored_cut_cache is None
        assert threshold != cache[1].fallback, "training must have stored the blend, not the sentinel"

        for k in (1, -3, 10):
            assert recut_detector_threshold(det_ctx, k) is None, k
