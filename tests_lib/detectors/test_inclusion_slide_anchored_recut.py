"""An inclusion slide re-derives the *shipped* safe threshold, faithfully.

With safe thresholds ON the shipped cutoff is the fold-anchored population cut
(``fold_anchored_gmm_threshold``), not the cross-calibration quantile.  The
fitted estimator is parked on ``DetectorContext.anchored_cut_cache``, so
``recompute_detector_thresholds_for_inclusion`` can re-cut it at the new
inclusion with no refit and no re-scoring - and land on exactly the value a
fresh retrain at that inclusion would have stored.

That is a change from the pre-fusion behaviour, where the slide re-derived the
*raw* cross-calibration aggregate and silently dropped the GMM component of the
blend (comprehensive-audit-2026-07 follow-up #1, resolved "skip blend on
slides").  The blend needed extra state to reapply; the anchored estimator
carries its own, so the divergence no longer has to exist.  These tests pin the
new semantics against both regressions: sliding back to the raw cross-cal value,
and freezing (no-op) on the stale threshold.
"""

from __future__ import annotations

import numpy as np

from vtscore.detectors.training import train_and_score
from vtscore.state.core import (
    DetectorContext,
    recompute_detector_thresholds_for_inclusion,
    register_detector_context,
)
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


class TestInclusionSlideRecutsTheAnchoredEstimator:
    def test_slide_matches_a_fresh_retrain_at_the_new_inclusion(self):
        """Slide from inclusion 0 to 4; the threshold must equal what training
        directly at inclusion 4 produces, not the raw cross-cal quantile."""
        rng = np.random.default_rng(7)
        # 20-media haystack so the population estimator has a real distribution.
        clips = _clips(rng, range(500, 520))
        good = {cid: None for cid in range(500, 504)}
        bad = {cid: None for cid in range(504, 508)}

        det_ctx = DetectorContext(detector_id="det-slide-anchored", media_type="audio")
        _results, at_zero, model = train_and_score(
            clips, good, bad, inclusion_value=0, safe_thresholds=True, det_ctx=det_ctx
        )
        assert model is not None
        assert det_ctx.anchored_cut_cache is not None, (
            "safe-on training must park the fitted estimator so a slide can re-cut it"
        )

        # What a fresh retrain at inclusion 4 stores, on its own context.
        retrained_ctx = DetectorContext(detector_id="det-slide-anchored-4", media_type="audio")
        _r2, at_four, _m2 = train_and_score(
            clips, good, bad, inclusion_value=4, safe_thresholds=True, det_ctx=retrained_ctx
        )

        # Drive the slide path exactly as the route does: the trained threshold
        # is stored on the (registered) context, then the slider fires recompute.
        det_ctx.threshold = at_zero
        register_detector_context(det_ctx)
        recompute_detector_thresholds_for_inclusion(4)

        assert det_ctx.threshold == at_four
        # ...and the slide is not simply the raw cross-calibration quantile,
        # which is what the pre-fusion recompute fell back to.
        cache = det_ctx.calibration_cache
        assert cache is not None and cache[1].fallback is None
        raw_xcal = threshold_from_fold_orderings(cache[1].orderings, 4)
        assert abs(det_ctx.threshold - raw_xcal) > 1e-9

    def test_slide_is_monotone_across_the_whole_knob(self):
        """Nested inclusion sets: the re-cut threshold never rises with k."""
        rng = np.random.default_rng(9)
        clips = _clips(rng, range(600, 620))
        good = {cid: None for cid in range(600, 604)}
        bad = {cid: None for cid in range(604, 608)}

        det_ctx = DetectorContext(detector_id="det-slide-monotone", media_type="audio")
        _results, threshold, model = train_and_score(
            clips, good, bad, inclusion_value=0, safe_thresholds=True, det_ctx=det_ctx
        )
        assert model is not None
        det_ctx.threshold = threshold
        register_detector_context(det_ctx)

        seen = []
        for k in range(-10, 11):
            recompute_detector_thresholds_for_inclusion(k)
            seen.append(det_ctx.threshold)
        assert all(b <= a + 1e-12 for a, b in zip(seen, seen[1:], strict=False)), seen

    def test_safe_off_still_slides_on_the_conformal_rule(self):
        """No population estimator without safe thresholds: the slide falls back
        to re-thresholding the cached fold orderings, as it always has."""
        rng = np.random.default_rng(11)
        clips = _clips(rng, range(700, 720))
        good = {cid: None for cid in range(700, 704)}
        bad = {cid: None for cid in range(704, 708)}

        det_ctx = DetectorContext(detector_id="det-slide-safe-off", media_type="audio")
        _results, threshold, model = train_and_score(
            clips, good, bad, inclusion_value=0, safe_thresholds=False, det_ctx=det_ctx
        )
        assert model is not None
        assert det_ctx.anchored_cut_cache is None

        det_ctx.threshold = threshold
        register_detector_context(det_ctx)
        recompute_detector_thresholds_for_inclusion(-3)

        cache = det_ctx.calibration_cache
        assert cache is not None
        assert det_ctx.threshold == threshold_from_fold_orderings(cache[1].orderings, -3)
