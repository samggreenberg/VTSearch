"""An inclusion slide deliberately drops the safe-threshold GMM blend.

Decision (docs/plans/comprehensive-audit-2026-07.md, open follow-up #1 -
resolved "skip blend on slides"): with safe-thresholds ON and 6 <= n < 20
labels, a *fresh* retrain stores a threshold blended between the
cross-calibration cutoff and a GMM cutoff (``calculate_safe_threshold``'s linear
ramp).  The fold-ordering cache on the detector context holds only the raw
cross-calibration orderings - never the GMM component - so an inclusion slide
(``recompute_detector_thresholds_for_inclusion``) re-derives the RAW
cross-calibration aggregate and does NOT reapply the blend.

That divergence is intentional: the slide is a cheap re-threshold over cached
orderings, not a re-blend, so the recompute stays stateless (no cached GMM
threshold / label count).  These tests pin the chosen semantics so the behaviour
can't silently flip to "re-blend on slide" (which would need the extra state) or
to "no-op on slide" (which would freeze the blended value and ignore the new
inclusion).
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


class TestInclusionSlideDropsSafeBlend:
    def test_slide_re_derives_raw_xcal_not_the_blend(self):
        """Safe on, 8 labels (inside the 6..19 blend window): the fresh-retrain
        cutoff is the GMM blend, but a slide re-derives the raw cross-cal
        aggregate over the cached fold orderings, dropping the blend."""
        rng = np.random.default_rng(7)
        # 20-media haystack so the GMM has a real score distribution to fit.
        clips = _clips(rng, range(500, 520))
        det_ctx = DetectorContext(detector_id="det-slide-blend", media_type="audio")

        good = {cid: None for cid in range(500, 504)}
        bad = {cid: None for cid in range(504, 508)}  # 8 labels: 6 <= n < 20
        _results, blended, model = train_and_score(
            clips, good, bad, inclusion_value=0, safe_thresholds=True, det_ctx=det_ctx
        )

        assert model is not None
        assert det_ctx.calibration_cache is not None, (
            "the >=6-label safe-on path must cache raw fold orderings so a slide can move the line"
        )
        _key, (orderings, fallback) = det_ctx.calibration_cache
        assert fallback is None and orderings

        raw = threshold_from_fold_orderings(orderings, 0)
        # The stored (fresh-retrain) cutoff is the GMM blend, distinct from the
        # raw cross-cal aggregate - otherwise this test proves nothing.
        assert abs(blended - raw) > 1e-9, (
            "expected the GMM blend to move the threshold off the raw cross-cal value "
            "in the 6..19 label window"
        )

        # Drive the slide path exactly as the route does: the trained threshold
        # is stored on the (registered) context, then the slider fires recompute.
        det_ctx.threshold = blended
        register_detector_context(det_ctx)
        recompute_detector_thresholds_for_inclusion(0)

        # The slide re-derived the RAW cross-cal aggregate; the blend is gone.
        assert det_ctx.threshold == raw
        assert det_ctx.threshold != blended

    def test_slide_below_ramp_floor_is_a_no_op(self):
        """Safe on, 4 labels (below the ramp floor): the fresh retrain clears
        the cache (pure-GMM regime), so a slide leaves the threshold untouched -
        no raw-xcal value to fall back to, and inclusion is irrelevant there."""
        rng = np.random.default_rng(9)
        clips = _clips(rng, range(600, 620))
        det_ctx = DetectorContext(detector_id="det-slide-floor", media_type="audio")

        good = {600: None, 601: None}
        bad = {602: None, 603: None}  # 4 labels < 6
        _results, gmm_threshold, model = train_and_score(
            clips, good, bad, inclusion_value=0, safe_thresholds=True, det_ctx=det_ctx
        )

        assert model is not None
        assert det_ctx.calibration_cache is None, "below the ramp floor the fold cache is cleared"

        det_ctx.threshold = gmm_threshold
        register_detector_context(det_ctx)
        recompute_detector_thresholds_for_inclusion(-10)

        # No cached orderings -> recompute skips this detector; the pure-GMM
        # value (inclusion-independent below the floor) stays put.
        assert det_ctx.threshold == gmm_threshold
