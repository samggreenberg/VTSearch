"""A small-label retrain must never leave a *stale* calibration-folds cache.

Regression: after un-voting from a larger label set down to a handful and
retraining, a subsequent inclusion slide
(``recompute_detector_thresholds_for_inclusion``) trusted whatever
``det_ctx.calibration_cache`` held - and used to re-threshold against fold
orderings computed for the *old* label set and model.

The cache is honest at every label count now, safe thresholds on or off:
``_train_and_score_xy`` computes the folds unconditionally (the shipped
fold-anchored threshold anchors on the fold models, so there is nothing to skip
at small counts), and the key is a fingerprint of the current labels - so a
stale entry is always *replaced*, never read.
"""

from __future__ import annotations

import numpy as np

from vtscore.detectors.training import train_and_score
from vtscore.state.core import DetectorContext
from vtscore.training.thresholds import CalibrationFolds

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


class TestCalibrationCacheStaysHonestAtSmallLabelCounts:
    def test_safe_off_small_label_training_replaces_stale_fold_cache(self):
        """Safe-thresholds off: the small-label retrain cross-calibrates, so a
        stale cache is overwritten with fresh orderings (never survives)."""
        rng = np.random.default_rng(42)
        clips = _clips(rng, range(100, 104))
        det_ctx = DetectorContext(detector_id="det-cache-test", media_type="audio")
        stale_orderings = [([0.9, 0.1], [1.0, 0.0])]
        det_ctx.calibration_cache = ("stale-key", CalibrationFolds(stale_orderings, None, []))

        good = {100: None, 101: None}
        bad = {102: None, 103: None}
        results, _threshold, model = train_and_score(clips, good, bad, det_ctx=det_ctx)

        assert model is not None
        assert det_ctx.calibration_cache is not None, (
            "small-label training cross-calibrates and must cache fresh fold orderings for the inclusion slide"
        )
        assert det_ctx.calibration_cache[0] != "stale-key", (
            "the stale fold orderings must be replaced by a cache keyed to the "
            "current label set, not left behind for the inclusion slide to read"
        )
        assert len(results) == len(clips)

    def test_safe_on_small_label_training_replaces_stale_fold_cache(self):
        """Safe-thresholds on: the folds feed the fold-anchored estimator, so
        they are computed here too and the stale entry is replaced."""
        rng = np.random.default_rng(44)
        clips = _clips(rng, range(300, 304))
        det_ctx = DetectorContext(detector_id="det-cache-test-safe", media_type="audio")
        stale_orderings = [([0.9, 0.1], [1.0, 0.0])]
        det_ctx.calibration_cache = ("stale-key", CalibrationFolds(stale_orderings, None, []))

        good = {300: None, 301: None}
        bad = {302: None, 303: None}
        _results, _threshold, model = train_and_score(clips, good, bad, safe_thresholds=True, det_ctx=det_ctx)

        assert model is not None
        assert det_ctx.calibration_cache is not None
        assert det_ctx.calibration_cache[0] != "stale-key", (
            "stale fold orderings must not survive a safe-on small-label "
            "training; the inclusion slide re-thresholds from any non-None cache"
        )

    def test_large_label_training_repopulates_cache(self):
        rng = np.random.default_rng(43)
        clips = _clips(rng, range(200, 208))
        det_ctx = DetectorContext(detector_id="det-cache-test-2", media_type="audio")

        good = {cid: None for cid in range(200, 204)}
        bad = {cid: None for cid in range(204, 208)}
        _, _, model = train_and_score(clips, good, bad, det_ctx=det_ctx)

        assert model is not None
        assert det_ctx.calibration_cache is not None, "training caches fold orderings for the inclusion slide"


class TestSafeOffSmallLabelSetCrossCalibrates:
    """#8: with safe-thresholds off, the vote/labelset path cross-calibrates at
    4-5 labels instead of hard-coding 0.5 - so it agrees with the Find path
    (``train_and_threshold``) for the same user state."""

    def test_safe_off_five_labels_stores_the_calibrated_aggregate(self):
        from vtscore.training.thresholds import threshold_from_fold_orderings

        rng = np.random.default_rng(11)
        clips = _clips(rng, range(400, 405))  # 5 media, none structural
        det_ctx = DetectorContext(detector_id="det-parity", media_type="audio")

        good = {400: None, 401: None}
        bad = {402: None, 403: None, 404: None}  # 5 labels < 6
        _results, threshold, model = train_and_score(
            clips, good, bad, inclusion_value=0, safe_thresholds=False, det_ctx=det_ctx
        )

        assert model is not None
        # Real fold orderings were computed and cached (the <6 skip is gone for
        # safe-off), so an inclusion slide can move the line below 6 labels too.
        assert det_ctx.calibration_cache is not None
        _key, folds = det_ctx.calibration_cache
        assert folds.fallback is None and folds.orderings, (
            "safe-thresholds-off below 6 labels must cross-calibrate, not hard-code 0.5"
        )
        # The stored cutoff is exactly the aggregate over those cached orderings
        # at the active inclusion - the value the Find path uses for this state.
        assert threshold == threshold_from_fold_orderings(folds.orderings, 0)
