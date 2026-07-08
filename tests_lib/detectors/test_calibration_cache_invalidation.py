"""The <6-label training path must drop a stale fold-ordering cache.

Regression: ``_train_and_score_xy`` skips k-fold calibration below the
safe-threshold ramp floor (threshold = 0.5) but used to leave
``det_ctx.calibration_cache`` untouched.  After un-voting from ≥6 labels
down to ≤5 and retraining, a subsequent inclusion slide
(``recompute_detector_thresholds_for_inclusion``) trusted the stale cache
and re-thresholded against fold orderings computed for the *old* label
set and model.
"""

from __future__ import annotations

import numpy as np

from vtscore.detectors.training import train_and_score
from vtscore.state.core import DetectorContext

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


class TestCalibrationCacheClearedBelowRampFloor:
    def test_small_label_training_clears_stale_fold_cache(self):
        rng = np.random.default_rng(42)
        clips = _clips(rng, range(100, 104))
        det_ctx = DetectorContext(detector_id="det-cache-test", media_type="audio")
        stale_orderings = [([0.9, 0.1], [1.0, 0.0])]
        det_ctx.calibration_cache = ("stale-key", (stale_orderings, None))

        good = {100: None, 101: None}
        bad = {102: None, 103: None}
        results, threshold, model = train_and_score(clips, good, bad, det_ctx=det_ctx)

        assert model is not None
        assert threshold == 0.5
        assert det_ctx.calibration_cache is None, (
            "stale fold orderings must not survive a below-ramp-floor "
            "training; the inclusion slide re-thresholds from any non-None cache"
        )
        assert len(results) == len(clips)

    def test_large_label_training_repopulates_cache(self):
        rng = np.random.default_rng(43)
        clips = _clips(rng, range(200, 208))
        det_ctx = DetectorContext(detector_id="det-cache-test-2", media_type="audio")

        good = {cid: None for cid in range(200, 204)}
        bad = {cid: None for cid in range(204, 208)}
        _, _, model = train_and_score(clips, good, bad, det_ctx=det_ctx)

        assert model is not None
        assert det_ctx.calibration_cache is not None, (
            "the ≥6-label path caches fold orderings for the inclusion slide"
        )
