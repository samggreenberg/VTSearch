"""Regression: reloading a saved detector lands on the **fold-anchored** cut.

Issue #3257 proposed measuring a "load-then-continue" trajectory on the premise
that a user who re-opens a saved detector starts the session on the *pooled
conformal* cut - `train_detector_from_origins`, where "the conformal cut ships
alone" - and that this worse starting threshold then steers acquisition.  The
premise is wrong about the app: `train_detector_from_origins` is the
**library-tier** re-derivation entry point and has no in-app caller.  The app's
load path is `POST /api/detectors/registry/load` ->
`vtscore.detectors.labelset_training.train_from_labelset(..., snap=snap)` ->
`train_and_threshold`, which is handed the active dataset's medias as a
haystack and therefore fits the fold-anchored population estimator - the same
threshold a freshly trained detector gets.

This test pins that, because the claim is otherwise only visible by following
three call sites: what a resumed session starts on is what decides whether a
whole class of "the load-time cut steers everything after" experiments has
anything to measure.  It asserts the provenance directly (a fitted
`FoldAnchoredCut` is parked on the context and the shipped threshold *is* its
cut) and, so the assertion cannot pass vacuously, that the pooled conformal cut
over the same fold orderings is a different number.

Labels follow `test_find_inclusion_slide.py`: 6 good / 6 bad keeps the
calibration folds non-separable, leaving ids 13-20 as the haystack the
population estimator is fitted on.
"""

from __future__ import annotations

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import get_detector_context
from vtsearch.state import snapshot_medias

_GOOD_IDS = [1, 2, 3, 4, 5, 6]
_BAD_IDS = [7, 8, 9, 10, 11, 12]


def _load_saved_detector(client, name: str):
    """Write a detector to disk, load it through the real route, return its ctx."""
    detector_id = setup_trainable_model_in_registry(
        name,
        good_ids=_GOOD_IDS,
        bad_ids=_BAD_IDS,
        snap=snapshot_medias(),
    )
    client.post("/api/inclusion", json={"inclusion": 0})
    load_detector_and_wait(client, detector_id)
    ctx = get_detector_context(detector_id)
    assert ctx is not None
    return ctx


def test_reloaded_detector_starts_on_the_anchored_cut(client):
    ctx = _load_saved_detector(client, "reload-anchored")

    assert ctx.anchored_cut_cache is not None, (
        "the registry load path must fit the fold-anchored population estimator; "
        "without it a resumed session would start on the pooled conformal cut"
    )
    assert ctx.threshold == ctx.anchored_cut_cache.threshold_at(0)


def test_reloaded_threshold_is_not_the_pooled_conformal_cut(client):
    from vtscore.training.thresholds import threshold_from_fold_orderings

    ctx = _load_saved_detector(client, "reload-not-conformal")

    # The folds the conformal rule would cut over are cached on the context, so
    # the counterfactual is computable here: it is what the load-time threshold
    # would have been had no haystack reached `train_and_threshold`.
    assert ctx.calibration_cache is not None
    folds = ctx.calibration_cache[1]
    assert folds.fallback is None and folds.orderings
    pooled = threshold_from_fold_orderings(folds.orderings, 0)

    assert ctx.threshold != pooled, (
        "the anchored and pooled cuts coincide on this fixture, so the "
        "provenance assertion above no longer distinguishes them"
    )
