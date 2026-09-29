"""Regression: a Find-mode floor change must actually move the cutoff.

The find-label / detector-load training path has to cache the fitted estimators
on the detector context, or ``recompute_detector_thresholds`` skips the detector
and a floor change is a silent no-op: the threshold never moves and the
good/bad split never changes.  Browse (which scopes over the good set) then
projects the *previous* cutoff's positives even though the user asked for more
precision.  (The regression was found on the Inclusion slider, which the floor
replaced; #4269 retired the slider.)

These tests pin (a) that find-label populates the cache and (b) that a floor
change re-derives the threshold and re-splits the unverified items, with the
``/api/votes`` good set (what Browse reads) staying in lock-step with the export
partition (what Export reads).

The fixture's labels carry no learned-sort provenance, so a real retrain never
gives the floor evidence and every floor would draw the same Inclusion 0 line.
A promised estimate is planted on the context instead
(:func:`~tests.helpers.planted_precision_floor_estimate`), and the stored cutoff
is poisoned before the change so a skipped recompute is observable.
"""

from __future__ import annotations

from tests import load_detector_and_wait
from tests.helpers import planted_precision_floor_estimate, setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context
from vtsearch.state import snapshot_medias


def _votes_good(client):
    return len(client.get("/api/votes").get_json()["good"])


def _export_good(client):
    resp = client.get("/api/labels/export?label_filter=unverified")
    return sum(1 for e in resp.get_json()["labels"] if e["label"] == "good")


# 6 good / 6 bad, leaving ids 13–20 unlabeled as the haystack a floor change
# re-splits.  The label count keeps the calibration folds non-separable, so the
# fitted estimators are real ones.
_GOOD_IDS = [1, 2, 3, 4, 5, 6]
_BAD_IDS = [7, 8, 9, 10, 11, 12]


def _run_find(client):
    detector_id = setup_trainable_model_in_registry(
        "floor-recut-regression",
        good_ids=_GOOD_IDS,
        bad_ids=_BAD_IDS,
        snap=snapshot_medias(),
    )
    load_detector_and_wait(client, detector_id)
    client.post("/api/find-label", json={"detector_id": detector_id})
    return detector_id


def test_find_label_populates_calibration_cache(client):
    _run_find(client)
    # The cache is what lets a later floor change re-derive the threshold;
    # without it the change is a no-op.
    ctx = get_active_detector_context()
    assert ctx.calibration_cache is not None
    assert ctx.anchored_cut_cache is not None
    assert ctx.precision_floor_cache is not None


def test_a_floor_change_recuts_and_resplits(client):
    _run_find(client)
    ctx = get_active_detector_context()
    ctx.precision_floor_cache = planted_precision_floor_estimate(n_pos_per_fold=8)
    promised = ctx.precision_floor_cache.cut(0.5)
    assert promised.threshold is not None and promised.status.value == "promised"

    # Poison the stored cutoff so a skipped recompute is observable no matter
    # where the floor's cut lands.
    ctx.threshold = -999.0

    # Change the floor WITHOUT re-running find-label (the pure re-cut path).
    resp = client.post("/api/min-precision", json={"min_precision": 0.5}).get_json()

    assert ctx.threshold == promised.threshold
    assert resp["threshold"] == ctx.threshold
    assert resp["status"] == "promised"
    # Browse (votes) and Export (partition) never diverge.
    assert _export_good(client) == _votes_good(client)
