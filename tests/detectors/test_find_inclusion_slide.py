"""Regression: a Find-mode Inclusion slide must actually move the cutoff.

Before the fix, the find-label / detector-load training path never cached the
K fold orderings on the detector context, so
``recompute_detector_thresholds_for_inclusion`` skipped the detector and an
Inclusion slide was a silent no-op: the threshold never moved and the good/bad
split never changed.  Browse (which scopes over the good set) then projected the
*previous* cutoff's positives even though the user had tightened Inclusion.

These tests pin (a) that find-label populates the cache and (b) that a slide
re-derives the threshold and re-splits the unverified items, with the
``/api/votes`` good set (what Browse reads) staying in lock-step with the export
partition (what Export reads).

**What "re-derives" can be asserted against has changed.** The shipped cut rule
is now the midpoint between the fitted component means, which ignores the cost
weights Inclusion arrives as - so the fused threshold is *numerically* constant
across the knob and "the number moved" can no longer stand in for "the slide
ran" (issue #2865).  The regression is therefore pinned structurally: the slide
must land on what the cached estimator says at the new inclusion, which is
exactly what the skipped-detector bug did not do.
"""

from __future__ import annotations

from helpers import setup_trainable_model_in_registry
from tests import load_detector_and_wait
from vtscore.state.core import get_active_detector_context
from vtsearch.state import snapshot_medias


def _votes_good(client):
    return len(client.get("/api/votes").get_json()["good"])


def _export_good(client):
    resp = client.get("/api/labels/export?label_filter=unverified")
    return sum(1 for e in resp.get_json()["labels"] if e["label"] == "good")


# 6 good / 6 bad, leaving ids 13–20 unlabeled as the haystack an Inclusion slide
# re-splits. The label count matters: with a tiny set (e.g. 4 good / 5 bad) the
# calibration folds are perfectly separable — at the optimal cut both FPR and
# FNR are 0, so no Inclusion weighting can move the threshold and the slide is a
# genuine (correct) no-op, which this regression can't observe. This split keeps
# the folds non-separable so tightening Inclusion demonstrably raises the cutoff.
_GOOD_IDS = [1, 2, 3, 4, 5, 6]
_BAD_IDS = [7, 8, 9, 10, 11, 12]


def _run_find(client):
    detector_id = setup_trainable_model_in_registry(
        "slide-regression",
        good_ids=_GOOD_IDS,
        bad_ids=_BAD_IDS,
        snap=snapshot_medias(),
    )
    load_detector_and_wait(client, detector_id)
    client.post("/api/inclusion", json={"inclusion": 10})
    client.post("/api/find-label", json={"detector_id": detector_id})
    return detector_id


def test_find_label_populates_calibration_cache(client):
    _run_find(client)
    # The cache is what lets a later slide re-derive the threshold; without it
    # the slide is a no-op.
    assert get_active_detector_context().calibration_cache is not None


def test_inclusion_slide_recuts_the_estimator_and_resplits(client):
    _run_find(client)
    ctx = get_active_detector_context()
    inclusive_threshold = ctx.threshold
    inclusive_good = _votes_good(client)
    assert _export_good(client) == inclusive_good

    # Poison the stored cutoff so a skipped recompute is observable.  The
    # original bug left the detector untouched; under an inclusion-invariant
    # cut rule that is indistinguishable from a correct re-derivation unless
    # the starting value is one no code path would produce.
    assert ctx.anchored_cut_cache is not None
    ctx.threshold = -999.0

    # Tighten Inclusion WITHOUT re-running find-label (the pure-slide path).
    resp = client.post("/api/inclusion", json={"inclusion": -10}).get_json()
    exclusive_threshold = ctx.threshold

    # The slide re-derived the cutoff from the cached estimator rather than
    # skipping the detector or dropping back to the raw cross-calibration
    # quantile.
    assert exclusive_threshold == ctx.anchored_cut_cache.threshold_at(-10)
    assert resp["threshold"] == exclusive_threshold
    # A more exclusive Inclusion never *lowers* the cutoff -> no more positives.
    assert exclusive_threshold >= inclusive_threshold
    exclusive_good = _votes_good(client)
    assert exclusive_good <= inclusive_good
    # Browse (votes) and Export (partition) never diverge.
    assert _export_good(client) == exclusive_good
