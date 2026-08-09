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

**The regression is pinned twice, structurally and behaviourally.** The stored
cutoff is poisoned before the slide so a skipped recompute is observable no
matter what the cut rule returns, and the slide must land on exactly what the
cached estimator says at the new inclusion.  On top of that, the shipped
``mid_tilt`` rule answers the knob again (issue #2865; the interim midpoint
rule was numerically constant across it), so "the number moved" is back as a
meaningful assertion and the full-range slide must strictly raise the cutoff.
"""

from __future__ import annotations

from tests.helpers import setup_trainable_model_in_registry
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

    # Poison the stored cutoff so a skipped recompute is observable no matter
    # what the cut rule returns.  The original bug left the detector
    # untouched, which an inclusion-invariant rule (or a small realized tilt)
    # could mask unless the starting value is one no code path would produce.
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
    # A more exclusive Inclusion raises the cutoff -> no more positives.  The
    # full-range slide (10 -> -10) must move the number, not just fail to
    # lower it: mid_tilt answers the knob (issue #2865).
    assert exclusive_threshold > inclusive_threshold
    exclusive_good = _votes_good(client)
    assert exclusive_good <= inclusive_good
    # Browse (votes) and Export (partition) never diverge.
    assert _export_good(client) == exclusive_good
