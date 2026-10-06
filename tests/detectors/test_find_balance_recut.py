"""Regression: a Find-mode balance change must actually move the cutoff.

The find-label / detector-load training path has to cache what a re-cut reads
on the detector context, or ``recompute_detector_thresholds`` skips the
detector and a balance change is a silent no-op: the threshold never moves and
the good/bad split never changes.  Browse (which scopes over the good set) then
projects the *previous* cutoff's positives even though the user asked for a
different balance.  (The regression was found on the Inclusion slider, which
the balance's predecessor replaced; #4269 retired the slider.)

These tests pin (a) that find-label populates the cache and (b) that a balance
change re-derives the threshold and re-splits the unverified items, with the
``/api/votes`` good set (what Browse reads) staying in lock-step with the export
partition (what Export reads).

Under a balance the line is the labels' line (#4452): the class model the
retrain's calibration folds imply, cut at the beta, or the fold-anchored cut
when the folds fit no class model.  The fixture leaves 8 unvoted items, so
where the line lands at one beta or another is whatever the fit says; the
stored cutoff is poisoned before the change so a skipped recompute is
observable, and the re-cut is pinned to land on the balance's own line.
"""

from __future__ import annotations

import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context, human_voted_ids, recut_detector_threshold
from vtsearch.state import snapshot_medias


def _votes_good(client):
    """The unverified Good votes, which is what the unverified export partition counts."""
    data = client.get("/api/votes").get_json()
    return len(set(data["good"]) - set(data["verified"]))


def _export_good(client):
    resp = client.get("/api/labels/export?label_filter=unverified")
    return sum(1 for e in resp.get_json()["labels"] if e["label"] == "good")


# 6 good / 6 bad, leaving ids 13–20 unlabeled as the haystack a balance change
# re-splits.  The label count keeps the calibration folds non-separable, so the
# fitted estimators are real ones.
_GOOD_IDS = [1, 2, 3, 4, 5, 6]
_BAD_IDS = [7, 8, 9, 10, 11, 12]


def _run_find(client):
    detector_id = setup_trainable_model_in_registry(
        "balance-recut-regression",
        good_ids=_GOOD_IDS,
        bad_ids=_BAD_IDS,
        snap=snapshot_medias(),
    )
    load_detector_and_wait(client, detector_id)
    client.post("/api/find-label", json={"detector_id": detector_id})
    return detector_id


def test_find_label_populates_calibration_cache(client):
    _run_find(client)
    # The cache is what lets a later balance change re-derive the threshold;
    # without it the change is a no-op.
    ctx = get_active_detector_context()
    assert ctx.calibration_cache is not None
    assert ctx.anchored_cut_cache is not None
    assert ctx.precision_floor_cache is None, "a retrain no longer parks the #4220 estimate (#4362)"
    assert ctx.line_ranking is not None, "the ranking the balance's state counts the line against"


def test_a_balance_change_recuts_and_resplits(client):
    _run_find(client)
    ctx = get_active_detector_context()
    # The balance's line at beta 1, through the same re-cut a change runs: the
    # labels' line, or the fold-anchored cut with no class model (#4452).
    kept = recut_detector_threshold(ctx, beta=1.0)
    assert kept is not None

    # Start from another balance, so the move to 1 is a change, then poison
    # the stored cutoff so a skipped recompute is observable no matter where
    # the balance's cut lands.
    client.post("/api/balance", json={"beta": 2.0})
    ctx.threshold = -999.0

    # Change the balance WITHOUT re-running find-label (the pure re-cut path).
    resp = client.post("/api/balance", json={"beta": 1.0}).get_json()

    assert ctx.threshold == kept
    assert resp["threshold"] == ctx.threshold
    assert resp["status"] == "unchecked" and resp["beta"] == 1.0
    # Browse (votes) and Export (partition) never diverge.
    assert _export_good(client) == _votes_good(client)


def test_a_find_pass_on_a_reused_head_has_a_ranking_and_can_be_checked(client):
    """A Find pass that reuses the cached head gets its ranking back, and can be checked (#4273).

    Ending a Find session drops the ranking the line is read against; the next
    pass reuses the head as it was (no retrain), so before the fix it came back
    with no ranking, and a spot check refused to start ("No ranking to check")
    straight after a Find pass.
    """
    detector_id = _run_find(client)
    ctx = get_active_detector_context()
    head = ctx.model
    assert client.post("/api/find/end-session").status_code == 200
    assert ctx.line_ranking is None, "ending the session drops the ranking"
    ctx.threshold = -999.0

    resp = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()

    assert ctx.model is head, "the pass reused the cached head rather than retraining"
    assert ctx.line_ranking is not None
    # The fixture's 8 unlabelled items are the whole unvoted remainder, and the
    # balance's state counts how many of them the line keeps (#4452).
    voted = human_voted_ids(ctx)
    ranking = get_active_detector_context().line_ranking  # re-read: the pass set it after the drop above
    assert ranking is not None
    assert sorted(ranking.candidate(32, voted)) == list(range(13, 21))
    assert ctx.threshold != -999.0 and resp["threshold"] == pytest.approx(ctx.threshold, abs=1e-4)
    kept = [i for i in ranking.unvoted_ids(voted) if ranking.score_of(int(i)) >= ctx.threshold]
    assert resp["balance"]["status"] == "unchecked" and resp["balance"]["count"] == len(kept)

    start = client.post("/api/precision-check/start", json={})
    assert start.status_code == 200, start.get_json()
    assert set(start.get_json()["check"]["picks"]) <= set(range(13, 21))
