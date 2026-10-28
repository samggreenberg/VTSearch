"""A detector that starts with labels gets Smart and Stable from its first sorts (#5358).

Loading a detector restores its saved labels into the votes without recording
them in the label history.  The Smart / Stable cache keyed every learned sort's
model by the votes it trained on, restored labels included, but built its own
label sets from the history alone, so it never found one: the indicators stayed
on "no history yet" and Autopilot never left Refine Boundary.  These run the
real load, vote, sort and status routes.
"""

from __future__ import annotations

import time

import vtscore.detectors.labeling_progress as lp
from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.concurrency.async_jobs import labeling_status_jobs
from vtscore.state.core import get_active_context, get_active_detector_context
from vtsearch.state import snapshot_medias

GOOD = [1, 2, 3]
BAD = [18, 19, 20]


def _load_trained_detector(client, name: str):
    detector_id = setup_trainable_model_in_registry(name, good_ids=GOOD, bad_ids=BAD, snap=snapshot_medias())
    load_detector_and_wait(client, detector_id)
    return get_active_detector_context()


def _vote_and_sort(client, votes: list[tuple[int, str]]) -> None:
    for cid, target in votes:
        assert client.post(f"/api/medias/{cid}/vote", json={"target": target}).status_code == 200
        assert client.post("/api/learned-sort", json={"wait": True}).status_code == 200


def _fresh_status(client) -> dict:
    """Poll ``/api/labeling-status`` until its background refresh has caught the cache up."""
    for _ in range(20):
        body = client.get("/api/labeling-status").get_json()
        if not body["stale"]:
            return body
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and labeling_status_jobs.active_jobs():
            for job in labeling_status_jobs.active_jobs():
                job.done_event.wait(timeout=1.0)
    raise AssertionError("labeling-status never caught up")


def _history(client, metric: str) -> list[dict]:
    body = client.get(f"/api/indicator-score-history?metric={metric}").get_json()
    assert body["complete"]
    return body["history"]


def test_load_records_the_restored_labels_as_the_base(client):
    ctx = _load_trained_detector(client, "restored-base-5358")

    assert ctx.label_history == []
    assert ctx.label_history_base == (frozenset(GOOD), frozenset(BAD))


def test_a_trained_detectors_sorts_are_plotted(client):
    _load_trained_detector(client, "restored-plot-5358")

    _vote_and_sort(client, [(4, "good"), (17, "bad"), (5, "good"), (16, "bad")])
    _fresh_status(client)

    assert len(_history(client, "smart")) == 4
    assert len(_history(client, "stable")) == 3


def test_a_rehydrate_starts_the_pair_over(client):
    """A dataset switch re-derives the votes: the base is the new restore, and the old steps go."""
    ctx = _load_trained_detector(client, "restored-rehydrate-5358")
    _vote_and_sort(client, [(4, "good"), (17, "bad")])
    _fresh_status(client)
    key = (get_active_context().dataset_id, ctx.detector_id)
    with lp._progress_lock:
        assert len(lp._caches[key].steps) == 2

    # The next request rehydrates, as it does when the detector comes back from another dataset.
    ctx.votes_dataset_id = "another-dataset"
    assert client.get("/api/votes").status_code == 200

    assert ctx.label_history == []
    assert ctx.label_history_base == (frozenset(GOOD + [4]), frozenset(BAD + [17]))
    with lp._progress_lock:
        assert key not in lp._caches
