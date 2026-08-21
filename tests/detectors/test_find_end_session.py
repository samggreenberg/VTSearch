"""Tests for ``POST /api/find/end-session`` (issue #3212).

A Find pass replaces the detector's in-memory votes with its own call for every
item in the dataset.  Returning to the Train window with those presumptions
still in place made the whole collection read as voted — Autopilot landed in a
terminal phase on arrival, and the find-mode write-back guard kept every new
training vote out of the labelset.  The Train window now ends the session on
entry; these tests pin what that does.
"""

from __future__ import annotations

import app as app_module
from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context
from vtsearch.state import snapshot_medias


def _detector_with_votes(client):
    """A loaded detector holding three good + four bad training votes."""
    detector_id = setup_trainable_model_in_registry("end-session", good_ids=[], bad_ids=[], snap=snapshot_medias())
    load_detector_and_wait(client, detector_id)
    for media_id in (1, 2, 3):
        assert client.post(f"/api/medias/{media_id}/vote", json={"target": "good"}).status_code == 200
    for media_id in (17, 18, 19, 20):
        assert client.post(f"/api/medias/{media_id}/vote", json={"target": "bad"}).status_code == 200
    return detector_id


class TestEndFindSession:
    """The Train window's on-entry hand-off out of a Find session."""

    def test_find_labels_all_items_then_end_session_restores_training_votes(self, client):
        detector_id = _detector_with_votes(client)

        resp = client.post("/api/find-label", json={"detector_id": detector_id})
        assert resp.status_code == 200, resp.get_json()
        after_find = client.get("/api/votes").get_json()
        # Every media carries a presumption, and they are not the training votes.
        assert len(after_find["good"]) + len(after_find["bad"]) == app_module.NUM_MEDIAS
        assert get_active_detector_context().find_mode is True

        ended = client.post("/api/find/end-session")
        assert ended.status_code == 200
        assert ended.get_json() == {"ok": True, "ended": True}

        after_end = client.get("/api/votes").get_json()
        assert set(after_end["good"]) == {1, 2, 3}
        assert set(after_end["bad"]) == {17, 18, 19, 20}
        assert after_end["verified"] == []
        assert get_active_detector_context().find_mode is False

    def test_end_session_lets_training_votes_reach_the_labelset_again(self, client):
        """A vote cast after the hand-off syncs; one cast in find mode does not."""
        detector_id = _detector_with_votes(client)
        client.post("/api/find-label", json={"detector_id": detector_id})

        client.post("/api/find/end-session")
        assert client.post("/api/medias/4/vote", json={"target": "good"}).status_code == 200

        votes = client.get("/api/votes").get_json()
        assert set(votes["good"]) == {1, 2, 3, 4}
        assert votes["labelset_good_count"] == 4
        assert votes["labelset_bad_count"] == 4

    def test_end_session_is_a_no_op_without_a_session(self, client):
        _detector_with_votes(client)

        resp = client.post("/api/find/end-session")
        assert resp.status_code == 200
        assert resp.get_json() == {"ok": True, "ended": False}

        votes = client.get("/api/votes").get_json()
        assert set(votes["good"]) == {1, 2, 3}
        assert set(votes["bad"]) == {17, 18, 19, 20}

    def test_end_session_drops_the_frozen_find_scores(self, client):
        """The work-queue / boundary-walk reads go empty with the session."""
        detector_id = _detector_with_votes(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        assert client.get("/api/find/queue-ids?filter=unverified_good").get_json()["count"] > 0

        client.post("/api/find/end-session")

        assert client.get("/api/find/queue-ids?filter=unverified_good").get_json()["count"] == 0
        ctx = get_active_detector_context()
        assert dict(ctx.find_scores) == {}
        assert dict(ctx.find_initial_labels) == {}
