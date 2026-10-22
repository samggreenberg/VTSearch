"""Tests for ``GET /api/votes?scope=verified`` (issue #5115).

``/api/find-label`` gives every item the detector's call, so in Find the full
``/api/votes`` payload lists every id with a click time each: over a megabyte
on a 50k-item dataset, which the Review tab used to poll every two seconds.
The verified scope carries only what the Review tab shows, the items the human
has verified; these tests pin what it holds.
"""

from __future__ import annotations

import json

from tests import load_detector_and_wait
from tests.fixtures.medias import NUM_MEDIAS
from tests.helpers import setup_trainable_model_in_registry
from vtsearch.state import snapshot_medias

VERIFIED = "/api/votes?scope=verified"


def _find_session(client) -> str:
    """A loaded detector with three good + four bad training votes, then scored by Find."""
    detector_id = setup_trainable_model_in_registry("votes-scope", good_ids=[], bad_ids=[], snap=snapshot_medias())
    load_detector_and_wait(client, detector_id)
    for media_id in (1, 2, 3):
        assert client.post(f"/api/medias/{media_id}/vote", json={"target": "good"}).status_code == 200
    for media_id in (17, 18, 19, 20):
        assert client.post(f"/api/medias/{media_id}/vote", json={"target": "bad"}).status_code == 200
    resp = client.post("/api/find-label", json={"detector_id": detector_id})
    assert resp.status_code == 200, resp.get_json()
    return detector_id


class TestVerifiedScope:
    def test_no_presumptions_before_anything_is_verified(self, client):
        _find_session(client)

        full = client.get("/api/votes").get_json()
        scoped = client.get(VERIFIED).get_json()

        # The full scope carries the detector's call on every item ...
        assert len(full["good"]) + len(full["bad"]) == NUM_MEDIAS
        assert len(full["click_times"]) == NUM_MEDIAS
        # ... the verified scope none of it.
        assert scoped["good"] == []
        assert scoped["bad"] == []
        assert scoped["verified"] == []
        assert scoped["click_times"] == {}
        assert scoped["learned_scores"] == {}
        assert scoped["good_region_boxes"] == {}
        # The labelset counts do not depend on the scope.
        assert scoped["labelset_good_count"] == full["labelset_good_count"]
        assert scoped["labelset_bad_count"] == full["labelset_bad_count"]

    def test_verified_items_carry_their_polarity_click_time_and_score(self, client):
        from vtsearch.state import last_learned_scores

        _find_session(client)
        # A learned sort's score for every item: only the verified ones' go out.
        for media_id in range(1, NUM_MEDIAS + 1):
            last_learned_scores[media_id] = media_id / 100
        client.post("/api/medias/5/vote", json={"target": "good"})
        client.post("/api/medias/9/vote", json={"target": "bad"})

        full = client.get("/api/votes").get_json()
        scoped = client.get(VERIFIED).get_json()

        assert scoped["good"] == [5]
        assert scoped["bad"] == [9]
        assert scoped["verified"] == [5, 9]
        assert scoped["click_times"] == {"5": full["click_times"]["5"], "9": full["click_times"]["9"]}
        assert scoped["learned_scores"] == {"5": 0.05, "9": 0.09}

    def test_unvoting_a_verified_item_drops_it(self, client):
        _find_session(client)
        client.post("/api/medias/5/vote", json={"target": "good"})
        client.post("/api/medias/9/vote", json={"target": "bad"})

        client.post("/api/medias/5/vote", json={"target": "none"})

        scoped = client.get(VERIFIED).get_json()
        assert scoped["good"] == []
        assert scoped["bad"] == [9]
        assert scoped["verified"] == [9]
        assert set(scoped["click_times"]) == {"9"}

    def test_region_box_of_a_verified_good(self, client):
        _find_session(client)
        client.post("/api/medias/5/vote", json={"target": "good", "region_box": [0.1, 0.2, 0.7, 0.8]})

        scoped = client.get(VERIFIED).get_json()
        assert scoped["good_region_boxes"] == {"5": [0.1, 0.2, 0.7, 0.8]}

    def test_non_finite_learned_score_stays_strict_json(self, client):
        """The scoped path keeps the full scope's guard (audit M13)."""
        from vtsearch.state import last_learned_scores

        _find_session(client)
        client.post("/api/medias/5/vote", json={"target": "good"})
        last_learned_scores[5] = float("nan")

        resp = client.get(VERIFIED)

        def _reject_constant(c):
            raise AssertionError(f"non-strict JSON token in /api/votes response: {c!r}")

        data = json.loads(resp.data.decode("utf-8"), parse_constant=_reject_constant)
        assert data["learned_scores"]["5"] == -1.0

    def test_outside_find_nothing_is_verified(self, client):
        """Training votes are not verifications: the scope is empty, the counts are not."""
        detector_id = setup_trainable_model_in_registry("votes-scope", good_ids=[], bad_ids=[], snap=snapshot_medias())
        load_detector_and_wait(client, detector_id)
        client.post("/api/medias/1/vote", json={"target": "good"})
        client.post("/api/medias/17/vote", json={"target": "bad"})

        full = client.get("/api/votes").get_json()
        scoped = client.get(VERIFIED).get_json()

        assert full["good"] == [1]
        assert full["bad"] == [17]
        assert scoped["good"] == []
        assert scoped["bad"] == []
        assert scoped["labelset_good_count"] == full["labelset_good_count"] == 1
        assert scoped["labelset_bad_count"] == full["labelset_bad_count"] == 1


class TestScopeParameter:
    def test_default_and_explicit_all_match(self, client):
        _find_session(client)
        client.post("/api/medias/5/vote", json={"target": "good"})

        assert client.get("/api/votes").get_json() == client.get("/api/votes?scope=all").get_json()

    def test_unknown_scope_is_rejected(self, client):
        resp = client.get("/api/votes?scope=presumed")
        assert resp.status_code == 422
