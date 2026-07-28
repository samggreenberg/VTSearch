"""Route tests for ``/api/find/queue-ids`` and ``/api/find/boundary-next``.

The server-side Find work-queue / boundary-walk endpoints (scalability.md
S3/S17/S19).  The underlying query logic is unit-tested in
``tests_lib/detectors/test_find_navigation.py``; these assert the HTTP layer:
schema validation, the detector-header gate, and that responses reflect the
active detector context's Find state.
"""

from __future__ import annotations

from vtscore.state.core import get_active_detector_context
from vtsearch.state import set_find_scores

_SCORES = {1: 0.9, 2: 0.8, 3: 0.6, 4: 0.3, 5: 0.1}


def _seed_find_state(verified: set[int], good: set[int], threshold: float = 0.5) -> None:
    ctx = get_active_detector_context()
    ctx.find_mode = True
    ctx.threshold = threshold
    set_find_scores(_SCORES)
    ctx.verified_ids.clear()
    ctx.verified_ids.update({mid: None for mid in verified})
    ctx.good_votes.clear()
    ctx.good_votes.update({mid: None for mid in good})


class TestQueueIdsRoute:
    def test_unverified_good(self, client):
        _seed_find_state(verified={1, 4}, good={1, 2, 3})
        resp = client.get("/api/find/queue-ids?filter=unverified_good")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["ids"] == [2, 3]
        assert body["count"] == 2

    def test_good(self, client):
        _seed_find_state(verified={1, 4}, good={1, 2, 3})
        body = client.get("/api/find/queue-ids?filter=good").get_json()
        assert body["ids"] == [1, 2, 3]
        assert body["count"] == 3

    def test_default_filter_is_unverified_good(self, client):
        _seed_find_state(verified={1, 4}, good={1, 2, 3})
        body = client.get("/api/find/queue-ids").get_json()
        assert body["ids"] == [2, 3]

    def test_invalid_filter_422(self, client):
        _seed_find_state(verified=set(), good=set())
        assert client.get("/api/find/queue-ids?filter=nonsense").status_code == 422

    def test_missing_detector_header_400(self, client):
        _seed_find_state(verified=set(), good=set())
        resp = client.get("/api/find/queue-ids", headers={"X-Detector-Id": ""})
        assert resp.status_code == 400


class TestBoundaryNextRoute:
    def test_above(self, client):
        _seed_find_state(verified={1, 4}, good={1, 2, 3})
        assert client.get("/api/find/boundary-next?side=above").get_json() == {"id": 3, "side": "above"}

    def test_below(self, client):
        _seed_find_state(verified={1, 4}, good={1, 2, 3})
        assert client.get("/api/find/boundary-next?side=below").get_json() == {"id": 5, "side": "below"}

    def test_exclude(self, client):
        _seed_find_state(verified={1, 4}, good={1, 2, 3})
        body = client.get("/api/find/boundary-next?side=above&exclude=3").get_json()
        assert body == {"id": 2, "side": "above"}

    def test_done_state(self, client):
        _seed_find_state(verified={1, 2, 3, 4, 5}, good={1, 2, 3})
        assert client.get("/api/find/boundary-next?side=above").get_json() == {"id": None, "side": None}

    def test_invalid_side_422(self, client):
        _seed_find_state(verified=set(), good=set())
        assert client.get("/api/find/boundary-next?side=sideways").status_code == 422
