"""The balance in the app (#4413, step 2): the setting, the line it draws, and the walk it runs.

``/api/balance`` reads and sets the active detector's beta beside the
deprecated ``/api/min-precision``; ``line_preference`` (``PUT /api/settings``)
says which of the two draws the line, the balance by default since the
switch's last step.  Under the balance the line is the mixture's F-beta argmax
under the cap, a check is the F-beta walk, and its result moves the line to
the walk's peak; under the floor a beta change moves nothing.
"""

from __future__ import annotations

import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import detector_balance_state, get_active_detector_context, human_voted_ids
from vtscore.training.thresholds import BALANCE_CHECKED, balance_line
from vtsearch.state import get_beta, get_line_preference, snapshot_medias


def _load_detector(client, name: str = "balance-carrier") -> str:
    detector_id = setup_trainable_model_in_registry(
        name, good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
    )
    load_detector_and_wait(client, detector_id)
    return detector_id


def _switch(client, preference: str) -> None:
    resp = client.put("/api/settings", json={"line_preference": preference})
    assert resp.status_code == 200, resp.get_json()
    assert get_line_preference() == preference


class TestTheSetting:
    def test_get_reports_the_default_balance_and_the_balance_preference(self, client):
        _load_detector(client)
        data = client.get("/api/balance").get_json()
        assert data["beta"] == 1.0 and data["line_preference"] == "balance"
        assert data["status"] == "unchecked" and data["schedule"]["candidate"] == 32
        assert data["precision"] is None and data["recall"] is None and data["fbeta"] is None

    def test_post_sets_the_balance_and_under_the_floor_moves_no_line(self, client, floor_preference):
        detector_id = _load_detector(client)
        before = client.post("/api/find-label", json={"detector_id": detector_id}).get_json()["threshold"]
        data = client.post("/api/balance", json={"beta": 2.0}).get_json()
        assert data["beta"] == 2.0 and get_beta() == 2.0
        assert data["schedule"]["candidate"] == 128, "recall-leaning: the larger cap"
        assert data["threshold"] == before, "the floor draws the line until the preference is switched"

    @pytest.mark.parametrize("bad", [True, None, "x"])
    def test_a_non_numeric_balance_is_refused(self, client, bad):
        _load_detector(client)
        assert client.post("/api/balance", json={"beta": bad}).status_code in (400, 422)

    def test_the_balance_is_clamped_like_the_floor(self, client):
        _load_detector(client)
        assert client.post("/api/balance", json={"beta": 9.0}).get_json()["beta"] == 4.0
        assert client.post("/api/balance", json={"beta": 0.01}).get_json()["beta"] == 0.25

    def test_an_unknown_preference_is_refused(self, client):
        _load_detector(client)
        assert client.put("/api/settings", json={"line_preference": "inclusion"}).status_code in (400, 422)


class TestTheLineUnderTheBalance:
    def test_switching_the_preference_moves_the_line_to_the_balances_set(self, client, floor_preference):
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        ctx = get_active_detector_context()
        _switch(client, "balance")
        data = client.get("/api/balance").get_json()
        expected = balance_line(ctx.line_ranking, 1.0, ctx.precision_check, human_voted_ids(ctx), proposal=None)
        state = detector_balance_state(ctx, 1.0)
        assert data["count"] == state["count"] and 1 <= data["count"] <= 8
        assert data["threshold"] == round(ctx.threshold, 4)
        assert ctx.threshold == balance_line(
            ctx.line_ranking, 1.0, ctx.precision_check, human_voted_ids(ctx), proposal=state["count"]
        )
        assert expected is not None
        # A beta change now moves the line (no retrain).
        moved = client.post("/api/balance", json={"beta": 0.5}).get_json()
        assert moved["beta"] == 0.5 and moved["threshold"] == round(ctx.threshold, 4)

    def test_a_check_under_the_balance_is_the_f_beta_walk_and_moves_the_line_to_its_peak(self, client):
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        start = client.post("/api/precision-check/start", json={})
        assert start.status_code == 200, start.get_json()
        check = start.get_json()["check"]
        assert check["beta"] == 1.0 and check["min_precision"] is None and check["status"] == "running"
        assert start.get_json()["balance"]["status"] == "unchecked"
        ctx = get_active_detector_context()
        positives = set(range(1, 7))
        data = start.get_json()
        for _ in range(20):
            if data["check"]["status"] != "running":
                break
            votes = [{"id": cid, "label": "good" if cid in positives else "bad"} for cid in data["check"]["picks"]]
            data = client.post("/api/precision-check/votes", json={"votes": votes}).get_json()
        assert data["check"]["status"] == BALANCE_CHECKED, "a balance walk ends checked, never short"
        assert data["balance"]["status"] == BALANCE_CHECKED and data["balance"]["count"] == data["check"]["candidate"]
        assert data["balance"]["fbeta"] is not None and data["balance"]["recall"] is not None
        assert ctx.threshold == ctx.line_ranking.threshold_for(data["balance"]["count"], human_voted_ids(ctx))
        assert data["floor"]["status"] == "unchecked", "the floor's own state is untouched by a balance walk"

    def test_under_the_floor_a_check_is_still_the_floors(self, client, floor_preference):
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        start = client.post("/api/precision-check/start", json={}).get_json()
        assert start["check"]["min_precision"] == 0.5 and "beta" not in start["check"]
        assert start["balance"]["status"] == "unchecked"
