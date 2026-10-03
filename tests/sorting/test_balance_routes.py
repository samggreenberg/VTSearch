"""The balance in the app (#4413): the setting, the line it draws, and the walk it runs.

``/api/balance`` reads and sets the active detector's beta, the one
preference every line is drawn at since the precision floor was retired
(#4421).  The line is the mixture's F-beta argmax under the cap, a check is
the F-beta walk, and how its result moves the line follows the preset
(#4427): advisory at beta 1 and below, a trim above.
"""

from __future__ import annotations

import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import detector_balance_state, get_active_detector_context, human_voted_ids
from vtscore.training.thresholds import BALANCE_CHECKED, CHECK_ADVISORY, CHECK_TRIM, balance_count, balance_line
from vtsearch.state import bad_votes, get_beta, good_votes, snapshot_medias


def _load_detector(client, name: str = "balance-carrier") -> str:
    detector_id = setup_trainable_model_in_registry(
        name, good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
    )
    load_detector_and_wait(client, detector_id)
    return detector_id


class TestTheSetting:
    def test_get_reports_the_default_balance(self, client):
        _load_detector(client)
        data = client.get("/api/balance").get_json()
        assert data["beta"] == 1.0 and "line_preference" not in data
        assert data["status"] == "unchecked" and data["schedule"]["candidate"] == 32
        assert data["precision"] is None and data["recall"] is None and data["fbeta"] is None

    def test_post_sets_the_balance_and_moves_the_line(self, client):
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        data = client.post("/api/balance", json={"beta": 2.0}).get_json()
        assert data["beta"] == 2.0 and get_beta() == 2.0
        assert data["schedule"]["candidate"] == 128, "recall-leaning: the larger cap"
        ctx = get_active_detector_context()
        state = detector_balance_state(ctx, 2.0)
        assert state is not None
        assert data["count"] == state["count"] and data["threshold"] == round(ctx.threshold, 4)
        assert ctx.threshold == balance_line(
            ctx.line_ranking, 2.0, ctx.precision_check, human_voted_ids(ctx), proposal=state["count"]
        )

    @pytest.mark.parametrize("bad", [True, None, "x"])
    def test_a_non_numeric_balance_is_refused(self, client, bad):
        _load_detector(client)
        assert client.post("/api/balance", json={"beta": bad}).status_code in (400, 422)

    def test_the_balance_is_clamped_to_its_range(self, client):
        _load_detector(client)
        assert client.post("/api/balance", json={"beta": 9.0}).get_json()["beta"] == 4.0
        assert client.post("/api/balance", json={"beta": 0.01}).get_json()["beta"] == 0.25


class TestTheLineUnderTheBalance:
    def test_a_find_pass_draws_the_balances_set(self, client):
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        ctx = get_active_detector_context()
        data = client.get("/api/balance").get_json()
        expected = balance_line(ctx.line_ranking, 1.0, ctx.precision_check, human_voted_ids(ctx), proposal=None)
        state = detector_balance_state(ctx, 1.0)
        assert state is not None
        assert data["count"] == state["count"] and 1 <= data["count"] <= 8
        assert data["threshold"] == round(ctx.threshold, 4)
        assert ctx.threshold == balance_line(
            ctx.line_ranking, 1.0, ctx.precision_check, human_voted_ids(ctx), proposal=state["count"]
        )
        assert expected is not None
        # A beta change now moves the line (no retrain).
        moved = client.post("/api/balance", json={"beta": 0.5}).get_json()
        assert moved["beta"] == 0.5 and moved["threshold"] == round(ctx.threshold, 4)

    def _walk(self, client, positives=frozenset(range(1, 7))):
        start = client.post("/api/precision-check/start", json={})
        assert start.status_code == 200, start.get_json()
        data = start.get_json()
        for _ in range(20):
            if data["check"]["status"] != "running":
                break
            votes = [{"id": cid, "label": "good" if cid in positives else "bad"} for cid in data["check"]["picks"]]
            data = client.post("/api/precision-check/votes", json={"votes": votes}).get_json()
        return start.get_json(), data

    def test_a_check_at_beta_one_is_the_f_beta_walk_as_advisory_and_the_line_keeps_its_count(self, client):
        """#4427: at beta <= 1 the walk audits and reports, its votes stay votes, and the line keeps the unchecked
        rule's count; the state says which set was audited."""
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        start, data = self._walk(client)
        check = start["check"]
        assert check["beta"] == 1.0 and "min_precision" not in check and check["status"] == "running"
        assert start["balance"]["status"] == "unchecked" and start["balance"]["shape"] == CHECK_ADVISORY
        assert start["balance"]["audited"] is None
        ctx = get_active_detector_context()
        assert data["check"]["status"] == BALANCE_CHECKED, "a balance walk ends checked, never short"
        assert data["balance"]["status"] == BALANCE_CHECKED and data["balance"]["shape"] == CHECK_ADVISORY
        assert data["balance"]["audited"] == data["check"]["candidate"], "the walk's end is what the ranges describe"
        unchecked_state = detector_balance_state(ctx, 1.0)
        assert unchecked_state is not None
        unchecked = balance_count(1.0, None, proposal=unchecked_state["count"])
        assert data["balance"]["count"] == unchecked, "the line keeps the unchecked rule's count"
        assert data["balance"]["fbeta"] is not None and data["balance"]["recall"] is not None
        assert ctx.threshold == ctx.line_ranking.threshold_for(data["balance"]["count"], human_voted_ids(ctx))
        assert "floor" not in data

    def test_a_check_at_beta_two_trims_and_moves_the_line_to_its_end(self, client):
        """#4427: above beta 1 the walk may only step shallower, and the line takes the set it ends on."""
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        assert client.post("/api/balance", json={"beta": 2.0}).get_json()["shape"] == CHECK_TRIM
        start, data = self._walk(client)
        assert start["check"]["beta"] == 2.0 and start["balance"]["shape"] == CHECK_TRIM
        ctx = get_active_detector_context()
        assert data["check"]["status"] == BALANCE_CHECKED
        assert data["check"]["candidate"] <= data["check"]["start_candidate"], "never deeper than its start"
        assert data["balance"]["status"] == BALANCE_CHECKED and data["balance"]["shape"] == CHECK_TRIM
        assert data["balance"]["count"] == data["balance"]["audited"] == data["check"]["candidate"]
        assert ctx.threshold == ctx.line_ranking.threshold_for(data["balance"]["count"], human_voted_ids(ctx))

    def test_a_check_starts_when_the_mixture_has_no_estimate(self, client):
        """#4419: on the 20-item corpus the fit collapses onto the top two scores; the walk reads recall against the cap."""
        from vtscore.state.core import detector_balance_positives, detector_walk_positives

        good_votes.update({k: None for k in [1, 2]})
        bad_votes.update({k: None for k in [3, 4]})
        assert client.post("/api/learned-sort", json={"wait": True}).get_json()["status"] == "done"
        ctx = get_active_detector_context()
        assert detector_balance_positives(ctx) is None, "the fixture: no estimate"
        assert detector_walk_positives(ctx, 1.0) == len(ctx.line_ranking.unvoted_ids(human_voted_ids(ctx)))
        start = client.post("/api/precision-check/start", json={})
        assert start.status_code == 200, start.get_json()
        assert start.get_json()["check"]["status"] == "running" and start.get_json()["check"]["beta"] == 1.0
