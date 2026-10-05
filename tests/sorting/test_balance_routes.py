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
from vtscore.training.thresholds import BALANCE_CHECKED, BALANCE_GATE, CHECK_ADVISORY
from vtsearch.state import bad_votes, get_beta, get_line_preference, good_votes, snapshot_medias


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
        state = detector_balance_state(ctx, 1.0)
        # The labels' line (#4452), fitted by the retrain under the floor and re-cut on the switch; this
        # fixture's held-out votes do not separate, so there is no class model and the retrain's fallback
        # answers - never a count on the ranking.
        if ctx.labels_line is not None:
            assert ctx.threshold == pytest.approx(ctx.labels_line.threshold(1.0), abs=1e-12)
        assert data["threshold"] == pytest.approx(ctx.threshold, abs=1e-4)
        assert data["count"] == state["count"] >= 0, "what the threshold keeps of the ranking, possibly none"
        # A beta change re-cuts without a retrain.
        moved = client.post("/api/balance", json={"beta": 0.5}).get_json()
        assert moved["beta"] == 0.5 and moved["threshold"] == pytest.approx(ctx.threshold, abs=1e-4)
        if ctx.labels_line is not None:
            assert ctx.threshold == pytest.approx(ctx.labels_line.threshold(0.5), abs=1e-12)

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
        before = get_active_detector_context().threshold
        start, data = self._walk(client)
        check = start["check"]
        assert check["beta"] == 1.0 and check["min_precision"] is None and check["status"] == "running"
        assert start["balance"]["status"] == "unchecked" and start["balance"]["shape"] == CHECK_ADVISORY
        assert start["balance"]["audited"] is None
        ctx = get_active_detector_context()
        assert data["check"]["status"] == BALANCE_CHECKED, "a balance walk ends checked, never short"
        assert data["balance"]["status"] == BALANCE_CHECKED and data["balance"]["shape"] == CHECK_ADVISORY
        assert data["balance"]["audited"] == data["check"]["candidate"], "the walk's end is what the ranges describe"
        assert data["balance"]["fbeta"] is not None and data["balance"]["recall"] is not None
        # The line stays where the labels put it (#4452), whatever the walk audited.  The walk's picks are
        # votes, so the retrain behind the check may refit; the line is that refit's labels line or, with
        # no class model (this fixture), its fallback - never the walk's end.
        if ctx.labels_line is not None:
            assert ctx.threshold == pytest.approx(ctx.labels_line.threshold(1.0), abs=1e-12)
        assert data["balance"]["count"] != data["check"]["candidate"] or ctx.threshold == pytest.approx(before)
        assert data["floor"]["status"] == "unchecked", "the floor's own state is untouched by a balance walk"

    def test_a_check_at_beta_two_is_advisory_too(self, client):
        """#4452: the line comes from the labels at every preset; a beta-2 check audits and never moves it."""
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        moved = client.post("/api/balance", json={"beta": 2.0}).get_json()
        assert moved["shape"] == CHECK_ADVISORY
        ctx = get_active_detector_context()
        before = ctx.threshold
        start, data = self._walk(client)
        assert start["check"]["beta"] == 2.0 and data["check"]["status"] == BALANCE_CHECKED
        assert data["balance"]["shape"] == CHECK_ADVISORY and data["balance"]["audited"] == data["check"]["candidate"]
        assert ctx.threshold == pytest.approx(before, abs=1e-9) or ctx.labels_line is None

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

    def test_a_line_with_no_ranking_offers_no_check_and_start_refuses_it(self, client):
        """#4489: a structural detector's line is the verification gate's boundary, so its rerank drops the
        ranking; the balance says a check cannot start there, as start's 409 does."""
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        ctx = get_active_detector_context()
        assert client.get("/api/balance").get_json()["checkable"] is True
        ctx.line_ranking = None  # what ``maybe_structural_rerank`` leaves on a structural detector
        assert client.get("/api/balance").get_json()["checkable"] is False
        assert client.get("/api/precision-check").get_json()["balance"]["checkable"] is False
        assert client.post("/api/precision-check/start", json={}).status_code == 409

    def test_a_gate_line_reports_how_many_pass_the_gate(self, client):
        """#4505: on a structural line the count is what the gate passes, not the count rule's 32."""
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        ctx = get_active_detector_context()
        unvoted = [cid for cid in snapshot_medias() if cid not in human_voted_ids(ctx)]
        # What ``maybe_structural_rerank`` leaves on a structural detector.
        ctx.line_ranking = None
        ctx.gate_passed = frozenset(unvoted[:3])
        for balance in (
            client.get("/api/balance").get_json(),
            client.get("/api/precision-check").get_json()["balance"],
        ):
            assert balance["status"] == BALANCE_GATE and balance["count"] == 3
            assert balance["checkable"] is False and balance["precision"] is None

    def test_under_the_floor_a_check_is_still_the_floors(self, client, floor_preference):
        detector_id = _load_detector(client)
        client.post("/api/find-label", json={"detector_id": detector_id})
        start = client.post("/api/precision-check/start", json={}).get_json()
        assert start["check"]["min_precision"] == 0.5 and "beta" not in start["check"]
        assert start["balance"]["status"] == "unchecked"
