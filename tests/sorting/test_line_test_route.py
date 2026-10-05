"""``/api/line-test`` (#4524): each state of Test mode's test of the line through the endpoints.

Start freezes the ranking off the Find pass's scores and the line at its
threshold and deals round one; votes land as session votes (provenance
``test``, verified in Find mode, never in the labelset) and move the ranges;
the ↓ key takes one back; a whole round deals the next, until the phase
machine says done; cancel drops a running test and keeps a finished one; the
result reads ``stale`` once corrections are folded in and ``moved`` once the
line no longer keeps the set it measured.

The fixture's Find pass scores a small corpus, so a test is a few rounds; the
long cases plant a 200-item ranking on the context instead.
"""

from __future__ import annotations

import pytest

from tests import load_detector_and_wait
from tests.helpers import setup_trainable_model_in_registry
from vtscore.state.core import get_active_detector_context
from vtscore.training.thresholds import PHASE_DONE, PHASE_MATCHES, PHASE_MISSES, PHASE_NOTHING
from vtsearch.state import snapshot_medias


def _run_find(client) -> str:
    """Score the fixture corpus with a fresh detector; returns its registry id, for a second pass."""
    detector_id = setup_trainable_model_in_registry(
        "test-route", good_ids=[1, 2, 3, 4, 5, 6], bad_ids=[7, 8, 9, 10, 11, 12], snap=snapshot_medias()
    )
    load_detector_and_wait(client, detector_id)
    _score(client, detector_id)
    return detector_id


def _score(client, detector_id: str) -> None:
    resp = client.post("/api/find-label", json={"detector_id": detector_id})
    assert resp.status_code == 200, resp.get_json()


def _plant_big_corpus(n: int = 200, line: int = 64) -> None:
    """Replace the Find pass's frozen scores with ids 1..n on a descending ladder, the line keeping the top *line*."""
    ctx = get_active_detector_context()
    ctx.find_scores = {cid: 1.0 - cid / (n + 1) for cid in range(1, n + 1)}
    ctx.threshold = 1.0 - line / (n + 1)
    ctx.labels_line = None
    ctx.line_test = None


def _votes(picks, right) -> dict:
    return {"votes": [{"id": cid, "label": "good" if right else "bad"} for cid in picks]}


def _start(client) -> dict:
    resp = client.post("/api/line-test/start", json={})
    assert resp.status_code == 200, resp.get_json()
    return resp.get_json()


def _vote_until_done(client, data: dict, right=lambda cid: True, limit: int = 60) -> dict:
    for _ in range(limit):
        test = data["test"]
        if test["phase"] in (PHASE_DONE, PHASE_NOTHING):
            return data
        picks = test["picks"]
        assert picks, test
        votes = {"votes": [{"id": cid, "label": "good" if right(cid) else "bad"} for cid in picks]}
        resp = client.post("/api/line-test/votes", json=votes)
        assert resp.status_code == 200, resp.get_json()
        data = resp.get_json()
    pytest.fail("the test did not finish")


class TestBeforeAnyTest:
    def test_get_reports_the_line_and_no_test(self, client):
        _run_find(client)
        data = client.get("/api/line-test").get_json()
        assert data["test"] is None and data["presets"] == []
        assert data["stale"] is False and data["moved"] is False
        assert data["line_count"] >= 0 and data["balance"]["status"] in ("unchecked", "checked", "gate")

    def test_start_needs_a_find_pass(self, client):
        resp = client.post("/api/line-test/start", json={})
        assert resp.status_code == 409
        assert "score the dataset" in resp.get_json()["message"]

    def test_votes_and_unvote_need_a_running_test(self, client):
        _run_find(client)
        assert client.post("/api/line-test/votes", json=_votes([1], True)).status_code == 409
        assert client.post("/api/line-test/unvote", json={"id": 1}).status_code == 409
        # Nothing running: cancel is a no-op.
        assert client.post("/api/line-test/cancel", json={}).get_json()["test"] is None


class TestOneRound:
    def test_start_freezes_the_ranking_and_deals_round_one_from_the_band_holding_the_line(self, client):
        _run_find(client)
        _plant_big_corpus()
        data = _start(client)
        test = data["test"]
        assert test["phase"] == PHASE_MATCHES and test["round"] == 1
        assert (test["line_count"], test["size"], test["picks_per_round"]) == (64, 200, 5)
        assert len(test["picks"]) == 5
        # The first round audits the band holding the line: ranks 33–64, the last band above it.
        assert test["band"] == {"index": 3, "side": "above", "lo": 33, "hi": 64}
        assert set(test["picks"]) <= set(range(33, 65))
        assert test["estimates"]["labelled"] == 0
        assert data["line_count"] == 64 and data["moved"] is False
        assert client.get("/api/line-test").get_json()["test"]["picks"] == test["picks"]

    def test_a_stray_vote_is_refused(self, client):
        _run_find(client)
        _plant_big_corpus()
        picks = _start(client)["test"]["picks"]
        stray = next(cid for cid in range(33, 65) if cid not in picks)
        assert client.post("/api/line-test/votes", json=_votes([stray], True)).status_code == 400

    def test_a_partial_round_moves_the_ranges_and_a_whole_round_deals_the_next(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        picks = _start(client)["test"]["picks"]

        data = client.post("/api/line-test/votes", json=_votes(picks[:2], True)).get_json()
        test = data["test"]
        assert test["picks"] == picks[2:] and test["labelled"] == 2 and test["round"] == 1
        assert test["bands"][3]["labelled"] == 2 and test["bands"][3]["right"] == 2
        assert test["estimates"]["labelled"] == 2

        data = client.post("/api/line-test/votes", json=_votes(picks[2:], True)).get_json()
        test = data["test"]
        assert test["round"] == 2 and test["labelled"] == 5
        assert len(test["picks"]) == 5 and not set(test["picks"]) & set(picks)
        # The next band above the line, working up.
        assert test["band"]["index"] == 2 and test["band"]["side"] == "above"

        # The votes are session votes: verified, tagged as the test's, never the line's movers.
        assert all(cid in ctx.good_votes and cid in ctx.verified_ids for cid in picks)
        assert all(ctx.vote_provenance[cid]["flow"] == "test" for cid in picks)
        assert ctx.threshold == pytest.approx(1.0 - 64 / 201)

    def test_the_down_key_takes_a_vote_back(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        picks = _start(client)["test"]["picks"]
        client.post("/api/line-test/votes", json=_votes(picks[:1], False))
        assert picks[0] in ctx.bad_votes and picks[0] in ctx.verified_ids

        data = client.post("/api/line-test/unvote", json={"id": picks[0]}).get_json()
        assert data["test"]["labelled"] == 0 and set(data["test"]["picks"]) == set(picks)
        assert picks[0] not in ctx.bad_votes and picks[0] not in ctx.verified_ids
        # Only a vote of this round can be taken back.
        assert client.post("/api/line-test/unvote", json={"id": picks[1]}).status_code == 400

    def test_cancel_drops_a_running_test_and_keeps_the_votes_cast(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        picks = _start(client)["test"]["picks"]
        client.post("/api/line-test/votes", json=_votes(picks[:2], True))
        data = client.post("/api/line-test/cancel", json={}).get_json()
        assert data["test"] is None and ctx.line_test is None
        assert all(cid in ctx.good_votes for cid in picks[:2])

    def test_a_running_test_is_replaced_by_a_new_start(self, client):
        _run_find(client)
        _plant_big_corpus()
        first = _start(client)["test"]
        client.post("/api/line-test/votes", json=_votes(first["picks"][:2], True))
        second = _start(client)["test"]
        assert second["labelled"] == 0 and second["round"] == 1


class TestToDone:
    def test_matches_then_misses_then_done_and_a_finished_test_is_kept(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        data = _start(client)
        phases = []
        for _ in range(60):
            phases.append(data["test"]["phase"])
            if data["test"]["phase"] == PHASE_DONE:
                break
            picks = data["test"]["picks"]
            # Everything above the line is right, nothing below it: a clean line.
            data = client.post("/api/line-test/votes", json=_votes(picks, picks[0] <= 64)).get_json()
        test = data["test"]
        assert test["phase"] == PHASE_DONE and test["picks"] == []
        assert PHASE_MATCHES in phases and PHASE_MISSES in phases
        assert phases.index(PHASE_MISSES) > phases.index(PHASE_MATCHES)
        report = test["report"]
        assert report["matches_stop"] is not None and report["misses_stop"] is not None
        est = test["estimates"]
        assert est["precision"]["lo"] > 0.5 and est["precision"]["hi"] <= 1.0
        assert est["found"] and est["at_edges"]
        # Done: a vote is refused, cancel keeps the result, and start returns it as it is.
        assert client.post("/api/line-test/votes", json=_votes([1], True)).status_code == 409
        assert client.post("/api/line-test/cancel", json={}).get_json()["test"]["phase"] == PHASE_DONE
        again = _start(client)["test"]
        assert again["labelled"] == test["labelled"] and ctx.line_test is not None

    def test_no_class_model_means_no_presets(self, client):
        _run_find(client)
        _plant_big_corpus()
        data = _vote_until_done(client, _start(client))
        assert data["presets"] == []

    def test_a_class_model_gives_the_line_each_preset_would_ship(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        from vtscore.training.thresholds import LabelsLine  # noqa: PLC0415
        from vtscore.training.thresholds.labels_line import ClassScoreModel  # noqa: PLC0415

        ctx.labels_line = LabelsLine(ClassScoreModel(2.0, -2.0, 1.0, 10, 10), 0.3)
        data = _vote_until_done(client, _start(client), right=lambda cid: cid <= 64)
        presets = data["presets"]
        assert [p["beta"] for p in presets] == [0.25, 1.0, 4.0]
        # A balance toward false positives keeps more, and every reading is a range.
        assert presets[0]["count"] <= presets[1]["count"] <= presets[2]["count"]
        for p in presets:
            assert 0.0 <= p["precision"]["lo"] <= p["precision"]["hi"] <= 1.0
            assert 0.0 <= p["recall"]["lo"] <= p["recall"]["hi"] <= 1.0
            assert p["found"]


class TestStaleAndMoved:
    def test_corrections_make_the_result_stale(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        _start(client)
        ctx.find_eval_stale = True
        assert client.get("/api/line-test").get_json()["stale"] is True

    def test_a_line_that_no_longer_keeps_the_tested_set_reads_moved(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        _start(client)
        assert client.get("/api/line-test").get_json()["moved"] is False
        ctx.threshold = 1.0 - 100 / 201
        data = client.get("/api/line-test").get_json()
        assert data["moved"] is True and data["line_count"] == 100
        # A fresh start tests the new line.
        assert _start(client)["test"]["line_count"] == 100

    def test_a_fresh_find_pass_drops_the_test(self, client):
        detector_id = _run_find(client)
        ctx = get_active_detector_context()
        _plant_big_corpus()
        _start(client)
        assert ctx.line_test is not None
        _score(client, detector_id)
        assert ctx.line_test is None
        assert client.get("/api/line-test").get_json()["test"] is None

    def test_clearing_the_votes_drops_the_test(self, client):
        _run_find(client)
        _plant_big_corpus()
        _start(client)
        assert client.post("/api/votes/clear").status_code == 200
        assert get_active_detector_context().line_test is None


class TestNothingToTest:
    def test_a_line_that_keeps_fewer_than_a_round_has_nothing_to_test(self, client):
        _run_find(client)
        _plant_big_corpus(line=3)
        data = _start(client)
        test = data["test"]
        assert test["phase"] == PHASE_NOTHING and test["picks"] == [] and test["estimates"] is None
        assert data["presets"] == []
        assert client.post("/api/line-test/votes", json=_votes([1], True)).status_code == 409


def _detector_file() -> tuple:
    """The active detector's JSON path and its parsed content."""
    from vtscore.detectors.registry import get_detector  # noqa: PLC0415
    from vtscore.detectors.store import _detector_path, _read_detector  # noqa: PLC0415

    entry = get_detector(get_active_detector_context().detector_id)
    assert entry is not None
    path = _detector_path(entry["name"])
    return path, _read_detector(path) or {}


def _kept():
    from vtscore.detectors.line_verdicts import read_verdicts  # noqa: PLC0415

    return read_verdicts(_detector_file()[1])


def _retrain_on_disk() -> None:
    """Change the detector file's labels, as a retrain does: one more label."""
    from vtscore.datasets.labelset import LabeledElement, LabelSet  # noqa: PLC0415
    from vtscore.detectors.store import _write_detector  # noqa: PLC0415

    path, data = _detector_file()
    labelset = LabelSet.from_dict(data["labelset"])
    data["labelset"] = LabelSet([*labelset.elements, LabeledElement(md5="later", label="good")]).to_dict()
    _write_detector(path, data)


def _new_session() -> None:
    """Another day on the same ranking: no test in memory, no session votes."""
    ctx = get_active_detector_context()
    ctx.line_test = None
    ctx.good_votes.clear()
    ctx.bad_votes.clear()
    ctx.verified_ids.clear()
    ctx.vote_provenance.clear()


def _clean_line(cid: int) -> bool:
    return cid <= 64


class TestTheVerdictIsKept:
    """#4526: the vote that finishes a test keeps its verdict on the detector, and the readers and a later test read it."""

    def test_the_vote_that_finishes_a_test_keeps_its_verdict(self, client):
        detector_id = _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        picks = _start(client)["test"]["picks"]
        data = client.post("/api/line-test/votes", json=_votes(picks, True)).get_json()
        assert _kept() == [], "nothing is kept before Done"

        data = _vote_until_done(client, data, _clean_line)
        test = data["test"]
        assert test["phase"] == PHASE_DONE and test["kept_at"] is None
        (verdict,) = _kept()
        assert verdict.dataset_id and verdict.dataset_id == ctx.votes_dataset_id
        assert (verdict.line_count, verdict.size, verdict.labelled) == (64, 200, test["labelled"])
        assert verdict.precision.as_dict() == test["estimates"]["precision"]
        assert verdict.fbeta.as_dict() == test["estimates"]["fbeta"]
        assert verdict.labels_digest is not None
        # The rest of the file is as it was, and the Find session survived the write.
        assert _detector_file()[1]["labelset"]["labels"]
        assert ctx.find_mode and ctx.line_test is not None

        stats = client.get(f"/api/detectors/registry/{detector_id}/stats").get_json()
        (row,) = stats["test_verdicts"]
        assert row["dataset_id"] == verdict.dataset_id and row["stale"] is False
        assert row["labelled"] == test["labelled"] and row["found"] == test["estimates"]["found"]
        assert row["precision"] == test["estimates"]["precision"]

    def test_a_newer_test_of_the_same_dataset_replaces_the_older(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        _vote_until_done(client, _start(client), _clean_line)
        ctx.threshold = 1.0 - 100 / 201
        _vote_until_done(client, _start(client), lambda cid: cid <= 100)
        (verdict,) = _kept()
        assert verdict.line_count == 100

    def test_the_autorun_tab_reads_the_latest_verdict_of_an_autorun_detector(self, client):
        detector_id = _run_find(client)
        _plant_big_corpus()

        def entry() -> dict:
            rows = client.get("/api/detectors/registry").get_json()["detectors"]
            return next(d for d in rows if d["id"] == detector_id)

        resp = client.put(f"/api/detectors/registry/{detector_id}/autofind", json={"autofind": True})
        assert resp.status_code == 200, resp.get_json()
        assert entry()["test_verdict"] is None, "an AutoRun detector never tested"
        _vote_until_done(client, _start(client), _clean_line)
        row = entry()["test_verdict"]
        assert row["dataset_id"] == _kept()[0].dataset_id and row["stale"] is False
        client.put(f"/api/detectors/registry/{detector_id}/autofind", json={"autofind": False})
        assert "test_verdict" not in entry(), "a draft reads no detector file for it"

    def test_a_retrain_marks_the_verdict_stale_and_it_stays(self, client):
        detector_id = _run_find(client)
        _plant_big_corpus()
        _vote_until_done(client, _start(client), _clean_line)
        _retrain_on_disk()
        (row,) = client.get(f"/api/detectors/registry/{detector_id}/stats").get_json()["test_verdicts"]
        assert row["stale"] is True
        assert len(_kept()) == 1

    def test_corrections_before_done_leave_the_verdict_stale_from_the_start(self, client):
        detector_id = _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        data = _start(client)
        ctx.find_eval_stale = True
        _vote_until_done(client, data, _clean_line)
        (verdict,) = _kept()
        assert verdict.labels_digest is None
        assert client.get(f"/api/detectors/registry/{detector_id}/stats").get_json()["test_verdicts"][0]["stale"]


class TestResumingFromTheKeptPicks:
    def test_a_later_test_of_the_same_ranking_resumes_from_the_kept_picks(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        done = _vote_until_done(client, _start(client), _clean_line)["test"]
        (verdict,) = _kept()

        _new_session()
        data = _start(client)
        test = data["test"]
        assert test["phase"] == PHASE_DONE and test["picks"] == []
        assert test["kept_at"] == pytest.approx(verdict.tested_at)
        assert test["labelled"] == done["labelled"]
        assert test["estimates"]["precision"] == done["estimates"]["precision"]
        # The kept picks are session votes again, in the Review tab's piles.
        for cid, right, _ in verdict.picks:
            assert cid in ctx.verified_ids and (cid in ctx.good_votes) == right
            assert ctx.vote_provenance[cid]["flow"] == "test"
        # Nothing new to keep: the verdict on disk is the one it resumed from.
        assert [v.to_dict() for v in _kept()] == [verdict.to_dict()]

    def test_a_pick_already_verified_this_session_keeps_the_session_vote(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        _vote_until_done(client, _start(client), _clean_line)
        cid, right, _ = _kept()[0].picks[0]

        _new_session()
        ctx.verified_ids[cid] = None
        (ctx.bad_votes if right else ctx.good_votes)[cid] = None
        _start(client)
        assert (cid in ctx.good_votes) != right

    def test_another_line_starts_afresh(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        _vote_until_done(client, _start(client), _clean_line)
        _new_session()
        ctx.threshold = 1.0 - 100 / 201
        test = _start(client)["test"]
        assert test["labelled"] == 0 and test["kept_at"] is None and test["phase"] == PHASE_MATCHES

    def test_a_stale_verdict_is_not_resumed(self, client):
        _run_find(client)
        _plant_big_corpus()
        _vote_until_done(client, _start(client), _clean_line)
        _retrain_on_disk()
        _new_session()
        test = _start(client)["test"]
        assert test["labelled"] == 0 and test["kept_at"] is None

    def test_forget_drops_the_kept_verdict_so_the_next_start_tests_afresh(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        _vote_until_done(client, _start(client), _clean_line)
        _new_session()
        assert _start(client)["test"]["kept_at"] is not None
        resumed = dict(ctx.good_votes)

        data = client.post("/api/line-test/forget", json={}).get_json()
        assert data["test"] is None and ctx.line_test is None and _kept() == []
        # The resumed picks stay session votes, as a cancelled test's do.
        assert dict(ctx.good_votes) == resumed
        test = _start(client)["test"]
        assert test["labelled"] == 0 and test["kept_at"] is None and test["phase"] == PHASE_MATCHES

    def test_forget_leaves_a_running_test_and_is_fine_with_nothing_kept(self, client):
        _run_find(client)
        _plant_big_corpus()
        ctx = get_active_detector_context()
        picks = _start(client)["test"]["picks"]
        data = client.post("/api/line-test/forget", json={}).get_json()
        assert data["test"]["picks"] == picks and ctx.line_test is not None
