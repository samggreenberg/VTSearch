"""Smart and Stable read the dry run past the opening (#5477).

Both lights need 5 Goods, and most faces have 3 or 4 in reach, so Autopilot
never told those sessions Done.  Once 16 votes in a row past the opening held no
Good, the target has shown what it can supply and the lights need only the Goods
in hand.  The labeling status reads the run off the label history and each
vote's provenance: Autopilot's Boundary and Diversity picks and spot-check votes
count, the opening and votes cast by hand do not.  These run the real vote and
status routes.
"""

from __future__ import annotations

import pytest

from vtscore.detectors.labeling_progress import hard_ran_dry
from vtscore.detectors.stability import HARD_DRY_RUN, minimum_reason

HARD = {"flow": "autopilot", "phase": "hard"}
NEW = {"flow": "autopilot", "phase": "new"}
CHECK = {"flow": "check"}
OPENING_BAD = {"flow": "autopilot", "phase": "bad"}
BY_HAND = {"flow": "list_review"}


def _history(labels: list[str]) -> list[tuple[int, str, float]]:
    return [(i, label, float(i)) for i, label in enumerate(labels)]


class TestReadOffTheHistory:
    def test_sixteen_bads_past_the_opening_run_dry(self):
        history = _history(["good"] * 3 + ["bad"] * HARD_DRY_RUN)
        provenance = {i: HARD for i in range(3, 3 + HARD_DRY_RUN)}
        assert hard_ran_dry(history, provenance)
        assert not hard_ran_dry(history[:-1], provenance)

    def test_diversity_and_spot_check_votes_count(self):
        history = _history(["bad"] * HARD_DRY_RUN)
        provenance = {i: (NEW if i % 2 else CHECK) for i in range(HARD_DRY_RUN)}
        assert hard_ran_dry(history, provenance)

    @pytest.mark.parametrize("other", [OPENING_BAD, BY_HAND, None])
    def test_the_opening_and_votes_by_hand_do_not_count(self, other):
        history = _history(["bad"] * HARD_DRY_RUN)
        provenance = {i: HARD for i in range(HARD_DRY_RUN)}
        if other is None:
            del provenance[5]
        else:
            provenance[5] = other
        assert not hard_ran_dry(history, provenance)

    def test_a_good_resets_the_run(self):
        labels = ["bad"] * 10 + ["good"] + ["bad"] * (HARD_DRY_RUN - 1)
        provenance = dict.fromkeys(range(len(labels)), HARD)
        assert not hard_ran_dry(_history(labels), provenance)
        assert hard_ran_dry(_history(labels + ["bad"]), dict.fromkeys(range(len(labels) + 1), HARD))

    def test_it_latches(self):
        labels = ["bad"] * HARD_DRY_RUN + ["good"]
        assert hard_ran_dry(_history(labels), dict.fromkeys(range(len(labels)), HARD))

    def test_no_provenance_never_runs_dry(self):
        assert not hard_ran_dry(_history(["bad"] * 40), None)
        assert not hard_ran_dry(_history(["bad"] * 40), {})


GOODS = [1, 2, 3]

#: The run the route tests shrink the dry run to: the fixture dataset has 20 items, too few for 16 Boundary Bads.
RUN = 4


@pytest.fixture
def short_run(monkeypatch):
    import functools

    import vtscore.detectors.labeling_progress as lp

    monkeypatch.setattr(lp, "HardDryRun", functools.partial(lp.HardDryRun, target=RUN))


def _vote(client, cid: int, target: str, provenance: dict) -> None:
    resp = client.post(f"/api/medias/{cid}/vote", json={"target": target, "provenance": provenance})
    assert resp.status_code == 200, resp.get_json()


def _status(client) -> dict:
    """The status computed inline over the whole history, the path a caught-up poll takes."""
    from vtscore.detectors.labeling_progress import compute_labeling_status
    from vtscore.state.core import get_active_detector_context
    from vtsearch.state import bad_votes, good_votes, label_history, snapshot_medias

    return compute_labeling_status(
        snapshot_medias(),
        label_history,
        good_votes,
        bad_votes,
        vote_provenance=dict(get_active_detector_context().vote_provenance),
    )


def _session(client, misses: int) -> list[int]:
    """Three Goods, five opening Bads, then *misses* Boundary Bads; the ids voted Bad."""
    from vtsearch.state import snapshot_medias

    ids = sorted(snapshot_medias())
    others = [i for i in ids if i not in GOODS]
    assert len(others) >= 5 + misses, f"the fixture dataset has {len(ids)} items"
    for cid in GOODS:
        _vote(client, cid, "good", {"flow": "autopilot", "phase": "good"})
    for cid in others[:5]:
        _vote(client, cid, "bad", OPENING_BAD)
    for cid in others[5 : 5 + misses]:
        _vote(client, cid, "bad", HARD)
    return others[: 5 + misses]


class TestTheLights:
    def test_three_goods_hold_the_lights_red_until_the_run(self, client, short_run):
        bads = _session(client, RUN - 1)
        status = _status(client)
        reason = minimum_reason(3, len(bads))
        assert status["smart"]["reason"] == reason and status["stable"]["reason"] == reason

    def test_the_run_lets_the_lights_read_three_goods(self, client, short_run):
        bads = _session(client, RUN)
        status = _status(client)
        for light in ("smart", "stable"):
            assert status[light]["reason"] != minimum_reason(3, len(bads))
            assert not status[light]["reason"].startswith("Need at least 5 good")

    def test_the_status_route_reads_the_provenance(self, client, monkeypatch):
        """The poll hands the lights the detector's vote provenance, inline and in the background refresh."""
        import vtscore.detectors.labeling_progress as lp

        seen = []
        real = lp.hard_ran_dry

        def spy(history, provenance):
            seen.append(provenance)
            return real(history, provenance)

        monkeypatch.setattr(lp, "hard_ran_dry", spy)
        _session(client, 2)
        for _ in range(3):
            assert client.get("/api/labeling-status").status_code == 200
        from vtscore.concurrency.async_jobs import labeling_status_jobs

        for job in labeling_status_jobs.active_jobs():
            job.done_event.wait(timeout=30)
        assert seen and any(p and p.get(GOODS[0], {}).get("phase") == "good" for p in seen)
