"""Issue #4664's early check: Autopilot checks every session once, from a fixed vote count on.

An experiment knob, not app behaviour: off by default, so the default arm is the
app's own run pick for pick.  The pick log's ``check_kind`` says which check a
check pick belongs to.
"""

from __future__ import annotations

from typing import Any

import pytest

OPENING = ("good", "bad", "more")


def _run(early_check: Any = None, *, max_steps: int = 60, **kw: Any) -> list[dict[str, Any]]:
    from tests_lib.detectors.test_startup_schedule import _seeded_dataset
    from vtscore.eval.voting_iterations import simulate_voting_iterations

    medias, seed_scores = _seeded_dataset(n_pos=40)
    picks: list[dict[str, Any]] = []
    simulate_voting_iterations(
        medias,
        target_category="target",
        seed=3,
        dataset_name="stub",
        max_steps=max_steps,
        seed_scores=seed_scores,
        atlas_min_node_size=8,
        early_check=early_check,
        pick_sink=picks,
        **kw,
    )
    return picks


def _kind_ts(picks, kind):
    """The vote counts a check's picks are logged at (a round's picks all carry its last vote)."""
    return [p["t"] for p in picks if p["check_kind"] == kind]


def _first(picks, kind):
    return next(i for i, p in enumerate(picks) if p["check_kind"] == kind)


# Labels that never separate weakly, so only the checks under test run mid-session.
_NEVER_WEAK: dict[str, Any] = {"weak_separation": -1e9}
# Labels that always do, so the weak prompt fires at its first chance, once.
_ALWAYS_WEAK: dict[str, Any] = {"weak_separation": 1e9, "weak_min_t": 1, "weak_repeat": 0}
_STEP_OVER_PARTS = 5


@pytest.fixture(scope="module")
def always_weak_ts() -> list[int]:
    """The vote counts the always-weak run's weak check is logged at."""
    return _kind_ts(_run(**_ALWAYS_WEAK), "weak")


class TestEarlyCheck:
    def test_off_by_default(self):
        assert not _kind_ts(_run(), "early")
        assert not _kind_ts(_run(**_NEVER_WEAK), "early")

    def test_checks_once_from_the_vote_count_past_the_opening(self):
        picks = _run(20, **_NEVER_WEAK)
        assert _kind_ts(picks, "early"), "the early check never ran"
        i = _first(picks, "early")
        n = len(_kind_ts(picks, "early"))
        # One run of consecutive prompted clicks, starting on the step from the 20th vote.
        assert all(p["check_kind"] == "early" and p["phase"] == "prompt" for p in picks[i : i + n])
        assert picks[i - 1]["t"] == 20
        assert picks[i - 1]["phase"] not in OPENING
        # The session goes on after it.
        assert any(p["phase"] not in ("prompt", "check") for p in picks[i + n :])

    def test_waits_for_the_opening(self):
        """Due inside the opening, it starts on the first step past it (a pick logs the phase it was picked in)."""
        plain, picks = _run(**_NEVER_WEAK), _run(2, **_NEVER_WEAK)
        i = _first(picks, "early")
        assert i > 2 and all(p["phase"] in OPENING for p in picks[:i])
        assert plain[i]["phase"] not in OPENING

    def test_the_run_before_it_is_the_apps(self):
        plain, early = _run(**_NEVER_WEAK), _run(20, **_NEVER_WEAK)
        i = _first(early, "early")
        assert [p["picked_id"] for p in plain[:i]] == [p["picked_id"] for p in early[:i]]

    def test_a_running_check_is_the_sessions_check(self):
        """At N votes mid-check, the running check counts: no second check follows it."""
        weak = _kind_ts(_run(**_ALWAYS_WEAK), "weak")
        assert weak, "the always-weak run never prompted"
        plain, picks = _run(**_ALWAYS_WEAK), _run(weak[0], **_ALWAYS_WEAK)
        assert not _kind_ts(picks, "early")
        assert [p["picked_id"] for p in plain] == [p["picked_id"] for p in picks]

    # Every N from four votes before the weak check to its last vote is a whole
    # session, ~20 of them: 312 s as one test on a Xeon Silver 4114 under the
    # suite, past the 300 s per-test timeout (#4925).  Dealt over parts, which
    # xdist runs side by side.
    @pytest.mark.parametrize("part", range(_STEP_OVER_PARTS))
    def test_a_round_that_steps_over_the_vote_count_is_the_sessions_check(self, always_weak_ts, part):
        """A check round casts its picks at once: one that reaches N votes counts, though no step starts at N."""
        weak = always_weak_ts
        for n in range(weak[0] - 4 + part, weak[-1] + 1, _STEP_OVER_PARTS):
            assert not _kind_ts(_run(n, **_ALWAYS_WEAK), "early"), n

    def test_no_cooldown_after_a_check_that_ended(self):
        """A check that ended just before N votes does not stand in for it (the cooldown variant is read offline)."""
        weak = _kind_ts(_run(**_ALWAYS_WEAK), "weak")
        picks = _run(weak[-1] + 1, **_ALWAYS_WEAK)
        assert _kind_ts(picks, "early")
        assert picks[_first(picks, "early") - 1]["t"] == weak[-1] + 1

    def test_not_started_past_the_budget(self):
        """A check whose opening bands do not fit the clicks left is skipped, as a prompted one is."""
        assert not _kind_ts(_run(58, **_NEVER_WEAK), "early")

    def test_the_kinds_of_check_picks(self):
        for picks in (_run(20, **_NEVER_WEAK), _run(**_ALWAYS_WEAK)):
            assert {p["check_kind"] for p in picks} <= {"", "weak", "done", "early", "end"}
            assert all(p["check_kind"] == "end" for p in picks if p["phase"] == "check")
            assert all(p["check_kind"] == "" for p in picks if p["phase"] not in ("check", "prompt"))
            assert all(p["check_kind"] in ("weak", "done", "early") for p in picks if p["phase"] == "prompt")
        assert _kind_ts(_run(**_ALWAYS_WEAK), "weak")

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "50", True])
    def test_refuses_what_it_cannot_run(self, bad):
        with pytest.raises(ValueError):
            _run(bad)

    @pytest.mark.parametrize("kw", [{"beta": "off"}, {"spot_check": "off"}])
    def test_needs_a_balance_and_a_run_that_checks(self, kw):
        with pytest.raises(ValueError):
            _run(30, **kw)


def test_the_flow_takes_an_early_check():
    from vtscore.eval.autopilot_flow import AutopilotFlow

    flow = AutopilotFlow()
    flow.begin_spot_check("early")
    assert flow.spot_check == "early"
    flow.end_spot_check()
    # Unlike the check before Done, it latches nothing.
    assert flow.spot_check is None and not flow.done_checked
    with pytest.raises(ValueError):
        flow.begin_spot_check("whenever")
