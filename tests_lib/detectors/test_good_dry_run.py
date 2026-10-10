"""Issue #4731: the Good walk runs dry, and the quota's second tier gives what it leaves a trained head.

Autopilot's Good phase used to end only at its third Good, so a target with fewer than three Goods
the sort can reach walked the seed sort for the whole session and never showed a detector. Now the
phase also ends after ``moreDryRun`` (16) picks in a row without a Good, with a Good in hand, and a
labelset of a Good and 16 Bads gets the trained head. The harness ports both and runs them by
default; ``"off"`` is the pre-#4731 arm.
"""

from __future__ import annotations

import pytest

from vtscore.detectors.label_quota import DRY_BAD_QUOTA
from vtscore.eval.autopilot_flow import BAD_TARGET, GOOD_TARGET, MORE_DRY_RUN, AutopilotFlow, next_phase
from vtscore.eval.voting_iterations import _quota_centroid


def _walk(flow: AutopilotFlow, outcomes: str, good: int = 0, bad: int = 0) -> list[str]:
    """Cast *outcomes* (``g`` a Good, ``b`` a Bad) one vote at a time; the phase after each."""
    phases = []
    for o in outcomes:
        good, bad = (good + 1, bad) if o == "g" else (good, bad + 1)
        phases.append(flow.update(good, bad, 500, None))
    return phases


class TestNextPhase:
    @staticmethod
    def _phase(good: int, bad: int, **kw) -> str:
        return next_phase(good, bad, remaining_unlabeled=500, smart="red", stable="red", span="red", **kw)

    def test_a_dry_walk_meets_the_good_target_with_what_it_found(self):
        assert self._phase(1, 20) == "good"
        assert self._phase(1, 20, good_ran_dry=True, more_done=True) == "hard"

    def test_it_never_hands_over_without_a_good(self):
        assert self._phase(0, 20, good_ran_dry=True, more_done=True) == "good"

    def test_the_bad_target_still_holds(self):
        assert self._phase(2, 1, good_ran_dry=True, more_done=True) == "bad"


class TestFlow:
    def test_the_app_runs_dry_at_more_dry_run(self):
        assert AutopilotFlow().good_dry_run == MORE_DRY_RUN == 16

    def test_runs_dry_after_its_run_of_misses(self):
        flow = AutopilotFlow()
        phases = _walk(flow, "g" + "b" * MORE_DRY_RUN)
        assert phases[:MORE_DRY_RUN] == ["good"] * MORE_DRY_RUN
        assert phases[MORE_DRY_RUN] == "hard"
        assert flow.good_dry and flow.more_done

    def test_the_pre_4731_arm_waits_for_three_goods(self):
        assert set(_walk(AutopilotFlow(good_dry_run=None), "g" + "b" * 100)) == {"good"}

    def test_never_hands_over_without_a_good(self):
        flow = AutopilotFlow(good_dry_run=8)
        assert set(_walk(flow, "b" * 40)) == {"good"}
        assert not flow.good_dry

    def test_the_more_walk_is_spent(self):
        """The seed sort ran dry in the Good phase; the walk down it is not taken again."""
        assert "more" not in _walk(AutopilotFlow(good_dry_run=8), "g" + "b" * 8 + "g" * 3)

    def test_misses_before_the_first_good_do_not_count(self):
        flow = AutopilotFlow(good_dry_run=8)
        assert set(_walk(flow, "b" * 30 + "g" + "b" * 7)) == {"good"}
        assert _walk(flow, "b", good=1, bad=37) == ["hard"]

    def test_a_good_restarts_the_run(self):
        flow = AutopilotFlow(good_dry_run=8)
        assert set(_walk(flow, "g" + "b" * 7 + "g" + "b" * 7)) == {"good"}
        assert not flow.good_dry

    def test_a_short_run_still_waits_for_the_bad_target(self):
        flow = AutopilotFlow(good_dry_run=2)
        assert _walk(flow, "gbb") == ["good", "good", "bad"]

    def test_the_third_good_ends_it_as_before(self):
        assert _walk(AutopilotFlow(), "g" * GOOD_TARGET + "b" * BAD_TARGET)[-1] == "more"

    def test_a_startup_schedule_owns_the_opening(self):
        from vtscore.eval.startup_schedule import StartupState, parse_startup_schedule

        flow = AutopilotFlow(startup=StartupState(parse_startup_schedule("g3@top,b4@mid")))
        assert flow.good_dry_run is None

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5, "8"])
    def test_refuses_a_value_that_is_not_a_run_length(self, bad):
        with pytest.raises(ValueError, match="good_dry_run"):
            AutopilotFlow(good_dry_run=bad)


class TestQuotaTier:
    def test_the_apps_second_quota(self):
        assert DRY_BAD_QUOTA == 16
        assert _quota_centroid(1, 15, DRY_BAD_QUOTA)
        assert not _quota_centroid(1, 16, DRY_BAD_QUOTA)
        assert not _quota_centroid(2, 40, DRY_BAD_QUOTA)

    def test_the_pre_4731_quota(self):
        assert _quota_centroid(1, 100, None)
        assert _quota_centroid(2, 4, None)
        assert not _quota_centroid(GOOD_TARGET, BAD_TARGET, None)

    def test_no_good_is_left_as_the_quota_has_it(self):
        """No Good is the quota's ``none`` tier, which the second quota does not touch."""
        assert _quota_centroid(0, 100, DRY_BAD_QUOTA) == _quota_centroid(0, 100, None)


class TestInTheHarness:
    """One example and one withheld photo: nothing more to find in the voting half."""

    def _run(self, **kw):
        from tests_lib.detectors.test_example_opening import _medias
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _medias(n_pos=2),
            target_category="person",
            seed=3,
            dataset_name="stub",
            max_steps=24,
            atlas_min_node_size=8,
            spot_check="off",
            seed_examples=1,
            stratify_target=True,
            pick_sink=picks,
            **kw,
        )
        return rows, picks

    def test_the_pre_4731_arm_never_leaves_the_good_phase(self):
        rows, picks = self._run(good_dry_run="off", quota_dry_bads="off")
        assert {p["phase"] for p in picks[1:]} == {"good"}
        assert not any(r["app_trained"] for r in rows)

    def test_the_app_hands_over_a_trained_head_and_agrees_until_then(self):
        _, before = self._run(good_dry_run="off", quota_dry_bads="off")
        rows, app = self._run()
        # The example, then 16 misses, then the learned sort.
        assert [p["phase"] for p in app[:17]] == ["example"] + ["good"] * 16
        assert [p["picked_id"] for p in app[:17]] == [p["picked_id"] for p in before[:17]]
        assert app[17]["phase"] == "hard"
        shown = [r for r in rows if r["app_trained"]]
        assert shown and min(r["t"] for r in shown) == 17
        assert {r["detector_tier"] for r in shown} == {"trained"}

    def test_without_the_second_quota_the_detector_shown_is_the_centroid(self):
        rows, _ = self._run(quota_dry_bads="off")
        shown = [r for r in rows if r["app_trained"]]
        assert shown and {r["detector_tier"] for r in shown} == {"centroid"}

    def test_an_arm_sets_another_length(self):
        _, picks = self._run(good_dry_run=8)
        assert picks[9]["phase"] == "hard"

    @pytest.mark.parametrize(
        "kw",
        [
            {"good_dry_run": 0},
            {"good_dry_run": "on"},
            {"quota_dry_bads": 0},
            {"good_dry_run": 8, "startup_schedule": "g3@top,b4@mid"},
        ],
    )
    def test_refuses_what_it_cannot_run(self, kw):
        with pytest.raises(ValueError):
            self._run(**kw)
