"""Issue #4731's arms: the Good phase runs dry, and the quota tier that leaves a trained head.

Autopilot's Good phase ends at its third Good. A target with fewer than three Goods the sort can
reach walks the seed sort for the whole session and never shows a detector. ``good_dry_run`` also
ends it after a run of misses, once a Good is in hand; ``quota_dry_bads`` gives the trained head to
the labelset that leaves. Both are experiment knobs, off by default, so the default arm is the app's
own session pick for pick.
"""

from __future__ import annotations

import pytest

from vtscore.eval.autopilot_flow import BAD_TARGET, GOOD_TARGET, AutopilotFlow, next_phase
from vtscore.eval.voting_iterations import _quota_centroid


def _walk(flow: AutopilotFlow, outcomes: str, good: int = 0, bad: int = 0) -> list[str]:
    """Cast *outcomes* (``g`` a Good, ``b`` a Bad) one vote at a time; the phase after each."""
    phases = []
    for o in outcomes:
        good, bad = (good + 1, bad) if o == "g" else (good, bad + 1)
        phases.append(flow.update(good, bad, 500, None))
    return phases


class TestNextPhase:
    def test_the_ported_machine_is_the_apps(self):
        """The arm lives in the flow, which lowers the Good target it passes; the port is untouched."""
        import inspect

        assert "good_dry" not in inspect.getsource(next_phase)
        phase = next_phase(
            1, 20, remaining_unlabeled=500, smart="red", stable="red", span="red", good_target=1, more_done=True
        )
        assert phase == "hard"


class TestFlow:
    def test_off_by_default(self):
        """The app: one Good and a hundred misses is still the Good phase."""
        assert set(_walk(AutopilotFlow(), "g" + "b" * 100)) == {"good"}

    def test_never_hands_over_without_a_good(self):
        flow = AutopilotFlow(good_dry_run=8)
        assert set(_walk(flow, "b" * 40)) == {"good"}
        assert not flow.good_dry

    def test_runs_dry_after_its_run_of_misses(self):
        flow = AutopilotFlow(good_dry_run=8)
        phases = _walk(flow, "g" + "b" * 8)
        assert phases[:8] == ["good"] * 8
        assert phases[8] == "hard"
        assert flow.good_dry and flow.more_done

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

    def test_the_third_good_ends_it_as_in_the_app(self):
        assert _walk(AutopilotFlow(good_dry_run=8), "g" * GOOD_TARGET + "b" * BAD_TARGET)[-1] == "more"

    @pytest.mark.parametrize("bad", [0, -1, True, 2.5, "8"])
    def test_refuses_a_value_that_is_not_a_run_length(self, bad):
        with pytest.raises(ValueError, match="good_dry_run"):
            AutopilotFlow(good_dry_run=bad)


class TestQuotaTier:
    def test_the_apps_quota_when_off(self):
        assert _quota_centroid(1, 100, None)
        assert _quota_centroid(2, 4, None)
        assert not _quota_centroid(GOOD_TARGET, BAD_TARGET, None)

    def test_a_good_and_enough_bads_get_the_trained_head(self):
        assert _quota_centroid(1, 15, 16)
        assert not _quota_centroid(1, 16, 16)
        assert not _quota_centroid(2, 40, 16)

    def test_no_good_is_left_as_the_quota_has_it(self):
        """No Good is the quota's ``none`` tier, which the dry tier does not touch."""
        assert _quota_centroid(0, 100, 16) == _quota_centroid(0, 100, None)

    def test_the_full_quota_still_trains(self):
        assert not _quota_centroid(GOOD_TARGET, BAD_TARGET, 16)


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
            max_steps=20,
            atlas_min_node_size=8,
            spot_check="off",
            seed_examples=1,
            stratify_target=True,
            pick_sink=picks,
            **kw,
        )
        return rows, picks

    def test_today_never_leaves_the_good_phase(self):
        rows, picks = self._run()
        assert {p["phase"] for p in picks[1:]} == {"good"}
        assert not any(r["app_trained"] for r in rows)

    def test_the_dry_run_hands_over_and_agrees_until_then(self):
        _, ctl = self._run()
        rows, dry = self._run(good_dry_run=8)
        # The example, then 8 misses, then the learned sort.
        assert [p["phase"] for p in dry[:9]] == ["example"] + ["good"] * 8
        assert [p["picked_id"] for p in dry[:9]] == [p["picked_id"] for p in ctl[:9]]
        assert dry[9]["phase"] == "hard"
        shown = [r for r in rows if r["app_trained"]]
        assert shown and min(r["t"] for r in shown) == 9
        # The quota is the app's: below three Goods, Test gives the centroid.
        assert {r["detector_tier"] for r in shown} == {"centroid"}

    def test_the_quota_tier_shows_a_trained_head(self):
        rows, _ = self._run(good_dry_run=8, quota_dry_bads=8)
        shown = [r for r in rows if r["app_trained"]]
        assert shown and {r["detector_tier"] for r in shown} == {"trained"}

    @pytest.mark.parametrize(
        "kw",
        [
            {"good_dry_run": 0},
            {"quota_dry_bads": 0},
            {"good_dry_run": 8, "startup_schedule": "g3@top,b4@mid"},
        ],
    )
    def test_refuses_what_it_cannot_run(self, kw):
        with pytest.raises(ValueError):
            self._run(**kw)
