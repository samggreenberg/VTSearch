"""The eval harness's default arm draws the line a live detector draws (#4245, #4272).

The app's reporting line under a precision floor keeps a set - the floor's
unchecked starting candidate, or the set its spot check ended on - and the
default user has a floor, so the harness's default arm has to do the same, or
every study measures a detector nobody ships.  The rule itself is shared
(``floor_line`` / ``floor_state`` / ``SpotCheck``); what these tests pin is the
harness's side of the plumbing: that the run's spot check draws its candidate
off the ranking, answers its picks from ground truth, casts them as votes and
retrains on them, and that the arms the knob names behave as named.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.eval.autopilot_flow import pick_provenance
from vtscore.eval.voting_iterations import simulate_voting_iterations
from vtscore.training.thresholds import (
    BALANCE_CHECKED,
    BALANCE_STATES,
    BAND_BASE,
    DEFAULT_BETA,
    DEFAULT_MIN_PRECISION,
    FLOOR_CONFIRMED,
    FLOOR_SHORT,
    FLOOR_STATES,
    FLOOR_UNCHECKED,
    NO_PRECISION_FLOOR,
    check_schedule,
)


def _separable(n_per_cat: int = 20, dim: int = 16) -> dict[int, dict]:
    rng = np.random.RandomState(0)
    medias: dict[int, dict] = {}
    for i in range(2 * n_per_cat):
        cat = "alpha" if i < n_per_cat else "beta"
        centre = 1.0 if cat == "alpha" else -1.0
        emb = rng.normal(centre, 0.6, dim).astype(np.float32)
        medias[i + 1] = {"id": i + 1, "embeddings": {"emb": emb}, "category": cat}
    return medias


class TestTheRetiredProvenanceRecord:
    """``pick_provenance`` fed the #4245 evidence filter, which retired with the estimate (#4362)."""

    @pytest.mark.parametrize(
        ("phase", "sort_kind", "select_mode"),
        [("good", "text", "top"), ("bad", "text", "hard"), ("hard", "learned", "hard"), ("new", "learned", "new")],
    )
    def test_it_still_answers_from_its_table_with_a_warning(self, phase, sort_kind, select_mode):
        with pytest.deprecated_call(match="pick_provenance"):
            recorded = pick_provenance(phase)
        assert recorded == {"flow": "autopilot", "phase": phase, "select_mode": select_mode, "sort_kind": sort_kind}

    @pytest.mark.parametrize("phase", [None, "idle", "done", "exhausted"])
    def test_no_labelling_phase_records_nothing(self, phase):
        with pytest.deprecated_call():
            assert pick_provenance(phase) is None


class TestTheArms:
    """The floor arm (``min_precision`` pinned; deprecated with the floor, #4413), and what the default arm is."""

    def _run(self, **kwargs):
        kwargs.setdefault("min_precision", DEFAULT_MIN_PRECISION)
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(), "alpha", seed=0, max_steps=15, calibrate_count=2, pick_sink=picks, **kwargs
        )
        return rows, picks

    def test_the_default_arm_is_the_apps_balance(self):
        """Neither knob pinned draws the line where the app does: the balance at DEFAULT_BETA (#4413)."""
        rows, _picks = self._run(min_precision=None)
        assert rows
        assert all(r["beta"] == DEFAULT_BETA and math.isnan(r["min_precision"]) for r in rows)
        assert {r["floor_status"] for r in rows} <= set(BALANCE_STATES)

    def test_a_pinned_floor_is_the_floor_arm(self):
        rows, _picks = self._run()
        assert rows
        assert all(r["min_precision"] == DEFAULT_MIN_PRECISION and math.isnan(r["beta"]) for r in rows)
        assert {r["floor_status"] for r in rows} <= set(FLOOR_STATES)

    def test_before_the_check_the_line_keeps_the_unchecked_starting_candidate(self, schedule_only):
        """What a headless run exports: the top K unvoted, K from the floor's schedule, capped by the corpus."""
        rows, _picks = self._run()
        steps = [r for r in rows if r["phase"] != "check"]
        assert steps and all(r["floor_status"] == FLOOR_UNCHECKED for r in steps)
        k = check_schedule(DEFAULT_MIN_PRECISION).candidate
        # `n_remainder` is the unvoted sim set after this step's vote: the candidate.
        assert all(r["floor_count"] == min(k, r["n_remainder"]) for r in steps)
        assert all(math.isnan(r["range_lo"]) and r["check_labelled"] == -1 and r["check_stale"] == -1 for r in steps)

    def test_before_the_check_the_mixture_can_only_lower_the_count(self):
        """The app's unchecked line: the smaller of the schedule's count and the vote-anchored mixture's (#4389).

        The harness hands the votes so far to the same ``mixture_count`` the
        app anchors on, so a headless study exports what AutoRun would.
        """
        rows, _picks = self._run()
        steps = [r for r in rows if r["phase"] != "check"]
        k = check_schedule(DEFAULT_MIN_PRECISION).candidate
        assert steps and all(1 <= r["floor_count"] <= min(k, r["n_remainder"]) for r in steps)
        assert any(r["floor_count"] < min(k, r["n_remainder"]) for r in steps), "the mixture lowered some line"

    def test_the_check_runs_once_the_steps_are_spent_and_its_votes_enter_training(self):
        rows, picks = self._run()
        steps = [r for r in rows if r["phase"] != "check"]
        check_rows = [r for r in rows if r["phase"] == "check"]
        assert check_rows, "the default arm checks the line"
        assert max(r["t"] for r in steps) == 15
        # One row per round, each after that round's picks were cast and the model retrained.
        assert all(r["t"] > 15 for r in check_rows)
        assert all(r["n_good"] + r["n_bad"] == r["t"] for r in check_rows)
        # The check's picks are logged as votes with the app's provenance flow.
        check_picks = [p for p in picks if p["phase"] == "check"]
        assert len(check_picks) == check_rows[-1]["t"] - 15
        assert all(p["t"] > 15 for p in check_picks)
        # The last row carries the result: the set the check ended on, and its range.
        last = check_rows[-1]
        assert last["floor_status"] in (FLOOR_CONFIRMED, FLOOR_SHORT)
        assert 0.0 <= last["range_lo"] <= last["range_hi"] <= 1.0
        assert last["check_labelled"] >= 1 and 0 <= last["check_right"] <= last["check_labelled"]
        assert last["check_stale"] in (0, 1)
        # The walk decides on the band-weighted share of its picks, not on the
        # range's lower end (#4388): a confirmed set's range can straddle the
        # floor, and a short one keeps the first band whatever its range says.
        assert last["floor_count"] >= 1
        if last["floor_status"] == FLOOR_SHORT:
            assert last["floor_count"] <= BAND_BASE

    def test_switching_the_check_off_leaves_the_run_unchecked(self):
        rows, picks = self._run(spot_check="off")
        assert all(r["phase"] != "check" for r in rows)
        assert all(r["floor_status"] == FLOOR_UNCHECKED for r in rows)
        assert all(p["phase"] != "check" for p in picks)

    def test_the_voting_steps_are_byte_identical_with_or_without_the_check(self):
        """The check is seeded after every trajectory draw: switching it on changes nothing before it."""
        with_check, _ = self._run()
        without, _ = self._run(spot_check="off")
        steps = [r for r in with_check if r["phase"] != "check"]
        assert [r["t"] for r in steps] == [r["t"] for r in without]
        # The plain frame carries no raw threshold; these all read it.
        for col in ("report_pool_percentile", "cost", "n_flagged", "acq_threshold", "floor_count"):
            assert [r[col] for r in steps] == [r[col] for r in without], col

    def test_an_unknown_check_knob_is_refused(self):
        with pytest.raises(ValueError, match="spot_check"):
            self._run(spot_check="sometimes")

    def test_the_inclusion_arm_reports_no_floor_and_never_checks(self):
        rows, _picks = self._run(min_precision=NO_PRECISION_FLOOR)
        assert rows
        assert all(math.isnan(r["min_precision"]) for r in rows)
        assert all(r["floor_status"] == "" and r["floor_count"] == -1 for r in rows)
        assert all(r["phase"] != "check" for r in rows)

    def test_a_pinned_floor_is_recorded_and_sizes_the_candidate(self, schedule_only):
        rows, _picks = self._run(min_precision=0.25)
        assert rows and all(r["min_precision"] == 0.25 for r in rows)
        steps = [r for r in rows if r["phase"] != "check"]
        assert all(r["floor_count"] == min(64, r["n_remainder"]) for r in steps)

    def test_a_malformed_floor_kills_the_cell_before_it_runs(self):
        with pytest.raises(ValueError):
            self._run(min_precision=1.5)


class TestTheBalanceArm:
    """#4413: the line drawn at F-beta's beta, and the end-of-run check as the F-beta walk."""

    def _run(self, **kwargs):
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(), "alpha", seed=0, max_steps=15, calibrate_count=2, pick_sink=picks, **kwargs
        )
        return rows, picks

    def test_the_line_is_the_balances_and_the_floor_is_unused(self):
        rows, _ = self._run(beta=1.0, min_precision=0.9)
        steps = [r for r in rows if r["phase"] not in ("check", "")]
        assert steps and all(r["beta"] == 1.0 and math.isnan(r["min_precision"]) for r in steps)
        assert all(r["floor_status"] == FLOOR_UNCHECKED for r in steps)
        assert all(1 <= r["floor_count"] <= min(32, r["n_remainder"]) for r in steps), "the cap holds"

    def test_the_check_is_the_f_beta_walk_and_ends_checked(self):
        rows, picks = self._run(beta=1.0)
        check_rows = [r for r in rows if r["phase"] == "check"]
        assert check_rows, "the balance arm checks the line"
        last = check_rows[-1]
        assert last["floor_status"] == BALANCE_CHECKED and last["beta"] == 1.0
        assert last["floor_count"] >= 1 and 0.0 <= last["range_lo"] <= last["range_hi"] <= 1.0
        assert last["check_labelled"] >= 5 and all(p["phase"] != "check" or p["t"] > 15 for p in picks)

    def test_a_recall_leaning_balance_keeps_at_least_as_much(self):
        lo, _ = self._run(beta=0.5)
        hi, _ = self._run(beta=2.0)
        at = lambda rows: [r["floor_count"] for r in rows if r["phase"] not in ("check", "")]  # noqa: E731
        assert sum(at(hi)) >= sum(at(lo))

    def test_a_balance_outside_the_range_kills_the_cell_before_it_runs(self):
        with pytest.raises(ValueError, match="beta must be in"):
            self._run(beta=9.0)


class TestThePAwareAcquisitionArm:
    """#4409: the acquisition cut at the depth where the session's mixture says the ranking stops being P right."""

    def _run(self, min_precision, **kwargs):
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(), "alpha", seed=0, max_steps=15, calibrate_count=2, pick_sink=picks,
            min_precision=min_precision, spot_check="off", **kwargs,
        )  # fmt: skip
        return rows, picks

    def test_a_higher_floor_samples_higher_up_the_ranking(self):
        hi, _ = self._run(0.9, acq_inclusion_offset=0, acq_p_crossing=1.0)
        lo, _ = self._run(0.1, acq_inclusion_offset=0, acq_p_crossing=1.0)
        a_hi = [r["acq_threshold"] for r in hi if r["t"] >= 5]
        a_lo = [r["acq_threshold"] for r in lo if r["t"] >= 5]
        assert a_hi and len(a_hi) == len(a_lo)
        assert sum(a_hi) / len(a_hi) > sum(a_lo) / len(a_lo), "90% samples above 10%"

    def test_it_is_a_live_arm_not_the_shipped_cut(self):
        shipped, picks_shipped = self._run(0.5)
        aware, picks_aware = self._run(0.5, acq_inclusion_offset=0, acq_p_crossing=0.5)
        # The cut differs; on a 40-item fixture the opening takes every pick, so
        # whether the picks follow is for the bench run to show.
        assert [r["acq_threshold"] for r in shipped] != [r["acq_threshold"] for r in aware]
        assert len(picks_shipped) == len(picks_aware)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"acq_p_crossing": 1.0}, "acq_inclusion_offset=0"),
            ({"acq_inclusion_offset": 0, "acq_p_crossing": 0.0}, "must be > 0"),
            ({"acq_inclusion_offset": 0, "acq_rank_percentile": 0.9, "acq_p_crossing": 1.0}, "acq_inclusion_offset=0"),
        ],
    )
    def test_a_malformed_arm_dies_before_it_runs(self, kwargs, match):
        with pytest.raises(ValueError, match=match):
            self._run(0.5, **kwargs)

    def test_it_needs_a_preference(self):
        with pytest.raises(ValueError, match="needs a preference"):
            self._run(NO_PRECISION_FLOOR, acq_inclusion_offset=0, acq_p_crossing=1.0)

    def test_under_a_balance_it_samples_at_the_f_beta_argmax(self):
        """#4413: a precision-leaning balance samples higher than a recall-leaning one."""
        hi, _ = self._run(NO_PRECISION_FLOOR, beta=0.5, acq_inclusion_offset=0, acq_p_crossing=1.0)
        lo, _ = self._run(NO_PRECISION_FLOOR, beta=2.0, acq_inclusion_offset=0, acq_p_crossing=1.0)
        a_hi = [r["acq_threshold"] for r in hi if r["t"] >= 5 and r["phase"] != "check"]
        a_lo = [r["acq_threshold"] for r in lo if r["t"] >= 5 and r["phase"] != "check"]
        assert a_hi and len(a_hi) == len(a_lo)
        assert sum(a_hi) / len(a_hi) >= sum(a_lo) / len(a_lo)
