"""The eval harness's default arm draws the line a live detector draws (#4245, #4272).

The app's reporting line under a precision floor keeps a set - the floor's
unchecked starting candidate, or the set its spot check ended on - and the
default user has a floor, so the harness's default arm has to do the same, or
every study measures a detector nobody ships.  The rule itself is shared
(``floor_line`` / ``floor_state`` / ``SpotCheck``); what these tests pin is the
harness's side of the plumbing: that each simulated click is recorded with the
provenance the app would record, that the run's spot check draws its candidate
off the ranking, answers its picks from ground truth, casts them as votes and
retrains on them, and that the arms the knob names behave as named.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.datasets.vote_provenance import calibrates_precision
from vtscore.eval.autopilot_flow import pick_provenance
from vtscore.eval.voting_iterations import _calibration_rows, simulate_voting_iterations
from vtscore.training.thresholds import (
    CHECK_PROVENANCE,
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


class TestWhatTheAppWouldRecord:
    @pytest.mark.parametrize(
        ("phase", "sort_kind", "select_mode"),
        [("good", "text", "top"), ("bad", "text", "hard"), ("hard", "learned", "hard"), ("new", "learned", "new")],
    )
    def test_each_phase_draws_off_the_sort_the_label_view_sets(self, phase, sort_kind, select_mode):
        assert pick_provenance(phase) == {
            "flow": "autopilot",
            "phase": phase,
            "select_mode": select_mode,
            "sort_kind": sort_kind,
        }

    @pytest.mark.parametrize("phase", [None, "idle", "done", "exhausted"])
    def test_no_labelling_phase_records_nothing(self, phase):
        assert pick_provenance(phase) is None

    def test_only_the_hard_phase_calibrates_the_estimate(self):
        calibrating = {p for p in ("good", "bad", "hard", "new") if calibrates_precision(pick_provenance(p))}
        assert calibrating == {"hard"}
        assert not calibrates_precision(CHECK_PROVENANCE)

    def test_rows_map_back_to_their_votes(self):
        provenance = {1: pick_provenance("good"), 2: pick_provenance("hard"), 3: pick_provenance("new")}
        details = {"row_votes": [1, 2, 3, 2]}
        assert _calibration_rows(details, provenance) == [False, True, False, True]
        # No phase machine: no app counterpart to filter on, so every vote serves.
        assert _calibration_rows(details, None) is None
        # A trainer that cannot name its rows' votes lets none serve.
        assert _calibration_rows({}, provenance) == []


class TestTheArms:
    def _run(self, **kwargs):
        picks: list[dict] = []
        rows = simulate_voting_iterations(
            _separable(), "alpha", seed=0, max_steps=15, calibrate_count=2, pick_sink=picks, **kwargs
        )
        return rows, picks

    def test_the_default_arm_is_the_apps_floor(self):
        rows, _picks = self._run()
        assert rows
        assert all(r["min_precision"] == DEFAULT_MIN_PRECISION for r in rows)
        assert {r["floor_status"] for r in rows} <= set(FLOOR_STATES)

    def test_before_the_check_the_line_keeps_the_unchecked_starting_candidate(self):
        """What a headless run exports: the top K unvoted, K from the floor's schedule, capped by the corpus."""
        rows, _picks = self._run()
        steps = [r for r in rows if r["phase"] != "check"]
        assert steps and all(r["floor_status"] == FLOOR_UNCHECKED for r in steps)
        k = check_schedule(DEFAULT_MIN_PRECISION).candidate
        # `n_remainder` is the unvoted sim set after this step's vote: the candidate.
        assert all(r["floor_count"] == min(k, r["n_remainder"]) for r in steps)
        assert all(math.isnan(r["range_lo"]) and r["check_labelled"] == -1 and r["check_stale"] == -1 for r in steps)

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
        assert (last["floor_status"] == FLOOR_CONFIRMED) == (last["range_lo"] >= DEFAULT_MIN_PRECISION - 1e-9)

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

    def test_a_pinned_floor_is_recorded_and_sizes_the_candidate(self):
        rows, _picks = self._run(min_precision=0.25)
        assert rows and all(r["min_precision"] == 0.25 for r in rows)
        steps = [r for r in rows if r["phase"] != "check"]
        assert all(r["floor_count"] == min(64, r["n_remainder"]) for r in steps)

    def test_a_malformed_floor_kills_the_cell_before_it_runs(self):
        with pytest.raises(ValueError):
            self._run(min_precision=1.5)
