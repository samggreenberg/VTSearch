"""The eval harness's default arm draws the line a live detector draws (#4245).

The app's reporting line is the precision floor's when one is set - and the
default user has one - so the harness's default arm has to be too, or every
study measures a detector nobody ships.  The rule itself is shared
(``reporting_line``); what these tests pin is the harness's side of the
plumbing: that each simulated click is recorded with the provenance the app
would record, that the evidence filter reads it through the app's own
``calibrates_precision``, and that the arms the knob names behave as named.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from vtscore.datasets.vote_provenance import calibrates_precision
from vtscore.eval.autopilot_flow import pick_provenance
from vtscore.eval.voting_iterations import _calibration_rows, simulate_voting_iterations
from vtscore.training.thresholds import DEFAULT_MIN_PRECISION, NO_PRECISION_FLOOR

_STATES = {"promised", "unreachable", "insufficient_evidence"}


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

    def test_only_the_hard_phase_calibrates_a_promise(self):
        calibrating = {p for p in ("good", "bad", "hard", "new") if calibrates_precision(pick_provenance(p))}
        assert calibrating == {"hard"}

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
        return simulate_voting_iterations(_separable(), "alpha", seed=0, max_steps=15, calibrate_count=2, **kwargs)

    def test_the_default_arm_is_the_apps_floor(self):
        rows = self._run()
        assert rows
        assert all(r["min_precision"] == DEFAULT_MIN_PRECISION for r in rows)
        assert {r["floor_status"] for r in rows} <= _STATES | {""}
        assert any(r["floor_status"] in _STATES for r in rows), "no step built a floor estimate"

    def test_below_the_gate_the_default_arm_is_the_inclusion_zero_arm_step_for_step(self):
        """A floor that promises nothing draws the Inclusion 0 line, and acquisition samples 4 below it.

        So until the gate opens the default arm and the Inclusion arm at 0 are
        the same run - which is the owner's ruling that an unmet floor never
        changes what the user gets (#4247).  Fifteen votes cannot hold ten
        calibration positives, so the gate stays shut here by construction.
        """
        floored = self._run()
        off = self._run(min_precision=NO_PRECISION_FLOOR, inclusion=0)
        assert all(r["floor_status"] != "promised" for r in floored)
        # The plain frame carries no raw threshold; these all read it.
        for col in ("report_pool_percentile", "n_flagged", "cost", "acq_threshold", "acq_pool_percentile"):
            assert [r[col] for r in floored] == [r[col] for r in off], col

    def test_the_inclusion_arm_reports_no_floor(self):
        rows = self._run(min_precision=NO_PRECISION_FLOOR)
        assert rows
        assert all(math.isnan(r["min_precision"]) for r in rows)
        assert all(r["floor_status"] == "" for r in rows)

    def test_a_pinned_floor_is_recorded(self):
        rows = self._run(min_precision=0.9)
        assert rows and all(r["min_precision"] == 0.9 for r in rows)

    def test_a_malformed_floor_kills_the_cell_before_it_runs(self):
        with pytest.raises(ValueError):
            self._run(min_precision=1.5)
