"""The stopping-rule columns the harness emits beside ``phase`` (issue #3560).

A study's "final cost" is the metric at a click budget nobody chose; the number
a user actually leaves with is the metric at the click the app's stopping rules
fired.  Deriving that needs two things from every row: the phase (already
emitted), and the three indicator lights behind it — because ``hard`` cannot
say whether Smart or Stable is what is holding a run there, and "why did this
run never stop?" is exactly that question.

What these pin:

1. The lights are **on every row**, and are the ones that produced the phase —
   ``done`` means all three green, ``new`` means Span alone is not, and nothing
   else can claim to be ``done``.
2. They are **blank rather than guessed** where no phase machine ran, so an
   analysis can tell "not green" from "not measured".
3. Emitting them does not perturb the trajectory: the same run with and without
   a reader of them is the same run.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.autopilot_flow import STOPPING_PHASE, AutopilotFlow, stopping_rule_fired
from vtscore.eval.voting_columns import IDENT_COLUMNS, VOTING_COLUMNS
from vtscore.eval.voting_iterations import simulate_voting_iterations

LIGHTS = ("smart", "stable", "span")


def _clips(n=120, dim=16, seed=7):
    """A separable two-class pool big enough to run a real trajectory."""
    rng = np.random.default_rng(seed)
    medias = {}
    for i in range(n):
        cat = "target" if i < n // 3 else "other"
        emb = rng.standard_normal(dim).astype(np.float32) + (1.0 if cat == "target" else -1.0)
        medias[i + 1] = {
            "id": i + 1,
            "embedder": "e5",
            "embeddings": {"e5": (emb / np.linalg.norm(emb)).astype(np.float32)},
            "category": cat,
        }
    return medias


@pytest.fixture(scope="module")
def rows():
    return simulate_voting_iterations(
        _clips(), "target", seed=3, dataset_name="synthetic", max_steps=40, calibrate_count=1
    )


class TestSchema:
    def test_lights_are_declared_beside_the_phase(self):
        for col in (*LIGHTS, "span_level", "span_depth"):
            assert col in IDENT_COLUMNS, f"{col} must ride the identifying prefix, like `phase`"
            assert col in VOTING_COLUMNS

    def test_every_row_carries_them(self, rows):
        assert rows
        for r in rows:
            assert set(r) == set(VOTING_COLUMNS)


class TestLightsAgreeWithPhase:
    """The phase is a function of the lights, so the two can be checked against
    each other on real rows - which is what catches a light read off a stale
    step, the one failure a schema test cannot see."""

    def test_done_means_all_three_green(self, rows):
        for r in rows:
            if stopping_rule_fired(r["phase"]):
                assert [r[k] for k in LIGHTS] == ["green"] * 3, r

    def test_new_means_span_alone_is_short(self, rows):
        for r in rows:
            if r["phase"] == "new":
                assert r["smart"] == "green" and r["stable"] == "green"
                assert r["span"] != "green", r

    def test_hard_means_smart_or_stable_is_short(self, rows):
        for r in rows:
            if r["phase"] == "hard":
                assert not (r["smart"] == "green" and r["stable"] == "green"), r

    def test_span_counts_are_consistent_with_the_light(self, rows):
        for r in rows:
            if r["span"] == "":
                continue
            assert r["span_depth"] >= 0 and r["span_level"] >= 0
            if r["span"] == "green":
                assert r["span_level"] >= min(40, r["span_depth"]) or r["span_depth"] == 0, r


class TestNotMeasuredIsNotNotGreen:
    def test_a_run_with_no_phase_machine_leaves_them_blank(self):
        rows = simulate_voting_iterations(
            _clips(),
            "target",
            seed=3,
            dataset_name="synthetic",
            max_steps=8,
            calibrate_count=1,
            autopilot_fidelity=False,
        )
        assert rows
        for r in rows:
            assert r["phase"] == ""
            assert [r[k] for k in LIGHTS] == ["", "", ""]
            assert r["span_level"] == -1 and r["span_depth"] == -1

    def test_a_startup_schedules_own_rounds_leave_them_blank(self):
        """A schedule owns the phase without consulting the indicators, so its
        rounds are 'not measured' - and because a schedule is always a prefix of
        the trajectory, blank is exactly the rounds and never a stale reading
        carried forward into a step the machine did evaluate."""
        medias = _clips()
        # A schedule names positions on the seed sort, so it needs one: rank the
        # pool by its own first coordinate, which is arbitrary but real.
        seed_scores = {mid: float(m["embeddings"]["e5"][0]) for mid, m in medias.items()}
        rows = simulate_voting_iterations(
            medias,
            "target",
            seed=3,
            dataset_name="synthetic",
            max_steps=30,
            calibrate_count=1,
            seed_scores=seed_scores,
            startup_schedule="n12@top",
        )
        assert rows
        blank = [r["t"] for r in rows if r["smart"] == ""]
        assert blank, "the schedule's own rounds should be in the frame"
        assert all(r["phase"].startswith("s") for r in rows if r["smart"] == "")
        # Blank is a prefix: once the machine starts evaluating, it never stops.
        assert blank == sorted(blank) and max(blank) < min(
            (r["t"] for r in rows if r["smart"] != ""), default=float("inf")
        )

    def test_a_fresh_flow_has_not_measured_anything_yet(self):
        flow = AutopilotFlow()
        assert [flow.smart, flow.stable, flow.span] == ["", "", ""]
        assert flow.span_level == -1 and flow.span_depth == -1


class TestObservationDoesNotPerturb:
    def test_the_trajectory_is_unchanged_by_recording_the_lights(self, rows):
        """Two identical runs agree on every non-timing column, lights included.

        The lights are read off state the phase machine already computed, so
        they cannot move a vote - but "cannot" is what a regression makes false
        quietly, and the trajectory is the thing every study rests on.
        """
        again = simulate_voting_iterations(
            _clips(), "target", seed=3, dataset_name="synthetic", max_steps=40, calibrate_count=1
        )
        assert len(again) == len(rows)
        for a, b in zip(rows, again, strict=True):
            for col in ("t", "phase", "n_good", "n_bad", "cost", *LIGHTS, "span_level", "span_depth"):
                assert a[col] == b[col] or (isinstance(a[col], float) and np.isnan(a[col]) and np.isnan(b[col])), col


class TestStoppingPhaseName:
    def test_exhausted_is_not_a_stop(self):
        """Running out of pool is the opposite result from converging, and
        folding the two together would report a starved run as a stopped one."""
        assert stopping_rule_fired(STOPPING_PHASE)
        assert not stopping_rule_fired("exhausted")
        assert not stopping_rule_fired("hard")
