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

And then the **margins** beside the lights: the continuous quantities each gate
is a threshold on.  A light says whether a rule fired; a study asking whether
the rule is any good needs how close it came, and "Stable held it" is three
different findings until the three rates behind it are on the row.  What these
pin, beyond the three above applied to eight more columns:

4. Every margin **agrees with the light beside it** — a green Smart implies its
   slope cleared the flatness cutoff or its t cleared the significance one, a
   green Stable implies all three of its rates are inside their gates, and a
   green Span implies ``span_level >= span_target``.  That is what makes them
   margins *of this rule* rather than eight numbers that happen to ride along.
5. They are ``NaN`` where the rule declined to fit one, never 0.0 — which on the
   Smart flatness test would read as the exact centre of the gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.autopilot_flow import (
    SMART_FLAT_THRESHOLD,
    SMART_SLOPE_T,
    STABLE_FALLING_RATIO,
    STABLE_MAX_THRESHOLD,
    STABLE_RATE_THRESHOLD,
    STOPPING_PHASE,
    AutopilotFlow,
    span_target,
    stopping_rule_fired,
)
from vtscore.eval.voting_columns import IDENT_COLUMNS, STOPPING_MARGIN_COLUMNS, VOTING_COLUMNS
from vtscore.eval.voting_iterations import simulate_voting_iterations

LIGHTS = ("smart", "stable", "span")
SPAN_COUNTS = ("span_level", "span_depth", "span_target")


def _fit(r, col):
    """Whether the rule actually produced *col* on this row."""
    return isinstance(r[col], float) and not np.isnan(r[col])


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
        for col in (*LIGHTS, *SPAN_COUNTS, *STOPPING_MARGIN_COLUMNS):
            assert col in IDENT_COLUMNS, f"{col} must ride the identifying prefix, like `phase`"
            assert col in VOTING_COLUMNS

    def test_margins_ride_the_calibration_frame_too(self):
        """A calibration study's main frame IS its variant frame - one row per
        `pool_variant` replaces the plain row - so margins that lived only on
        the plain schema would be missing from exactly the studies that ask
        where the rules fired."""
        from vtscore.eval.voting_columns import CALIBRATION_COLUMNS

        for col in (*STOPPING_MARGIN_COLUMNS, "span_target"):
            assert col in CALIBRATION_COLUMNS

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
            # The bar is emitted rather than assumed to be 40: it is capped at
            # the tree size, and the goal itself is a per-run knob.
            assert r["span_target"] == span_target(r["span_depth"])
            assert (r["span"] == "green") == (r["span_level"] >= r["span_target"]), r


class TestMarginsAgreeWithTheLights:
    """The margins are the quantities the gates are thresholds on, so each light
    is re-derivable from them.  That is the check that catches a margin read off
    a stale step - the one failure a schema test cannot see, and the one that
    would make every 'how close did it come' number quietly describe the
    previous click."""

    def test_smart_green_means_one_of_its_two_gates_cleared(self, rows):
        for r in rows:
            if r["smart"] != "green" or not _fit(r, "smart_slope"):
                continue
            assert r["smart_slope"] >= SMART_FLAT_THRESHOLD or r["smart_slope_t"] > -SMART_SLOPE_T, r

    def test_smart_yellow_with_a_fitted_window_means_neither_cleared(self, rows):
        for r in rows:
            if r["smart"] != "yellow" or not _fit(r, "smart_slope"):
                continue
            assert r["smart_slope"] < SMART_FLAT_THRESHOLD and r["smart_slope_t"] <= -SMART_SLOPE_T, r

    def test_stable_green_means_all_three_of_its_gates_cleared(self, rows):
        for r in rows:
            if r["stable"] != "green" or not _fit(r, "stable_confident_flip_rate"):
                continue
            assert r["stable_confident_flip_rate"] < STABLE_RATE_THRESHOLD, r
            assert r["stable_max_confident_flip_rate"] < STABLE_MAX_THRESHOLD, r
            still_falling = r["stable_flip_rate_late"] >= STABLE_RATE_THRESHOLD and (
                r["stable_flip_rate_late"] < STABLE_FALLING_RATIO * r["stable_flip_rate_early"]
            )
            assert not still_falling, r

    def test_stable_yellow_means_at_least_one_did_not(self, rows):
        for r in rows:
            if r["stable"] != "yellow" or not _fit(r, "stable_confident_flip_rate"):
                continue
            blocked = (
                r["stable_confident_flip_rate"] >= STABLE_RATE_THRESHOLD
                or r["stable_max_confident_flip_rate"] >= STABLE_MAX_THRESHOLD
                or (
                    r["stable_flip_rate_late"] >= STABLE_RATE_THRESHOLD
                    and r["stable_flip_rate_late"] < STABLE_FALLING_RATIO * r["stable_flip_rate_early"]
                )
            )
            assert blocked, r

    def test_the_average_never_exceeds_its_own_worst_step(self, rows):
        """A window average above the window max is the signature of two
        different windows - the arithmetic cannot produce it."""
        for r in rows:
            if not _fit(r, "stable_confident_flip_rate"):
                continue
            assert r["stable_confident_flip_rate"] <= r["stable_max_confident_flip_rate"] + 1e-9, r

    def test_a_run_actually_exercises_them(self, rows):
        """Guards the four tests above from passing vacuously on a trajectory
        where no rule ever fitted a margin."""
        assert sum(1 for r in rows if _fit(r, "smart_slope")) >= 5
        assert sum(1 for r in rows if _fit(r, "stable_confident_flip_rate")) >= 5


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
            assert r["span_level"] == -1 and r["span_depth"] == -1 and r["span_target"] == -1
            # NaN, not 0.0: zero is a real slope, and on the flatness test it is
            # the most converged reading there is.
            for col in STOPPING_MARGIN_COLUMNS:
                assert np.isnan(r[col]), col

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
        assert flow.span_level == -1 and flow.span_depth == -1 and flow.span_target == -1
        for col in STOPPING_MARGIN_COLUMNS:
            assert np.isnan(getattr(flow, col)), col

    def test_too_little_history_is_not_measured_either(self):
        """The rules refuse to fit below their per-class minimum and their
        history minimums, and the harness reports that refusal as NaN rather
        than as a margin of zero."""
        flow = AutopilotFlow()
        flow.update(1, 1, remaining_unlabeled=50, span={"level": 0, "depth": 12})
        assert flow.smart == "red" and flow.stable == "red"
        assert np.isnan(flow.smart_slope) and np.isnan(flow.stable_confident_flip_rate)
        # Span is a count, not a fit: it is measured from the first update.
        assert flow.span_level == 0 and flow.span_target == 12


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
            for col in ("t", "phase", "n_good", "n_bad", "cost", *LIGHTS, *SPAN_COUNTS, *STOPPING_MARGIN_COLUMNS):
                assert a[col] == b[col] or (isinstance(a[col], float) and np.isnan(a[col]) and np.isnan(b[col])), col


class TestStoppingPhaseName:
    def test_exhausted_is_not_a_stop(self):
        """Running out of pool is the opposite result from converging, and
        folding the two together would report a starved run as a stopped one."""
        assert stopping_rule_fired(STOPPING_PHASE)
        assert not stopping_rule_fired("exhausted")
        assert not stopping_rule_fired("hard")
