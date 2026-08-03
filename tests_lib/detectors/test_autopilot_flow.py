"""Parity tests for the eval harness's port of the app's Autopilot flow.

These pin :mod:`vtscore.eval.autopilot_flow` against the two sources it ports —
``AutopilotStateService.checkPhaseTransition`` (TypeScript, so it cannot be
imported) and the indicator rules in
:mod:`vtscore.detectors.labeling_progress`.  The constants are asserted
explicitly rather than read from the module, so a drift in either direction
fails here instead of silently changing what every future study measures.

Background: the harness used to train from the first ``(good, bad)`` pair and
interleave Hard/New on step parity, neither of which the app does.  That made
issue #2788's cold-start degenerate thresholds look like a user-facing bug when
they were an artifact of a flow no user takes.
"""

import math

import numpy as np
import pytest

from vtscore.eval.al_strategies import ALContext, select_next
from vtscore.eval.autopilot_flow import (
    BAD_TARGET,
    GOOD_TARGET,
    MIN_PER_CLASS,
    SMART_FLAT_THRESHOLD,
    SPAN_YELLOW,
    STABLE_MAX_THRESHOLD,
    STABLE_RATE_THRESHOLD,
    AutopilotFlow,
    Status,
    app_has_detector,
    next_phase,
    smart_status,
    span_status,
    stable_status,
)


def _sim_clips(n=160, dim=16):
    """A small, well-separated two-class collection for end-to-end runs."""
    rng = np.random.default_rng(0)
    clips = {}
    for i in range(n):
        pos = i % 4 == 0
        center = np.ones(dim, np.float32) if pos else -np.ones(dim, np.float32)
        clips[i] = {
            "embeddings": {"e": center + rng.standard_normal(dim).astype(np.float32) * 1.5},
            "embedder_names": ["e"],
            "category": "cat" if pos else "other",
        }
    return clips


def _fidelity_rows():
    """One faithful-flow trajectory, shared by the integration assertions."""
    from vtscore.eval.voting_iterations import simulate_voting_iterations

    clips = _sim_clips()
    seed_scores = {i: float(np.mean(clips[i]["embeddings"]["e"])) for i in clips}
    return simulate_voting_iterations(
        clips,
        "cat",
        seed=1,
        max_steps=20,
        atlas_min_node_size=5,
        seed_scores=seed_scores,
    )


@pytest.fixture(scope="module")
def rows():
    """One faithful-flow trajectory, computed once for the whole module."""
    return _fidelity_rows()


def _phase(
    good: int,
    bad: int,
    *,
    smart: Status = "red",
    stable: Status = "red",
    span: Status = "red",
    remaining: float = 1000,
) -> str:
    return next_phase(good, bad, remaining_unlabeled=remaining, smart=smart, stable=stable, span=span)


class TestPortedConstants:
    """The app's numbers, asserted literally so a drift fails loudly."""

    def test_vote_targets_match_autopilot_initial_state(self):
        # frontend INITIAL_STATE: goodToStart 3, badToStart 4.
        assert (GOOD_TARGET, BAD_TARGET) == (3, 4)

    def test_indicator_gates_match_labeling_progress(self):
        # _compute_smart_status / _compute_stable_status: "Need at least 5 good
        # and 5 bad"; FLAT_THRESHOLD -0.015; stable rate/max 0.005 / 0.01.
        assert MIN_PER_CLASS == 5
        assert SMART_FLAT_THRESHOLD == -0.015
        assert STABLE_RATE_THRESHOLD == 0.005
        assert STABLE_MAX_THRESHOLD == 0.01
        assert SPAN_YELLOW == 10

    def test_quorum_always_clears_the_calibrator_fold_split_guard(self):
        """The reason #2788's cold start is not an Autopilot-path problem.

        ``compute_fold_orderings`` needs >=4 votes and >=2 per class before it
        can form a stratified split; below that it returns the flat 0.5
        ``too_few_default``.  Autopilot's first learned sort happens only once
        both targets are met, which clears both bounds by construction.
        """
        assert GOOD_TARGET + BAD_TARGET >= 4
        assert min(GOOD_TARGET, BAD_TARGET) >= 2


class TestPhaseMachine:
    def test_good_phase_until_the_good_target(self):
        assert _phase(0, 0) == "good"
        assert _phase(GOOD_TARGET - 1, 0) == "good"

    def test_bad_phase_until_the_bad_target(self):
        assert _phase(GOOD_TARGET, 0) == "bad"
        assert _phase(GOOD_TARGET, BAD_TARGET - 1) == "bad"

    def test_hard_once_quorum_is_reached(self):
        assert _phase(GOOD_TARGET, BAD_TARGET) == "hard"

    def test_new_only_once_smart_and_stable_are_green(self):
        """The app switches to diversity on the indicators, never on parity."""
        assert _phase(9, 9, smart="green", stable="yellow") == "hard"
        assert _phase(9, 9, smart="yellow", stable="green") == "hard"
        assert _phase(9, 9, smart="green", stable="green") == "new"

    def test_done_when_all_three_are_green(self):
        assert _phase(9, 9, smart="green", stable="green", span="green") == "done"

    def test_exhausted_when_nothing_left_and_not_all_green(self):
        assert _phase(9, 9, remaining=0) == "exhausted"
        # All-green still wins over exhausted, as in the app's branch order.
        assert _phase(9, 9, smart="green", stable="green", span="green", remaining=0) == "done"

    def test_targets_are_capped_by_what_the_collection_can_supply(self):
        """A tiny collection must not strand the flow in an early phase.

        With one good vote and nothing left to label, the 3-good / 4-bad
        targets are both unreachable.  Uncapped they would pin the phase at
        ``good`` forever; capped, the flow advances and lands in the terminal
        ``exhausted`` state.
        """
        assert _phase(1, 0, remaining=0) == "exhausted"
        # Same counts, but with items still available the targets bind normally.
        assert _phase(1, 0, remaining=100) == "good"

    def test_unknown_collection_size_leaves_targets_uncapped(self):
        assert _phase(0, 0, remaining=math.inf) == "good"

    def test_phase_can_regress_when_votes_are_removed(self):
        assert _phase(GOOD_TARGET, BAD_TARGET) == "hard"
        assert _phase(GOOD_TARGET - 1, BAD_TARGET) == "good"


class TestDetectorVisibility:
    def test_no_detector_before_the_first_learned_sort(self):
        assert not app_has_detector("good")
        assert not app_has_detector("bad")

    def test_detector_from_the_hard_phase_on(self):
        assert app_has_detector("hard")
        assert app_has_detector("new")
        assert app_has_detector("done")


class TestSmartStatus:
    def test_red_below_five_per_class(self):
        assert smart_status([0.5, 0.4, 0.3], MIN_PER_CLASS - 1, 9) == "red"
        assert smart_status([0.5, 0.4, 0.3], 9, MIN_PER_CLASS - 1) == "red"

    def test_yellow_without_enough_history(self):
        assert smart_status([0.5, 0.4], 9, 9) == "yellow"

    def test_yellow_while_the_error_cost_is_still_falling(self):
        assert smart_status([0.9, 0.7, 0.5, 0.3], 9, 9) == "yellow"

    def test_green_once_the_error_cost_levels_off(self):
        assert smart_status([0.30, 0.30, 0.30, 0.30], 9, 9) == "green"

    def test_green_when_the_cost_is_rising(self):
        assert smart_status([0.2, 0.3, 0.4, 0.5], 9, 9) == "green"


class TestStableStatus:
    def test_red_below_five_per_class(self):
        entries = [{"num_flips": 0, "num_unlabeled": 100}] * 6
        assert stable_status(entries, MIN_PER_CLASS - 1, 9) == "red"

    def test_yellow_without_enough_entries(self):
        entries = [{"num_flips": 0, "num_unlabeled": 100}] * 4
        assert stable_status(entries, 9, 9) == "yellow"

    def test_green_once_predictions_settle(self):
        entries = [{"num_flips": 0, "num_unlabeled": 1000}] * 6
        assert stable_status(entries, 9, 9) == "green"

    def test_yellow_while_predictions_still_flip(self):
        entries = [{"num_flips": 50, "num_unlabeled": 1000}] * 6
        assert stable_status(entries, 9, 9) == "yellow"

    def test_a_single_spike_blocks_green(self):
        """``max_flip_rate`` is a separate gate from the average."""
        entries = [{"num_flips": 0, "num_unlabeled": 1000}] * 5
        entries.append({"num_flips": 20, "num_unlabeled": 1000})  # 2% > 1% max
        assert stable_status(entries, 9, 9) == "yellow"

    def test_zero_unlabeled_counts_as_no_flips(self):
        entries = [{"num_flips": 7, "num_unlabeled": 0}] * 6
        assert stable_status(entries, 9, 9) == "green"


class TestSpanStatus:
    def test_thresholds(self):
        assert span_status(0, 100) == "red"
        assert span_status(SPAN_YELLOW - 1, 100) == "red"
        assert span_status(SPAN_YELLOW, 100) == "yellow"
        assert span_status(40, 100) == "green"

    def test_green_target_is_capped_at_the_tree_size(self):
        """A tree smaller than the goal must still be able to go green."""
        assert span_status(5, 5) == "green"

    def test_degenerate_tree_is_green(self):
        assert span_status(0, 0) == "green"


class TestAutopilotFlow:
    def test_starts_in_the_good_phase(self):
        assert AutopilotFlow().phase == "good"

    def test_advances_through_the_initial_phases(self):
        flow = AutopilotFlow()
        assert flow.update(GOOD_TARGET, 0, 500, None) == "bad"
        assert flow.update(GOOD_TARGET, BAD_TARGET, 500, None) == "hard"

    def test_records_flip_rates_between_consecutive_steps(self):
        flow = AutopilotFlow()
        flow.record_step(0.4, {1: 1, 2: 0, 3: 1})
        assert flow.stability == []  # nothing to compare the first step against
        flow.record_step(0.3, {1: 0, 2: 0, 3: 1})  # id 1 flipped
        assert flow.stability == [{"num_flips": 1, "num_unlabeled": 3}]

    def test_ignores_steps_with_no_model(self):
        flow = AutopilotFlow()
        flow.record_step(None, None)
        assert flow.error_costs == []
        assert flow.stability == []

    def test_span_drives_the_new_to_done_transition(self):
        flow = AutopilotFlow()
        flow.error_costs = [0.3] * 4
        flow.stability = [{"num_flips": 0, "num_unlabeled": 1000}] * 6
        assert flow.update(9, 9, 500, {"level": 0, "depth": 100}) == "new"
        assert flow.update(9, 9, 500, {"level": 40, "depth": 100}) == "done"


class TestSelectorPhaseParity:
    """The selector's picks must match the app's Sort + Select pairing."""

    @staticmethod
    def _ctx(pool, *, phase, scores=None, seed_scores=None, threshold=0.5, model=None):
        ids = list(pool)
        return ALContext(
            pool_ids=ids,
            embeddings={i: np.zeros(4, np.float32) for i in ids},
            labeled={},
            scores=scores or {},
            model=model,
            threshold=threshold,
            atlas=None,
            rng=np.random.RandomState(0),
            pool_labels=None,
            seed_scores=seed_scores,
            phase=phase,
        )

    def test_good_phase_takes_the_top_of_the_text_sort(self):
        seed = {1: 0.1, 2: 0.9, 3: 0.5}
        ctx = self._ctx([1, 2, 3], phase="good", seed_scores=seed)
        assert select_next("autopilot", ctx) == 2

    def test_bad_phase_never_consults_the_detector(self):
        """The app is still on the text sort here, so a model must be ignored.

        This is the divergence behind issue #2788: the old harness trained at
        3 good + 1 bad and ranked bads by that model, a state the app never
        reaches.
        """
        seed = {i: float(i) for i in range(1, 11)}
        # A model that would rank id 10 lowest; the faithful pick must not use it.
        scores = {i: (0.0 if i == 10 else 1.0) for i in range(1, 11)}
        ctx = self._ctx(list(range(1, 11)), phase="bad", seed_scores=seed, scores=scores, model=object())
        assert select_next("autopilot", ctx) != 10

    def test_bad_phase_picks_the_middle_of_the_text_sort_not_the_bottom(self):
        """Select ``hard`` on a text sort lands at the cutoff, not the tail."""
        # Two well-separated clusters so the GMM cut falls between them; the
        # pick should sit at that seam rather than at the lowest-scoring item.
        seed = {i: 0.0 + 0.01 * i for i in range(10)}
        seed.update({i: 1.0 + 0.01 * i for i in range(10, 20)})
        ctx = self._ctx(list(range(20)), phase="bad", seed_scores=seed)
        pick = select_next("autopilot", ctx)
        assert pick not in (0, 19), "picked an extreme of the ranking, not the cutoff"

    def test_hard_phase_picks_by_rank_not_by_score_distance(self):
        """The app measures distance in rank space; scores cluster unevenly.

        Here id 3 is nearest the threshold in raw score, but the ranked
        positions put id 2 adjacent to the cutoff.  A score-space ``argmin``
        would pick 3; the app picks by index.
        """
        # Descending order: 1 (0.99), 2 (0.98), 3 (0.10), 4 (0.01)
        scores = {1: 0.99, 2: 0.98, 3: 0.10, 4: 0.01}
        ctx = self._ctx([1, 2, 3, 4], phase="hard", scores=scores, threshold=0.5, model=object())
        # threshold_index = first score <= 0.5 -> index 2 (id 3).
        # Nearest unlabeled by index is id 3 itself, so make it labeled-out by
        # removing it from the pool: then ids 2 and 4 tie at distance 1 and the
        # first in ranked order (id 2) wins.
        ctx.pool_ids = [1, 2, 4]
        assert select_next("autopilot", ctx) == 2

    def test_legacy_mode_is_unchanged_when_no_phase_is_supplied(self):
        """``phase=None`` must keep the pre-alignment behaviour intact."""
        seed = {i: float(i) for i in range(1, 6)}
        ctx = ALContext(
            pool_ids=[1, 2, 3, 4, 5],
            embeddings={i: np.zeros(4, np.float32) for i in range(1, 6)},
            labeled={100: 1.0, 101: 1.0, 102: 1.0},  # 3 goods -> legacy bad phase
            scores={},
            model=None,
            threshold=0.5,
            atlas=None,
            rng=np.random.RandomState(0),
            pool_labels=None,
            seed_scores=seed,
            phase=None,
        )
        # Legacy bad phase takes the *bottom* of the text ranking.
        assert select_next("autopilot", ctx) == 1


class TestHarnessIntegration:
    """End-to-end: the simulated trajectory visits the app's trained states."""

    def test_no_user_visible_threshold_before_quorum(self, rows):
        for r in rows:
            if r["app_trained"]:
                assert r["n_good"] >= GOOD_TARGET and r["n_bad"] >= BAD_TARGET

    def test_the_first_user_visible_step_can_be_calibrated(self, rows):
        """Every app-visible step clears the fold-split guard, so the flat 0.5
        ``too_few_default`` cut cannot arise on the Autopilot path."""
        trained = [r for r in rows if r["app_trained"]]
        assert trained, "the run never reached a trained phase"
        first = trained[0]
        assert first["n_good"] + first["n_bad"] >= 4
        assert min(first["n_good"], first["n_bad"]) >= 2

    def test_metrics_are_still_recorded_before_quorum(self, rows):
        """Fidelity changes the vote order, not the measurement coverage."""
        pre = [r for r in rows if not r["app_trained"]]
        assert pre, "expected pre-quorum steps to still be measured"
        assert all(r["cost"] is not None for r in pre)

    def test_legacy_mode_trains_from_the_first_vote_pair(self):
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        clips = _sim_clips()
        seed_scores = {i: float(np.mean(clips[i]["embeddings"]["e"])) for i in clips}
        rows = simulate_voting_iterations(
            clips,
            "cat",
            seed=1,
            max_steps=8,
            atlas_min_node_size=5,
            seed_scores=seed_scores,
            autopilot_fidelity=False,
        )
        assert rows[0]["app_trained"] == 1
        assert min(rows[0]["n_good"], rows[0]["n_bad"]) == 1
