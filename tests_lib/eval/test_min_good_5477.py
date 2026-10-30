"""The Goods minimum behind Smart and Stable (issue #5477): ``min_good`` and ``hard_dry_run``.

Both lights stay red until the labelset holds ``MIN_PER_CLASS`` (5) Goods, and a
target with only 3 or 4 to find - most FHIBE people - never gets there, so
Autopilot never says Done.  ``min_good`` lowers the Goods the lights need;
``hard_dry_run`` waives the minimum down to the Goods in hand once that many
learned picks in a row held no Good.  Unset, both are the app.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.detectors.cost_trend import rising_status_from_scores, smart_status_from_costs
from vtscore.detectors.stability import MIN_PER_CLASS, stable_status_from_entries
from vtscore.eval.autopilot_flow import AutopilotFlow

FLAT = [0.9] * 5
_QUIET = {"num_flips": 0, "num_confident_flips": 0, "num_unlabeled": 90, "num_pool": 100}


class TestTheRules:
    def test_the_default_needs_five_goods(self):
        assert stable_status_from_entries([_QUIET] * 6, MIN_PER_CLASS - 1, 9)["status"] == "red"
        assert rising_status_from_scores(FLAT, MIN_PER_CLASS - 1, 9)["status"] == "red"
        assert smart_status_from_costs([0.1] * 5, MIN_PER_CLASS - 1, 9)["status"] == "red"

    def test_a_lower_minimum_reads_the_light(self):
        assert stable_status_from_entries([_QUIET] * 6, 3, 9, min_good=3)["status"] == "green"
        assert rising_status_from_scores(FLAT, 3, 9, min_good=3)["status"] == "green"
        assert smart_status_from_costs([0.1] * 5, 3, 9, min_good=3)["status"] == "green"

    def test_the_bads_minimum_is_unchanged(self):
        result = stable_status_from_entries([_QUIET] * 6, 9, MIN_PER_CLASS - 1, min_good=1)
        assert result["status"] == "red"
        assert result["reason"] == "Need at least 1 good and 5 bad. Currently 9g, 4b."


def _flow(**kwargs):
    return AutopilotFlow(**kwargs)


def _vote(flow, good, bad):
    """One vote's worth of a converged session: a flat Smart window and no flips."""
    flow.record_step(None, {i: 0.0 for i in range(90)}, 0.5, num_pool=100, windows={"fbeta_best": FLAT})
    return flow.update(good, bad, 90, None)


def _session(flow, goods=3, misses=0):
    """Good to *goods*, Bad to the target, then *misses* learned picks without a Good; the phases."""
    phases = [_vote(flow, g, 0) for g in range(1, goods + 1)]
    phases += [_vote(flow, goods, b) for b in range(1, 5)]
    phases += [_vote(flow, goods, b) for b in range(5, 5 + misses)]
    return phases


class TestTheFlow:
    def test_three_goods_never_leave_hard_in_the_app(self):
        phases = _session(_flow(), misses=40)
        assert phases[-1] == "hard"

    def test_a_minimum_of_three_reaches_the_check(self):
        assert _session(_flow(min_good=3), misses=10)[-1] == "spot"

    def test_a_dry_run_waives_the_minimum_after_its_misses(self):
        flow = _flow(hard_dry_run=6)
        phases = _session(flow, misses=5)
        assert phases[-1] == "hard" and not flow.hard_dry
        assert _vote(flow, 3, 10) == "spot"
        assert flow.hard_dry and flow.min_good_now(3) == 3

    def test_a_good_resets_the_run(self):
        flow = _flow(hard_dry_run=6)
        _session(flow, misses=5)
        _vote(flow, 4, 9)
        assert flow.hard_misses == 0
        assert [_vote(flow, 4, b) for b in range(10, 15)][-1] == "hard"
        assert _vote(flow, 4, 15) == "spot"

    def test_the_opening_does_not_count_toward_the_run(self):
        flow = _flow(hard_dry_run=2)
        _session(flow, misses=0)
        assert flow.hard_misses <= 1 and not flow.hard_dry

    def test_the_waiver_never_raises_the_minimum(self):
        flow = _flow(min_good=2, hard_dry_run=1)
        _session(flow, misses=2)
        assert flow.min_good_now(4) == 2

    @pytest.mark.parametrize("kwargs", [{"min_good": 0}, {"min_good": 6}, {"min_good": True}, {"hard_dry_run": 0}])
    def test_a_bad_knob_is_refused(self, kwargs):
        with pytest.raises(ValueError):
            _flow(**kwargs)


def _sparse_clips(n=160, n_pos=4, dim=16):
    """A collection with only *n_pos* positives, well apart from the rest."""
    rng = np.random.default_rng(0)
    clips = {}
    for i in range(n):
        pos = i < n_pos
        center = np.ones(dim, np.float32) if pos else -np.ones(dim, np.float32)
        clips[i] = {
            "embeddings": {"e": center + rng.standard_normal(dim).astype(np.float32) * 0.5},
            "embedder_names": ["e"],
            "category": "cat" if pos else "other",
        }
    return clips


def _run(**kwargs):
    from vtscore.eval.voting_iterations import simulate_voting_iterations

    clips = _sparse_clips()
    seed_scores = {i: float(clips[i]["embeddings"]["e"].mean()) for i in clips}
    return simulate_voting_iterations(
        clips, "cat", seed=1, max_steps=50, atlas_min_node_size=5, seed_scores=seed_scores, **kwargs
    )


class TestTheHarness:
    def test_unset_is_the_app(self):
        assert [r["phase"] for r in _run(min_good=MIN_PER_CLASS)] == [r["phase"] for r in _run()]

    def test_a_scarce_target_is_told_done_only_under_an_arm(self):
        assert "done" not in {r["phase"] for r in _run()}
        assert "done" in {r["phase"] for r in _run(hard_dry_run=8)}

    @pytest.mark.parametrize("kwargs", [{"min_good": 0}, {"min_good": 1.5}, {"hard_dry_run": -1}])
    def test_a_bad_knob_is_refused(self, kwargs):
        with pytest.raises(ValueError, match="min_good|hard_dry_run"):
            _run(**kwargs)
