"""Issue #4197's opening-diversity knob: the text-sort walk passes over the Bads' near neighbours.

An experiment knob, not app behaviour: off by default, so the default arm is the
app's own opening pick for pick.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.al_strategies import ALContext, _pick_good_phase, _select_phase_faithful
from vtscore.eval.voting_iterations import _parse_opening_diversity

SIB = np.array([1.0, 0.0, 0.0], dtype=np.float32)  # the sibling cluster's direction
POS = np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _ctx(labeled: dict[int, float], *, diversity=None) -> ALContext:
    """Pool 10..14 ranked 14 > 13 > ... by the text sort; 14 and 13 are siblings, 12 is a positive."""
    emb = {
        1: SIB,  # a Bad already voted: a sibling
        2: SIB + np.float32(0.05) * POS,  # a second sibling Bad
        14: SIB + np.float32(0.02) * POS,
        13: SIB + np.float32(0.03) * POS,
        12: POS,
        11: POS + np.float32(0.1) * SIB,
        10: np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    pool = [10, 11, 12, 13, 14]
    return ALContext(
        pool_ids=pool,
        embeddings=emb,
        labeled=labeled,
        scores={},
        model=None,
        threshold=0.5,
        atlas=None,
        seed_scores={i: float(i) for i in pool},
        phase="good",
        opening_diversity=diversity,
    )


class TestPick:
    def test_off_by_default_takes_the_plain_top(self):
        assert _pick_good_phase(_ctx({1: 0.0})) == 14

    def test_passes_over_candidates_near_a_bad(self):
        assert _pick_good_phase(_ctx({1: 0.0}, diversity=(0.9, 1))) == 12

    def test_needs_k_near_bads_before_passing_over(self):
        # One sibling Bad is not a cluster yet at k = 2 ...
        assert _pick_good_phase(_ctx({1: 0.0}, diversity=(0.9, 2))) == 14
        # ... two are.
        assert _pick_good_phase(_ctx({1: 0.0, 2: 0.0}, diversity=(0.9, 2))) == 12

    def test_goods_are_not_avoided(self):
        assert _pick_good_phase(_ctx({1: 1.0}, diversity=(0.9, 1))) == 14

    def test_without_bads_it_is_the_plain_top(self):
        assert _pick_good_phase(_ctx({}, diversity=(0.9, 1))) == 14

    def test_falls_back_to_the_plain_top_when_everything_is_near(self):
        assert _pick_good_phase(_ctx({1: 0.0}, diversity=(-1.0, 1))) == 14

    def test_the_more_walk_uses_it_too(self):
        ctx = _ctx({1: 0.0}, diversity=(0.9, 1))
        ctx.phase = "more"
        assert _select_phase_faithful(ctx, "more") == 12


class TestSpec:
    def test_parses(self):
        assert _parse_opening_diversity("0.85/1") == (0.85, 1)
        assert _parse_opening_diversity(None) is None
        assert _parse_opening_diversity("") is None

    @pytest.mark.parametrize("spec", ["0.85", "0.85/0", "1.5/1", "x/1", "0.8/2/3"])
    def test_rejects_junk(self, spec):
        with pytest.raises(ValueError):
            _parse_opening_diversity(spec)


class TestInTheHarness:
    @pytest.fixture
    def walk(self, monkeypatch):
        """#4282's More walk (20 Goods), which the diversity pass acts on: off on photos since #4740."""
        from functools import partial

        from vtscore.eval import voting_iterations as VI
        from vtscore.eval.autopilot_flow import AutopilotFlow

        monkeypatch.setattr(VI, "AutopilotFlow", partial(AutopilotFlow, more_target=20))

    def _run(self, spec):
        from tests_lib.detectors.test_startup_schedule import _seeded_dataset
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        medias, seed_scores = _seeded_dataset(n_pos=20)
        picks: list = []
        simulate_voting_iterations(
            medias,
            target_category="target",
            seed=3,
            dataset_name="stub",
            max_steps=30,
            seed_scores=seed_scores,
            atlas_min_node_size=8,
            opening_diversity=spec,
            pick_sink=picks,
        )
        return picks

    def test_the_knob_is_inert_without_a_walk(self):
        """Today's opening takes its Goods before any Bad, so there is no Bad for the pass to steer by (#4740)."""
        plain, diverse = self._run(None), self._run("0.3/1")
        assert [p["picked_id"] for p in plain] == [p["picked_id"] for p in diverse]

    def test_the_knob_reaches_the_opening_walk(self, walk):
        """A loose tau on the synthetic pool moves a pick inside the opening's walk."""
        plain, diverse = self._run(None), self._run("0.3/1")
        diffs = [(a["phase"], a["picked_id"], b["picked_id"]) for a, b in zip(plain, diverse)]
        first = next(d for d in diffs if d[1] != d[2])
        assert first[0] in ("good", "more")
