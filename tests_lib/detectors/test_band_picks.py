"""Issue #4482's band picks: one in N of Autopilot's picks past the opening, drawn the way the spot check draws.

An experiment knob, not app behaviour: off by default, so the default arm is the
app's own pick for pick.
"""

from __future__ import annotations

import numpy as np
import pytest

from vtscore.eval.al_strategies import band_pick
from vtscore.training.thresholds.spot_check import band_edges


def _pool(n: int = 100) -> tuple[list[int], dict[int, float]]:
    """Ids 0..n-1 scored so that id i ranks i-th (the highest score first)."""
    ids = list(range(n))
    return ids[::-1], {i: 1.0 - i / n for i in ids}


def _pick(ids: list[int], scores: dict[int, float], candidate: int, k: int, rng: np.random.Generator) -> int:
    """:func:`band_pick` on a pool it can always draw from."""
    got = band_pick(ids, scores, candidate, k, rng)
    assert got is not None
    return got


class TestBandPick:
    def test_cycles_through_the_bands_that_hold_the_cap(self):
        """At the top-32 cap the bands are ranks [0, 8), [8, 16), [16, 32); the picks visit them in turn."""
        ids, scores = _pool()
        rng = np.random.default_rng(0)
        got = [_pick(ids, scores, 32, k, rng) for k in range(6)]
        assert [0 <= got[0] < 8, 8 <= got[1] < 16, 16 <= got[2] < 32] == [True] * 3
        assert [0 <= got[3] < 8, 8 <= got[4] < 16, 16 <= got[5] < 32] == [True] * 3

    def test_the_recall_cap_reaches_deeper_bands(self):
        ids, scores = _pool(300)
        rng = np.random.default_rng(1)
        edges = band_edges(300)
        got = [_pick(ids, scores, 128, k, rng) for k in range(5)]
        for k, g in enumerate(got):
            assert edges[k] <= g < edges[k + 1]

    def test_uniform_within_the_band(self):
        ids, scores = _pool()
        rng = np.random.default_rng(2)
        draws = [_pick(ids, scores, 32, 2, rng) for _ in range(3000)]
        counts = np.bincount(draws, minlength=32)[16:32]
        assert counts.min() > 0.6 * counts.mean()

    def test_only_scored_items_and_none_when_nothing_is_scored(self):
        ids, scores = _pool(20)
        assert band_pick([999, 1000], scores, 32, 0, np.random.default_rng(0)) is None
        assert band_pick(ids, scores, 32, 0, np.random.default_rng(0)) in set(range(8))

    def test_a_short_ranking_is_one_band(self):
        ids, scores = _pool(5)
        assert band_pick(ids, scores, 32, 4, np.random.default_rng(0)) in set(range(5))


class TestInTheHarness:
    def _run(self, band_share, **kw):
        from tests_lib.detectors.test_startup_schedule import _seeded_dataset
        from vtscore.eval.voting_iterations import simulate_voting_iterations

        medias, seed_scores = _seeded_dataset(n_pos=40)
        picks: list = []
        simulate_voting_iterations(
            medias,
            target_category="target",
            seed=3,
            dataset_name="stub",
            max_steps=40,
            seed_scores=seed_scores,
            atlas_min_node_size=8,
            spot_check="off",
            band_share=band_share,
            pick_sink=picks,
            **kw,
        )
        return picks

    def test_off_by_default(self):
        assert not any(p["phase"] == "band" for p in self._run(None))

    def test_every_nth_pick_past_the_opening_is_a_band_pick(self):
        picks = self._run(3)
        past = [p for p in picks if p["phase"] not in ("good", "bad", "more")]
        assert past, "the stub run never left the opening"
        assert [p["phase"] == "band" for p in past] == [(i + 1) % 3 == 0 for i in range(len(past))]

    def test_the_opening_is_untouched(self):
        plain, banded = self._run(None), self._run(2)
        n = next(i for i, p in enumerate(plain) if p["phase"] not in ("good", "bad", "more"))
        assert [p["picked_id"] for p in plain[:n]] == [p["picked_id"] for p in banded[:n]]

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "8", True])
    def test_refuses_what_it_cannot_run(self, bad):
        with pytest.raises(ValueError):
            self._run(bad)

    def test_needs_a_balance(self):
        with pytest.raises(ValueError):
            self._run(4, beta="off")
