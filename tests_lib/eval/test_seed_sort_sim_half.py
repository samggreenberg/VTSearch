"""The simulated session walks its seed sort over the voting half alone (#5004).

The app sorts the dataset it has loaded, which ``simulate_voting_iterations``
models with the sim half.  So the Bad phase's cut, the rank distances the pick
on the sort measures, a schedule's cuts and the pick log's ranks are all read
over that half, and the withheld half's cosines, whatever they are, cannot move
a pick.  The typed query's Test still reads them: it searches the withheld half.
"""

from __future__ import annotations

import numpy as np
import pytest

import vtscore.eval.voting_iterations as vi
from vtscore.detectors.label_quota import TIER_QUERY
from vtscore.eval.example_opening import EXAMPLE_PHASE
from vtscore.eval.startup_schedule import parse_startup_schedule, round_cut
from vtscore.eval.voting_iterations import simulate_voting_iterations

DIM = 16
SEED = 3
#: Far outside a cosine's range, so the withheld half would dominate any cut or rank it reached.
EXTREME = 10.0
#: Every cut kind a round reads off the sort: ``top``, the GMM midpoint, a rank quantile.
SCHEDULE = "g3@top,b4@mid,n6@q0.5"


def _world(n_pos: int = 40, n_neg: int = 120, seed: int = 0) -> tuple[dict[int, dict], dict[int, float]]:
    """A single-vector dataset and a typed query's cosines that rank the target near the top."""
    rng = np.random.default_rng(seed)
    centre = rng.standard_normal(DIM).astype(np.float32)
    query = centre + 0.6 * rng.standard_normal(DIM).astype(np.float32)
    query /= np.linalg.norm(query)
    medias: dict[int, dict] = {}
    scores: dict[int, float] = {}
    for i in range(n_pos + n_neg):
        positive = i < n_pos
        vec = (centre if positive else 0.0) + rng.standard_normal(DIM).astype(np.float32)
        vec = (vec / np.linalg.norm(vec)).astype(np.float32)
        medias[i + 1] = {"id": i + 1, "category": "target" if positive else "other", "embeddings": {"stub": vec}}
        scores[i + 1] = float(vec @ query)
    return medias, scores


def _run(seed_scores, *, max_steps: int = 18, **kwargs):
    """One session, with every ``ALContext`` the selector was handed (its phase, seed-sort ids and pool)."""
    medias, _ = _world()
    contexts: list[tuple] = []
    real = vi.select_next

    def spy(strategy, ctx):
        keys = None if ctx.seed_scores is None else frozenset(ctx.seed_scores)
        contexts.append((ctx.phase, keys, frozenset(ctx.pool_ids)))
        return real(strategy, ctx)

    picks: list[dict] = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(vi, "select_next", spy)
        rows = simulate_voting_iterations(
            medias,
            "target",
            seed=SEED,
            max_steps=max_steps,
            calibrate_count=2,
            spot_check="off",
            pick_sink=picks,
            seed_scores=seed_scores,
            **kwargs,
        )
    return rows, picks, contexts


def _halves(contexts) -> tuple[frozenset, frozenset]:
    """The run's voting and withheld halves, read off the first pick's pool (no vote cast yet)."""
    medias, _ = _world()
    sim = contexts[0][2]
    return sim, frozenset(medias) - sim


def _extreme(scores: dict[int, float], test: frozenset) -> dict[int, float]:
    """*scores* with the withheld half pushed to the extremes, by truth: its typed-query set is then exact."""
    medias, _ = _world()
    out = dict(scores)
    for cid in test:
        out[cid] = EXTREME if medias[cid]["category"] == "target" else -EXTREME
    return out


@pytest.fixture(scope="module")
def runs():
    """The same session on the plain cosines and with the withheld half's pushed to the extremes."""
    _medias, scores = _world()
    plain = _run(scores, query_hold=0)
    _sim, test = _halves(plain[2])
    return plain, _run(_extreme(scores, test), query_hold=0)


@pytest.fixture(scope="module")
def schedule_runs():
    _medias, scores = _world()
    plain = _run(scores, startup_schedule=SCHEDULE, max_steps=24)
    _sim, test = _halves(plain[2])
    return plain, _run(_extreme(scores, test), startup_schedule=SCHEDULE, max_steps=24)


def _columns(picks, cols):
    """*cols* of every pick, NaN read as ``None`` so two logs compare by value."""
    return [tuple(None if isinstance(p[c], float) and np.isnan(p[c]) else p[c] for c in cols) for p in picks]


def _seed_columns(picks):
    return _columns(
        picks, ("t", "phase", "picked_id", "picked_seed_rank", "picked_seed_percentile", "picked_seed_score")
    )


class TestTheSelectorSortsTheVotingHalf:
    def test_every_context_holds_only_sim_ids(self, runs):
        (_rows, _picks, contexts), (_xrows, _xpicks, xcontexts) = runs
        sim, test = _halves(contexts)
        assert sim and test
        for _phase, keys, _pool in contexts + xcontexts:
            assert keys == sim

    def test_the_bad_phase_pick_ignores_the_withheld_cosines(self, runs):
        (_rows, picks, _contexts), (_xrows, xpicks, _xcontexts) = runs
        bad = [p["picked_id"] for p in picks if p["phase"] == "bad"]
        assert bad, "the session reaches the Bad phase"
        assert [p["picked_id"] for p in xpicks if p["phase"] == "bad"] == bad
        assert [(p["t"], p["picked_id"]) for p in xpicks] == [(p["t"], p["picked_id"]) for p in picks]

    def test_the_pick_log_ranks_the_voting_half(self, runs):
        (_rows, picks, contexts), (_xrows, xpicks, _xcontexts) = runs
        sim, _test = _halves(contexts)
        assert _seed_columns(xpicks) == _seed_columns(picks)
        assert max(p["picked_seed_rank"] for p in picks) < len(sim)
        assert all(abs(p["picked_seed_score"]) <= 1.0 for p in xpicks)


class TestTheTypedQueryTestReadsTheWithheldHalf:
    def test_the_query_rows_use_the_withheld_cosines(self, runs):
        (rows, _picks, _contexts), (xrows, _xpicks, _xcontexts) = runs
        query = [r for r in rows if r["detector_tier"] == TIER_QUERY]
        xquery = [r for r in xrows if r["detector_tier"] == TIER_QUERY]
        assert query and len(xquery) == len(query)
        # At the extremes the withheld half splits exactly on its truth.
        assert all(r["precision"] == 1.0 and r["recall"] == 1.0 for r in xquery)
        assert any(r["precision"] < 1.0 or r["recall"] < 1.0 for r in query)


class TestAScheduleCutsTheVotingHalf:
    def test_the_cuts_and_the_picks_ignore_the_withheld_cosines(self, schedule_runs):
        (_rows, picks, _contexts), (_xrows, xpicks, _xcontexts) = schedule_runs
        cols = ("t", "startup_round", "startup_cut", "startup_cut_percentile", "picked_id", "picked_seed_rank")
        assert _columns(xpicks, cols) == _columns(picks, cols)

    def test_each_rounds_cut_is_drawn_over_the_voting_half(self, schedule_runs):
        (_rows, picks, contexts), _extreme_run = schedule_runs
        sim, _test = _halves(contexts)
        _medias, scores = _world()
        cuts = [round_cut([scores[cid] for cid in sim], rnd) for rnd in parse_startup_schedule(SCHEDULE)]
        seen = {p["startup_round"]: p["startup_cut"] for p in picks if p["startup_round"] >= 0}
        assert set(seen) == {0, 1, 2}, "the session walks every round"
        for rnd, cut in seen.items():
            assert cut == pytest.approx(cuts[rnd], abs=1e-6) or (np.isinf(cut) and np.isinf(cuts[rnd]))


def test_an_example_opening_sorts_the_voting_half():
    _rows, picks, contexts = _run(None, seed_examples=2)
    examples = {p["picked_id"] for p in picks if p["phase"] == EXAMPLE_PHASE}
    assert len(examples) == 2
    sim = contexts[0][2] | examples
    medias, _ = _world()
    assert frozenset(medias) - sim
    for _phase, keys, _pool in contexts:
        assert keys == sim
