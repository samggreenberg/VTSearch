"""Tests for the autopilot vote-order strategy.

All tests build small, controlled :class:`ALContext` states so the phase logic
is checked directly - no model downloads.  ``model`` is a bare sentinel object
wherever the selector only needs "a detector exists" (it never runs the model;
it reads ``scores``), so these stay hermetic.
"""

import numpy as np
import pytest

from vtscore.eval.al_strategies import (
    STRATEGIES,
    ALContext,
    available_strategies,
    select_next,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _ctx(
    pool_ids,
    embeddings=None,
    *,
    labeled=None,
    scores=None,
    model=None,
    threshold=0.5,
    atlas=None,
    pool_labels=None,
    seed_scores=None,
    seed=0,
):
    embeddings = embeddings if embeddings is not None else {i: np.zeros(4, np.float32) for i in pool_ids}
    return ALContext(
        pool_ids=list(pool_ids),
        embeddings=embeddings,
        labeled=labeled or {},
        scores=scores or {},
        model=model,
        threshold=threshold,
        atlas=atlas,
        rng=np.random.RandomState(seed),
        pool_labels=pool_labels,
        seed_scores=seed_scores,
    )


def _labels(n_good, n_bad, start=1000):
    """A ``labeled`` dict with *n_good* good and *n_bad* bad votes on out-of-pool ids."""
    labeled = {}
    cid = start
    for _ in range(n_good):
        labeled[cid] = 1.0
        cid += 1
    for _ in range(n_bad):
        labeled[cid] = 0.0
        cid += 1
    return labeled


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------


class TestRegistry:
    def test_autopilot_is_the_only_strategy(self):
        assert set(STRATEGIES) == {"autopilot"}
        assert available_strategies() == ["autopilot"]

    def test_random_and_academic_strategies_are_gone(self):
        for name in ["random", "margin", "entropy", "bald", "eig", "coreset", "balanced", "ensemble_std"]:
            assert name not in STRATEGIES

    def test_unknown_strategy_raises(self):
        with pytest.raises(KeyError):
            select_next("random", _ctx([1, 2]))

    def test_empty_pool_raises(self):
        with pytest.raises(ValueError):
            select_next("autopilot", _ctx([]))


# ------------------------------------------------------------------
# Seed / Good phase
# ------------------------------------------------------------------


class TestGoodSeedPhase:
    def test_no_text_seeds_a_ground_truth_positive(self):
        # Fewer than 3 goods, no text sort -> draw a random known-good example.
        ctx = _ctx([1, 2, 3, 4], pool_labels={1: 0.0, 2: 1.0, 3: 0.0, 4: 1.0}, labeled={})
        assert select_next("autopilot", ctx) in {2, 4}

    def test_no_text_seed_is_seed_deterministic(self):
        pl = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}
        a = select_next("autopilot", _ctx(list(pl), pool_labels=pl, seed=7))
        b = select_next("autopilot", _ctx(list(pl), pool_labels=pl, seed=7))
        assert a == b

    def test_no_text_no_pool_labels_falls_back_to_any_pick(self):
        ctx = _ctx([1, 2, 3], pool_labels=None, labeled={})
        assert select_next("autopilot", ctx) in {1, 2, 3}

    def test_text_seed_picks_top_of_the_ranking(self):
        # Text sort available: the top-ranked item is the most-likely good.
        ctx = _ctx([1, 2, 3, 4], seed_scores={1: 0.1, 2: 0.9, 3: 0.5, 4: 0.2}, labeled={})
        assert select_next("autopilot", ctx) == 2


# ------------------------------------------------------------------
# Bad phase
# ------------------------------------------------------------------


class TestBadPhase:
    def test_with_detector_picks_lowest_scored(self):
        # 3 goods, 0 bads -> bad phase.  A detector exists, so its lowest-scored
        # item is the likely bad.
        ctx = _ctx(
            [1, 2, 3],
            labeled=_labels(3, 0),
            scores={1: 0.9, 2: 0.1, 3: 0.5},
            model=object(),
        )
        assert select_next("autopilot", ctx) == 2

    def test_text_no_detector_picks_bottom_of_ranking(self):
        # 3 goods, 0 bads, still no detector, but text sort exists -> the bottom
        # of the ranking (least similar to the query) is the likely bad.
        ctx = _ctx([1, 2, 3], labeled=_labels(3, 0), seed_scores={1: 0.9, 2: 0.1, 3: 0.5})
        assert select_next("autopilot", ctx) == 2

    def test_no_text_no_detector_picks_least_similar_to_goods(self):
        # Good centroid points along +e0; the pool item pointing along -e0 is
        # least similar and gets the Bad vote.
        e_pos = np.array([1.0, 0.0], np.float32)
        e_neg = np.array([-1.0, 0.0], np.float32)
        embeddings = {
            1: np.array([0.9, 0.1], np.float32),  # near the goods
            2: e_neg,  # opposite the goods
            10: e_pos,
            11: e_pos,
            12: e_pos,
        }
        ctx = _ctx([1, 2], embeddings=embeddings, labeled={10: 1.0, 11: 1.0, 12: 1.0})
        assert select_next("autopilot", ctx) == 2


# ------------------------------------------------------------------
# Hard / New interleave
# ------------------------------------------------------------------


def _two_cluster_atlas():
    """A coverage atlas over two well-separated clusters (labels start empty)."""
    from vtscore.state.coverage_atlas import CoverageAtlas

    rng = np.random.default_rng(0)
    vectors = {}
    vid = 1
    for _ in range(8):
        v = np.zeros(8, np.float32)
        v[0] = 1.0
        vectors[vid] = (v + rng.standard_normal(8).astype(np.float32) * 0.02).astype(np.float32)
        vid += 1
    for _ in range(8):
        v = np.zeros(8, np.float32)
        v[0] = -1.0
        vectors[vid] = (v + rng.standard_normal(8).astype(np.float32) * 0.02).astype(np.float32)
        vid += 1
    return CoverageAtlas(vectors, k=2, max_depth=4, min_node_size=3), vectors


class TestHardNewInterleave:
    def test_hard_picks_item_nearest_threshold(self):
        # Even total past the quorum, atlas absent -> Hard (margin): the score
        # closest to the threshold wins.
        ctx = _ctx(
            [1, 2, 3],
            labeled=_labels(3, 5),  # total 8, even
            scores={1: 0.05, 2: 0.55, 3: 0.95},
            model=object(),
            threshold=0.5,
            atlas=None,
        )
        assert select_next("autopilot", ctx) == 2

    def test_odd_step_uses_the_coverage_atlas(self):
        atlas, vectors = _two_cluster_atlas()
        pool = list(vectors)
        scores = {cid: 0.5 for cid in pool}
        ctx = _ctx(
            pool,
            embeddings=vectors,
            labeled=_labels(3, 4),  # total 7, odd -> New phase
            scores=scores,
            model=object(),
            atlas=atlas,
        )
        expected = atlas.next_sample(scores, 0.5)
        assert expected in pool
        assert select_next("autopilot", ctx) == expected

    def test_new_falls_back_to_hard_when_atlas_exhausted(self):
        # An atlas whose every node already carries evidence returns no sample,
        # so an odd step falls back to the Hard (nearest-threshold) pick.
        atlas, vectors = _two_cluster_atlas()
        for cid in vectors:
            atlas.label(cid, good=True)
        pool = [1, 2, 3]
        ctx = _ctx(
            pool,
            embeddings=vectors,
            labeled=_labels(3, 4),  # odd
            scores={1: 0.05, 2: 0.52, 3: 0.95},
            model=object(),
            threshold=0.5,
            atlas=atlas,
        )
        assert select_next("autopilot", ctx) == 2


# ------------------------------------------------------------------
# Selection always yields a valid id
# ------------------------------------------------------------------


class TestSelectionValidity:
    @pytest.mark.parametrize(
        "labeled",
        [
            {},  # good phase
            _labels(3, 0),  # bad phase
            _labels(3, 5),  # hard/new phase
        ],
    )
    def test_returns_a_pool_id(self, labeled):
        ctx = _ctx(
            [1, 2, 3],
            labeled=labeled,
            pool_labels={1: 1.0, 2: 0.0, 3: 1.0},
            scores={1: 0.3, 2: 0.6, 3: 0.9},
            model=object(),
        )
        assert select_next("autopilot", ctx) in {1, 2, 3}
