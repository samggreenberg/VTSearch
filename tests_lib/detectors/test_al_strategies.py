"""Tests for the active-learning acquisition strategies.

All tests build small, controlled :class:`ALContext` states so the acquisition
maths is checked directly - no model downloads.  The model-driven samplers
(``bald``, ``eig``) train a tiny MLP on a handful of separable points.
"""

import numpy as np
import pytest

from vtscore.eval.al_strategies import (
    STRATEGIES,
    ALContext,
    _score_coreset,
    _score_entropy,
    _score_margin,
    acquisition_utilities,
    available_strategies,
    density_weights,
    select_next,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _ctx(
    pool_ids,
    embeddings,
    *,
    labeled=None,
    scores=None,
    model=None,
    threshold=0.5,
    atlas=None,
    seed=0,
    input_dim=None,
    hidden_dim=8,
):
    dim = input_dim if input_dim is not None else (len(next(iter(embeddings.values()))) if embeddings else 4)
    return ALContext(
        pool_ids=list(pool_ids),
        embeddings=embeddings,
        labeled=labeled or {},
        scores=scores or {},
        model=model,
        threshold=threshold,
        input_dim=dim,
        hidden_dim=hidden_dim,
        atlas=atlas,
        rng=np.random.RandomState(seed),
    )


def _trained_model(dim=8, seed=0):
    """Train a tiny separable MLP; return ``(model, pos_vec, neg_vec)``."""
    import torch

    from vtscore.training.mlp import train_model

    rng = np.random.default_rng(seed)
    pos = np.zeros(dim, dtype=np.float32)
    pos[0] = 1.0
    neg = np.zeros(dim, dtype=np.float32)
    neg[0] = -1.0
    X_rows = []
    y_rows = []
    for _ in range(6):
        X_rows.append(pos + rng.standard_normal(dim).astype(np.float32) * 0.05)
        y_rows.append(1.0)
        X_rows.append(neg + rng.standard_normal(dim).astype(np.float32) * 0.05)
        y_rows.append(0.0)
    X = torch.tensor(np.array(X_rows), dtype=torch.float32)
    y = torch.tensor(y_rows, dtype=torch.float32).unsqueeze(1)
    model = train_model(X, y, dim, hidden_dim=8)
    return model, pos, neg


def _score_pool(model, pool_ids, embeddings):
    import torch

    X = torch.tensor(np.array([embeddings[i] for i in pool_ids]), dtype=torch.float32)
    with torch.no_grad():
        s = torch.sigmoid(model(X)).squeeze(1).cpu().tolist()
    return dict(zip(pool_ids, s, strict=True))


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------


class TestRegistry:
    def test_base_strategies_present(self):
        for name in ["random", "margin", "entropy", "bald", "eig", "coreset"]:
            assert name in STRATEGIES

    def test_density_variants_present(self):
        for name in ["density_margin", "density_entropy", "density_bald", "density_coreset"]:
            assert name in STRATEGIES

    def test_random_and_eig_have_no_density_variant(self):
        assert "density_random" not in STRATEGIES
        assert "density_eig" not in STRATEGIES

    def test_available_strategies_sorted_and_complete(self):
        names = available_strategies()
        assert names == sorted(names)
        assert set(names) == set(STRATEGIES)


# ------------------------------------------------------------------
# Pointwise scorers (no model needed)
# ------------------------------------------------------------------


class TestMargin:
    def test_picks_score_closest_to_threshold(self):
        emb = {
            1: np.array([1.0, 0.0], np.float32),
            2: np.array([0.0, 1.0], np.float32),
            3: np.array([1.0, 1.0], np.float32),
        }
        # threshold 0.5: id 2's score (0.55) is closest.
        scores = {1: 0.05, 2: 0.55, 3: 0.95}
        ctx = _ctx([1, 2, 3], emb, scores=scores, threshold=0.5)
        util = _score_margin(ctx)
        assert ctx.pool_ids[int(np.argmax(util))] == 2
        assert np.all(util >= 0.0)  # non-negative for the density multiply

    def test_respects_non_half_threshold(self):
        emb = {1: np.zeros(2, np.float32), 2: np.zeros(2, np.float32)}
        scores = {1: 0.5, 2: 0.85}
        ctx = _ctx([1, 2], emb, scores=scores, threshold=0.9)
        # id 2 (0.85) sits nearest the 0.9 threshold.
        util = _score_margin(ctx)
        assert ctx.pool_ids[int(np.argmax(util))] == 2


class TestEntropy:
    def test_picks_most_uncertain(self):
        emb = {1: np.zeros(2, np.float32), 2: np.zeros(2, np.float32), 3: np.zeros(2, np.float32)}
        scores = {1: 0.5, 2: 0.99, 3: 0.01}
        ctx = _ctx([1, 2, 3], emb, scores=scores)
        util = _score_entropy(ctx)
        assert ctx.pool_ids[int(np.argmax(util))] == 1
        assert np.all(util >= 0.0)


class TestCoreset:
    def test_picks_farthest_from_labeled(self):
        emb = {
            1: np.array([0.1, 0.0], np.float32),  # near the labeled point
            2: np.array([5.0, 0.0], np.float32),  # far
            10: np.array([0.0, 0.0], np.float32),  # labeled
        }
        ctx = _ctx([1, 2], emb, labeled={10: 0.0})
        util = _score_coreset(ctx)
        assert ctx.pool_ids[int(np.argmax(util))] == 2

    def test_falls_back_to_random_without_labels(self):
        emb = {1: np.zeros(2, np.float32), 2: np.ones(2, np.float32)}
        ctx = _ctx([1, 2], emb, labeled={})
        util = _score_coreset(ctx)
        assert util.shape == (2,)
        assert np.all(np.isfinite(util))


# ------------------------------------------------------------------
# Model-driven scorers
# ------------------------------------------------------------------


class TestBald:
    def test_mutual_information_non_negative_and_finite(self):
        from vtscore.eval.al_strategies import _score_bald

        model, pos, neg = _trained_model()
        emb = {1: pos, 2: neg, 3: (pos + neg) / 2.0}
        scores = _score_pool(model, [1, 2, 3], emb)
        ctx = _ctx([1, 2, 3], emb, scores=scores, model=model, seed=1)
        util = _score_bald(ctx)
        assert util.shape == (3,)
        assert np.all(util >= 0.0)
        assert np.all(np.isfinite(util))

    def test_select_returns_valid_pool_id(self):
        model, pos, neg = _trained_model()
        emb = {1: pos, 2: neg, 3: (pos + neg) / 2.0}
        scores = _score_pool(model, [1, 2, 3], emb)
        ctx = _ctx([1, 2, 3], emb, scores=scores, model=model, seed=2)
        assert select_next("bald", ctx) in {1, 2, 3}


class TestEig:
    def test_utilities_finite_for_small_pool(self):
        from vtscore.eval.al_strategies import _score_eig

        model, pos, neg = _trained_model()
        emb = {1: pos, 2: neg, 3: (pos + neg) / 2.0, 10: pos, 11: neg}
        scores = _score_pool(model, [1, 2, 3], emb)
        ctx = _ctx(
            [1, 2, 3],
            emb,
            scores=scores,
            model=model,
            labeled={10: 1.0, 11: 0.0},
            seed=0,
        )
        util = _score_eig(ctx)
        assert util.shape == (3,)
        # Pool <= _EIG_MAX_CANDIDATES, so every candidate is evaluated (finite).
        assert np.all(np.isfinite(util))
        assert select_next("eig", ctx) in {1, 2, 3}


# ------------------------------------------------------------------
# Density weighting
# ------------------------------------------------------------------


def _clustered_atlas(min_node_size=3):
    """Build a coverage atlas with two well-separated clusters of unequal size."""
    from vtscore.state.coverage_atlas import CoverageAtlas

    rng = np.random.default_rng(0)
    vectors = {}
    vid = 1
    # Big cluster around +e0 (15 pts), small cluster around -e0 (4 pts).
    for _ in range(15):
        v = np.zeros(8, np.float32)
        v[0] = 1.0
        vectors[vid] = (v + rng.standard_normal(8).astype(np.float32) * 0.02).astype(np.float32)
        vid += 1
    for _ in range(4):
        v = np.zeros(8, np.float32)
        v[0] = -1.0
        vectors[vid] = (v + rng.standard_normal(8).astype(np.float32) * 0.02).astype(np.float32)
        vid += 1
    atlas = CoverageAtlas(vectors, k=2, max_depth=4, min_node_size=min_node_size)
    return atlas, vectors


class TestDensityWeights:
    def test_weight_is_inverse_sqrt_cell_count(self):
        atlas, vectors = _clustered_atlas()
        ctx = _ctx(list(vectors), vectors, atlas=atlas)
        weights = density_weights(ctx)
        for i, cid in enumerate(ctx.pool_ids):
            leaf = atlas.vector_to_leaf[cid]
            expected = 1.0 / np.sqrt(atlas.nodes[leaf]["n"])
            assert weights[i] == pytest.approx(expected)

    def test_no_atlas_gives_unit_weights(self):
        emb = {1: np.zeros(4, np.float32), 2: np.ones(4, np.float32)}
        ctx = _ctx([1, 2], emb, atlas=None)
        assert np.array_equal(density_weights(ctx), np.ones(2))

    def test_density_variant_reweights_base_utility(self):
        atlas, vectors = _clustered_atlas()
        pool = list(vectors)
        scores = {cid: 0.5 for cid in pool}
        # Non-None model just satisfies the needs-model gate; margin reads scores.
        ctx = _ctx(pool, vectors, scores=scores, model=object(), atlas=atlas)
        base = acquisition_utilities("margin", ctx)
        dens = acquisition_utilities("density_margin", ctx)
        np.testing.assert_allclose(dens, base * density_weights(ctx))


# ------------------------------------------------------------------
# Selection semantics / cold start
# ------------------------------------------------------------------


class TestSelection:
    @pytest.mark.parametrize("strategy", available_strategies())
    def test_cold_start_falls_back_to_random(self, strategy):
        """With no model yet, every strategy still returns a valid pool id."""
        emb = {1: np.zeros(4, np.float32), 2: np.ones(4, np.float32), 3: np.full(4, 2.0, np.float32)}
        ctx = _ctx([1, 2, 3], emb, model=None, atlas=None, seed=5)
        assert select_next(strategy, ctx) in {1, 2, 3}

    def test_empty_pool_raises(self):
        ctx = _ctx([], {}, model=None)
        with pytest.raises(ValueError):
            select_next("random", ctx)

    def test_unknown_strategy_raises(self):
        emb = {1: np.zeros(4, np.float32)}
        ctx = _ctx([1], emb)
        with pytest.raises(KeyError):
            select_next("nope", ctx)

    def test_random_is_seed_deterministic(self):
        emb = {i: np.full(4, float(i), np.float32) for i in range(1, 8)}
        a = select_next("random", _ctx(list(emb), emb, seed=3))
        b = select_next("random", _ctx(list(emb), emb, seed=3))
        assert a == b
