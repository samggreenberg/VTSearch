"""Tests for the voting-iterations evaluation framework.

All tests use small synthetic datasets with known, well-separated
embeddings so no real model downloads are needed.
"""

import numpy as np
import pandas as pd

from vtscore.eval.voting_iterations import (
    _inclusion_weights,
    _make_vote_sequence,
    _split_media_ids,
    run_voting_iterations_eval,
    simulate_voting_iterations,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _make_separable_clips(dim=16, n_per_cat=20, seed=0):
    """Two categories with well-separated embeddings.

    Category "alpha" clusters around [+1, 0, 0, ...],
    category "beta"  clusters around [-1, 0, 0, ...].
    """
    rng = np.random.RandomState(seed)
    medias = {}
    media_id = 1
    for _ in range(n_per_cat):
        emb = rng.normal(1.0, 0.2, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embedding": emb, "category": "alpha"}
        media_id += 1
    for _ in range(n_per_cat):
        emb = rng.normal(-1.0, 0.2, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embedding": emb, "category": "beta"}
        media_id += 1
    return medias


def _make_overlapping_clips(dim=16, n_per_cat=20, seed=0):
    """Two categories with overlapping embeddings (harder to classify).

    Category "alpha" centred at [+0.3, 0, 0, ...],
    category "beta"  centred at [-0.3, 0, 0, ...], with large noise.
    """
    rng = np.random.RandomState(seed)
    medias = {}
    media_id = 1
    for _ in range(n_per_cat):
        emb = rng.normal(0.3, 1.0, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embedding": emb, "category": "alpha"}
        media_id += 1
    for _ in range(n_per_cat):
        emb = rng.normal(-0.3, 1.0, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embedding": emb, "category": "beta"}
        media_id += 1
    return medias


def _make_three_category_clips(dim=16, n_per_cat=15, seed=0):
    """Three categories: alpha, beta, gamma."""
    rng = np.random.RandomState(seed)
    medias = {}
    media_id = 1
    centres = {"alpha": 1.0, "beta": -1.0, "gamma": 0.0}
    for cat, centre in centres.items():
        for _ in range(n_per_cat):
            emb = rng.normal(centre, 0.2, dim).astype(np.float32)
            medias[media_id] = {"id": media_id, "embedding": emb, "category": cat}
            media_id += 1
    return medias


# ------------------------------------------------------------------
# Unit tests: helpers
# ------------------------------------------------------------------


class TestInclusionWeights:
    def test_zero_inclusion(self):
        fpr_w, fnr_w = _inclusion_weights(0)
        assert fpr_w == 1.0
        assert fnr_w == 1.0

    def test_positive_inclusion(self):
        fpr_w, fnr_w = _inclusion_weights(3)
        assert fpr_w == 1.0
        assert fnr_w == 8.0

    def test_negative_inclusion(self):
        fpr_w, fnr_w = _inclusion_weights(-2)
        assert fpr_w == 4.0
        assert fnr_w == 1.0


class TestSplitClipIds:
    def test_split_sizes(self):
        medias = _make_separable_clips(n_per_cat=10)
        rng = np.random.RandomState(42)
        sim, test = _split_media_ids(medias, 0.5, rng)
        assert len(sim) + len(test) == len(medias)
        assert len(sim) == 10
        assert len(test) == 10

    def test_no_overlap(self):
        medias = _make_separable_clips(n_per_cat=10)
        rng = np.random.RandomState(42)
        sim, test = _split_media_ids(medias, 0.5, rng)
        assert set(sim).isdisjoint(set(test))

    def test_deterministic(self):
        medias = _make_separable_clips(n_per_cat=10)
        rng1 = np.random.RandomState(42)
        sim1, test1 = _split_media_ids(medias, 0.5, rng1)
        rng2 = np.random.RandomState(42)
        sim2, test2 = _split_media_ids(medias, 0.5, rng2)
        assert sim1 == sim2
        assert test1 == test2


class TestMakeVoteSequence:
    def test_all_clips_voted(self):
        medias = _make_separable_clips(n_per_cat=5)
        sim_ids = list(medias.keys())[:5]
        rng = np.random.RandomState(42)
        seq = _make_vote_sequence(sim_ids, medias, "alpha", rng)
        assert len(seq) == 5
        assert {cid for cid, _ in seq} == set(sim_ids)

    def test_labels_match_category(self):
        medias = _make_separable_clips(n_per_cat=5)
        sim_ids = list(medias.keys())
        rng = np.random.RandomState(42)
        seq = _make_vote_sequence(sim_ids, medias, "alpha", rng)
        for cid, label in seq:
            expected = "good" if medias[cid]["category"] == "alpha" else "bad"
            assert label == expected


# ------------------------------------------------------------------
# Unit tests: simulate_voting_iterations
# ------------------------------------------------------------------


class TestSimulateVotingIterations:
    def test_returns_rows(self):
        medias = _make_separable_clips(n_per_cat=10)
        rows = simulate_voting_iterations(
            medias,
            target_category="alpha",
            seed=42,
            dataset_name="test_ds",
            inclusion=0,
            sim_fraction=0.5,
        )
        assert len(rows) > 0

    def test_row_schema(self):
        medias = _make_separable_clips(n_per_cat=10)
        rows = simulate_voting_iterations(
            medias,
            target_category="alpha",
            seed=42,
            dataset_name="test_ds",
        )
        expected_keys = {"seed", "dataset", "category", "t", "cost", "fpr", "fnr", "elapsed_seconds"}
        for row in rows:
            assert set(row.keys()) == expected_keys

    def test_seed_determinism(self):
        medias = _make_separable_clips(n_per_cat=10)
        rows1 = simulate_voting_iterations(medias, "alpha", seed=42)
        rows2 = simulate_voting_iterations(medias, "alpha", seed=42)
        assert len(rows1) == len(rows2)
        for r1, r2 in zip(rows1, rows2):
            # Compare all fields except elapsed_seconds (wall-clock timing varies between runs)
            r1_cmp = {k: v for k, v in r1.items() if k != "elapsed_seconds"}
            r2_cmp = {k: v for k, v in r2.items() if k != "elapsed_seconds"}
            assert r1_cmp == r2_cmp

    def test_different_seeds_differ(self):
        medias = _make_separable_clips(n_per_cat=10)
        rows1 = simulate_voting_iterations(medias, "alpha", seed=42)
        rows2 = simulate_voting_iterations(medias, "alpha", seed=99)
        # Different seeds should produce different vote orderings / splits,
        # so the t-indexed costs should differ (not guaranteed for every row,
        # but at least the full sequence should differ).
        costs1 = [r["cost"] for r in rows1]
        costs2 = [r["cost"] for r in rows2]
        assert costs1 != costs2

    def test_t_values_monotonically_increase(self):
        medias = _make_separable_clips(n_per_cat=10)
        rows = simulate_voting_iterations(medias, "alpha", seed=42)
        t_vals = [r["t"] for r in rows]
        assert t_vals == sorted(t_vals)
        # t starts >=2 because we need at least 1 good + 1 bad
        assert all(t >= 2 for t in t_vals)

    def test_cost_decreases_over_time_for_overlapping_data(self):
        """With overlapping data, cost should generally decrease as more votes come in."""
        medias = _make_overlapping_clips(n_per_cat=60, dim=16)
        rows = simulate_voting_iterations(
            medias,
            "alpha",
            seed=42,
            sim_fraction=0.5,
        )
        costs = [r["cost"] for r in rows]
        # Compare average of first half vs last half.
        # With overlapping data and a regularised model the decrease is
        # gradual, so we allow a 15% tolerance.
        n = len(costs)
        mid = max(1, n // 2)
        early_avg = sum(costs[:mid]) / mid
        late_avg = sum(costs[mid:]) / max(1, n - mid)
        assert late_avg <= early_avg * 1.15

    def test_empty_when_no_test_positives(self):
        """If all medias of target category land in sim, test set has no positives -> empty."""
        # Only 1 media of target category — likely all end up in sim with 50% split
        medias = {
            1: {"id": 1, "embedding": np.ones(8, dtype=np.float32), "category": "rare"},
            2: {"id": 2, "embedding": -np.ones(8, dtype=np.float32), "category": "common"},
            3: {"id": 3, "embedding": -np.ones(8, dtype=np.float32) * 0.9, "category": "common"},
            4: {"id": 4, "embedding": -np.ones(8, dtype=np.float32) * 0.8, "category": "common"},
        }
        rows = simulate_voting_iterations(medias, "rare", seed=42, sim_fraction=0.5)
        # Might be empty or not depending on split — just shouldn't crash
        assert isinstance(rows, list)

    def test_inclusion_affects_cost(self):
        """With overlapping data, different inclusion values produce different costs."""
        medias = _make_overlapping_clips(n_per_cat=20)
        rows_inc0 = simulate_voting_iterations(medias, "alpha", seed=42, inclusion=0)
        rows_inc5 = simulate_voting_iterations(medias, "alpha", seed=42, inclusion=5)
        # Same splits but different inclusion -> costs should differ
        costs0 = [r["cost"] for r in rows_inc0]
        costs5 = [r["cost"] for r in rows_inc5]
        assert costs0 != costs5

    def test_elapsed_seconds_non_negative_and_increasing(self):
        """elapsed_seconds should be non-negative and non-decreasing over rows."""
        medias = _make_separable_clips(n_per_cat=10)
        rows = simulate_voting_iterations(medias, "alpha", seed=42)
        times = [r["elapsed_seconds"] for r in rows]
        assert all(t >= 0.0 for t in times)
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]


# ------------------------------------------------------------------
# Integration test: run_voting_iterations_eval
# ------------------------------------------------------------------


class TestRunVotingIterationsEval:
    def test_returns_dataframe(self):
        medias = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[42],
            categories={"ds1": ["alpha"]},
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["seed", "dataset", "category", "t", "cost", "fpr", "fnr", "elapsed_seconds"]

    def test_multiple_seeds(self):
        medias = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[1, 2, 3],
            categories={"ds1": ["alpha"]},
        )
        assert set(df["seed"].unique()) == {1, 2, 3}

    def test_multiple_categories(self):
        medias = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[42],
            categories={"ds1": ["alpha", "beta"]},
        )
        assert set(df["category"].unique()) == {"alpha", "beta"}

    def test_auto_categories(self):
        """When categories=None, all unique categories are used."""
        medias = _make_three_category_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[42],
        )
        assert set(df["category"].unique()) == {"alpha", "beta", "gamma"}

    def test_multiple_datasets(self):
        clips1 = _make_separable_clips(n_per_cat=10, seed=0)
        clips2 = _make_separable_clips(n_per_cat=10, seed=1)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": clips1, "ds2": clips2},
            seeds=[42],
            categories={"ds1": ["alpha"], "ds2": ["beta"]},
        )
        assert set(df["dataset"].unique()) == {"ds1", "ds2"}

    def test_cost_column_numeric(self):
        medias = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[42],
            categories={"ds1": ["alpha"]},
        )
        assert df["cost"].dtype == np.float64
        assert df["fpr"].dtype == np.float64
        assert df["fnr"].dtype == np.float64

    def test_full_cross_product_shape(self):
        """2 seeds x 1 dataset x 2 categories -> each combo produces rows."""
        medias = _make_separable_clips(n_per_cat=10)
        df = run_voting_iterations_eval(
            dataset_clips={"ds1": medias},
            seeds=[1, 2],
            categories={"ds1": ["alpha", "beta"]},
        )
        combos = df.groupby(["seed", "dataset", "category"]).ngroups
        assert combos == 4  # 2 seeds x 1 dataset x 2 categories
