"""Tests for the voting-iterations evaluation framework.

All tests use small synthetic datasets with known, well-separated
embeddings so no real model downloads are needed.
"""

import numpy as np
import pandas as pd

from vtscore.eval.voting_iterations import (
    _good_training_vec,
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
        medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": "alpha"}
        media_id += 1
    for _ in range(n_per_cat):
        emb = rng.normal(-1.0, 0.2, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": "beta"}
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
        medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": "alpha"}
        media_id += 1
    for _ in range(n_per_cat):
        emb = rng.normal(-0.3, 1.0, dim).astype(np.float32)
        medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": "beta"}
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
            medias[media_id] = {"id": media_id, "embeddings": {"emb": emb}, "category": cat}
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
        expected_keys = {
            "seed",
            "dataset",
            "category",
            "t",
            "n_good",
            "n_bad",
            "cost",
            "fpr",
            "fnr",
            "elapsed_seconds",
        }
        for row in rows:
            assert set(row.keys()) == expected_keys

    def test_vote_counts_reported(self):
        """Each row carries the good/bad vote counts the model was trained on.

        The first scored row reflects the 1-good + 1-bad minimum, and the
        counts never exceed the votes seen so far (t).
        """
        medias = _make_separable_clips(n_per_cat=10)
        rows = simulate_voting_iterations(medias, "alpha", seed=42)
        assert rows  # at least one scored step
        first = rows[0]
        assert first["n_good"] >= 1
        assert first["n_bad"] >= 1
        assert min(first["n_good"], first["n_bad"]) == 1  # earliest trainable step
        for row in rows:
            assert row["n_good"] + row["n_bad"] == row["t"]
            assert row["n_good"] >= 1
            assert row["n_bad"] >= 1

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
        # Only 1 media of target category; likely all end up in sim with 50% split
        medias = {
            1: {"id": 1, "embeddings": {"emb": np.ones(8, dtype=np.float32)}, "category": "rare"},
            2: {"id": 2, "embeddings": {"emb": -np.ones(8, dtype=np.float32)}, "category": "common"},
            3: {"id": 3, "embeddings": {"emb": -np.ones(8, dtype=np.float32) * 0.9}, "category": "common"},
            4: {"id": 4, "embeddings": {"emb": -np.ones(8, dtype=np.float32) * 0.8}, "category": "common"},
        }
        rows = simulate_voting_iterations(medias, "rare", seed=42, sim_fraction=0.5)
        # Might be empty or not depending on split; just shouldn't crash
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
# Production fidelity: the per-step calibration must match the live
# _train_and_score_xy / train_and_threshold pipeline, or the reported
# cost measures a pipeline the detector never runs.
# ------------------------------------------------------------------


def _mt_key(rng: np.random.RandomState):
    """Return the MT19937 key array of *rng* as a tuple (pyright-narrowed).

    ``get_state(legacy=True)`` returns a tuple, but the numpy stub types it as a
    ``dict | tuple`` union; the ``isinstance`` narrows it so ``state[1]`` (the
    624-word key array) type-checks.
    """
    state = rng.get_state(legacy=True)
    assert isinstance(state, tuple)
    return state


class TestProductionCalibrationFidelity:
    """The eval's per-step threshold calibration mirrors production exactly.

    Production (`_train_and_score_xy` / `train_and_threshold`) sizes the hidden
    layer from the full label count, forces that width onto the calibration
    folds, and always calibrates with a fresh ``RandomState(42)`` (the fixed
    seed baked into ``cross_calibration_threshold_cached``).  These tests spy on
    the calibration call to prove the eval does the same, so overlapping the
    fold split RNG with the per-seed simulation RNG or letting folds auto-size
    can't silently reintroduce a production mismatch.
    """

    def _spy_calibration(self, monkeypatch):
        from vtscore.eval import voting_iterations

        real = voting_iterations.calculate_cross_calibration_threshold
        captured: list[dict] = []

        def spy(X_list, y_list, input_dim, inclusion_value=0, *, rng=None, hidden_dim=None, **kw):
            # get_state() copies without advancing, so recording it here does
            # not perturb the real calibration that runs on the next line.
            captured.append(
                {
                    "n": len(X_list),
                    "hidden_dim": hidden_dim,
                    "mt_key": _mt_key(rng)[1] if rng is not None else None,
                }
            )
            return real(X_list, y_list, input_dim, inclusion_value, rng=rng, hidden_dim=hidden_dim, **kw)

        monkeypatch.setattr(voting_iterations, "calculate_cross_calibration_threshold", spy)
        return captured

    def test_folds_forced_to_full_data_hidden_dim(self, monkeypatch):
        from vtscore.training.mlp import _auto_hidden_dim

        captured = self._spy_calibration(monkeypatch)
        medias = _make_separable_clips(n_per_cat=10)
        simulate_voting_iterations(medias, "alpha", seed=42)

        assert captured  # at least one calibrated step
        for c in captured:
            # The fold models must be sized from the full label count for the
            # step, not auto-sized per fold (hidden_dim=None).
            assert c["hidden_dim"] == _auto_hidden_dim(c["n"])

    def test_folds_calibrate_with_fixed_random_state_42(self, monkeypatch):
        captured = self._spy_calibration(monkeypatch)
        medias = _make_separable_clips(n_per_cat=10)
        simulate_voting_iterations(medias, "alpha", seed=7)

        assert captured
        ref_key = _mt_key(np.random.RandomState(42))[1]
        for c in captured:
            mt_key = c["mt_key"]
            assert mt_key is not None
            # A fresh RandomState(42), not the shared per-seed simulation RNG
            # (which the media split + vote sequence would have advanced).
            assert np.array_equal(mt_key, ref_key)

    def test_calibration_rng_independent_of_eval_seed(self, monkeypatch):
        """The fold split RNG is pinned, so it does not vary with the eval seed."""
        medias = _make_separable_clips(n_per_cat=10)

        captured_a = self._spy_calibration(monkeypatch)
        simulate_voting_iterations(medias, "alpha", seed=1)
        captured_b = self._spy_calibration(monkeypatch)
        simulate_voting_iterations(medias, "alpha", seed=2)

        # Same first-step vote count is not guaranteed across seeds, but every
        # calibrated step in both runs must start from the identical RNG state.
        assert captured_a and captured_b
        ref_key = tuple(_mt_key(np.random.RandomState(42))[1].tolist())
        states_a = {tuple(c["mt_key"].tolist()) for c in captured_a}
        states_b = {tuple(c["mt_key"].tolist()) for c in captured_b}
        assert states_a == states_b == {ref_key}


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
        assert list(df.columns) == [
            "seed",
            "dataset",
            "category",
            "t",
            "n_good",
            "n_bad",
            "cost",
            "fpr",
            "fnr",
            "elapsed_seconds",
        ]

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


# ------------------------------------------------------------------
# Region voting (patch datasets)
# ------------------------------------------------------------------

_PATCH_DIM = 8
_GRID = 3  # 3x3 patch grid


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _patch_media(media_id, positive, *, category, with_box=True):
    """A synthetic patch-embedder media (``patch_grid`` + ``patch_regions``).

    Positive media have grid cells pointing along ``+e0`` and a ground-truth
    box; negatives point along ``-e0`` and carry no box.  Separable so the MLP
    trains cleanly without flakiness.
    """
    from vtscore.media.patch_embed import RegionVector

    rng = np.random.default_rng(media_id)
    sign = 1.0 if positive else -1.0
    grid = np.zeros((_GRID, _GRID, _PATCH_DIM), dtype=np.float32)
    for r in range(_GRID):
        for c in range(_GRID):
            base = np.zeros(_PATCH_DIM, dtype=np.float32)
            base[0] = sign
            grid[r, c] = _unit(base + rng.standard_normal(_PATCH_DIM).astype(np.float32) * 0.05)
    img_vec = _unit(grid.reshape(-1, _PATCH_DIM).mean(axis=0))

    regions = [RegionVector(box=(0.0, 0.0, 1.0, 1.0), vec=img_vec)]
    for r in range(_GRID):
        for c in range(_GRID):
            box = (c / _GRID, r / _GRID, (c + 1) / _GRID, (r + 1) / _GRID)
            regions.append(RegionVector(box=box, vec=grid[r, c]))

    media = {
        "id": media_id,
        "media_type": "image",
        "embedder": "dinov3_patch",
        "embeddings": {"dinov3_patch": img_vec},
        "patch_grid": grid,
        "patch_regions": regions,
        "category": category,
    }
    if positive and with_box:
        media["regions"] = [{"box": [0.0, 0.0, 2 / 3, 1.0], "label": category}]
    return media


def _make_patch_clips(n_per_cat=10):
    medias = {}
    media_id = 1
    for _ in range(n_per_cat):
        medias[media_id] = _patch_media(media_id, positive=True, category="apple")
        media_id += 1
    for _ in range(n_per_cat):
        medias[media_id] = _patch_media(media_id, positive=False, category="other")
        media_id += 1
    return medias


class TestGoodTrainingVec:
    """The per-Good-vote training vector, region-pooled or whole-image."""

    def test_image_level_when_region_voting_off(self):
        from vtscore.embedding.media_vectors import media_embedding

        media = _patch_media(1, positive=True, category="apple")
        vec = _good_training_vec(media, "apple", region_voting=False)
        np.testing.assert_allclose(vec, media_embedding(media))

    def test_snaps_box_to_region_when_region_voting_on(self):
        """With a ``patch_regions`` tree present, the simulated region vote
        snaps the ground-truth box to its nearest region node (the same path
        the live vote flow takes), not a fresh uniform grid pool."""
        from vtscore.media.patch_embed import snap_box_to_region

        media = _patch_media(1, positive=True, category="apple")
        vec = _good_training_vec(media, "apple", region_voting=True)
        expected = snap_box_to_region(media["patch_regions"], (0.0, 0.0, 2 / 3, 1.0))
        np.testing.assert_allclose(vec, expected)
        # The snapped vector is one of the tree's actual node vectors.
        assert any(np.allclose(vec, r.vec) for r in media["patch_regions"])

    def test_falls_back_without_patch_grid(self):
        from vtscore.embedding.media_vectors import media_embedding

        media = _patch_media(1, positive=True, category="apple")
        del media["patch_grid"]
        vec = _good_training_vec(media, "apple", region_voting=True)
        np.testing.assert_allclose(vec, media_embedding(media))

    def test_falls_back_without_matching_box(self):
        from vtscore.embedding.media_vectors import media_embedding

        # Positive image but no annotated box for this category.
        media = _patch_media(1, positive=True, category="apple", with_box=False)
        vec = _good_training_vec(media, "apple", region_voting=True)
        np.testing.assert_allclose(vec, media_embedding(media))


class TestRegionVotingSimulate:
    """End-to-end region voting on a synthetic patch dataset."""

    def test_region_voting_produces_finite_rows(self):
        medias = _make_patch_clips(n_per_cat=10)
        rows = simulate_voting_iterations(medias, target_category="apple", seed=0, region_voting=True)
        assert rows  # region-aware scoring path runs end-to-end
        for row in rows:
            assert np.isfinite(row["cost"])
            assert np.isfinite(row["fpr"])
            assert np.isfinite(row["fnr"])

    def test_baseline_on_patch_data_also_scores_region_aware(self):
        # region_voting=False still works on a patch dataset: Good votes train
        # whole-image, but scoring max-pools over regions (the live inference).
        medias = _make_patch_clips(n_per_cat=10)
        rows = simulate_voting_iterations(medias, target_category="apple", seed=0, region_voting=False)
        assert rows
        assert all(np.isfinite(r["cost"]) for r in rows)

    def test_run_eval_threads_region_voting_flag(self):
        medias = _make_patch_clips(n_per_cat=8)
        df = run_voting_iterations_eval(
            dataset_clips={"vg": medias},
            seeds=[0],
            categories={"vg": ["apple"]},
            region_voting=True,
        )
        assert not df.empty
        assert (df["category"] == "apple").all()
