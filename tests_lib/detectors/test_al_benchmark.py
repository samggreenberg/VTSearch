"""Tests for the hermetic active-learning benchmark harness."""

import numpy as np
import pandas as pd
import pytest

from vtscore.eval.al_benchmark import (
    _final_cost_summary,
    precomputed_source,
    precomputed_source_from_npz,
    run_al_benchmark,
    synthetic_source,
)
from vtscore.embedding.media_vectors import media_embedding


# ------------------------------------------------------------------
# synthetic_source
# ------------------------------------------------------------------


class TestSyntheticSource:
    def test_shape_and_counts(self):
        clips = synthetic_source(n_per_cat=10, dim=16, n_categories=3, seed=0)
        assert set(clips) == {"synthetic"}
        medias = clips["synthetic"]
        assert len(medias) == 30
        cats = {m["category"] for m in medias.values()}
        assert cats == {"cat0", "cat1", "cat2"}
        for m in medias.values():
            vec = media_embedding(m)
            assert vec is not None
            assert vec.shape == (16,)

    def test_seed_reproducible(self):
        a = synthetic_source(n_per_cat=5, dim=8, seed=42)["synthetic"]
        b = synthetic_source(n_per_cat=5, dim=8, seed=42)["synthetic"]
        for cid in a:
            np.testing.assert_array_equal(media_embedding(a[cid]), media_embedding(b[cid]))

    def test_separation_is_separable(self):
        # High separation, low noise => classes are cleanly split in feature space.
        clips = synthetic_source(n_per_cat=20, dim=8, n_categories=2, separation=3.0, noise=0.1, seed=1)["synthetic"]
        cat0 = np.array([media_embedding(m) for m in clips.values() if m["category"] == "cat0"])
        cat1 = np.array([media_embedding(m) for m in clips.values() if m["category"] == "cat1"])
        # Between-class centroid distance dwarfs within-class spread.
        centroid_gap = np.linalg.norm(cat0.mean(0) - cat1.mean(0))
        within = np.linalg.norm(cat0 - cat0.mean(0), axis=1).mean()
        assert centroid_gap > within

    def test_rejects_bad_params(self):
        with pytest.raises(ValueError):
            synthetic_source(n_categories=1)
        with pytest.raises(ValueError):
            synthetic_source(n_per_cat=0)


# ------------------------------------------------------------------
# precomputed_source
# ------------------------------------------------------------------


class TestPrecomputedSource:
    def test_wraps_features_and_labels(self):
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((6, 4)).astype(np.float32)
        labels = ["a", "a", "b", "b", "c", "c"]
        clips = precomputed_source(feats, labels, name="pc")["pc"]
        assert len(clips) == 6
        assert {m["category"] for m in clips.values()} == {"a", "b", "c"}
        # Media ids default to 1..n and the vectors round-trip.
        np.testing.assert_array_equal(media_embedding(clips[1]), feats[0])

    def test_custom_ids(self):
        feats = np.zeros((2, 3), dtype=np.float32)
        clips = precomputed_source(feats, ["x", "y"], ids=[100, 200])["precomputed"]
        assert set(clips) == {100, 200}

    def test_rejects_shape_mismatch(self):
        with pytest.raises(ValueError):
            precomputed_source(np.zeros((3, 4), np.float32), ["a", "b"])  # 3 rows, 2 labels
        with pytest.raises(ValueError):
            precomputed_source(np.zeros(4, np.float32), ["a"])  # 1D features

    def test_npz_roundtrip_without_pickle(self, tmp_path):
        rng = np.random.default_rng(1)
        feats = rng.standard_normal((5, 4)).astype(np.float32)
        labels = np.array(["p", "q", "p", "q", "p"])
        path = tmp_path / "features.npz"
        np.savez(path, features=feats, labels=labels)
        clips = precomputed_source_from_npz(path)
        medias = clips["features"]  # default name = file stem
        assert len(medias) == 5
        assert {m["category"] for m in medias.values()} == {"p", "q"}

    def test_npz_missing_keys_raises(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(path, wrong=np.zeros((2, 2)))
        with pytest.raises(ValueError):
            precomputed_source_from_npz(path)


# ------------------------------------------------------------------
# run_al_benchmark + summary
# ------------------------------------------------------------------


class TestRunAlBenchmark:
    def test_multiple_strategies_and_strategy_column(self):
        clips = synthetic_source(n_per_cat=12, dim=12, seed=0)
        df = run_al_benchmark(
            clips,
            strategies=["random", "margin"],
            seeds=[0, 1],
            calibrate_count=1,
            max_steps=8,
        )
        assert not df.empty
        assert set(df["strategy"].unique()) == {"random", "margin"}
        assert "cost" in df.columns

    def test_density_strategy_runs(self):
        clips = synthetic_source(n_per_cat=15, dim=12, seed=2)
        df = run_al_benchmark(
            clips,
            strategies=["density_margin"],
            seeds=[0],
            calibrate_count=1,
            max_steps=8,
            atlas_min_node_size=3,
        )
        assert not df.empty
        assert (df["strategy"] == "density_margin").all()

    def test_plot_dir_writes_pngs(self, tmp_path):
        clips = synthetic_source(n_per_cat=10, dim=8, seed=0)
        run_al_benchmark(
            clips,
            strategies=["random", "margin"],
            seeds=[0],
            calibrate_count=1,
            max_steps=6,
            plot_dir=tmp_path,
        )
        pngs = list(tmp_path.glob("*.png"))
        assert {p.name for p in pngs} == {"voting_iterations_cost.png", "voting_iterations_fpr_fnr.png"}

    def test_final_cost_summary_ranks_strategies(self):
        clips = synthetic_source(n_per_cat=12, dim=12, seed=0)
        df = run_al_benchmark(clips, strategies=["random", "margin"], seeds=[0, 1], calibrate_count=1, max_steps=8)
        summary = _final_cost_summary(df)
        assert list(summary.columns) == ["strategy", "final_cost"]
        assert set(summary["strategy"]) == {"random", "margin"}
        # Sorted ascending by final cost.
        assert summary["final_cost"].is_monotonic_increasing

    def test_final_cost_summary_empty(self):
        empty = pd.DataFrame(
            columns=["seed", "dataset", "category", "strategy", "t", "cost", "fpr", "fnr"]  # pyright: ignore[reportArgumentType]
        )
        summary = _final_cost_summary(empty)
        assert summary.empty
