"""Tests for the vtsearch.eval.visualize module.

All tests use synthetic data structures — no real model downloads or
embeddings required.  Charts are written to a temporary directory and
verified for existence, correct filenames, and basic file validity.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vtsearch.eval.metrics import DatasetResult, LearnedSortMetrics, QueryMetrics
from vtsearch.eval.visualize import plot_eval_results, plot_voting_iterations


@pytest.fixture()
def tmp_dir():
    """Create a temporary directory, cleaned up after each test."""
    d = tempfile.mkdtemp(prefix="vtsearch_viz_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------
# Fixtures: synthetic DatasetResult objects
# ------------------------------------------------------------------


def _make_text_sort_result(dataset_id="test_ds", media_type="image") -> DatasetResult:
    dr = DatasetResult(dataset_id=dataset_id, media_type=media_type)
    dr.text_sort = [
        QueryMetrics(
            query_text="a photograph of a cat",
            target_category="cat",
            average_precision=0.85,
            precision_at_k={5: 0.8, 10: 0.7, 20: 0.5},
            recall_at_k={5: 0.4, 10: 0.7, 20: 1.0},
            num_relevant=10,
            num_total=50,
        ),
        QueryMetrics(
            query_text="a photograph of a dog",
            target_category="dog",
            average_precision=0.72,
            precision_at_k={5: 0.6, 10: 0.5, 20: 0.4},
            recall_at_k={5: 0.3, 10: 0.5, 20: 0.8},
            num_relevant=10,
            num_total=50,
        ),
    ]
    return dr


def _make_learned_sort_result(dataset_id="test_ds", media_type="image") -> DatasetResult:
    dr = DatasetResult(dataset_id=dataset_id, media_type=media_type)
    dr.learned_sort = [
        LearnedSortMetrics(
            accuracy=0.9,
            precision=0.85,
            recall=0.8,
            f1=0.82,
            num_train=25,
            num_test=25,
            target_category="cat",
        ),
        LearnedSortMetrics(
            accuracy=0.88,
            precision=0.82,
            recall=0.78,
            f1=0.80,
            num_train=25,
            num_test=25,
            target_category="dog",
        ),
    ]
    return dr


def _make_voting_iterations_df(n_seeds=2, n_steps=10) -> pd.DataFrame:
    rows = []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        for t in range(2, 2 + n_steps):
            cost = max(0, 1.0 - t * 0.08 + rng.normal(0, 0.05))
            fpr = max(0, 0.5 - t * 0.04 + rng.normal(0, 0.03))
            fnr = max(0, 0.5 - t * 0.04 + rng.normal(0, 0.03))
            rows.append(
                {"seed": seed, "dataset": "ds1", "category": "alpha", "t": t, "cost": cost, "fpr": fpr, "fnr": fnr}
            )
    return pd.DataFrame(rows, columns=["seed", "dataset", "category", "t", "cost", "fpr", "fnr"])


# ------------------------------------------------------------------
# plot_eval_results
# ------------------------------------------------------------------


class TestPlotEvalResults:
    def test_text_sort_generates_four_plots(self, tmp_dir):
        results = [_make_text_sort_result()]
        paths = plot_eval_results(results, output_dir=tmp_dir)
        assert len(paths) == 4
        expected = {
            "text_sort_map_by_dataset.png",
            "text_sort_ap_by_query.png",
            "text_sort_precision_at_k.png",
            "text_sort_recall_at_k.png",
        }
        assert {p.name for p in paths} == expected
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_learned_sort_generates_two_plots(self, tmp_dir):
        results = [_make_learned_sort_result()]
        paths = plot_eval_results(results, output_dir=tmp_dir)
        assert len(paths) == 2
        expected = {
            "learned_sort_f1_by_category.png",
            "learned_sort_metrics_breakdown.png",
        }
        assert {p.name for p in paths} == expected
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_both_modes_generates_six_plots(self, tmp_dir):
        dr = _make_text_sort_result()
        dr_learned = _make_learned_sort_result()
        # Combine into one result with both
        dr.learned_sort = dr_learned.learned_sort
        paths = plot_eval_results([dr], output_dir=tmp_dir)
        assert len(paths) == 6

    def test_empty_results_generates_no_plots(self, tmp_dir):
        paths = plot_eval_results([], output_dir=tmp_dir)
        assert paths == []

    def test_creates_output_dir(self, tmp_dir):
        nested = tmp_dir / "sub" / "dir"
        results = [_make_text_sort_result()]
        paths = plot_eval_results(results, output_dir=nested)
        assert nested.is_dir()
        assert len(paths) > 0

    def test_multiple_datasets(self, tmp_dir):
        r1 = _make_text_sort_result(dataset_id="animals_images")
        r2 = _make_text_sort_result(dataset_id="vehicles_images")
        paths = plot_eval_results([r1, r2], output_dir=tmp_dir)
        assert len(paths) == 4
        for p in paths:
            assert p.exists()

    def test_output_files_are_valid_png(self, tmp_dir):
        results = [_make_text_sort_result()]
        paths = plot_eval_results(results, output_dir=tmp_dir)
        for p in paths:
            header = p.read_bytes()[:8]
            # PNG magic bytes
            assert header[:4] == b"\x89PNG"

    def test_string_output_dir(self, tmp_dir):
        """Accepts str path, not just Path objects."""
        results = [_make_text_sort_result()]
        paths = plot_eval_results(results, output_dir=str(tmp_dir))
        assert len(paths) > 0


# ------------------------------------------------------------------
# plot_voting_iterations
# ------------------------------------------------------------------


class TestPlotVotingIterations:
    def test_generates_two_plots(self, tmp_dir):
        df = _make_voting_iterations_df()
        paths = plot_voting_iterations(df, output_dir=tmp_dir)
        assert len(paths) == 2
        expected = {
            "voting_iterations_cost.png",
            "voting_iterations_fpr_fnr.png",
        }
        assert {p.name for p in paths} == expected
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0

    def test_empty_dataframe_generates_no_plots(self, tmp_dir):
        df = pd.DataFrame(columns=["seed", "dataset", "category", "t", "cost", "fpr", "fnr"])
        paths = plot_voting_iterations(df, output_dir=tmp_dir)
        assert paths == []

    def test_non_dataframe_generates_no_plots(self, tmp_dir):
        paths = plot_voting_iterations(None, output_dir=tmp_dir)
        assert paths == []

    def test_single_seed(self, tmp_dir):
        df = _make_voting_iterations_df(n_seeds=1)
        paths = plot_voting_iterations(df, output_dir=tmp_dir)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_multiple_categories(self, tmp_dir):
        df1 = _make_voting_iterations_df(n_seeds=1)
        df2 = df1.copy()
        df2["category"] = "beta"
        df = pd.concat([df1, df2], ignore_index=True)
        paths = plot_voting_iterations(df, output_dir=tmp_dir)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_many_groups_uses_combined_chart(self, tmp_dir):
        """When > 4 (dataset, category) groups, FPR/FNR uses a combined chart."""
        frames = []
        for i in range(6):
            df = _make_voting_iterations_df(n_seeds=1)
            df["category"] = f"cat_{i}"
            frames.append(df)
        combined = pd.concat(frames, ignore_index=True)
        paths = plot_voting_iterations(combined, output_dir=tmp_dir)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()

    def test_output_files_are_valid_png(self, tmp_dir):
        df = _make_voting_iterations_df()
        paths = plot_voting_iterations(df, output_dir=tmp_dir)
        for p in paths:
            header = p.read_bytes()[:8]
            assert header[:4] == b"\x89PNG"
