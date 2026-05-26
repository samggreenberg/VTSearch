"""Tests for the eval CLI entry points.

* ``python -m vtscore.eval``; :mod:`vtscore.eval.__main__`
* ``python -m vtscore.eval.label_curve_main``

The full ``run_eval`` / ``run_label_curve_eval`` calls download demo
datasets and run real embedders, so we test argument parsing, ``--list``
output, and the JSON / CSV serialisation paths with the heavy entry
points stubbed.  This still exercises the argparse wiring and the
result-printing branches that previously had 0% coverage.
"""

from __future__ import annotations

import json
import sys

import pytest

from vtscore.eval import label_curve_main as lc_main
from vtscore.eval import __main__ as eval_main
from vtscore.eval.metrics import DatasetResult, LearnedSortMetrics, QueryMetrics


# ---------------------------------------------------------------------------
# vtscore.eval.__main__
# ---------------------------------------------------------------------------


class TestEvalMainList:
    """``--list`` enumerates available eval datasets and exits 0."""

    def test_list_exits_zero_and_prints_known_id(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["prog", "--list"])
        with pytest.raises(SystemExit) as excinfo:
            eval_main.main()
        assert excinfo.value.code == 0
        out = capsys.readouterr().out
        assert "Available eval datasets:" in out
        # One of the canonical demo IDs must appear.
        assert "esc50_s" in out


class TestEvalMainRun:
    """The body of ``main`` after argument parsing: printing, JSON write,
    and plot generation. Exercised with ``run_eval`` stubbed."""

    @pytest.fixture
    def stub_run_eval(self, monkeypatch):
        """Stub run_eval to return one populated DatasetResult."""

        result = DatasetResult(dataset_id="esc50_s", media_type="audio")
        result.text_sort = [
            QueryMetrics(
                query_text="a dog barking",
                target_category="dog",
                num_relevant=1,
                num_total=10,
                average_precision=0.5,
                precision_at_k={5: 0.4, 10: 0.3},
                recall_at_k={5: 0.2, 10: 0.4},
                elapsed_seconds=0.1,
            ),
        ]
        result.learned_sort = [
            LearnedSortMetrics(
                target_category="dog",
                num_train=10,
                num_test=5,
                accuracy=0.8,
                precision=0.7,
                recall=0.6,
                f1=0.65,
                elapsed_seconds=0.2,
            ),
        ]

        def _fake(**kwargs):
            return [result]

        monkeypatch.setattr(eval_main, "run_eval", _fake)
        return result

    def test_main_prints_summary(self, monkeypatch, capsys, stub_run_eval):
        monkeypatch.setattr(sys, "argv", ["prog", "--datasets", "esc50_s"])
        eval_main.main()
        out = capsys.readouterr().out
        assert "SUMMARY" in out
        assert "esc50_s" in out
        # text-sort surfaces the mAP line; learned-sort surfaces mean_F1.
        assert "mAP=" in out
        assert "mean_F1=" in out

    def test_output_writes_json_file(self, monkeypatch, capsys, tmp_path, stub_run_eval):
        out_path = tmp_path / "results.json"
        monkeypatch.setattr(sys, "argv", ["prog", "--output", str(out_path)])
        eval_main.main()
        assert out_path.is_file()
        # The CLI uses format_results_json which is JSON-valid.
        data = json.loads(out_path.read_text())
        assert isinstance(data, list)
        assert data[0]["dataset_id"] == "esc50_s"

    def test_plot_dir_invokes_plotter(self, monkeypatch, capsys, tmp_path, stub_run_eval):
        from vtscore.eval import visualize

        calls = {}

        def _fake_plot(results, output_dir):
            calls["results"] = results
            calls["output_dir"] = output_dir
            return [tmp_path / "plot.png"]

        monkeypatch.setattr(visualize, "plot_eval_results", _fake_plot)
        monkeypatch.setattr(sys, "argv", ["prog", "--plot-dir", str(tmp_path)])
        eval_main.main()
        assert calls["output_dir"] == str(tmp_path)
        out = capsys.readouterr().out
        assert "Plots written" in out

    def test_no_plot_flag_skips_plotter(self, monkeypatch, tmp_path, stub_run_eval):
        from vtscore.eval import visualize

        called = {"yes": False}

        def _fake_plot(results, output_dir):
            called["yes"] = True
            return []

        monkeypatch.setattr(visualize, "plot_eval_results", _fake_plot)
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--plot-dir", str(tmp_path), "--no-plot"],
        )
        eval_main.main()
        assert called["yes"] is False


# ---------------------------------------------------------------------------
# vtscore.eval.label_curve_main
# ---------------------------------------------------------------------------


class TestLabelCurveMain:
    @pytest.fixture
    def stub_pipeline(self, monkeypatch):
        """Stub the demo-dataset load and the eval to make ``main`` headless."""
        import pandas as pd

        monkeypatch.setattr(lc_main, "_load_dataset", lambda demo_id: {1: {"id": 1, "category": "x"}})

        df = pd.DataFrame(
            [
                {
                    "dataset": "esc50_s",
                    "category": "dog",
                    "trainer": "mlp",
                    "n_labels": 10,
                    "auroc_mean": 0.8,
                    "auroc_std": 0.05,
                    "average_precision_mean": 0.7,
                    "average_precision_std": 0.04,
                    "best_f1_mean": 0.6,
                    "f1_at_xcal_mean": 0.55,
                    "f1_at_xcal_std": 0.03,
                    "train_seconds_mean": 0.1,
                },
            ]
        )

        monkeypatch.setattr(lc_main, "run_label_curve_eval", lambda **kw: df)

        # summarise just returns the same df shape we built; pass through.
        monkeypatch.setattr(lc_main, "summarise", lambda d: d)
        return df

    def test_main_smoke(self, capsys, stub_pipeline):
        rc = lc_main.main(["--datasets", "esc50_s", "--label-counts", "10", "--seeds", "0"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Loading esc50_s" in out
        assert "SUMMARY" in out
        # The summary row is printed via _print_summary.
        assert "esc50_s" in out
        assert "mlp" in out

    def test_main_csv_output(self, tmp_path, stub_pipeline):
        out_path = tmp_path / "curve.csv"
        rc = lc_main.main(
            [
                "--datasets",
                "esc50_s",
                "--label-counts",
                "10",
                "--seeds",
                "0",
                "--output",
                str(out_path),
            ]
        )
        assert rc == 0
        assert out_path.is_file()
        text = out_path.read_text()
        assert "trainer" in text  # header row from to_csv

    def test_main_json_output(self, tmp_path, stub_pipeline):
        out_path = tmp_path / "curve.json"
        rc = lc_main.main(
            [
                "--datasets",
                "esc50_s",
                "--label-counts",
                "10",
                "--seeds",
                "0",
                "--output",
                str(out_path),
            ]
        )
        assert rc == 0
        # JSON output is well-formed.
        data = json.loads(out_path.read_text())
        assert isinstance(data, list)
        assert data[0]["trainer"] == "mlp"

    def test_main_no_datasets_loaded_returns_2(self, monkeypatch, capsys):
        """If every dataset fails to load, exit code is 2."""

        def _boom(demo_id):
            raise RuntimeError("nope")

        monkeypatch.setattr(lc_main, "_load_dataset", _boom)
        rc = lc_main.main(["--datasets", "esc50_s", "--label-counts", "10", "--seeds", "0"])
        assert rc == 2
        err = capsys.readouterr().err
        assert "ERROR loading esc50_s" in err
        assert "No datasets loaded" in err

    def test_print_summary_empty_dataframe(self, capsys):
        import pandas as pd

        lc_main._print_summary(pd.DataFrame())
        out = capsys.readouterr().out
        assert "no rows" in out
