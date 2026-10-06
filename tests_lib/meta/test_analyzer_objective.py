"""The calibration analyzers default to the objective on a frame that carries a beta (#4584).

The app draws its line at an F-beta balance and every review reads the objective,
F-beta of the withheld half above the threshold the app holds (#4427).  The
shared analyzers under ``scripts/experiments/calibration/`` defaulted to
``cost``, priced at Inclusion 0 whatever beta drew the line, so a study that
named no metric read a preference the app no longer holds.  Pinned here:

* ``objective.py``'s rule: the objective on a frame with a balance, cost on one
  without, and the columns filled from the rates on a frame that predates them;
* ``curves``, ``stopping`` and ``viewer`` follow it when no metric is named, and
  still honour one that is.

Meta-group: the subject is repo tooling under ``scripts/``, loaded by path with
``common`` stubbed inert, as ``test_ab_resolution_gate.py`` loads ``analyze_ab``.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"


def _load(name: str):
    """Import one calibration script by path, ``common`` stubbed and its directory importable while it loads."""
    stub: Any = types.ModuleType("common")
    stub.setup_env = lambda: None
    stub.log = lambda _msg: None
    stub.RESULTS = Path(".")
    saved = sys.modules.get("common")
    sys.modules["common"] = stub
    sys.path.insert(0, str(_CALIB))
    try:
        spec = importlib.util.spec_from_file_location(f"_objective_test_{name}", _CALIB / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_CALIB))
        if saved is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = saved


@pytest.fixture(scope="module")
def objective():
    return _load("objective")


@pytest.fixture(scope="module")
def curves():
    return _load("curves")


@pytest.fixture(scope="module")
def stopping():
    return _load("stopping")


def _frame(beta: float | None, *, n_seeds: int = 3, t_max: int = 6) -> pd.DataFrame:
    """A run's main frame as a balance-era runner wrote it: rates and a beta, no ``fbeta`` column."""
    rng = np.random.default_rng(42)
    rows = []
    for seed in range(n_seeds):
        for t in range(1, t_max + 1):
            rows.append(
                {
                    "arm": "prod",
                    "dataset": "coco_better",
                    "embedder": "siglip",
                    "category": "cat",
                    "seed": seed,
                    "t": t,
                    "beta": np.nan if beta is None else beta,
                    "precision": 0.3 + 0.05 * t,
                    "recall": 0.2 + 0.05 * t,
                    "f1": 0.0,
                    "cost": 0.6 - 0.05 * t + 0.01 * rng.standard_normal(),
                    "average_precision": 0.5,
                    "phase": "done" if t >= 4 else "hard",
                    "n_good": t,
                    "n_bad": t,
                }
            )
    return pd.DataFrame(rows)


class TestTheRule:
    def test_a_frame_with_a_balance_is_decided_on_the_objective(self, objective):
        assert objective.primary_metric(_frame(1.0)) == "fbeta"
        assert objective.primary_metric(_frame(None)) == "cost"
        assert objective.primary_metric(_frame(1.0).drop(columns="beta")) == "cost"

    def test_each_metric_points_its_own_way(self, objective):
        assert objective.lower_is_better("cost") and objective.lower_is_better("regret")
        assert not objective.lower_is_better("fbeta") and not objective.lower_is_better("fbeta_b025")
        assert not objective.lower_is_better("average_precision")

    def test_a_frame_that_predates_the_columns_is_filled_from_its_rates(self, objective):
        out = objective.with_objective(_frame(0.25))
        assert set(objective.preset_columns()) | {"fbeta"} <= set(out.columns)
        p, r = out["precision"].to_numpy(), out["recall"].to_numpy()
        want = 1.0625 * p * r / (0.0625 * p + r)
        np.testing.assert_allclose(out["fbeta"], want)
        np.testing.assert_allclose(out["fbeta_b025"], want)
        np.testing.assert_allclose(out["fbeta_b1"], 2 * p * r / (p + r))

    def test_a_column_the_runner_wrote_is_left_as_written(self, objective):
        df = _frame(1.0).assign(fbeta=0.123, fbeta_b025=0.1, fbeta_b1=0.2, fbeta_b4=0.3)
        assert objective.with_objective(df) is df

    def test_no_balance_is_no_objective_but_the_presets_are_still_read(self, objective):
        out = objective.with_objective(_frame(None))
        assert out["fbeta"].isna().all() and out["fbeta_b1"].notna().all()

    def test_a_csv_row_reads_the_same_number(self, objective):
        row = {k: str(v) for k, v in _frame(1.0).iloc[0].items()}
        p, r = float(row["precision"]), float(row["recall"])
        assert objective.row_objective(row) == pytest.approx(2 * p * r / (p + r))
        assert objective.row_objective({**row, "fbeta": "0.42"}) == pytest.approx(0.42)
        assert objective.rows_carry_beta([row]) and not objective.rows_carry_beta([{**row, "beta": "nan"}])


class TestCurves:
    def test_no_metric_named_draws_the_objective_and_anchors_it_at_the_runs_preset(self, curves):
        main, metric, lower, col = curves.resolve_metric(_frame(1.0), None, None, None)
        assert (metric, lower, col) == ("fbeta", False, "text_fbeta_b1")
        assert main["fbeta"].notna().all()

    def test_a_frame_without_a_balance_still_draws_cost(self, curves):
        _main, metric, lower, col = curves.resolve_metric(_frame(None), None, None, None)
        assert (metric, lower, col) == ("cost", True, None)

    def test_a_named_metric_is_honoured(self, curves):
        _main, metric, lower, _col = curves.resolve_metric(_frame(1.0), "cost", None, None)
        assert (metric, lower) == ("cost", True)

    def test_a_beta_off_the_presets_has_no_text_sort_anchor(self, curves):
        assert curves.objective_anchor_column(_frame(0.5)) is None
        assert curves.objective_anchor_column(_frame(4.0)) == "text_fbeta_b4"


class TestStopping:
    def test_the_stopping_point_carries_the_objective_beside_cost(self, stopping):
        stops = stopping.stopping_points(_frame(1.0), keys=("arm", "dataset", "embedder", "category", "seed"))
        assert {"fbeta_at_stop", "cost_at_stop"} <= set(stops.columns)
        assert stops["fbeta_at_stop"].notna().all()
        table = stopping.stopping_table(stopping.summarise(stops))
        assert "fbeta at stop" in table and "cost at stop" not in table

    def test_a_run_without_a_balance_reports_cost(self, stopping):
        stops = stopping.stopping_points(_frame(None), keys=("arm", "dataset", "embedder", "category", "seed"))
        assert "fbeta_at_stop" not in stops.columns
        assert "cost at stop" in stopping.stopping_table(stopping.summarise(stops))


def _payload(path: Path) -> dict:
    m = re.search(r'type="application/json">(.*?)</script>', path.read_text(encoding="utf-8"), re.S)
    assert m, "no payload script tag"
    return json.loads(m.group(1))


@pytest.mark.parametrize(("beta", "opens_on"), [(1.0, "fbeta"), (None, None)], ids=("balance", "no-balance"))
def test_the_viewer_opens_on_the_objective_on_a_balance_run(tmp_path, beta, opens_on):
    viewer = _load("viewer")
    out = viewer.build_viewer(_frame(beta), tmp_path / "viewer.html", arms=["prod"])
    payload = _payload(out)
    assert payload.get("view", {}).get("metric") == opens_on
    keys = [m["key"] for m in payload["metrics"]]
    assert ("fbeta" in keys) is (beta is not None)


def test_the_viewer_still_opens_where_the_study_says(tmp_path):
    viewer = _load("viewer")
    out = viewer.build_viewer(_frame(1.0), tmp_path / "viewer.html", arms=["prod"], default_metric="cost")
    assert _payload(out)["view"]["metric"] == "cost"
