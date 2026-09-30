"""The trajectory A/B sizing gate: `preflight.sh` check 17 and what it counts (#4111).

#3840 turned an A/B's resolution into a rule, validated on 399 fresh cells
against a pre-registered prediction: SE of the paired mean Δcost is σ/√n with
σ ≈ 0.04, so resolving δ at 2 SE takes n = (2σ/δ)² paired cells.  #3825 ran 114
cells on a question that needed 400 and found its floor in the write-up.  The
rule only helps before submission, so it is a preflight check.

What is pinned here:

* the gate's arithmetic lands on the report's own table at every boundary,
  including the ones floating point puts a hair past a whole number;
* the count it reads is the number of cells ``analyze_ab.py`` *pairs*, one per
  style, not the array-task count ``--print-cells`` sizes an array with;
* ``analyze_ab.py`` prints the floor beside the Δ, so a report cannot quote one
  without the other.

Meta-group: nothing here imports shipped ``vtsearch``/``vtscore`` code.  The
subject is repo tooling under ``scripts/``, which has no other test or type
coverage (``pyrightconfig.json`` excludes it).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_PREFLIGHT = _ROOT / "scripts" / "experiments" / "preflight.sh"
_CALIB = _ROOT / "scripts" / "experiments" / "calibration"


def _preflight(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run the preflight with no VTS_REPO, so only the explicit-count path is live."""
    return subprocess.run(  # noqa: S603  # fixed argv, repo-local script path, no shell
        ["bash", str(_PREFLIGHT), "--exp", str(tmp_path / "exp"), *args],  # noqa: S607 - bash from PATH
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)},
    )


class TestPreflightGate:
    """Check 17 on an explicit `--paired-cells` count."""

    @pytest.mark.parametrize(
        ("delta", "need"),
        (("0.02", 16), ("0.01", 64), ("0.005", 256), ("0.004", 400), ("0.002", 1600)),
    )
    def test_the_boundary_is_the_reports_table(self, tmp_path, delta, need):
        """n = (2σ/δ)² exactly: `need` cells pass and one fewer fails.

        2*0.04/0.004 is 20.000000000000004 in floating point, so a bare ceil
        would ask for 401 where the table says 400.
        """
        ok = _preflight(tmp_path, "--resolve-delta", delta, "--paired-cells", str(need)).stdout
        assert f"ok    A/B resolves δ={delta} at 2 SE: {need} paired cells >= {need}" in ok
        short = _preflight(tmp_path, "--resolve-delta", delta, "--paired-cells", str(need - 1)).stdout
        assert f"FAIL  this A/B has {need - 1} paired cells; resolving δ={delta} at 2 SE (σ=0.04) needs {need}" in short

    def test_3825s_grid_is_refused_with_the_delta_it_could_resolve(self, tmp_path):
        """114 cells against δ = 0.004: the floor #3840 quotes is about 0.0075."""
        proc = _preflight(tmp_path, "--resolve-delta", "0.004", "--paired-cells", "114")
        assert "needs 400" in proc.stdout
        assert "resolves δ ≈ 0.00749" in proc.stdout
        assert "PREFLIGHT FAILED" in proc.stdout
        assert proc.returncode == 1

    def test_sigma_is_overridable(self, tmp_path):
        """#3796's 0.056 on `vg_scale_any`: (2 · 0.056 / 0.01)² = 125.44, so 126."""
        short = _preflight(tmp_path, "--resolve-delta", "0.01", "--sigma", "0.056", "--paired-cells", "125").stdout
        assert "(σ=0.056) needs 126" in short
        ok = _preflight(tmp_path, "--resolve-delta", "0.01", "--sigma", "0.056", "--paired-cells", "126").stdout
        assert "ok    A/B resolves δ=0.01 at 2 SE: 126 paired cells >= 126 (σ=0.056" in ok

    def test_warn_only_downgrades_it(self, tmp_path):
        proc = _preflight(tmp_path, "--warn-only", "--resolve-delta", "0.004", "--paired-cells", "114")
        assert "WARN  this A/B has 114 paired cells" in proc.stdout
        assert "preflight OK" in proc.stdout

    def test_the_check_is_opt_in(self, tmp_path):
        """Most launches are not A/Bs, and none of them may meet this gate."""
        assert "A/B" not in _preflight(tmp_path).stdout

    def test_no_count_and_no_grid_to_count_is_a_failure_naming_the_fix(self, tmp_path):
        """An explicit `--resolve-delta` asks for the check now; it cannot pass unlooked."""
        out = _preflight(tmp_path, "--resolve-delta", "0.004").stdout
        assert "FAIL  --resolve-delta: no prepare_info.json" in out
        assert "pass --paired-cells N" in out

    @pytest.mark.parametrize(
        "args",
        (
            ("--sigma", "0.04"),
            ("--paired-cells", "400"),
            ("--resolve-delta", "0"),
            ("--resolve-delta", "-0.01"),
            ("--resolve-delta", "abc"),
            ("--resolve-delta", "0.01", "--sigma", "0"),
            ("--resolve-delta", "0.01", "--paired-cells", "1e3"),
            ("--resolve-delta", "0.01", "--paired-cells", "-4"),
        ),
        ids=("sigma-alone", "cells-alone", "zero-delta", "negative-delta", "word", "zero-sigma", "float-cells", "neg"),
    )
    def test_a_request_that_cannot_mean_what_it_says_is_a_usage_error(self, tmp_path, args):
        proc = _preflight(tmp_path, *args)
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert "preflight:" not in proc.stdout  # refused before any check ran


# --- What the gate counts ------------------------------------------------------


def _load(name: str, path: Path):
    """Import one calibration script by path, with its directory importable only while it loads."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _with_stubs(name: str, path: Path, **modules: Any):
    """Load *path* with ``common`` stubbed inert and *modules* pre-installed."""
    stub: Any = types.ModuleType("common")
    stub.setup_env = lambda: None
    stub.log = lambda _msg: None
    stub.RESULTS = Path(".")
    installed = {"common": stub, **modules}
    saved = {k: sys.modules.get(k) for k in installed}
    sys.modules.update(installed)
    try:
        return _load(name, path)
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


#: The #3585 environments #3840 sized its rule on: boxed VG and COCO run a
#: patch embedder under both styles, boxless Caltech runs it whole-image only.
_GRID_3585 = {
    "CALIB_DATASETS": "visual_genome_m,caltech101_m,coco_val",
    "CALIB_VG_EMBEDDERS": "siglip,dinov3_patch",
    "CALIB_CALTECH_EMBEDDERS": "siglip,dinov3_patch",
    "CALIB_COCO_EMBEDDERS": "siglip,dinov3_patch",
    "CALIB_PATCH_STYLES": "whole_image,max_patch",
}
_CATS = {"visual_genome_m": 8, "caltech101_m": 6, "coco_val": 7}


def _prepare_info() -> dict:
    return {
        "datasets": {
            ds: {emb: {"selected_categories": [f"{ds}-{i}" for i in range(n)]} for emb in ("siglip", "dinov3_patch")}
            for ds, n in _CATS.items()
        }
    }


@pytest.fixture
def grid(monkeypatch):
    """``(experiment_config, run_cells)`` for the #3585 grid under extra env *overrides*."""

    def build(**overrides: str):
        for key in ("CALIB_TRAINER", "CALIB_CALIBRATION_SEEDS", "CALIB_TRAIN_MIXES"):
            monkeypatch.delenv(key, raising=False)
        for key, value in {**_GRID_3585, "CALIB_N_SEEDS": "2", **overrides}.items():
            monkeypatch.setenv(key, value)
        cfg = _load("_ab_gate_experiment_config", _CALIB / "experiment_config.py")
        rc = _with_stubs("_ab_gate_run_cells", _CALIB / "run_cells.py", experiment_config=cfg)
        return cfg, rc

    return build


class TestPairedCellCount:
    def test_3825s_grid_is_114_paired_cells_from_84_tasks(self, grid):
        """57 paired cells a seed (#3840) against 42 array tasks: styles are cells."""
        cfg, rc = grid()
        tasks = cfg.array_cells(rc._categories_by_dataset(_prepare_info()))
        assert len(tasks) == 2 * 2 * sum(_CATS.values()) == 84
        assert rc.paired_cell_count(_prepare_info()) == 2 * 57 == 114

    def test_3840s_nine_seeds_are_its_513(self, grid):
        _cfg, rc = grid(CALIB_N_SEEDS="9")
        assert rc.paired_cell_count(_prepare_info()) == 513

    def test_a_boxless_patch_arm_is_one_cell_not_two(self, grid):
        """`styles_for` falls a patch embedder back to whole_image without boxes."""
        _cfg, rc = grid(CALIB_DATASETS="caltech101_m")
        assert rc.paired_cell_count(_prepare_info()) == 2 * 2 * 6

    def test_a_standalone_trainer_runs_one_style_less_cell(self, grid):
        """A #3959 trainer has no detection style, so a task is one paired cell."""
        cfg, rc = grid(CALIB_TRAINER="gp_rbf", CALIB_VG_EMBEDDERS="siglip", CALIB_COCO_EMBEDDERS="siglip")
        assert cfg.cell_styles("visual_genome_m", "siglip") == [None]
        tasks = cfg.array_cells(rc._categories_by_dataset(_prepare_info()))
        assert rc.paired_cell_count(_prepare_info()) == len(tasks)

    def test_calibration_seed_draws_share_one_paired_cell(self, grid):
        """`analyze_ab` joins on (arm, category, seed): #3796's draws average into one."""
        cfg, rc = grid(CALIB_CALIBRATION_SEEDS="42,7")
        tasks = cfg.array_cells(rc._categories_by_dataset(_prepare_info()))
        assert len(tasks) == 2 * 84
        assert rc.paired_cell_count(_prepare_info()) == 114

    def test_the_count_keys_on_what_analyze_ab_pairs_on(self):
        """If the analyzer's pairing unit moves, the gate's count has to move with it."""
        ab = _with_stubs("_ab_gate_analyze_ab", _CALIB / "analyze_ab.py")
        assert ab.CELL_KEYS == ("arm", "category", "seed")


# --- The floor beside every Δ ----------------------------------------------------


@pytest.fixture(scope="module")
def ab():
    return _with_stubs("_ab_gate_analyze_ab_mod", _CALIB / "analyze_ab.py")


def test_the_floor_is_twice_the_paired_se(ab):
    d = np.random.default_rng(42).normal(0.001, 0.04, 400)
    se, floor = ab._resolution(d)
    assert se == pytest.approx(d.std(ddof=1) / np.sqrt(d.size))
    assert floor == pytest.approx(2 * se)


def test_one_cell_has_no_floor(ab):
    assert all(np.isnan(v) for v in ab._resolution(np.array([0.01, np.nan])))


def _pooled(delta: float, se: float, n: int = 400) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": "app_visible",
                "window": "all_steps",
                "metric": "cost",
                "n_cells": n,
                "delta_on_minus_off": delta,
                "se": se,
                "resolvable_delta_2se": 2 * se,
            }
        ]
    )


def test_a_delta_inside_its_floor_says_so(ab):
    """#3825's +0.0049 at SE 0.0037: printed, and printed as unresolved."""
    line = ab.pooled_line(_pooled(0.0049, 0.0037, 114))
    assert "+0.0049 ± 0.0037 over 114 cells" in line
    assert "resolvable δ at 2 SE 0.0074" in line
    assert "not resolved" in line


def test_a_delta_past_its_floor_says_so(ab):
    """#3825's `ll1e-3` arm: +0.026 at SE 0.0062."""
    assert "(resolved at 2 SE)" in ab.pooled_line(_pooled(0.026, 0.0062, 114))


def test_no_paired_cells_is_said_rather_than_raised(ab):
    assert "no paired cells" in ab.pooled_line(pd.DataFrame())
