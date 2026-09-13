#!/usr/bin/env python
"""Planted-answer test for ``analyze_3585.py``.

    python selftest_analyze_3585.py

The gate's verdict is a handful of counts read off two joins, which is exactly
the kind of arithmetic that is wrong in a way nobody notices: an inner join that
silently drops the arm rows with no baseline partner turns "12 of 400 changed"
into "12 of 380", and both look like results.  So the tables are computed here
over a frame whose answers were written down first.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_3585 import bench_table, fit_table, gate_by_env, gate_table  # noqa: E402

FAILURES: list[str] = []


def check(name: str, got, want) -> None:
    if isinstance(got, float) and isinstance(want, float):
        ok = abs(got - want) < 1e-9
    else:
        ok = got == want
    print(f"{'ok  ' if ok else 'FAIL'} {name}: got {got!r}, want {want!r}")
    if not ok:
        FAILURES.append(name)


def _cut_row(cell, kind, case, arm, thr, adm, seconds=0.01, provenance="fold_anchored[2/2]", n=1000):
    return {
        "cell": cell,
        "dataset": "d",
        "embedder": "e",
        "category": "c",
        "seed": 0,
        "style": "whole_image",
        "kind": kind,
        "case": case,
        "arm": arm,
        "n": n,
        "n_folds": 2,
        "n_anchored": 2,
        "provenance": provenance,
        "seconds": seconds,
        "threshold_i-3": thr,
        "threshold_i0": thr,
        "threshold_i3": thr,
        "n_admitted_i-3": adm,
        "n_admitted_i0": adm,
        "n_admitted_i3": adm,
    }


def main() -> int:
    # Four fold cases.  Case 1: the candidate moves the threshold but not the
    # admitted set (what the snap is for).  Case 2: it moves one media.  Cases
    # 3 and 4: identical.  So 1 of 4 changed, and the largest move is 0.5.
    cuts = pd.DataFrame(
        [
            _cut_row("cell_0", "fold", "0000", "baseline", 0.50, 100, seconds=0.04),
            _cut_row("cell_0", "fold", "0000", "native", 0.60, 100, seconds=0.01),
            _cut_row("cell_0", "fold", "0005", "baseline", 0.50, 100, seconds=0.04),
            _cut_row("cell_0", "fold", "0005", "native", 0.51, 101, seconds=0.01),
            _cut_row("cell_1", "fold", "0000", "baseline", 0.20, 50, seconds=0.04),
            _cut_row("cell_1", "fold", "0000", "native", 0.20, 50, seconds=0.02),
            _cut_row("cell_1", "fold", "0005", "baseline", 0.10, 10, seconds=0.04),
            _cut_row("cell_1", "fold", "0005", "native", 0.60, 10, provenance="no_cut", seconds=0.01),
            # A sort case, and a baseline-only row that must not become a case.
            _cut_row("cell_1", "sort", "0000", "baseline", 0.30, 7, seconds=0.02),
            _cut_row("cell_1", "sort", "0000", "native", 0.30, 7, seconds=0.01),
            _cut_row("cell_1", "sort", "0010", "baseline", 0.30, 7, seconds=0.02),
        ]
    )
    g = gate_table(cuts).set_index(["arm", "kind"])
    check("fold cases", int(g.loc[("native", "fold"), "cases"]), 4)
    check("fold changed", int(g.loc[("native", "fold"), "changed"]), 1)
    check("fold changed pct", float(g.loc[("native", "fold"), "changed_pct"]), 25.0)
    check("largest threshold move", float(g.loc[("native", "fold"), "d_thr_max"]), 0.5)
    check("provenance changes", int(g.loc[("native", "fold"), "prov_changed"]), 1)
    check("median speedup", float(g.loc[("native", "fold"), "speedup_median"]), 4.0)
    check("a rate over 4 cases is flagged", bool(g.loc[("native", "fold"), "rate_reportable"]), False)
    check("an unpaired baseline row is not a case", int(g.loc[("native", "sort"), "cases"]), 1)

    env = gate_by_env(cuts)
    check("env rows", len(env), 2)
    check("env fold changed", int(env[env["kind"] == "fold"]["changed"].iloc[0]), 1)

    # Likelihood: the candidate wins one, ties one, loses one.
    def _fit_row(case, sample, arm, loglik, seconds=0.01):
        return {
            "cell": "cell_0",
            "dataset": "d",
            "embedder": "e",
            "category": "c",
            "seed": 0,
            "style": "whole_image",
            "kind": "fold",
            "case": case,
            "sample": sample,
            "arm": arm,
            "n": 1000,
            "seconds": seconds,
            "loglik": loglik,
            "w_lo": 0.5,
            "mu_lo": 0.1,
            "var_lo": 0.01,
            "w_hi": 0.5,
            "mu_hi": 0.9,
            "var_hi": 0.01,
            "midpoint": 0.5,
        }

    fits = pd.DataFrame(
        [
            _fit_row("0000", "hay0", "baseline", 1.0, seconds=0.04),
            _fit_row("0000", "hay0", "native", 1.5, seconds=0.01),
            _fit_row("0000", "hay1", "baseline", 2.0, seconds=0.04),
            _fit_row("0000", "hay1", "native", 2.0, seconds=0.01),
            _fit_row("0005", "hay0", "baseline", 3.0, seconds=0.04),
            _fit_row("0005", "hay0", "native", 2.5, seconds=0.01),
        ]
    )
    f = fit_table(fits).set_index(["arm", "kind"])
    check("loglik better", int(f.loc[("native", "fold"), "better"]), 1)
    check("loglik tied", int(f.loc[("native", "fold"), "tied"]), 1)
    check("loglik worse", int(f.loc[("native", "fold"), "worse"]), 1)
    check("loglik median delta", float(f.loc[("native", "fold"), "d_loglik_median"]), 0.0)

    bench = pd.DataFrame(
        [
            {
                "sample": "s",
                "dataset": "d",
                "embedder": "e",
                "style": "w",
                "kind": "sort",
                "case": "0",
                "n": 1000,
                "resampled": 0,
                "arm": "baseline",
                "seconds": 0.10,
                "midpoint": 0.5,
                "loglik": 1.0,
            },
            {
                "sample": "s",
                "dataset": "d",
                "embedder": "e",
                "style": "w",
                "kind": "sort",
                "case": "0",
                "n": 1000,
                "resampled": 0,
                "arm": "native",
                "seconds": 0.01,
                "midpoint": 0.5,
                "loglik": 1.0,
            },
        ]
    )
    b = bench_table(bench).set_index(["arm", "n_bucket"])
    # 1000 scores lands in the "1k-5k" band; a measured size is banded by
    # decade, only a resampled one keeps its exact n as its label.
    check("bench speedup", float(b.loc[("native", "1k-5k"), "speedup_median"]), 10.0)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {', '.join(FAILURES)}")
        return 1
    print("all planted answers reproduced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
