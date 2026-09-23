#!/usr/bin/env python
"""Planted answers for ``resolution_3840.py``'s estimators.

    python selftest_3840.py

A spread is the statistic that looks plausible when computed on the wrong axis
(#3796), so every estimator here is checked on a fixture where the right answer
is planted and a wrong-axis estimator would miss it:

* ``summarise`` / ``curve``: Δ drawn with a known σ, so SE must come out at
  σ/√n and the subsampled SE at k must track σ/√k.
* ``components``: a category effect and a seed effect with *different* planted
  sds and several seeds per category, so a pooled sd cannot pass.
* ``clustered_se``: must exceed the naive SE when the category component is
  large, and agree with it when it is zero.
* ``allocation``: two environments with a known σ and cost ratio, where the
  Neyman fractions have closed forms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import resolution_3840 as R


def _deltas(n_env: int, n_cat: int, n_seed: int, sd_cat: float, sd_seed: float, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rows = []
    for e in range(n_env):
        for c in range(n_cat):
            a = rng.normal(0, sd_cat)
            for s in range(n_seed):
                rows.append((f"env{e}", f"cat{c}", s, a + rng.normal(0, sd_seed)))
    f = pd.DataFrame(rows, columns=["arm", "category", "seed", "d"])
    return f.set_index(["arm", "category", "seed"])["d"]


def main() -> int:
    # 1. SE and the subsampled curve.
    d = _deltas(4, 50, 2, 0.0, 0.04)
    s = R.summarise(d)
    assert abs(s["sd"] - 0.04) < 0.004, s
    assert abs(s["se"] - s["sd"] / np.sqrt(len(d))) < 1e-12
    for row in R.curve(d, np.random.default_rng(1), reps=2000):
        assert abs(row["se_true"] / (s["sd"] / np.sqrt(row["k"])) - 1) < 0.1, row
        assert 0.85 < row["coverage_2se"] <= 1.0, row

    # 2. Components: planted sd_cat 0.03, sd_seed 0.01, 8 seeds.
    c = R.components(_deltas(3, 60, 8, 0.03, 0.01, seed=2))
    assert abs(c["sd_seed"] - 0.01) < 0.0015, c
    assert abs(c["sd_category"] - 0.03) < 0.005, c
    assert c["icc"] > 0.8, c
    c0 = R.components(_deltas(3, 60, 8, 0.0, 0.03, seed=3))
    assert c0["icc"] < 0.05, c0

    # 3. Clustered SE: larger than naive with a category effect, equal without.
    dc = _deltas(3, 60, 8, 0.03, 0.01, seed=4)
    assert R.clustered_se(dc) > 2 * R.summarise(dc)["se"]
    d0 = _deltas(3, 60, 8, 0.0, 0.03, seed=5)
    assert abs(R.clustered_se(d0) / R.summarise(d0)["se"] - 1) < 0.2

    # 4. Allocation: equal weights, σ 1 vs 3, costs 1 vs 9.
    t = pd.DataFrame({"n": [10, 10], "sd": [1.0, 3.0], "cell_seconds": [1.0, 9.0]}, index=["a", "b"])
    al = R.allocation(t)
    # cells: (Σ W σ)² / Σ W σ² = (2)² / 5 = 0.8
    assert abs(al["cells_needed_neyman"] - 0.8) < 1e-9, al
    # cost: (Σ W σ √c)² / (Σ W σ² · Σ W c) = (0.5*1 + 0.5*9)² / (5 * 5) = 25 / 25 = 1
    assert abs(al["cost_needed_neyman_by_cost"] - 1.0) < 1e-9, al
    print("selftest_3840: all planted answers recovered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
