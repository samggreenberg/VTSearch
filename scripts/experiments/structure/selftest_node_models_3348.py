#!/usr/bin/env python
"""Planted-answer self-test for ``node_models_3348`` (#3348).

    python selftest_node_models_3348.py

Two answers are known in advance on synthetic sphere data, so the instrument
can be checked before it is pointed at the pile:

1. **Validity.**  ``knn_global`` and every ``*_split_deepest`` variant are
   split-conformal, so on a held-out draw from the build distribution their
   rate below 0.05 must sit at 0.05 within sampling error.  If one does not,
   the routing or the rank is broken, not the model.
2. **Power.**  A query set drawn from a different mixture must be called
   ``shifted`` by ``knn_global`` -- the planted shift is gross.

It also checks the bookkeeping the run relies on: ``grid_mean_shipped`` equals
the mean of ``path_pvalues`` (the #3329 consistency check), and the ``loo``
vMF node calibration equals the atlas's own leave-one-out ``t_quantiles``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "calibration"))

import common  # noqa: E402

common.setup_env()

import node_models_3348 as nm  # noqa: E402
from run_node_models_3348 import Candidates  # noqa: E402


def mixture(rng: np.random.Generator, centers: np.ndarray, n: int, spread: float) -> np.ndarray:
    lab = rng.integers(0, centers.shape[0], n)
    # Anisotropic: each cluster spreads along a few private directions, the
    # shape a single mean direction describes badly.
    x = centers[lab] + spread * rng.standard_normal((n, centers.shape[1]))
    return nm.normalize_rows(x)


def main() -> int:
    rng = np.random.default_rng(0)
    d = 64
    centers = rng.standard_normal((12, d))
    other = rng.standard_normal((12, d))
    vecs = mixture(rng, centers, 4000, 0.35)
    ids = np.arange(vecs.shape[0], dtype=np.int64)
    perm = rng.permutation(vecs.shape[0])
    hold, build = perm[:800], perm[800:]

    rows: list[dict] = []

    def add(family, scope, statistic, value, n=0):
        rows.append({"family": family, "scope": scope, "statistic": statistic, "value": value})

    c = Candidates(ids, build, vecs, 0, add)
    timing: dict[str, float] = {}
    pin = c.pvalues(vecs[hold], timing)
    pout = c.pvalues(mixture(rng, other, 800, 0.35), timing)

    failures = []
    # Binomial sd at n=800, p=0.05 is 0.0077; allow ~3 sd.
    for var in ["knn_global"] + [f"{s}_split_deepest" for s in nm.STATISTICS]:
        rate = float(np.mean(pin[var] < 0.05))
        ok = abs(rate - 0.05) <= 0.025
        print(f"validity  {var:22s} frac<0.05 = {rate:.3f}  {'ok' if ok else 'FAIL'}")
        if not ok:
            failures.append(var)
    v = nm.shift_verdict(pout["knn_global"])
    print(f"power     knn_global shifted={v['shifted']} frac={v['frac_atypical']:.3f}")
    if not v["shifted"]:
        failures.append("power")
    diff = timing["grid_mean_path_diff"]
    print(f"bookkeep  grid mean vs path mean max|diff| = {diff:.2e}")
    if diff > 1e-5:
        failures.append("grid_path")

    # loo vMF calibration == the atlas's own LOO t-quantiles (same numbers, exact ranks vs a grid).
    from vtscore.coverage.atlas import _CALIBRATION_GRID  # noqa: PLC0415

    m = c.models["vmf_loo"]
    worst = 0.0
    for name, cal in m.cal.items():
        q = np.quantile(cal, _CALIBRATION_GRID)
        worst = max(worst, float(np.max(np.abs(q - np.asarray(c.full.nodes[name]["t_quantiles"])))))
    print(f"bookkeep  vmf_loo cal vs atlas t_quantiles max|diff| = {worst:.2e}")
    if worst > 1e-4:
        failures.append("vmf_loo_cal")

    print("SELFTEST", "FAILED: " + ", ".join(failures) if failures else "PASSED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
