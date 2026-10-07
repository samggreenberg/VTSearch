#!/usr/bin/env python3
"""#4584: the per-cell SD of the paired Δ-objective in ``analyze_ab``'s decision window.

Scope ``app_visible``, all steps (``n_votes >= 2``), each arm's own rows, one
mean per run, at the arms' own beta; paired on (category, seed).  This is the σ
``preflight.sh --resolve-delta`` sizes a balance A/B with.

    python sigma_objective_4584.py --pair DIR_A=DIR_B [--pair ...] --out sigma.csv

Each DIR is an arm directory holding ``results/cells``.  Results for the
committed defaults: ``docs/experiments/2026-10-07-objective-sigma-4584/``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import common

common.setup_env()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WINDOWS = {"all_steps": (2, 10**6), "ramp_6_20": (6, 20), "post_ramp_21_plus": (21, 10**6)}


def run_means(arm: Path) -> tuple[float, dict[str, pd.Series]]:
    import _cells_io  # noqa: PLC0415
    from vtscore.eval.calibration_metrics import fbeta_from_rates  # noqa: PLC0415

    f, prov = _cells_io.load_arm(arm / "results")
    print(f"{arm.name}: {_cells_io.describe_load(prov)}", flush=True)
    f = f[f["app_trained"].fillna(0).astype(int) == 1].copy()
    f["n_votes"] = f["n_good"] + f["n_bad"]
    betas = pd.to_numeric(f["beta"], errors="coerce").dropna().unique()
    if len(betas) != 1:
        raise SystemExit(f"{arm}: expected one beta, found {sorted(betas)}")
    beta = float(betas[0])
    f["objective"] = fbeta_from_rates(f["precision"].to_numpy(float), f["recall"].to_numpy(float), beta)
    return beta, {
        w: f[(f["n_votes"] >= lo) & (f["n_votes"] <= hi)].groupby(["category", "seed"])["objective"].mean()
        for w, (lo, hi) in WINDOWS.items()
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", action="append", required=True, help="DIR_A=DIR_B; Δ = A - B")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(list(argv) if argv is not None else None)
    rows = []
    for spec in args.pair:
        a_dir, b_dir = (Path(x) for x in spec.split("=", 1))
        ba, a = run_means(a_dir)
        bb, b = run_means(b_dir)
        if ba != bb:
            raise SystemExit(f"{spec}: the arms ran at different betas ({ba}, {bb})")
        for w in WINDOWS:
            d = (a[w] - b[w]).dropna()
            sd = float(d.std(ddof=1)) if len(d) > 1 else np.nan
            rows.append(
                {
                    "a": a_dir.name,
                    "b": b_dir.name,
                    "beta": ba,
                    "window": w,
                    "n": len(d),
                    "mean": float(d.mean()) if len(d) else np.nan,
                    "sd": sd,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.4g")
    print(out.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
