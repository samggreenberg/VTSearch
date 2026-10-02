#!/usr/bin/env python
"""The owner's metric, literally (#4427): F-beta of the withheld images above the app's threshold.

Each session row carries ``precision`` / ``recall`` of the withheld test set at that step's threshold.  The
last ordinary row is the unchecked line's threshold; the last ``check`` row is the threshold after the walk.
Per arm: mean F-beta before and after the check, the walk's paired effect, and the count above the threshold.

    python withheld_at_threshold.py --beta 1 name=<run dir> [name=<run dir> ...]
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd


def fbeta(p: float, r: float, beta: float) -> float:
    if not (p > 0 or r > 0) or np.isnan(p) or np.isnan(r):
        return 0.0 if not (np.isnan(p) or np.isnan(r)) else float("nan")
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) > 0 else 0.0


def per_cell(run: Path, beta: float) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(run / "results" / "cells" / "task_*.csv"))):
        if "__" in Path(f).name:
            continue
        d = pd.read_csv(f, low_memory=False)
        d = d[d["gmm_variant"].isna() & (d["pool_variant"] == "max")] if "gmm_variant" in d else d
        if d.empty or "precision" not in d:
            continue
        d = d.sort_values("t")
        ordinary = d[d["phase"].astype(str) != "check"]
        check = d[d["phase"].astype(str) == "check"]
        if ordinary.empty:
            continue
        before = ordinary.iloc[-1]
        after = check.iloc[-1] if not check.empty else before
        n_pos, n_neg = float(before.get("n_test_pos", np.nan)), float(before.get("n_test_neg", np.nan))
        rows.append(
            {
                "category": before["category"],
                "seed": int(before["seed"]),
                "f_before": fbeta(float(before["precision"]), float(before["recall"]), beta),
                "f_after": fbeta(float(after["precision"]), float(after["recall"]), beta),
                "p_after": float(after["precision"]),
                "r_after": float(after["recall"]),
                "k_before": float(before["recall"]) * n_pos + float(before["fpr"]) * n_neg,
                "k_after": float(after["recall"]) * n_pos + float(after["fpr"]) * n_neg,
                "checked": int(not check.empty),
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("runs", nargs="+", help="name=dir")
    args = ap.parse_args()
    tables = {}
    for spec in args.runs:
        name, _, d = spec.partition("=")
        tables[name] = per_cell(Path(d), args.beta)
    ctl_name = next(iter(tables))
    ctl = tables[ctl_name]
    print(
        f"| arm | cells | F{args.beta:g} above the threshold, before the check | after the check | walk's effect (paired) | precision after | recall after | returned before | returned after |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, t in tables.items():
        eff = t["f_after"] - t["f_before"]
        print(
            f"| {name} | {len(t)} | {t['f_before'].mean():.3f} | **{t['f_after'].mean():.3f}** | {eff.mean():+.3f} ± {eff.std(ddof=1) / math.sqrt(len(eff)):.3f} | "
            f"{t['p_after'].mean():.3f} | {t['r_after'].mean():.3f} | {t['k_before'].mean():.1f} | {t['k_after'].mean():.1f} |"
        )
    print()
    print(f"Paired against {ctl_name}, F{args.beta:g} after the check:")
    for name, t in tables.items():
        if name == ctl_name:
            continue
        m = ctl.merge(t, on=["category", "seed"], suffixes=("_c", "_a"))
        diff = m["f_after_a"] - m["f_after_c"]
        print(f"  {name}: {diff.mean():+.4f} ± {diff.std(ddof=1) / math.sqrt(len(diff)):.4f} (n={len(diff)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
