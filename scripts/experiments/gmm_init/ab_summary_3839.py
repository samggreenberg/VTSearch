#!/usr/bin/env python
"""Pool ``analyze_ab.py``'s paired cells into the #3825-style A/B line (#3839).

    python ab_summary_3839.py --paired <ab dir>/agg/ab_paired_cells.csv --out <dir>

``analyze_ab.py`` reports per environment and window; a decision is taken on the
pooled paired mean over every cell at ``app_visible``/``all_steps`` with
SE = sd/√n - the line #3825 quoted and #3840 validated.  Also written: the
share of cells whose Δ is exactly zero (a raised cap only acts on a trajectory
in which some fold reaches 200 iterations, so most cells should not move at
all - a check that the two grids are the same harness), and the same line per
environment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("cost", "regret", "fnr", "fpr", "average_precision")


def line(f: pd.DataFrame) -> dict:
    out: dict = {"n": len(f)}
    for m in METRICS:
        d = (f[f"{m}_on"] - f[f"{m}_off"]).to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        out[m] = float(d.mean())
        out[f"{m}_se"] = float(d.std(ddof=1) / np.sqrt(d.size))
    d = (f["cost_on"] - f["cost_off"]).to_numpy(dtype=float)
    out["cost_sd"] = float(d.std(ddof=1))
    out["frac_zero"] = float(np.mean(np.abs(d) < 1e-12))
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--paired", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    p = pd.read_csv(args.paired)
    p = p[(p["scope"] == "app_visible") & (p["window"] == "all_steps")]
    summary = {"pooled": line(p), "by_env": {a: line(g) for a, g in p.groupby("arm")}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ab_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["pooled"], indent=2))
    for a, s in summary["by_env"].items():
        print(f"{a:44s} n={s['n']:3d} Δcost {s['cost']:+.4f} ± {s['cost_se']:.4f}  zero {s['frac_zero']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
