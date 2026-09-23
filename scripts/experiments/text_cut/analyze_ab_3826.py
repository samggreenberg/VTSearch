#!/usr/bin/env python
"""The pre-registered ship decision for the #3826 trajectory A/B.

    python analyze_ab_3826.py --on <guarded results> --off <midpoint results> --out <analysis>

Reads ``calibration/analyze_ab.py``'s per-cell frame (``agg/ab_paired_cells.csv``
under ``--out``).  That frame holds one row per (scope, window, cell), with each
metric's mean for the ON (``guarded_tail``) and OFF (``gmm_midpoint``) run.  It
applies the rule posted on issue #3826 *before* the arrays were submitted:

    PRIMARY    scope app_visible, window all_steps, metric cost, d = ON - OFF per
               (dataset, category, seed) cell:  ship iff mean(d) + 2 SE <= +0.010
    SECONDARY  scope app_visible, window ramp_6_20 (the first detectors, whose
               votes the opening chose):         mean(d) + 2 SE <= +0.020
    Ship the guarded line as the default only if BOTH hold.  Otherwise it stays
    behind VTSEARCH_TEXT_SORT_CUT, off by default.

SE is the standard error over cells (the independent units: the two arms vote
on different items, so steps within a cell are not independent).  Everything
else it writes - AP, FNR, FPR, per-dataset rows, the curve - is for reading,
not deciding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MARGIN_PRIMARY = 0.010
MARGIN_SECONDARY = 0.020


def _stat(d: np.ndarray) -> dict:
    d = d[np.isfinite(d)]
    n = int(d.size)
    se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return {
        "n_cells": n,
        "mean": float(d.mean()) if n else float("nan"),
        "se": se,
        "sd": float(d.std(ddof=1)) if n > 1 else float("nan"),
    }


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(list(argv) if argv is not None else None)
    out = Path(args.out)
    cells = pd.read_csv(out / "agg" / "ab_paired_cells.csv")
    cells["dataset"] = cells["arm"].str.split("/").str[0]

    rows = []
    for (scope, window), g in cells.groupby(["scope", "window"]):
        for metric in ("cost", "fnr", "fpr", "average_precision", "auroc"):
            if f"{metric}_on" not in g.columns:
                continue
            s = _stat((g[f"{metric}_on"] - g[f"{metric}_off"]).to_numpy(float))
            rows.append({"scope": scope, "window": window, "metric": metric, "dataset": "all", **s})
            for ds, h in g.groupby("dataset"):
                s = _stat((h[f"{metric}_on"] - h[f"{metric}_off"]).to_numpy(float))
                rows.append({"scope": scope, "window": window, "metric": metric, "dataset": ds, **s})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(out / "ab_3826_deltas.csv", index=False)

    def pick(window: str) -> dict:
        r = tbl[(tbl.scope == "app_visible") & (tbl.window == window) & (tbl.metric == "cost") & (tbl.dataset == "all")]
        return r.iloc[0].to_dict() if len(r) else {}

    prim, sec = pick("all_steps"), pick("ramp_6_20")
    prim_ok = bool(prim) and prim["mean"] + 2 * prim["se"] <= MARGIN_PRIMARY
    sec_ok = bool(sec) and sec["mean"] + 2 * sec["se"] <= MARGIN_SECONDARY
    decision = {
        "rule": "ship iff PRIMARY (app_visible/all_steps cost: mean+2SE <= +0.010) AND SECONDARY (app_visible/ramp_6_20 cost: mean+2SE <= +0.020)",
        "primary": {**prim, "upper_2se": prim.get("mean", np.nan) + 2 * prim.get("se", np.nan), "passes": prim_ok},
        "secondary": {**sec, "upper_2se": sec.get("mean", np.nan) + 2 * sec.get("se", np.nan), "passes": sec_ok},
        "ship_as_default": bool(prim_ok and sec_ok),
    }
    (out / "ab_3826_decision.json").write_text(json.dumps(decision, indent=1, default=float))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print(tbl[(tbl.scope == "app_visible") & (tbl.dataset == "all")].to_string(index=False))
    print(json.dumps(decision, indent=1, default=float))

    # The curve behind the windows: mean cost per vote count, both arms, and the paired difference.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cur = pd.read_csv(out / "agg" / "ab_curves_vs_votes.csv")
        cur["dataset"] = cur["arm"].str.split("/").str[0]
        fig, axes = plt.subplots(1, cur["dataset"].nunique(), figsize=(13, 3.8), squeeze=False, sharey=True)
        colors = {"guarded_tail": "#2a78d6", "gmm_midpoint": "#e34948"}
        for ax, (ds, g) in zip(axes[0], cur.groupby("dataset")):
            for run, h in g.groupby("run"):
                h = h.groupby("n_votes")["cost"].mean()
                label = "guarded_tail" if run == "safe_on" else "gmm_midpoint"
                ax.plot(h.index, h.values, color=colors[label], lw=2, label=label)
            ax.set_title(ds, fontsize=9)
            ax.set_xlabel("votes")
        axes[0][0].set_ylabel("cost (FPR+FNR), mean over cells")
        axes[0][0].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        fig.savefig(out / "ab_3826_cost_vs_votes.png", dpi=130)
    except Exception as exc:  # noqa: BLE001 - figures are for reading, the decision is above
        print(f"figure skipped: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
