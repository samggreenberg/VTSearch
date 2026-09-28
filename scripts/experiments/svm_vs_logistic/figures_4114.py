"""#4114 figure: the paired head difference over clicks, one panel per scale band.

The averaged quality-over-clicks pair (``curves.py``) hides the study's finding,
which is that the converged logistic head's gain over the SVM lives in one band.
This draws ``other - svm`` per click for cost, oracle cost and AP, paired on the
cell (category, seed) at each click, with a +-2 SE band clustered on category -
the same pairing and clustering as ``analyze_stage_b.py``.

    python figures_4114.py --root /expscratch/$USER/logreg-4114 --out docs/experiments/<study>/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "calibration"))

KEYS = ["dataset", "embedder", "category", "seed"]
METRICS = (("cost", "cost"), ("oracle_cost", "oracle cost (ranking)"), ("average_precision", "average precision"))
BANDS = ("small", "medium", "large")
COLORS = {"lrconv": "#d9730d", "linear": "#2e8b57"}


def load(root: Path, arms: list[str]) -> pd.DataFrame:
    import _cells_io

    parts = []
    for arm in ["svm", *arms]:
        df, _ = _cells_io.load_arm(root / arm / "results")
        df = df[[*KEYS, "t", *(m for m, _ in METRICS)]].copy()
        df["arm"] = arm
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df["band"] = df["category"].astype(str).str.extract(r"@(\w+)$", expand=False)
    return df


def paired_by_click(df: pd.DataFrame, metric: str, arm: str) -> pd.DataFrame:
    """Per click: mean of ``arm - svm`` over cells measured by both, and its clustered SE."""
    w = df.pivot_table(index=[*KEYS, "t"], columns="arm", values=metric)
    x = (w[arm] - w["svm"]).dropna().rename("d").reset_index()
    g = x.groupby(["t", "dataset", "embedder", "category"])["d"].mean().reset_index()
    out = g.groupby("t")["d"].agg(["mean", "std", "count"])
    out["se"] = out["std"] / np.sqrt(out["count"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", default="lrconv,linear")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arms = [a for a in args.arms.split(",") if a]
    df = load(args.root, arms)
    args.out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(METRICS), len(BANDS), figsize=(11, 8), sharex=True, sharey="row")
    rows = []
    for i, (metric, label) in enumerate(METRICS):
        for j, band in enumerate(BANDS):
            ax = axes[i, j]
            sub = df[df["band"] == band]
            for arm in arms:
                s = paired_by_click(sub, metric, arm)
                s = s[s["count"] >= 10]  # a click few categories reach is not a mean
                ax.plot(s.index, s["mean"], color=COLORS.get(arm), lw=1.4, label=f"{arm} − svm")
                ax.fill_between(
                    s.index, s["mean"] - 2 * s["se"], s["mean"] + 2 * s["se"], color=COLORS.get(arm), alpha=0.2, lw=0
                )
                rows.append(s.assign(metric=metric, band=band, arm=arm).reset_index())
            ax.axhline(0, color="black", lw=0.8)
            if i == 0:
                ax.set_title(f"{band} band", fontsize=10)
            if j == 0:
                ax.set_ylabel(f"Δ {label}", fontsize=9)
            if i == len(METRICS) - 1:
                ax.set_xlabel("clicks")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Paired difference from the shipped SVM, per scale band (±2 SE, clustered on category)\n"
        "cost rows: below 0 = the other head is better · AP row: above 0 = the other head is better",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out / "paired_delta_by_band.png", dpi=130)
    pd.concat(rows).to_csv(args.out / "paired_delta_by_band.csv", index=False)
    print(f"wrote {args.out / 'paired_delta_by_band.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
