#!/usr/bin/env python
"""The objective over clicks per arm (#4427 / #4428): F-beta of the withheld set above the app's threshold, the
unchecked line at each click (curves.csv thr_fbeta) with the post-check value marked at the end.

    python figs_objective.py --beta 1 --out fig.png name=<analysis dir> [name=<analysis dir> ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PALETTE = ["#555555", "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--title", default=None)
    ap.add_argument("runs", nargs="+", help="name=analysis dir")
    args = ap.parse_args()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, spec in enumerate(args.runs):
        name, _, d = spec.partition("=")
        d = Path(d)
        curves = pd.read_csv(d / "curves.csv")
        cells = pd.read_csv(d / "cells.csv")
        cells = cells[~cells["never_trained"].astype(bool)]
        m = curves.groupby("t")["thr_fbeta"].mean()
        color = PALETTE[i % len(PALETTE)]
        ax.plot(m.index, m.to_numpy(), color=color, lw=2, label=f"{name}: unchecked line")
        end = float(cells["thr_fbeta_final"].mean())
        ax.plot([m.index.max() + 3], [end], marker="D", markersize=7, color=color)
        ax.annotate(f"{end:.2f} after the check", (m.index.max() + 3, end), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=color)  # fmt: skip
    ax.set_ylim(0, 0.75)
    ax.set_xlim(0, max(ax.get_xlim()[1], 150) * 1.3)
    ax.set_xlabel("clicks")
    ax.set_ylabel(f"F{args.beta:g} of the withheld set above the threshold")
    ax.set_title(
        args.title or f"beta {args.beta:g}: the objective over clicks (mean over runs)", loc="left", fontsize=10
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=120)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
