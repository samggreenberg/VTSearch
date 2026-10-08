#!/usr/bin/env python
"""Figures for the acquisition pricing (#4409 / #4413 step 5): per beta, the three arms over clicks.

Left: positives found (Goods) at 25 / 50 / 100 / 150 clicks.  Middle: the returned set's F-beta at its own beta over
clicks.  Right: its share of the best cut's F-beta over clicks.  Means over the trained cells (5 seeds x 144 cells).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path("/expscratch/sgreenberg/p-aware-acq-4409")
ARMS = [("ctl", "line − 4 (shipped)", "#555555"), ("1.0", "argmax ×1.0", "#1f77b4"), ("0.5", "argmax ×0.5", "#d62728")]


def fig(beta: str, out: Path) -> None:
    f, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for arm, label, color in ARMS:
        d = ROOT / f"b{beta}-x{arm}" / "analysis-binary"
        if not (d / "cells.csv").exists():
            continue
        c = pd.read_csv(d / "cells.csv")
        c = c[~c["never_trained"].astype(bool)]
        ts = [10, 25, 50, 100, 150]
        axes[0].plot(ts, [c[f"goods_{t}"].mean() for t in ts], "o-", color=color, label=label)
        s = pd.read_csv(d / "balance_steps.csv")
        s = s[s["beta"].sub(float(beta)).abs() < 1e-9]
        g = s.groupby("t")[["fbeta", "fb_share"]].mean()
        axes[1].plot(g.index, g["fbeta"], "-", color=color, label=label)
        axes[2].plot(g.index, g["fb_share"], "-", color=color, label=label)
    axes[0].set(title="positives found (Goods)", xlabel="clicks", ylabel="mean over cells")
    axes[1].set(title=f"F{beta} at the line", xlabel="clicks")
    axes[2].set(title=f"share of the best F{beta} cut", xlabel="clicks", ylim=(0.5, 1.0))
    axes[0].legend(fontsize=8, loc="upper left")
    f.suptitle(f"beta {beta}: acquisition at the F-beta argmax vs the shipped line − 4 (Binary, 5 seeds)")
    f.tight_layout()
    f.savefig(out, dpi=110)
    plt.close(f)


if __name__ == "__main__":
    for b in sys.argv[1:] or ["0.5", "1", "2"]:
        fig(b, ROOT / f"acq_beta{b}.png")
        print("wrote", ROOT / f"acq_beta{b}.png")
