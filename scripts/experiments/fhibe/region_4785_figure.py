#!/usr/bin/env python
"""#4785's Region check: Find's F-beta and AP at every vote, plain SVM against the head start.

Reads ``prior_4776.py``'s ``prior_curves.csv`` for the Region pair and writes ``fig_region_by_vote.png``:
two panels (F-beta at beta 1, AP), the plain SVM in grey and the head started from the Goods' centroid
in the deck's blue, every line labelled at its end. ``--scale`` multiplies both curves (runs with rows /
all runs: a run that never finds a Good gives Test nothing under either head, so it counts as 0).

Usage::

    python region_4785_figure.py --curves <prior_curves.csv> --set region-b1 --scale 0.969 --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
GREY, BLUE = "#8a929e", "#2F6DB5"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--set", default="region-b1")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": INK2})  # fmt: skip
    c = pd.read_csv(args.curves)
    c = c[c["set"] == args.set]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, (metric, title) in zip(
        axes, (("F", "What a Find returns: F-beta, beta 1"), ("AP", "AP of the withheld half"))
    ):
        s = c[c["metric"] == metric].sort_values("vote")
        k = args.scale if metric == "F" else 1.0
        ax.plot(s["vote"], s["control"] * k, color=GREY, lw=2)
        ax.plot(s["vote"], s["arm"] * k, color=BLUE, lw=2.2)
        ends = sorted(
            [(float(s["control"].iloc[-1]) * k, "plain SVM", GREY), (float(s["arm"].iloc[-1]) * k, "head start", BLUE)]
        )
        lo, hi = ax.get_ylim()
        gap = 0.06 * (hi - lo)
        ys = [ends[0][0], max(ends[1][0], ends[0][0] + gap)]
        for (_, label, color), y in zip(ends, ys):
            ax.text(152, y, label, color=color, va="center", fontsize=10)
        ax.set_xlim(0, 150)
        ax.set_xlabel("vote")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(axis="y", color="#e4e3df", lw=0.8)
    fig.suptitle("COCO Better, Region path (DINOv3 patches), beta 1: 288 runs a head")
    fig.tight_layout(rect=(0, 0, 0.95, 0.94))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig_region_by_vote.png", dpi=150)
    print(f"wrote {args.out / 'fig_region_by_vote.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
