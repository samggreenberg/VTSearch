#!/usr/bin/env python
"""#4731's FHIBE figure: mean F-beta over clicks per arm, one panel per photo-count stratum and dataset.

Reads ``curves.csv`` from ``analyze_good_dry_4731.py``.

    python figures_good_dry_4731.py --analysis <analysis dir> --out <report dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

#: Categorical slots 1-3 of the reference palette, in fixed order; checked with the dataviz validator.
#: Each arm's end label and legend line. #4731 shipped dry run + head, so ``app`` (#4743's run of the
#: shipped app) is the same rule and wears its colour; ``ctl`` is the app before #4731 in both studies.
ARMS = {
    "ctl": ("before #4731", "#2a78d6", "before #4731 (the Good phase waits for 3 Goods)"),
    "dry16": ("dry run", "#1baf7a", "dry run: the Good phase also ends after 16 misses with a Good in hand"),
    "dry16q16": ("dry run + head", "#eb6834", "dry run + head: and 1 Good with 16 Bads gets the trained head"),
    "app": ("today", "#eb6834", "today: the dry run and the trained head at 1 Good + 16 Bads, shipped by #4731"),
}
DATASETS = {
    "fhibe_1024": "whole photo (SigLIP)",
    "fhibe_faces_1024": "face crop, 1024 px (FaceNet)",
    "fhibe_faces_640": "face crop, 640 px (FaceNet)",
}
STRATA = {"2-4": "2-4 photos", "5-6": "5-6 photos", "7-": "7+ photos"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", default="fig_fhibe_curves.png")
    a = ap.parse_args()
    cur = pd.read_csv(a.analysis / "curves.csv")
    arms = {k: v for k, v in ARMS.items() if k in set(cur["arm"])}
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.5), sharex=True, sharey=True)
    for i, (st, st_name) in enumerate(STRATA.items()):
        for j, (ds, ds_name) in enumerate(DATASETS.items()):
            ax = axes[i, j]
            ends = []
            for arm, (name, color, _) in arms.items():
                g = cur[(cur["arm"] == arm) & (cur["stratum"] == st) & (cur["dataset"] == ds)]
                if g.empty:
                    continue
                ax.plot(g["t"], g["fbeta"], color=color, lw=2)
                ends.append([float(g["fbeta"].iloc[-1]), name, color])
            # Labels at the lines' ends in the right-hand column, nudged apart where they would overlap
            # (no leader lines); the legend above names the arms for the other columns.
            if j < len(DATASETS) - 1:
                ends = []
            ends.sort()
            for k in range(1, len(ends)):
                if ends[k][0] - ends[k - 1][0] < 0.07:
                    ends[k][0] = ends[k - 1][0] + 0.07
            for y, name, color in ends:
                ax.text(152, y, name, color=INK, fontsize=7.5, va="center")
                ax.plot([149, 151], [y, y], color=color, lw=2)
            ax.set_xlim(0, 150)
            ax.set_ylim(0, 1)
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["left", "bottom"]].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=8)
            if i == 0:
                ax.set_title(ds_name, fontsize=10, color=INK)
            if j == 0:
                ax.set_ylabel(f"{st_name}\nmean F-beta", fontsize=9, color=INK)
            if i == 2:
                ax.set_xlabel("click (the example is click 1)", fontsize=9, color=MUTED)
    fig.legend(
        handles=[plt.Line2D([], [], color=c, lw=2) for _, c, _ in arms.values()],
        labels=[legend for _, _, legend in arms.values()],
        loc="upper center",
        ncol=1,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.45, 1.02),
    )
    fig.subplots_adjust(right=0.88, wspace=0.12, hspace=0.18, top=0.9)
    a.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out / a.name, dpi=150, bbox_inches="tight")
    print(f"wrote {a.out / a.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
