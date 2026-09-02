#!/usr/bin/env python3
"""Figures for #3521, generated from ``summary.json`` by the analysis job.

Two, because the study has two claims:

1. **bar_error_by_branch.png** — the fraction of the progress bar each arm
   budgets to the wrong step, one panel per task, bars grouped by the branch the
   held-out runs took. This is the figure that says a profile is not wrong in
   general: it is wrong about a branch it never measured.
2. **observed_vs_predicted.png** — predicted against observed seconds per step,
   coloured by arm, log-log with the identity line. A point far below the line
   is a step the profile thinks is free.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: One colour per arm, stable across both figures so a reader learns them once.
_ARM_COLOURS = {"old": "#c2532f", "new": "#2f6fc2", "shipped": "#8a8a8a"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp", required=True)
    args = ap.parse_args()
    exp = Path(args.exp)
    summary = json.loads((exp / "summary.json").read_text())
    figdir = exp / "figures"
    figdir.mkdir(exist_ok=True)

    # --- 1. bar error by branch --------------------------------------------
    grouped: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for key, vals in summary["by_arm_branch"].items():
        arm, task, branch = key.split("|", 2)
        if vals["bar_error"] is not None:
            grouped[task][branch][arm] = vals["bar_error"]
    tasks = sorted(grouped)
    if tasks:
        fig, axes = plt.subplots(1, len(tasks), figsize=(4.2 * len(tasks), 4.0), squeeze=False)
        for ax, task in zip(axes[0], tasks):
            branches = sorted(grouped[task])
            width = 0.26
            for offset, arm in enumerate(("old", "new", "shipped")):
                xs = [i + (offset - 1) * width for i in range(len(branches))]
                ys = [grouped[task][b].get(arm, 0.0) for b in branches]
                ax.bar(xs, ys, width=width, label=arm, color=_ARM_COLOURS[arm])
            ax.set_xticks(range(len(branches)))
            ax.set_xticklabels([b.replace("=", "\n") for b in branches], fontsize=7)
            ax.set_title(task, fontsize=10)
            ax.set_ylabel("fraction of the bar in the wrong step")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)
        axes[0][0].legend(fontsize=8, title="profile")
        fig.suptitle(
            "Bar error by the branch the held-out runs took "
            f"(device {summary['device']}); lower is better, 1.0 = every second budgeted to the wrong step",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(figdir / "bar_error_by_branch.png", dpi=130)
        plt.close(fig)

    # --- 2. observed vs predicted ------------------------------------------
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    drawn = False
    for arm in ("shipped", "old", "new"):
        xs, ys = [], []
        for rec in summary["records"]:
            if rec["arm"] != arm:
                continue
            for step, err in rec["steps"].items():
                observed = rec.get("observed", {}).get(step)
                if observed:
                    xs.append(observed)
                    ys.append(max(1e-4, observed * (1 + err)))
        if xs:
            ax.scatter(xs, ys, s=14, alpha=0.6, label=arm, color=_ARM_COLOURS[arm])
            drawn = True
    if drawn:
        lo, hi = 1e-2, 1e3
        ax.plot([lo, hi], [lo, hi], color="#333", lw=1, ls="--", label="perfect")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("observed seconds")
        ax.set_ylabel("predicted seconds (magnitude of the error)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "observed_vs_predicted.png", dpi=130)
    plt.close(fig)

    print(f"wrote figures to {figdir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
