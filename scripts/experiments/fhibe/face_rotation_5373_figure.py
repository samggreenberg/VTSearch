#!/usr/bin/env python
"""Figures for the #5373 report: what each quality cut and minimum gain buys (cuts.csv), and the
bench's paired gain after the check for the people the retry touched (paired.csv).

    python face_rotation_5373_figure.py docs/experiments/2026-11-06-face-rotation-5373
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    d = Path(sys.argv[1])
    t = pd.read_csv(d / "cuts.csv")
    t = t[t["size"] == 1024]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    for gain, colour in ((0, "#9aa0a6"), (2, "#1a73e8"), (5, "#e37400")):
        g = t[t["gain"] == gain]
        a.plot(g["cut"], g["sum_dap"], "o-", color=colour, label=f"min gain {gain}")
        b.plot(g["cut"], g["worse_cos"], "o-", color=colour, label=f"min gain {gain}")
    a.axhline(t["sum_dap"].max(), color="#5f6368", lw=0.8, ls=":")
    a.set(
        xlabel="retry faces under Face Quality",
        ylabel="summed AP gain over own photos",
        title="Gain kept (1024 px, 9,986 faces)",
    )
    b.set(
        xlabel="retry faces under Face Quality",
        ylabel="faces that match worse after the switch",
        title="Switches that hurt",
    )
    for ax in (a, b):
        ax.axvline(18, color="#188038", lw=0.8)
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(d / "cuts.webp", dpi=110)

    p = pd.read_csv(d / "paired.csv")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cells = [(k, s) for s in ("1024", "640") for k in (1, 4)]
    for j, (beta, colour) in enumerate(((0.25, "#9aa0a6"), (1.0, "#1a73e8"), (4.0, "#e37400"))):
        for people, x0, alpha in (("touched", 0.0, 1.0), ("untouched", 0.3, 0.45)):
            r = [
                p[(p.K == k) & (p["size"].astype(str) == s) & (p.beta == beta) & (p.people == people)].iloc[0]
                for k, s in cells
            ]
            xs = [i * 1.0 + (j - 1) * 0.1 + x0 for i in range(len(cells))]
            ax.errorbar(
                xs,
                [q["d_thr_fbeta_final"] for q in r],
                yerr=[q["se_thr_fbeta_final"] for q in r],
                fmt="o",
                color=colour,
                alpha=alpha,
                capsize=2,
                label=f"\u03b2 {beta:g}" if people == "touched" else None,
            )
    ax.axhline(0, color="#5f6368", lw=0.8)
    ax.set_xticks([i + 0.15 for i in range(len(cells))], [f"K {k}, {s} px" for k, s in cells])
    ax.set(
        ylabel="F-beta after the check, rot \u2212 ctl",
        title="People with a turned face (solid) and without (faint), \u00b1 1 SE",
    )
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(d / "bench.webp", dpi=110)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
