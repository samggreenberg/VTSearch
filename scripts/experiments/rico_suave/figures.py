#!/usr/bin/env python
"""Rico Suave v0's curve figure (#4356): what Find returns at every vote, on app screens and on photos.

Reads ``curves.csv`` from ``analyze.py`` and writes ``fig_fbeta_by_vote.png``: one panel per balance
preset. The thick blue line is every Rico Suave class pooled, the faint blue lines are the classes one
by one, and the grey dashed line is the coco_better reference read by the same code. Lines are
labelled at their right ends, not through leader lines (slides/STYLE.md).

Usage::

    python figures.py --curves <analysis>/curves.csv --out <dir>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, GREY = "#2a78d6", "#8a8984"
PRESETS = {"b025": "balance 1/4", "b1": "balance 1", "b4": "balance 4"}


def _spread(ys: list[float], gap: float) -> list[float]:
    """Nudge end-label heights apart by at least *gap*, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < gap:
            out[b] = out[a] + gap
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    curves = pd.read_csv(args.curves)
    presets = [p for p in PRESETS if p in set(curves["preset"])]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
        }
    )
    fig, axes = plt.subplots(1, len(presets), figsize=(5.2 * len(presets), 4.2), sharey=True, squeeze=False)
    for ax, p in zip(axes[0], presets):
        g = curves[curves["preset"] == p]
        rico = g[g["dataset"] == "rico_suave"]
        for _, d in rico[rico["category"] != "all"].groupby("category"):
            d = d.sort_values("t")
            ax.plot(d["t"], d["fbeta"], color=BLUE, lw=0.8, alpha=0.25)
        ends = []
        for ds, colour, ls, name in (
            ("rico_suave", BLUE, "-", "app screens"),
            ("coco_better", GREY, "--", "photos (coco_better)"),
        ):
            d = g[(g["dataset"] == ds) & (g["category"] == "all")].sort_values("t")
            if d.empty:
                continue
            ax.plot(d["t"], d["fbeta"], color=colour, ls=ls, lw=2.2)
            ends.append((float(d["fbeta"].iloc[-1]), name, colour))
        for y, (_, name, colour) in zip(_spread([e[0] for e in ends], 0.06), ends):
            ax.text(152, y, name, color=INK2, va="center", fontsize=9)
            ax.plot([149.5], [y], marker="s", ms=4, color=colour, clip_on=False)
        ax.set_title(PRESETS[p], color=INK)
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 1)
        ax.set_xlabel("votes")
        ax.grid(True, color=GRID, lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0][0].set_ylabel("F-beta of what Find returns (withheld half)")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig_fbeta_by_vote.png", dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out / 'fig_fbeta_by_vote.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
