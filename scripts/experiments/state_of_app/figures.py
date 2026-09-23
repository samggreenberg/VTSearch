#!/usr/bin/env python3
"""State of the App (#4159): the report's figures, from ``analyze.py``'s tables.

    python figures.py --analysis <exp>/analysis --out <report dir>/figures

* ``cost_over_clicks.png`` / ``f1_over_clicks.png`` -- the mean curve per path,
  click 0 (text only) to 150, with each path's full-label ceiling dashed. Two
  figures, not one with two y-axes: cost and F1 are different measures.
* ``per_cell.png`` -- every class x band: text only (hollow), after the clicks
  (filled), full labels (tick), one panel per path.

Colours are the dataviz reference palette's first two categorical slots, which
validate for both CVD and normal vision (validate_palette.js, 2026-09-23).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"SigLIP binary": "#2a78d6", "DINOv3 region": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def _axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def over_clicks(curves: pd.DataFrame, cells: pd.DataFrame, metric: str, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _axes(ax)
    mean = curves.groupby(["arm", "t"])[metric].mean().unstack(0)
    for arm, color in COLORS.items():
        if arm not in mean:
            continue
        ax.plot(mean.index, mean[arm], color=color, linewidth=2)
        ax.annotate(
            f"{arm}  {mean[arm].iloc[-1]:.2f}",
            (mean.index[-1], mean[arm].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )
        if metric == "cost":
            ceil = cells[cells["arm"] == arm]["ceiling_cost"].mean()
            ax.axhline(ceil, color=color, linewidth=1.2, linestyle="--")
            ax.annotate(
                f"full labels {ceil:.2f}", (2, ceil), xytext=(0, 4), textcoords="offset points", color=INK, fontsize=8
            )
    t0 = mean.iloc[0].mean()
    ax.plot([0], [t0], marker="o", markersize=8, color=INK, zorder=5)
    ax.annotate(f"text only {t0:.2f}", (0, t0), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=8)
    ax.set_xlim(0, 175)
    ax.set_xlabel("clicks", color=INK)
    label = {"cost": "cost at the shipped cut (lower is better)", "f1": "F1 at the shipped cut (higher is better)"}[
        metric
    ]
    ax.set_ylabel(label, color=INK)
    n = cells.groupby("arm").size().to_dict()
    ax.set_title(
        f"Mean {metric} over clicks  (runs: " + ", ".join(f"{k} {v}" for k, v in n.items()) + ")",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def per_cell(cells: pd.DataFrame, out: Path) -> None:
    cells = cells.copy()
    order = cells.groupby("category")["final_cost"].mean().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.28 * len(order) + 1.4), sharey=True, facecolor=SURFACE)
    for ax, (arm, color) in zip(axes, COLORS.items(), strict=True):
        _axes(ax)
        a = cells[cells["arm"] == arm].set_index("category").reindex(order)
        y = range(len(order))
        for yi, (tc, fc, cc) in zip(
            y, a[["text_cost", "final_cost", "ceiling_cost"]].itertuples(index=False), strict=True
        ):
            ax.plot([cc, fc], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.scatter(
            a["text_cost"], y, s=36, facecolors="none", edgecolors=MUTED, linewidths=1.2, zorder=2, label="text only"
        )
        ax.scatter(a["final_cost"], y, s=40, color=color, zorder=3, label="after 150 clicks")
        ax.scatter(a["ceiling_cost"], y, s=60, marker="|", color=INK, zorder=3, label="full labels")
        never = a[a["never_trained"].astype("boolean").fillna(False).astype(bool)]
        for cat in never.index:
            ax.annotate(
                "never found a positive",
                (never.loc[cat, "text_cost"], order.index(cat)),
                xytext=(-10, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=7,
                color=INK,
            )
        ax.set_title(arm, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("cost (lower is better)", color=INK)
        ax.set_xlim(0, 1)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=8)
    axes[0].invert_yaxis()
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.analysis / "cells.csv")
    curves = pd.read_csv(args.analysis / "curves.csv")
    over_clicks(curves, cells, "cost", args.out / "cost_over_clicks.png")
    over_clicks(curves, cells, "f1", args.out / "f1_over_clicks.png")
    per_cell(cells, args.out / "per_cell.png")
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
