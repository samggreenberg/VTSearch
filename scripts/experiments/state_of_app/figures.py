#!/usr/bin/env python3
"""State of the App (#4159): the report's figures, from ``analyze.py``'s tables.

    python figures.py --analysis <exp>/analysis --out <report dir>/figures

* ``ap_over_clicks.png`` / ``goods_over_clicks.png`` -- the mean curve per path,
  click 0 (text only) to the last click, with each path's full-label ceiling AP
  dashed. Two figures, not one with two y-axes: AP and a count of Goods are
  different measures.
* ``line_at_floors.png`` -- the line on a fresh corpus, one panel per floor the
  app offers: how right the set the line keeps is, from click 0 to the final
  line, against the floor (dotted) and the full-label ceiling's line (dashed).
  Each panel's title carries the share of sessions whose final line meets X.
* ``compare_ap.png`` (with ``--compare <other analysis dir>``) -- both paths'
  mean AP on the SAME seeds, the ones this analysis has.
* ``per_cell.png`` -- every class x band: text only (hollow), after the clicks
  (filled), full labels (tick), one panel per path.

No FPR + FNR anywhere (owner, 2026-09-30, #4357). Colours are the dataviz
reference palette's first two categorical slots, which validate for both CVD
and normal vision (validate_palette.js, 2026-09-23).
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
#: The line's points left to right, as ``analyze.py`` names them; the ceiling is drawn apart.
LINE_POINTS = ("text", "10", "25", "50", "100", "150", "final")


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
        fmt = f"{mean[arm].iloc[-1]:.2f}" if metric == "ap" else f"{mean[arm].iloc[-1]:.1f}"
        ax.annotate(
            f"{arm}  {fmt}",
            (mean.index[-1], mean[arm].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            color=INK,
            fontsize=9,
        )
        if metric == "ap":
            ceil = cells[cells["arm"] == arm]["ceiling_ap"].mean()
            if pd.notna(ceil):
                ax.axhline(ceil, color=color, linewidth=1.2, linestyle="--")
                ax.annotate(
                    f"full labels {ceil:.2f}",
                    (2, ceil),
                    xytext=(0, 4),
                    textcoords="offset points",
                    color=INK,
                    fontsize=8,
                )
    if metric == "ap":
        t0 = mean.iloc[0].mean()
        ax.plot([0], [t0], marker="o", markersize=8, color=INK, zorder=5)
        ax.annotate(f"text only {t0:.2f}", (0, t0), xytext=(8, 6), textcoords="offset points", color=INK, fontsize=8)
        ax.set_ylim(0, 1.02)
    ax.set_xlim(0, mean.index.max() * 1.17)
    ax.set_xlabel("clicks", color=INK)
    label = {
        "ap": "average precision on the test half (higher is better)",
        "goods": "Goods found (higher is better)",
    }[metric]
    ax.set_ylabel(label, color=INK)
    n = cells.groupby("arm").size().to_dict()
    name = {"ap": "AP", "goods": "Goods found"}[metric]
    ax.set_title(
        f"Mean {name} over clicks  (runs: " + ", ".join(f"{k} {v}" for k, v in n.items()) + ")",
        color=INK,
        fontsize=10,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def line_at_floors(lines: pd.DataFrame, out: Path) -> bool:
    """One panel per floor: the kept set's precision from click 0 to the final line.

    Returns False (and draws nothing) when the run recorded no rank frames.
    """
    known = lines[(lines["point"] != "text") & lines["precision"].notna()]
    if known.empty:
        return False
    floors = sorted(lines["floor"].unique())
    fig, axes = plt.subplots(1, len(floors), figsize=(4.0 * len(floors), 3.8), sharey=True, facecolor=SURFACE)
    axes = list(axes) if len(floors) > 1 else [axes]
    for ax, x in zip(axes, floors, strict=True):
        _axes(ax)
        sub = lines[lines["floor"] == x]
        met = []
        for arm, color in COLORS.items():
            a = sub[sub["arm"] == arm]
            if a.empty:
                continue
            m = a.groupby("point")["precision"].mean().reindex(LINE_POINTS)
            ax.plot(range(len(LINE_POINTS)), m.to_numpy(), color=color, linewidth=2, marker="o", markersize=5)
            ceil = a[a["point"] == "ceiling"]["precision"].mean()
            if pd.notna(ceil):
                ax.axhline(ceil, color=color, linewidth=1.2, linestyle="--")
            final = a[a["point"] == "final"]["meets"].mean()
            if pd.notna(final):
                met.append(f"{arm} {final:.0%}")
        ax.axhline(x, color=INK, linewidth=1, linestyle=":")
        ax.annotate(f"floor {x:.0%}", (0, x), xytext=(2, 4), textcoords="offset points", color=INK, fontsize=8)
        ax.set_xticks(range(len(LINE_POINTS)))
        ax.set_xticklabels(["text", *LINE_POINTS[1:-1], "final"], fontsize=8)
        ax.set_xlabel("clicks", color=INK)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"X = {x:.0%}: meets X at the end: " + ", ".join(met), color=INK, fontsize=9, loc="left")
    axes[0].set_ylabel("share of the kept set that is right", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return True


def per_cell(cells: pd.DataFrame, out: Path) -> None:
    cells = cells.copy()
    order = cells.groupby("category")["final_ap"].mean().sort_values(ascending=False).index.tolist()
    arms = [(a, c) for a, c in COLORS.items() if a in set(cells["arm"])]
    fig, axes = plt.subplots(
        1, len(arms), figsize=(5.2 * len(arms), 0.2 * len(order) + 1.4), sharey=True, facecolor=SURFACE, squeeze=False
    )
    axes = axes[0]
    for ax, (arm, color) in zip(axes, arms, strict=True):
        _axes(ax)
        sub = cells[cells["arm"] == arm].assign(
            never_trained=lambda d: d["never_trained"].astype("boolean").fillna(False).astype(bool)
        )
        # Mean over seeds; a cell "never found a positive" only if it failed in every seed.
        a = (
            sub.groupby("category")
            .agg(
                text_ap=("text_ap", "mean"),
                final_ap=("final_ap", "mean"),
                ceiling_ap=("ceiling_ap", "mean"),
                never_trained=("never_trained", "all"),
            )
            .reindex(order)
        )
        y = range(len(order))
        for yi, (fa, ca) in zip(y, a[["final_ap", "ceiling_ap"]].itertuples(index=False), strict=True):
            ax.plot([fa, ca], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.scatter(
            a["text_ap"], y, s=36, facecolors="none", edgecolors=MUTED, linewidths=1.2, zorder=2, label="text only"
        )
        ax.scatter(a["final_ap"], y, s=40, color=color, zorder=3, label="after the clicks")
        ax.scatter(a["ceiling_ap"], y, s=60, marker="|", color=INK, zorder=3, label="full labels")
        never = a[a["never_trained"].astype("boolean").fillna(False).astype(bool)]
        for cat in never.index:
            ax.annotate(
                "never found a positive",
                (never.loc[cat, "text_ap"], order.index(cat)),
                xytext=(10, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=7,
                color=INK,
            )
        ax.set_title(arm, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("average precision (higher is better)", color=INK)
        ax.set_xlim(0, 1.02)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels(order, fontsize=6 if len(order) > 60 else 8)
    axes[0].invert_yaxis()
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None, help="the other path's analysis dir, for compare_ap.png")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(args.analysis / "cells.csv")
    curves = pd.read_csv(args.analysis / "curves.csv")
    lines = pd.read_csv(args.analysis / "lines.csv", dtype={"point": str})
    over_clicks(curves, cells, "ap", args.out / "ap_over_clicks.png")
    over_clicks(curves, cells, "goods", args.out / "goods_over_clicks.png")
    if not line_at_floors(lines, args.out / "line_at_floors.png"):
        print("no rank frames in this run: line_at_floors.png skipped")
    per_cell(cells, args.out / "per_cell.png")
    if args.compare is not None:
        seeds = set(cells["seed"])
        other_cells = pd.read_csv(args.compare / "cells.csv")
        other_curves = pd.read_csv(args.compare / "curves.csv")
        both_cells = pd.concat([cells, other_cells[other_cells["seed"].isin(seeds)]], ignore_index=True)
        both_curves = pd.concat([curves, other_curves[other_curves["seed"].isin(seeds)]], ignore_index=True)
        over_clicks(both_curves, both_cells, "ap", args.out / "compare_ap.png")
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
