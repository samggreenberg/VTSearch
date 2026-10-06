#!/usr/bin/env python3
"""Figures for the Test-arm pricing (#4523), from ``analyze_line_test_4523.py``'s tables.

    python figure_line_test_4523.py --analysis DIR --out FIGDIR

* ``coverage_vs_picks.png`` - per world, the precision range's coverage against
  the picks a session spends, one point per grid point (width x budget), the
  nominal 95% level drawn: the figure the budget decision reads.
* ``picks_by_world.png`` - the picks to Done at the default budgets, per world
  and beta, as the distribution across sessions (what a user pays).
* ``recall_estimate.png`` - the recall estimate against the truth at the default
  budgets, per world: what the model-assisted tail is worth where positives are
  scarce.
* ``verdicts.png`` - the verdict's reading against the oracle's at the default
  budgets, per world and beta.

Colour is the world (fixed order), hue never cycled; everything else is a panel.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DPI = 130
#: Categorical slots in fixed order (the dataviz reference palette): the worlds take the first three.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#dcdbd6"
DEFAULT_WIDTH, DEFAULT_BUDGET = 0.20, 40


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _worlds(df: pd.DataFrame) -> list[str]:
    order = {"0.1%": 0, "0.44%": 1, "5%": 2}
    return sorted(df["world"].unique(), key=lambda w: (order.get(str(w).split("@")[0], 9), str(w)))


def coverage_vs_picks(summary: pd.DataFrame, out: Path) -> None:
    betas = sorted(summary["beta"].unique())
    worlds = _worlds(summary)
    fig, axes = plt.subplots(1, len(betas), figsize=(4.2 * len(betas), 3.6), sharey=True, squeeze=False)
    for ax, beta in zip(axes[0], betas):
        _style(ax)
        for i, w in enumerate(worlds):
            g = summary[(summary["world"] == w) & (summary["beta"] == beta)]
            ax.scatter(
                g["picks_mean"],
                g["precision_cov"],
                s=28,
                color=SERIES[i],
                label=w,
                zorder=3,
                edgecolor="white",
                linewidth=0.8,
            )
            d = g[(g["width"] == DEFAULT_WIDTH) & (g["budget"] == DEFAULT_BUDGET)]
            if len(d):
                ax.scatter(
                    d["picks_mean"],
                    d["precision_cov"],
                    s=90,
                    facecolor="none",
                    edgecolor=SERIES[i],
                    linewidth=1.5,
                    zorder=4,
                )
            for _, r in g.iterrows():
                ax.annotate(
                    f"{r['width']:g}/{int(r['budget'])}",
                    (r["picks_mean"], r["precision_cov"]),
                    fontsize=6,
                    color=MUTED,
                    xytext=(3, 2),
                    textcoords="offset points",
                )
        ax.axhline(0.95, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.set_title(f"beta {beta:g}", fontsize=10, color=INK)
        ax.set_xlabel("picks to Done (mean)", fontsize=9, color=MUTED)
    axes[0][0].set_ylabel("precision range held the truth (share)", fontsize=9, color=MUTED)
    axes[0][0].legend(frameon=False, fontsize=8, title="withheld half", title_fontsize=8)
    fig.suptitle(
        "Coverage of the precision range against its cost, per target width / budget (ring = the plan's 0.20/40)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out / "coverage_vs_picks.png", dpi=DPI)
    plt.close(fig)


def picks_by_world(rows: pd.DataFrame, out: Path) -> None:
    d = rows[(rows["width"] == DEFAULT_WIDTH) & (rows["budget"] == DEFAULT_BUDGET) & (rows["phase"] == "done")]
    betas = sorted(d["beta"].unique())
    worlds = _worlds(d)
    fig, axes = plt.subplots(1, len(betas), figsize=(4.2 * len(betas), 3.4), sharey=True, squeeze=False)
    for ax, beta in zip(axes[0], betas):
        _style(ax)
        data = [d[(d["world"] == w) & (d["beta"] == beta)]["picks_total"].to_numpy() for w in worlds]
        parts = ax.boxplot(
            data, widths=0.5, showfliers=False, patch_artist=True, medianprops={"color": INK, "linewidth": 1.2}
        )
        for patch, i in zip(parts["boxes"], range(len(worlds))):
            patch.set_facecolor(SERIES[i])
            patch.set_alpha(0.55)
            patch.set_edgecolor(SERIES[i])
        for key in ("whiskers", "caps"):
            for line in parts[key]:
                line.set_color(MUTED)
        ax.set_xticks(range(1, len(worlds) + 1), worlds, fontsize=8)
        ax.set_title(f"beta {beta:g}", fontsize=10, color=INK)
    axes[0][0].set_ylabel("picks to Done (0.20 / 40)", fontsize=9, color=MUTED)
    fig.suptitle(
        "What a Test costs at the plan's budgets: picks to Done across sessions (box = quartiles)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out / "picks_by_world.png", dpi=DPI)
    plt.close(fig)


def recall_estimate(rows: pd.DataFrame, out: Path) -> None:
    d = rows[(rows["width"] == DEFAULT_WIDTH) & (rows["budget"] == DEFAULT_BUDGET) & (rows["phase"] == "done")]
    worlds = _worlds(d)
    fig, axes = plt.subplots(1, len(worlds), figsize=(3.6 * len(worlds), 3.6), sharey=True, squeeze=False)
    for ax, w, i in zip(axes[0], worlds, range(len(worlds))):
        _style(ax)
        g = d[d["world"] == w]
        ax.plot([0, 1], [0, 1], color=GRID, linewidth=1)
        held = g["recall_held"] == 1
        ax.scatter(
            g.loc[held, "recall_true"],
            g.loc[held, "recall_point"],
            s=10,
            color=SERIES[i],
            alpha=0.5,
            label="range held",
            edgecolor="none",
        )
        ax.scatter(
            g.loc[~held, "recall_true"],
            g.loc[~held, "recall_point"],
            s=14,
            facecolor="none",
            edgecolor=INK,
            linewidth=0.6,
            label="range missed",
        )
        cov = float(g["recall_held"].mean()) if len(g) else float("nan")
        ax.set_title(f"{w}: range held {cov:.0%}", fontsize=10, color=INK)
        ax.set_xlabel("true recall", fontsize=9, color=MUTED)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    axes[0][0].set_ylabel("estimated recall (point)", fontsize=9, color=MUTED)
    axes[0][0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(
        "The recall estimate against the truth at the plan's budgets (one point per session x Test seed)",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out / "recall_estimate.png", dpi=DPI)
    plt.close(fig)


def verdicts(rows: pd.DataFrame, out: Path) -> None:
    d = rows[(rows["width"] == DEFAULT_WIDTH) & (rows["budget"] == DEFAULT_BUDGET) & (rows["phase"] == "done")]
    worlds = _worlds(d)
    betas = sorted(d["beta"].unique())
    kinds = ("ship", "lean", "retrain")
    fig, axes = plt.subplots(1, len(betas), figsize=(4.2 * len(betas), 3.4), sharey=True, squeeze=False)
    for ax, beta in zip(axes[0], betas):
        _style(ax)
        x = np.arange(len(worlds))
        for j, (col, label, alpha) in enumerate(
            (("verdict", "the Test reads", 0.9), ("oracle_verdict", "the truth says", 0.45))
        ):
            bottom = np.zeros(len(worlds))
            for k, kind in enumerate(kinds):
                share = np.array([(d[(d["world"] == w) & (d["beta"] == beta)][col] == kind).mean() for w in worlds])
                ax.bar(
                    x + (j - 0.5) * 0.36,
                    share,
                    0.34,
                    bottom=bottom,
                    color=SERIES[3 + k],
                    alpha=alpha,
                    edgecolor="white",
                    linewidth=1,
                    label=f"{kind} ({label})" if True else None,
                )
                bottom += share
        ax.set_xticks(x, worlds, fontsize=8)
        ax.set_title(f"beta {beta:g}", fontsize=10, color=INK)
    axes[0][0].set_ylabel("share of sessions", fontsize=9, color=MUTED)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "The verdict's reading (left bar) against the truth under the same rule (right bar), at 0.20 / 40",
        fontsize=10,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "verdicts.png", dpi=DPI)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--analysis", type=Path, required=True, help="analyze_line_test_4523.py's --out")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.analysis / "summary.csv")
    rows = pd.read_csv(args.analysis / "rows.csv.gz")
    # The width x budget grid at the plan's walk below the line; the walk variants are the report's table.
    if "budget_below" in summary.columns:
        summary = summary[summary["budget_below"] == summary["budget"]]
        rows = rows[rows["budget_below"] == rows["budget"]]
    if "walk" in summary.columns:
        default_walk = summary.loc[summary["width"].eq(DEFAULT_WIDTH) & summary["budget"].eq(DEFAULT_BUDGET), "walk"]
        plan = sorted(default_walk.unique(), key=lambda w: (not str(w).startswith("d0.05/w5/"), str(w)))[0]
        summary = summary[summary["walk"] == plan]
        rows = rows[rows["walk"] == plan]
    coverage_vs_picks(summary, args.out)
    picks_by_world(rows, args.out)
    recall_estimate(rows, args.out)
    verdicts(rows, args.out)
    print(f"wrote 4 figures to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
