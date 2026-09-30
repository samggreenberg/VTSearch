#!/usr/bin/env python3
"""Figures for #4383 from ``analyze_line_estimate_4383.py``'s ``summary.csv``.

    python figure_line_estimate_4383.py --summary OUT/summary.csv --out FIGDIR [--t 150] [--floor 0.5]

* ``f1_by_size.png`` - the returned set's F1 against corpus size, one line per
  rule, one panel per world, with the best cut's F1 dashed: does the rule keep
  up with the oracle as the corpus grows, where a fixed count cannot?
* ``recall_share_by_size.png`` - recall over the oracle's recall at P, the same
  layout.
* ``shortfall_vs_votes.png`` - at the whole test half: mean shortfall below P
  against audit votes spent, one point per rule, per world. Down and left is
  better.

Colours are the dataviz reference palette's categorical slots (validated for
CVD and normal vision); the baselines are black.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
SIZE_ORDER = ["32", "320", "3200", "full", "x10"]
SIZE_LABEL = {
    "32": "32",
    "320": "320",
    "3,200": "3,200",
    "3200": "3,200",
    "full": "test half",
    "x10": "x10 (bootstrap)",
}
COLORS = {
    "fixed": INK,
    "check": "#6b6b6b",
    "grow": "#2a78d6",
    "grow-lb": "#7fb0e6",
    "bands-iso": "#eb6834",
    "bands-iso-lb": "#f3a98a",
    "gmm": "#2a9d5c",
    "gmm+shift5": "#7fc9a0",
    "gmm+shift": "#0f6b3a",
    "post": "#8e5bd1",
    "post+shift5": "#c2a5e8",
    "post+shift": "#5a2f99",
    "grow-fine": "#1b4f8f",
    "gmm-grow": "#d62b6b",
}
#: The rules the report's figures show; the rest are in ``summary.csv``.
MAIN_RULES = ("fixed", "check", "grow", "grow-fine", "gmm", "gmm+shift5", "gmm-grow", "bands-iso-lb")
STYLE = {"fixed": "-", "check": "--"}


def _axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def by_size(s: pd.DataFrame, metric: str, out: Path, title: str, oracle_col: str | None, rules=()) -> None:
    worlds = [w for w in ("0.1%", "0.44%", "5%") if w in set(s["world"])]
    fig, axes = plt.subplots(1, len(worlds), figsize=(4.6 * len(worlds), 4.0), sharey=True, facecolor=SURFACE)
    axes = list(axes) if len(worlds) > 1 else [axes]
    for ax, w in zip(axes, worlds, strict=True):
        _axes(ax)
        sub = s[s["world"] == w]
        x = range(len(SIZE_ORDER))
        for rule, color in COLORS.items():
            if rules and rule not in rules:
                continue
            r = sub[sub["rule"] == rule].set_index("size").reindex(SIZE_ORDER)
            if r[metric].isna().all():
                continue
            ax.plot(x, r[metric].to_numpy(), color=color, linewidth=2.2 if rule in ("fixed", "gmm-grow") else 1.3,
                    linestyle=STYLE.get(rule, "-"), marker="o", markersize=3.5, label=rule)  # fmt: skip
        if oracle_col:
            r = sub[sub["rule"] == "fixed"].set_index("size").reindex(SIZE_ORDER)
            ax.plot(x, r[oracle_col].to_numpy(), color=INK, linewidth=1.2, linestyle=":", label="oracle")
        ax.set_xticks(list(x))
        ax.set_xticklabels([SIZE_LABEL[z] for z in SIZE_ORDER], fontsize=8)
        ax.set_xlabel("corpus size", color=INK)
        ax.set_title(f"{w} positives", color=INK, fontsize=10, loc="left")
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel(title, color=INK)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def shortfall_vs_votes(s: pd.DataFrame, out: Path, rules=()) -> None:
    worlds = [w for w in ("0.1%", "0.44%", "5%") if w in set(s["world"])]
    fig, axes = plt.subplots(1, len(worlds), figsize=(4.6 * len(worlds), 4.0), sharey=True, facecolor=SURFACE)
    axes = list(axes) if len(worlds) > 1 else [axes]
    for ax, w in zip(axes, worlds, strict=True):
        _axes(ax)
        sub = s[(s["world"] == w) & (s["size"] == "full")]
        for rule, color in COLORS.items():
            if rules and rule not in rules:
                continue
            r = sub[sub["rule"] == rule]
            if r.empty:
                continue
            ax.scatter(r["votes"], r["shortfall"], color=color, s=36, zorder=3)
            ax.annotate(rule, (float(r["votes"].iloc[0]), float(r["shortfall"].iloc[0])), xytext=(4, 3),
                        textcoords="offset points", fontsize=7, color=INK)  # fmt: skip
        ax.set_xlabel("audit votes spent", color=INK)
        ax.set_title(f"{w} positives, the test half", color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("mean shortfall below P (lower is better)", color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--t", type=int, default=150)
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--all-rules", action="store_true", help="every rule, not just MAIN_RULES")
    args = ap.parse_args()
    rules = () if args.all_rules else MAIN_RULES
    args.out.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(args.summary, dtype={"size": str})
    s = s[(s["t"] == args.t) & (s["floor"] == args.floor)]
    tag = f"p{round(args.floor * 100):d}"
    by_size(
        s,
        "f1",
        args.out / f"f1_by_size_{tag}.png",
        f"F1 of the returned set at P = {args.floor:.0%}",
        "oracle_f1",
        rules,
    )
    by_size(
        s,
        "recall_share",
        args.out / f"recall_share_by_size_{tag}.png",
        f"recall ÷ oracle at P = {args.floor:.0%}",
        None,
        rules,
    )
    by_size(
        s,
        "meets",
        args.out / f"meets_by_size_{tag}.png",
        f"share of sessions at or above P = {args.floor:.0%}",
        None,
        rules,
    )
    shortfall_vs_votes(s, args.out / f"shortfall_vs_votes_{tag}.png", rules)
    print(f"figures -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
