#!/usr/bin/env python
"""#4759's figures: each centroid count against the shipped one, at every vote, per bench and balance.

Reads ``curves.csv`` from ``analyze_centroid_count_4759.py`` and writes ``fig_centroid_count.png``: one
row per balance (beta 1/4, 1, 4), one column per panel named with ``--panel run/dataset=Title``. Each
line is a candidate's mean F-beta minus ``count``'s at that vote, over every run (zero wherever Find
does not give the centroid). Every curve returns to zero when the tier ends, so end labels would stack;
one legend names the lines. ``gap_floor`` and ``count_gap`` are left out: they are ``count_gap_floor``
below and above beta 1.

Usage::

    python figures_centroid_count_4759.py --curves <analysis>/curves.csv --out <dir> \\
        --panel "fhibe-k1/fhibe_faces_1024=FHIBE faces, 1 example" --panel "coco-b1/coco_better=COCO Binary"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
#: Candidate -> (colour, label, width). The gap family in the warm slots, the bar family in the cool ones.
RULES = {
    "count_gap_floor": ("#eb6834", "larger count, floored at the gap (shipped)", 2.4),
    "gap": ("#f2a541", "gap alone", 1.3),
    "count_z3.5": ("#2a78d6", "bar at z 3.5", 1.3),
    "count_z3": ("#1d4f91", "bar at z 3", 1.3),
    "mirror_z3.5": ("#6aa6e8", "mirrored bulk, z 3.5", 1.3),
}
BETAS = {0.25: "β ¼", 1.0: "β 1", 4.0: "β 4"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK2,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )


#: Colours for ``--rule`` lines, in order: the first is drawn thick (the candidate a study ships).
PALETTE = ("#eb6834", "#f2a541", "#2a78d6", "#1d4f91", "#6aa6e8", "#8a8984")


def curves_figure(
    curves: pd.DataFrame,
    panels: list[tuple[str, str, str]],
    out: Path,
    max_t: int,
    rules: dict[str, tuple[str, str, float]] | None = None,
    reference: str = "count",
    name: str = "fig_centroid_count.png",
) -> Path:
    rules = RULES if rules is None else rules
    fig, axes = plt.subplots(
        len(BETAS), len(panels), figsize=(4.4 * len(panels), 2.6 * len(BETAS)), squeeze=False, sharex=True
    )
    d = curves[(curves["metric"] == "d_ref") & (curves["t"] <= max_t)]
    for j, (run, ds, title) in enumerate(panels):
        for i, (beta, bname) in enumerate(BETAS.items()):
            ax = axes[i][j]
            sub = d[(d["run"] == run) & (d["dataset"] == ds) & (d["beta"] == beta)]
            for rule, (color, _name, width) in rules.items():
                c = sub[sub["rule"] == rule].sort_values("t")
                if not c.empty:
                    ax.plot(c["t"], c["value"], color=color, lw=width)
            ax.axhline(0, color=INK2, lw=0.8)
            ax.grid(axis="y", color=GRID, lw=0.8)
            if i == 0:
                ax.set_title(title, fontsize=10, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f"{bname}: minus {reference}", fontsize=8 if len(reference) > 8 else 10)
            if i == len(BETAS) - 1:
                ax.set_xlabel("vote")
            ax.set_xlim(1, max_t)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    handles = [plt.Line2D([], [], color=c, lw=w, label=n) for c, n, w in rules.values()]
    fig.legend(handles=handles, loc="upper center", ncol=len(rules), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--curves", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--panel", action="append", required=True, help="run/dataset=Title")
    ap.add_argument("--max-t", type=int, default=20, help="the last vote drawn (the tier ends by ~17)")
    ap.add_argument(
        "--rule", action="append", default=[], help="rule=Label, in order (the first thick); default #4759's set"
    )
    ap.add_argument("--reference", default="count", help="the rule the curves are differences to (the y label)")
    ap.add_argument("--name", default="fig_centroid_count.png")
    args = ap.parse_args(argv)
    panels = []
    for spec in args.panel:
        key, _, title = spec.partition("=")
        run, _, ds = key.partition("/")
        panels.append((run, ds, title or key))
    _style()
    rules = None
    if args.rule:
        rules = {}
        for i, spec in enumerate(args.rule):
            rule, _, label = spec.partition("=")
            rules[rule] = (PALETTE[i % len(PALETTE)], label or rule, 2.4 if i == 0 else 1.3)
    print(curves_figure(pd.read_csv(args.curves), panels, args.out, args.max_t, rules, args.reference, args.name))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
