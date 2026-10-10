#!/usr/bin/env python
"""#4732's figure: what a Find returns at every vote under each centroid line, per bench.

Reads ``curves.csv`` from ``analyze_centroid_line_4732.py`` and writes ``fig_centroid_line.png``. The
top row is F-beta at the run's balance (beta 1) over votes 1..H, one line per rule; the bottom row is
each rule's difference to the midpoint at beta 1/4, 1 and 4, which is zero off the centroid's
clicks. One column per panel named with ``--panel run/dataset=Title``. Lines are labelled at their
right ends, not through leader lines (slides/STYLE.md).

Usage::

    python figures_centroid_line_4732.py --curves <analysis>/curves.csv --out <dir> \\
        --panel "fhibe-k1/fhibe_faces_1024=FHIBE faces, 1 example" --panel "coco-b1/coco_better=COCO Binary"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
#: Rule -> colour (categorical slots 1 and 2 for the candidates, ink for today's line).
RULES = {"midpoint": ("#8a8984", "midpoint (today)"), "guarded": ("#2a78d6", "guarded"), "count": ("#eb6834", "count")}
#: Beta -> line style in the difference row.
BETAS = {0.25: (":", "β¼"), 1.0: ("-", "β1"), 4.0: ("--", "β4")}


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
    ap.add_argument("--panel", action="append", required=True, help="run/dataset=Title")
    ap.add_argument("--run-beta", type=float, default=1.0)
    args = ap.parse_args(argv)
    curves = pd.read_csv(args.curves)
    panels = []
    for spec in args.panel:
        key, _, title = spec.partition("=")
        run, _, ds = key.partition("/")
        panels.append((run, ds, title or key))

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
    fig, axes = plt.subplots(2, len(panels), figsize=(4.6 * len(panels), 6.4), squeeze=False, sharex=True)
    for j, (run, ds, title) in enumerate(panels):
        sub = curves[(curves["run"] == run) & (curves["dataset"] == ds)]
        top, bottom = axes[0][j], axes[1][j]
        ends, names, colors = [], [], []
        for rule, (color, name) in RULES.items():
            c = sub[(sub["rule"] == rule) & (sub["beta"] == args.run_beta) & (sub["metric"] == "fbeta")]
            if c.empty:
                continue
            top.plot(c["t"], c["value"], color=color, lw=1.8)
            ends.append(float(c["value"].iloc[-1]))
            names.append(name)
            colors.append(color)
        x_end = float(sub["t"].max()) if len(sub) else 1.0
        for y, name, color in zip(_spread(ends, 0.04), names, colors):
            top.text(x_end + 0.6, y, name, color=color, va="center", fontsize=9)
        top.set_title(title, fontsize=10, color=INK, loc="left")
        top.set_ylim(0, 1)
        top.grid(axis="y", color=GRID, lw=0.8)
        if j == 0:
            top.set_ylabel(f"F-beta at beta {args.run_beta:g}, what a Find returns")
        for rule, (color, _name) in RULES.items():
            if rule == "midpoint":
                continue
            for beta, (style, _bname) in BETAS.items():
                d = sub[(sub["rule"] == rule) & (sub["beta"] == beta) & (sub["metric"] == "d_fbeta")]
                if not d.empty:
                    bottom.plot(d["t"], d["value"], color=color, ls=style, lw=1.5)
        bottom.axhline(0, color=INK2, lw=0.8)
        bottom.grid(axis="y", color=GRID, lw=0.8)
        bottom.set_xlabel("vote")
        if j == 0:
            bottom.set_ylabel("F-beta minus the midpoint's")
    handles = [plt.Line2D([], [], color=INK2, ls=s, label=f"difference at {n}") for s, n in BETAS.values()]
    axes[1][-1].legend(handles=handles, frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / "fig_centroid_line.png", dpi=150, facecolor=SURFACE)
    print(args.out / "fig_centroid_line.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
