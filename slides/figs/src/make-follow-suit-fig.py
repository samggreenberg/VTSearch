#!/usr/bin/env python
"""Slide 32, "Follow Suit": the line follows the user's preference (#4548, #4519).

Run from the repo root, once the study's curve CSV is committed:

    python slides/figs/src/make-follow-suit-fig.py

The owner wanted slide 32 to show what the work since cross-calibration bought.
Measured on the 1% pool (#4548), it bought a line that follows the user's preset:

- at beta 1/4 today's app beats cross-calibration by +0.17 F-beta at vote 150;
- at beta 1 by +0.03;
- at beta 4 not at all, because cross-calibration's one long, recall-heavy line already suits that user.

One row per preset, as Beta Max stacks them (beta rising down the slide). Each row is that preset's F-beta over
votes, filled before a detector shows, for two lines:

- cross-calibration (grey): the same sessions whatever the user wants, scored at the row's beta;
- today's app at that preset (the deck's blue), its own sessions.

Both start at the typed query, the notch at click 0. Page a draws cross-calibration alone ("one line, whatever
you want"); page b adds today's app. A reveal adds ink and restyles nothing, so every label holds its final
place from the page it appears on.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save, spread_labels  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
#: The 1% study (#4519, #4548): cross-calibration and today's app at each preset, analyzed together.
CSV = REPO / "docs" / "experiments" / "2026-10-05-ladder-fbeta-4519" / "presets" / "progression_curve.csv"
#: One row per preset, top to bottom as Beta Max stacks them: (beta label, the F column scored, today's rung).
ROWS: tuple[tuple[str, str, str], ...] = (
    ("¼", "f025_mean", "r8_labels_b025"),
    ("1", "f1_mean", "r8_labels"),
    ("4", "f4_mean", "r8_labels_b4"),
)
XCAL = "r1_xcal"
#: The deck's `.cut` blue (`themes/vtsearch.css`): today's app, the shipped thing.
SHIPPED = "#2F6DB5"
GREY = "#8a929e"
#: The rows' shared F range; the highest point drawn is ~0.62.
YLIM = (0.0, 0.8)

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 17,
        "text.color": INK,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "axes.labelsize": 17,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def read_curves(path: Path) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """``(rung, column) -> (votes, mean)`` for every rung and F column the slide draws."""
    if not path.exists():
        raise SystemExit(f"{path.relative_to(REPO)} does not exist yet: run the #4548 analysis and commit it")
    by: dict[tuple[str, str], list[tuple[int, float]]] = {}
    cols = {c for _, c, _ in ROWS}
    with path.open() as f:
        for row in csv.DictReader(f):
            for col in cols:
                by.setdefault((row["rung"], col), []).append((int(row["t"]), float(row[col])))
    out = {}
    for key, pts in by.items():
        pts.sort()
        out[key] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    missing = [f"{r} {c}" for _, c, r in ROWS if (r, c) not in out]
    missing += [f"{XCAL} {c}" for _, c, _ in ROWS if (XCAL, c) not in out]
    if missing:
        raise SystemExit(f"{path}: no curve for {', '.join(missing)}")
    return out


def _figure(curves: dict, today: bool) -> Figure:
    """The slide; *today* adds today's app to every row (page b)."""
    fig = plt.figure(figsize=(12.8, 7.2))
    # Under the title notch, with a right column for the line names.
    top, bottom, left, right, gap = 0.73, 0.13, 0.17, 0.74, 0.035
    h = (top - bottom - 2 * gap) / 3
    t_max = int(max(curves[(XCAL, ROWS[0][1])][0].max(), 150))
    for i, (beta, col, rung) in enumerate(ROWS):
        ax = fig.add_axes([left, top - (i + 1) * h - i * gap, right - left, h])
        ax.set_xlim(0, t_max)
        ax.set_ylim(*YLIM)
        ax.set_yticks([0, 0.4, 0.8])
        ax.yaxis.grid(True, color="#e3e7ec", lw=1.0)
        ax.set_axisbelow(True)
        ticks = [0, 25, 50, 100, 150]
        ax.set_xticks(ticks)
        if i == len(ROWS) - 1:
            ax.set_xticklabels(["typed\nquery", "25", "50", "100", "150"])
            ax.set_xlabel("Votes")
        else:
            ax.set_xticklabels([])
        # The preset this row is, where Beta Max names its rows: left of the row.
        ax.text(-0.09, 0.5, f"β = {beta}", transform=ax.transAxes, ha="right", va="center", fontsize=18)
        xt, xy = curves[(XCAL, col)]
        tt, ty = curves[(rung, col)]
        ends = [float(xy[-1]), float(ty[-1])]
        # Both names at their ends, fanned only as far as the type needs where the
        # two ends meet (beta 4: the lines coincide). Computed from both, so page a
        # puts cross-calibration's name where page b keeps it.
        span = YLIM[1] - YLIM[0]
        lab = spread_labels(ends, gap=0.2 * span)
        ax.plot(xt, xy, color=GREY, lw=2.6, solid_capstyle="round", zorder=3)
        ax.text(t_max * 1.02, lab[0], "cross-calibration", color=INK, ha="left", va="center", clip_on=False)
        if today:
            ax.plot(tt, ty, color=SHIPPED, lw=3.6, solid_capstyle="round", zorder=4)
            ax.text(t_max * 1.02, lab[1], "today's app", color=SHIPPED, fontweight="bold", ha="left",
                    va="center", clip_on=False)  # fmt: skip
        # The notch: the typed query, where both lines start.
        ax.plot([0], [float(xy[0])], marker="o", color=INK, markersize=8, zorder=5, clip_on=False)
    fig.text(0.02, (top + bottom) / 2, r"$\mathregular{F_\beta}$ at the row's β", rotation=90, ha="center",
             va="center", fontsize=17)  # fmt: skip
    return fig


def main() -> int:
    curves = read_curves(CSV)
    # Saved at the canvas's declared bounds: the top margin keeps the title notch
    # empty, and both pages frame the rows identically.
    save(_figure(curves, today=True), OUT, "follow-suit.png", column=FULL_BLEED, tight=False)
    save(_figure(curves, today=False), OUT, "follow-suit.build1.png", column=FULL_BLEED, tight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
