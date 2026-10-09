#!/usr/bin/env python
"""Experimentation's "Up and Up": what the work since the first rule bought, in the score the user sets (#4668).

Run from the repo root, once the study's curve CSV is committed:

    python slides/figs/src/make-up-and-up-fig.py [CSV]

The owner wanted a slide whose takeaway is "the score they care about improved over and over" (2026-10-08). #4668
ran the F-beta era as a cumulative build-up, each change added to the one before. Two of the changes carry it, and
the owner picked the slide that draws those two at beta 1/4:

- the labels line at the preset (#4452), which lifts the second half of the session;
- the typed query's line at the preset (#4603), which lifts the opening, where no detector is on screen yet.

The relative floor (#4492), the weak check (#4496) and even-odds asking (#3546) sit between them in the build-up. Each
moves the curve by less than 0.02, so the slide does not draw them and the notes name them.

One panel, the objective (F-beta 1/4 of the withheld half above the line the app shows) over votes, filled from the
typed query's set until a detector shows. A three-page build, each page adding a line:

- a: cross-calibration (rung b1), grey;
- b: plus the labels line at the preset (b2), dark;
- c: today's app (b6), the deck's blue.

Lines a and b start at the typed query's old line (the mixture midpoint) and line c at today's, so each line carries
its own click-0 dot.
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
CSV = REPO / "docs" / "experiments" / "2026-10-08-fbeta-buildup-4668" / "buildup_curve.csv"
PRESET = "b025"
#: (rung, label, colour, line width, bold label), in build order.
LINES: tuple[tuple[str, str, str, float, bool], ...] = (
    ("b1", "cross-calibration", "#8a929e", 2.6, False),
    ("b2", "+ the labels line", "#3d4450", 2.8, False),
    ("b6", "today's app", "#2F6DB5", 3.6, True),
)
YLIM = (0.0, 0.7)
#: The canvas: the whole slide, at 100 slide pixels to the inch.
FIG_W, FIG_H = 12.8, 7.2

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


def read_curves(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """``rung -> (votes, mean)`` at the slide's preset."""
    if not path.exists():
        raise SystemExit(f"{path} does not exist yet: run analyze_buildup_4668.py and commit its CSV")
    by: dict[str, list[tuple[int, float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["preset"] == PRESET:
                by.setdefault(row["rung"], []).append((int(row["t"]), float(row["mean"])))
    missing = [r for r, *_ in LINES if r not in by]
    if missing:
        raise SystemExit(f"{path}: no {PRESET} curve for {', '.join(missing)}")
    out = {}
    for rung, pts in by.items():
        pts.sort()
        out[rung] = (np.array([p[0] for p in pts]), np.array([p[1] for p in pts]))
    return out


def _figure(curves: dict[str, tuple[np.ndarray, np.ndarray]], shown: int) -> Figure:
    """The slide with the first *shown* lines drawn (pages a, b, c)."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # Up to the title notch's foot (172 of 720 px), the full width but a column on the right for the line names.
    ax = fig.add_axes((0.105, 0.135, 0.665, 0.62))
    t_max = 150
    ax.set_xlim(0, t_max)
    ax.set_ylim(*YLIM)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.yaxis.grid(True, color="#e3e7ec", lw=1.0)
    ax.set_axisbelow(True)
    ax.set_xticks([0, 25, 50, 100, 150])
    ax.set_xticklabels(["typed\nquery", "25", "50", "100", "150"])
    ax.set_xlabel("Votes")
    ax.set_ylabel("Fβ at β = ¼")
    ends = [float(curves[r][1][-1]) for r, *_ in LINES]
    # Every name at its line's end, fanned only where ends meet; computed from all three so a name holds its
    # place from the page it appears on.
    lab = spread_labels(ends, gap=0.075 * (YLIM[1] - YLIM[0]))
    for i, (rung, label, colour, lw, bold) in enumerate(LINES[:shown]):
        t, y = curves[rung]
        ax.plot(t, y, color=colour, lw=lw, solid_capstyle="round", zorder=3 + i)
        ax.plot([0], [float(y[0])], marker="o", color=colour, markersize=9, zorder=6, clip_on=False)
        ax.text(t_max * 1.02, lab[i], label, color=colour if bold else INK, fontweight="bold" if bold else "normal",
                ha="left", va="center", clip_on=False)  # fmt: skip
    return fig


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    curves = read_curves(path)
    save(_figure(curves, 3), OUT, "up-and-up.png", column=FULL_BLEED, tight=False)
    for k in (1, 2):
        save(_figure(curves, k), OUT, f"up-and-up.build{k}.png", column=FULL_BLEED, tight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
