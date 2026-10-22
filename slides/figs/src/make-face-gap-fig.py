#!/usr/bin/env python
"""Mind the Gap: how the face path does for the people it does worst for, and what closes it.

    python slides/figs/src/make-face-gap-fig.py

Writes `figs/face-gap.webp` and one build stage, from the face-gap study's committed curves
(`docs/experiments/2026-10-12-face-gap-4780/curves_by_group.csv`): what the session shows at every
click, F-beta at β 1, from one starting photo at 1024 px, for FHIBE's she/her people (283) and he/him
people (278). The study ran today's app at one commit, so the left panel is the face path as it ships
(FaceNet on crops turned upright, the head started from the Goods' centroid, no prompted check). The
right panel is the same sessions with every face's vector swapped for ArcFace's, a face model the app
does not ship: its licence is non-commercial research. Click 0 is the starting photo's own sort, as on
Face Value.

Page a draws FaceNet alone; page b adds ArcFace beside it, on the same axes. The two groups are told
apart by weight and darkness, never hue: the deck's hues mean the line, the Goods and the Bads.

`EXPECT` pins the numbers the presenter notes say, as `make-sota-figs.py` does: a rerun of the study
is a rerun of this script, then a look at the notes.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save, spread_labels  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
CURVES = REPO / "docs" / "experiments" / "2026-10-12-face-gap-4780" / "curves_by_group.csv"

#: The panels: the set's name in the CSV and its title on the slide.
PANELS = (("fn", "FaceNet: the app"), ("arc", "ArcFace"))
#: The groups: the CSV's pronoun, the slide's label, and how heavy and dark its line is drawn.
GROUPS = (
    ("She/her/hers", "she/her", INK, 3.4),
    ("He/him/his", "he/him", "#8a929e", 3.4),
)
#: From one starting photo, as every face slide reads it.
K = "1"
#: The study's 150 votes, and click 0 for the starting photo's own sort.
CLICKS = 150

#: What the presenter notes say, rounded as they say it: each group's level at the last click.
EXPECT = {
    ("fn", "She/her/hers"): 0.76,
    ("fn", "He/him/his"): 0.88,
    ("arc", "She/her/hers"): 0.93,
    ("arc", "He/him/his"): 0.95,
}

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
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

#: The canvas, the slide's 16:9, and the two panels below the title notch: Running Start's band.
FIG_SIZE = (12.8, 7.2)
TOP, BOTTOM, LEFT, RIGHT, GAP = 0.68, 0.14, 0.085, 0.86, 0.13
#: The y axis both panels share, and the faint lines it is read against (Photo Finish's grid).
Y_SPAN, Y_GRID = (0.4, 1.0), "#bcc4ce"


def read_curves(path: Path = CURVES) -> dict[tuple[str, str], np.ndarray]:
    """``(set, pronoun) -> F-beta at clicks 0..150``, from one starting photo."""
    if not path.exists():
        raise SystemExit(f"{path.relative_to(REPO)} does not exist: it is the face-gap study's committed curves")
    by: dict[tuple[str, str], dict[int, float]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["K"] == K:
                by.setdefault((row["set"], row["pronoun"]), {})[int(row["t"])] = float(row["mean"])
    out = {}
    for name, _title in PANELS:
        for pronoun, *_ in GROUPS:
            curve = by.get((name, pronoun), {})
            if sorted(curve) != list(range(CLICKS + 1)):
                raise SystemExit(f"{path}: {name} / {pronoun} does not cover clicks 0..{CLICKS}")
            out[(name, pronoun)] = np.array([curve[t] for t in range(CLICKS + 1)])
    wrong = {k: (round(float(out[k][-1]), 2), v) for k, v in EXPECT.items() if round(float(out[k][-1]), 2) != v}
    if wrong:
        raise SystemExit(f"make-face-gap-fig: the study no longer says what the notes quote (got, pinned): {wrong}")
    return out


def figure(curves: dict[tuple[str, str], np.ndarray], panels: int) -> Figure:
    fig = plt.figure(figsize=FIG_SIZE)
    w = (RIGHT - LEFT - GAP) / 2
    clicks = np.arange(CLICKS + 1)
    for i, (name, title) in enumerate(PANELS[:panels]):
        ax = fig.add_axes([LEFT + i * (w + GAP), BOTTOM, w, TOP - BOTTOM])
        ax.set_xlim(0, CLICKS)
        ax.set_ylim(*Y_SPAN)
        ax.set_xticks([0, 50, 100, 150])
        ax.set_xticklabels(["start\nphoto", "50", "100", "150"])
        ax.set_yticks(np.arange(Y_SPAN[0], Y_SPAN[1] + 1e-9, 0.2))
        ax.set_yticks(np.arange(Y_SPAN[0], Y_SPAN[1] + 1e-9, 0.1), minor=True)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.yaxis.grid(True, which="both", color=Y_GRID, lw=1.3)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=18, pad=12)
        ax.set_xlabel("Clicks")
        if i == 0:
            ax.set_ylabel("Fβ, β = 1")
        ends = [float(curves[(name, pronoun)][-1]) for pronoun, *_ in GROUPS]
        heights = spread_labels(ends, gap=0.075 * (Y_SPAN[1] - Y_SPAN[0]))
        for (pronoun, label, colour, weight), y in zip(GROUPS, heights, strict=True):
            ax.plot(clicks, curves[(name, pronoun)], color=colour, lw=weight, solid_capstyle="round", zorder=3)
            ax.annotate(label, (CLICKS, y), xytext=(8, 0), textcoords="offset points", ha="left", va="center",
                        fontsize=16, color=colour, annotation_clip=False)  # fmt: skip
    return fig


def main() -> int:
    curves = read_curves()
    save(figure(curves, 2), OUT, "face-gap.webp", column=FULL_BLEED, tight=False)
    save(figure(curves, 1), OUT, "face-gap.build1.webp", column=FULL_BLEED, tight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
