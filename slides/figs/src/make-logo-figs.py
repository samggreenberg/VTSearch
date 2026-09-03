#!/usr/bin/env python
"""The two figures behind the *what does "logo detection" mean* appendix.

    python slides/figs/src/make-logo-figs.py

Writes `figs/logo-hits.png` and `figs/logo-jobs.png`, each with its build
stages. They exist to make one argument concretely rather than by assertion:
**"the Coke logo" is not a set of images, and no amount of model capability
turns it into one.**

The material is a real image search for *Coke logo*, and the eight results it
returned. Nothing of that search is committed and nothing of the mark is
redrawn — a slide that reproduced eight thumbnails of a live trademark at
150px would be illegible from the third row *and* be reproducing somebody's
mark to make a point about ambiguity. What the figures draw instead is the
part the argument actually needs, and the part a photograph hides: **which
visual attributes each result has**, and **which of them each job counts as a
positive**. That is the same move `calib-cost-knob` makes when it takes the
photographs away and leaves the marks — see `make-calib-figs.py` — and for the
same reason: the disagreement is over decisions, not over pixels.

The deck's own running example is `book`, and this appendix is the same claim
in a domain where the audience expects it *not* to bite. Everyone accepts that
"book" has a fuzzy edge. Almost nobody expects "the Coke logo" to, because a
logo is a designed artefact with a spec sheet — and it does anyway, harder,
because the disagreement is not at the edge of one concept but between four
different concepts sharing three words.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import (  # noqa: E402
    FULL_BLEED,
    LABEL_GAP_PT,
    save,
)

OUT = Path(__file__).resolve().parent.parent

INK = "#14181f"
SOFT = "#5b6472"
RULE = "#d8dee6"
NEUTRAL_FILL = "#e8ebef"
RED = "#b91c1c"  # the reject side, everywhere else in this deck

#: Shared with the calibration schematics so a label set at 16pt here is the
#: size it is there: the axes fill the figure, so the canvas is just the figure
#: rescaled. See `make-calib-figs.FLOW_UNIT_PT`.
UNIT_PT = 38.0

#: Exactly 16:9, and both figures are written with `tight=False`, so the canvas
#: maps one-to-one onto the 1280x720 slide. That is what makes `NOTCH` below a
#: rectangle in these coordinates rather than something to be discovered by
#: running the check.
CANVAS = (19.8, 11.0)

#: `slide_figure.TITLE_NOTCH_PX` converted into canvas units: the top-left
#: corner the slide overlays its headline on. `save()` enforces it; this is
#: here so the layouts can be *written* clear of it instead of nudged until
#: they pass. x in [0.93, 5.57], y in [7.26, 10.35].
_PX = CANVAS[0] / 1280.0
NOTCH_X1 = (60.0 + 300.0) * _PX
NOTCH_Y0 = CANVAS[1] - (42.0 + 200.0) * _PX

#: `slide_figure`'s spacing standard in these drawing units.
LABEL_GAP = LABEL_GAP_PT / UNIT_PT

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 15,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def _canvas() -> tuple[plt.Figure, plt.Axes]:
    """A blank 16:9 drawing whose axes fill the figure, in `CANVAS` units."""
    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.set_axis_off()
    return fig, ax


# ── figure 1: eight results, and no attribute they share ─────────────────────

#: How many stages `logo-hits` reveals in: the query and what came back; the
#: attributes each result actually has; and the reading — no column is full.
HITS_STAGES = 3

#: The attributes the eight results vary over, as (heading, ...) — one column
#: each, headings broken across two lines where a single word will not fit the
#: column pitch. Chosen as the things a *detector* would have to fire on or
#: not: what shape carries the mark, what colour it is, what word it spells.
HIT_ATTRS = (
    "script",
    "red",
    "red\nfield",
    "disc",
    "ribbon",
    "bottle",
    '"Coke"',
)

#: The eight results the search returned, each as (what it is, attributes it
#: has). Read off the actual hits, not invented: the wordmark reversed out of
#: a solid red field; the round red button badge; that badge with a contour
#: bottle beside it; the wordmark over the dynamic ribbon; "Coke" set in a
#: heavy sans that is not the script at all; the wordmark in flat black; the
#: wordmark in red on white; and Diet Coke, which is a different product.
#:
#: The column sums are the whole figure: script 7, red 7, and every other
#: column in low single figures. **No attribute is in all eight** — so there is
#: no feature a detector could key on that would return this set, and the set
#: is what a person typing the query got.
HITS = (
    ("white on a solid red field", ("script", "red", "red\nfield", "ribbon")),
    ("the round red badge", ("script", "red", "red\nfield", "disc")),
    ("that badge, with a bottle", ("script", "red", "red\nfield", "disc", "bottle")),
    ("wordmark over the ribbon", ("script", "red", "ribbon")),
    ('"Coke" in a heavy sans', ("red", '"Coke"')),
    ("the wordmark in black", ("script",)),
    ("the wordmark in red", ("script", "red")),
    ("Diet Coke", ("script", "red", '"Coke"')),
)

#: Where the matrix sits. The row labels run left from `LABEL_RIGHT` and the
#: tick columns run right from `TICKS_LEFT`, which is clear of `NOTCH_X1`; the
#: first row sits below `NOTCH_Y0`, so the column headings are the only thing
#: at notch height and they are all to the right of it.
LABEL_RIGHT = 6.6
TICKS_LEFT, TICKS_RIGHT = 7.6, 18.3
ROW_TOP, ROW_PITCH = 6.75, 0.72


def _col_x(index: int) -> float:
    """The centre of attribute column *index*, evenly spread across the ticks."""
    span = TICKS_RIGHT - TICKS_LEFT
    return TICKS_LEFT + span * (index + 0.5) / len(HIT_ATTRS)


def hits_fig() -> None:
    """Eight results for one query, and the attribute they have in common."""
    for stage in range(1, HITS_STAGES):
        save(_hits_stage(stage), OUT, f"logo-hits.build{stage}.png", column=FULL_BLEED, tight=False)
    save(_hits_stage(HITS_STAGES), OUT, "logo-hits.png", column=FULL_BLEED, tight=False)


def _hits_stage(stage: int) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the hits figure."""
    fig, ax = _canvas()

    # ── stage 1: the query, and the eight things that came back ──────────────
    # The search box is drawn because the argument is about a *query*: these
    # eight are not a curated set of hard cases, they are what one ordinary
    # search returned, in the order it returned them.
    ax.add_patch(
        FancyBboxPatch(
            (TICKS_LEFT, 9.35),
            7.0,
            0.86,
            boxstyle="round,pad=0,rounding_size=0.43",
            facecolor="white",
            edgecolor=SOFT,
            linewidth=1.6,
        )
    )
    ax.text(TICKS_LEFT + 0.42, 9.78, "Coke logo", ha="left", va="center", fontsize=17, color=INK)
    ax.text(
        TICKS_LEFT + 7.0 + 4 * LABEL_GAP,
        9.78,
        "8 results",
        ha="left",
        va="center",
        fontsize=15,
        color=SOFT,
    )

    for row, (name, _) in enumerate(HITS):
        ax.text(LABEL_RIGHT, ROW_TOP - row * ROW_PITCH, name, ha="right", va="center", fontsize=16, color=INK)

    # ── stage 2: what each of them actually has ──────────────────────────────
    if stage >= 2:
        for index, heading in enumerate(HIT_ATTRS):
            ax.text(
                _col_x(index),
                ROW_TOP + 0.62,
                heading,
                ha="center",
                va="bottom",
                fontsize=15,
                color=SOFT,
                linespacing=1.15,
            )
        for row, (_, attrs) in enumerate(HITS):
            y = ROW_TOP - row * ROW_PITCH
            for index, heading in enumerate(HIT_ATTRS):
                if heading not in attrs:
                    continue
                ax.text(_col_x(index), y, "●", ha="center", va="center", fontsize=17, color=INK)

    # ── stage 3: the reading — every column has a hole in it ─────────────────
    # The two nearly-full columns are the ones worth marking: they are the two
    # attributes anybody would name if asked what a Coke logo *is*, and each of
    # them is missing from a result the search returned as one.
    if stage >= HITS_STAGES:
        for index, heading in enumerate(HIT_ATTRS):
            missing = [row for row, (_, attrs) in enumerate(HITS) if heading not in attrs]
            if len(missing) > 2:
                continue
            for row in missing:
                ax.text(
                    _col_x(index),
                    ROW_TOP - row * ROW_PITCH,
                    "✗",
                    ha="center",
                    va="center",
                    fontsize=19,
                    color=RED,
                    fontweight="bold",
                )
        ax.text(
            TICKS_LEFT,
            ROW_TOP - (len(HITS) - 1) * ROW_PITCH - 0.95,
            "no attribute is in all eight",
            ha="left",
            va="top",
            fontsize=17,
            color=RED,
        )
    return fig


# ── figure 2: four jobs, and the four different answers they want ────────────

#: How many stages `logo-jobs` reveals in: the six candidates in order; then
#: one job's positives per stage, three of them nested and the fourth not.
JOBS_STAGES = 5

#: The candidates, ordered left to right by how unmistakably each *is* the
#: mark. The order is a claim the figure makes and the slide defends: it is not
#: the only defensible order, and it does not have to be — the argument is that
#: **no single cut on it serves all four jobs**, which survives reordering.
#:
#: Written two lines deep and hung on **alternating ranks** below the axis,
#: each on its own leader. Six labels sharing one rank get 2.1 units apiece —
#: about nine characters — and "delivery truck" is fourteen, so a single rank
#: printed them through one another. Two ranks double the pitch to 4.2 units,
#: which every line here clears; the leader is what keeps a label bound to its
#: own tick once it no longer sits directly under it.
CANDIDATES = (
    "a red\ndelivery truck",
    "a rival's\nswash script",
    '"Coke" set\nin Arial',
    "Diet Coke",
    "a red disc,\ntoo far to read",
    "the wordmark,\nto spec",
)

#: Each job, as (name, the candidate indices it counts as a **positive**, is it
#: a cut on this axis at all). The first three are nested — one slider, three
#: settings, which is the Inclusion knob of Part 3 wearing a different hat. The
#: fourth is the point of the slide: trademark enforcement wants the passing-off
#: cases and does *not* want the genuine mark, so its positives are exactly the
#: items every other job throws away. It is not a looser or tighter cut. It is a
#: different concept that the same three words also name.
JOBS = (
    ("brand compliance", (5,), True),
    ("sponsorship seconds", (3, 4, 5), True),
    ("archive retrieval", (0, 1, 2, 3, 4, 5), True),
    ("trademark enforcement", (1, 2), False),
)

AXIS_LEFT, AXIS_RIGHT, AXIS_Y = 6.6, 18.3, 9.55
BAR_TOP, BAR_PITCH, BAR_H = 5.85, 1.32, 0.5

#: How far below the axis each rank of candidate labels hangs. The axis starts
#: right of `NOTCH_X1` for the same reason the calibration panels do — a top
#: row that spans the drawing cannot be panned out of the title notch — and
#: the leftmost label is what actually sets `AXIS_LEFT`, since it is centred
#: half a column left of the first tick.
LABEL_RANKS = (0.34, 1.42)


def _cand_x(index: int) -> float:
    """The centre of candidate *index* on the axis."""
    span = AXIS_RIGHT - AXIS_LEFT
    return AXIS_LEFT + span * (index + 0.5) / len(CANDIDATES)


def jobs_fig() -> None:
    """One phrase, four jobs, and the four different sets they are asking for."""
    for stage in range(1, JOBS_STAGES):
        save(_jobs_stage(stage), OUT, f"logo-jobs.build{stage}.png", column=FULL_BLEED, tight=False)
    save(_jobs_stage(JOBS_STAGES), OUT, "logo-jobs.png", column=FULL_BLEED, tight=False)


def _jobs_stage(stage: int) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the jobs figure."""
    fig, ax = _canvas()

    # ── stage 1: the candidates, in order ────────────────────────────────────
    # The axis is indented from the left rather than spanning the drawing: a
    # top row that spans cannot be panned out of the title notch, and the right
    # margin is what pays for it (`slides/STYLE.md`).
    ax.annotate(
        "",
        xy=(AXIS_RIGHT, AXIS_Y),
        xytext=(AXIS_LEFT, AXIS_Y),
        arrowprops={"arrowstyle": "-|>", "color": INK, "linewidth": 1.8, "shrinkA": 0, "shrinkB": 0},
    )
    ax.text(
        AXIS_RIGHT,
        AXIS_Y + 0.36,
        "more unmistakably the mark",
        ha="right",
        va="bottom",
        fontsize=15,
        color=SOFT,
    )
    for index, name in enumerate(CANDIDATES):
        x, drop = _cand_x(index), LABEL_RANKS[index % len(LABEL_RANKS)]
        ax.plot([x, x], [AXIS_Y - 0.16, AXIS_Y + 0.16], color=INK, linewidth=1.8, zorder=3)
        # The leader runs from under the tick to a label gap above the label,
        # so the label's nearest ink is its own leader and not its neighbour.
        ax.plot(
            [x, x],
            [AXIS_Y - 0.16, AXIS_Y - drop + LABEL_GAP],
            color=RULE,
            linewidth=1.2,
            zorder=1,
        )
        ax.text(x, AXIS_Y - drop, name, ha="center", va="top", fontsize=15, color=INK, linespacing=1.2)

    # ── stages 2-5: one job's positives per stage ────────────────────────────
    span = (AXIS_RIGHT - AXIS_LEFT) / len(CANDIDATES)
    for job, (name, keeps, is_cut) in enumerate(JOBS):
        if stage < job + 2:
            continue
        y = BAR_TOP - job * BAR_PITCH
        colour = INK if is_cut else RED
        # A wash across the whole axis first, so the bar reads as a selection
        # *out of* the six rather than as a bar chart of some quantity.
        ax.add_patch(
            Rectangle(
                (AXIS_LEFT, y - BAR_H / 2),
                AXIS_RIGHT - AXIS_LEFT,
                BAR_H,
                facecolor=NEUTRAL_FILL,
                edgecolor=RULE,
                linewidth=1.0,
                zorder=1,
            )
        )
        left = _cand_x(min(keeps)) - span / 2
        ax.add_patch(
            Rectangle(
                (left, y - BAR_H / 2),
                span * len(keeps),
                BAR_H,
                facecolor=colour,
                edgecolor="none",
                zorder=2,
            )
        )
        ax.text(AXIS_LEFT - 4 * LABEL_GAP, y, name, ha="right", va="center", fontsize=16, color=colour)
        if not is_cut:
            ax.text(
                AXIS_RIGHT,
                y - BAR_H / 2 - 3 * LABEL_GAP,
                "not a looser cut — the sign has flipped",
                ha="right",
                va="top",
                fontsize=16,
                color=RED,
            )
    return fig


def main() -> None:
    hits_fig()
    jobs_fig()


if __name__ == "__main__":
    main()
