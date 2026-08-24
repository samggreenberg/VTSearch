#!/usr/bin/env python
"""Mechanism figures for the calibration deck.

Uses matplotlib and vtscore, both already project dependencies, so a normal
checkout install is enough. Run from the repo root:

    python slides/figs/src/make-calib-figs.py

The loop schematic is hand-drawn: it is the deck's introduction to the
application, so it carries plain-English labels rather than the notation the
later schematics share.

The two mixture figures run the *real* vtscore estimators (`fit_score_gmm`,
`fit_anchored_score_gmm`) on labelled synthetic data — schematic inputs, real
code. The schedule figure plots the shipped `blend_schedules` registry verbatim.
The decomposition figure re-plots published numbers from
`docs/experiments/gmm-cut/REPORT-2881.md` (the #2879 re-measure).
"""

import functools
import sys
from pathlib import Path

# Ensure the repo root (where app.py lives) is importable no matter the cwd:
# ``python slides/figs/src/x.py`` only puts the script's own dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from matplotlib.patches import Ellipse, FancyArrow, FancyArrowPatch, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import (  # noqa: E402
    FULL_BLEED,
    LABEL_GAP_PT,
    OBJECT_GAP_PT,
    SIDEBAR,
    SIDEBAR_WIDE,
    save,
    tight_box,
)

from vtscore.training.blend_schedules import BlendContext, get_schedule
from vtscore.training.thresholds import (
    FOLD_ANCHOR_WEIGHT,
    FoldAnchoredCut,
    GmmFit1D,
    acquisition_inclusion,
    conformal_threshold,
    fit_anchored_score_gmm,
    fit_score_gmm,
    gmm_cut_from_fit,
    inclusion_cost_weights,
)

OUT = Path(__file__).resolve().parent.parent

INK = "#14181f"
SOFT = "#5b6472"
RULE = "#d8dee6"
NEUTRAL_FILL = "#e8ebef"
BLUE = "#0b5fa5"  # production / the shipped thing
RUST = "#b45309"  # the Bad component / cross-calibration
GREEN = "#0d8a5f"  # the Good component

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        # Subscripts and the parentheses around them are set as mathtext, which
        # sizes them as a unit (see `_sub`); this keeps that in the figure's own
        # face rather than switching to Computer Modern.
        "mathtext.fontset": "dejavusans",
        "font.size": 15,
        "text.color": INK,
        "axes.edgecolor": SOFT,
        "axes.labelcolor": INK,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": SOFT,
        "ytick.color": SOFT,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)


def gaussian(x: np.ndarray, mu: float, var: float) -> np.ndarray:
    return np.exp(-0.5 * (x - mu) ** 2 / var) / np.sqrt(2 * np.pi * var)


#: How many build stages the loop schematic reveals in: the corpus; the
#: detector; the scores; the cut; what the cut keeps; what the cut asks about
#: next; the retrain that closes the loop. Stage 7 is the committed final
#: figure. The fork at stages 5-6 is the point of the slide — one cut, two
#: jobs — so those two steps are deliberately separate reveals.
LOOP_STAGES = 7

#: How many build stages the cross-calibration schematic reveals in
#: (issue #3208): D₀ → M₀; the split; the fold models; M₁ scores D₂; θ₁;
#: M₂ scores D₁ and θ₂; the average. Stage 7 is the committed final figure.
XCAL_FLOW_STAGES = 7

#: The points one drawing unit of a *schematic* figure is drawn at. The axes
#: fill the figure (no subplot margins), so the figure size is just the canvas
#: rescaled — which is what keeps every geometry constant below in a fixed
#: relationship to the font sizes, whatever the figure's inches happen to be.
#:
#: Every schematic in the calibration progression shares this scale and this
#: canvas *height*: the deep-dive deck fits each one into the same
#: `bg right:70%` slot, so a figure that is height-limited at the same number
#: of units is scaled by the same factor as its neighbours, and a 16pt label
#: renders at the same size on every slide of the progression. Widths may
#: differ; heights may not. (A figure re-used at a *narrower* slot elsewhere —
#: `xcal.short` — still has to clear the type floor there; see
#: `xcal_flow_fig`.)
FLOW_UNIT_PT = 38.0
FLOW_CANVAS_H = 11.0

XCAL_CANVAS = (13.2, FLOW_CANVAS_H)

#: `slide_figure`'s spacing standard, in a schematic's drawing units.
LABEL_GAP = LABEL_GAP_PT / FLOW_UNIT_PT
OBJECT_GAP = OBJECT_GAP_PT / FLOW_UNIT_PT

#: The fold-model boxes. `_box_edge` has to agree with what `_model_box` draws.
MODEL_W, MODEL_H = 0.85, 0.62

#: The outlined block arrows: shaft width, then the head's width and length.
ARROW_W, ARROW_HEAD_W, ARROW_HEAD_L = 0.5, 0.66, 0.32

#: A 16pt label's cap height in drawing units, and how far a score line's own
#: label sits above the line: clear of the tallest check mark by one label gap.
CAP_16 = 0.36
SCORE_LABEL_LIFT = 0.08 + 0.29 + LABEL_GAP

#: The half-width the held-out score marks below are quoted at, and the marks
#: themselves: where each fold's Bad and Good votes land on its own score line,
#: and where that fold's cut goes. Quoted once and rescaled by `_line_marks`,
#: because two figures draw the same two lines at two different lengths — the
#: cross-calibration schematic can spend 2.6 units on a line, while the blend
#: schematic has to fit a whole second branch beside them. One ordering is
#: imperfect on purpose: a Bad lands above θ₂.
SCORE_MARK_HALF = 2.6
SCORE_MARKS_1 = {"bad": (-2.25, -1.53, -0.81, -0.09), "good": (0.81, 1.53, 2.25), "theta": 0.36}
SCORE_MARKS_2 = {"bad": (-2.25, -1.62, -0.98, 0.48), "good": (0.0, 1.1, 1.82), "theta": -0.48}


def _line_marks(marks: dict, half: float) -> dict:
    """`marks` rescaled from `SCORE_MARK_HALF` to a line of half-width *half*."""
    k = half / SCORE_MARK_HALF
    return {
        "bad": [x * k for x in marks["bad"]],
        "good": [x * k for x in marks["good"]],
        "theta_x": marks["theta"] * k,
    }


def _sub(expr: str) -> str:
    """A subscripted label, set in the figure's own sans face.

    Unicode subscript glyphs (`M₁(D₂)`) leave the parentheses at their plain
    height while the content they enclose reaches a subscript's depth, so the
    parens read as too small for what they hold (issue #3217). mathtext sizes
    the two together. This is *not* the `$…$` that `slides/STYLE.md` bans:
    that rule is about Marp slide prose, where MathJax would switch the
    formula to Computer Modern mid-sentence. Here `mathtext.fontset` is the
    same DejaVu Sans as every other label in the figure.
    """
    return rf"$\mathregular{{{expr}}}$"


def xcal_flow_fig() -> None:
    """Schematic of the original cross-calibration idea (issues #3207/#3208).

    Follows the issue's hand mockup: this is the *iteration-1* slide, so it
    draws the simple initial version — partition the labelled data in half,
    train a model per half, score the half each model never saw, find a cut
    per half, and average the two cuts. The refinements the shipped code adds
    on top (pooled folds, the conformal quantile, re-drawn splits) are later
    iterations of the deck's story and deliberately absent here.

    Besides the final figure, this writes the build stages the deck's
    `<!-- build: ... -->` markers reveal through: stage k draws the first k
    steps of the mechanism onto the same fixed canvas, and every stage is
    cropped to the *final* stage's box, so across the build slides the
    drawing assembles in place rather than recentring itself.
    """
    # Type-floor-checked against `SIDEBAR`, not `SIDEBAR_WIDE`, because this one
    # figure appears at two widths: the deep-dive slide gives it 70% and drops
    # its bullets, while `xcal.short` keeps the bullets that carry the brief
    # deck's argument and so can only spare 56%. A figure used at two slot
    # widths has to clear the floor at the narrower of them.
    final = _xcal_flow_stage(XCAL_FLOW_STAGES)
    box = tight_box(final)
    for stage in range(1, XCAL_FLOW_STAGES):
        save(
            _xcal_flow_stage(stage),
            OUT,
            f"calib-xcal-flow.build{stage}.png",
            column=SIDEBAR,
            box=box,
        )
    save(final, OUT, "calib-xcal-flow.png", column=SIDEBAR, box=box)


def _data_block(ax: plt.Axes, x0: float, y0: float, w: float, h: float, split: bool = False) -> None:
    """A block of labelled media: an amorphous green (Good) region over a red (Bad) one."""
    good_h = 0.42 * h
    ax.add_patch(
        Rectangle(
            (x0, y0 + h - good_h),
            w,
            good_h,
            facecolor="white",
            edgecolor=GREEN,
            hatch="//////",
            linewidth=0,
            zorder=2,
        )
    )
    ax.add_patch(
        Rectangle((x0, y0), w, h - good_h, facecolor="white", edgecolor=RUST, hatch="\\\\\\", linewidth=0, zorder=2)
    )
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=3))
    ax.plot([x0, x0 + w], [y0 + h - good_h] * 2, color=INK, linewidth=1.0, zorder=3)
    if not split:
        return
    ax.plot([x0 + w / 2] * 2, [y0, y0 + h], color=INK, linewidth=1.6, zorder=3)
    # D₁ and D₂ ride *inside* the halves they name, each on a white disc that
    # clears the Good/Bad divider and the hatching under it (issue #3217).
    # Hung below the block instead, each label sat further from the half it
    # named than from the train arrow leaving that half.
    for cx, name in ((x0 + w / 4, "D_1"), (x0 + 3 * w / 4, "D_2")):
        _disc_label(ax, cx, y0 + h / 2, name)


def _disc_label(ax: plt.Axes, cx: float, cy: float, name: str) -> None:
    """Name a data block from *inside* it, on a white disc.

    The disc clears the Good/Bad divider and the hatching under it, so the
    name reads cleanly without being hung outside the block — where it would
    sit as close to whatever arrow arrives next as to the thing it names
    (issue #3217). Used for the halves D₁/D₂, and for any whole block that has
    an arrow coming into the space above it.
    """
    ax.add_patch(Ellipse((cx, cy), 0.95, 0.62, facecolor="white", edgecolor="none", zorder=4))
    ax.text(cx, cy, _sub(name), ha="center", va="center", fontsize=16, color=INK, zorder=5)


def _box_edge(cx: float, cy: float, toward: tuple[float, float], gap: float) -> tuple[float, float]:
    """The point `gap` outside a model box, on the ray from its centre to `toward`.

    Measuring an arrow's clearance from where its own line leaves the box —
    rather than from the box's nearest flat edge — is what lets one spacing
    constant hold for every arrow whatever angle it arrives at.
    """
    dx, dy = toward[0] - cx, toward[1] - cy
    ux, uy = np.array([dx, dy]) / float(np.hypot(dx, dy))
    t = min(
        MODEL_W / 2 / abs(ux) if ux else np.inf,
        MODEL_H / 2 / abs(uy) if uy else np.inf,
    )
    return cx + (t + gap) * ux, cy + (t + gap) * uy


def _arrow(ax: plt.Axes, xy_from: tuple[float, float], xy_to: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(xy_from, xy_to, arrowstyle="-|>", mutation_scale=14, color=INK, linewidth=1.6, zorder=2)
    )


def _labeled_arrow(
    ax: plt.Axes, xy_from: tuple[float, float], xy_to: tuple[float, float], label: str, z: float = 2.0
) -> None:
    """An outlined block arrow with an open core, *label* written along the shaft.

    The label sits at the centre of the *whole* arrow — head included — both
    length-wise and width-wise, rotated to the arrow's own angle. Centring it
    in the shaft alone (which is what the arithmetic wants to do, since the
    shaft is the part with room in it) reads as a label shoved towards the
    tail, because the eye counts the head as space like everything else
    (issue #3217). Every arrow here is drawn long enough for that to fit.
    """
    (x0, y0), (x1, y1) = xy_from, xy_to
    dx, dy = x1 - x0, y1 - y0
    ax.add_patch(
        FancyArrow(
            x0,
            y0,
            dx,
            dy,
            width=ARROW_W,
            head_width=ARROW_HEAD_W,
            head_length=ARROW_HEAD_L,
            length_includes_head=True,
            facecolor="white",
            edgecolor=INK,
            linewidth=1.4,
            zorder=z,
        )
    )
    angle = float(np.degrees(np.arctan2(dy, dx)))
    if angle < -90 or angle > 90:  # keep the label reading left-to-right
        angle += 180
    ax.text(
        x0 + dx * 0.5,
        y0 + dy * 0.5,
        label,
        rotation=angle,
        ha="center",
        va="center",
        fontsize=15,
        color=INK,
        zorder=z + 0.05,
    )


def _model_box(ax: plt.Axes, cx: float, cy: float, label: str) -> None:
    ax.add_patch(
        Rectangle(
            (cx - MODEL_W / 2, cy - MODEL_H / 2),
            MODEL_W,
            MODEL_H,
            facecolor="white",
            edgecolor=INK,
            linewidth=1.6,
            zorder=3,
        )
    )
    ax.text(cx, cy, _sub(label), ha="center", va="center", fontsize=16, color=INK, zorder=4)


def _score_line(
    ax: plt.Axes,
    cx: float,
    y: float,
    half: float,
    label: str,
    bad: list[float],
    good: list[float],
    theta_x: float,
    theta: str,
    cut: bool = True,
) -> None:
    """Held-out scores on a number line: Bad low, Good high, a cut between.

    Everything here is a label of the line, so it sits a label gap from it:
    the ticks and marks touch the line, and the two texts clear the tallest
    mark on their side by `LABEL_GAP`.
    """
    ax.plot([cx - half, cx + half], [y, y], color=INK, linewidth=1.8, zorder=2)
    ax.text(cx, y + SCORE_LABEL_LIFT, _sub(label), ha="center", va="bottom", fontsize=16, color=INK)
    for x in bad:
        ax.text(cx + x, y - 0.12, "✗", ha="center", va="top", fontsize=16, color=RUST, fontweight="bold")
    for x in good:
        ax.text(cx + x, y + 0.08, "✓", ha="center", va="bottom", fontsize=16, color=GREEN, fontweight="bold")
    if not cut:
        return
    ax.plot([cx + theta_x] * 2, [y - 0.32, y], color=INK, linewidth=2.2, zorder=3)
    ax.text(cx + theta_x, y - 0.32 - LABEL_GAP, _sub(theta), ha="center", va="top", fontsize=16, color=INK)


def _xcal_flow_stage(stage: int) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic.

    Layout rules carried over from the mockup on purpose: the split is drawn,
    not written — D₀ is one block with a centre divider whose halves are
    labelled D₁ and D₂ inside themselves; the train arrows D₁ → M₁ and
    D₂ → M₂ are vertical, so the only diagonals are the two scoring paths
    D₂ → M₁ → M₁(D₂) and D₁ → M₂ → M₂(D₁), which cross into the X that names
    cross-calibration; and one held-out ordering is imperfect — a Bad lands
    above θ₂ — because a fold model's ranking of votes it never saw is not
    trivially clean. Green = Good media, red = Bad media (amorphous regions);
    everything else ink on white.

    Which half of a scoring path gets the outlined block arrow is forced by
    the geometry, not taste (issue #3217). The two D → M legs mirror each
    other about the block's divider, so their midpoints — where a centred
    label has to go — sit within a few tenths of that divider whatever the
    slope: two block arrows there would print "score" twice on the same spot.
    The M → scores legs diverge instead, so that is where the two labels can
    both be read. The X is then drawn as two plain strokes crossing, which is
    what an X wants to be anyway.
    """
    fig, ax = plt.subplots(figsize=tuple(c * FLOW_UNIT_PT / 72 for c in XCAL_CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, XCAL_CANVAS[0])
    ax.set_ylim(0, XCAL_CANVAS[1])
    ax.set_axis_off()

    # ── layout ────────────────────────────────────────────────────────────────
    # Everything is derived from the block downwards, so the two spacing
    # constants really are the only spacings in the figure.
    bx, block_w, block_h, block_top = 6.5, 4.8, 1.05, 10.35
    block_y0, block_x0 = block_top - block_h, bx - block_w / 2
    m1x, m2x = bx - block_w / 4, bx + block_w / 4  # the halves, and the models under them

    train_len, score_len, slope = 1.75, 1.95, 0.75  # slope = dx per dy of a scoring stroke
    train_tail_y = block_y0 - OBJECT_GAP
    my = train_tail_y - train_len - OBJECT_GAP - MODEL_H / 2

    down_left = (m1x - slope, my - 1.0)
    exit1 = _box_edge(m1x, my, down_left, OBJECT_GAP)
    ux, uy = np.array([slope, 1.0]) / float(np.hypot(slope, 1.0))
    tip1 = (exit1[0] - score_len * ux, exit1[1] - score_len * uy)
    # The arrow points at the score-line *group*, so it stops an object gap
    # above the group's topmost ink — which is its label, not the line.
    line_y, line_half = tip1[1] - (OBJECT_GAP + CAP_16 + SCORE_LABEL_LIFT), 2.6
    cx1 = tip1[0] - 0.4 * line_half
    cx2 = 2 * bx - cx1

    data_block = functools.partial(_data_block, ax)
    arrow = functools.partial(_arrow, ax)
    labeled_arrow = functools.partial(_labeled_arrow, ax)
    model_box = functools.partial(_model_box, ax)
    score_line = functools.partial(_score_line, ax, y=line_y, half=line_half)

    # ── stage 1: D0 and the model trained on all of it ────────────────────────
    ax.text(bx, block_top + LABEL_GAP, _sub("D_0"), ha="center", va="bottom", fontsize=16, color=INK)
    data_block(block_x0, block_y0, block_w, block_h, split=stage >= 2)
    good_h = 0.42 * block_h
    ax.text(block_x0 - LABEL_GAP, block_top - good_h / 2, "Good", ha="right", va="center", fontsize=15, color=GREEN)
    ax.text(
        block_x0 - LABEL_GAP,
        block_y0 + (block_h - good_h) / 2,
        "Bad",
        ha="right",
        va="center",
        fontsize=15,
        color=RUST,
    )
    train_x = bx + block_w / 2 + OBJECT_GAP
    labeled_arrow((train_x, block_y0 + block_h / 2), (train_x + 2.4, block_y0 + block_h / 2), "train")
    model_box(train_x + 2.4 + OBJECT_GAP + MODEL_W / 2, block_y0 + block_h / 2, "M_0")

    # ── stage 2: split D0 in half — D1 and D2 ─────────────────────────────────
    # Drawn by `_data_block(split=True)` above: the divider and the two labels
    # inside the halves.

    # ── stage 3: train a model per half ───────────────────────────────────────
    # The train arrows are vertical; the only diagonals are the scoring paths
    # D₂ → M₁ → M₁(D₂) and D₁ → M₂ → M₂(D₁), whose crossing draws the X that
    # names cross-calibration.
    if stage >= 3:
        for mx, name in ((m1x, "M_1"), (m2x, "M_2")):
            model_box(mx, my, name)
            labeled_arrow((mx, train_tail_y), (mx, train_tail_y - train_len), "train")

    # ── stage 4: M1 scores the half it never saw ──────────────────────────────
    # Each scoring path is one geometric line: it runs from under the opposite
    # half, through the M box, on down to the score line — the entry stroke and
    # the exit arrow are collinear through the box's centre, so the eye reads
    # one straight path and, once both are drawn, the two paths cross in a
    # symmetric X.
    if stage >= 4:
        entry_tail = (m1x + slope * (train_tail_y - my), train_tail_y)
        arrow(entry_tail, _box_edge(m1x, my, entry_tail, OBJECT_GAP))
        labeled_arrow(exit1, tip1, "score", z=2.1)
        # ── stage 5: cut M1's held-out scores at θ1 ───────────────────────────
        score_line(
            cx1,
            label="M_1(D_2)",
            **_line_marks(SCORE_MARKS_1, line_half),
            theta=r"\theta_1",
            cut=stage >= 5,
        )

    # ── stage 6: the same for M2 — score D1, cut at θ2 ────────────────────────
    if stage >= 6:
        entry_tail = (m2x - slope * (train_tail_y - my), train_tail_y)
        arrow(entry_tail, _box_edge(m2x, my, entry_tail, OBJECT_GAP))
        exit2 = (2 * bx - exit1[0], exit1[1])
        labeled_arrow(exit2, (2 * bx - tip1[0], tip1[1]), "score", z=2.1)
        score_line(
            cx2,
            label="M_2(D_1)",
            **_line_marks(SCORE_MARKS_2, line_half),
            theta=r"\theta_2",
        )

    # ── stage 7: average the cuts ─────────────────────────────────────────────
    if stage >= 7:
        theta_bottom = line_y - 0.32 - LABEL_GAP - CAP_16
        avg_y = theta_bottom - OBJECT_GAP - 0.21
        ax.text(
            bx,
            avg_y,
            _sub(r"\theta_0 = avg(\theta_1,\, \theta_2)"),
            ha="center",
            va="center",
            fontsize=16.5,
            color=INK,
        )
        ax.text(
            bx,
            avg_y - 0.21 - OBJECT_GAP - 0.23,
            "return " + _sub(r"(M_0,\, \theta_0)"),
            ha="center",
            va="center",
            fontsize=18,
            color=INK,
        )

    return fig


#: How many build stages the label-free mixture schematic reveals in: the
#: unlabeled haystack; the votes drawn out of it and the model they train; the
#: haystack scored into a histogram; the two modes *claimed* as Bad and Good;
#: the cut between them.
GMM_FLOW_STAGES = 5

#: Same height as `XCAL_CANVAS`, deliberately — see `FLOW_CANVAS_H`.
GMM_CANVAS = (13.2, FLOW_CANVAS_H)

#: Bins in the schematic's score histogram. Enough to show two modes and the
#: dip between them; few enough that one bar is still a visible object at the
#: rendered size.
GMM_FLOW_BINS = 44


def gmm_flow_fig() -> None:
    """Schematic of the label-free mixture cut — iteration 2 of the line.

    The second figure of the calibration progression (issue #3218), and drawn
    to be recognisably the *first* one with a new source of data bolted on:
    `D₀ —train→ M₀` is carried over from `calib-xcal-flow` unchanged, and above
    it sits `D₋₁`, the unlabeled haystack the votes were drawn out of. The
    mechanism is then the other direction round the loop — `M₀` scores all of
    `D₋₁`, and a cut is read off the shape of the resulting score histogram
    without any label being consulted.

    The haystack block is the same height as the votes' block and carries no
    Good/Bad hatching, which is the one thing the figure says by shape alone:
    these are the same kind of media, there are far more of them, and their
    classes are unknown — not absent.

    What the figure deliberately leaves to the speaker is that naming the low
    mode Bad and the high mode Good is this estimator's *assumption*. It reads
    the shape of the distribution and nothing else, so the identification is
    exactly the part it cannot support, and #2836 measured it wrong: a fitted
    high-component weight of 0.35 against a true prevalence of 0.09 — "confi-
    dently scored", not "true match". That is the ceiling iteration 4 lifts by
    letting votes pin the components down.
    """
    fit, scores = _haystack_scores()
    final = _gmm_flow_stage(GMM_FLOW_STAGES, fit, scores)
    box = tight_box(final)
    for stage in range(1, GMM_FLOW_STAGES):
        save(
            _gmm_flow_stage(stage, fit, scores),
            OUT,
            f"calib-gmm-flow.build{stage}.png",
            column=SIDEBAR_WIDE,
            box=box,
        )
    save(final, OUT, "calib-gmm-flow.png", column=SIDEBAR_WIDE, box=box)


def _haystack_scores() -> tuple[GmmFit1D, np.ndarray]:
    """A synthetic haystack and the **real** `fit_score_gmm` fit of it.

    Schematic input, real estimator — the house rule for the mixture figures.
    The positive rate is chosen so both modes are visible at slide size —
    density-normalised, a realistic 2-5% positive haystack draws its high mode
    at a fortieth of the low one's height and the picture stops being bimodal
    to look at. Real haystacks are sparser than this, which only sharpens the
    identification problem the slide is there to raise.
    """
    rng = np.random.default_rng(42)
    neg = rng.normal(0.26, 0.082, 4800)
    pos = rng.normal(0.74, 0.078, 1200)
    scores = np.clip(np.concatenate([neg, pos]), 0.0, 1.0)
    fit = fit_score_gmm(scores)
    assert fit is not None
    return fit, scores


def _haystack_block(ax: plt.Axes, x0: float, y0: float, w: float, h: float) -> None:
    """A block of **unlabeled** media: the same outline as `_data_block`, flat.

    The absence of the Good/Bad hatching is the whole content of this shape.
    Beside a hatched `_data_block` it reads as "same kind of thing, classes
    unknown", which is exactly what the haystack is.
    """
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor=NEUTRAL_FILL, edgecolor=INK, linewidth=1.6, zorder=2))


def _staircase(x0: float, y_base: float, w: float, sy: float, edges, density, first: int, last: int) -> "Polygon":
    """The histogram's own outline over bins ``first..last``, as a closed shape.

    Drawing the bars as one polygon rather than N rectangles is what lets the
    same silhouette be re-drawn between build stages — bare while it is just
    "the shape of the data", then split in two and hatched once the fit claims
    which half is which — without the outline moving by a pixel. It is also
    what keeps the unfitted form's bar separators single-stroked; see
    `_score_histogram`.
    """
    pts = [(x0 + edges[first] * w, y_base)]
    for i in range(first, last + 1):
        top = y_base + density[i] * sy
        pts.append((x0 + edges[i] * w, top))
        pts.append((x0 + edges[i + 1] * w, top))
    pts.append((x0 + edges[last + 1] * w, y_base))
    return Polygon(pts, closed=True)


#: Height below which a fitted component's tail stops being drawn, as a
#: fraction of the panel height. See `_score_histogram`.
TAIL_FLOOR = 0.012

#: Stroke width of an unfitted histogram's bars. The narrowest panel in the
#: progression is `calib-quantile-flow`'s, where `GMM_FLOW_BINS` bars share
#: 4.85 drawing units — about 4pt each, or six slide pixels — so the two edges
#: of a bar and the white between them are all within a pixel or two of each
#: other. Any heavier and the bars close up into the solid block they replaced.
BAR_EDGE_LW = 0.9

#: Nominal height of one "?" in the hatch, in drawing units, and the lattice
#: pitch as a multiple of it.
QUERY_GLYPH = 0.19
QUERY_PITCH = 1.32


def _question_hatch(ax: plt.Axes, region: "Polygon", color: str, z: float) -> None:
    """Fill *region* with a lattice of small question marks in *color*.

    The figure's one texture that is also an argument: the mixture has read no
    labels, so "this hump is the Bad one" is a guess, and the fill says so at
    a glance rather than in the speaker notes. It is drawn on the same footing
    as `_data_block`'s hatching — a property of the region, not a label of it.

    Deliberately a scatter of mathtext *markers* rather than `ax.text` glyphs.
    A marker is a path, so it is texture the way a hatch is texture; text would
    be read by `slide_figure.enforce_type_floor` as the smallest label in the
    figure and would fail the 20px floor — correctly, for a label, which this
    is not. Rows are staggered so the lattice reads as a fill rather than as a
    grid of columns.
    """
    # Bounds come off the polygon's own vertices, which are in drawing units.
    # `Patch.get_extents()` would report *display* pixels once the patch has
    # been added to an Axes, and a lattice laid out on those numbers lands
    # thousands of units off-canvas, where the clip silently eats all of it.
    xy = region.get_xy()
    bx0, by0 = xy[:, 0].min(), xy[:, 1].min()
    bx1, by1 = xy[:, 0].max(), xy[:, 1].max()
    pitch = QUERY_GLYPH * QUERY_PITCH
    rows = np.arange(by0 + 0.35 * pitch, by1, pitch)
    xs, ys = [], []
    for k, y in enumerate(rows):
        for x in np.arange(bx0 + (0.5 * pitch if k % 2 else 0.0), bx1, pitch):
            xs.append(x)
            ys.append(y)
    if not xs:
        return
    dots = ax.scatter(
        xs,
        ys,
        marker="$?$",
        s=(QUERY_GLYPH * FLOW_UNIT_PT) ** 2,
        color=color,
        linewidths=0,
        zorder=z,
    )
    dots.set_clip_path(region)


def _score_histogram(
    ax: plt.Axes,
    x0: float,
    y_base: float,
    w: float,
    h: float,
    fit: "GmmFit1D | None",
    scores: np.ndarray,
    *,
    fill: str,
    mu_labels: bool = True,
) -> None:
    """The haystack's score histogram, drawn in the schematic's drawing units.

    A `bg`-slot schematic cannot host a real Axes without inheriting its own
    scales and margins, so the distribution is rasterised by hand into the
    rectangle `(x0, y_base, w, h)`: score 0-1 maps across `w`, and the density
    is scaled so the tallest bar is exactly `h`. The fitted component curves
    ride the same scaling, so curve and bars are directly comparable.

    `fill` names what the silhouette is filled with, and across the
    progression the three fills *are* the argument:

    * `"plain"` — hollow bars in outline: the shape of the data, which is all
      anyone actually has before a fit is claimed.
    * `"query"` — the same silhouette re-filled, split at the components'
      crossing and hatched with question marks in rust or green. The mixture
      has read no labels, so "this hump is the Bad one" is a guess and the
      texture says so (the label-free figure, `calib-gmm-flow`).
    * `"class"` — the same two humps hatched the way `_data_block` hatches
      Good and Bad media. Votes inside the humps have pinned the components
      to actual classes, so the fill is the block's own texture rather than a
      question mark (the fold-anchored figure, `calib-fold-anchored-flow`).

    `mu_labels` writes the component means under the baseline. Off where two
    panels share a row and the μ names would crowd the cuts ticked beside
    them; the dashed stems stay either way, because the cut is drawn midway
    between them and needs them visible to read as a midpoint.
    """
    if fill not in ("plain", "query", "class"):
        raise ValueError(f"unknown histogram fill {fill!r}; expected 'plain', 'query' or 'class'")
    density, edges = np.histogram(scores, bins=GMM_FLOW_BINS, range=(0.0, 1.0), density=True)
    sy = h / float(density.max())

    if fill == "plain":
        # Bars in outline rather than a solid black mass. The silhouette is the
        # same polygon the fitted forms re-draw, so nothing moves between build
        # stages; what makes it read as *bars* is one separator per internal bin
        # boundary. Each separator stops at the shorter of the two bars it
        # divides, because the staircase's own riser already carries the rest of
        # that boundary — drawn full height they would double-stroke every riser
        # in the figure and the outline would thicken wherever the data steps.
        bars = _staircase(x0, y_base, w, sy, edges, density, 0, len(density) - 1)
        bars.set(facecolor="white", edgecolor=INK, linewidth=BAR_EDGE_LW, zorder=2)
        ax.add_patch(bars)
        for i in range(1, len(density)):
            top = y_base + float(min(density[i - 1], density[i])) * sy
            if top > y_base:
                ax.plot([x0 + edges[i] * w] * 2, [y_base, top], color=INK, linewidth=BAR_EDGE_LW, zorder=2)
        ax.plot([x0, x0 + w], [y_base] * 2, color=INK, linewidth=1.8, zorder=5)
        return

    # Where the fit stops calling a score Bad and starts calling it Good. The
    # two humps are split here, so between them the fill changes colour exactly
    # once and at the place the mixture itself puts the boundary.
    xs = np.linspace(0.0, 1.0, 600)
    lo_d = fit.w_lo * gaussian(xs, fit.mu_lo, fit.var_lo)
    hi_d = fit.w_hi * gaussian(xs, fit.mu_hi, fit.var_hi)
    crossing = int(np.argmax(hi_d > lo_d)) if (hi_d > lo_d).any() else len(xs)

    for lo_i, hi_i, mu, var, weight, color, hatch, name in (
        (0, crossing, fit.mu_lo, fit.var_lo, fit.w_lo, RUST, "\\\\\\", r"\mu_{lo}"),
        (crossing, len(xs), fit.mu_hi, fit.var_hi, fit.w_hi, GREEN, "//////", r"\mu_{hi}"),
    ):
        curve = y_base + weight * gaussian(xs, mu, var) * sy
        seg_x, seg_y = xs[lo_i:hi_i], curve[lo_i:hi_i]
        if seg_x.size:
            # The hatch is clipped to the area under this component's own
            # curve, not to the histogram's silhouette. The silhouette is the
            # taller of the two wherever the data outruns the fit, so clipping
            # to it let question marks stand above the very line that is
            # supposed to bound them.
            pts = [(x0 + seg_x[0] * w, y_base)]
            pts += [(x0 + xx * w, yy) for xx, yy in zip(seg_x, seg_y)]
            pts.append((x0 + seg_x[-1] * w, y_base))
            hump = Polygon(pts, closed=True)
            if fill == "query":
                hump.set(facecolor="white", edgecolor="none", zorder=2)
                ax.add_patch(hump)
                _question_hatch(ax, hump, color, z=2.5)
            else:
                # `_data_block`'s own hatches, so a hump reads as the same
                # stuff as the half of D₀ that named it.
                hump.set(facecolor="white", edgecolor=color, hatch=hatch, linewidth=0, zorder=2)
                ax.add_patch(hump)
        # Each Gaussian is drawn only where it is visibly off the baseline.
        # Plotted over the full axis, a component's far tail lies flat along
        # the bottom of the *other* hump, and a green line running under the
        # rust distribution reads as a stray mark rather than as the tail of
        # something that is genuinely still there.
        visible = np.flatnonzero(curve > y_base + TAIL_FLOOR * h)
        if visible.size:
            lo_v, hi_v = visible[0], visible[-1] + 1
            ax.plot(x0 + xs[lo_v:hi_v] * w, curve[lo_v:hi_v], color=color, linewidth=2.4, zorder=3)
        # Black, not the component's colour: against a rust or green hatch a
        # matching dashed line stops reading as a separate mark, and this one
        # has a job of its own — θ_G is ticked midway between the two.
        peak = y_base + weight * gaussian(np.array([mu]), mu, var)[0] * sy
        ax.plot([x0 + mu * w] * 2, [y_base, peak], color=INK, linewidth=1.6, linestyle=(0, (2, 2)), zorder=4)
        if mu_labels:
            ax.text(x0 + mu * w, y_base - LABEL_GAP, _sub(name), ha="center", va="top", fontsize=15, color=INK)

    ax.plot([x0, x0 + w], [y_base] * 2, color=INK, linewidth=1.8, zorder=5)


def _gmm_flow_stage(stage: int, fit: "GmmFit1D", scores: np.ndarray) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic.

    The loop runs anticlockwise — haystack down to votes, votes right into the
    model, model back down across the figure into the histogram — so the two
    things the mixture cut trades off sit on opposite sides of it: the sliver
    of labelled data on the upper left, the whole scored haystack along the
    bottom. `D₋₁ → M₀` is a plain stroke and `M₀ → M₀(D₋₁)` is the labelled
    block arrow, matching `_xcal_flow_stage`'s rule that a scoring path enters
    a model bare and leaves it labelled.
    """
    fig, ax = plt.subplots(figsize=tuple(c * FLOW_UNIT_PT / 72 for c in GMM_CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, GMM_CANVAS[0])
    ax.set_ylim(0, GMM_CANVAS[1])
    ax.set_axis_off()

    # ── layout ────────────────────────────────────────────────────────────────
    # Derived top-down from the haystack, then bottom-up from the closing line,
    # so the histogram takes whatever vertical room is actually left over.
    # `block_h` matches `_xcal_flow_stage`'s block, so D₀ is literally the same
    # object across the two figures — and so the disc naming it clears the
    # hatching by the same margin there as here.
    bx, block_w, block_h = 4.2, 4.0, 1.05
    # The haystack is the same *height* as the votes' block and much wider:
    # the two are the same kind of thing — media — so the shape that differs
    # between them should be how many there are, not how tall the box is.
    # Both blocks are named on discs *inside* themselves and share a left edge,
    # so the only difference the eye has to read between them is width — and
    # neither spends vertical budget on a label hung above it.
    hay_x0, hay_top = bx - block_w / 2, 10.9
    hay_w, hay_h = 10.1, block_h
    hay_y0 = hay_top - hay_h
    vote_len = 1.45
    block_top = hay_y0 - OBJECT_GAP - vote_len - OBJECT_GAP
    block_y0 = block_top - block_h
    row_y = block_y0 + block_h / 2

    train_len = 1.75
    train_x = bx + block_w / 2 + OBJECT_GAP
    m0x = train_x + train_len + OBJECT_GAP + MODEL_W / 2

    # One closing line, not two. `_xcal_flow_stage` can afford `θ₀ = avg(θ₁, θ₂)`
    # above its `return` because averaging two cuts is arithmetic the picture
    # cannot show; the midpoint *is* shown here — θ_G is ticked exactly between
    # two named means — so spelling it out again would cost the histogram a
    # third of its height to restate what the reader can already see.
    return_y = 0.32
    theta_label_top = return_y + 0.23 + OBJECT_GAP + CAP_16
    y_base = theta_label_top + LABEL_GAP + 0.32

    # D₋₁ → M₀ → M₀(D₋₁) is one straight vertical drop, not a dogleg: the
    # haystack falls into the model and the scores fall out of it on the same
    # line, which is the whole path the slide is about. The price is paid in
    # the panel's *position* rather than in the drawing — the histogram slides
    # right to sit under that line, so the arrow lands between θ_G and μ_hi
    # instead of over the middle of the distribution. Pointing at a group's
    # centre is worth less than the path reading as one stroke, and a vertical
    # arrow is also the cheapest possible use of the drop it costs: every unit
    # of height becomes arrow length, none of it spent going sideways.
    panel_x0, panel_w, panel_h = 3.8, 8.0, 1.9
    tip = (m0x, y_base + panel_h + OBJECT_GAP + CAP_16 + LABEL_GAP)

    labeled_arrow = functools.partial(_labeled_arrow, ax)

    # ── stage 1: the unlabeled haystack ───────────────────────────────────────
    # "Unlabeled" sits where D₀'s Good/Bad sit, capitalised to match them,
    # because it answers the same question about the same slot: what the
    # classes in this block are.
    _haystack_block(ax, hay_x0, hay_y0, hay_w, hay_h)
    _disc_label(ax, hay_x0 + hay_w / 2, hay_y0 + hay_h / 2, "D_{-1}")
    ax.text(
        hay_x0 - LABEL_GAP,
        hay_y0 + hay_h / 2,
        "Unlabeled",
        ha="right",
        va="center",
        fontsize=15,
        color=SOFT,
    )

    # ── stage 2: vote a sliver of it into D0, and train M0 on that ────────────
    # Lifted from `_xcal_flow_stage`'s stage 1, with the votes' provenance
    # added above it: this is the same D₀ and the same M₀ the last figure kept.
    if stage >= 2:
        labeled_arrow((bx, hay_y0 - OBJECT_GAP), (bx, hay_y0 - OBJECT_GAP - vote_len), "vote")
        _data_block(ax, bx - block_w / 2, block_y0, block_w, block_h)
        good_h = 0.42 * block_h
        ax.text(
            bx - block_w / 2 - LABEL_GAP,
            block_top - good_h / 2,
            "Good",
            ha="right",
            va="center",
            fontsize=15,
            color=GREEN,
        )
        ax.text(
            bx - block_w / 2 - LABEL_GAP,
            block_y0 + (block_h - good_h) / 2,
            "Bad",
            ha="right",
            va="center",
            fontsize=15,
            color=RUST,
        )
        _disc_label(ax, bx, row_y, "D_0")
        labeled_arrow((train_x, row_y), (train_x + train_len, row_y), "train")
        _model_box(ax, m0x, row_y, "M_0")

    # ── stage 3: M0 scores the whole haystack ─────────────────────────────────
    # The haystack drops into M₀ as a bare stroke and leaves it as the labelled
    # "score" arrow, so the eye reads one path with the model on it.
    if stage >= 3:
        _arrow(ax, (m0x, hay_y0 - OBJECT_GAP), _box_edge(m0x, row_y, (m0x, hay_y0), OBJECT_GAP))
        labeled_arrow(_box_edge(m0x, row_y, tip, OBJECT_GAP), tip, "score", z=2.1)
        # Left-aligned, unlike the centred labels on `_score_line`: the cut is
        # carried up through the histogram and lands within a hair of the panel's
        # own midpoint, so a centred label would sit on top of it.
        ax.text(
            panel_x0,
            y_base + panel_h + LABEL_GAP,
            _sub("M_0(D_{-1})"),
            ha="left",
            va="bottom",
            fontsize=16,
            color=INK,
        )
        _score_histogram(ax, panel_x0, y_base, panel_w, panel_h, fit, scores, fill="query" if stage >= 4 else "plain")

    # ── stage 5: cut at the midpoint between the two claimed modes ────────────
    if stage >= 5:
        mid = 0.5 * (fit.mu_lo + fit.mu_hi)
        theta_x = panel_x0 + mid * panel_w
        # A notch under the baseline, exactly as `_score_line` ticks a cut.
        # Carrying the cut up through the histogram would say more about this
        # one — its whole claim is that it divides the distribution — but the
        # next two figures put this very panel beside score lines whose cuts
        # are notched, and a cut that changes its mark between figures reads
        # as a different kind of thing rather than as the same θ. One mark,
        # one meaning; the two named means either side of it are what make it
        # a midpoint, and they are drawn.
        ax.plot([theta_x] * 2, [y_base - 0.32, y_base], color=INK, linewidth=2.2, zorder=6)
        ax.text(theta_x, y_base - 0.32 - LABEL_GAP, _sub(r"\theta_G"), ha="center", va="top", fontsize=16, color=INK)
        ax.text(
            panel_x0 + panel_w / 2,
            return_y,
            "return " + _sub(r"(M_0,\, \theta_G)"),
            ha="center",
            va="center",
            fontsize=18,
            color=INK,
        )

    return fig


#: How many build stages the blend schematic reveals in: the spine the two
#: rival estimators share; the mixture branch; its label-free cut θ_G; the fold
#: models and their crossed scoring paths; the held-out cuts averaged into θ_X;
#: and the weighted average that settles between the two.
BLEND_FLOW_STAGES = 6

#: Wider than the two figures it assembles (13.2 each) because it holds both of
#: them, and — uniquely in the progression — 0.4 units taller than
#: `FLOW_CANVAS_H`, which is a deliberate exception to the rule stated there.
#: What that rule is really protecting is the *rendered* label size, and the
#: trade here was made by measuring it: at this height, cropped, in a
#: `bg right:70%` slot, a 15pt label renders at 23.8px against the two
#: parents' 24.7px. The 0.4 units buys back the parents' own block heights
#: (1.05) and their 15pt arrow labels — this figure stacks more arrows than
#: either of them, and a block arrow has to be longer than the word written
#: along it. Spending 4% of size to avoid shrinking the type and cramping
#: every arrow is the better half of that trade; spending much more would not
#: be, and the fix past this point is to cut content rather than add canvas.
BLEND_CANVAS = (13.6, 11.4)

#: The blend schematic's score lines, shorter than the cross-calibration
#: figure's 2.6 because the mixture panel shares their row.
BLEND_LINE_HALF = 1.45

#: The conclusion row's type.
CONCLUSION_PT = 17.0

#: How much of the figure's bottom-right corner to leave empty, in drawing
#: units. A `bg` figure is laid *under* the slide, and the theme puts the page
#: number at `right: 70px; bottom: 26px` — inside the figure's own bottom-right
#: corner once it is fitted into a 70% slot. The two parent figures are clear
#: of it by luck (their conclusion lines are centred over the middle of the
#: canvas); this one ends with a line that would otherwise run right through
#: it, so the clearance is measured rather than eyeballed: fitted, this figure
#: renders at ~60px per drawing unit, and the badge's ink starts ~73px in from
#: the right edge and ~50px up from the bottom.
PAGE_NUMBER_CLEAR = 1.35


def blend_flow_fig() -> None:
    """Schematic of the blend — iteration 3 of the line (issue #3218).

    The third figure of the calibration progression, and the first that is an
    *assembly* rather than a new mechanism: its left half is `calib-xcal-flow`
    and its right half is `calib-gmm-flow`, sharing the spine both of those
    figures already drew — the haystack D₋₁, the votes D₀ taken out of it, and
    the model M₀ trained on those votes.

    From that shared spine the two estimators go their separate ways and are
    drawn going them: the cross-calibration flow **down** (split the votes,
    train a fold model per half, score the half it never saw, cut, average)
    ending at θ_X, and the mixture flow **right, then down** (M₀ scores the
    whole haystack, fit two components, cut at their midpoint) ending at θ_G.
    Neither is the answer; the answer is the weighted average of the two, which
    is what shipped as safe thresholds (#2798/#2799).

    Three geometric choices carry the argument:

    * **The three evidence displays share one baseline.** M₁'s held-out score
      line, M₂'s, and the haystack histogram all sit on the same rule, so the
      three cuts θ₁, θ₂ and θ_G are ticked at the same height, with the same
      mark, and read as three answers to one question rather than as two
      unrelated pictures.
    * **The rivals are the same size.** The mixture panel gets the width the
      two score lines get, because the slide's claim is that neither estimator
      dominates — one is starved, the other is biased.
    * **Each conclusion sits under the branch that produced it**, and the
      arrow between them is the whole iteration.

    What the figure deliberately does *not* say is what w is. `avg_w` names a
    weighted average and stops there: the weight's shape over the vote count —
    the hand-tuned ramp, and iteration 3½'s finding that the shipped curves
    never hand over completely — is the next slide's business, and a ramp drawn
    here would spend this slide's ink getting ahead of it.
    """
    fit, scores = _haystack_scores()
    final = _blend_flow_stage(BLEND_FLOW_STAGES, fit, scores)
    box = tight_box(final)
    for stage in range(1, BLEND_FLOW_STAGES):
        save(
            _blend_flow_stage(stage, fit, scores),
            OUT,
            f"calib-blend-flow.build{stage}.png",
            column=SIDEBAR_WIDE,
            box=box,
        )
    save(final, OUT, "calib-blend-flow.png", column=SIDEBAR_WIDE, box=box)


def _blend_flow_stage(stage: int, fit: "GmmFit1D", scores: np.ndarray) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic."""
    fig, ax = plt.subplots(figsize=tuple(c * FLOW_UNIT_PT / 72 for c in BLEND_CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, BLEND_CANVAS[0])
    ax.set_ylim(0, BLEND_CANVAS[1])
    ax.set_axis_off()

    # ── layout ────────────────────────────────────────────────────────────────
    # Top-down for the spine, then the fold branch (which fixes how much width
    # is left), then the mixture branch into what remains, then the conclusion.
    bx, block_w, block_h = 4.4, 4.0, 1.05
    block_x0 = bx - block_w / 2
    hay_x0, hay_top, hay_h = block_x0, BLEND_CANVAS[1], block_h
    hay_w = BLEND_CANVAS[0] - 0.2 - hay_x0
    hay_y0 = hay_top - hay_h
    vote_len = 1.15
    block_top = hay_y0 - OBJECT_GAP - vote_len - OBJECT_GAP
    block_y0 = block_top - block_h
    row_y = block_y0 + block_h / 2

    # ── the cross-calibration branch, straight down ───────────────────────────
    fold_train_len, score_len, slope = 1.15, 1.3, 0.75
    m1x, m2x = bx - block_w / 4, bx + block_w / 4
    train_tail_y = block_y0 - OBJECT_GAP
    my = train_tail_y - fold_train_len - OBJECT_GAP - MODEL_H / 2

    down_left = (m1x - slope, my - 1.0)
    exit1 = _box_edge(m1x, my, down_left, OBJECT_GAP)
    ux, uy = np.array([slope, 1.0]) / float(np.hypot(slope, 1.0))
    tip1 = (exit1[0] - score_len * ux, exit1[1] - score_len * uy)
    line_y, line_half = tip1[1] - (OBJECT_GAP + CAP_16 + SCORE_LABEL_LIFT), BLEND_LINE_HALF
    cx1 = tip1[0] - 0.4 * line_half
    cx2 = 2 * bx - cx1

    # ── the mixture branch, into the width the fold branch leaves ─────────────
    # Its baseline is the score lines' baseline: the two fold lines and the
    # haystack histogram are three readings of the same quantity, so they are
    # drawn on one rule and their three cuts are ticked at one height.
    panel_x0 = cx2 + line_half + OBJECT_GAP
    panel_w = BLEND_CANVAS[0] - 0.2 - panel_x0
    panel_h = 2.0
    y_base = line_y
    # Where the mixture puts its cut: midway between the two component means,
    # which is `calculate_gmm_threshold`'s rule and the previous figure's θ_G.
    theta_g_x = panel_x0 + 0.5 * (fit.mu_lo + fit.mu_hi) * panel_w

    # D₋₁ → M₀ → M₀(D₋₁) is one straight vertical drop, exactly as the mixture
    # figure draws it. That figure could put M₀ where the train arrow happened
    # to leave it and slide the *panel* under the drop; here the panel's
    # position is already spoken for — it takes the width the score lines
    # leave — so it is M₀ that moves, out to the point of the histogram the
    # drop should land on: past θ_G, short of μ_hi, off the tall bars. The
    # train arrow reaching it is however long that takes.
    m0x = panel_x0 + 0.62 * panel_w
    train_x = bx + block_w / 2 + OBJECT_GAP
    train_len = m0x - MODEL_W / 2 - OBJECT_GAP - train_x
    tip0 = (m0x, y_base + panel_h + OBJECT_GAP + CAP_16 + LABEL_GAP)

    # ── the conclusion ────────────────────────────────────────────────────────
    # One row, each half under the branch that produced it: what the folds
    # settled on, and what the two rivals settle on together.
    theta_bottom = line_y - 0.32 - LABEL_GAP - CAP_16
    conclusion_y = theta_bottom - OBJECT_GAP - CONCLUSION_PT / 72
    eq_x = panel_x0 + panel_w / 2

    data_block = functools.partial(_data_block, ax)
    arrow = functools.partial(_arrow, ax)
    labeled_arrow = functools.partial(_labeled_arrow, ax)
    model_box = functools.partial(_model_box, ax)
    score_line = functools.partial(_score_line, ax, y=line_y, half=line_half)

    # ── stage 1: the spine both estimators are built on ───────────────────────
    # Carried over wholesale from the mixture figure: the unlabeled haystack,
    # the sliver of it that got voted, and the model trained on those votes.
    _haystack_block(ax, hay_x0, hay_y0, hay_w, hay_h)
    _disc_label(ax, hay_x0 + hay_w / 2, hay_y0 + hay_h / 2, "D_{-1}")
    ax.text(hay_x0 - LABEL_GAP, hay_y0 + hay_h / 2, "Unlabeled", ha="right", va="center", fontsize=15, color=SOFT)
    labeled_arrow((bx, hay_y0 - OBJECT_GAP), (bx, hay_y0 - OBJECT_GAP - vote_len), "vote")
    data_block(block_x0, block_y0, block_w, block_h, split=stage >= 4)
    good_h = 0.42 * block_h
    ax.text(block_x0 - LABEL_GAP, block_top - good_h / 2, "Good", ha="right", va="center", fontsize=15, color=GREEN)
    ax.text(
        block_x0 - LABEL_GAP,
        block_y0 + (block_h - good_h) / 2,
        "Bad",
        ha="right",
        va="center",
        fontsize=15,
        color=RUST,
    )
    # D₀ is named from *above*, not on a disc inside itself as the mixture
    # figure names it: the inside of this block is about to be divided into D₁
    # and D₂, which is where those discs go. Left-aligned rather than centred
    # because the centre of the space above the block is where the vote arrow
    # lands.
    ax.text(block_x0, block_top + LABEL_GAP, _sub("D_0"), ha="left", va="bottom", fontsize=16, color=INK)
    labeled_arrow((train_x, row_y), (train_x + train_len, row_y), "train")
    model_box(m0x, row_y, "M_0")

    # ── stage 2: M0 scores the whole haystack ─────────────────────────────────
    if stage >= 2:
        arrow((m0x, hay_y0 - OBJECT_GAP), _box_edge(m0x, row_y, (m0x, hay_y0), OBJECT_GAP))
        labeled_arrow(_box_edge(m0x, row_y, tip0, OBJECT_GAP), tip0, "score", z=2.1)
        ax.text(
            panel_x0,
            y_base + panel_h + LABEL_GAP,
            _sub("M_0(D_{-1})"),
            ha="left",
            va="bottom",
            fontsize=16,
            color=INK,
        )
        _score_histogram(ax, panel_x0, y_base, panel_w, panel_h, fit, scores, fill="query")

    # ── stage 3: cut it where the two fitted components meet ──────────────────
    if stage >= 3:
        # A notch under the baseline, exactly as `_score_line` ticks a cut: the
        # cuts in this progression are one mark with one meaning, so they are
        # drawn one way whether the evidence above the line is a row of votes
        # or a fitted mixture.
        ax.plot([theta_g_x] * 2, [y_base - 0.32, y_base], color=INK, linewidth=2.2, zorder=6)
        ax.text(theta_g_x, y_base - 0.32 - LABEL_GAP, _sub(r"\theta_G"), ha="center", va="top", fontsize=16, color=INK)

    # ── stage 4: split the votes and train a fold model on each half ──────────
    if stage >= 4:
        for mx, name, sign in ((m1x, "M_1", 1.0), (m2x, "M_2", -1.0)):
            model_box(mx, my, name)
            labeled_arrow((mx, train_tail_y), (mx, train_tail_y - fold_train_len), "train")
            entry_tail = (mx + sign * slope * (train_tail_y - my), train_tail_y)
            arrow(entry_tail, _box_edge(mx, my, entry_tail, OBJECT_GAP))
        labeled_arrow(exit1, tip1, "score", z=2.1)
        exit2 = (2 * bx - exit1[0], exit1[1])
        labeled_arrow(exit2, (2 * bx - tip1[0], tip1[1]), "score", z=2.1)
        score_line(cx1, label="M_1(D_2)", **_line_marks(SCORE_MARKS_1, line_half), theta=r"\theta_1", cut=stage >= 5)
        score_line(cx2, label="M_2(D_1)", **_line_marks(SCORE_MARKS_2, line_half), theta=r"\theta_2", cut=stage >= 5)

    # ── stage 5: cut each fold's held-out scores, and average the two ─────────
    if stage >= 5:
        theta_x_text = ax.text(
            bx,
            conclusion_y,
            _sub(r"\theta_X = avg(\theta_1,\, \theta_2)"),
            ha="center",
            va="center",
            fontsize=16.5,
            color=INK,
        )

    # ── stage 6: settle between the rivals ────────────────────────────────────
    # `avg_w` and no more: this slide's claim is that the answer is between the
    # two rivals, not where between them.
    if stage >= 6:
        equation = ax.text(
            eq_x,
            conclusion_y,
            _sub(r"\theta_0 = avg_w(\theta_X,\, \theta_G)"),
            ha="center",
            va="center",
            fontsize=CONCLUSION_PT,
            color=INK,
        )
        # Everything below is measured off the two texts' own ink, because the
        # rendered width of a mathtext run is not something to guess at: the
        # equation is pulled left if it would reach into the page number's
        # corner, and the arrow between the two is then fitted to what is left.
        fig.canvas.draw()
        to_units = ax.transData.inverted()
        eq_box = equation.get_window_extent().transformed(to_units)
        overhang = eq_box.x1 - (BLEND_CANVAS[0] - 0.2 - PAGE_NUMBER_CLEAR)
        if overhang > 0:
            equation.set_x(eq_x - overhang)
            fig.canvas.draw()
            eq_box = equation.get_window_extent().transformed(to_units)
        xcal_box = theta_x_text.get_window_extent().transformed(to_units)
        arrow((xcal_box.x1 + OBJECT_GAP, conclusion_y), (eq_box.x0 - OBJECT_GAP, conclusion_y))

    return fig


#: How many build stages the fold-anchored schematic reveals in: the spine
#: carried over from the blend; the split and the two fold models; each fold
#: model scoring the whole haystack into a shape; the held-out votes arriving
#: to name the two components; the per-fold cuts; the average.
XSEMI_FLOW_STAGES = 6

#: Taller than any of its parents, and the one figure in the progression that
#: could not be talked down to `FLOW_CANVAS_H`. The blend fits in 11.4 because
#: its mixture panel sits *beside* the fold branch, sharing that branch's
#: vertical budget; here both evidence displays are mixture panels and both
#: hang *below* the fold models, so the panel's height is spent on top of the
#: whole spine rather than alongside it. Measured rather than eyeballed, on
#: the same rule the blend's exception was measured by: fitted into a
#: `bg right:70%` slot a 15pt label renders at 22.6px here, against the
#: blend's 23.8px and the theme's 20px floor. The alternatives were cutting
#: the panels to a size where the votes inside the humps stop being legible,
#: or dropping a beat the figure exists to make.
XSEMI_CANVAS = (13.6, 12.75)

#: The fold panels' height. Shorter than the blend's 2.0 because there are two
#: of them stacked under the flow rather than one beside it, and no shorter,
#: because the Good hump has to stay tall enough to hold a vote *inside* it —
#: at a fifth of the haystack it draws at about a quarter of the Bad hump's
#: height, which is the real floor on this number.
XSEMI_PANEL_H = 1.75

#: Each fold model's haystack scores, as `((mu, sigma), (mu, sigma))` for the
#: Bad and Good populations. Fold 1 is the mixture figure's own haystack,
#: unchanged, so the audience recognises the picture; fold 2 is drawn from a
#: model that scores the same media differently — its Bads sit higher and its
#: Goods sit higher still. That the two folds disagree is the point of
#: averaging their cuts at all, and the quantile figure that follows is where
#: the disagreement stops being survivable by an average of raw scores.
XSEMI_POPULATIONS = (
    ((0.26, 0.082), (0.74, 0.078)),
    ((0.36, 0.095), (0.80, 0.075)),
)

#: Each fold's *held-out* votes, as scores on that fold's own scale — the
#: cross-calibration figure's ✗s and ✓s, moved onto a score axis. The counts
#: and the imperfection are carried over from `SCORE_MARKS_2`: fold 2 ranks a
#: Bad above its own cut, because a fold model's ranking of votes it never
#: saw is not trivially clean, and an anchored fit has to survive that. No
#: Good sits at its fold's μ_hi, because a ✓ drawn above the baseline lands on
#: the dashed stem that marks the mean and neither mark survives the overlap.
XSEMI_ANCHORS = (
    {"bad": (0.12, 0.20, 0.28, 0.36), "good": (0.63, 0.71, 0.85)},
    {"bad": (0.22, 0.30, 0.38, 0.70), "good": (0.61, 0.74, 0.91)},
)


def xsemi_flow_fig() -> None:
    """Schematic of the fold-anchored mixture — iteration 4 of the line (#3218).

    The fourth figure of the calibration progression, and the one where the
    two rivals of `calib-blend-flow` stop being rivals: instead of averaging a
    label-free cut against a label-only cut, each *fold* model fits the whole
    haystack **and** its own held-out votes at once, so one estimator reads
    both sources of evidence. This is the shipped threshold path
    (`fold_anchored_gmm_threshold`), drawn at its shipped settings: anchors at
    mass κ = `FOLD_ANCHOR_WEIGHT`, the cut at the midpoint of the two fitted
    means, one fit per fold.

    What survives from the previous figures, deliberately: the haystack D₋₁
    and the votes D₀ taken out of it; D₀ —train→ M₀; the split into D₁/D₂ and
    the crossed scoring paths that name cross-calibration. What is gone is the
    horizontal rival branch — M₀ no longer scores the haystack here, because
    the mixture it used to fit has moved inside the folds. What is new is the
    pair of bare strokes running down the outside: *each fold model* scores
    the whole haystack, which is the fit's population.

    Three things the geometry is doing:

    * **The evidence displays upgrade in place.** Where the cross-calibration
      figure put a score line with ✗s and ✓s on it, this one puts the same
      marks inside a mixture panel. The votes have not changed and neither has
      their arrangement; what has changed is that a whole scored haystack is
      now drawn behind them.
    * **The humps take the block's own hatching.** The label-free figure
      filled them with question marks because the low/high = Bad/Good reading
      was an assumption. Here the votes are *in* the humps, so the fill is the
      same Good/Bad texture D₀ is drawn with — the identification is shown
      being earned rather than assumed.
    * **The two cuts are ticked on one baseline**, as the blend's three were,
      and they visibly disagree. Averaging them is the closing line, and it is
      a *strawman*: the quantile figure that follows shows what a cardinal
      average of two models' raw scores actually does. That is also why this
      figure does not close on a `return`, as its parents do — it ends on the
      thing the next figure corrects.

    One honesty note for the speaker, recorded here because the figure cannot
    say it: production never shipped this cardinal average. Quantile transfer
    was in the fold-anchored design from the start; the average is drawn plain
    here because the mistake teaches the fix, not because it is history.
    """
    folds = _xsemi_folds()
    final = _xsemi_flow_stage(XSEMI_FLOW_STAGES, folds)
    box = tight_box(final)
    for stage in range(1, XSEMI_FLOW_STAGES):
        save(
            _xsemi_flow_stage(stage, folds),
            OUT,
            f"calib-fold-anchored-flow.build{stage}.png",
            column=SIDEBAR_WIDE,
            box=box,
        )
    save(final, OUT, "calib-fold-anchored-flow.png", column=SIDEBAR_WIDE, box=box)


def _xsemi_folds() -> list[tuple[GmmFit1D, np.ndarray, dict]]:
    """Per fold: the **real** anchored fit, its haystack sample, and its votes.

    Schematic input, real estimator — the house rule, and here it runs the
    shipped entry point (`fit_anchored_score_gmm` at `FOLD_ANCHOR_WEIGHT`) on
    each fold's own population and each fold's own held-out anchors, which is
    exactly what `fit_fold_anchored_cut` does per fold. At κ = 0.3 a handful
    of votes barely moves a fit this well separated, and that is the honest
    picture: what the anchors buy here is *identity*, not displacement. The
    figure that shows anchors moving a fit is `calib-anchored-em`.
    """
    rng = np.random.default_rng(4)
    folds = []
    for (neg, pos), anchors in zip(XSEMI_POPULATIONS, XSEMI_ANCHORS, strict=True):
        scores = np.clip(np.concatenate([rng.normal(neg[0], neg[1], 4800), rng.normal(pos[0], pos[1], 1200)]), 0.0, 1.0)
        a_scores = np.array(anchors["bad"] + anchors["good"], dtype=float)
        a_labels = np.concatenate([np.zeros(len(anchors["bad"])), np.ones(len(anchors["good"]))])
        fit, provenance = fit_anchored_score_gmm(scores, a_scores, a_labels, anchor_weight=FOLD_ANCHOR_WEIGHT)
        assert fit is not None and provenance == "anchored"
        folds.append((fit, scores, anchors))
    return folds


def _hump_marks(ax: plt.Axes, x0: float, y_base: float, w: float, anchors: dict) -> None:
    """This fold's held-out votes, drawn on the panel's baseline.

    The panel's baseline *is* the cross-calibration figure's score line: the
    same ✗s below it and ✓s above it, at the same offsets and the same weight,
    because this is the identical evidence and the reader has already learnt
    to read it that way. All that has changed is that the fold model's whole
    scored haystack is now drawn standing on the same line, so the votes and
    the population they anchor are read against one axis.

    The one thing the panel adds is that a vote's position now means a score
    rather than a rank, which is what lets a vote the fold model ranked wrongly
    land visibly to the far side of its own fold's cut.

    The glyphs are knocked out of whatever they land on with a white outline.
    On the cross-calibration figure they sat on white paper; here a ✓ sits
    just above the baseline where the Good hump's green hatch starts, and
    green on green is not a mark. The halo is the smallest change that keeps
    the glyph itself identical.
    """
    halo = [patheffects.withStroke(linewidth=4.0, foreground="white")]
    for key, color, glyph, dy, va in (("bad", RUST, "✗", -0.12, "top"), ("good", GREEN, "✓", 0.08, "bottom")):
        for score in anchors[key]:
            ax.text(
                x0 + score * w,
                y_base + dy,
                glyph,
                ha="center",
                va=va,
                fontsize=16,
                color=color,
                fontweight="bold",
                zorder=6,
                path_effects=halo,
            )


def _mark_legend(ax: plt.Axes, x_outer: float, y: float, name: str, *, mirrored: bool) -> None:
    """Name whose scores the ✗s and ✓s in a panel are, with the glyphs themselves.

    The panel's own label says what the *humps* are — `Mᵢ(D₋₁)`, the whole
    scored haystack. The marks in them are the other half of the fit and are a
    different quantity, so they are named too, in the same two glyphs the
    cross-calibration figure put on a score line.

    Set as a second line above the panel's own label, at the panel's outer
    top corner: the two labels name the two things in the panel, so they read
    as one group, and the corner is the only place in this figure where a
    two-line group fits. Inside the panel it would sit on a hump — the two
    panels are mirror images in every quadrant *except* their humps, which
    both lean left — and hung at the inner corners the two legends meet in
    the middle of the figure and read as one four-glyph cluster belonging to
    neither panel.
    """
    step, gap = 0.30, 0.30
    d = -1.0 if mirrored else 1.0
    pair = (("✓", GREEN), ("✗", RUST)) if mirrored else (("✗", RUST), ("✓", GREEN))
    for k, (glyph, color) in enumerate(pair):
        ax.text(
            x_outer + d * (0.15 + step * k),
            y,
            glyph,
            ha="center",
            va="baseline",
            fontsize=16,
            color=color,
            fontweight="bold",
        )
    ax.text(
        x_outer + d * (0.15 + step + gap),
        y,
        _sub(name),
        ha="right" if mirrored else "left",
        va="baseline",
        fontsize=15,
        color=INK,
    )


def _xsemi_flow_stage(stage: int, folds: list) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic."""
    fig, ax = plt.subplots(figsize=tuple(c * FLOW_UNIT_PT / 72 for c in XSEMI_CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, XSEMI_CANVAS[0])
    ax.set_ylim(0, XSEMI_CANVAS[1])
    ax.set_axis_off()

    # ── layout ────────────────────────────────────────────────────────────────
    # Top-down for the spine and the fold branch, exactly as the blend figure
    # derives them, then the panels from where the fold branch's score arrows
    # land, then the conclusion under both.
    canvas_w, canvas_h = XSEMI_CANVAS
    bx, block_w, block_h = canvas_w / 2, 4.0, 1.05
    block_x0 = bx - block_w / 2
    # Centred on the spine rather than run out to the canvas edge: the disc
    # naming the haystack sits at the block's own centre, so centring the
    # block is what puts that name directly above the vote arrow drawn out of
    # it. Wide enough to stay 2.4× the votes' block — "far more of them" is
    # the one thing this shape says — and no wider, since past the point the
    # ratio is legible the extra length only pulls D₋₁'s name off the spine.
    hay_w, hay_h = 2.4 * block_w, block_h
    hay_x0 = bx - hay_w / 2
    hay_y0 = canvas_h - hay_h
    vote_len = 1.15
    block_top = hay_y0 - OBJECT_GAP - vote_len - OBJECT_GAP
    block_y0 = block_top - block_h
    row_y = block_y0 + block_h / 2

    fold_train_len, score_len, slope = 1.15, 1.3, 0.75
    m1x, m2x = bx - block_w / 4, bx + block_w / 4
    train_tail_y = block_y0 - OBJECT_GAP
    my = train_tail_y - fold_train_len - OBJECT_GAP - MODEL_H / 2

    down_left = (m1x - slope, my - 1.0)
    exit1 = _box_edge(m1x, my, down_left, OBJECT_GAP)
    ux, uy = np.array([slope, 1.0]) / float(np.hypot(slope, 1.0))
    tip1 = (exit1[0] - score_len * ux, exit1[1] - score_len * uy)

    # The two panels are a mirror pair about the figure's centre line, and
    # their width is whatever makes the left one's arrow land where the blend
    # figure's did — 0.62 of the way across, past the cut and short of μ_hi,
    # off the tall bars. Solving that for the width is what keeps the pair
    # centred *and* the arrows honest.
    panel_h = XSEMI_PANEL_H
    panel_gap = 2 * OBJECT_GAP
    panel_w = (bx - panel_gap / 2 - tip1[0]) / 0.38
    panel_top = tip1[1] - (OBJECT_GAP + CAP_16 + LABEL_GAP)
    y_base = panel_top - panel_h
    panel_x = (bx - panel_gap / 2 - panel_w, bx + panel_gap / 2)

    # M₀ still gets trained on all of D₀ — it is what the threshold is *for* —
    # but nothing flows into it from the haystack any more: the mixture that
    # used to be fitted on M₀'s scores has moved inside the folds. That
    # collapse is the whole of what iteration 4 removes.
    train_x = block_x0 + block_w + OBJECT_GAP
    train_len = 1.5
    m0x = train_x + train_len + OBJECT_GAP + MODEL_W / 2

    theta_bottom = y_base - 0.32 - LABEL_GAP - CAP_16
    conclusion_y = theta_bottom - OBJECT_GAP - CONCLUSION_PT / 72

    data_block = functools.partial(_data_block, ax)
    arrow = functools.partial(_arrow, ax)
    labeled_arrow = functools.partial(_labeled_arrow, ax)
    model_box = functools.partial(_model_box, ax)

    # ── stage 1: the spine, carried over from the blend ───────────────────────
    _haystack_block(ax, hay_x0, hay_y0, hay_w, hay_h)
    _disc_label(ax, hay_x0 + hay_w / 2, hay_y0 + hay_h / 2, "D_{-1}")
    ax.text(hay_x0 - LABEL_GAP, hay_y0 + hay_h / 2, "Unlabeled", ha="right", va="center", fontsize=15, color=SOFT)
    labeled_arrow((bx, hay_y0 - OBJECT_GAP), (bx, hay_y0 - OBJECT_GAP - vote_len), "vote")
    data_block(block_x0, block_y0, block_w, block_h, split=stage >= 2)
    good_h = 0.42 * block_h
    ax.text(block_x0 - LABEL_GAP, block_top - good_h / 2, "Good", ha="right", va="center", fontsize=15, color=GREEN)
    ax.text(
        block_x0 - LABEL_GAP,
        block_y0 + (block_h - good_h) / 2,
        "Bad",
        ha="right",
        va="center",
        fontsize=15,
        color=RUST,
    )
    ax.text(block_x0, block_top + LABEL_GAP, _sub("D_0"), ha="left", va="bottom", fontsize=16, color=INK)
    labeled_arrow((train_x, row_y), (train_x + train_len, row_y), "train")
    model_box(m0x, row_y, "M_0")

    # ── stage 2: split the votes, train a fold model on each half ─────────────
    if stage >= 2:
        for mx, name in ((m1x, "M_1"), (m2x, "M_2")):
            model_box(mx, my, name)
            labeled_arrow((mx, train_tail_y), (mx, train_tail_y - fold_train_len), "train")

    # ── stage 3: each fold model scores the whole haystack ────────────────────
    # Where the haystack enters the fold models is left to the panel's own
    # label. Drawn, that path has to come down the outside of the figure and
    # in along the models' row — the space directly above each fold model is
    # spoken for by the half of D₀ that trains it — and two L-shaped strokes
    # that size buy a fact the audience already has from three figures of the
    # same haystack: `Mᵢ(D₋₁)` over a histogram says where it came from.
    for i, (fit, scores, _anchors) in enumerate(folds):
        if stage < 3:
            break
        sign = -1.0 if i == 0 else 1.0
        exit_i = (2 * bx - exit1[0], exit1[1]) if i else exit1
        tip_i = (2 * bx - tip1[0], tip1[1]) if i else tip1
        labeled_arrow(exit_i, tip_i, "score", z=2.1)
        ax.text(
            panel_x[i] + (panel_w if sign > 0 else 0.0),
            panel_top + LABEL_GAP,
            _sub(f"M_{i + 1}(D_{{-1}})"),
            ha="right" if sign > 0 else "left",
            va="bottom",
            fontsize=16,
            color=INK,
        )
        _score_histogram(
            ax,
            panel_x[i],
            y_base,
            panel_w,
            panel_h,
            fit,
            scores,
            fill="class" if stage >= 4 else "plain",
            mu_labels=False,
        )

    # ── stage 4: the held-out votes arrive and name the two components ────────
    if stage >= 4:
        for i, (mx, sign) in enumerate(((m1x, 1.0), (m2x, -1.0))):
            entry_tail = (mx + sign * slope * (train_tail_y - my), train_tail_y)
            arrow(entry_tail, _box_edge(mx, my, entry_tail, OBJECT_GAP))
            _hump_marks(ax, panel_x[i], y_base, panel_w, folds[i][2])
            # Held out, and named: the votes in fold i's panel are the ones
            # its own model never trained on — the crossed strokes above are
            # where they came from.
            _mark_legend(
                ax,
                panel_x[i] + (0.0 if i == 0 else panel_w),
                panel_top + LABEL_GAP + CAP_16 + LABEL_GAP,
                f"M_{i + 1}(D_{2 - i})",
                mirrored=i == 1,
            )
        # How much weight the votes carry — κ, and the share γ = κn/(κn+N)
        # they end up holding — is deliberately *not* annotated here. It is
        # the number that replaces the blend's hand-tuned ramp, which makes it
        # a claim to argue on a slide of its own rather than a quantity the
        # drawing needs: what this figure has to show is that the votes are in
        # the fit at all.

    # ── stage 5: cut each fold at the midpoint of its two fitted means ────────
    if stage >= 5:
        for i, (fit, _scores, _anchors) in enumerate(folds):
            theta_x = panel_x[i] + 0.5 * (fit.mu_lo + fit.mu_hi) * panel_w
            ax.plot([theta_x] * 2, [y_base - 0.32, y_base], color=INK, linewidth=2.2, zorder=6)
            ax.text(
                theta_x,
                y_base - 0.32 - LABEL_GAP,
                _sub(rf"\theta_{i + 1}"),
                ha="center",
                va="top",
                fontsize=16,
                color=INK,
            )

    # ── stage 6: average the two cuts ─────────────────────────────────────────
    # One line, and no `return (M₀, θ₀)` after it. Its parents close on a
    # return because they are each a whole algorithm; this one is a beat in the
    # middle of an argument, and the quantile figure that follows takes this
    # very average apart. Ending on the thing about to be corrected is the
    # point, and an arrow onward to a return would spend the slide's last
    # words settling something the next slide unsettles.
    if stage >= 6:
        ax.text(
            bx,
            conclusion_y,
            _sub(r"\theta_0 = avg(\theta_1,\, \theta_2)"),
            ha="center",
            va="center",
            fontsize=CONCLUSION_PT,
            color=INK,
        )

    return fig


#: How many build stages the quantile schematic reveals in: the spine; the
#: split and the two fold models; both fold panels fitted, voted and cut (one
#: advance, because every stroke of it is `calib-fold-anchored-flow`
#: recapitulated); M₀ scoring the haystack into a third, unfitted shape; the
#: cardinal average dropped onto that shape, where it lands in the Good mound;
#: the two cuts re-read as quantiles and averaged there; the mean quantile
#: realized on M₀'s own distribution, and the return.
XQUANT_FLOW_STAGES = 7

#: Wider than any of its parents and, at 13.4, marginally taller than the
#: fold-anchored figure it extends. The width is what carries the argument:
#: three panels of the **same** width on one baseline, so "0.58" is at the same
#: place along every axis and the eye can compare where a raw score lands in
#: three different models' distributions. Fitted into a `bg right:70%` slot a
#: 15pt label renders at 21.2px here, against the fold-anchored figure's 22.6px
#: and the theme's 20px floor; the figure is height-limited in that slot, so
#: the extra width is free and only the 0.65 units of extra height are spent.
XQUANT_CANVAS = (16.65, 13.0)

#: The three panels' shared height, and the gauge row's bar height. The panels
#: are `XSEMI_PANEL_H` unchanged — the fold half of this figure is the previous
#: figure's fold half, drawn at the same size so it is recognised rather than
#: re-read.
XQUANT_PANEL_H = XSEMI_PANEL_H
XQUANT_GAUGE_H = 0.30

#: The gauge's width as a fraction of its panel's. Narrower than the panel, and
#: left-aligned under it, for two reasons: a gauge is a *rank* axis and must not
#: invite being read as the score axis it sits under, and the figure's
#: bottom-right corner is where the theme prints the page number (see
#: `PAGE_NUMBER_CLEAR`), which a full-width bar ran straight through.
XQUANT_GAUGE_W = 0.58

#: What the two fold gauges are **labelled** with, and filled to.
#:
#: Printed as percentages, which is the point of them being percentages: a
#: share and a score are different kinds of quantity, and every other number on
#: this figure is a score. `81%` cannot be mistaken for a cut at 0.81.
#:
#: These two are the figure's one invented pair of numbers, and they are here
#: because the true ones teach badly. A midpoint cut on a well-separated mixture
#: at this prevalence always lands at very nearly the same quantile — that is
#: precisely *why* a quantile survives the crossing between two models and a raw
#: score does not — so the real fold quantiles for these two populations are
#: 0.796 and 0.798, which both print as `0.80`. A gauge row reading
#: `0.80`, `0.80`, `avg = 0.80` makes the combine step look like a no-op and
#: invites the audience to wonder what the averaging is for.
#:
#: So the two fold gauges are drawn straddling the truth rather than on it: they
#: bracket the real pair, and their mean is the real combined quantile to the two
#: decimals the figure prints. Nothing downstream is invented — the M₀ gauge, and
#: θ₀ realized from it, come from a live `FoldAnchoredCut` at inclusion 0.
XQUANT_SHOWN_QUANTILES = (0.79, 0.81)

#: How far the gauge's cut stands proud of the bar, and where the realizing
#: stroke starts from.
GAUGE_STUB = 0.16

#: The two fold models' haystack scores and M₀'s, as `((mu, sigma), (mu,
#: sigma))` for the Bad and Good populations. This is the figure's premise, so
#: the three are deliberately **on three different scales**: fold 1 spreads
#: (σ ≈ 0.10), fold 2 is right-shifted and tight, and M₀ — trained on all of
#: D₀ rather than half of it — separates the same media in a lower, narrower
#: band. Nothing about the media changed; three models scored them.
#:
#: The numbers are chosen so the strawman bites at the one place it can be
#: seen: fold 1 cuts at ≈0.50 and fold 2 at ≈0.66, so the *cardinal* average
#: of the two lands at ≈0.58 — which is M₀'s Good mean. Averaging raw scores
#: does not merely lose a little accuracy here; it puts the threshold through
#: the middle of the mound it is supposed to keep.
XQUANT_POPULATIONS = (
    ((0.24, 0.105), (0.76, 0.100)),
    ((0.46, 0.070), (0.86, 0.065)),
)
XQUANT_FINAL_POPULATION = ((0.20, 0.070), (0.58, 0.070))

#: Each fold's *held-out* votes, on that fold's own scale — the same seven
#: marks the progression has carried since the cross-calibration figure, moved
#: onto each model's axis. Fold 2 keeps the imperfect ordering: a Bad at 0.72
#: sits above its own fold's cut. No Good sits at its fold's μ_hi, where the
#: dashed mean stem would eat the glyph.
XQUANT_ANCHORS = (
    {"bad": (0.10, 0.19, 0.27, 0.36), "good": (0.64, 0.72, 0.88)},
    {"bad": (0.34, 0.41, 0.48, 0.72), "good": (0.75, 0.82, 0.94)},
)


def xquant_flow_fig() -> None:
    """Schematic of quantile transfer — iteration 5, the super-figure (#3218).

    The fifth and last figure of the calibration progression, and the one that
    **is** the shipped algorithm: fold-anchored fits at κ = `FOLD_ANCHOR_WEIGHT`,
    the cut at the midpoint of each fold's two fitted means, the per-fold cuts
    combined as a mean *quantile* rather than a mean score, and that mean
    quantile realized on the final model's own distribution.

    It is an assembly, and the geometry says so. Its left two thirds are
    `calib-fold-anchored-flow` at the same size, stroke for stroke: the
    haystack, the votes drawn out of it, the split, the crossed fold models,
    and a fitted, voted, cut mixture panel under each. Its right third is
    `calib-blend-flow`'s mixture branch returning — D₀ –train→ M₀, then a
    straight drop into a panel on the shared baseline — but doing a different
    job. In the blend, M₀ scored the haystack so a mixture could be fitted on
    it; here nothing is fitted on it, which is why the panel is drawn as bare
    bars with no curve over them. M₀'s distribution is not evidence in this
    figure. It is the *scale the answer has to be spoken in*.

    The argument runs in three moves, and the drawing is arranged so each is a
    comparison the audience makes rather than a claim they are told:

    * **The strawman is a location, not an assertion.** The three panels are
      the same width and each maps score 0–1 across it, so the average of two
      raw cuts can simply be dropped onto the third panel and looked at. It
      lands on M₀'s Good mound. Nothing has to say the cardinal average is
      wrong; it is drawn being wrong, and it stays on the final frame beside
      the answer that replaces it.
    * **The quantile is drawn as what it is** — the share of the corpus the cut
      calls Bad. Each panel's gauge is that share, hatched in `_data_block`'s
      own Bad and Good textures, so the reader can see the two folds agreeing
      on the fraction while disagreeing on the number. That is the whole of why
      a quantile survives the crossing between two models and a score does not.
    * **Realization is a stroke.** The mean quantile is a position on M₀'s
      gauge; θ₀ is where that position lands on M₀'s distribution, and the arrow
      between them is the last operation the algorithm performs.

    One honesty note, recorded here because it is the figure's only fiction:
    production never went through a cardinal-averaging stage. Quantile transfer
    was in the fold-anchored design from the start. The strawman is drawn
    because it teaches the fix, not because it happened.
    """
    folds, final = _xquant_populations()
    final_stage = _xquant_flow_stage(XQUANT_FLOW_STAGES, folds, final)
    box = tight_box(final_stage)
    for stage in range(1, XQUANT_FLOW_STAGES):
        save(
            _xquant_flow_stage(stage, folds, final),
            OUT,
            f"calib-quantile-flow.build{stage}.png",
            column=SIDEBAR_WIDE,
            box=box,
        )
    save(final_stage, OUT, "calib-quantile-flow.png", column=SIDEBAR_WIDE, box=box)


def _xquant_populations() -> tuple[list[tuple[GmmFit1D, np.ndarray, dict]], np.ndarray]:
    """The two fold fits (real estimator) and M₀'s unfitted haystack scores.

    Schematic input, real estimator, as everywhere else in the progression:
    each fold runs the shipped `fit_anchored_score_gmm` at `FOLD_ANCHOR_WEIGHT`
    on its own population and its own held-out anchors. M₀'s scores are *not*
    fitted — the figure's point is that no mixture is estimated on them.
    """
    rng = np.random.default_rng(11)

    def sample(neg: tuple[float, float], pos: tuple[float, float]) -> np.ndarray:
        return np.clip(np.concatenate([rng.normal(*neg, 4800), rng.normal(*pos, 1200)]), 0.0, 1.0)

    folds = []
    for (neg, pos), anchors in zip(XQUANT_POPULATIONS, XQUANT_ANCHORS, strict=True):
        scores = sample(neg, pos)
        a_scores = np.array(anchors["bad"] + anchors["good"], dtype=float)
        a_labels = np.concatenate([np.zeros(len(anchors["bad"])), np.ones(len(anchors["good"]))])
        fit, provenance = fit_anchored_score_gmm(scores, a_scores, a_labels, anchor_weight=FOLD_ANCHOR_WEIGHT)
        assert fit is not None and provenance == "anchored"
        folds.append((fit, scores, anchors))
    return folds, sample(*XQUANT_FINAL_POPULATION)


def _xquant_cut(folds: list, final: np.ndarray) -> "FoldAnchoredCut":
    """The **real** shipped estimator over this figure's three populations."""
    return FoldAnchoredCut(
        fits=tuple(f for f, _s, _a in folds),
        fold_haystacks=tuple(np.sort(s) for _f, s, _a in folds),
        final_haystack=np.sort(final),
        n_anchored=len(folds),
    )


def _quantile_gauge(
    ax: plt.Axes, x0: float, y0: float, w: float, h: float, q: float, label: str
) -> matplotlib.text.Text:
    """The corpus, sorted by score and cut at quantile *q*.

    A quantile is a *share of the population*, and the figure has to show that
    rather than assert it — the whole argument is that the share is what two
    models agree on when their scores do not. So the gauge is the population
    itself: the same Bad and Good hatching `_data_block` draws labelled media
    with, split left-to-right at the fraction the cut admits, and notched at
    that split with the identical mark every cut in the progression carries.

    Drawn under its panel and the width of it, so the fraction can be read
    against the distribution it was measured on; but it is a rank axis, not a
    score axis, and the two do not agree — which is exactly why the fold
    panels' notches sit at different places along their axes while their
    gauges' notches sit at the same place along theirs.
    """
    ax.add_patch(
        Rectangle((x0, y0), q * w, h, facecolor="white", edgecolor=RUST, hatch="\\\\\\", linewidth=0, zorder=2)
    )
    ax.add_patch(
        Rectangle(
            (x0 + q * w, y0), (1 - q) * w, h, facecolor="white", edgecolor=GREEN, hatch="////", linewidth=0, zorder=2
        )
    )
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=INK, linewidth=1.6, zorder=3))
    # The cut runs the full depth of the bar and stands `GAUGE_STUB` proud of
    # its top: on a panel the notch hangs below the baseline because the
    # distribution is above it, and here the population is below, so the same
    # mark is drawn on the same side of the thing it cuts.
    ax.plot([x0 + q * w] * 2, [y0, y0 + h + GAUGE_STUB], color=INK, linewidth=2.2, zorder=4)
    return ax.text(x0 + w / 2, y0 - LABEL_GAP, label, ha="center", va="top", fontsize=15, color=INK)


def _theta_notch(ax: plt.Axes, x: float, y_base: float, label: str, ha: str = "center") -> None:
    """The progression's one cut mark: a notch under the baseline, then a name."""
    ax.plot([x] * 2, [y_base - 0.32, y_base], color=INK, linewidth=2.2, zorder=6)
    dx = 0.0 if ha == "center" else (-0.10 if ha == "right" else 0.10)
    ax.text(x + dx, y_base - 0.32 - LABEL_GAP, label, ha=ha, va="top", fontsize=16, color=INK)


def _xquant_numbers(folds: list, final: np.ndarray) -> tuple[list[float], float, float]:
    """The per-fold cuts, the combined quantile, and the threshold it realizes.

    All three from the shipped estimator: the midpoint of each fold\'s own fitted
    means, and then a live `FoldAnchoredCut` at inclusion 0 for the mean quantile
    and the threshold realized from it, so the figure cannot drift from the
    algorithm it claims to draw. The only numbers on the drawing that do *not*
    come from here are the two fold gauges\' labels — see
    `XQUANT_SHOWN_QUANTILES` for what they are and why.
    """
    cut = _xquant_cut(folds, final)
    thetas = [gmm_cut_from_fit(fit, "mid", 1.0, 1.0)[0] for fit, _s, _a in folds]
    return thetas, cut.quantile_at(0), cut.threshold_at(0)


def _xquant_fold_panel(
    ax: plt.Axes,
    i: int,
    fold: tuple,
    theta: float,
    *,
    x0: float,
    y_base: float,
    w: float,
    h: float,
    top: float,
    score_from: tuple[float, float],
    score_to: tuple[float, float],
) -> None:
    """One fold's whole evidence display: `calib-fold-anchored-flow`'s panel.

    Every stroke here is the previous figure — the score arrow into the panel,
    the fitted and hatched mixture, the held-out votes on the baseline, the
    panel's name, and the midpoint cut — so it is drawn from one call and
    revealed in one advance. The quantile figure has its own three moves to spend
    advances on, and re-walking a picture the audience was already walked through
    would spend them on nothing.

    Both of the panel's names are kept — `Mᵢ(D₋₁)` for the humps and `Mᵢ(D_j)`
    for the votes standing in them — because a panel holding two quantities has
    to name both, and this is the figure where a reader most needs to know that
    the votes came from the *other* half. What goes is `_mark_legend`'s pair of
    glyphs beside that second name. There they were the first ✗ and ✓ to appear
    inside a mixture panel rather than on a score line, so a key was worth its
    space; two figures on, they are simply the evidence, and a floating key for
    a mark the audience already reads is ink competing with the three moves this
    figure exists to make.

    Both names hang at the panel's own outer corner, which `_mark_legend` could
    not do — it had glyphs to fit in first, so its name sat inboard of them. Two
    labels on one edge read as one group naming one panel.
    """
    fit, scores, anchors = fold
    outer = x0 + (w if i else 0.0)
    _labeled_arrow(ax, score_from, score_to, "score", z=2.1)
    ax.text(
        outer,
        top + LABEL_GAP,
        _sub(f"M_{i + 1}(D_{{-1}})"),
        ha="right" if i else "left",
        va="bottom",
        fontsize=16,
        color=INK,
    )
    ax.text(
        outer,
        top + LABEL_GAP + CAP_16 + LABEL_GAP,
        _sub(f"M_{i + 1}(D_{2 - i})"),
        ha="right" if i else "left",
        va="baseline",
        fontsize=15,
        color=INK,
    )
    _score_histogram(ax, x0, y_base, w, h, fit, scores, fill="class", mu_labels=False)
    _hump_marks(ax, x0, y_base, w, anchors)
    _theta_notch(ax, x0 + theta * w, y_base, _sub(rf"\theta_{i + 1} = {theta:.2f}"))


def _xquant_gauges(ax: plt.Axes, xs: tuple[float, ...], y0: float, w: float, q_0: float, clear_x: float) -> None:
    """The three gauges of the combine step, in one row under the three panels.

    Drawn together because they are one comparison, not three readings: what the
    row has to show is two folds agreeing on a fraction they disagree about the
    score of, and a third bar carrying that agreed fraction over to the model the
    threshold will be applied by. The third is `q₀` and not a bare `q` for the
    same reason the threshold read off it is `θ₀`: both belong to M₀, and a
    subscript that names its model is what says the share has *moved* rather
    than merely been averaged.
    """
    shown = XQUANT_SHOWN_QUANTILES
    texts = []
    for x, q, label in zip(
        xs,
        (*shown, q_0),
        (
            _sub(rf"q_1 = {shown[0]:.0%}".replace("%", r"\%")),
            _sub(rf"q_2 = {shown[1]:.0%}".replace("%", r"\%")),
            _sub(rf"q_0 = avg(q_1,\, q_2) = {q_0:.0%}".replace("%", r"\%")),
        ),
        strict=True,
    ):
        texts.append(_quantile_gauge(ax, x, y0, w, XQUANT_GAUGE_H, q, label))
    # The last of the three is the widest and sits in the figure\'s bottom-right
    # corner, which is where the theme prints the page number over it. Measured
    # off its own ink and pulled left if it overhangs, exactly as the blend
    # figure does with its conclusion: the rendered width of a mathtext run is
    # not something to leave to an estimate.
    ax.figure.canvas.draw()
    box = texts[-1].get_window_extent().transformed(ax.transData.inverted())
    if box.x1 > clear_x:
        texts[-1].set_x(texts[-1].get_position()[0] - (box.x1 - clear_x))


def _xquant_flow_stage(stage: int, folds: list, final: np.ndarray) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic."""
    fig, ax = plt.subplots(figsize=tuple(c * FLOW_UNIT_PT / 72 for c in XQUANT_CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, XQUANT_CANVAS[0])
    ax.set_ylim(0, XQUANT_CANVAS[1])
    ax.set_axis_off()

    canvas_w, canvas_h = XQUANT_CANVAS
    panel_h, panel_gap = XQUANT_PANEL_H, 2 * OBJECT_GAP

    # ── layout ────────────────────────────────────────────────────────────────
    # The fold half is `_xsemi_flow_stage`'s arithmetic verbatim, so the two
    # figures' shared strokes land on the same drawing units and the reader
    # recognises the picture instead of re-reading it. Only the spine's x moves:
    # here the fold pair is pushed right by nothing and M₀'s branch takes the
    # width to its right, so `bx` is fixed by the left panel's own margin.
    block_w, block_h = 4.0, 1.05
    fold_train_len, score_len, slope = 1.15, 1.3, 0.75
    # Solve the panel width from where the fold branch's score arrow has to
    # land — 0.62 across the panel, past the cut and short of μ_hi — exactly as
    # the fold-anchored figure solves it, which is what keeps the two figures'
    # panels the same size.
    ux, uy = np.array([slope, 1.0]) / float(np.hypot(slope, 1.0))
    t_edge = min(MODEL_W / 2 / abs(ux), MODEL_H / 2 / abs(uy)) + OBJECT_GAP
    dx_to_tip = block_w / 4 + t_edge * ux + score_len * ux  # bx − tip1.x
    panel_w = (dx_to_tip - panel_gap / 2) / 0.38
    bx = 0.2 + panel_w + panel_gap / 2

    block_x0 = bx - block_w / 2
    hay_x0 = block_x0
    hay_w = canvas_w - 0.2 - hay_x0
    hay_h = block_h
    hay_y0 = canvas_h - hay_h
    vote_len = 1.15
    block_top = hay_y0 - OBJECT_GAP - vote_len - OBJECT_GAP
    block_y0 = block_top - block_h
    row_y = block_y0 + block_h / 2

    m1x, m2x = bx - block_w / 4, bx + block_w / 4
    train_tail_y = block_y0 - OBJECT_GAP
    my = train_tail_y - fold_train_len - OBJECT_GAP - MODEL_H / 2

    down_left = (m1x - slope, my - 1.0)
    exit1 = _box_edge(m1x, my, down_left, OBJECT_GAP)
    tip1 = (exit1[0] - score_len * ux, exit1[1] - score_len * uy)

    panel_top = tip1[1] - (OBJECT_GAP + CAP_16 + LABEL_GAP)
    y_base = panel_top - panel_h
    fold_x = (bx - panel_gap / 2 - panel_w, bx + panel_gap / 2)
    final_x = fold_x[1] + panel_w + panel_gap

    # M₀'s branch, returning from the blend figure: train out to the right, then
    # one straight drop onto the distribution it produces. The blend aims its own
    # drop off the tall bars (0.62 across) because a mixture is about to be
    # fitted under it and the fit is what the eye has to reach; nothing is fitted
    # here, so the drop can land on the shape itself — at its **centre of mass**,
    # which is the honest middle of a histogram this left-weighted and is nearly
    # a third of the panel left of its geometric centre. That also takes some
    # three units off the train arrow reaching M₀.
    m0x = final_x + float(np.mean(final)) * panel_w
    train_x = block_x0 + block_w + OBJECT_GAP
    train_len = m0x - MODEL_W / 2 - OBJECT_GAP - train_x
    # The head clears the panel's own name by an object gap instead of abutting
    # the row it sits in. This drop is the longest arrow in the progression and
    # every unit off it is worth having.
    tip0 = (m0x, panel_top + 2 * OBJECT_GAP + CAP_16 + LABEL_GAP)

    theta_bottom = y_base - 0.32 - LABEL_GAP - CAP_16
    gauge_top = theta_bottom - OBJECT_GAP - GAUGE_STUB
    gauge_y0 = gauge_top - XQUANT_GAUGE_H
    gauge_w = XQUANT_GAUGE_W * panel_w
    # Where the bottom-right corner stops being available: the theme prints the
    # page number over a `bg` figure, and this one ends with a row of gauges and
    # their labels that would otherwise run straight through it. See
    # `PAGE_NUMBER_CLEAR` for the measurement; fitted, this figure renders at
    # ~52px per drawing unit, so the badge\'s ink reaches ~2.1 units in from the
    # cropped right edge, which is the panel\'s right edge plus the crop\'s pad.
    clear_x = final_x + panel_w + 0.27 - 2.1

    data_block = functools.partial(_data_block, ax)
    arrow = functools.partial(_arrow, ax)
    labeled_arrow = functools.partial(_labeled_arrow, ax)
    model_box = functools.partial(_model_box, ax)

    thetas, q_0, theta_0 = _xquant_numbers(folds, final)
    theta_cardinal = float(np.mean(thetas))

    # ── stage 1: the spine, unchanged since the mixture figure ────────────────
    _haystack_block(ax, hay_x0, hay_y0, hay_w, hay_h)
    _disc_label(ax, hay_x0 + hay_w / 2, hay_y0 + hay_h / 2, "D_{-1}")
    ax.text(hay_x0 - LABEL_GAP, hay_y0 + hay_h / 2, "Unlabeled", ha="right", va="center", fontsize=15, color=SOFT)
    labeled_arrow((bx, hay_y0 - OBJECT_GAP), (bx, hay_y0 - OBJECT_GAP - vote_len), "vote")
    data_block(block_x0, block_y0, block_w, block_h, split=stage >= 2)
    good_h = 0.42 * block_h
    ax.text(block_x0 - LABEL_GAP, block_top - good_h / 2, "Good", ha="right", va="center", fontsize=15, color=GREEN)
    ax.text(
        block_x0 - LABEL_GAP, block_y0 + (block_h - good_h) / 2, "Bad", ha="right", va="center", fontsize=15, color=RUST
    )
    ax.text(block_x0, block_top + LABEL_GAP, _sub("D_0"), ha="left", va="bottom", fontsize=16, color=INK)
    labeled_arrow((train_x, row_y), (train_x + train_len, row_y), "train")
    model_box(m0x, row_y, "M_0")

    # ── stage 2: split the votes, train a fold model on each half ─────────────
    if stage >= 2:
        for mx, name, sign in ((m1x, "M_1", 1.0), (m2x, "M_2", -1.0)):
            model_box(mx, my, name)
            labeled_arrow((mx, train_tail_y), (mx, train_tail_y - fold_train_len), "train")
            entry_tail = (mx + sign * slope * (train_tail_y - my), train_tail_y)
            arrow(entry_tail, _box_edge(mx, my, entry_tail, OBJECT_GAP))

    # ── stage 3: the fold-anchored figure, recapitulated in one advance ───────
    # Fitted, voted and cut together: every stroke here is the previous figure
    # and the audience has already been walked through it a step at a time.
    if stage >= 3:
        for i, (fit, scores, anchors) in enumerate(folds):
            _xquant_fold_panel(
                ax,
                i,
                (fit, scores, anchors),
                thetas[i],
                x0=fold_x[i],
                y_base=y_base,
                w=panel_w,
                h=panel_h,
                top=panel_top,
                score_from=(2 * bx - exit1[0], exit1[1]) if i else exit1,
                score_to=(2 * bx - tip1[0], tip1[1]) if i else tip1,
            )

    # ── stage 4: M₀ scores the haystack — a scale, not evidence ───────────────
    # Bare bars, and no fitted curves over them: nothing is estimated on this
    # distribution. It is here because the threshold has to be a number M₀ can
    # apply, and M₀'s numbers are its own.
    if stage >= 4:
        arrow((m0x, hay_y0 - OBJECT_GAP), _box_edge(m0x, row_y, (m0x, hay_y0), OBJECT_GAP))
        labeled_arrow(_box_edge(m0x, row_y, tip0, OBJECT_GAP), tip0, "score", z=2.1)
        # At the panel's outer corner, as the fold panels' names are: the drop now
        # comes down over the left third of this panel, which is where a
        # left-aligned name would be standing.
        ax.text(
            final_x + panel_w,
            panel_top + LABEL_GAP,
            _sub("M_0(D_{-1})"),
            ha="right",
            va="bottom",
            fontsize=16,
            color=INK,
        )
        _score_histogram(ax, final_x, y_base, panel_w, panel_h, None, final, fill="plain", mu_labels=False)

    # ── stage 5: the strawman — average the two numbers, and look ─────────────
    # Offered the way every other cut in the progression is offered: a dashed
    # stem up through the distribution, and the same notch under the baseline.
    # Grey, and struck through, so the proposal and its refusal arrive together.
    # Its value is left off the rule — θ₁ and θ₂ are printed two panels away, and
    # an audience that averages them itself is more convinced than one told the
    # answer. Below the baseline rather than above it because M₀'s drop now lands
    # in the middle of the panel, and a label wide enough to hold this one has
    # nowhere to stand up there that the arrow does not already occupy.
    if stage >= 5:
        cx = final_x + theta_cardinal * panel_w
        ax.plot([cx, cx], [y_base, panel_top], color=SOFT, linewidth=2.0, linestyle=(0, (4, 3)), zorder=6)
        ax.plot([cx, cx], [y_base - 0.32, y_base], color=SOFT, linewidth=2.2, zorder=6)
        strike = ax.text(
            final_x + panel_w,
            y_base - 0.32 - LABEL_GAP,
            _sub(r"avg(\theta_1,\, \theta_2)"),
            ha="right",
            va="top",
            fontsize=15,
            color=SOFT,
        )
        # Struck through, and measured rather than guessed: the rendered width
        # of a mathtext run is not a number to estimate, and a rule that
        # overhangs its own text reads as a different mark entirely.
        fig.canvas.draw()
        sb = strike.get_window_extent().transformed(ax.transData.inverted())
        ax.plot(
            [sb.x0 - 0.06, sb.x1 + 0.06],
            [(sb.y0 + sb.y1) / 2 + 0.02] * 2,
            color=RUST,
            linewidth=1.5,
            zorder=7,
            solid_capstyle="round",
        )

    # ── stage 6: re-read each cut as a share of the corpus, and average ───────
    if stage >= 6:
        _xquant_gauges(ax, (*fold_x, final_x), gauge_y0, gauge_w, q_0, clear_x)

    # ── stage 7: realize the mean share on M₀'s own distribution ─────────────
    if stage >= 7:
        theta_x = final_x + theta_0 * panel_w
        _theta_notch(ax, theta_x, y_base, _sub(rf"\theta_0 = {theta_0:.2f}"), ha="right")
        arrow((final_x + q_0 * gauge_w, gauge_top + GAUGE_STUB + OBJECT_GAP / 2), (theta_x, y_base - 0.32))

    return fig


# ═══════════════════════════════════════════════════════════════════════════
# Part 2 — the Inclusion progression (issue #3218).
#
# The calibration progression above asks "where does the line go?"; this one
# asks the second question the room always has — "what if I wanted more, or
# fewer, false positives?" — and answers it by walking the *same* machinery a
# second time. Every figure here is a piece of `calib-quantile-flow` with the
# knob turned on it: Figs F and G are its left panel (one fold model's corpus
# and the votes standing on that corpus's baseline), Fig H is the whole of it,
# and Fig I is its right panel.
#
# **These figures share Fig E's canvas *width* rather than Part 1's canvas
# height.** Part 1's schematics are height-limited in a `bg right:70%` slot, so
# pinning `FLOW_CANVAS_H` is what keeps a 15pt label the same size on every
# slide of it. These four are much wider than they are tall — a score axis with
# things hung under it — so they are *width*-limited in the same slot, and it is
# the width that has to be pinned instead. At `XQUANT_CANVAS[0]` a 15pt label
# renders at 21.2px, which is `calib-quantile-flow`'s own number: the type does
# not change size when the talk crosses from Part 1 into Part 2. Heights may
# differ, and must stay under `INCL_CANVAS_W * 720 / 896` (13.37) or the figure
# becomes height-limited and the pin stops holding.
INCL_CANVAS_W = XQUANT_CANVAS[0]

#: The panel every Part 2 figure hangs its argument on, in drawing units.
INCL_PANEL_X0, INCL_PANEL_W, INCL_PANEL_H = 1.4, 13.85, 2.2

#: The three stops each figure reads its gauges at. The retired rule returns one
#: answer at every stop, so the knob's two ends and its middle are the fairest
#: three to draw it at; the conformal rule's motion is all below inclusion 0 on
#: cleanly separated votes (its false-negative cap is an upper bound and goes
#: unspent when the negatives force no sacrifice), so the walk is read at the
#: three stops where it has something to show. The pile of stops at ``k >= 0``
#: is drawn rather than hidden — see `_walk_flow_stage`.
KNOB_STOPS = (10, 0, -10)
WALK_STOPS = (0, -5, -10)

#: The knob's own range, from `vtscore.training.thresholds.INCLUSION_MIN/MAX`.
#: Drawn as a tick per stop, so a rule that returns one cut for the whole
#: slider draws twenty-one ticks in one place.
INCL_KNOB = tuple(range(-10, 11))

#: Fold 1's held-out votes and the corpus it scored, from `calib-quantile-flow`
#: — the same seven ✗s and ✓s the progression has carried since
#: `calib-xcal-flow`, standing on the same distribution. Part 2 opens on a
#: picture the audience has already been walked through twice.
INCL_VOTES = XQUANT_ANCHORS[0]
INCL_POPULATION = XQUANT_POPULATIONS[0]

#: How many build stages each of the four reveal in.
KNOB_FLOW_STAGES = 5
WALK_FLOW_STAGES = 6
TILT_FLOW_STAGES = 6
ACQ_FLOW_STAGES = 5

#: The gauge row's geometry: three bars across the panel's width.
INCL_GAUGE_GAP = 0.8
INCL_GAUGE_W = (INCL_PANEL_W - 2 * INCL_GAUGE_GAP) / 3
INCL_GAUGE_H = XQUANT_GAUGE_H


def _incl_corpus() -> np.ndarray:
    """Fold 1's scored corpus, redrawn exactly as `_xquant_populations` draws it.

    Same generator, same seed, same first draw, so the histogram in Part 2 is
    the *same shape* as the left panel of `calib-quantile-flow` rather than a
    lookalike. Nothing is fitted on it here: the two cut rules Part 2 compares
    read the seven votes and nothing else, and the corpus is drawn as bare bars
    for the reason `calib-quantile-flow` draws M₀ that way — it is not evidence,
    it is what the cut is applied to.
    """
    rng = np.random.default_rng(11)
    neg, pos = INCL_POPULATION
    return np.clip(np.concatenate([rng.normal(*neg, 4800), rng.normal(*pos, 1200)]), 0.0, 1.0)


def _incl_votes() -> tuple[np.ndarray, np.ndarray]:
    """The seven held-out votes as `(scores, labels)`, in the estimators' order."""
    scores = np.array(INCL_VOTES["bad"] + INCL_VOTES["good"], dtype=float)
    labels = np.concatenate([np.zeros(len(INCL_VOTES["bad"])), np.ones(len(INCL_VOTES["good"]))])
    return scores, labels


def _argmin_cut(scores: np.ndarray, labels: np.ndarray, inclusion: int) -> float:
    """The **retired** min-cost threshold search, reconstructed for the figure.

    This is the one place in the deck's generators that cannot delegate to
    `vtscore`, because the code it draws no longer exists: `find_optimal_threshold`
    was deleted when the conformal rule shipped (#2693), which is the whole
    point of the slide. It is reconstructed from the rule as
    `docs/experiments/inclusion-knob/REPORT.md` states it — the minimum of
    ``fpr_weight·FPR + fnr_weight·FNR`` over the observed held-out cut points,
    with the weights from the shipped :func:`inclusion_cost_weights` so the
    *knob* half of the picture is still the live definition.

    The tie-break is the failure the figure is about and is therefore not
    incidental: on cleanly ranked votes every cut in the band between the two
    classes has cost zero under every weighting, so the search returns whichever
    of them it happens to see first and returns *that same one* at all twenty-one
    stops of the knob.
    """
    fpr_weight, fnr_weight = inclusion_cost_weights(inclusion)
    n_neg, n_pos = float((labels == 0).sum()), float((labels == 1).sum())
    best, best_cost = float(scores.max()), np.inf
    for cut in np.sort(scores):
        predicted = scores > cut - 1e-9
        fpr = float(((labels == 0) & predicted).sum()) / n_neg
        fnr = float(((labels == 1) & ~predicted).sum()) / n_pos
        cost = fpr_weight * fpr + fnr_weight * fnr
        if cost < best_cost - 1e-12:
            best, best_cost = float(cut), cost
    return best


def _normalised_cost(scores: np.ndarray, labels: np.ndarray, inclusion: int, grid: np.ndarray) -> np.ndarray:
    """``(w_f·FPR + w_n·FNR) / (w_f + w_n)`` over a grid of cut positions.

    Divided by the weights' sum only so the two ends of the knob — which price
    the two errors a thousand to one in opposite directions — can be drawn on
    one vertical scale. It is a rescaling by a positive constant, so it moves
    neither the curve's shape nor its argmin, which are the two things the
    figure reads off it.
    """
    fpr_weight, fnr_weight = inclusion_cost_weights(inclusion)
    neg, pos = scores[labels == 0], scores[labels == 1]
    fpr = np.array([float((neg > t).mean()) for t in grid])
    fnr = np.array([float((pos <= t).mean()) for t in grid])
    return (fpr_weight * fpr + fnr_weight * fnr) / (fpr_weight + fnr_weight)


def _incl_panel(ax: plt.Axes, corpus: np.ndarray, *, y_base: float, top: float, votes: bool = True) -> None:
    """`calib-quantile-flow`'s left panel, unfitted: the corpus, and the votes on it.

    The bars carry no fill and no fitted curve because nothing is estimated on
    them — the cut rules Part 2 compares read the seven marks on the baseline
    and nothing else.
    Both names are kept for the reason the quantile figure keeps both: a panel
    holding two quantities has to name both, and which model scored which half
    of the votes is the fact the whole progression turns on.
    """
    x0, w = INCL_PANEL_X0, INCL_PANEL_W
    _score_histogram(ax, x0, y_base, w, INCL_PANEL_H, None, corpus, fill="plain", mu_labels=False)
    ax.text(x0, top + LABEL_GAP, _sub("M_1(D_{-1})"), ha="left", va="bottom", fontsize=16, color=INK)
    if votes:
        ax.text(
            x0,
            top + LABEL_GAP + CAP_16 + LABEL_GAP,
            _sub("M_1(D_2)"),
            ha="left",
            va="baseline",
            fontsize=15,
            color=INK,
        )
        _hump_marks(ax, x0, y_base, w, INCL_VOTES)


def _incl_gauges(
    ax: plt.Axes,
    corpus: np.ndarray,
    stops: "tuple[int, ...]",
    cuts: "list[float]",
    *,
    y0: float,
    stop_label_y: float,
) -> None:
    """One gauge per stop: what the knob admits, at three settings of it.

    The same bar `calib-quantile-flow` reads a quantile off, doing the job it
    was built for one figure earlier — the corpus sorted by score and split at
    the cut. Three of them in a row is the comparison the section exists to
    make: under the retired rule they are the same picture three times.
    """
    for i, (inclusion, cut) in enumerate(zip(stops, cuts, strict=True)):
        x0 = INCL_PANEL_X0 + i * (INCL_GAUGE_W + INCL_GAUGE_GAP)
        q = float((corpus <= cut).mean())
        ax.text(
            x0 + INCL_GAUGE_W / 2,
            stop_label_y,
            _sub(rf"k = {inclusion:+d}" if inclusion else "k = 0"),
            ha="center",
            va="bottom",
            fontsize=16,
            color=INK,
        )
        _quantile_gauge(
            ax,
            x0,
            y0,
            INCL_GAUGE_W,
            INCL_GAUGE_H,
            q,
            _sub(rf"admits\ {1 - q:.0%}".replace("%", r"\%")),
        )


def _incl_figure(canvas_h: float) -> tuple[plt.Figure, plt.Axes]:
    """A Part 2 canvas: Fig E's width, this figure's own height."""
    fig, ax = plt.subplots(figsize=tuple(c * FLOW_UNIT_PT / 72 for c in (INCL_CANVAS_W, canvas_h)))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, INCL_CANVAS_W)
    ax.set_ylim(0, canvas_h)
    ax.set_axis_off()
    return fig, ax


#: The knob pair's shared canvas height. `calib-knob-flow` and `calib-walk-flow`
#: are a matched pair — the same panel, the same votes, the same gauge row, with
#: one row swapped between them — so every row lands on the same drawing unit in
#: both and the deck's flip from one to the other moves only the thing that
#: changed. `_incl_rows` is where that is enforced; the height is set so the
#: lower of the two figures' conclusion lines ends just inside the canvas.
INCL_CANVAS_H = 11.2


def _incl_rows() -> dict:
    """Every shared y in the knob pair, so the two figures overlay exactly.

    `calib-walk-flow` has no cut notch hanging under its panel and
    `calib-knob-flow` has no two-line anchor names under its middle row; both
    reserve the other's space anyway. Spending a few empty drawing units is what
    buys the property the pair exists for — flipping between the two slides
    moves the middle row and nothing else.
    """
    panel_top = INCL_CANVAS_H - (LABEL_GAP + CAP_16 + LABEL_GAP + CAP_16)
    y_base = panel_top - INCL_PANEL_H
    # Reserved on both: a cut notch under the panel and its name (knob only).
    theta_bottom = y_base - 0.32 - LABEL_GAP - CAP_16
    mid_label_y = theta_bottom - OBJECT_GAP - CAP_16
    mid_top = mid_label_y - LABEL_GAP
    mid_h = 1.9
    mid_base = mid_top - mid_h
    # Reserved on both: two rows of anchor names under the middle row (walk only).
    mid_bottom = mid_base - 0.32 - LABEL_GAP - CAP_16 - LABEL_GAP - CAP_16
    stop_label_y = mid_bottom - OBJECT_GAP - CAP_16
    gauge_top = stop_label_y - LABEL_GAP
    gauge_y0 = gauge_top - INCL_GAUGE_H
    conclusion_y = gauge_y0 - LABEL_GAP - CAP_16 - OBJECT_GAP - 0.23
    return {
        "panel_top": panel_top,
        "y_base": y_base,
        "theta_bottom": theta_bottom,
        "mid_label_y": mid_label_y,
        "mid_top": mid_top,
        "mid_h": mid_h,
        "mid_base": mid_base,
        "mid_bottom": mid_bottom,
        "stop_label_y": stop_label_y,
        "gauge_top": gauge_top,
        "gauge_y0": gauge_y0,
        "conclusion_y": conclusion_y,
    }


def knob_flow_fig() -> None:
    """Schematic of the knob that did not turn — Part 2's opening figure (#3218).

    The Inclusion slider is defined as a trade between the two error *rates*,
    ``cost = w_f·FPR + w_n·FNR``, each ``+1`` step doubling the price of a miss
    and each ``-1`` the price of a false alarm
    (:func:`vtscore.training.thresholds.inclusion_cost_weights`, which this
    figure calls rather than restates). The rule that first answered it took the
    minimum of that cost over the observed held-out cut points — and had exactly
    as many distinct optima as the calibration set had ranking errors.

    The drawing is that sentence. Seven cleanly ranked votes leave an empty band
    between the classes; both ends of the knob price the two errors a thousand
    to one in opposite directions, and *both cost curves are zero across the
    whole band*, so every cut in it ties at every setting. Twenty-one stops, one
    answer, three identical gauges (#2693,
    ``docs/experiments/inclusion-knob/REPORT.md``: 100% flat sweeps on the
    separable arm, and ~1.8 distinct admitted sizes across eleven stops on real
    embeddings).
    """
    corpus = _incl_corpus()
    scores, labels = _incl_votes()
    final = _knob_flow_stage(KNOB_FLOW_STAGES, corpus, scores, labels)
    box = tight_box(final)
    for stage in range(1, KNOB_FLOW_STAGES):
        save(
            _knob_flow_stage(stage, corpus, scores, labels),
            OUT,
            f"calib-knob-flow.build{stage}.png",
            column=SIDEBAR_WIDE,
            box=box,
        )
    save(final, OUT, "calib-knob-flow.png", column=SIDEBAR_WIDE, box=box)


def _knob_flow_stage(stage: int, corpus: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic."""
    fig, ax = _incl_figure(INCL_CANVAS_H)
    x0, w = INCL_PANEL_X0, INCL_PANEL_W

    # The cost panel's own labels sit above it, so the object gap below the cut's
    # name is measured to *them* rather than to the curves they name; `_incl_rows`
    # holds that arithmetic, because the walk figure has to land on it too.
    rows = _incl_rows()
    panel_top, y_base = rows["panel_top"], rows["y_base"]
    cost_label_y, cost_h, cost_base = rows["mid_label_y"], rows["mid_h"], rows["mid_base"]
    stop_label_y, gauge_y0, conclusion_y = rows["stop_label_y"], rows["gauge_y0"], rows["conclusion_y"]

    # ── stage 1: the corpus, and the seven held-out votes standing on it ──────
    _incl_panel(ax, corpus, y_base=y_base, top=panel_top)

    # ── stage 2: the only cuts the search can return ──────────────────────────
    # A tick per observed vote score. Shorter than the progression's own cut
    # notch and in soft grey: these are candidates, not a decision.
    if stage >= 2:
        for score in scores:
            ax.plot([x0 + score * w] * 2, [y_base - 0.18, y_base], color=SOFT, linewidth=1.6, zorder=5)

    # ── stage 3: what a cut costs, at the two ends of the knob ────────────────
    # Drawn together because the whole content is that they agree. At k = +10 a
    # miss is priced 1024:1 and the curve is essentially the false-negative
    # rate; at k = -10 it is essentially the false-positive rate. Between the
    # top ✗ and the bottom ✓ neither error is possible, so both curves sit on
    # zero — and every cut in that band is optimal at every setting of a knob
    # whose two ends are three orders of magnitude apart.
    band_lo, band_hi = float(scores[labels == 0].max()), float(scores[labels == 1].min())
    band_mid = (band_lo + band_hi) / 2
    if stage >= 3:
        ax.add_patch(
            Rectangle(
                (x0 + band_lo * w, cost_base),
                (band_hi - band_lo) * w,
                cost_h,
                facecolor=NEUTRAL_FILL,
                edgecolor="none",
                zorder=1,
            )
        )
        # The band is a fact about the *votes*, so it is tied back to them: two
        # soft rules running from the panel's baseline down the figure, standing
        # in the gap between the top ✗ and the bottom ✓. Without them the shaded
        # rectangle reads as a feature of the cost curves rather than as the
        # place on the score axis where the calibration data runs out.
        for edge in (band_lo, band_hi):
            ax.plot(
                [x0 + edge * w] * 2,
                [cost_base, y_base],
                color=RULE,
                linewidth=1.4,
                zorder=0,
            )
        grid = np.linspace(0.0, 1.0, 800)
        for inclusion, style, name_at in ((10, "solid", 0.965), (-10, (0, (4, 3)), 0.035)):
            curve = _normalised_cost(scores, labels, inclusion, grid)
            ax.plot(
                x0 + grid * w,
                cost_base + curve * cost_h,
                color=INK,
                linewidth=2.2,
                linestyle=style,
                zorder=3,
            )
            ax.text(
                x0 + name_at * w,
                cost_label_y,
                _sub(rf"k = {inclusion:+d}"),
                ha="right" if inclusion > 0 else "left",
                va="bottom",
                fontsize=15,
                color=INK,
            )
        ax.plot([x0, x0 + w], [cost_base] * 2, color=INK, linewidth=1.8, zorder=4)
        # The row's name is the definition of the knob, which is the one thing
        # every rule in this section shares — and putting it here rather than in
        # the slide's own copy means the pair of figures carries it, so the walk
        # figure's identical row is read against the same sentence.
        ax.text(
            x0 + band_mid * w,
            cost_label_y,
            _sub(r"cost = w_f\cdot FPR + w_n\cdot FNR"),
            ha="center",
            va="bottom",
            fontsize=16,
            color=INK,
        )
        ax.text(
            x0 + band_mid * w,
            cost_base + LABEL_GAP,
            "zero, either way",
            ha="center",
            va="bottom",
            fontsize=15,
            color=SOFT,
        )

    # ── stage 4: turn the knob, twenty-one times, and watch ───────────────────
    # The two lines under the cost row are the mechanism, and they stand in the
    # rows `_incl_rows` reserves for the walk figure's anchor names — so the pair
    # keeps its overlay and neither figure carries an empty band.
    if stage >= 4:
        ax.text(
            x0 + band_mid * w,
            cost_base - 0.32 - LABEL_GAP,
            "no ✗ ranks above a ✓",
            ha="center",
            va="top",
            fontsize=16,
            color=INK,
        )
        ax.text(
            x0 + band_mid * w,
            cost_base - 0.32 - LABEL_GAP - CAP_16 - LABEL_GAP,
            "so the search has one optimum, at every price",
            ha="center",
            va="top",
            fontsize=15,
            color=SOFT,
        )
        cuts = [_argmin_cut(scores, labels, k) for k in INCL_KNOB]
        for cut in cuts:
            ax.plot([x0 + cut * w] * 2, [y_base - 0.32, y_base], color=INK, linewidth=2.2, zorder=6)
        theta = cuts[0]
        ax.text(
            x0 + theta * w + 0.10,
            y_base - 0.32 - LABEL_GAP,
            _sub(r"\theta\ at\ all\ 21\ stops"),
            ha="left",
            va="top",
            fontsize=16,
            color=INK,
        )

    # ── stage 5: three settings of the knob, three identical answers ──────────
    if stage >= 5:
        cuts = [_argmin_cut(scores, labels, k) for k in KNOB_STOPS]
        _incl_gauges(ax, corpus, KNOB_STOPS, cuts, y0=gauge_y0, stop_label_y=stop_label_y)
        ax.text(
            x0 + w / 2,
            conclusion_y,
            "one answer, whichever way you turn it",
            ha="center",
            va="center",
            fontsize=17,
            color=INK,
        )

    return fig


#: The false-positive guard's and the walk's endpoints, as
#: `vtscore.training.thresholds.conformal_threshold` defines them. Quoted here
#: only so the figure can *label* them; every cut it draws comes from calling
#: that function, not from re-deriving it.
WALK_GUARD_Q = 1.0 - 0.25  # the 1 - BASE*2^k quantile of the negatives, at k = 0
WALK_TOP_Q = 0.75  # CONFORMAL_QPOS_MAX: the positives' quantile the k = -10 end walks to


def walk_flow_fig() -> None:
    """Schematic of the conformal quantile walk — Part 2's repair (#2693, #3218).

    Same panel as `calib-knob-flow`, same seven votes, same corpus: only the
    middle row changes, from what the retired rule *computed* to what the
    shipped one computes. That is the comparison the pair exists to make.

    The rule is :func:`vtscore.training.thresholds.conformal_threshold`, which
    this figure calls at each of the knob's twenty-one stops rather than
    restating, and it is drawn in the order it composes:

    * the **false-positive guard**, the negatives' 1 − 0.25·2^k quantile — a
      *quantile* of the ✗s and deliberately not their maximum, for the same
      reason the midpoint below is not the lowest ✓;
    * the **band** between that guard and the lowest ✓, which the calibration
      set cannot resolve, and the **gap midpoint** in it that inclusion 0 cuts
      at — the max-margin choice among cuts the data calls equal, where the
      retired rule took the band's top edge and got an extreme order statistic
      over a handful of votes;
    * the **walk** up from that midpoint to the positives' 75th percentile at
      k = −10 — "just the surest matches" — which is where the knob's
      resolution comes from: a quantile moves whenever the scores have any
      spread, and eleven stops of the slider land in eleven distinct places;
    * the **false-negative cap**, the positives' α(k) = 0.25·2^−k quantile,
      halving per step. It is a ceiling, not a target: on votes this cleanly
      ranked it sits above the walk and never binds, which is why the stops at
      k ≥ 0 pile on the midpoint. The figure draws that pile rather than hiding
      it — the cap is what gives the positive half of the knob its meaning once
      the classes overlap, and this vote set has no overlap to spend it on.
    """
    corpus = _incl_corpus()
    scores, labels = _incl_votes()
    final = _walk_flow_stage(WALK_FLOW_STAGES, corpus, scores, labels)
    box = tight_box(final)
    for stage in range(1, WALK_FLOW_STAGES):
        save(
            _walk_flow_stage(stage, corpus, scores, labels),
            OUT,
            f"calib-walk-flow.build{stage}.png",
            column=SIDEBAR_WIDE,
            box=box,
        )
    save(final, OUT, "calib-walk-flow.png", column=SIDEBAR_WIDE, box=box)


def _walk_flow_stage(stage: int, corpus: np.ndarray, scores: np.ndarray, labels: np.ndarray) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic."""
    fig, ax = _incl_figure(INCL_CANVAS_H)
    x0, w = INCL_PANEL_X0, INCL_PANEL_W

    # The middle row is `calib-knob-flow`'s cost panel — same height, same place,
    # same label row above it — holding the new rule instead of the old one's
    # arithmetic. That the two figures differ in exactly one row is the whole of
    # what the pair has to say, so both take their rows from `_incl_rows`.
    rows = _incl_rows()
    panel_top, y_base = rows["panel_top"], rows["y_base"]
    rule_label_y, rule_h, rule_base = rows["mid_label_y"], rows["mid_h"], rows["mid_base"]
    stop_label_y, gauge_y0, conclusion_y = rows["stop_label_y"], rows["gauge_y0"], rows["conclusion_y"]

    neg, pos = scores[labels == 0], scores[labels == 1]
    guard = float(np.quantile(neg, WALK_GUARD_Q))
    lowest_good = float(pos.min())
    top_quarter = float(np.quantile(pos, WALK_TOP_Q))
    band_mid = (guard + lowest_good) / 2
    stops = {k: conformal_threshold(scores.tolist(), labels.tolist(), k) for k in INCL_KNOB}

    def guide(score: float) -> None:
        """Tie a mark on the rule row back to the votes on the panel above it."""
        ax.plot([x0 + score * w] * 2, [rule_base, y_base], color=RULE, linewidth=1.4, zorder=0)

    def mark(score: float, name: str, sub: str) -> None:
        """One anchor of the rule: a full-height tick, named in two lines below."""
        # Stops short of the row's top so the walk's own arrow and label have a
        # clear band to run in; the shaded rectangle carries the full height.
        ax.plot([x0 + score * w] * 2, [rule_base - 0.32, rule_base + rule_h * 0.3], color=INK, linewidth=2.2, zorder=5)
        ax.text(x0 + score * w, rule_base - 0.32 - LABEL_GAP, name, ha="center", va="top", fontsize=16, color=INK)
        ax.text(
            x0 + score * w,
            rule_base - 0.32 - LABEL_GAP - CAP_16 - LABEL_GAP,
            sub,
            ha="center",
            va="top",
            fontsize=15,
            color=SOFT,
        )

    # ── stage 1: the same panel the retired rule was drawn on ─────────────────
    _incl_panel(ax, corpus, y_base=y_base, top=panel_top)
    ax.plot([x0, x0 + w], [rule_base] * 2, color=INK, linewidth=1.8, zorder=4)
    ax.text(
        x0 + band_mid * w, rule_label_y, "where the rule may cut", ha="center", va="bottom", fontsize=15, color=SOFT
    )

    # ── stage 2: the false-positive guard — a quantile of the ✗s, not their max ─
    if stage >= 2:
        guide(guard)
        mark(guard, "guard", "¾ of the ✗s")

    # ── stage 3: the band the votes cannot resolve, and its midpoint ──────────
    # Every cut between the guard and the lowest ✓ has the same empirical error
    # on this calibration set — which is what the previous figure's flat cost
    # floor was. The retired rule took the band's top edge; this one sits in the
    # middle of it, because that top edge is a single held-out vote and moves
    # violently when the next one arrives.
    #
    # The mark is named `k ≥ 0`, not `k = 0`, and that is exact rather than
    # cautious: above inclusion 0 the rule returns ``min(cap, θ(0))``, and the
    # gap midpoint is below every calibration positive, so on cleanly ranked
    # votes the cap can never bind and *every* non-negative stop lands here.
    if stage >= 3:
        guide(lowest_good)
        ax.add_patch(
            Rectangle(
                (x0 + guard * w, rule_base),
                (lowest_good - guard) * w,
                rule_h,
                facecolor=NEUTRAL_FILL,
                edgecolor="none",
                zorder=1,
            )
        )
        ax.text(
            x0 + band_mid * w,
            rule_base + rule_h * 0.3 + LABEL_GAP,
            "the votes cannot tell these apart",
            ha="center",
            va="bottom",
            fontsize=15,
            color=SOFT,
        )
        mark(stops[0], _sub(r"k \geq 0"), "midpoint")

    # ── stage 4: walk up the positives as false alarms get dearer ─────────────
    # Every stop of the knob below 0 is its own quantile, so every stop is its
    # own cut — the whole of the repair.
    if stage >= 4:
        guide(top_quarter)
        for k in INCL_KNOB:
            ax.plot([x0 + stops[k] * w] * 2, [rule_base - 0.32, rule_base], color=INK, linewidth=2.2, zorder=5)
        mark(stops[-10], _sub("k = -10"), "top ¼ of the ✓s")
        arrow_y = rule_base + rule_h * 0.685
        _arrow(
            ax,
            (x0 + stops[0] * w + OBJECT_GAP, arrow_y),
            (x0 + stops[-10] * w - OBJECT_GAP, arrow_y),
        )
        ax.text(
            x0 + (stops[0] + stops[-10]) / 2 * w,
            arrow_y + LABEL_GAP,
            "one stop, one quantile, one cut",
            ha="center",
            va="bottom",
            fontsize=15,
            color=INK,
        )

    # ── stage 5: the other half of the knob, and why it is quiet here ─────────
    # The rule's false-negative cap is a quantile of the ✓s — never above the
    # α(k) = 0.25·2^-k of them, halving per step — so it is drawn where it
    # lives: a bracket over the positives in the panel, not a tick on the axis
    # below. It is a ceiling, not a target, and on votes this cleanly ranked it
    # never binds: the gap midpoint is under every ✓, so nothing above
    # inclusion 0 has anything to give back, which is exactly the pile of ticks
    # `k ≥ 0` names. Under overlap it is what the positive half of the slider
    # spends, and a tick per stop would have to be redrawn for every one of
    # them inside a fifth of a score unit.
    if stage >= 5:
        brace_y = y_base + INCL_PANEL_H * 0.72
        lo, hi = x0 + float(pos.min()) * w, x0 + float(pos.max()) * w
        ax.plot(
            [lo, lo, hi, hi], [brace_y - 0.12, brace_y, brace_y, brace_y - 0.12], color=SOFT, linewidth=1.6, zorder=6
        )
        ax.text(
            (lo + hi) / 2,
            brace_y + LABEL_GAP,
            _sub(r"cap:\ \alpha(k)\ of\ these,\ halving\ per\ step"),
            ha="center",
            va="bottom",
            fontsize=15,
            color=SOFT,
        )

    # ── stage 6: three settings of the knob, three different answers ──────────
    if stage >= 6:
        cuts = [stops[k] for k in WALK_STOPS]
        _incl_gauges(ax, corpus, WALK_STOPS, cuts, y0=gauge_y0, stop_label_y=stop_label_y)
        ax.text(
            x0 + w / 2,
            conclusion_y,
            "and the sets nest: cut at 1, verify up to 4",
            ha="center",
            va="center",
            fontsize=17,
            color=INK,
        )

    return fig


#: The tilt figure stacks its two panels instead of setting them side by side,
#: which is what lets both be `INCL_PANEL_W` wide — the same panel the knob pair
#: draws, in the same place, so all three figures of the section share one
#: geometry. It also buys the comb its legibility: eleven cuts spanning a fifth
#: of the axis are 13px apart across a full-width panel and 8px apart across a
#: half-width one. The height that costs is height the figure wanted anyway; at
#: 12.1 units it very nearly fills a `bg right:70%` slot, where the side-by-side
#: version sat in a thin band across the middle of it.
TILT_CANVAS_H = 12.1
TILT_PANEL_H = 2.6

#: Which stops of the knob the comb on M₀'s panel draws. Every other one: at
#: twenty-one the strokes touch and the comb reads as a bar, and the claim the
#: figure makes is "a cut per stop", which a comb of eleven makes and a solid
#: block does not.
TILT_COMB = tuple(range(-10, 11, 2))

#: Which stops the *rate* rule's crossing is drawn at on the fold panel. Zero is
#: deliberately absent, and not for room: on populations this close in variance
#: the rate crossing at equal cost weights sits within a thousandth of the
#: midpoint, so a stem drawn there would land on the midpoint notch and invite
#: exactly the reading `mid_tilt` exists to refuse — that the shipped rule is
#: `rate`. What the figure needs from `rate` is its *displacement*, which is the
#: only thing `mid_tilt` takes from it.
TILT_RATE_STOPS = (-10, -5, 5, 10)

#: The grey caret that marks the one cut the inclusion-blind midpoint returns.
TILT_CARET_H = 0.26


def tilt_flow_fig() -> None:
    """Schematic of the tilt that gave the knob back — Part 2's repair, on the
    super-figure (#2865, #3218).

    Part 1 ended by replacing the blend with one fused fit, and that fit reads
    no cost weights: `mid` is the midpoint of two fitted component means, and
    the Inclusion knob arrives as a pair of cost weights it never looks at. So
    the knob went silent again — measured, not argued: one admitted set for the
    whole slider in **65,671 of 65,671** cell-steps across four environments, at
    up to **+0.18±0.02** regret away from inclusion 0
    (``docs/experiments/inclusion-cut-rule/REPORT.md``).

    The shipped rule (:data:`~vtscore.training.thresholds.FOLD_ANCHOR_CUT_RULE`,
    ``"mid_tilt"``) keeps the measured winner exactly where it was measured and
    borrows only the motion:
    ``q(k) = q_mid + (q_rate(k) - q_rate(0))`` in fold-quantile space. At
    inclusion 0 the bracket is *identically* zero — both terms are the same
    computation on the same fits — so the threshold is bit-for-bit the arm the
    anchor-mass run scored; everywhere else it inherits the rate-optimal
    crossing's tilt without inheriting its location.

    The drawing is that sentence, stacked: the rate crossing fans across the
    fold panel, the midpoint does not move, the formula carries the fan and not
    the location down to the model that applies it, and M₀'s panel grows a comb
    where it had one notch. Every number on it comes from a live
    :class:`~vtscore.training.thresholds.FoldAnchoredCut` over
    `calib-quantile-flow`'s own three populations, so the figure cannot drift
    from the estimator it claims to draw.
    """
    folds, final = _xquant_populations()
    last = _tilt_flow_stage(TILT_FLOW_STAGES, folds, final)
    box = tight_box(last)
    for stage in range(1, TILT_FLOW_STAGES):
        save(
            _tilt_flow_stage(stage, folds, final),
            OUT,
            f"calib-tilt-flow.build{stage}.png",
            column=SIDEBAR_WIDE,
            box=box,
        )
    save(last, OUT, "calib-tilt-flow.png", column=SIDEBAR_WIDE, box=box)


def _tilt_flow_stage(stage: int, folds: list, final: np.ndarray) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic."""
    fig, ax = _incl_figure(TILT_CANVAS_H)
    x0, w = INCL_PANEL_X0, INCL_PANEL_W

    fit, scores, anchors = folds[0]
    cut = _xquant_cut(folds, final)
    theta_mid = gmm_cut_from_fit(fit, "mid", 1.0, 1.0)[0]
    theta = {k: cut.threshold_at(k) for k in TILT_COMB}

    # ── layout ────────────────────────────────────────────────────────────────
    # Two panels on one axis, stacked, with the rule that carries the cut from
    # the upper to the lower one written in the gap between them.
    fold_top = TILT_CANVAS_H - (LABEL_GAP + CAP_16 + LABEL_GAP + CAP_16)
    fold_base = fold_top - TILT_PANEL_H
    mid_label_bottom = fold_base - 0.32 - LABEL_GAP - CAP_16

    formula_y = mid_label_bottom - OBJECT_GAP - 0.24
    caret_top = formula_y - 0.24 - OBJECT_GAP - CAP_16 - LABEL_GAP
    final_top = caret_top - TILT_CARET_H - OBJECT_GAP
    final_base = final_top - TILT_PANEL_H
    comb_label_y = final_base - 0.32 - LABEL_GAP
    caption_y = comb_label_y - CAP_16 - LABEL_GAP
    free_y = caption_y - CAP_16 - OBJECT_GAP - 0.24

    x_mid = x0 + float(theta[0]) * w

    # ── stage 1: the super-figure's conclusion, stacked into two rows ─────────
    _score_histogram(ax, x0, fold_base, w, TILT_PANEL_H, fit, scores, fill="class", mu_labels=False)
    _hump_marks(ax, x0, fold_base, w, anchors)
    ax.text(x0, fold_top + LABEL_GAP, _sub("M_1(D_{-1})"), ha="left", va="bottom", fontsize=16, color=INK)
    ax.text(
        x0,
        fold_top + LABEL_GAP + CAP_16 + LABEL_GAP,
        _sub("M_1(D_2)"),
        ha="left",
        va="baseline",
        fontsize=15,
        color=INK,
    )
    _theta_notch(ax, x0 + theta_mid * w, fold_base, "mid")

    _score_histogram(ax, x0, final_base, w, TILT_PANEL_H, None, final, fill="plain", mu_labels=False)
    ax.text(x0 + w, final_top + LABEL_GAP, _sub("M_0(D_{-1})"), ha="right", va="bottom", fontsize=16, color=INK)
    # Anchored on the panels' shared left edge and spanning the whole gap, so
    # it reads as one axis handing its cut down to the next rather than as a
    # mark floating between two drawings.
    _arrow(ax, (x0, fold_base - OBJECT_GAP), (x0, final_top + OBJECT_GAP))
    ax.plot([x_mid] * 2, [final_base - 0.32, final_base], color=INK, linewidth=2.2, zorder=6)
    ax.text(x_mid, comb_label_y, _sub(r"\theta_0"), ha="center", va="top", fontsize=16, color=INK)

    # ── stage 2: the knob arrives, and the midpoint does not hear it ──────────
    # A caret over M₀'s distribution, pointing at the one cut the fused fit
    # returns however far the slider is dragged. It stays on the final frame:
    # the comb the last stage draws is centred on it, because inclusion 0 is
    # where the two rules agree by construction.
    if stage >= 2:
        ax.add_patch(
            Polygon(
                [
                    (x_mid - TILT_CARET_H * 0.6, caret_top),
                    (x_mid + TILT_CARET_H * 0.6, caret_top),
                    (x_mid, caret_top - TILT_CARET_H),
                ],
                closed=True,
                facecolor=SOFT,
                edgecolor="none",
                zorder=6,
            )
        )
        ax.text(
            x_mid,
            caret_top + LABEL_GAP,
            "mid: this one cut, at every stop of the knob",
            ha="center",
            va="bottom",
            fontsize=15,
            color=SOFT,
        )

    # ── stage 3: the rate-optimal crossing does read the weights ─────────────
    if stage >= 3:
        for k in TILT_RATE_STOPS:
            rate = gmm_cut_from_fit(fit, "rate", *inclusion_cost_weights(k))[0]
            ax.plot(
                [x0 + rate * w] * 2,
                [fold_base, fold_top],
                color=SOFT,
                linewidth=2.0,
                linestyle=(0, (4, 3)),
                zorder=6,
            )
        ax.text(x0 + w, fold_top + LABEL_GAP, _sub("rate(k)"), ha="right", va="bottom", fontsize=15, color=SOFT)

    # ── stage 4: keep the location that was measured, borrow only the motion ──
    if stage >= 4:
        ax.text(
            x0 + w / 2,
            formula_y,
            _sub(r"q(k) = q_{mid} + \left(q_{rate}(k) - q_{rate}(0)\right)"),
            ha="center",
            va="center",
            fontsize=17,
            color=INK,
        )

    # ── stage 5: realized on M₀ — one notch becomes a comb ───────────────────
    if stage >= 5:
        for k in TILT_COMB:
            ax.plot(
                [x0 + float(theta[k]) * w] * 2,
                [final_base - 0.32, final_base],
                color=INK,
                linewidth=2.2,
                zorder=6,
            )
        # The comb spans a fifth of M₀'s axis, which is narrower than its two end
        # labels together; so the names are hung *outward* from its ends rather
        # than centred on them, and θ₀ keeps the middle.
        for k, ha, dx in ((10, "right", -LABEL_GAP), (-10, "left", LABEL_GAP)):
            ax.text(
                x0 + float(theta[k]) * w + dx,
                comb_label_y,
                _sub(rf"k = {k:+d}"),
                ha=ha,
                va="top",
                fontsize=16,
                color=INK,
            )
        ax.text(
            x_mid,
            caption_y,
            "a stop, a quantile, a cut — as before",
            ha="center",
            va="top",
            fontsize=15,
            color=SOFT,
        )

    # ── stage 6: and it costs nothing, because the fits do not move ───────────
    if stage >= 6:
        ax.text(
            INCL_CANVAS_W / 2,
            free_y,
            "the fits are inclusion-independent: re-cutting is arithmetic, not a retrain",
            ha="center",
            va="center",
            fontsize=17,
            color=INK,
        )

    return fig


#: Taller than the rest of the section, and the extra is spent at the bottom
#: rather than on content. This is the only Part 2 figure whose last line runs
#: along the foot of the canvas, and the theme prints the page number over a
#: `bg` figure's bottom-right corner (see `PAGE_NUMBER_CLEAR`): fitted into a
#: 70% slot this renders at ~52px per drawing unit, and the badge's ink reaches
#: about 0.56 units up from the canvas's bottom edge, so the closing line has to
#: sit clear of that. 13.26 units is the ceiling — past it the figure stops
#: being width-limited in the slot and the type starts shrinking.
ACQ_CANVAS_H = 13.0

#: The acquisition figure's panel and the ranking bar under it.
ACQ_PANEL_X0, ACQ_PANEL_W, ACQ_PANEL_H = 4.9, 10.3, 2.0
ACQ_GAUGE_H = 0.34

#: The zoom strip: how many items of the ranking it shows, and how tall a cell
#: is. Twenty-one is chosen from the drawing's own numbers rather than for
#: looks — one step of the knob moves this estimator's cut about eight items of
#: six thousand, so a window of twenty-one is the smallest that holds both cuts
#: *and* the gap between them at their true separation.
ACQ_ZOOM_CELLS = 21
ACQ_ZOOM_H = 0.62

#: How far the zoomed ranking hangs below the full one, leaving room for the
#: callout lines that tie the two together and for the pick's own name.
ACQ_ZOOM_DROP = 1.2

#: Which cells of the zoom carry votes rather than unlabeled media, as
#: `(index, is_good)`. Two of twenty-one: near the cut almost everything is
#: unlabeled, which is the whole reason there is something to ask about.
ACQ_ZOOM_VOTES = ((3, False), (17, True))


def acq_flow_fig() -> None:
    """Schematic of the second cut, and the loop it closes — Part 2's last
    figure (#2876, #3218).

    The same fitted estimator yields **two** thresholds, because the two jobs
    named at the top of the talk read a threshold differently. Reporting reads
    it as a decision boundary: everything above it comes back. Autopilot's
    ``hard`` pick reads it as a **rank position** — it ranks the corpus
    descending, finds the first position at or below the cut, and takes the
    unlabeled item nearest that position *by index*
    (:func:`vtscore.eval.al_strategies._hard_pick_by_index`, mirroring the app's
    ``autoSelectNext``). So the acquisition cut is taken
    :data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET` steps
    below the reporting one — a *negative* offset, which prices false alarms
    higher, **raises** the cut, moves it up the ranking, and returns more
    positives to vote on.

    The direction is the opposite of the intuition from the cost weights, so the
    figure draws it at its true size rather than at a legible one: on this
    corpus one step of the knob is about eight items in six thousand, which is
    why the ranking is zoomed rather than merely notched twice. The gap is small
    and it compounds — every pick it changes changes a vote, and every vote
    retrains the model, which is the arrow that closes the loop back to D₀.

    Measured record, and it is not a clean one:
    ``coco_val × siglip2`` found an interior optimum at −3 (positives per 100
    votes 4 → 18, average precision 0.696 → 0.817), ``visual_genome_m × siglip``
    rejected −3 against a +0.01 tolerance, and only −1 passed in both. The
    region-voting leg is void pending a re-run (#2943), and a supply-dependent
    offset is the open frontier (#2910). See
    ``docs/experiments/acquisition-inclusion/REPORT.md``.
    """
    folds, final = _xquant_populations()
    last = _acq_flow_stage(ACQ_FLOW_STAGES, folds, final)
    box = tight_box(last)
    for stage in range(1, ACQ_FLOW_STAGES):
        save(
            _acq_flow_stage(stage, folds, final),
            OUT,
            f"calib-acq-flow.build{stage}.png",
            column=SIDEBAR_WIDE,
            box=box,
        )
    save(last, OUT, "calib-acq-flow.png", column=SIDEBAR_WIDE, box=box)


def _acq_cell(ax: plt.Axes, x0: float, y0: float, w: float, h: float, kind: str, lw: float = 1.2) -> None:
    """One item of the zoomed ranking: unlabeled, or a vote already cast."""
    if kind == "unlabeled":
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=NEUTRAL_FILL, edgecolor=INK, linewidth=lw, zorder=3))
        return
    color, hatch = (GREEN, "//////") if kind == "good" else (RUST, "\\\\\\")
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=color, hatch=hatch, linewidth=0, zorder=3))
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="none", edgecolor=INK, linewidth=lw, zorder=4))


def _acq_flow_stage(stage: int, folds: list, final: np.ndarray) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the schematic."""
    fig, ax = _incl_figure(ACQ_CANVAS_H)
    x0, w = ACQ_PANEL_X0, ACQ_PANEL_W

    cut = _xquant_cut(folds, final)
    k_acq = acquisition_inclusion(0)
    q_report, q_acq = cut.quantile_at(0), cut.quantile_at(k_acq)
    theta_report = cut.threshold_at(0)

    # ── layout ────────────────────────────────────────────────────────────────
    # Two rows of evidence — the distribution the cut is a *number* in, and the
    # ranking it is a *position* in — with the loop's return routed down the left
    # margin past both. The row names go in that margin too: the zoom's callout
    # lines need the space under the ranking bar, and a name hung there would be
    # the thing they ran through.
    block_w, block_h = 3.0, 1.05
    block_x0 = 1.0
    block_top = ACQ_CANVAS_H - LABEL_GAP - CAP_16
    block_y0 = block_top - block_h
    row_y = block_y0 + block_h / 2

    m0x = x0 + 0.30 * w
    score_len = 1.2
    panel_top = row_y - MODEL_H / 2 - OBJECT_GAP - score_len - OBJECT_GAP - CAP_16 - LABEL_GAP
    y_base = panel_top - ACQ_PANEL_H

    theta_bottom = y_base - 0.32 - LABEL_GAP - CAP_16
    gauge_top = theta_bottom - OBJECT_GAP - GAUGE_STUB
    gauge_y0 = gauge_top - ACQ_GAUGE_H

    zoom_top = gauge_y0 - ACQ_ZOOM_DROP
    zoom_y0 = zoom_top - ACQ_ZOOM_H
    cell_w = w / ACQ_ZOOM_CELLS
    # The window is centred on the reporting cut and holds the acquisition cut at
    # its true distance: one step of the knob is about eight items in six
    # thousand here, and drawing that gap wider than it is would be the one lie
    # the figure could tell that actually matters.
    gap_cells = (q_acq - q_report) * final.size
    report_cell = (ACQ_ZOOM_CELLS - gap_cells) / 2
    zoom_report_x = x0 + report_cell * cell_w
    zoom_acq_x = zoom_report_x + gap_cells * cell_w
    # The app ranks descending and takes the first position at or below the cut,
    # so the pick is the item the cut falls *into*, not the one above it.
    pick_index = int(np.floor(report_cell + gap_cells))

    cut_label_bottom = zoom_y0 - 0.32 - LABEL_GAP - CAP_16
    rail_x = 0.45
    ask_y = cut_label_bottom - OBJECT_GAP - CAP_16 - LABEL_GAP
    conclusion_y = ask_y - OBJECT_GAP - 0.24

    def row_name(y: float, text: str, size: float = 15.0) -> None:
        ax.text(x0 - LABEL_GAP, y, text, ha="right", va="center", fontsize=size, color=SOFT)

    # ── stage 1: where the calibration talk left off ──────────────────────────
    ax.text(block_x0, block_top + LABEL_GAP, _sub("D_0"), ha="left", va="bottom", fontsize=16, color=INK)
    _data_block(ax, block_x0, block_y0, block_w, block_h)
    train_x = block_x0 + block_w + OBJECT_GAP
    _labeled_arrow(ax, (train_x, row_y), (m0x - MODEL_W / 2 - OBJECT_GAP, row_y), "train")
    _model_box(ax, m0x, row_y, "M_0")
    _labeled_arrow(
        ax,
        (m0x, row_y - MODEL_H / 2 - OBJECT_GAP),
        (m0x, panel_top + OBJECT_GAP + CAP_16 + LABEL_GAP),
        "score",
        z=2.1,
    )
    ax.text(x0 + w, panel_top + LABEL_GAP, _sub("M_0(D_{-1})"), ha="right", va="bottom", fontsize=16, color=INK)
    _score_histogram(ax, x0, y_base, w, ACQ_PANEL_H, None, final, fill="plain", mu_labels=False)
    _theta_notch(ax, x0 + theta_report * w, y_base, _sub(r"\theta_{report}"))
    _quantile_gauge(ax, x0, gauge_y0, w, ACQ_GAUGE_H, q_report, "")
    row_name(gauge_y0 + ACQ_GAUGE_H / 2, "the corpus, ranked")

    # ── stage 2: job one — the cut read as a decision boundary ───────────────
    if stage >= 2:
        brace_x0, brace_x1 = x0 + q_report * w, x0 + w
        brace_y = gauge_top + GAUGE_STUB + OBJECT_GAP
        ax.plot(
            [brace_x0, brace_x0, brace_x1, brace_x1],
            [brace_y - 0.12, brace_y, brace_y, brace_y - 0.12],
            color=INK,
            linewidth=1.6,
            zorder=5,
        )
        ax.text(
            (brace_x0 + brace_x1) / 2,
            brace_y + LABEL_GAP,
            "what you keep",
            ha="center",
            va="bottom",
            fontsize=15,
            color=INK,
        )

    # ── stage 3: job two reads the same number as a rank, so zoom in ─────────
    if stage >= 3:
        for i in range(ACQ_ZOOM_CELLS):
            kind = "unlabeled"
            for idx, good in ACQ_ZOOM_VOTES:
                if idx == i:
                    kind = "good" if good else "bad"
            _acq_cell(ax, x0 + i * cell_w, zoom_y0, cell_w, ACQ_ZOOM_H, kind)
        for target in (x0, x0 + w):
            ax.plot(
                [x0 + q_report * w, target],
                [gauge_y0, zoom_top],
                color=RULE,
                linewidth=1.4,
                zorder=0,
            )
        row_name(zoom_y0 + ACQ_ZOOM_H / 2, "zoomed at the cut")
        ax.plot([zoom_report_x] * 2, [zoom_y0 - 0.32, zoom_top], color=INK, linewidth=2.2, zorder=6)
        ax.text(
            zoom_report_x - LABEL_GAP,
            zoom_y0 - 0.32 - LABEL_GAP,
            _sub(r"\theta_{report}"),
            ha="right",
            va="top",
            fontsize=16,
            color=INK,
        )

    # ── stage 4: the second cut, one step of the knob further up the ranking ──
    if stage >= 4:
        ax.plot([zoom_acq_x] * 2, [zoom_y0 - 0.32, zoom_top], color=INK, linewidth=2.2, zorder=6)
        ax.text(
            zoom_acq_x + LABEL_GAP,
            zoom_y0 - 0.32 - LABEL_GAP,
            _sub(rf"\theta_{{acq}}\ \ (k = {k_acq})"),
            ha="left",
            va="top",
            fontsize=16,
            color=INK,
        )
        _acq_cell(ax, x0 + pick_index * cell_w, zoom_y0, cell_w, ACQ_ZOOM_H, "unlabeled", lw=3.2)
        ax.text(
            x0 + (pick_index + 0.5) * cell_w,
            zoom_top + LABEL_GAP,
            "ask about this one",
            ha="center",
            va="bottom",
            fontsize=15,
            color=INK,
        )

    # ── stage 5: the vote goes back to D₀, and the loop closes ───────────────
    # Routed down the left margin, as the loop schematic routes its own return:
    # a straight diagonal would cross both rows of evidence, and what the last
    # step has to say is that the threshold chooses what gets voted on, which a
    # clean rectangular return says more plainly than a shortcut.
    if stage >= 5:
        pick_cx = x0 + (pick_index + 0.5) * cell_w
        ax.plot(
            [pick_cx, pick_cx, rail_x, rail_x],
            [cut_label_bottom - OBJECT_GAP, ask_y, ask_y, row_y],
            color=INK,
            linewidth=1.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )
        _arrow(ax, (rail_x, row_y), (block_x0 - OBJECT_GAP, row_y))
        ax.text(
            (rail_x + pick_cx) / 2,
            ask_y + LABEL_GAP,
            "vote, and train again",
            ha="center",
            va="bottom",
            fontsize=15,
            color=INK,
        )
        # Centred on the *canvas*, not on the panel, and kept short: a line
        # hung off centre or run out to the panel's right edge widens the saved
        # figure past its canvas, and every label in the slot shrinks with it.
        ax.text(
            INCL_CANVAS_W / 2,
            conclusion_y,
            "the cut chooses the next question",
            ha="center",
            va="center",
            fontsize=17,
            color=INK,
        )

    return fig


def blend_schedule_fig() -> None:
    n = np.arange(0, 121)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for name, color, style, label, xy, ha, va in (
        ("prod", SOFT, (0, (4, 3)), "historical ramp:\npure x-cal by 20 votes", (40, 0.96), "left", "top"),
        ("cap50", BLUE, (0, (1, 1.6)), "cap50 — binary voting", (23, 0.42), "left", "top"),
        ("slow_cap50", BLUE, "solid", "slow_cap50 — region voting", (44, 0.55), "left", "bottom"),
    ):
        w = [
            get_schedule(name).weight(BlendContext(n_labels=int(k), n_good=int(k) // 2, n_bad=int(k) - int(k) // 2))
            for k in n
        ]
        ax.plot(n, w, color=color, linestyle=style, linewidth=2.4)
        ax.annotate(label, xy=xy, ha=ha, va=va, fontsize=15, color=color)
    ax.set_xlim(0, 122)
    ax.set_ylim(-0.02, 1.08)
    ax.set_yticks([0, 0.5, 1.0], ["pure\nGMM", "0.5", "pure\nx-cal"])
    ax.set_xlabel("votes")
    # Short enough not to overflow the (shortened) axes above the title notch;
    # the tick labels already read "pure GMM" to "pure x-cal", so the axis name
    # only has to name the quantity, not re-explain the ends.
    ax.set_ylabel("weight on the x-cal cut")
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    # Full-bleed: no in-figure title (the slide's headline is the title, and it
    # is drawn over this band), and the plot is pushed below TITLE_NOTCH_PX
    # rather than tight-cropped, so the reserved corner survives the write.
    fig.subplots_adjust(left=0.20, right=0.97, top=0.69, bottom=0.13)
    save(fig, OUT, "calib-blend-schedule.png", column=FULL_BLEED, tight=False)


def anchored_fig() -> None:
    # Haystack whose negatives are themselves bimodal: unanchored EM splits the
    # negative bulk and parks the cut inside it. A handful of votes re-identify
    # the components.
    rng = np.random.default_rng(7)
    neg = np.concatenate([rng.normal(0.18, 0.06, 1300), rng.normal(0.42, 0.07, 650)])
    pos = rng.normal(0.74, 0.08, 55)
    scores = np.clip(np.concatenate([neg, pos]), 0.0, 1.0)

    good_anchors = np.clip(rng.normal(0.72, 0.07, 6), 0, 1)
    bad_anchors = np.clip(np.concatenate([rng.normal(0.2, 0.05, 9), rng.normal(0.44, 0.05, 6)]), 0, 1)
    a_scores = np.concatenate([good_anchors, bad_anchors])
    a_labels = np.concatenate([np.ones_like(good_anchors), np.zeros_like(bad_anchors)])

    plain = fit_score_gmm(scores)
    anchored, prov = fit_anchored_score_gmm(scores, a_scores, a_labels, anchor_weight=30.0)
    assert plain is not None and anchored is not None and prov == "anchored"
    plain_mid = 0.5 * (plain.mu_lo + plain.mu_hi)
    anch_mid = 0.5 * (anchored.mu_lo + anchored.mu_hi)
    print(f"unanchored cut {plain_mid:.3f}  anchored cut {anch_mid:.3f}")

    x = np.linspace(0, 1, 400)
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.hist(scores, bins=48, density=True, color=NEUTRAL_FILL, zorder=1)
    ax.plot(
        x,
        plain.w_lo * gaussian(x, plain.mu_lo, plain.var_lo) + plain.w_hi * gaussian(x, plain.mu_hi, plain.var_hi),
        color=SOFT,
        linewidth=1.8,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    ax.plot(x, anchored.w_lo * gaussian(x, anchored.mu_lo, anchored.var_lo), color=RUST, linewidth=2.4, zorder=3)
    ax.plot(x, anchored.w_hi * gaussian(x, anchored.mu_hi, anchored.var_hi), color=GREEN, linewidth=2.4, zorder=3)
    ax.axvline(plain_mid, color=SOFT, linewidth=2.0, linestyle=(0, (4, 3)), zorder=4)
    ax.axvline(anch_mid, color=BLUE, linewidth=2.4, zorder=4)

    ymax = ax.get_ylim()[1]
    for s in bad_anchors:
        ax.plot([s, s], [-0.05 * ymax, 0.035 * ymax], color=RUST, linewidth=2.2, zorder=5, clip_on=False)
    for s in good_anchors:
        ax.plot([s, s], [-0.05 * ymax, 0.035 * ymax], color=GREEN, linewidth=2.2, zorder=5, clip_on=False)
    ax.annotate(
        "unanchored fit\ncuts inside the\nnegative bulk",
        xy=(plain_mid, ymax * 0.8),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        color=SOFT,
        fontsize=15,
    )
    ax.annotate(
        "anchored cut",
        xy=(anch_mid, ymax * 0.8),
        xytext=(10, 0),
        textcoords="offset points",
        ha="left",
        color=BLUE,
        fontsize=15,
    )
    ax.annotate(
        "votes, clamped one-hot at mass κ",
        xy=(0.44, -0.175 * ymax),
        ha="center",
        va="top",
        color=INK,
        fontsize=15,
        annotation_clip=False,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("detector score", labelpad=44)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_title("Votes re-identify the components", loc="left", pad=14)
    fig.tight_layout()
    save(fig, OUT, "calib-anchored-em.png")


def decomposition_fig() -> None:
    # docs/experiments/gmm-cut/REPORT-2881.md — the #2879 re-measure of #2836's
    # decomposition (region arm, ramp 6-20): total excess cost 0.0686.
    terms = [
        ("identification", 0.0057),
        ("prior / loss", 0.0111),
        ("Gaussian\nmisspecification", 0.0129),
        ("sim → test\ntransfer", 0.0389),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ys = np.arange(len(terms))
    for y, (name, v) in zip(ys, terms):
        emphasized = name.startswith("sim")
        ax.barh(y, v, height=0.55, color=BLUE if emphasized else "#9aa4b0", zorder=2)
        ax.annotate(
            f"{v:.4f}",
            xy=(v, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=15,
            color=BLUE if emphasized else SOFT,
            fontweight="bold" if emphasized else "normal",
        )
    ax.set_yticks(ys, [t[0] for t in terms])
    ax.set_xlim(0, 0.047)
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(left=False)
    # Full-bleed: the in-figure title is gone (the slide's headline says the
    # same thing, over this band), but the units line stays — it is the one
    # thing the bars do not say for themselves.
    ax.annotate(
        "excess cost vs the test oracle, region arm",
        xy=(0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=15,
        color=SOFT,
    )
    fig.subplots_adjust(left=0.28, right=0.97, top=0.645, bottom=0.05)
    save(fig, OUT, "calib-error-decomposition.png", column=FULL_BLEED, tight=False)


#: The loop schematic's canvas. Wider than the calibration schematics because
#: the cut forks left and right on the same row; the height matches theirs so a
#: 16pt label renders at the same size on every schematic in the deck.
LOOP_CANVAS = (13.6, FLOW_CANVAS_H)

#: How far the whole-corpus score line's own label sits above it. The
#: calibration schematics clear a check mark (`SCORE_LABEL_LIFT`); this line
#: carries plain grey ticks instead, because the corpus is unlabeled and
#: nothing on it is known to be Good or Bad.
LOOP_TICK_H = 0.16
LOOP_LABEL_LIFT = LOOP_TICK_H + LABEL_GAP


def vts_loop_fig() -> None:
    """The application loop the whole deck sits inside.

    Deliberately not in the notation the calibration schematics share: this is
    the slide that introduces the tool, so the corpus, the detector and the
    votes are named in words. What it *does* share is the vocabulary of shapes
    — a grey bar for unlabeled media, a green-over-rust hatched block for
    votes, an outlined box for a model, a number line with a cut on it — so
    that the later schematics are already half-read when they arrive.

    The one argument the figure makes is the fork under the cut: the same
    threshold decides what the search returns *and* which item the user is
    asked about next, which is why it is worth a talk. The build reveals those
    two arrows as separate steps.
    """
    final = _vts_loop_stage(LOOP_STAGES)
    box = tight_box(final)
    for stage in range(1, LOOP_STAGES):
        save(_vts_loop_stage(stage), OUT, f"vts-loop.build{stage}.png", column=SIDEBAR_WIDE, box=box)
    save(final, OUT, "vts-loop.png", column=SIDEBAR_WIDE, box=box)


def _vts_loop_stage(stage: int) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative) of the loop."""
    fig, ax = plt.subplots(figsize=tuple(c * FLOW_UNIT_PT / 72 for c in LOOP_CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, LOOP_CANVAS[0])
    ax.set_ylim(0, LOOP_CANVAS[1])
    ax.set_axis_off()

    arrow = functools.partial(_arrow, ax)
    labeled_arrow = functools.partial(_labeled_arrow, ax)

    # ── layout ────────────────────────────────────────────────────────────────
    # One spine down the middle (corpus, detector, scores, cut), then a fork.
    bx = 7.4
    pool_x0, pool_w, pool_y0, pool_h = 2.6, 9.6, 9.95, 0.85
    det_cy, det_w, det_h = 8.2, 2.9, 0.78
    score_len = 1.5

    score_tail = det_cy - det_h / 2 - OBJECT_GAP
    score_tip = score_tail - score_len
    # The arrow points at the score-line *group*, so it stops an object gap
    # above the group's topmost ink — which is the line's label, not the line.
    line_y = score_tip - (OBJECT_GAP + CAP_16 + LOOP_LABEL_LIFT)
    line_half = 3.5
    theta_x = bx + 0.1 * line_half

    # Both forks leave from directly under the cut's own label.
    fork = (theta_x, line_y - 0.32 - LABEL_GAP - CAP_16 - OBJECT_GAP)

    votes_x0, votes_w, votes_y0, votes_h = 2.6, 3.2, 0.85, 0.95
    votes_cx = votes_x0 + votes_w / 2
    keep_cx, keep_cy, keep_w, keep_h = 11.5, 1.6, 3.2, 0.8

    # ── stage 1: the corpus — everything the user has, none of it labelled ────
    ax.add_patch(
        Rectangle(
            (pool_x0, pool_y0),
            pool_w,
            pool_h,
            facecolor=NEUTRAL_FILL,
            edgecolor=INK,
            linewidth=1.6,
            zorder=2,
        )
    )
    ax.text(
        pool_x0 + pool_w / 2,
        pool_y0 + pool_h / 2,
        "everything you have, unlabeled",
        ha="center",
        va="center",
        fontsize=16,
        color=INK,
        zorder=3,
    )

    # ── stage 2: the detector — a small head trained on the votes so far ──────
    if stage >= 2:
        arrow((bx, pool_y0 - OBJECT_GAP), (bx, det_cy + det_h / 2 + OBJECT_GAP))
        ax.add_patch(
            Rectangle(
                (bx - det_w / 2, det_cy - det_h / 2),
                det_w,
                det_h,
                facecolor="white",
                edgecolor=INK,
                linewidth=1.6,
                zorder=3,
            )
        )
        ax.text(bx, det_cy, "detector", ha="center", va="center", fontsize=16, color=INK, zorder=4)

    # ── stage 3: it scores the whole corpus ──────────────────────────────────
    # Grey ticks, not checks and crosses: the corpus is unlabeled, so the shape
    # of the scores is all anyone has. The two mounds are what iteration 2 goes
    # on to fit.
    if stage >= 3:
        labeled_arrow((bx, score_tail), (bx, score_tip), "score")
        ax.plot([bx - line_half, bx + line_half], [line_y] * 2, color=INK, linewidth=1.8, zorder=2)
        ax.text(
            bx,
            line_y + LOOP_LABEL_LIFT,
            "the whole corpus, scored",
            ha="center",
            va="bottom",
            fontsize=16,
            color=INK,
        )
        rng = np.random.default_rng(0)
        draws = np.concatenate([rng.normal(-0.55, 0.22, 46), rng.normal(0.60, 0.17, 9)])
        for u in np.clip(draws, -0.97, 0.97):
            x = bx + u * line_half
            ax.plot([x, x], [line_y, line_y + LOOP_TICK_H], color=SOFT, linewidth=1.3, zorder=3)

    # ── stage 4: the cut ─────────────────────────────────────────────────────
    if stage >= 4:
        ax.plot([theta_x] * 2, [line_y - 0.32, line_y], color=BLUE, linewidth=2.6, zorder=4)
        ax.text(
            theta_x,
            line_y - 0.32 - LABEL_GAP,
            _sub(r"\theta"),
            ha="center",
            va="top",
            fontsize=16,
            color=BLUE,
        )

    # ── stage 5: job one — what the search gives back ────────────────────────
    if stage >= 5:
        labeled_arrow(fork, (keep_cx - keep_w / 2 + 0.7, keep_cy + keep_h / 2 + OBJECT_GAP), "keep")
        ax.add_patch(
            Rectangle(
                (keep_cx - keep_w / 2, keep_cy - keep_h / 2),
                keep_w,
                keep_h,
                facecolor="white",
                edgecolor=INK,
                linewidth=1.6,
                zorder=3,
            )
        )
        ax.text(keep_cx, keep_cy, "what you keep", ha="center", va="center", fontsize=16, color=INK, zorder=4)

    # ── stage 6: job two — which item you are asked about next ───────────────
    if stage >= 6:
        labeled_arrow(fork, (votes_cx, votes_y0 + votes_h + LABEL_GAP + CAP_16 + OBJECT_GAP), "ask next")
        _data_block(ax, votes_x0, votes_y0, votes_w, votes_h)
        ax.text(
            votes_cx,
            votes_y0 + votes_h + LABEL_GAP,
            "your votes",
            ha="center",
            va="bottom",
            fontsize=16,
            color=INK,
        )
        good_h = 0.42 * votes_h
        ax.text(
            votes_x0 + votes_w + LABEL_GAP,
            votes_y0 + votes_h - good_h / 2,
            "Good",
            ha="left",
            va="center",
            fontsize=15,
            color=GREEN,
        )
        ax.text(
            votes_x0 + votes_w + LABEL_GAP,
            votes_y0 + (votes_h - good_h) / 2,
            "Bad",
            ha="left",
            va="center",
            fontsize=15,
            color=RUST,
        )

    # ── stage 7: the vote retrains the detector, and it all goes round again ─
    # Routed as a rail down the left margin: a straight diagonal would cut
    # through the score line, and the point of the last step is that the loop
    # closes, which a clean rectangular return says more plainly.
    if stage >= 7:
        rail_x = 0.75
        rail_y = votes_y0 + votes_h / 2
        head_from = bx - det_w / 2 - OBJECT_GAP - 0.55
        ax.plot(
            [votes_x0 - OBJECT_GAP, rail_x, rail_x, head_from],
            [rail_y, rail_y, det_cy, det_cy],
            color=INK,
            linewidth=1.6,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=2,
        )
        arrow((head_from, det_cy), (bx - det_w / 2 - OBJECT_GAP, det_cy))
        ax.text(
            rail_x - LABEL_GAP,
            (rail_y + det_cy) / 2,
            "retrain",
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=15,
            color=INK,
        )

    return fig


if __name__ == "__main__":
    vts_loop_fig()
    xcal_flow_fig()
    gmm_flow_fig()
    blend_flow_fig()
    xsemi_flow_fig()
    xquant_flow_fig()
    knob_flow_fig()
    walk_flow_fig()
    tilt_flow_fig()
    acq_flow_fig()
    blend_schedule_fig()
    anchored_fig()
    decomposition_fig()
    print("wrote figures to", OUT)
