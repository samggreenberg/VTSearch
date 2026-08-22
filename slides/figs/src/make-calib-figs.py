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
from matplotlib.patches import Ellipse, FancyArrow, FancyArrowPatch, Polygon, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import LABEL_GAP_PT, OBJECT_GAP_PT, SIDEBAR, SIDEBAR_WIDE, save, tight_box  # noqa: E402

from vtscore.training.blend_schedules import BlendContext, get_schedule
from vtscore.training.thresholds import GmmFit1D, fit_anchored_score_gmm, fit_score_gmm

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
    same silhouette be re-filled between build stages — black while it is just
    "the shape of the data", then split in two and hatched once the fit claims
    which half is which — without the outline moving by a pixel.
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
    fit: "GmmFit1D",
    scores: np.ndarray,
    *,
    colored: bool,
) -> None:
    """The haystack's score histogram, drawn in the schematic's drawing units.

    A `bg`-slot schematic cannot host a real Axes without inheriting its own
    scales and margins, so the distribution is rasterised by hand into the
    rectangle `(x0, y_base, w, h)`: score 0-1 maps across `w`, and the density
    is scaled so the tallest bar is exactly `h`. The fitted component curves
    ride the same scaling, so curve and bars are directly comparable.

    `colored` splits stage 3 from stage 4 of the build, and the split is the
    argument of the whole slide. Stage 3 is the histogram in flat black: the
    shape of the data, which is all anyone actually has. Stage 4 keeps that
    silhouette exactly and re-fills it — split at the components' crossing,
    each half hatched with question marks in rust or green and traced by its
    fitted Gaussian. Everything that arrives in stage 4 is inference, and
    none of it has read a label.
    """
    density, edges = np.histogram(scores, bins=GMM_FLOW_BINS, range=(0.0, 1.0), density=True)
    sy = h / float(density.max())

    if not colored:
        black = _staircase(x0, y_base, w, sy, edges, density, 0, len(density) - 1)
        black.set(facecolor=INK, edgecolor="none", zorder=2)
        ax.add_patch(black)
        ax.plot([x0, x0 + w], [y_base] * 2, color=INK, linewidth=1.8, zorder=5)
        return

    # Where the fit stops calling a score Bad and starts calling it Good. The
    # two humps are split here, so between them the fill changes colour exactly
    # once and at the place the mixture itself puts the boundary.
    xs = np.linspace(0.0, 1.0, 600)
    lo_d = fit.w_lo * gaussian(xs, fit.mu_lo, fit.var_lo)
    hi_d = fit.w_hi * gaussian(xs, fit.mu_hi, fit.var_hi)
    crossing = int(np.argmax(hi_d > lo_d)) if (hi_d > lo_d).any() else len(xs)

    for lo_i, hi_i, mu, var, weight, color, name in (
        (0, crossing, fit.mu_lo, fit.var_lo, fit.w_lo, RUST, r"\mu_{lo}"),
        (crossing, len(xs), fit.mu_hi, fit.var_hi, fit.w_hi, GREEN, r"\mu_{hi}"),
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
            hump.set(facecolor="white", edgecolor="none", zorder=2)
            ax.add_patch(hump)
            _question_hatch(ax, hump, color, z=2.5)
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
        _score_histogram(ax, panel_x0, y_base, panel_w, panel_h, fit, scores, colored=stage >= 4)

    # ── stage 5: cut at the midpoint between the two claimed modes ────────────
    if stage >= 5:
        mid = 0.5 * (fit.mu_lo + fit.mu_hi)
        theta_x = panel_x0 + mid * panel_w
        # Carried up through the histogram, not just ticked below it: the cut's
        # whole claim is that it divides this distribution into the two classes,
        # and a tick under the baseline leaves that to be taken on trust.
        ax.plot([theta_x] * 2, [y_base - 0.32, y_base + panel_h], color=INK, linewidth=2.2, zorder=6)
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
      three cuts θ₁, θ₂ and θ_G are ticked at the same height and read as
      three answers to one question rather than as two unrelated pictures.
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
        _score_histogram(ax, panel_x0, y_base, panel_w, panel_h, fit, scores, colored=True)

    # ── stage 3: cut it where the two fitted components meet ──────────────────
    if stage >= 3:
        ax.plot([theta_g_x] * 2, [y_base - 0.32, y_base + panel_h], color=INK, linewidth=2.2, zorder=6)
        ax.text(theta_g_x, y_base - 0.32 - LABEL_GAP, _sub(r"\theta_G"), ha="center", va="top", fontsize=16, color=INK)

    # ── stage 4: split the votes and train a fold model on each half ──────────
    if stage >= 4:
        for mx, name in ((m1x, "M_1"), (m2x, "M_2")):
            model_box(mx, my, name)
            labeled_arrow((mx, train_tail_y), (mx, train_tail_y - fold_train_len), "train")
        for mx, sign in ((m1x, 1.0), (m2x, -1.0)):
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
    ax.set_ylabel("weight on the cross-calibration cut")
    ax.set_title("Measured schedules never hand over", loc="left", pad=14, fontsize=16)
    ax.grid(axis="y", color=RULE, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, OUT, "calib-blend-schedule.png")


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
    ax.set_title("Where the threshold error lives", loc="left", pad=34, fontsize=16)
    ax.annotate(
        "excess cost vs the test oracle, region arm",
        xy=(0, 1.0),
        xycoords="axes fraction",
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=15,
        color=SOFT,
    )
    fig.tight_layout()
    save(fig, OUT, "calib-error-decomposition.png")


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
    blend_schedule_fig()
    anchored_fig()
    decomposition_fig()
    print("wrote figures to", OUT)
