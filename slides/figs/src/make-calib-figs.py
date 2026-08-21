#!/usr/bin/env python
"""Mechanism figures for the calibration decks.

Uses matplotlib and vtscore, both already project dependencies, so a normal
checkout install is enough. Run from the repo root:

    python slides/figs/src/make-calib-figs.py

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
from matplotlib.patches import Ellipse, FancyArrow, FancyArrowPatch, Rectangle

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
            bad=[-2.25, -1.53, -0.81, -0.09],
            good=[0.81, 1.53, 2.25],
            theta_x=0.36,
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
            bad=[-2.25, -1.62, -0.98, 0.48],
            good=[0.0, 1.1, 1.82],
            theta_x=-0.48,
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

    `colored` splits stage 3 from stage 4 of the build: before it the bars are
    bare and the figure has asserted only "here is the shape"; after it the two
    fitted components are drawn over them in the same rust and green the rest
    of the deck uses for Bad and Good. That reveal *is* the claim the estimator
    makes — and, per the speaker notes, the one it cannot support, since
    nothing here has read a label.
    """
    density, edges = np.histogram(scores, bins=GMM_FLOW_BINS, range=(0.0, 1.0), density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_w = edges[1] - edges[0]
    sy = h / float(density.max())

    for c, d in zip(centers, density):
        ax.add_patch(
            Rectangle(
                (x0 + (c - bin_w / 2) * w, y_base),
                bin_w * w,
                d * sy,
                facecolor=NEUTRAL_FILL,
                edgecolor="none",
                zorder=2,
            )
        )

    if colored:
        xs = np.linspace(0.0, 1.0, 400)
        for mu, var, weight, color, name in (
            (fit.mu_lo, fit.var_lo, fit.w_lo, RUST, r"\mu_{lo}"),
            (fit.mu_hi, fit.var_hi, fit.w_hi, GREEN, r"\mu_{hi}"),
        ):
            ax.plot(x0 + xs * w, y_base + weight * gaussian(xs, mu, var) * sy, color=color, linewidth=2.4, zorder=3)
            peak = weight * gaussian(np.array([mu]), mu, var)[0] * sy
            ax.plot(
                [x0 + mu * w] * 2,
                [y_base, y_base + peak],
                color=color,
                linewidth=1.6,
                linestyle=(0, (2, 2)),
                zorder=4,
            )
            # The means are named because the closing line's formula is written
            # in terms of them; they sit under the line they drop to, a label
            # gap below it, and clear of the cut's own deeper tick.
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
    hay_x0, hay_top = bx - block_w / 2, 10.75
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
    return_y = 0.40
    theta_label_top = return_y + 0.23 + OBJECT_GAP + CAP_16
    y_base = theta_label_top + LABEL_GAP + 0.32

    panel_x0, panel_w, panel_h = hay_x0, 7.6, 2.3
    # The score arrow's angle is the figure's one real trade-off. Two arrows
    # leaving the same small box are always nearer each other than either is to
    # the box, and the shallower this one runs the more of its own shaft width
    # it turns towards the train arrow's head — so too shallow breaks the object
    # gap between them. But a steep arrow is a *short* arrow for a given drop,
    # and it has to be long enough to hold the word "score", which forces the
    # drop to grow and takes the height straight out of the histogram — leaving
    # a void where the flow crosses back over the figure. This angle is the
    # shallowest that still clears the head, which is also the one that spends
    # the least height and sweeps the void.
    tip = (m0x - 1.6, y_base + panel_h + OBJECT_GAP + CAP_16 + LABEL_GAP)

    labeled_arrow = functools.partial(_labeled_arrow, ax)

    # ── stage 1: the unlabeled haystack ───────────────────────────────────────
    # "unlabeled" sits where D₀'s Good/Bad sit, because it answers the same
    # question about the same slot: what the classes in this block are.
    _haystack_block(ax, hay_x0, hay_y0, hay_w, hay_h)
    _disc_label(ax, bx, hay_y0 + hay_h / 2, "D_{-1}")
    ax.text(
        hay_x0 - LABEL_GAP,
        hay_y0 + hay_h / 2,
        "unlabeled",
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


if __name__ == "__main__":
    xcal_flow_fig()
    gmm_flow_fig()
    blend_schedule_fig()
    anchored_fig()
    decomposition_fig()
    print("wrote figures to", OUT)
