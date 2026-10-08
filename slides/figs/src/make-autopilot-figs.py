#!/usr/bin/env python
"""Figures for the Navigation slides: what Autopilot does, and when it says stop (#4610).

Run from the repo root:

    python slides/figs/src/make-autopilot-figs.py

Autopilot picks every item a user is shown in the Train window, in six steps,
and it tells them when they can stop. The steps are
`AutopilotStateService.checkPhaseTransition`; the stop is three lights the
labeling status computes, Smart (`vtscore/detectors/cost_trend.py`), Stable
(`vtscore/detectors/stability.py`) and Span (the coverage atlas), all green.

Schematic inputs, real code, as in `make-test-figs.py`:

* `autopilot-steps` walks the harness's port of the phase machine
  (`vtscore.eval.autopilot_flow.next_phase`, pinned to the app by
  `scripts/check-eval-app-sync.py`) through the six steps, and every target it
  prints is that module's constant.
* `autopilot-smart` runs the shipped `smart_status_from_costs` on a planted
  session's error costs, and asserts the lights the slide draws.
* `autopilot-stable` runs the shipped `count_flips` on two planted retrains,
  and asserts the flips the slide counts.
* `autopilot-span` counts the walk's run over the first 40 cells of an atlas,
  as `CoverageAtlas.coverage_level` does, and reads the light from the
  harness's port of the Span rule.
* `autopilot-done` is the hand-off, in the app's own words
  (`autopilot-panel.component.ts`).

The last figure, `autopilot-stop`, is the only measurement: where the stop
fired in today's app, at each radio's beta, and what clicking on bought. It is
re-plotted from the 2026-10-08 Binary Photo review's committed tables
(`docs/experiments/2026-10-08-state-of-the-app-binary-photo/`:
`stopping_by_preset.csv` and `objective_by_click.csv`). `EXPECT` pins the numbers the
notes quote.

Every canvas is the whole 1280x720 slide at 100 px to the inch, saved at its
declared bounds, so every stage of a build shares one framing by construction
and `save()` checks the title notch against exactly what the slide shows.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

# Ensure the repo root is importable no matter the cwd: ``python
# slides/figs/src/x.py`` only puts the script's own dir on sys.path.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save, spread_labels  # noqa: E402

from vtscore.detectors.cost_trend import (  # noqa: E402
    SMART_FLAT_THRESHOLD,
    SMART_SLOPE_T,
    SMART_WINDOW,
    smart_status_from_costs,
)
from vtscore.detectors.stability import (  # noqa: E402
    STABLE_BAND_STD_FRACTION,
    STABLE_MAX_THRESHOLD,
    STABLE_RATE_THRESHOLD,
    STABLE_WINDOW,
    ScoredSnapshot,
    count_flips,
)
from vtscore.eval.autopilot_flow import (  # noqa: E402
    BAD_TARGET,
    GOOD_TARGET,
    MORE_DRY_RUN,
    MORE_TARGET,
    SPAN_GREEN_DEFAULT,
    SPAN_YELLOW,
    next_phase,
    span_status,
)

OUT = Path(__file__).resolve().parent.parent
#: The State of the App review the measured slide reads: Binary Photo on COCO Better, one session set per radio.
REPORT_SOTA = _REPO_ROOT / "docs" / "experiments" / "2026-10-08-state-of-the-app-binary-photo"

# The deck's palette (`make-calib-figs.py`, `themes/vtsearch.css`): blue is the
# line, red the losing arm and a Bad, green a Good. The lights borrow red and
# green from it and add an amber, and every light on a slide is named in words
# beside it, so no state is told by hue alone.
BLUE = "#0b5fa5"
RED = "#b91c1c"
GREEN = "#0d8a5f"
AMBER = "#c98a06"
RULE = "#d8dee6"
UNLABELED_FILL = "#dae0e8"
#: The ambiguity band around the line: a tint of the line's own blue, since it
#: is the line's band.
BAND_FILL = "#d6e4f2"
LIGHT = {"red": RED, "yellow": AMBER, "green": GREEN}

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

#: The canvas: the whole slide, at 100 slide pixels to the inch, in drawing
#: units of one inch. 15pt is the smallest type here: 20.8px on the slide.
W, H = 12.8, 7.2
#: Where a statement over the drawing starts: right of the title notch
#: (`slide_figure.TITLE_NOTCH_PX`, 60-360 px across), with a margin.
STATEMENT_X, STATEMENT_Y = 4.0, 6.35
#: A chart slide's numbers stack under the notch in this column (figure
#: fractions), as the Test slides' do.
SIDE_X, SIDE_TOP = 0.04, 0.70


# ── shared drawing ───────────────────────────────────────────────────────────


def _canvas() -> tuple[Figure, plt.Axes]:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    return fig, ax


def _save(stages: int, draw, name: str) -> None:
    """Every stage of a build at the canvas's declared bounds, the last as the slide's own figure."""
    for stage in range(1, stages):
        save(draw(stage), OUT, f"{name}.build{stage}.png", column=FULL_BLEED, tight=False)
    save(draw(stages), OUT, f"{name}.png", column=FULL_BLEED, tight=False)


def _statement(ax: plt.Axes, words: str) -> None:
    ax.text(STATEMENT_X, STATEMENT_Y, words, ha="left", va="center", fontsize=19)


def _side_block(fig: Figure, lines: list[tuple[str, str, float]], top: float = SIDE_TOP) -> float:
    """A column of ``(words, style, gap before)`` lines under the notch; returns the last line's height.

    ``head`` is a bold heading in ink and ``body`` plain ink indented under it.
    """
    y = top
    for words, style, gap in lines:
        y -= gap
        if style == "body":
            fig.text(SIDE_X + 0.015, y, words, ha="left", va="center", fontsize=16)
        else:
            fig.text(SIDE_X, y, words, ha="left", va="center", fontsize=17, fontweight="bold")
    return y


def _light(ax: plt.Axes, x: float, y: float, colour: str, r: float = 0.13) -> None:
    """One indicator light: a disc in the light's colour, and a white tick on green, as the app draws it."""
    ax.add_patch(Circle((x, y), r, facecolor=LIGHT[colour], edgecolor="none", zorder=6))
    if colour == "green":
        ax.plot(
            [x - 0.55 * r, x - 0.12 * r, x + 0.58 * r],
            [y + 0.02 * r, y - 0.42 * r, y + 0.5 * r],
            color="white",
            linewidth=2.4,
            solid_capstyle="round",
            zorder=7,
        )


def _disc(ax: plt.Axes, x: float, y: float, number: int, r: float = 0.22) -> None:
    ax.add_patch(Circle((x, y), r, facecolor=INK, edgecolor="none", zorder=6))
    ax.text(x, y, f"{number}", ha="center", va="center", fontsize=16, color="white", fontweight="bold", zorder=7)


def _line(ax: plt.Axes, x: float, y0: float, y1: float) -> None:
    """The line, in the palette's blue: the one shipped decision on every drawing."""
    ax.plot([x, x], [y0, y1], color=BLUE, linewidth=2.8, zorder=6, solid_capstyle="butt")


# ── Flight Plan: the six steps ───────────────────────────────────────────────

STEPS_STAGES = 4

#: The rows, top to bottom: the step's name as the panel's short label shows
#: it, what it asks about, the drawing in the middle column, and what ends it.
#: The targets are the harness's constants, which mirror the app's.
STEPS = (
    ("good", "Good", "the top of your query", "query-top", f"{GOOD_TARGET} Goods"),
    ("bad", "Bad", "at your query's line", "query-line", f"{BAD_TARGET} Bads"),
    ("more", "More", "the top, again", "query-top", f"{MORE_TARGET} Goods, or {MORE_DRY_RUN} in a row without one"),
    ("hard", "Boundary", "at even odds", "detector-odds", "Smart and Stable both green"),
    ("new", "Diversity", "a cell nobody voted in", "atlas", "Span green as well"),
    ("done", "Done", "every light green", "lights", "you say: keep going, or leave"),
)
STEP_ROWS_Y = (4.95, 4.1, 3.25, 2.4, 1.55, 0.7)
#: Which row each build stage reveals up to: the opening on the query, then
#: one step a stage.
STEP_STAGE_ROWS = {1: 3, 2: 4, 3: 5, 4: 6}
NAME_X, PICTURE_X0, PICTURE_X1, ENDS_X = 1.25, 4.3, 7.6, 7.95
STRIP_H = 0.34
#: Where each ranking's line falls along its strip (best on the right).
QUERY_LINE, DETECTOR_LINE = 0.62, 0.68
#: Where Boundary asks on the detector's strip: the first item, best first, the
#: line's own fit calls even odds (#3546, #4632). Inside the line, as it is at
#: the middle radio every detector starts on; at β ¼ the two nearly meet.
DETECTOR_ODDS = 0.80


def _check_step_order() -> None:
    """The six steps in the order the phase machine takes them, from its own rule.

    A session walked the way the slide narrates it: the counts reach each
    target in turn, then Smart and Stable go green, then Span.
    """
    many = math.inf
    walk = (
        dict(good_count=0, bad_count=0, smart="red", stable="red", span="red"),
        dict(good_count=GOOD_TARGET, bad_count=0, smart="red", stable="red", span="red"),
        dict(good_count=GOOD_TARGET, bad_count=BAD_TARGET, smart="red", stable="red", span="red"),
        dict(good_count=MORE_TARGET, bad_count=9, smart="yellow", stable="red", span="red", more_done=True),
        dict(good_count=24, bad_count=30, smart="green", stable="green", span="yellow", more_done=True),
        dict(good_count=30, bad_count=50, smart="green", stable="green", span="green", more_done=True),
    )
    phases = tuple(next_phase(remaining_unlabeled=many, **state) for state in walk)
    assert phases == tuple(step[0] for step in STEPS), phases


def steps_fig() -> None:
    """The six steps: what each asks about, and what ends it.

    Four stages: the opening three steps, all on the query's own ranking, with
    no detector shown yet; then Boundary, the first step on the detector's
    ranking; then Diversity, off the atlas; then Done.
    """
    _check_step_order()
    _save(STEPS_STAGES, _steps_stage, "autopilot-steps")


def _strip(ax: plt.Axes, y: float, words: str, line_at: float) -> float:
    """A ranking as a strip, best on the right, its line drawn; returns the line's x."""
    width = PICTURE_X1 - PICTURE_X0
    ax.add_patch(Rectangle((PICTURE_X0, y - STRIP_H / 2), width, STRIP_H, facecolor=UNLABELED_FILL,
                           edgecolor="none", zorder=2))  # fmt: skip
    ax.text(PICTURE_X0 + 0.1, y, words, ha="left", va="center", fontsize=15, color=SOFT, zorder=3)
    x = PICTURE_X0 + line_at * width
    _line(ax, x, y - STRIP_H / 2 - 0.08, y + STRIP_H / 2 + 0.08)
    return x


def _pick(ax: plt.Axes, x: float, y: float) -> None:
    """Where the step asks: a pointer onto the strip from above."""
    ax.plot(x, y + STRIP_H / 2 + 0.2, marker="v", markersize=14, color=INK, zorder=7, linestyle="none")


def _mini_tree(ax: plt.Axes, y: float) -> None:
    """Three levels of an atlas, the first cell nobody has voted in filled."""
    levels = (1, 3, 9)
    ys = (y + 0.27, y, y - 0.27)
    span = PICTURE_X1 - PICTURE_X0 - 0.4
    xs = []
    for count in levels:
        step = span / count
        xs.append([PICTURE_X0 + 0.2 + step * (i + 0.5) for i in range(count)])
    for level in range(1, len(levels)):
        for i, x in enumerate(xs[level]):
            parent = xs[level - 1][i // 3]
            ax.plot([parent, x], [ys[level - 1], ys[level]], color=RULE, linewidth=1.4, zorder=2)
    # Voted in, filled as the Span slide fills them: the root, its first two
    # children, and three cells under those. The walk's next cell is ringed.
    covered = {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 4)}
    first_empty = (1, 2)
    for level, row in enumerate(xs):
        for i, x in enumerate(row):
            face = INK if (level, i) in covered else "white"
            ax.add_patch(Circle((x, ys[level]), 0.075, facecolor=face, edgecolor=SOFT, linewidth=1.2, zorder=4))
    ax.add_patch(Circle((xs[1][first_empty[1]], ys[1]), 0.13, facecolor="none", edgecolor=INK, linewidth=2.4,
                        zorder=5))  # fmt: skip


def _steps_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    _statement(ax, "Autopilot picks every item you are shown")
    centre = (PICTURE_X0 + PICTURE_X1) / 2
    ax.text(centre, 5.62, "where it asks", ha="center", va="center", fontsize=15, color=SOFT)
    ax.text(ENDS_X, 5.62, "until", ha="left", va="center", fontsize=15, color=SOFT)
    for row in range(STEP_STAGE_ROWS[stage]):
        _phase, name, asks, picture, ends = STEPS[row]
        y = STEP_ROWS_Y[row]
        _disc(ax, 0.85, y, row + 1)
        ax.text(NAME_X, y + 0.14, name, ha="left", va="center", fontsize=18, fontweight="bold")
        ax.text(NAME_X, y - 0.2, asks, ha="left", va="center", fontsize=15, color=SOFT)
        if picture == "query-top":
            _strip(ax, y, "your query", QUERY_LINE)
            _pick(ax, PICTURE_X1 - 0.12, y)
        elif picture == "query-line":
            _pick(ax, _strip(ax, y, "your query", QUERY_LINE), y)
        elif picture == "detector-odds":
            _strip(ax, y, "the detector", DETECTOR_LINE)
            _pick(ax, PICTURE_X0 + DETECTOR_ODDS * (PICTURE_X1 - PICTURE_X0), y)
        elif picture == "atlas":
            _mini_tree(ax, y)
        else:
            for k, word in enumerate(("Smart", "Stable", "Span")):
                x = PICTURE_X0 + 0.2 + k * 1.13
                _light(ax, x, y, "green")
                ax.text(x + 0.2, y, word, ha="left", va="center", fontsize=16)
        ax.text(ENDS_X, y, ends, ha="left", va="center", fontsize=16, fontweight="bold" if row == 5 else "normal")
    return fig


# ── Diminishing Returns: Smart ───────────────────────────────────────────────

SMART_STAGES = 2
#: A planted session's error cost, one point a retrain: falling fast, then
#: flat with the step-to-step scatter a real labelset gives it.
SMART_RETRAINS = 30
SMART_SEED = 7
#: The two windows the slide fits, as the last retrain each one ends on.
SMART_EARLY_END, SMART_LATE_END = 14, 30


def _smart_costs() -> np.ndarray:
    rng = np.random.default_rng(SMART_SEED)
    t = np.arange(1, SMART_RETRAINS + 1)
    return 0.16 + 0.55 * np.exp(-t / 5.5) + rng.normal(0.0, 0.008, size=t.size)


def _smart_window(costs: np.ndarray, end: int) -> tuple[np.ndarray, dict]:
    """The last ``SMART_WINDOW`` retrains up to *end* (1-based), and the shipped Smart status over them."""
    window = costs[end - SMART_WINDOW : end]
    return window, smart_status_from_costs(list(window), good=10, bad=10)


def smart_fig() -> None:
    """Smart: the error cost of the last ten detectors, and the trend fitted through it.

    Two stages: early in a session, the window is still falling (yellow); later,
    it has levelled off (green).
    """
    # The notes say "two standard errors" and "the 1.5% line".
    assert (SMART_SLOPE_T, SMART_FLAT_THRESHOLD) == (2.0, -0.015), (SMART_SLOPE_T, SMART_FLAT_THRESHOLD)
    costs = _smart_costs()
    early, late = _smart_window(costs, SMART_EARLY_END), _smart_window(costs, SMART_LATE_END)
    assert early[1]["status"] == "yellow", early[1]
    assert late[1]["status"] == "green" and not late[1]["drift_within_noise"], late[1]
    _save(SMART_STAGES, lambda stage: _smart_stage(stage, costs), "autopilot-smart")


def _fit_line(window: np.ndarray, x0: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(x0, x0 + window.size)
    slope, intercept = np.polyfit(xs, window, 1)
    return xs, slope * xs + intercept


def _smart_stage(stage: int, costs: np.ndarray) -> Figure:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.40, 0.14, 0.56, 0.62))
    ax.set_xlim(0.3, SMART_RETRAINS + 0.7)
    ax.set_ylim(0.0, SMART_Y_TOP)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    ax.set_xlabel("retrain")
    ax.set_ylabel("mistakes, FPR + FNR")
    fig.text(0.40, 0.86, "Is the detector still getting better?", ha="left", va="center", fontsize=19)
    overlay = _overlay(fig)
    _side_block(
        fig,
        [
            ("Each point", "head", 0.0),
            ("one retrain's detector,", "body", 0.055),
            ("scored on today's votes", "body", 0.05),
            ("The fit", "head", 0.085),
            (f"through the last {SMART_WINDOW}", "body", 0.055),
        ],
    )
    shown = SMART_EARLY_END if stage == 1 else SMART_RETRAINS
    xs = np.arange(1, shown + 1)
    ax.plot(xs, costs[:shown], linestyle="none", marker="o", markersize=8, color=INK, zorder=4)
    windows = [(SMART_EARLY_END, 0.56)] + ([(SMART_LATE_END, 0.36)] if stage >= 2 else [])
    for end, label_y in windows:
        window, status = _smart_window(costs, end)
        x0 = end - SMART_WINDOW + 1
        ax.add_patch(Rectangle((x0 - 0.45, 0.0), SMART_WINDOW - 0.1, SMART_Y_TOP, facecolor="#eef1f5",
                               edgecolor="none", zorder=1))  # fmt: skip
        fx, fy = _fit_line(window, x0)
        ax.plot(fx, fy, color=INK, linewidth=2.6, zorder=5)
        cx, cy = _to_canvas(fig, ax, x0 + (SMART_WINDOW - 1) / 2, label_y)
        _labelled_light(overlay, cx, cy, status["status"], _signed(100 * status["slope"]) + "% a retrain")
    if stage >= 2:
        _side_block(
            fig,
            [
                ("Green", "head", 0.0),
                (f"flatter than {_signed(100 * SMART_FLAT_THRESHOLD)}% a retrain,", "body", 0.055),
                ("or a fall smaller than", "body", 0.05),
                ("its own scatter", "body", 0.05),
            ],
            top=SIDE_TOP - 0.36,
        )
    return fig


#: The top of the Smart chart's cost axis: clear of the session's first point.
SMART_Y_TOP = 0.7


def _signed(value: float) -> str:
    """A signed number with a real minus sign, one decimal."""
    return f"{value:+.1f}".replace("-", "\u2212")


def _overlay(fig: Figure) -> plt.Axes:
    """An invisible axes over the whole canvas, in inches, for drawing on top of a chart."""
    overlay = fig.add_axes((0, 0, 1, 1), zorder=10)
    overlay.set_xlim(0, W)
    overlay.set_ylim(0, H)
    overlay.set_axis_off()
    return overlay


def _to_canvas(fig: Figure, ax: plt.Axes, x: float, y: float) -> tuple[float, float]:
    """A point in *ax*'s data, in the canvas's inches."""
    fx, fy = fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    return fx * W, fy * H


def _labelled_light(overlay: plt.Axes, cx: float, cy: float, colour: str, words: str) -> None:
    """A light with its colour's name beside it, and *words* under both, centred on (*cx*, *cy*)."""
    name = overlay.text(cx + 0.15, cy, colour, ha="center", va="center", fontsize=17, fontweight="bold")
    overlay.figure.canvas.draw()
    box = name.get_window_extent().transformed(overlay.transData.inverted())
    _light(overlay, box.x0 - 0.2, cy, colour)
    overlay.text(cx, cy - 0.36, words, ha="center", va="center", fontsize=16)


# ── Change of Heart: Stable ──────────────────────────────────────────────────

STABLE_STAGES = 4
#: The planted pool both retrains score: every item's score under the last
#: retrain, and how far this one moved it. Most barely move; four wobble across
#: the line from just beside it; two cross it from well clear on both sides.
STABLE_SEED = 3
STABLE_POOL = 36
STABLE_CUT = 0.5
#: Two retrains drawn one above the other; scores run left to right, best on
#: the right, across this span of the canvas.
STABLE_Y_LAST, STABLE_Y_THIS = 4.55, 1.95
STABLE_X0, STABLE_X1 = 4.9, 12.2
#: The column the counts and the rule stand in, left of the two retrains.
STABLE_TEXT_X = 0.7


def _stable_retrains() -> tuple[ScoredSnapshot, ScoredSnapshot]:
    rng = np.random.default_rng(STABLE_SEED)
    last = np.sort(rng.uniform(0.04, 0.96, size=STABLE_POOL))
    move = rng.normal(0.0, 0.012, size=STABLE_POOL)
    this = last + move
    # Keep everything else on its own side, so the flips are only the planted ones.
    this = np.where(last >= STABLE_CUT, np.maximum(this, STABLE_CUT + 0.002), np.minimum(this, STABLE_CUT - 0.002))
    near = np.argsort(np.abs(last - STABLE_CUT))
    for k, i in enumerate(near[:4]):
        # The four nearest the line wobble across it, landing just the other side.
        this[i] = STABLE_CUT - 0.03 if last[i] >= STABLE_CUT else STABLE_CUT + 0.03 + 0.01 * k
    above = [i for i in np.argsort(last) if last[i] > STABLE_CUT + 0.2]
    below = [i for i in np.argsort(last) if last[i] < STABLE_CUT - 0.2]
    # Two cross from well clear: one Good the detector now calls Bad, and back.
    this[above[1]] = 0.24
    this[below[-2]] = 0.79
    return (
        ScoredSnapshot.from_scores(dict(enumerate(last)), STABLE_CUT),
        ScoredSnapshot.from_scores(dict(enumerate(this)), STABLE_CUT),
    )


def stable_fig() -> None:
    """Stable: two retrains' calls on the same pool, the items that changed side, and which of those count.

    Four stages: the two retrains, every item joined to itself; the items that
    changed side; the band around the line, and the flips clear of it both
    times, the only ones that count; and the rule over the last ten retrains.
    """
    # The notes say "a quarter of the scores' spread either side".
    assert STABLE_BAND_STD_FRACTION == 0.25, STABLE_BAND_STD_FRACTION
    last, this = _stable_retrains()
    flips, confident = count_flips(last, this)
    assert (flips, confident) == (6, 2), (flips, confident)
    _save(STABLE_STAGES, lambda stage: _stable_stage(stage, last, this), "autopilot-stable")


def _score_x(score: float) -> float:
    return STABLE_X0 + score * (STABLE_X1 - STABLE_X0)


def _stable_stage(stage: int, last: ScoredSnapshot, this: ScoredSnapshot) -> Figure:
    fig, ax = _canvas()
    _statement(ax, "Does the detector still change its mind?")
    _stable_retrain_axes(ax, stage, last, this)
    flipped, confident = _stable_items(ax, last, this)
    if stage >= 2:
        for x0, x1 in flipped:
            ax.plot([x0, x1], [STABLE_Y_LAST, STABLE_Y_THIS], color=SOFT, linewidth=2.6, zorder=5)
        ax.text(STABLE_TEXT_X, 4.7, f"{len(flipped)} changed side", ha="left", va="center", fontsize=17,
                fontweight="bold")  # fmt: skip
    if stage >= 3:
        for x0, x1 in confident:
            ax.plot([x0, x1], [STABLE_Y_LAST, STABLE_Y_THIS], color=INK, linewidth=4.2, zorder=6)
        for k, words in enumerate((f"{len(confident)} were clear of the", "band both times:", "only those count")):
            ax.text(STABLE_TEXT_X, 4.15 - 0.35 * k, words, ha="left", va="center", fontsize=17, fontweight="bold")
    if stage >= 4:
        _stable_rule(ax)
    return fig


def _stable_retrain_axes(ax: plt.Axes, stage: int, last: ScoredSnapshot, this: ScoredSnapshot) -> None:
    """The two retrains' score axes, each with its line, and from stage 3 its band."""
    for y, snap, words, words_y in (
        (STABLE_Y_LAST, last, "last retrain", STABLE_Y_LAST + 0.45),
        (STABLE_Y_THIS, this, "this retrain", STABLE_Y_THIS - 0.45),
    ):
        ax.plot([STABLE_X0, STABLE_X1], [y, y], color=SOFT, linewidth=1.4, zorder=2)
        ax.text(STABLE_X0, words_y, words, ha="left", va="center", fontsize=16, color=SOFT)
        if stage >= 3:
            x0, x1 = _score_x(snap.threshold - snap.band), _score_x(snap.threshold + snap.band)
            ax.add_patch(Rectangle((x0, y - 0.24), x1 - x0, 0.48, facecolor=BAND_FILL, edgecolor="none", zorder=1))
        _line(ax, _score_x(snap.threshold), y - 0.3, y + 0.3)
    ax.text(STABLE_X1, STABLE_Y_LAST + 0.45, "best", ha="right", va="center", fontsize=15, color=SOFT)


def _stable_items(
    ax: plt.Axes, last: ScoredSnapshot, this: ScoredSnapshot
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Every item joined to itself; returns the ends of the flips, and of the confident ones among them."""
    flipped, confident = [], []
    for cid in sorted(this.scores):
        x0, x1 = _score_x(last.scores[cid]), _score_x(this.scores[cid])
        ax.plot([x0, x1], [STABLE_Y_LAST, STABLE_Y_THIS], color=RULE, linewidth=1.2, zorder=3)
        for x, y in ((x0, STABLE_Y_LAST), (x1, STABLE_Y_THIS)):
            ax.plot(x, y, marker="o", markersize=7, color=SOFT, zorder=4, linestyle="none")
        if last.predicted(cid) != this.predicted(cid):
            flipped.append((x0, x1))
            if last.confident(cid) and this.confident(cid):
                confident.append((x0, x1))
    return flipped, confident


def _stable_rule(ax: plt.Axes) -> None:
    """Stable's green, in words, from the shipped constants."""
    lines = (
        ("Green, over the last", True),
        (f"{STABLE_WINDOW} retrains:", True),
        ("flips that count average", False),
        (f"under {100 * STABLE_RATE_THRESHOLD:.1f}% of the pool,", False),
        (f"never {100 * STABLE_MAX_THRESHOLD:.0f}% in one;", False),
        ("and all flips have", False),
        ("stopped falling", False),
    )
    for k, (words, bold) in enumerate(lines):
        ax.text(STABLE_TEXT_X if bold else STABLE_TEXT_X + 0.15, 2.75 - 0.35 * k, words, ha="left", va="center",
                fontsize=16, fontweight="bold" if bold else "normal")  # fmt: skip


# ── Lay of the Land: Span ────────────────────────────────────────────────────

SPAN_STAGES = 4
#: The atlas the slide draws: three children a cell, the first four levels,
#: which are exactly the 40 cells Span's default green asks for.
SPAN_K, SPAN_LEVELS = 3, 4
#: The whole atlas the light is computed against: thousands of cells, as a
#: COCO-sized collection builds, so the green bar is the default and not the
#: tree's size.
SPAN_ATLAS_CELLS = 2000
SPAN_LEVEL_Y = (4.95, 3.85, 2.75, 1.65)
SPAN_LEAF_X0, SPAN_LEAF_X1 = 1.9, 12.3
#: Cells as (level, index within the level), the walk's breadth-first order
#: being level by level, left to right (biggest sibling first). The cells the
#: Boundary step's votes landed in, before Diversity begins: deep in the first
#: branch, and one deep in the second.
SPAN_BOUNDARY_VOTES = ((3, 0), (3, 1), (3, 4), (3, 5), (3, 13))
#: The picks Diversity then makes, each the first cell the walk finds empty,
#: up to the stage the slide stops on.
SPAN_STAGE_PICKS = {2: 0, 3: 4, 4: None}


def _span_cells() -> list[tuple[int, int]]:
    return [(level, i) for level in range(SPAN_LEVELS) for i in range(SPAN_K**level)]


def _span_ancestors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    level, i = cell
    out = [cell]
    while level > 0:
        level, i = level - 1, i // SPAN_K
        out.append((level, i))
    return out


def _span_covered(stage: int) -> tuple[set[tuple[int, int]], int, list[tuple[int, int]]]:
    """The cells holding a vote (with everything above them), the walk's run, and the picks Diversity made."""
    covered: set[tuple[int, int]] = set()
    if stage >= 2:
        for cell in SPAN_BOUNDARY_VOTES:
            covered.update(_span_ancestors(cell))
    picks: list[tuple[int, int]] = []
    budget = SPAN_STAGE_PICKS.get(stage, 0)
    order = _span_cells()
    while budget is None or len(picks) < budget:
        empty = next((cell for cell in order if cell not in covered), None)
        if empty is None:
            break
        # The walk takes the first empty cell; a vote there marks it and every cell above.
        picks.append(empty)
        covered.update(_span_ancestors(empty))
    run = 0
    for cell in order:
        if cell not in covered:
            break
        run += 1
    return covered, run, picks


def span_fig() -> None:
    """Span: the first 40 cells of an atlas, the cells the votes reach, and the walk's run.

    Four stages: the cells, in the walk's order; after Boundary, a few covered
    and the run stopped at the first gap (red); four Diversity picks later, the
    run at 10 (yellow); all 40 (green).
    """
    expect = {2: (3, "red"), 3: (10, "yellow"), 4: (40, "green")}
    for stage, (run, light) in expect.items():
        _covered, got, _picks = _span_covered(stage)
        assert got == run and span_status(got, SPAN_ATLAS_CELLS) == light, (stage, got)
    assert SPAN_YELLOW == 10 and SPAN_GREEN_DEFAULT == len(_span_cells()) == 40
    _save(SPAN_STAGES, _span_stage, "autopilot-span")


def _span_xy() -> dict[tuple[int, int], tuple[float, float]]:
    leaves = SPAN_K ** (SPAN_LEVELS - 1)
    step = (SPAN_LEAF_X1 - SPAN_LEAF_X0) / (leaves - 1)
    xy = {(SPAN_LEVELS - 1, i): (SPAN_LEAF_X0 + i * step, SPAN_LEVEL_Y[-1]) for i in range(leaves)}
    for level in range(SPAN_LEVELS - 2, -1, -1):
        for i in range(SPAN_K**level):
            kids = [xy[(level + 1, i * SPAN_K + j)][0] for j in range(SPAN_K)]
            xy[(level, i)] = (sum(kids) / SPAN_K, SPAN_LEVEL_Y[level])
    return xy


def _span_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    _statement(ax, "Have your votes been everywhere?")
    xy = _span_xy()
    for (level, i), (x, y) in xy.items():
        if level:
            px, py = xy[(level - 1, i // SPAN_K)]
            ax.plot([px, x], [py, y], color=RULE, linewidth=1.5, zorder=2)
    for level, y in enumerate(SPAN_LEVEL_Y):
        count = SPAN_K**level
        ax.text(0.7, y, f"{count} cell" if count == 1 else f"{count} cells", ha="left", va="center", fontsize=15,
                color=SOFT)  # fmt: skip
    for i in range(SPAN_K ** (SPAN_LEVELS - 1)):
        x, y = xy[(SPAN_LEVELS - 1, i)]
        ax.plot([x - 0.08, x, x + 0.08], [y - 0.42, y, y - 0.42], color=RULE, linewidth=1.2, zorder=1)
    ax.text((SPAN_LEAF_X0 + SPAN_LEAF_X1) / 2, 0.85, "and thousands of smaller cells below", ha="center",
            va="center", fontsize=15, color=SOFT)  # fmt: skip
    covered, _run, _picks = _span_covered(stage)
    for cell, (x, y) in xy.items():
        face = INK if cell in covered else "white"
        ax.add_patch(Circle((x, y), 0.13, facecolor=face, edgecolor=SOFT, linewidth=1.4, zorder=4))
    if stage >= 2:
        # Where the walk starts: the first empty cell once Boundary is over. The
        # ring stays as the cell fills, so the build only ever adds.
        _covered, run, _picks = _span_covered(2)
        x, y = xy[_span_cells()[run]]
        ax.add_patch(Circle((x, y), 0.22, facecolor="none", edgecolor=INK, linewidth=2.6, zorder=5))
    for shown in range(2, stage + 1):
        _covered, run, picks = _span_covered(shown)
        words = {2: "after Boundary", 3: f"{len(picks)} picks later", 4: "Span is green"}[shown]
        y = SPAN_NOTE_Y[shown]
        _light(ax, SPAN_NOTE_X, y, span_status(run, SPAN_ATLAS_CELLS))
        ax.text(SPAN_NOTE_X + 0.25, y, f"{run} of 40, {words}", ha="left", va="center", fontsize=16,
                fontweight="bold")  # fmt: skip
    return fig


#: Where each stage's count sits: a column right of the root, one line a
#: stage, so the build only adds.
SPAN_NOTE_X, SPAN_NOTE_Y = 8.75, {2: 5.75, 3: 5.35, 4: 4.95}


# ── Cleared to Land: Done ────────────────────────────────────────────────────

DONE_STAGES = 3
DIALOG_X0, DIALOG_X1 = 7.0, 12.3


def done_fig() -> None:
    """The ways a run ends, and what the app says at each, in the app's own words.

    Three stages: every light green on a photo collection; a document
    collection's dry run, the same dialog; and every item labeled, the other.
    """
    _save(DONE_STAGES, _done_stage, "autopilot-done")


def _button(ax: plt.Axes, x: float, y: float, words: str, primary: bool) -> float:
    """A dialog button with its left edge at *x*, sized to its words; returns its right edge."""
    face, ink = (INK, "white") if primary else ("white", INK)
    label = ax.text(x + BUTTON_PAD, y, words, ha="left", va="center", fontsize=15, color=ink, zorder=7)
    ax.figure.canvas.draw()
    width = label.get_window_extent().transformed(ax.transData.inverted()).width + 2 * BUTTON_PAD
    ax.add_patch(FancyBboxPatch((x, y - 0.2), width, 0.4, boxstyle="round,pad=0,rounding_size=0.07",
                                facecolor=face, edgecolor=INK, linewidth=1.4, zorder=6))  # fmt: skip
    return x + width


#: A button's padding either side of its words, in inches.
BUTTON_PAD = 0.16


def _dialog(ax: plt.Axes, y_top: float, heading: str, body: list[str], stay: str) -> tuple[float, float]:
    """The hand-off as the app draws it, top edge at *y_top*; returns its left edge's middle."""
    height = 1.15 + 0.36 * len(body)
    y0 = y_top - height
    ax.add_patch(FancyBboxPatch((DIALOG_X0, y0), DIALOG_X1 - DIALOG_X0, height,
                                boxstyle="round,pad=0,rounding_size=0.12", facecolor="white", edgecolor=SOFT,
                                linewidth=1.6, zorder=4))  # fmt: skip
    ax.text(DIALOG_X0 + 0.25, y_top - 0.33, heading, ha="left", va="center", fontsize=18, fontweight="bold", zorder=5)
    for k, words in enumerate(body):
        ax.text(DIALOG_X0 + 0.25, y_top - 0.75 - 0.34 * k, words, ha="left", va="center", fontsize=15, zorder=5)
    right = _button(ax, DIALOG_X0 + 0.25, y0 + 0.36, stay, primary=False)
    _button(ax, right + 0.18, y0 + 0.36, "Head to Dashboard", primary=True)
    return DIALOG_X0, y0 + height / 2


def _arrow(ax: plt.Axes, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=INK, linewidth=1.8, mutation_scale=22,
                                shrinkA=0, shrinkB=0))  # fmt: skip


def _done_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    _statement(ax, "Three ways a run ends, each said once")
    trained_x, trained_y = _dialog(
        ax,
        5.65,
        "Detector Trained",
        ["Every quality indicator is green.", "Nothing here expires."],
        "Continue Training",
    )
    # Photos: every light green.
    ax.text(0.7, 4.95, "Photos", ha="left", va="center", fontsize=18, fontweight="bold")
    for k, word in enumerate(("Smart", "Stable", "Span")):
        x = 0.85 + k * 1.45
        _light(ax, x, 4.4, "green")
        ax.text(x + 0.2, 4.4, word, ha="left", va="center", fontsize=16)
    _arrow(ax, 5.25, 4.55, trained_x - 0.15, trained_y + 0.15)
    if stage >= 2:
        ax.text(0.7, 3.45, "Documents", ha="left", va="center", fontsize=18, fontweight="bold")
        ax.text(0.7, 3.0, f"{MORE_DRY_RUN} best matches in a row,", ha="left", va="center", fontsize=16)
        ax.text(0.7, 2.65, "not one of them Good", ha="left", va="center", fontsize=16)
        _arrow(ax, 5.25, 3.05, trained_x - 0.15, trained_y - 0.25)
    if stage >= 3:
        # Clear of the page number, which the theme sets in the bottom-right corner.
        ax.text(0.7, 1.8, "Anything", ha="left", va="center", fontsize=18, fontweight="bold")
        ax.text(0.7, 1.35, "every item has a vote", ha="left", va="center", fontsize=16)
        left_x, left_y = _dialog(
            ax, 2.55, "Nothing Left to Label", ["Autopilot has labeled every item", "in this dataset."], "Stay Here"
        )
        _arrow(ax, 5.25, 1.55, left_x - 0.15, left_y)
    return fig


# ── Past the Exit: where the stop fired, measured ────────────────────────────

STOP_STAGES = 2
#: The three radios, as the review's `beta` column names them, and what the slide calls each
#: (`make-sota-figs.RADIOS`).
STOP_RADIOS = ((0.25, "β = ¼"), (1.0, "β = 1"), (4.0, "β = 4"))
#: The review's click budget: where its sessions end, not where a user must.
STOP_BUDGET = 150

#: The numbers the notes quote, pinned per radio as (fire rate, KM median stop, mean Fβ gain after the stop,
#: its SE, share ending 0.02 or more worse, share ending 0.02 or more better): `main` fails if the review's
#: committed table says anything else, so a new State of the App review is a re-run of this script and then
#: a look at the notes.
EXPECT = {
    0.25: (0.83, 81, 0.075, 0.005, 0.15, 0.60),
    1.0: (0.82, 83, 0.076, 0.005, 0.09, 0.69),
    4.0: (0.79, 93, 0.046, 0.003, 0.10, 0.58),
}


def _stop_rows() -> dict[float, dict[str, str]]:
    """The review's stopping table over every session, per radio (`stops_by_preset.py`)."""
    path = REPORT_SOTA / "stopping_by_preset.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(_REPO_ROOT)} is missing: it is the review's stopping table")
    with path.open() as handle:
        return {float(row["beta"]): row for row in csv.DictReader(handle) if row["scope"] == "all"}


def _stop_curves() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """The objective a user has in hand at every click, mean over every session, per radio (`by_click.py`)."""
    path = REPORT_SOTA / "objective_by_click.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(_REPO_ROOT)} is missing: it is the review's per-click objective")
    curves: dict[float, list[tuple[float, float]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            curves.setdefault(float(row["beta"]), []).append((float(row["t"]), float(row["fbeta"])))
    return {beta: (np.array([t for t, _ in pts]), np.array([f for _, f in pts])) for beta, pts in curves.items()}


def stop_fig() -> None:
    """Where today's app told its sessions they could stop, and what clicking on did.

    Two stages: the objective along a session at each radio, with each radio's median stop; then what
    the sessions that were told to stop gained by clicking on to the review's budget.
    """
    rows, curves = _stop_rows(), _stop_curves()
    got = {
        beta: (
            round(float(r["fire_rate"]), 2),
            round(float(r["km_t_stop"])),
            round(float(r["mean_delta"]), 3),
            round(float(r["se_delta"]), 3),
            round(float(r["share_worse"]), 2),
            round(float(r["share_better"]), 2),
        )
        for beta, r in rows.items()
    }
    assert got == EXPECT, got
    _save(STOP_STAGES, lambda stage: _stop_stage(stage, rows, curves), "autopilot-stop")


def _span(values: list[float], fmt: str) -> str:
    """A range over the radios, or one number when they agree at the precision shown."""
    lo, hi = fmt.format(min(values)), fmt.format(max(values))
    return lo if lo == hi else f"{lo} to {hi}"


def _stop_stage(stage: int, rows: dict, curves: dict) -> Figure:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.40, 0.14, 0.43, 0.68))
    ax.set_xlim(0, STOP_BUDGET)
    ax.set_ylim(0.3, 0.7)
    ax.set_xticks([0, 50, 100, 150])
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7])
    ax.set_xlabel("click")
    ax.set_ylabel("Fβ at that β")
    stops = {beta: float(rows[beta]["km_t_stop"]) for beta, _ in STOP_RADIOS}
    ends = []
    for beta, words in STOP_RADIOS:
        t, f = curves[beta]
        ax.plot(t, f, color=INK, linewidth=2.2, zorder=4)
        ax.plot(stops[beta], float(np.interp(stops[beta], t, f)), marker="o", markersize=12,
                markerfacecolor="white", markeredgecolor=INK, markeredgewidth=2.2, zorder=6,
                linestyle="none")  # fmt: skip
        ends.append((float(f[-1]), words))
    heights = spread_labels([f for f, _ in ends], gap=0.03)
    for (_, words), y in zip(ends, heights, strict=True):
        ax.text(STOP_BUDGET + 3, y, words, ha="left", va="center", fontsize=16, clip_on=False)
    ax.text(min(stops.values()), 0.705, "○ the stop, median click", ha="left", va="bottom", fontsize=15,
            color=SOFT, clip_on=False)  # fmt: skip
    fired = [100 * float(rows[beta]["fire_rate"]) for beta, _ in STOP_RADIOS]
    _side_block(
        fig,
        [
            ("The stop fired", "head", 0.0),
            (f"in {_span(fired, '{:.0f}')}% of sessions,", "body", 0.055),
            (f"at median click {_span(list(stops.values()), '{:.0f}')}", "body", 0.05),
        ],
    )
    if stage >= 2:
        first = min(stops.values())
        ax.add_patch(Rectangle((first, 0.3), STOP_BUDGET - first, 0.4, facecolor="#eef1f5", edgecolor="none",
                               zorder=1))  # fmt: skip
        gain = [float(rows[beta]["mean_delta"]) for beta, _ in STOP_RADIOS]
        se = max(float(rows[beta]["se_delta"]) for beta, _ in STOP_RADIOS)
        better = [100 * float(rows[beta]["share_better"]) for beta, _ in STOP_RADIOS]
        worse = [100 * float(rows[beta]["share_worse"]) for beta, _ in STOP_RADIOS]
        _side_block(
            fig,
            [
                (f"Clicking on to {STOP_BUDGET}", "head", 0.0),
                (f"Fβ {_span(gain, '{:+.2f}')} on average", "body", 0.055),
                (f"(± {se:.3f} at most);", "body", 0.05),
                (f"{_span(better, '{:.0f}')}% of sessions end", "body", 0.05),
                ("0.02 or more better,", "body", 0.05),
                (f"{_span(worse, '{:.0f}')}% that much worse", "body", 0.05),
            ],
            top=SIDE_TOP - 0.25,
        )
    return fig


if __name__ == "__main__":
    steps_fig()
    smart_fig()
    stable_fig()
    span_fig()
    done_fig()
    stop_fig()
    print("wrote figures to", OUT)
