#!/usr/bin/env python
"""Figures for the Navigation slides: what Autopilot does, and when it says stop (#4610).

Run from the repo root:

    python slides/figs/src/make-autopilot-figs.py

Autopilot picks every item a user is shown in the Train window, in four steps
and a check, and it tells them when they can stop. The steps are
`AutopilotStateService.checkPhaseTransition`; the stop is three lights the
labeling status computes, Smart (`vtscore/detectors/cost_trend.py`), Stable
(`vtscore/detectors/stability.py`) and Span (the coverage atlas), all green.

Schematic inputs, real code, as in `make-test-figs.py`:

* `autopilot-steps` walks the harness's port of the phase machine
  (`vtscore.eval.autopilot_flow.next_phase`, pinned to the app by
  `scripts/check-eval-app-sync.py`) through the steps, and every target it
  prints is that module's constant.
* `autopilot-smart` is Smart's idea, a student's Spanish classes, and
  `autopilot-smart-trend` runs the shipped `rising_status_from_scores` on a
  planted session's F-beta points (#5134), and asserts the lights the slide draws.
* `autopilot-stable` runs the shipped `count_flips` on two planted retrains,
  and asserts the flips the slide counts.
* `autopilot-span` counts the walk's run over the first 40 cells of an atlas,
  as `CoverageAtlas.coverage_level` does, and reads the light from the
  harness's port of the Span rule.
* `autopilot-done` is the three ways a run ends: every light green, a
  document walk run dry (the harness's `MORE_DRY_RUN`), every item voted.

The last figure, `autopilot-stop`, is the only measurement: where the stop
fired in today's app, at each radio's beta, and what it gave up against
clicking on to 150. It is re-plotted from committed tables in the State of the
App review's shape (`stopping_by_preset.csv` and `objective_by_click.csv`,
written by `stops_by_preset.py` and `by_click.py`). Since #5141 they are
#5134's shipped arm, Smart on F-beta, read at the review's 150-click budget
(`docs/experiments/2026-10-20-smart-fbeta-5134/stop/`); the chart runs on to
the app's 200-click cap from that study's own tables (#5171). The 2026-10-16
Binary Photo review measured the app before Smart's switch. `EXPECT` pins the
numbers the slide and notes quote.

Every canvas is the whole 1280x720 slide at 100 px to the inch, saved at its
declared bounds, so every stage of a build shares one framing by construction
and `save()` checks the title notch against exactly what the slide shows.
"""

from __future__ import annotations

import csv
import gzip
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
from matplotlib.patches import Circle, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save, spread_labels  # noqa: E402

from vtscore.detectors.cost_trend import (  # noqa: E402
    SMART_RISE_THRESHOLD,
    SMART_SLOPE_T,
    SMART_WINDOW,
    rising_status_from_scores,
)
from vtscore.detectors.stability import (  # noqa: E402
    STABLE_BAND_STD_FRACTION,
    ScoredSnapshot,
    count_flips,
)
from vtscore.eval.autopilot_flow import (  # noqa: E402
    BAD_TARGET,
    GOOD_TARGET,
    MORE_DRY_RUN,
    SPAN_GREEN_DEFAULT,
    STOP_CAP,
    next_phase,
)

OUT = Path(__file__).resolve().parent.parent
#: The tables the measured slide reads: Binary Photo on COCO Better, one session set per radio, in the review's shape.
#: #5134's shipped arm (Smart on F-beta) read at click 150 (#5141), until the next State of the App review.
REPORT_STOP = _REPO_ROOT / "docs" / "experiments" / "2026-10-20-smart-fbeta-5134" / "stop"

# The deck's palette (`make-calib-figs.py`, `themes/vtsearch.css`): blue is the
# line, red the losing arm and a Bad, green a Good. The lights borrow red and
# green from it and add an amber, and every light on a slide is named in words
# beside it, so no state is told by hue alone.
BLUE = "#0b5fa5"
RED = "#b91c1c"
GREEN = "#0d8a5f"
AMBER = "#c98a06"
RULE = "#d8dee6"
#: An atlas's edges: dark enough to read as the tree's own structure from the
#: back of a room, still lighter than the cells they join (owner, #4884: the
#: rule grey was too faint).
EDGE = "#9aa3ae"
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
        save(draw(stage), OUT, f"{name}.build{stage}.webp", column=FULL_BLEED, tight=False)
    save(draw(stages), OUT, f"{name}.webp", column=FULL_BLEED, tight=False)


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


def _disc(ax: plt.Axes, x: float, y: float, mark: int | str, r: float = 0.22) -> None:
    ax.add_patch(Circle((x, y), r, facecolor=INK, edgecolor="none", zorder=6))
    ax.text(x, y, f"{mark}", ha="center", va="center", fontsize=16, color="white", fontweight="bold", zorder=7)


def _line(ax: plt.Axes, x: float, y0: float, y1: float) -> None:
    """The line, in the palette's blue: the one shipped decision on every drawing."""
    ax.plot([x, x], [y0, y1], color=BLUE, linewidth=2.8, zorder=6, solid_capstyle="butt")


# ── Flight Plan: the steps ───────────────────────────────────────────────────

STEPS_STAGES = 4

#: The rows, top to bottom: the step's mark, its name as the panel's short
#: label shows it, the sort it asks from, the drawing in the Question column,
#: and what ends it. The
#: targets are the harness's constants, which mirror the app's. Spot Check is
#: lettered, not numbered: it runs once every light is green, but Autopilot
#: also runs it earlier when the labels separate weakly (#4496), so it is not
#: a fifth step in a fixed order (owner, #4872). Its "until" is what the check
#: is for: it walks bands down from the top while its estimate of the kept
#: set's F-beta rises, and stops at that estimate's peak (`spot_check`), so the
#: slide says "Fβ: estimated", in the lights' "name: state" form (owner,
#: #5171). Each light's "until" names what green
#: means for it, not the colour (owner, #4884), so the slide reads without the
#: three light slides before it; a two-part rule takes two lines.
STEPS = (
    # The Good walk can also run dry once a Good is in (#4731).
    ("good", "1", "Good", "Query", "query-top", f"{GOOD_TARGET} Goods OR\n{MORE_DRY_RUN} in a row without one"),
    ("bad", "2", "Bad", "Query", "query-line", f"{BAD_TARGET} Bads"),
    ("hard", "3", "Boundary", "Detector", "detector-odds", "Smart: flat AND\nStable: low"),
    ("new", "4", "Diversity", "Atlas", "atlas", "Span: full"),
    ("spot", "A", "Spot Check", "Detector", "bands", "Fβ: estimated"),
)
#: No More walk on photos since #4740 (``MORE_TARGET`` 0): the opening is Good and
#: Bad on the query, then the detector takes over. Five rows down the slide,
#: with no statement over them (owner, #4872: the room has met Autopilot by now),
#: under a header row and its rule, as a table (owner, #5171: the old soft
#: "where it asks" and "until" just floated). The header sits under the title
#: notch, since its first column starts inside the notch's span.
STEP_ROWS_Y = (4.4, 3.45, 2.5, 1.55, 0.6)
STEP_HEAD_Y, STEP_RULE_Y = 5.2, 4.95
STEP_RULE_X0, STEP_RULE_X1 = 0.6, 12.4
STEP_HEADS = ("Sort", "Question", "Stopping")
#: Which row each build stage reveals up to: the opening on the query, then
#: one step a stage.
STEP_STAGE_ROWS = {1: 2, 2: 3, 3: 4, 4: 5}
#: The Question column is the point of the slide, so it takes every inch the
#: Sort and Stopping columns leave it (owner, #4884).
NAME_X, SORT_X, PICTURE_X0, PICTURE_X1, ENDS_X = 1.25, 3.0, 4.2, 9.3, 9.65
STRIP_H = 0.34
#: Where each ranking's line falls along its strip (best on the right).
QUERY_LINE, DETECTOR_LINE = 0.62, 0.68
#: Where Boundary asks on the detector's strip: the first item, best first, the
#: line's own fit calls even odds (#3546, #4632). Inside the line, as it is at
#: the middle radio every detector starts on; at β ¼ the two nearly meet.
DETECTOR_ODDS = 0.80
#: Spot Check's bands on the detector's strip, as fractions of it from the
#: right: the top 8, the next 8, then 16, then the band past the line, each
#: twice the last (`calib-floor-check`'s bands, where the line keeps 32).
SPOT_BANDS = (0.0, 0.08, 0.16, 0.32, 0.64)
#: Where in each band its few random picks fall, as a share of the band's
#: width: drawn once, so every page of the build shows the same picks.
SPOT_PICKS = ((0.5,), (0.5,), (0.25, 0.75), (0.2, 0.5, 0.8))
#: The asked item's mark, the deck's "?" in a ring (owner, #4872: the (?) for
#: where it asks, not a pointer); 15pt is the smallest type this canvas holds.
ASK_R, ASK_PT = 0.125, 15


def _check_step_order() -> None:
    """The steps in the order the phase machine takes them, from its own rule.

    A session walked the way the slide narrates it: the counts reach each
    target in turn, then Smart and Stable go green, then Span, then the check
    every light green calls for runs.
    """
    many = math.inf
    walk = (
        dict(good_count=0, bad_count=0, smart="red", stable="red", span="red"),
        dict(good_count=GOOD_TARGET, bad_count=0, smart="red", stable="red", span="red"),
        dict(good_count=GOOD_TARGET, bad_count=BAD_TARGET, smart="red", stable="red", span="red"),
        dict(good_count=24, bad_count=30, smart="green", stable="green", span="yellow"),
        dict(good_count=30, bad_count=50, smart="green", stable="green", span="green", done_checked=False),
    )
    phases = tuple(next_phase(remaining_unlabeled=many, **state) for state in walk)
    assert phases == tuple(step[0] for step in STEPS), phases
    # And the check also runs mid-run, which is why it is lettered: a weak-separation
    # check holds the phase at Spot Check whatever the counts.
    early = next_phase(remaining_unlabeled=many, checking=True, **walk[3])
    assert early == "spot", early


def steps_fig() -> None:
    """The steps: what each asks about, and what ends it.

    Four stages: the opening's two steps, both on the query's own ranking, with
    no detector shown yet; then Boundary, the first step on the detector's
    ranking; then Diversity, off the atlas; then Spot Check, which every light
    green calls for (#4836) and weak separation can call earlier (#4496).
    """
    _check_step_order()
    _save(STEPS_STAGES, _steps_stage, "autopilot-steps")


def _strip(ax: plt.Axes, y: float, line_at: float) -> float:
    """A ranking as a strip, best on the right, its line drawn; returns the line's x.

    Which ranking it is goes in the Sort column, not on the strip.
    """
    width = PICTURE_X1 - PICTURE_X0
    ax.add_patch(Rectangle((PICTURE_X0, y - STRIP_H / 2), width, STRIP_H, facecolor=UNLABELED_FILL,
                           edgecolor="none", zorder=2))  # fmt: skip
    x = PICTURE_X0 + line_at * width
    _line(ax, x, y - STRIP_H / 2 - 0.08, y + STRIP_H / 2 + 0.08)
    return x


def _ask(ax: plt.Axes, x: float, y: float) -> None:
    """Where the step asks: the deck's (?) on the item it picks."""
    ax.add_patch(Circle((x, y), ASK_R, facecolor="white", edgecolor=INK, linewidth=2.2, zorder=8))
    ax.text(x, y, "?", ha="center", va="center_baseline", fontsize=ASK_PT, fontweight="bold", color=INK, zorder=9)


def _spot_bands(ax: plt.Axes, y: float) -> None:
    """The detector's strip with Spot Check's bands bracketed from the top, and a few random picks in each."""
    _strip(ax, y, DETECTOR_LINE)
    width = PICTURE_X1 - PICTURE_X0
    # Each band's bracket sits over the strip, its picks on the strip under it.
    top = y + STRIP_H / 2 + 0.1
    for k in range(len(SPOT_BANDS) - 1):
        right = PICTURE_X1 - SPOT_BANDS[k] * width
        left = PICTURE_X1 - SPOT_BANDS[k + 1] * width
        lo, hi = left + 0.025, right - 0.025
        ax.plot([lo, lo, hi, hi], [top, top + 0.08, top + 0.08, top], color=INK, linewidth=1.4, zorder=6)
        for share in SPOT_PICKS[k]:
            _ask(ax, left + share * (right - left), y)


def _mini_tree(ax: plt.Axes, y: float) -> None:
    """Three levels of an atlas, the first cell nobody has voted in asked about."""
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
            ax.plot([parent, x], [ys[level - 1], ys[level]], color=EDGE, linewidth=1.4, zorder=2)
    # Voted in, filled as the Span slide fills them: the root, its first two
    # children, and three cells under those. The walk's next cell carries the (?).
    covered = {(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 4)}
    first_empty = (1, 2)
    for level, row in enumerate(xs):
        for i, x in enumerate(row):
            if (level, i) == first_empty:
                _ask(ax, x, ys[level])
                continue
            face = INK if (level, i) in covered else "white"
            ax.add_patch(Circle((x, ys[level]), 0.075, facecolor=face, edgecolor=SOFT, linewidth=1.2, zorder=4))


def _steps_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    for x, head in zip((SORT_X, PICTURE_X0, ENDS_X), STEP_HEADS, strict=True):
        ax.text(x, STEP_HEAD_Y, head, ha="left", va="center", fontsize=18, fontweight="bold")
    ax.plot([STEP_RULE_X0, STEP_RULE_X1], [STEP_RULE_Y, STEP_RULE_Y], color=INK, linewidth=1.4, zorder=2)
    for row in range(STEP_STAGE_ROWS[stage]):
        _phase, mark, name, sort, picture, ends = STEPS[row]
        y = STEP_ROWS_Y[row]
        _disc(ax, 0.85, y, mark)
        ax.text(NAME_X, y, name, ha="left", va="center", fontsize=18, fontweight="bold")
        ax.text(SORT_X, y, sort, ha="left", va="center", fontsize=17)
        if picture == "query-top":
            _strip(ax, y, QUERY_LINE)
            _ask(ax, PICTURE_X1 - ASK_R - 0.03, y)
        elif picture == "query-line":
            _ask(ax, _strip(ax, y, QUERY_LINE), y)
        elif picture == "detector-odds":
            _strip(ax, y, DETECTOR_LINE)
            _ask(ax, PICTURE_X0 + DETECTOR_ODDS * (PICTURE_X1 - PICTURE_X0), y)
        elif picture == "atlas":
            _mini_tree(ax, y)
        else:
            _spot_bands(ax, y)
        ax.text(ENDS_X, y, ends, ha="left", va="center", fontsize=17, linespacing=1.35)
    return fig


# ── Rate Expectations, then Diminishing Returns: Smart ───────────────────────

#: The idea first, then the implementation (owner, #4884): eight pages of a
#: student taking Spanish 101 to 104, and then, on a slide of its own (owner,
#: #5132), the two pages of the retrain chart.
SPANISH_PAGES = 8
SMART_TREND_STAGES = 2
#: The classes along the time axis, and the ability axis's words, bottom up.
SPANISH_CLASSES = ("Spanish101", "Spanish102", "Spanish103", "Spanish104")
SPANISH_LEVELS = ("Bad", "Ok", "Good", "Great")
#: The ability axis on the last page, after Spanish 104: in Spanish (owner, #5132).
SPANISH_AXIS = "Spanish ability"
SPANISH_AXIS_ES = "Habilidad en Español"
SPANISH_LEVELS_ES = ("Malo", "Ok", "Bueno", "Genial")
#: Each test the student takes, in order: (class, before or after it, ability).
#: Ability is in a fixed unit nobody sees — how much Spanish they know — and the
#: axis shows it against everything they have met so far.
SPANISH_TESTS = (
    (0, "before", 0.0),
    (0, "after", 1.0),
    (1, "before", 1.0),
    (1, "after", 3.0),
    (2, "before", 3.0),
    (2, "after", 5.0),
    (3, "before", 5.0),
    (3, "after", 5.0),
)
#: How the axis reads ability while each class is the hardest one met, as
#: (ability at "Bad", axis steps per unit). A new class shows how much Spanish
#: there is, so the axis takes the student's ability on starting it as "Bad" and
#: everything they knew before shrinks below it; 104 brings nothing they did not
#: already know, so its axis is 103's.
SPANISH_SCALES = ((0.0, 3.0), (1.0, 1.0), (3.0, 0.5), (3.0, 0.5))
#: Where a test sits along its class: just inside its start or its end.
SPANISH_INSET = 0.12
SPANISH_YLIM = (-1.85, 3.35)
#: The ability each page shows, as the notes describe it: the class's words, once a test is the newest on the page.
SPANISH_SHOWN = ("Bad", "Great", "Bad", "Good", "Bad", "Ok", "Ok", "Ok")
#: A planted session (#5134): retrain *k*'s detector, scored on the votes as
#: they stood at a later retrain *t*. Smart re-scores its window on *today's*
#: votes, and Autopilot spends the clicks between retrains on the photos it is
#: least sure of, so the votes keep getting harder: an old detector that never
#: trained on the newer ones loses ground on them. The chart at retrain 14 and
#: the chart at retrain 30 are two different tests, not one curve extended
#: (owner, #5171). A detector's skill rises and levels off
#: (``SMART_SKILL``: top, rise, time constant), the votes harden
#: (``SMART_HARDER``: by how much, how fast), and every re-scoring draws its own
#: step-to-step scatter.
SMART_RETRAINS = 30
SMART_SEED = 7
SMART_SKILL = (0.85, 0.55, 5.0)
SMART_HARDER = (0.35, 12.0)
SMART_SCATTER = 0.008
#: The two days the slide shows, each the last retrain its window ends on.
SMART_EARLY_END, SMART_LATE_END = 14, 30


def _spanish_axis(page: int) -> tuple[float, float]:
    """The axis on *page* (1-based): the scale of the hardest class whose test is on it."""
    newest = SPANISH_TESTS[page - 1][0]
    return SPANISH_SCALES[newest]


def _spanish_points(page: int) -> list[tuple[float, float]]:
    """``(time, axis height)`` of every test on *page* (1-based), read on that page's axis."""
    bad, steps = _spanish_axis(page)
    out = []
    for klass, when, ability in SPANISH_TESTS[:page]:
        x = klass + (SPANISH_INSET if when == "before" else 1 - SPANISH_INSET)
        out.append((x, (ability - bad) * steps))
    return out


def _smart_fbetas(today: int) -> np.ndarray:
    """F-beta of retrains 1 to *today*, every one scored on the votes as they stand at retrain *today*."""
    scatter = np.random.default_rng(SMART_SEED).normal(0.0, SMART_SCATTER, size=(SMART_RETRAINS, SMART_RETRAINS))
    top, rise, pace = SMART_SKILL
    harder, harden = SMART_HARDER
    k = np.arange(1, today + 1)
    skill = top - rise * np.exp(-k / pace)
    # What the votes asked since retrain k, which its detector never trained on.
    unseen = harder * (np.exp(-k / harden) - np.exp(-today / harden))
    return skill - unseen + scatter[today - 1, :today]


def _smart_window(fbetas: np.ndarray, end: int) -> tuple[np.ndarray, dict]:
    """The last ``SMART_WINDOW`` retrains up to *end* (1-based), and the shipped Smart status over them."""
    window = fbetas[end - SMART_WINDOW : end]
    return window, rising_status_from_scores(list(window), good=10, bad=10)


def smart_fig() -> None:
    """Smart: the idea, in Spanish classes, and then the F-beta of the last ten detectors.

    Two slides. Eight pages of the idea (owner, #4884): a student whose ability
    axis rescales each time a class shows them how much more Spanish there is,
    until a class brings nothing new and the curve goes flat, its axis by then
    in Spanish (#5132). Then, on its own slide (#5132), the shipped measure,
    early in a session (still rising) and later (flat).
    """
    # Every test on its own page reads as the notes say it does.
    for page, words in enumerate(SPANISH_SHOWN, start=1):
        height = _spanish_points(page)[-1][1]
        assert height == SPANISH_LEVELS.index(words), (page, height, words)
    # And each rescale leaves everything before it below the new "Bad".
    for page in (3, 5):
        assert all(y < 0 for _x, y in _spanish_points(page)[: page - 2]), page
    # The notes say "two standard errors" and "the 1.5% line".
    assert (SMART_SLOPE_T, SMART_RISE_THRESHOLD) == (2.0, 0.015), (SMART_SLOPE_T, SMART_RISE_THRESHOLD)
    early_day, late_day = _smart_fbetas(SMART_EARLY_END), _smart_fbetas(SMART_LATE_END)
    early, late = _smart_window(early_day, SMART_EARLY_END), _smart_window(late_day, SMART_LATE_END)
    assert early[1]["status"] == "yellow", early[1]
    assert late[1]["status"] == "green" and not late[1]["drift_within_noise"], late[1]
    # The notes' claim: on the later, harder votes every old detector scores lower than it did.
    assert (late_day[:SMART_EARLY_END] < early_day).all(), late_day[:SMART_EARLY_END] - early_day
    assert 0.0 < late_day.min() and early_day.max() < SMART_Y_TOP, (late_day.min(), early_day.max())
    _save(SPANISH_PAGES, _spanish_stage, "autopilot-smart")
    _save(SMART_TREND_STAGES, _smart_trend_stage, "autopilot-smart-trend")


def _spanish_stage(page: int) -> Figure:
    """The student's tests up to *page*, on that page's axis; a rescale shows where the old dots were.

    The last page, after Spanish 104, reads its ability axis in Spanish.
    """
    spanish = page == SPANISH_PAGES
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.15, 0.15, 0.79, 0.58))
    ax.set_xlim(0, len(SPANISH_CLASSES))
    ax.set_ylim(*SPANISH_YLIM)
    ax.set_yticks(range(len(SPANISH_LEVELS)))
    ax.set_yticklabels(SPANISH_LEVELS_ES if spanish else SPANISH_LEVELS, fontsize=17)
    ax.tick_params(axis="y", colors=INK)
    ax.set_xticks([k + 0.5 for k in range(len(SPANISH_CLASSES))])
    ax.set_xticklabels(SPANISH_CLASSES, fontsize=17, color=INK)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.set_ylabel(SPANISH_AXIS_ES if spanish else SPANISH_AXIS)
    for k in range(1, len(SPANISH_CLASSES)):
        ax.axvline(k, color=RULE, linewidth=1.2, zorder=1)
    for level in range(len(SPANISH_LEVELS)):
        ax.axhline(level, color="#eef1f5", linewidth=1.0, zorder=0)
    points = _spanish_points(page)
    if page > 1 and _spanish_axis(page) != _spanish_axis(page - 1):
        # The rescale: every earlier test where the last page had it, hollow, and an arrow down to where it is now.
        # The hollow dots keep the thin line that joined them, so what came down is visibly the curve the last
        # page drew (owner, #5171).
        ghosts = _spanish_points(page - 1)
        ax.plot(*zip(*ghosts, strict=True), color=SOFT, linewidth=1.2, zorder=2)
        for (x, was), (_x, now) in zip(ghosts, points, strict=False):
            ax.plot(x, was, marker="o", markersize=12, markerfacecolor="white", markeredgecolor=SOFT,
                    markeredgewidth=1.6, linestyle="none", zorder=3)  # fmt: skip
            ax.annotate("", xy=(x, now + 0.16), xytext=(x, was - 0.16),
                        arrowprops=dict(arrowstyle="-|>", color=SOFT, linewidth=1.6, mutation_scale=16,
                                        shrinkA=0, shrinkB=0), zorder=3)  # fmt: skip
    xs, ys = zip(*points, strict=True)
    ax.plot(xs, ys, color=INK, linewidth=2.4, zorder=4)
    ax.plot(xs, ys, marker="o", markersize=12, color=INK, linestyle="none", zorder=5)
    return fig


def _fit_line(window: np.ndarray, x0: int) -> tuple[np.ndarray, np.ndarray]:
    xs = np.arange(x0, x0 + window.size)
    slope, intercept = np.polyfit(xs, window, 1)
    return xs, slope * xs + intercept


def _smart_trend_stage(stage: int) -> Figure:
    """The retrain chart on one day's votes: retrain 14's, still rising (stage 1); retrain 30's, flat (stage 2).

    Stage 2 re-scores every old detector on the harder votes, and keeps stage 1's points where they stood,
    hollow and joined as stage 1 drew them, so the room sees the same detectors drop rather than an old
    curve levelling off (owner, #5171).
    """
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.40, 0.14, 0.56, 0.62))
    ax.set_xlim(0.3, SMART_RETRAINS + 0.7)
    ax.set_ylim(0.0, SMART_Y_TOP)
    ax.set_xticks([1, 10, 20, 30])
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("retrain")
    ax.set_ylabel("F-beta on today's votes")
    fig.text(0.40, 0.86, "Is the detector still getting better?", ha="left", va="center", fontsize=19)
    overlay = _overlay(fig)
    today = SMART_EARLY_END if stage == 1 else SMART_LATE_END
    _side_block(
        fig,
        [
            ("Each point", "head", 0.0),
            ("one retrain's detector,", "body", 0.055),
            ("scored on today's votes", "body", 0.05),
            ("Today", "head", 0.085),
            (f"retrain {today}", "body", 0.055),
            ("The fit", "head", 0.085),
            (f"through the last {SMART_WINDOW}", "body", 0.055),
        ],
    )
    fbetas = _smart_fbetas(today)
    if stage == SMART_TREND_STAGES:
        before = _smart_fbetas(SMART_EARLY_END)
        xs = np.arange(1, SMART_EARLY_END + 1)
        ax.plot(xs, before, color=SOFT, linewidth=1.2, zorder=3)
        ax.plot(xs, before, linestyle="none", marker="o", markersize=8, markerfacecolor="white",
                markeredgecolor=SOFT, markeredgewidth=1.6, zorder=3)  # fmt: skip
        # Over the hollow curve's end rather than past it: past it are the re-scored points.
        ax.text(SMART_EARLY_END, before[-1] + 0.05, f"on retrain {SMART_EARLY_END}'s votes", ha="center",
                va="bottom", fontsize=15, color=SOFT, zorder=6)  # fmt: skip
    ax.plot(np.arange(1, today + 1), fbetas, linestyle="none", marker="o", markersize=8, color=INK, zorder=4)
    label_y, words = (0.42, "still rising") if stage == 1 else (0.6, "flat")
    window, status = _smart_window(fbetas, today)
    x0 = today - SMART_WINDOW + 1
    ax.add_patch(Rectangle((x0 - 0.45, 0.0), SMART_WINDOW - 0.1, SMART_Y_TOP, facecolor="#eef1f5",
                           edgecolor="none", zorder=1))  # fmt: skip
    fx, fy = _fit_line(window, x0)
    ax.plot(fx, fy, color=INK, linewidth=2.6, zorder=5)
    cx, cy = _to_canvas(fig, ax, x0 + (SMART_WINDOW - 1) / 2, label_y)
    overlay.text(cx, cy, words, ha="center", va="center", fontsize=17, fontweight="bold")
    overlay.text(cx, cy - 0.36, _signed(100 * status["slope"]) + "% a retrain", ha="center", va="center",
                 fontsize=16)  # fmt: skip
    return fig


#: The top of the Smart chart's F-beta axis: F-beta's own.
SMART_Y_TOP = 1.0


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


# ── Change of Heart: Stable ──────────────────────────────────────────────────

STABLE_STAGES = 5
STABLE_CUT = 0.5
#: The planted pool both retrains score, one photo a letter, best on the right.
#: Lettered left to right on the last retrain, so the room can follow each one
#: down to this retrain (owner, #4884): A and B swap places, which changes
#: nothing anyone gets back; H, I, J and K wobble across the line from inside
#: the band; D and N cross it from well clear on both sides, the two that count.
#: Placed by hand rather than drawn at random, because every circle has to
#: clear its neighbours on both lines (`STABLE_MIN_GAP`), and the line between
#: them (`STABLE_LINE_CLEAR`).
STABLE_LAST = (0.04, 0.10, 0.16, 0.22, 0.28, 0.335, 0.385, 0.43, 0.47, 0.53, 0.57, 0.62, 0.68, 0.74, 0.80, 0.86,
               0.92, 0.97)  # fmt: skip
STABLE_THIS = (0.10, 0.04, 0.16, 0.785, 0.29, 0.34, 0.385, 0.53, 0.57, 0.43, 0.47, 0.615, 0.67, 0.225, 0.725, 0.85,
               0.915, 0.965)  # fmt: skip
STABLE_LETTERS = "ABCDEFGHIJKLMNOPQR"
#: Two retrains drawn one above the other; scores run left to right, best on
#: the right, across this span of the canvas. Wide, so each photo's lettered
#: circle clears the next.
STABLE_Y_LAST, STABLE_Y_THIS = 4.75, 2.15
STABLE_X0, STABLE_X1 = 4.25, 12.3
STABLE_ITEM_R = 0.13
STABLE_MIN_GAP = 2 * STABLE_ITEM_R + 0.04
STABLE_LINE_CLEAR = STABLE_ITEM_R + 0.08
#: The column the counts stand in, left of the two retrains.
STABLE_TEXT_X = 0.7
#: The last page's chart: a planted session's count of flips that count, one a
#: retrain, jiggling down to none (owner, #4884: fabricated, it is a picture of
#: the idea). Retrains from the hand-over to the detector to the 150th click.
STABLE_CHART_SEED = 31
STABLE_CHART_CLICKS = (8, 150)


def _stable_retrains() -> tuple[ScoredSnapshot, ScoredSnapshot]:
    assert len(STABLE_LAST) == len(STABLE_THIS) == len(STABLE_LETTERS)
    return (
        ScoredSnapshot.from_scores(dict(enumerate(STABLE_LAST)), STABLE_CUT),
        ScoredSnapshot.from_scores(dict(enumerate(STABLE_THIS)), STABLE_CUT),
    )


def _stable_chart() -> tuple[np.ndarray, np.ndarray]:
    """The super flips at every retrain of a planted session: a few early, jiggling, flattening to none."""
    rng = np.random.default_rng(STABLE_CHART_SEED)
    t = np.arange(STABLE_CHART_CLICKS[0], STABLE_CHART_CLICKS[1] + 1)
    mean = 7.0 * np.exp(-(t - t[0]) / 26.0)
    count = np.maximum(0, np.round(mean + rng.normal(0.0, 1.0, size=t.size) * 0.8 * np.sqrt(mean))) + 0.0
    return t, count


def stable_fig() -> None:
    """Stable: two retrains' calls on the same pool, the photos that changed side, and which of those count.

    Five stages: the last retrain alone (owner, #5132); then this one under
    it, every photo joined to itself by its letter; the photos that changed side; the band around the line, and the flips clear
    of it both times, the super flips, the only ones that count; and those over
    a session, falling to none.
    """
    # The notes say "a quarter of the scores' spread either side".
    assert STABLE_BAND_STD_FRACTION == 0.25, STABLE_BAND_STD_FRACTION
    last, this = _stable_retrains()
    flips, confident = count_flips(last, this)
    assert (flips, confident) == (6, 2), (flips, confident)
    for scores in (STABLE_LAST, STABLE_THIS):
        xs = sorted(_score_x(v) for v in scores)
        assert min(b - a for a, b in zip(xs, xs[1:])) >= STABLE_MIN_GAP, "two lettered circles touch"
        assert min(abs(x - _score_x(STABLE_CUT)) for x in xs) >= STABLE_LINE_CLEAR, "the line cuts a circle"
    t, count = _stable_chart()
    # The session settles: no flip that counts over its last forty retrains.
    assert count[0] >= 5 and not count[-40:].any(), count
    _save(STABLE_STAGES, lambda stage: _stable_stage(stage, last, this), "autopilot-stable")


def _score_x(score: float) -> float:
    return STABLE_X0 + score * (STABLE_X1 - STABLE_X0)


def _runs(ax: plt.Axes, x: float, y: float, runs: tuple[tuple[str, str], ...], size: float = 17) -> None:
    """One line of bold text from *x*, set in runs of ``(words, "normal" | "italic")``."""
    renderer = ax.figure.canvas.get_renderer()
    for words, style in runs:
        label = ax.text(x, y, words, ha="left", va="center", fontsize=size, fontweight="bold", fontstyle=style)
        box = label.get_window_extent(renderer).transformed(ax.transData.inverted())
        x = box.x1


def _stable_stage(stage: int, last: ScoredSnapshot, this: ScoredSnapshot) -> Figure:
    if stage == STABLE_STAGES:
        return _stable_chart_stage()
    # Step 0 is the last retrain alone; from step 1 this one is under it.
    step = stage - 1
    fig, ax = _canvas()
    _stable_retrain_axes(ax, step, last, this)
    flipped, confident = _stable_items(ax, last, this, both=step >= 1)
    if step >= 2:
        for x0, x1 in flipped:
            ax.plot([x0, x1], [STABLE_Y_LAST, STABLE_Y_THIS], color=SOFT, linewidth=2.6, zorder=5)
        ax.text(STABLE_TEXT_X, 4.0, f"{len(flipped)} changed side", ha="left", va="center", fontsize=17,
                fontweight="bold")  # fmt: skip
    if step >= 3:
        for x0, x1 in confident:
            ax.plot([x0, x1], [STABLE_Y_LAST, STABLE_Y_THIS], color=INK, linewidth=4.2, zorder=6)
        _runs(ax, STABLE_TEXT_X, 3.4, ((f"{len(confident)} ", "normal"), ("super", "italic"),
                                       (" changed side", "normal")))  # fmt: skip
    return fig


def _stable_retrain_axes(ax: plt.Axes, step: int, last: ScoredSnapshot, this: ScoredSnapshot) -> None:
    """The retrains' score axes, each with its line: the last alone at step 0, both after; bands from step 3."""
    for y, snap, words, words_y in (
        (STABLE_Y_LAST, last, "last retrain", STABLE_Y_LAST + 0.5),
        (STABLE_Y_THIS, this, "this retrain", STABLE_Y_THIS - 0.5),
    )[: 1 if step == 0 else 2]:
        ax.plot([STABLE_X0 - 0.2, STABLE_X1 + 0.2], [y, y], color=SOFT, linewidth=1.4, zorder=2)
        ax.text(STABLE_X0 - 0.2, words_y, words, ha="left", va="center", fontsize=16, color=SOFT)
        if step >= 3:
            x0, x1 = _score_x(snap.threshold - snap.band), _score_x(snap.threshold + snap.band)
            ax.add_patch(Rectangle((x0, y - 0.3), x1 - x0, 0.6, facecolor=BAND_FILL, edgecolor="none", zorder=1))
        _line(ax, _score_x(snap.threshold), y - 0.36, y + 0.36)
    ax.text(STABLE_X1 + 0.2, STABLE_Y_LAST + 0.5, "best", ha="right", va="center", fontsize=15, color=SOFT)


def _stable_items(
    ax: plt.Axes, last: ScoredSnapshot, this: ScoredSnapshot, *, both: bool = True
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Every photo joined to itself, its letter on both lines; returns the ends of the flips, and of the confident ones among them.

    With *both* false only the last retrain's letters are drawn, and nothing joins them.
    """
    flipped, confident = [], []
    for cid in sorted(this.scores):
        x0, x1 = _score_x(last.scores[cid]), _score_x(this.scores[cid])
        if both:
            ax.plot([x0, x1], [STABLE_Y_LAST, STABLE_Y_THIS], color=RULE, linewidth=1.4, zorder=3)
        for x, y in ((x0, STABLE_Y_LAST), (x1, STABLE_Y_THIS))[: 2 if both else 1]:
            ax.add_patch(Circle((x, y), STABLE_ITEM_R, facecolor="white", edgecolor=SOFT, linewidth=1.6, zorder=7))
            ax.text(x, y, STABLE_LETTERS[cid], ha="center", va="center_baseline", fontsize=15, fontweight="bold",
                    color=INK, zorder=8)  # fmt: skip
        if last.predicted(cid) != this.predicted(cid):
            flipped.append((x0, x1))
            if last.confident(cid) and this.confident(cid):
                confident.append((x0, x1))
    return flipped, confident


def _stable_chart_stage() -> Figure:
    """The super flips over a session, one point a retrain, falling to none."""
    t, count = _stable_chart()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.10, 0.14, 0.84, 0.58))
    ax.set_xlim(0, STABLE_CHART_CLICKS[1] + 2)
    ax.set_ylim(0, max(count) + 1)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150])
    ax.set_yticks(range(0, int(max(count)) + 2, 2))
    ax.set_xlabel("clicks")
    ax.set_ylabel("# super changed side")
    ax.plot(t, count, color=INK, linewidth=2.4, solid_joinstyle="round", zorder=4)
    fig.text(STATEMENT_X / W, STATEMENT_Y / H, "# Super Changed Side vs Clicks", ha="left", va="center", fontsize=19)
    return fig


# ── Tree's Company: Span ─────────────────────────────────────────────────────

SPAN_STAGES = 4
#: The atlas the slide draws: three children a cell, the first four levels,
#: which are exactly the 40 cells Span's default full asks for.
SPAN_K, SPAN_LEVELS = 3, 4
SPAN_LEVEL_Y = (4.95, 3.85, 2.75, 1.65)
SPAN_LEAF_X0, SPAN_LEAF_X1 = 1.9, 12.3
#: Cells as (level, index within the level), the walk's breadth-first order
#: being level by level, left to right (biggest sibling first). The cells the
#: Boundary step's votes landed in, before Diversity begins: deep in the first
#: branch, and one deep in the second, off its middle (owner, #5132: a path
#: straight down the middle of the tree looked chosen, not landed in).
SPAN_BOUNDARY_VOTES = ((3, 0), (3, 1), (3, 4), (3, 5), (3, 16))
#: The picks Diversity then makes, each the first cell the walk finds empty, and
#: the drawn leaf the picked item sits in. A vote is on one item, which sits in
#: one leaf of the atlas, and it marks that leaf and every cell above it
#: (`CoverageAtlas.label`), so a pick in a big empty cell fills a path down
#: through it, not the cell alone. Which child the item sits in is the item's
#: business, so the path wanders (owner, #4884: not always the leftmost child):
#: the pick at cell 02 lands in 021 and then 0212.
SPAN_PICK_LEAVES = {(1, 2): (3, 23)}
#: The four pages: the empty atlas; the cells the Boundary votes reached; where
#: the walk asks next; and that one pick, with where it asks after it (owner,
#: #5132: one pick, not a jump of four). No lights and no bar: the slide shows
#: the measure, not the rule. Per page: how many picks are in, and whether the
#: (?) is drawn.
SPAN_PAGES = {1: (None, False), 2: (0, False), 3: (0, True), 4: (1, True)}
#: The runs the notes quote: from the top to the first empty cell, after the
#: Boundary votes and after the pick.
SPAN_RUNS = {0: 3, 1: 6}


def _span_cells() -> list[tuple[int, int]]:
    return [(level, i) for level in range(SPAN_LEVELS) for i in range(SPAN_K**level)]


def _span_ancestors(cell: tuple[int, int]) -> list[tuple[int, int]]:
    level, i = cell
    out = [cell]
    while level > 0:
        level, i = level - 1, i // SPAN_K
        out.append((level, i))
    return out


def _span_covered(picks_in: int | None) -> tuple[set[tuple[int, int]], int, list[tuple[int, int]]]:
    """The cells holding a vote (each with its path), the walk's run, and the picks Diversity made.

    *picks_in* is how many picks Diversity has made; None is the empty atlas,
    before any vote.
    """
    covered: set[tuple[int, int]] = set()
    picks: list[tuple[int, int]] = []
    if picks_in is not None:
        for cell in SPAN_BOUNDARY_VOTES:
            covered.update(_span_ancestors(cell))
        order = _span_cells()
        while len(picks) < picks_in:
            # The walk takes the first empty cell; the vote there marks the
            # leaf its item sits in, and every cell from that leaf up to the root.
            empty = next(cell for cell in order if cell not in covered)
            leaf = SPAN_PICK_LEAVES[empty]
            assert empty in _span_ancestors(leaf), (empty, leaf)
            picks.append(empty)
            covered.update(_span_ancestors(leaf))
    run = 0
    for cell in _span_cells():
        if cell not in covered:
            break
        run += 1
    return covered, run, picks


def span_fig() -> None:
    """Span: the first 40 cells of an atlas, the cells the votes reach, and where the walk asks.

    Four stages: the empty cells; after asking at the line, a few darkened; the
    first empty cell, where the walk asks; and its pick there, a path down 02,
    021, 0212, with the next ask (owner, #4884, #5132).
    """
    for picks_in, run in SPAN_RUNS.items():
        _covered, got, picks = _span_covered(picks_in)
        assert got == run and len(picks) == picks_in, (picks_in, got)
    # Every pick the slide draws has its leaf named, and no leaf is named for a cell the walk never picks.
    assert set(_span_covered(max(SPAN_RUNS))[2]) == set(SPAN_PICK_LEAVES)
    assert SPAN_GREEN_DEFAULT == len(_span_cells()) == 40
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


#: A cell's radius, and the (?) drawn in place of the cell the walk asks next:
#: the deck's mark for where a step asks, as Flight Plan draws it.
SPAN_CELL_R = 0.13


def _span_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    xy = _span_xy()
    for (level, i), (x, y) in xy.items():
        if level:
            px, py = xy[(level - 1, i // SPAN_K)]
            ax.plot([px, x], [py, y], color=EDGE, linewidth=1.6, zorder=2)
    for level, y in enumerate(SPAN_LEVEL_Y):
        count = SPAN_K**level
        ax.text(0.7, y, f"{count} cell" if count == 1 else f"{count} cells", ha="left", va="center", fontsize=15,
                color=SOFT)  # fmt: skip
    for i in range(SPAN_K ** (SPAN_LEVELS - 1)):
        x, y = xy[(SPAN_LEVELS - 1, i)]
        ax.plot([x - 0.08, x, x + 0.08], [y - 0.42, y, y - 0.42], color=EDGE, linewidth=1.3, zorder=1)
    picks_in, asking = SPAN_PAGES[stage]
    covered, run, _picks = _span_covered(picks_in)
    ask = _span_cells()[run] if asking else None
    for cell, (x, y) in xy.items():
        if cell == ask:
            _ask(ax, x, y)
            continue
        face = INK if cell in covered else "white"
        ax.add_patch(Circle((x, y), SPAN_CELL_R, facecolor=face, edgecolor=SOFT, linewidth=1.4, zorder=4))
    return fig


# ── Cleared to Land: Done ────────────────────────────────────────────────────

DONE_STAGES = 3
#: The three endings, side by side: each column's left edge, and the heights of
#: its heading, its picture and its two lines of words. The picture sits low
#: enough that the first column clears the title notch.
DONE_COL_X = (0.75, 4.85, 8.95)
DONE_HEAD_Y, DONE_PICTURE_Y, DONE_WORDS_Y = 4.75, 3.95, (3.15, 2.72)
#: A vote in the Documents and Anything pictures: a disc in its class's hue,
#: red a Bad and green a Good, as everywhere in the deck.
DONE_VOTE_R, DONE_VOTE_PITCH = 0.085, 0.215
#: Anything's collection, every item voted: a grid of Goods and Bads.
DONE_GRID_ROWS, DONE_GRID_COLS, DONE_GRID_GOOD = 3, 14, 0.3


def done_fig() -> None:
    """The ways a run ends: every light green on photos, a document walk run dry, every item voted.

    Toasty's hand-off says the same thing whichever ending it was (#5143), so
    the slide draws the endings alone. Three stages, one ending each.
    """
    _save(DONE_STAGES, _done_stage, "autopilot-done")


def _done_column(ax: plt.Axes, x: float, head: str, words: tuple[str, str]) -> None:
    ax.text(x, DONE_HEAD_Y, head, ha="left", va="center", fontsize=20, fontweight="bold")
    for y, line in zip(DONE_WORDS_Y, words, strict=True):
        ax.text(x, y, line, ha="left", va="center", fontsize=17)


def _vote(ax: plt.Axes, x: float, y: float, good: bool) -> None:
    ax.add_patch(Circle((x, y), DONE_VOTE_R, facecolor=GREEN if good else RED, edgecolor="none", zorder=6))


def _done_stage(stage: int) -> Figure:
    fig, ax = _canvas()
    _statement(ax, "Three ways a run ends")
    # Photos: every light green.
    x = DONE_COL_X[0]
    _done_column(ax, x, "Photos", ("all three lights", "are green"))
    for k, word in enumerate(("Smart", "Stable", "Span")):
        lx = x + 0.15 + k * 1.15
        _light(ax, lx, DONE_PICTURE_Y, "green")
        ax.text(lx + 0.2, DONE_PICTURE_Y, word, ha="left", va="center", fontsize=16)
    if stage >= 2:
        # Documents: the walk's best matches, a run of Bads as long as the dry-run stop.
        x = DONE_COL_X[1]
        _done_column(ax, x, "Documents", (f"{MORE_DRY_RUN} best matches in a row,", "not one of them Good"))
        for k in range(MORE_DRY_RUN):
            _vote(ax, x + DONE_VOTE_R + k * DONE_VOTE_PITCH, DONE_PICTURE_Y, good=False)
    if stage >= 3:
        # Anything: nothing left to show.
        x = DONE_COL_X[2]
        _done_column(ax, x, "Anything", ("every item", "has a vote"))
        good = np.random.default_rng(42).random((DONE_GRID_ROWS, DONE_GRID_COLS)) < DONE_GRID_GOOD
        for row in range(DONE_GRID_ROWS):
            for col in range(DONE_GRID_COLS):
                y = DONE_PICTURE_Y + (1 - row) * DONE_VOTE_PITCH
                _vote(ax, x + DONE_VOTE_R + col * DONE_VOTE_PITCH, y, good=bool(good[row, col]))
    return fig


# ── Past the Exit: where the stop fired, measured ────────────────────────────

STOP_STAGES = 2
#: The three radios, as the review's `beta` column names them, and what the slide calls each
#: (`make-sota-figs.RADIOS`).
STOP_RADIOS = ((0.25, "β = ¼"), (1.0, "β = 1"), (4.0, "β = 4"))
#: The review's click budget, where its stopping table reads a session's end: the "vs 150" the slide quotes.
STOP_BUDGET = 150
#: Every session stops by the app's cap (#4903), whatever its lights say, so the chart runs that far (owner,
#: #5171). #5134's sessions ran to it; the per-click table the review commits stops at its 150-click budget,
#: and past that the curve is the study's own every-5-click table for today's app plus the A/B's paired
#: difference for the shipped arm (`_stop_curves`).
STOP_AXIS = STOP_CAP
REPORT_STUDY = REPORT_STOP.parent / "tables"

#: The numbers the slide prints and the notes quote, pinned per radio as (median stop, median Fβ at the
#: budget minus at the stop): `main` fails if the review's committed table says anything else, so a new
#: State of the App review is a re-run of this script and then a look at the notes.
EXPECT = {0.25: (54, 0.019), 1.0: (59, 0.026), 4.0: (64, 0.030)}


def _stop_rows() -> dict[float, dict[str, str]]:
    """The review's stopping table over every session, per radio (`stops_by_preset.py`)."""
    path = REPORT_STOP / "stopping_by_preset.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(_REPO_ROOT)} is missing: it is the review's stopping table")
    with path.open() as handle:
        return {float(row["beta"]): row for row in csv.DictReader(handle) if row["scope"] == "all"}


def _stop_curves() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """The objective a user has in hand at every click to the cap, mean over every session, per radio.

    To click 150 it is the review's per-click table (`by_click.py`). Past it, the same sessions read from
    #5134's every-5-click table of the reading pass, which is today's app before the switch, plus the A/B's
    paired difference of the shipped arm against it, click by click. The two agree where they overlap, and
    the splice asserts it.
    """
    path = REPORT_STOP / "objective_by_click.csv"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(_REPO_ROOT)} is missing: it is the review's per-click objective")
    curves: dict[float, list[tuple[float, float]]] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            curves.setdefault(float(row["beta"]), []).append((float(row["t"]), float(row["fbeta"])))
    with (REPORT_STUDY / "over_clicks.csv").open() as handle:
        control = {(float(r["beta"]), int(r["click"])): float(r["f"]) for r in csv.DictReader(handle)}
    with gzip.open(REPORT_STUDY / "ab_curves.csv.gz", "rt") as handle:
        shift = {
            (float(r["beta"]), int(r["click"])): float(r["d_f"])
            for r in csv.DictReader(handle)
            if r["arm"] == "fbeta_best"
        }
    out = {}
    for beta, pts in curves.items():
        t, f = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        assert t[-1] == STOP_BUDGET, (beta, t[-1])
        joined = control[(beta, STOP_BUDGET)] + shift[(beta, STOP_BUDGET)]
        assert abs(joined - f[-1]) < 0.001, (beta, joined, f[-1])
        later = np.arange(STOP_BUDGET + 5, STOP_AXIS + 1, 5)
        out[beta] = (
            np.concatenate([t, later]),
            np.concatenate([f, [control[(beta, c)] + shift[(beta, c)] for c in later]]),
        )
    return out


def stop_fig() -> None:
    """Where today's app told its sessions they could stop, and what that cost against clicking on.

    Two stages: the objective along a session at each radio to the cap, with each radio's median stop; then
    the median session's F-beta at the stop against at the review's budget.
    """
    rows, curves = _stop_rows(), _stop_curves()
    got = {
        beta: (round(float(r["km_t_stop"])), round(float(r["median_fbeta_delta"]), 3)) for beta, r in rows.items()
    }
    assert got == EXPECT, got
    _save(STOP_STAGES, lambda stage: _stop_stage(stage, rows, curves), "autopilot-stop")


#: The table of medians under the title notch, in figure fractions: each column's left edge, the header's
#: height, the rule under it, and the rows below.
STOP_TABLE_X = (0.04, 0.14, 0.255)
STOP_TABLE_HEAD_Y, STOP_TABLE_RULE_Y, STOP_TABLE_ROW_Y, STOP_TABLE_ROW_GAP = 0.66, 0.63, 0.575, 0.065


def _stop_stage(stage: int, rows: dict, curves: dict) -> Figure:
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0.43, 0.14, 0.42, 0.68))
    ax.set_xlim(0, STOP_AXIS)
    ax.set_ylim(0.3, 0.7)
    ax.set_xticks([0, 50, 100, STOP_BUDGET, STOP_AXIS])
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
        ax.text(STOP_AXIS + 4, y, words, ha="left", va="center", fontsize=16, clip_on=False)
    ax.text(min(stops.values()), 0.705, "○ the stop, median click", ha="left", va="bottom", fontsize=15,
            color=SOFT, clip_on=False)  # fmt: skip
    heads = ["Median", "Stop"] + (["Fβ vs 150"] if stage >= 2 else [])
    if stage >= 2:
        ax.axvline(STOP_BUDGET, color=SOFT, linewidth=1.4, linestyle=(0, (5, 4)), zorder=2)
    for x, head in zip(STOP_TABLE_X, heads, strict=False):
        fig.text(x, STOP_TABLE_HEAD_Y, head, ha="left", va="center", fontsize=17, fontweight="bold")
    fig.add_artist(plt.Line2D([STOP_TABLE_X[0], 0.36], [STOP_TABLE_RULE_Y] * 2, color=INK, linewidth=1.2))
    for k, (beta, words) in enumerate(STOP_RADIOS):
        y = STOP_TABLE_ROW_Y - k * STOP_TABLE_ROW_GAP
        fig.text(STOP_TABLE_X[0], y, words, ha="left", va="center", fontsize=16)
        fig.text(STOP_TABLE_X[1], y, f"{stops[beta]:.0f} clicks", ha="left", va="center", fontsize=16)
        if stage >= 2:
            # The stop against the budget, from the stop's side: what stopping there gave up.
            cost = -float(rows[beta]["median_fbeta_delta"])
            fig.text(STOP_TABLE_X[2], y, _signed2(cost), ha="left", va="center", fontsize=16)
    return fig


def _signed2(value: float) -> str:
    """A signed number with a real minus sign, two decimals."""
    return f"{value:+.2f}".replace("-", "\u2212")


if __name__ == "__main__":
    steps_fig()
    smart_fig()
    stable_fig()
    span_fig()
    done_fig()
    stop_fig()
    print("wrote figures to", OUT)
