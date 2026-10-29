#!/usr/bin/env python
"""The detector's head, a linear SVM: what it fits, and how fast (#5132).

Run from the repo root:

    python slides/figs/src/make-svm-figs.py             # draw both slides from the committed timings
    python slides/figs/src/make-svm-figs.py --measure   # re-time the shipped code first, then draw

Two slides, at the end of section II (Decision), after Pics on a Plane has shown a detector as a
boundary that moves with every vote.

**Marginal at Best** (once Margin Call; owner, #5171) is what that boundary is. Pages: a, the votes on a
plane; b, three lines that each get every vote right, so the votes alone do not pick one; c, the SVM's
line, the one with the widest street between the Goods and the Bads, with the votes on the curb ringed,
since they are the only ones that set it; d, the line alone, with the photos nobody voted on; e, each of
those scored by which side of the line it is on and how far, drawn as the perpendicular to the line,
one dot product each (owner, #5171).

Page c takes the candidate lines away again, and page d the street, its curbs and the rings. Those are
the breaks from "a reveal adds ink" (`slides/STYLE.md`), and they are the mechanism: of all those lines,
this one; and then, of all that geometry, only the line scores anything. Everything else is pinned across
the five pages.

Unlike Pics on a Plane, the line is a straight line, because in the app it is one: a linear SVM in
SigLIP's 768 dimensions draws a flat hyperplane, and two dimensions draw a flat line. It is a real fit,
`sklearn`'s linear SVC on the votes drawn (a near-hard margin, so the curb is exact), and `_check`
asserts the claims the notes make: every candidate line separates the votes, the ringed votes are the
fit's support vectors and sit on the curb, and the scored photos land where their labels say. The
shipped head is the soft kind, started from the Goods' average (`vtscore/training/svm_prior.py`); Head
Start, with faces, is the slide for that.

**Off With Their Heads** (owner, #5171) answers the question the room asks next: why a straight line?
The same votes, fitted by five heads, each drawing its own boundary: the linear SVM that ships, and the
four others tried over the head's life. The logistic regression, the MLP (the head until #2790) and the
Gaussian process have reports under `docs/experiments/` (#3197, #4114, #4219; 2026-07-22; #3954, #3959),
which the notes summarise; the random forest is the owner's. Every boundary is a real `sklearn` fit,
drawn where its probability (the SVM: its score) crosses even, over the field of that probability.

**Speed Reading** is the shipped code timed: `fit_linear_svm_prior_head`, the head every production fit
trains, against the number of votes; and `score_rows_with_model`, the pass every Find and every retrain
makes over the pile, against the pile's size. 768-d unit vectors, as SigLIP's are; the votes are 40%
Goods, shifted along one direction, the rest scattered. Every click fits three of these heads (the two
calibration folds and the final one) and scores the pile three times, which the notes add up. Page a,
the fit; b, the scoring; c, a blink, for scale.

The timings are a measurement of whatever machine ran ``--measure``, so they are committed beside this
file (`svm-timings.json`, with the date and the machine's core count) and the chart is drawn from them:
re-running the drawing does not move a number, and the slide's date is the date they were taken.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.gaussian_process import GaussianProcessClassifier  # noqa: E402
from sklearn.gaussian_process.kernels import RBF  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC))
from slide_figure import BOUNDARY_DOTS, BOUNDARY_LW, CUT_WASH, FULL_BLEED, INK, SOFT, save  # noqa: E402

TIMINGS = SRC / "svm-timings.json"

#: The palette of Pics on a Plane (`make-intro-figs.py`): the detector's line in round black dots
#: (`slide_figure.BOUNDARY_LW`), as a plane draws a threshold (#5206), and its street a neutral
#: wash with grey curbs; green the Good side, red the Bad side; ghost a line the slide is not
#: choosing. Blue is left only on Speed Reading, where it is the shipped code's timings, not a line.
BLUE = "#0b5fa5"
BAND = CUT_WASH
GHOST = "#aab3c0"
#: The line, wherever a slide here draws a head's boundary.
LINE = {"color": INK, "linestyle": BOUNDARY_DOTS, "dash_capstyle": "round"}
RED = "#b91c1c"
GREEN = "#0d8a5f"

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

#: The canvas: the whole slide, at 100 slide pixels to the inch.
FIG_W, FIG_H = 12.8, 7.2
#: Speed Reading carries a one-line headline, so it takes the deck plane's measured one-line notch
#: (`make-intro-figs.VOTE_NOTCH_PX`, see `slides/STYLE.md`); "Marginal at Best" takes two lines and
#: `save()`'s standard reserve.
NOTCH = (60.0, 42.0, 300.0, 88.0)

# ────────────────────────────────────────────────────────────────────────────────── Marginal at Best

#: The plane, in its own units: the whole slide, one unit to 80 slide pixels, so the title notch
#: (the standard two-line reserve under "Marginal at Best") is x 0.75-4.5, y above 6.85.
PLANE_W, PLANE_H = 16.0, 9.0
NOTCH_X1, NOTCH_Y0 = 4.5, 6.85
#: Item radius, in plane units, and the check and cross drawn to the same half-width.
R = 0.21
#: The votes. Bads to the left, Goods to the right, with a gap a line can pass through at many angles.
BADS = np.array([[2.3, 1.5], [3.9, 2.6], [1.9, 3.9], [3.6, 4.9], [5.2, 1.3], [5.9, 3.9], [2.8, 6.3], [5.6, 6.6]])
GOODS = np.array([[9.4, 2.4], [11.0, 1.3], [12.6, 3.0], [10.4, 4.6], [13.7, 5.0], [11.8, 6.4], [9.7, 7.1],
                  [14.3, 1.9]])  # fmt: skip
#: Three lines that each get every vote right, as (a point on the line, its angle in degrees).
CANDIDATES = (((7.3, 4.5), 62.0), ((7.4, 4.5), 102.0), ((7.75, 4.4), 80.0))
#: The photos nobody voted on, shown on page d and scored on page e: two in the street, where the line
#: cannot tell, and the rest out on either side. Each score is written over its photo, so a near-upright
#: line never runs through one, and each photo has a clear lane to the line for its perpendicular
#: (`_check`): no vote, photo or score in the way.
UNVOTED = np.array([[1.27, 2.33], [4.58, 3.54], [4.2, 0.7], [1.34, 5.48], [7.09, 5.6], [8.53, 2.9], [10.6, 6.0],
                    [12.39, 4.91], [14.6, 3.5], [13.01, 7.32]])  # fmt: skip
#: How far above its photo's top a score label sits, in plane units.
SCORE_DY = 0.1
MARGIN_STAGES = 5
LABEL_PT = 15
#: How far a perpendicular keeps from any other item, beyond that item's radius; and from a score.
LANE_CLEAR, LANE_LABEL_CLEAR = 0.2, 0.1
#: A score's box over its photo, for those clearances: its width, and its height above the gap.
SCORE_W, SCORE_H = 0.9, 0.42
#: The square in the corner where a perpendicular meets the line, its side in plane units.
RIGHT_ANGLE = 0.16


def _fit() -> SVC:
    """The max-margin line through the votes: a linear SVC with a near-hard margin."""
    X = np.vstack([BADS, GOODS])
    y = np.r_[np.zeros(len(BADS)), np.ones(len(GOODS))]
    return SVC(kernel="linear", C=1e4).fit(X, y)


def _score(model: SVC, p: np.ndarray) -> float:
    """The fit's score for *p*: which side of the line, and how far, in streets (the curb is ±1)."""
    return float(model.decision_function(np.atleast_2d(p))[0])


def _line_through(point: tuple[float, float], angle: float) -> tuple[np.ndarray, float]:
    """A line as (unit normal, offset), so that ``normal @ p - offset`` is the signed distance."""
    theta = np.radians(angle)
    direction = np.array([np.cos(theta), np.sin(theta)])
    normal = np.array([direction[1], -direction[0]])
    return normal, float(normal @ np.asarray(point))


def _foot(model: SVC, p: np.ndarray) -> np.ndarray:
    """Where the perpendicular from *p* meets the fit's line."""
    w = model.coef_[0]
    norm = float(np.linalg.norm(w))
    return p - (float(w @ p) + float(model.intercept_[0])) / norm * (w / norm)


def _lane(model: SVC, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The perpendicular page e draws for *p*: from its circle's edge to the line."""
    foot = _foot(model, p)
    return p + (foot - p) / np.linalg.norm(foot - p) * R, foot


def _score_box(p: np.ndarray) -> tuple[float, float, float, float]:
    """The box a score written over *p* fills, as (x0, y0, x1, y1)."""
    y0 = p[1] + R + SCORE_DY
    return p[0] - SCORE_W / 2, y0, p[0] + SCORE_W / 2, y0 + SCORE_H


def _box_gap(box: tuple[float, float, float, float], q: np.ndarray) -> float:
    dx = max(box[0] - q[0], 0.0, q[0] - box[2])
    dy = max(box[1] - q[1], 0.0, q[1] - box[3])
    return float(np.hypot(dx, dy))


def _along(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a + np.linspace(0.0, 1.0, 80)[:, None] * (b - a)


def _check() -> SVC:
    """Every claim the notes make about the drawing, asserted on the fit."""
    model = _fit()
    for point, angle in CANDIDATES:
        normal, offset = _line_through(point, angle)
        bad, good = BADS @ normal - offset, GOODS @ normal - offset
        assert (bad.max() < -R and good.min() > R) or (bad.min() > R and good.max() < -R), (point, angle)
    on_curb = np.abs(np.abs(model.decision_function(np.vstack([BADS, GOODS]))) - 1.0) < 1e-3
    assert on_curb.sum() == len(model.support_) and 2 <= len(model.support_) <= 4, model.support_
    scores = [_score(model, p) for p in UNVOTED]
    assert sum(-1 < s < 1 for s in scores) == 2 and min(scores) < -1 and max(scores) > 1, scores
    # A label over a photo in the street is about 0.8 units wide: it has to clear the line and the curb.
    half = 1 / float(np.linalg.norm(model.coef_[0]))
    for s in scores:
        if -1 < s < 1:
            assert 0.45 < abs(s) * half < half - 0.45, s
    for p in np.vstack([BADS, GOODS, UNVOTED]):
        assert not (p[0] < NOTCH_X1 + R and p[1] > NOTCH_Y0 - R), p
    # Page e: every perpendicular has a clear lane to the line, past every other item and every score,
    # and no score sits across the line.
    items = np.vstack([BADS, GOODS, UNVOTED])
    w, b0 = model.coef_[0], float(model.intercept_[0])
    for i, p in enumerate(UNVOTED):
        lane = _along(*_lane(model, p))
        others = [q for q in items if not np.array_equal(q, p)]
        for q in others:
            gap = float(np.min(np.linalg.norm(lane - q, axis=1)))
            assert gap > R + LANE_CLEAR, (p, q, gap)
        for j, q in enumerate(UNVOTED):
            if j != i:
                assert min(_box_gap(_score_box(q), x) for x in lane) > LANE_LABEL_CLEAR, (p, q)
        box = _score_box(p)
        corners = [np.array([x, y]) for x in box[::2] for y in box[1::2]]
        assert len({np.sign(w @ c + b0) for c in corners}) == 1, ("a score crosses the line", p)
        assert all(_box_gap(box, q) > R for q in others), ("a score covers an item", p)
    return model


def _clip_line(normal: np.ndarray, offset: float) -> np.ndarray:
    """The two ends of ``normal @ p = offset`` where it leaves the plane."""
    # Parametrise along the line's direction and clip to the plane's box.
    direction = np.array([-normal[1], normal[0]])
    base = normal * offset
    ts = []
    for axis, lo, hi in ((0, 0.0, PLANE_W), (1, 0.0, PLANE_H)):
        if abs(direction[axis]) > 1e-9:
            ts += [(lo - base[axis]) / direction[axis], (hi - base[axis]) / direction[axis]]
    pts = [base + t * direction for t in ts]
    pts = [p for p in pts if -1e-6 <= p[0] <= PLANE_W + 1e-6 and -1e-6 <= p[1] <= PLANE_H + 1e-6]
    pts.sort(key=lambda p: p[1])
    return np.array([pts[0], pts[-1]])


def _cross(ax: plt.Axes, p: np.ndarray) -> None:
    for dx in (-1, 1):
        ax.plot([p[0] - dx * R, p[0] + dx * R], [p[1] - R, p[1] + R], color=RED, lw=3.4, solid_capstyle="round",
                zorder=6)  # fmt: skip


def _tick(ax: plt.Axes, p: np.ndarray) -> None:
    x, y = p
    ax.plot([x - R, x - 0.28 * R, x + R], [y + 0.05 * R, y - 0.85 * R, y + R], color=GREEN, lw=3.6,
            solid_capstyle="round", solid_joinstyle="miter", zorder=6)  # fmt: skip


def _margin_line(model: SVC) -> tuple[np.ndarray, np.ndarray]:
    """The fit's unit normal, and its line's two ends where it leaves the plane."""
    w = model.coef_[0]
    norm = float(np.linalg.norm(w))
    return w / norm, _clip_line(w / norm, -float(model.intercept_[0]) / norm)


def _margin_street(ax: plt.Axes, model: SVC) -> None:
    """Page c's street: the band between the curbs, the curbs grey and dashed, and the votes on them ringed.

    The curbs are not cuts, so they take neither the line's dots nor its ink: they are the edges of
    its band, which is a neutral wash (#5206).
    """
    w = model.coef_[0]
    b0 = float(model.intercept_[0])
    norm = float(np.linalg.norm(w))
    lower = _clip_line(w / norm, (-1 - b0) / norm)
    upper = _clip_line(w / norm, (1 - b0) / norm)
    ax.add_patch(Polygon([lower[0], lower[1], upper[1], upper[0]], closed=True, facecolor=BAND, edgecolor="none",
                         zorder=1))  # fmt: skip
    for edge in (lower, upper):
        ax.plot(edge[:, 0], edge[:, 1], color=SOFT, lw=1.8, ls=(0, (5, 4)), zorder=2)
    for p in model.support_vectors_:
        ax.add_patch(Circle(tuple(p), 2.1 * R, facecolor="none", edgecolor=INK, lw=2.0, zorder=5))


def _margin_scores(ax: plt.Axes, model: SVC, normal: np.ndarray) -> None:
    """Page e: every unvoted photo's perpendicular to the line, its right angle, and its score."""
    direction = np.array([-normal[1], normal[0]])
    for p in UNVOTED:
        start, foot = _lane(model, p)
        ax.plot([start[0], foot[0]], [start[1], foot[1]], color=SOFT, lw=1.6, zorder=2)
        # The right angle where it meets the line, on the photo's side.
        toward = (p - foot) / np.linalg.norm(p - foot) * RIGHT_ANGLE
        along = direction * RIGHT_ANGLE
        corner = np.array([foot + along, foot + along + toward, foot + toward])
        ax.plot(corner[:, 0], corner[:, 1], color=SOFT, lw=1.2, zorder=2)
        s = _score(model, p)
        ax.text(p[0], p[1] + R + SCORE_DY, f"{s:+.1f}".replace("-", "−"), ha="center", va="bottom",
                fontsize=LABEL_PT, color=SOFT, zorder=6)  # fmt: skip
    # Top right: the bottom-right corner is where the slide prints its page number.
    ax.text(PLANE_W - 0.45, PLANE_H - 0.45, "score = w · x + b", ha="right", va="top", fontsize=20, color=INK,
            fontweight="bold", zorder=6)  # fmt: skip


def _margin_call(stage: int, model: SVC) -> Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, PLANE_W)
    ax.set_ylim(0, PLANE_H)
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ── a: the votes ─────────────────────────────────────────────────────────────────────────────
    for p in BADS:
        _cross(ax, p)
    for p in GOODS:
        _tick(ax, p)

    # ── b: three lines, each right about every vote ──────────────────────────────────────────────
    if stage == 2:
        for point, angle in CANDIDATES:
            a, b = _clip_line(*_line_through(point, angle))
            ax.plot([a[0], b[0]], [a[1], b[1]], color=GHOST, lw=2.6, zorder=2)

    # ── c: the widest street, set by the votes on its curb; from d, the line alone ─────────────────
    if stage >= 3:
        normal, line = _margin_line(model)
        if stage == 3:
            _margin_street(ax, model)
        ax.plot(line[:, 0], line[:, 1], lw=BOUNDARY_LW, zorder=3, **LINE)

    # ── d: the photos nobody voted on ────────────────────────────────────────────────────────────
    if stage >= 4:
        for p in UNVOTED:
            ax.add_patch(Circle(tuple(p), R, facecolor="white", edgecolor=INK, lw=1.8, zorder=4))

    # ── e: each scored by its side and its distance, straight across to the line ─────────────────
    if stage >= 5:
        _margin_scores(ax, model, normal)
    return fig


# ──────────────────────────────────────────────────────────────────────────────── Off With Their Heads

#: The five heads, in reading order: the one the current app uses first, then the others tried. Each is its name on
#: the slide and a factory for the fit, seeded where the fit is random.
HEADS = (
    ("Linear SVM (Current app)", lambda: SVC(kernel="linear", C=1e4)),
    ("Logistic regression", lambda: LogisticRegression(C=1.0)),
    ("MLP", lambda: MLPClassifier(hidden_layer_sizes=(32, 32), solver="lbfgs", max_iter=5000, random_state=0)),
    ("Random forest", lambda: RandomForestClassifier(n_estimators=25, max_features=1, random_state=0)),
    ("Gaussian process", lambda: GaussianProcessClassifier(1.0 * RBF(2.0), random_state=0)),
)
#: The slots, as (left, bottom) in inches on the slide: a 3x2 grid whose top-left slot is the title's, so
#: the heads take the other five, and each panel's plane and its name above it.
HEADS_SLOTS = ((4.55, 3.85), (8.45, 3.85), (0.65, 0.35), (4.55, 0.35), (8.45, 0.35))
HEADS_PANEL_W = 3.7
HEADS_PANEL_H = HEADS_PANEL_W * PLANE_H / PLANE_W
HEADS_NAME_PT = 18
#: Under the title, in its slot: what the five panels have in common.
HEADS_STATEMENT = ("Same votes,", "five heads")
HEADS_STATEMENT_XY = (0.65, (4.85, 4.3))


#: Each panel is shaded by what its head thinks of every point on the plane, from the Bad side's red
#: through white at even odds to the Good side's green, because on votes this clean every head draws
#: nearly the same line: what tells them apart is the field around it (the forest's boxes, the Gaussian
#: process going back to even odds away from the votes, the logistic's long ramp).
HEADS_FIELD = LinearSegmentedColormap.from_list("heads", ["#f2c9c9", "#ffffff", "#c6e8d9"])


def _heads_field(model, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """The fit's probability of Good over a grid of the plane; the SVM's score through a logistic."""
    grid = np.c_[xs.ravel(), ys.ravel()]
    if isinstance(model, SVC):
        return (1.0 / (1.0 + np.exp(-2.0 * model.decision_function(grid)))).reshape(xs.shape)
    return model.predict_proba(grid)[:, 1].reshape(xs.shape)


def _off_with_their_heads() -> Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    X = np.vstack([BADS, GOODS])
    y = np.r_[np.zeros(len(BADS)), np.ones(len(GOODS))]
    xs, ys = np.meshgrid(np.linspace(0, PLANE_W, 400), np.linspace(0, PLANE_H, 225))
    titles = []
    for (name, make), (left, bottom) in zip(HEADS, HEADS_SLOTS, strict=True):
        model = make().fit(X, y)
        assert (model.predict(X) == y).all(), name
        ax = fig.add_axes((left / FIG_W, bottom / FIG_H, HEADS_PANEL_W / FIG_W, HEADS_PANEL_H / FIG_H))
        ax.set_xlim(0, PLANE_W)
        ax.set_ylim(0, PLANE_H)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_color("#d8dee6")
        ships = isinstance(model, SVC)
        field = _heads_field(model, xs, ys)
        ax.imshow(field, extent=(0, PLANE_W, 0, PLANE_H), origin="lower", cmap=HEADS_FIELD, vmin=0.0, vmax=1.0,
                  interpolation="nearest", zorder=1)  # fmt: skip
        # Every head's boundary is a threshold, so each is drawn as one (#5206); which head ships is
        # its bold name, not its line.
        boundary = ax.contour(xs, ys, field, levels=[0.5], colors=[INK], linewidths=BOUNDARY_LW,
                              linestyles=[BOUNDARY_DOTS], zorder=3)  # fmt: skip
        boundary.set_capstyle("round")
        for p in BADS:
            _cross(ax, p)
        for p in GOODS:
            _tick(ax, p)
        title = ax.set_title(name, loc="left", fontsize=HEADS_NAME_PT, fontweight="bold" if ships else "normal", pad=8)
        titles.append((ax, title))
    x, (y0, y1) = HEADS_STATEMENT_XY
    for yy, words in zip((y0, y1), HEADS_STATEMENT, strict=True):
        fig.text(x / FIG_W, yy / FIG_H, words, ha="left", va="center", fontsize=20)
    # Each head's name stays over its own panel, so it cannot run into the next one's.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, title in titles:
        assert title.get_window_extent(renderer).x1 <= ax.get_window_extent(renderer).x1, title.get_text()
    return fig


# ────────────────────────────────────────────────────────────────────────────────────── Speed Reading

#: SigLIP's width, and the rows each measurement is repeated over (the median is drawn).
DIM = 768
VOTE_COUNTS = (10, 20, 50, 100, 200, 500)
PILE_SIZES = (1_000, 10_000, 100_000, 1_000_000)
FIT_REPEATS, SCORE_REPEATS = 15, 5
#: What one click costs: the two calibration folds and the final head, each fitted, and the pile scored
#: by each (`vtscore/training/thresholds/fold_orderings.py`, `calibrate_count=2`).
FITS_PER_CLICK = SCORE_PASSES_PER_CLICK = 3
#: The points that carry their number: every pile size, and the vote counts the notes quote.
LABELLED = {10, 50, 500, *PILE_SIZES}
#: A blink, for scale: about a tenth of a second at its quickest.
BLINK_MS = 100.0
SPEED_STAGES = 3


def _unit(a: np.ndarray) -> np.ndarray:
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def measure() -> dict:
    """Time the shipped fit and scoring pass on this machine, and return the table `TIMINGS` holds."""
    from vtscore.detectors.training import ScoringRows, score_rows_with_model  # noqa: PLC0415
    from vtscore.training.svm_prior import fit_linear_svm_prior_head  # noqa: PLC0415

    rng = np.random.default_rng(42)
    direction = _unit(rng.standard_normal((1, DIM)))[0]

    def votes(n: int) -> tuple[np.ndarray, np.ndarray]:
        goods = max(3, round(0.4 * n))
        X = _unit(rng.standard_normal((n, DIM)))
        X[:goods] = _unit(X[:goods] + 0.75 * direction)
        y = np.zeros(n)
        y[:goods] = 1.0
        return X.astype(np.float32), y

    def median_ms(fn, repeats: int) -> float:
        fn()  # warm
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        return round(statistics.median(times) * 1000, 3)

    fit = {}
    for n in VOTE_COUNTS:
        X, y = votes(n)
        fit[str(n)] = median_ms(lambda X=X, y=y: fit_linear_svm_prior_head(X, y, DIM), FIT_REPEATS)
    X, y = votes(50)
    model = fit_linear_svm_prior_head(X, y, DIM)
    score = {}
    for size in PILE_SIZES:
        matrix = np.empty((size, DIM), dtype=np.float32)
        for start in range(0, size, 100_000):
            stop = min(size, start + 100_000)
            matrix[start:stop] = _unit(rng.standard_normal((stop - start, DIM)).astype(np.float32))
        rows = ScoringRows(list(range(size)), matrix, np.arange(size), np.zeros(size, dtype=np.int64))
        score[str(size)] = median_ms(lambda rows=rows: score_rows_with_model(model, rows), SCORE_REPEATS)
        del matrix, rows
    return {
        "measured": dt.date.today().isoformat(),
        "cpus": os.cpu_count(),
        "dim": DIM,
        "fit_ms_by_votes": fit,
        "score_ms_by_photos": score,
    }


def _ms_label(ms: float) -> str:
    return f"{ms / 1000:g} s" if ms >= 1000 else f"{ms:g} ms"


def _speed_reading(stage: int, timings: dict) -> Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # Two panels sharing one time axis, right of the title notch's column and full height below it.
    top, bottom, left, right, gap = 0.74, 0.14, 0.115, 0.95, 0.07
    w = (right - left - gap) / 2
    fit = {int(k): v for k, v in timings["fit_ms_by_votes"].items()}
    score = {int(k): v for k, v in timings["score_ms_by_photos"].items()}
    panels = (
        ("Training", "votes", fit, (6, 800), (10, 50, 100, 500), ("10", "50", "100", "500")),
        ("Scoring", "photos", score, (400, 2.4e6), PILE_SIZES, ("1k", "10k", "100k", "1M")),
    )
    for i, (title, xlabel, series, xlim, xticks, xticklabels) in enumerate(panels):
        ax = fig.add_axes((left + i * (w + gap), bottom, w, top - bottom))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(0.1, 2000)
        ax.xaxis.set_major_locator(FixedLocator(xticks))
        ax.set_xticklabels(xticklabels)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_locator(FixedLocator([1, 10, 100, 1000]))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_yticklabels(["1 ms", "10 ms", "100 ms", "1 s"] if i == 0 else [])
        ax.yaxis.grid(True, color="#e3e7ec", lw=1.0)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=19, pad=12)
        ax.set_xlabel(xlabel)
        if stage < i + 1:
            continue
        xs = sorted(series)
        ys = [series[x] for x in xs]
        ax.plot(xs, ys, color=BLUE, lw=3.0, marker="o", ms=8, zorder=4)
        for x, y in zip(xs, ys, strict=True):
            if x not in LABELLED:
                continue
            words = _ms_label(round(y, 1) if y < 10 else round(y))
            if i == 1 and x != xs[-1]:
                # The pile's line climbs steeply, so a number over its point sits on the next segment:
                # it goes below and right instead, under the climb.
                ax.text(x * 1.3, y / 1.3, words, ha="left", va="top", fontsize=15, color=SOFT, zorder=5)
            else:
                ax.text(x, y * 1.55, words, ha="center", va="bottom", fontsize=15, color=SOFT, zorder=5)
        if stage >= 3:
            ax.axhline(BLINK_MS, color=RED, lw=1.8, ls=(0, (6, 4)), zorder=3)
            if i == 1:
                ax.text(xlim[0] * 1.15, BLINK_MS * 0.78, "a blink", ha="left", va="top", fontsize=17, color=RED,
                        zorder=5)  # fmt: skip
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--measure", action="store_true", help="re-time the shipped code and rewrite the timings")
    args = parser.parse_args()
    if args.measure:
        TIMINGS.write_text(json.dumps(measure(), indent=2) + "\n")
        print("wrote", TIMINGS)
    model = _check()
    save(_margin_call(MARGIN_STAGES, model), OUT, "svm-margin.webp", column=FULL_BLEED, tight=False)
    for stage in range(1, MARGIN_STAGES):
        save(_margin_call(stage, model), OUT, f"svm-margin.build{stage}.webp", column=FULL_BLEED, tight=False)
    save(_off_with_their_heads(), OUT, "svm-heads.webp", column=FULL_BLEED, tight=False)
    timings = json.loads(TIMINGS.read_text())
    # Saved at the canvas's declared bounds, so every page frames the panels identically.
    save(_speed_reading(SPEED_STAGES, timings), OUT, "svm-speed.webp", column=FULL_BLEED, tight=False, notch=NOTCH)
    for stage in range(1, SPEED_STAGES):
        save(_speed_reading(stage, timings), OUT, f"svm-speed.build{stage}.webp", column=FULL_BLEED, tight=False,
             notch=NOTCH)  # fmt: skip
    print("wrote figures to", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
