#!/usr/bin/env python
"""The detector's head, a linear SVM: what it fits, and how fast (#5132).

Run from the repo root:

    python slides/figs/src/make-svm-figs.py             # draw both slides from the committed timings
    python slides/figs/src/make-svm-figs.py --measure   # re-time the shipped code first, then draw

Two slides, at the end of section II (Decision), after Pics on a Plane has shown a detector as a
boundary that moves with every vote.

**Margin Call** is what that boundary is. Pages: a, the votes on a plane; b, three lines that each get
every vote right, so the votes alone do not pick one; c, the SVM's line, the one with the widest street
between the Goods and the Bads, with the votes on the curb ringed, since they are the only ones that set
it; d, every other photo scored by which side of the line it is on and how far, one dot product each.

Page c takes the candidate lines away again. That is the one break from "a reveal adds ink"
(`slides/STYLE.md`), and it is the mechanism: of all those lines, this one. Everything else is pinned
across the four pages.

Unlike Pics on a Plane, the line is a straight line, because in the app it is one: a linear SVM in
SigLIP's 768 dimensions draws a flat hyperplane, and two dimensions draw a flat line. It is a real fit,
`sklearn`'s linear SVC on the votes drawn (a near-hard margin, so the curb is exact), and `_check`
asserts the claims the notes make: every candidate line separates the votes, the ringed votes are the
fit's support vectors and sit on the curb, and the scored photos land where their labels say. The
shipped head is the soft kind, started from the Goods' average (`vtscore/training/svm_prior.py`); Head
Start, with faces, is the slide for that.

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
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC))
from slide_figure import FULL_BLEED, INK, SOFT, save  # noqa: E402

TIMINGS = SRC / "svm-timings.json"

#: The palette of Pics on a Plane (`make-intro-figs.py`): blue is the detector's line, the shipped
#: decision, and its street the band; green the Good side, red the Bad side; ghost a line the slide
#: is not choosing.
BLUE = "#0b5fa5"
BAND = "#dce7f2"
GHOST = "#aab3c0"
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
#: Both slides carry one-line headlines, so both take the deck plane's measured one-line notch
#: (`make-intro-figs.VOTE_NOTCH_PX`, see `slides/STYLE.md`) rather than the two-line reserve.
NOTCH = (60.0, 42.0, 300.0, 88.0)

# ─────────────────────────────────────────────────────────────────────────────────────── Margin Call

#: The plane, in its own units: the whole slide, one unit to 80 slide pixels, so the title notch
#: (`NOTCH`, one line tall under "Margin Call") is x 0.75-4.5, y above 7.9.
PLANE_W, PLANE_H = 16.0, 9.0
NOTCH_X1, NOTCH_Y0 = 4.5, 7.9
#: Item radius, in plane units, and the check and cross drawn to the same half-width.
R = 0.21
#: The votes. Bads to the left, Goods to the right, with a gap a line can pass through at many angles.
BADS = np.array([[2.3, 1.5], [3.9, 2.6], [1.9, 3.9], [3.6, 4.9], [5.2, 1.3], [5.9, 3.9], [2.8, 6.3], [5.6, 6.6]])
GOODS = np.array([[9.4, 2.4], [11.0, 1.3], [12.6, 3.0], [10.4, 4.6], [13.7, 5.0], [11.8, 6.4], [9.7, 7.1],
                  [14.3, 1.9]])  # fmt: skip
#: Three lines that each get every vote right, as (a point on the line, its angle in degrees).
CANDIDATES = (((7.3, 4.5), 62.0), ((7.4, 4.5), 102.0), ((7.75, 4.4), 80.0))
#: The photos nobody voted on, scored on page d: two in the street, where the line cannot tell, and
#: the rest out on either side. Each score is written over its photo, so a near-upright line never
#: runs through one.
UNVOTED = np.array([[1.3, 2.6], [4.6, 3.7], [4.2, 0.7], [1.3, 5.3], [7.09, 5.6], [8.53, 2.9], [10.6, 6.0],
                    [12.4, 4.6], [14.6, 3.5], [13.0, 7.3]])  # fmt: skip
#: How far above its photo's top a score label sits, in plane units.
SCORE_DY = 0.1
MARGIN_STAGES = 4
LABEL_PT = 15


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

    # ── c: the widest street, set by the votes on its curb ───────────────────────────────────────
    if stage >= 3:
        w = model.coef_[0]
        b0 = float(model.intercept_[0])
        norm = float(np.linalg.norm(w))
        normal = w / norm
        line = _clip_line(normal, -b0 / norm)
        lower = _clip_line(normal, (-1 - b0) / norm)
        upper = _clip_line(normal, (1 - b0) / norm)
        ax.add_patch(Polygon([lower[0], lower[1], upper[1], upper[0]], closed=True, facecolor=BAND,
                             edgecolor="none", zorder=1))  # fmt: skip
        for edge in (lower, upper):
            ax.plot(edge[:, 0], edge[:, 1], color=BLUE, lw=1.8, ls=(0, (5, 4)), zorder=2)
        ax.plot(line[:, 0], line[:, 1], color=BLUE, lw=3.4, zorder=3)
        for p in model.support_vectors_:
            ax.add_patch(Circle(tuple(p), 2.1 * R, facecolor="none", edgecolor=INK, lw=2.0, zorder=5))

    # ── d: everything else, scored by its side and its distance ──────────────────────────────────
    if stage >= 4:
        for p in UNVOTED:
            ax.add_patch(Circle(tuple(p), R, facecolor="white", edgecolor=INK, lw=1.8, zorder=4))
            s = _score(model, p)
            ax.text(p[0], p[1] + R + SCORE_DY, f"{s:+.1f}".replace("-", "−"), ha="center", va="bottom",
                    fontsize=LABEL_PT, color=SOFT, zorder=6)  # fmt: skip
        # Top right: the bottom-right corner is where the slide prints its page number.
        ax.text(PLANE_W - 0.45, PLANE_H - 0.45, "score = w · x + b", ha="right", va="top", fontsize=20, color=INK,
                fontweight="bold", zorder=6)  # fmt: skip
    return fig


# ────────────────────────────────────────────────────────────────────────────────────── Speed Reading

#: SigLIP's width, and the rows each measurement is repeated over (the median is drawn).
DIM = 768
VOTE_COUNTS = (10, 20, 50, 100, 200, 500)
PILE_SIZES = (1_000, 10_000, 100_000, 1_000_000)
FIT_REPEATS, SCORE_REPEATS = 15, 5
#: What one click costs: the two calibration folds and the final head, each fitted, and the pile scored
#: by each (`vtscore/training/thresholds/conformal.py`, `calibrate_count=2`).
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
        ("Fit the line", "votes", fit, (6, 800), (10, 50, 100, 500), ("10", "50", "100", "500")),
        ("Score the pile", "photos", score, (400, 2.4e6), PILE_SIZES, ("1k", "10k", "100k", "1M")),
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
    save(_margin_call(MARGIN_STAGES, model), OUT, "svm-margin.webp", column=FULL_BLEED, tight=False, notch=NOTCH)
    for stage in range(1, MARGIN_STAGES):
        save(_margin_call(stage, model), OUT, f"svm-margin.build{stage}.webp", column=FULL_BLEED, tight=False,
             notch=NOTCH)  # fmt: skip
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
