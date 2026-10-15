#!/usr/bin/env python
"""The trained head, before and after it started from the Goods' average (#4776, #4785).

Run from the repo root (the results slide reads the #4776 report's committed curves):

    python slides/figs/src/make-head-start-figs.py

Two slides.

**Head Start** (section X, after the FHIBE card) is a schematic of how the detector's line is placed
once there are enough votes to train it, drawn twice on the same plane of items, the same three Goods
(a person's photos) and the same four Bads (four strangers):

- *Before*, the plain linear SVM. With a handful of votes in hundreds of dimensions it learns roughly
  "the Goods minus the Bads": the line faces from the Bads' average toward the Goods' average. Four
  strangers are a poor sample of everyone who is not this person, so their average sits wherever they
  happened to stand, and the line tilts with it: it keeps strangers off to one side and misses one of
  the person's own photos.
- *After*, the SVM started from the Goods' average. The direction from the middle of everything to the
  Goods' average costs the fit almost nothing, every other direction its full price, so the line
  starts out facing the Goods' average and turns only where the Bads pay for it.

It is a drawing, not a fit: in two dimensions the two fits barely differ, because the trouble is the
Bads' average pointing somewhere unrelated to the person, which takes the room hundreds of dimensions
give it. So the panels draw the mechanism each head reduces to with a few votes, on planted points, and
`_check` asserts the outcome the notes describe (the before line keeps strangers and misses one of the
person's photos; the after line keeps every photo and no stranger). Pages: a, the votes; b, the plain
SVM's line; c, the Goods' average; d, the line started from it.

**Running Start** (section X, after Head Start) is what that bought, from the report's
`prior_curves.csv`: what a Find returns at every vote, F-beta at β 1, today's head against the head
started from the Goods' average, on faces from 1 and from 4 starting photos (562 people) and on COCO
Better's Binary opening (720 runs; the 18 that never find a Good return nothing under either head and
count as zero). Page a draws the plain SVM alone; page b adds the head the app now ships.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save, spread_labels  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent
REPO = SRC.parents[2]
CURVES = REPO / "docs" / "experiments" / "2026-10-11-face-handover-4776" / "prior_curves.csv"

#: The palette of Pics on a Plane (`make-intro-figs.py`): blue is the detector's line, the shipped
#: decision; ghost is a line as it was; green the Good side, red the Bad side.
BLUE = "#0b5fa5"
BAND = "#dce7f2"
GHOST = "#aab3c0"
GHOST_BAND = "#eef0f3"
RED = "#b91c1c"
GREEN = "#0d8a5f"
WIRE = "#98a2b0"
#: The deck's `.cut` blue (`themes/vtsearch.css`) for the shipped line on a chart, and the grey of a
#: line the app no longer draws (Follow Suit's cross-calibration).
SHIPPED = "#2F6DB5"
GREY = "#8a929e"

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

# ---------------------------------------------------------------------------------------- Head Start

#: A panel's plane, in its own units (aspect matches the axes box below).
PLANE_W, PLANE_H = 10.0, 8.4
#: Item radius, in plane units.
R = 0.21
#: The middle of everything: where the Goods' average is measured from.
MIDDLE = np.array([4.6, 3.9])
#: Three of the person's photos, voted Good.
GOODS = np.array([[7.6, 6.7], [8.4, 6.2], [8.0, 7.2]])
#: The rest of the person's photos, never voted: what a Find should return.
HIDDEN = np.array([[8.8, 7.0], [7.15, 7.15], [8.2, 7.75], [9.4, 5.6]])
#: Four strangers, voted Bad: from the middle of the example sort, so somewhat like the person. Three
#: happen to stand below and right of them and one up and left, which is all a handful can be: their
#: average is wherever they happened to stand.
BADS = np.array([[9.0, 3.2], [6.9, 2.3], [9.1, 4.1], [5.0, 6.1]])
#: Everyone else, never voted.
OTHERS = np.array(
    [
        [1.0, 7.4],
        [2.2, 6.9],
        [3.4, 7.6],
        [1.6, 5.8],
        [2.9, 5.5],
        [4.4, 6.4],
        [5.6, 7.5],
        [3.9, 4.8],
        [1.0, 4.2],
        [1.9, 3.2],
        [1.5, 2.3],
        [3.0, 2.2],
        [0.9, 0.9],
        [2.5, 0.8],
        [4.1, 1.1],
        [3.7, 3.1],
        [5.5, 4.6],
        [6.4, 5.2],
        [5.9, 3.4],
        [7.4, 1.3],
        [8.6, 0.9],
        [9.4, 2.4],
        [7.6, 4.3],
        [5.5, 8.15],
        [4.9, 8.0],
        [1.9, 8.1],
        [9.6, 1.7],
        [6.6, 6.5],
    ]
)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _line(direction: np.ndarray, goods: np.ndarray, bads: np.ndarray) -> tuple[np.ndarray, float]:
    """``(u, c)``: the line ``x · u = c`` facing *direction*, midway between the lowest Good and the highest Bad."""
    u = _unit(direction)
    return u, float((goods @ u).min() + (bads @ u).max()) / 2


BEFORE = _line(GOODS.mean(0) - BADS.mean(0), GOODS, BADS)
AFTER = _line(GOODS.mean(0) - MIDDLE, GOODS, BADS)


def _kept(line: tuple[np.ndarray, float], pts: np.ndarray) -> np.ndarray:
    u, c = line
    return pts @ u > c


def _check() -> None:
    """The drawing says what the notes say: before keeps strangers and misses a photo; after neither."""
    if _kept(BEFORE, HIDDEN).all() or not _kept(BEFORE, OTHERS).any():
        raise SystemExit("Head Start: the before line should miss one of the person's photos and keep strangers")
    if not _kept(AFTER, HIDDEN).all() or _kept(AFTER, OTHERS).any():
        raise SystemExit("Head Start: the after line should keep every photo of the person and no stranger")


def _clip_line(u: np.ndarray, c: float) -> tuple[np.ndarray, np.ndarray]:
    """The segment of ``x · u = c`` inside the plane."""
    pts = []
    for x in (0.0, PLANE_W):
        if abs(u[1]) > 1e-9:
            y = (c - u[0] * x) / u[1]
            if 0 <= y <= PLANE_H:
                pts.append((x, y))
    for y in (0.0, PLANE_H):
        if abs(u[0]) > 1e-9:
            x = (c - u[1] * y) / u[0]
            if 0 <= x <= PLANE_W:
                pts.append((x, y))
    pts = sorted(set(pts))
    return np.array(pts[0]), np.array(pts[-1])


def _kept_side(u: np.ndarray, c: float) -> np.ndarray:
    """The polygon of the plane on the kept side of the line."""
    corners = [(0, 0), (PLANE_W, 0), (PLANE_W, PLANE_H), (0, PLANE_H)]
    a, b = _clip_line(u, c)
    poly = [np.array(p, dtype=float) for p in corners if np.dot(p, u) > c]
    poly += [a, b]
    centre = np.mean(poly, axis=0)
    return np.array(sorted(poly, key=lambda p: np.arctan2(p[1] - centre[1], p[0] - centre[0])))


def _items(ax) -> None:
    for p in OTHERS:
        ax.add_patch(Circle(p, R, facecolor="white", edgecolor=WIRE, lw=1.8, zorder=3))
    for p in HIDDEN:
        ax.add_patch(Circle(p, R, facecolor="white", edgecolor=GREEN, lw=2.6, zorder=3))
    for p in GOODS:
        ax.add_patch(Circle(p, R * 1.25, facecolor=GREEN, edgecolor=GREEN, lw=1.5, zorder=4))
        ax.text(*p, "✓", color="white", ha="center", va="center", fontsize=17, fontweight="bold", zorder=5)
    for p in BADS:
        ax.add_patch(Circle(p, R * 1.25, facecolor=RED, edgecolor=RED, lw=1.5, zorder=4))
        ax.text(*p, "✗", color="white", ha="center", va="center", fontsize=17, fontweight="bold", zorder=5)


def _mean_mark(ax, p: np.ndarray, color: str) -> None:
    ax.plot(*p, marker="D", markersize=13, color=color, markeredgecolor="white", markeredgewidth=1.5, zorder=6)


def _arrow(ax, a: np.ndarray, b: np.ndarray, color: str) -> None:
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=26, lw=3.0, color=color, zorder=5, shrinkA=9, shrinkB=11
        )  # fmt: skip
    )


def _panel(fig: Figure, rect: list[float], caption: str):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, PLANE_W)
    ax.set_ylim(0, PLANE_H)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Polygon([(0, 0), (PLANE_W, 0), (PLANE_W, PLANE_H), (0, PLANE_H)], closed=True, fill=False,
                         edgecolor="#d5dae1", lw=1.4, zorder=1))  # fmt: skip
    ax.text(0.0, -0.3, caption, ha="left", va="top", fontsize=20, fontweight="bold", color=INK)
    return ax


def _head_start(stage: int) -> Figure:
    """Stage 1: the votes; 2: the plain SVM's line; 3: the Goods' average; 4: the line started from it."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # Both panels sit below the title notch, captioned underneath, with one key in the band right of
    # the notch; nothing moves between pages.
    left = _panel(fig, [0.035, 0.085, 0.455, 0.665], "Before: the votes alone")
    _items(left)
    if stage >= 2:
        u, c = BEFORE
        left.add_patch(Polygon(_kept_side(u, c), closed=True, facecolor=GHOST_BAND, edgecolor="none", zorder=0))
        a, b = _clip_line(u, c)
        left.plot([a[0], b[0]], [a[1], b[1]], color=GHOST, lw=4.0, ls=(0, (6, 3)), zorder=2)
        g, bm = GOODS.mean(0), BADS.mean(0)
        _mean_mark(left, bm, RED)
        _mean_mark(left, g, GREEN)
        _arrow(left, bm, g, SOFT)
    if stage >= 3:
        right = _panel(fig, [0.51, 0.085, 0.455, 0.665], "After: from the Goods' average")
        _items(right)
        g = GOODS.mean(0)
        right.plot(*MIDDLE, marker="+", markersize=20, mew=3.0, color=INK, zorder=6)
        right.text(*(MIDDLE + np.array([-0.4, 0.0])), "middle", ha="right", va="center", fontsize=18, color=INK)
        _mean_mark(right, g, GREEN)
        _arrow(right, MIDDLE, g, SOFT)
        if stage >= 4:
            u, c = AFTER
            right.add_patch(Polygon(_kept_side(u, c), closed=True, facecolor=BAND, edgecolor="none", zorder=0))
            a, b = _clip_line(u, c)
            right.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=4.0, zorder=2)
    _key(fig)
    return fig


def _key(fig: Figure) -> None:
    """One key for both panels, on every page (it names what page a already draws)."""
    # In the band right of the title notch, which the panels' tops leave free.
    ax = fig.add_axes([0.305, 0.80, 0.66, 0.08])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y, x, ms = 0.5, 0.0, 21
    entries = [("good", "Good"), ("bad", "Bad"), ("hidden", "same person"), ("other", "someone else"),
               ("mean", "average")]  # fmt: skip
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_ax = ax.transAxes.inverted()
    for kind, label in entries:
        if kind in ("good", "bad"):
            color = GREEN if kind == "good" else RED
            ax.plot(x + 0.01, y, marker="o", markersize=ms, color=color, transform=ax.transAxes, clip_on=False)
            ax.text(x + 0.01, y, "✓" if kind == "good" else "✗", color="white", ha="center", va="center",
                    fontsize=15, fontweight="bold", transform=ax.transAxes)  # fmt: skip
            x += 0.04
        elif kind in ("hidden", "other"):
            ax.plot(x + 0.01, y, marker="o", markersize=ms - 3, mfc="white", mec=GREEN if kind == "hidden" else WIRE,
                    mew=2.6 if kind == "hidden" else 1.8, transform=ax.transAxes, clip_on=False)  # fmt: skip
            x += 0.04
        else:
            for dx, color in ((0.01, GREEN), (0.035, RED)):
                ax.plot(x + dx, y, marker="D", markersize=13, color=color, mec="white", transform=ax.transAxes)
            x += 0.06
        t = ax.text(x, y, label, ha="left", va="center", fontsize=17, color=INK, transform=ax.transAxes)
        bb = t.get_window_extent(renderer)
        x = float(to_ax.transform((bb.x1, bb.y0))[0]) + 0.04


# ------------------------------------------------------------------------------------- Running Start

#: (set in prior_curves.csv, panel title, runs counted, runs with rows): COCO's 18 runs a preset that
#: never find a Good return nothing under either head, so its curves are scaled to every run (#4631).
PANELS: tuple[tuple[str, str, int, int], ...] = (
    ("k1-b1-1024", "Faces, 1 starting photo", 562, 562),
    ("k4-b1-1024", "Faces, 4 starting photos", 562, 562),
    ("coco-b1", "COCO Better photos", 720, 702),
)


def read_curves(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``set -> (votes, today, new head)`` for Find's F-beta."""
    if not path.exists():
        raise SystemExit(f"{path.relative_to(REPO)} does not exist: it is the #4776 report's committed curves")
    by: dict[str, list[tuple[int, float, float]]] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row["metric"] == "F":
                by.setdefault(row["set"], []).append((int(row["vote"]), float(row["control"]), float(row["arm"])))
    out = {}
    for name, _title, n_all, n_rows in PANELS:
        if name not in by:
            raise SystemExit(f"{path}: no F curve for {name}")
        pts = sorted(by[name])
        scale = n_rows / n_all
        out[name] = (
            np.array([p[0] for p in pts]),
            np.array([p[1] for p in pts]) * scale,
            np.array([p[2] for p in pts]) * scale,
        )
    return out


def _running_start(curves: dict, new: bool) -> Figure:
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    # Three panels in a row, right of the title notch's column and full height below it.
    top, bottom, left, right, gap = 0.68, 0.13, 0.075, 0.86, 0.075
    w = (right - left - 2 * gap) / 3
    for i, (name, title, _n, _r) in enumerate(PANELS):
        ax = fig.add_axes([left + i * (w + gap), bottom, w, top - bottom])
        votes, today, head = curves[name]
        ax.set_xlim(0, 150)
        ax.set_ylim(0, 1.0 if name.startswith("k") else 0.6)
        ax.set_xticks([0, 50, 100, 150])
        ax.yaxis.grid(True, color="#e3e7ec", lw=1.0)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=18, pad=12)
        if i == 0:
            ax.set_ylabel("What Find returns: Fβ, β = 1")
        ax.set_xlabel("Votes")
        span = ax.get_ylim()[1] - ax.get_ylim()[0]
        lab = spread_labels([float(today[-1]), float(head[-1])], gap=0.09 * span)
        ax.plot(votes, today, color=GREY, lw=2.6, zorder=3)
        if i == len(PANELS) - 1:
            ax.text(152, lab[0], "plain SVM", color=INK, ha="left", va="center", clip_on=False)
        if new:
            ax.plot(votes, head, color=SHIPPED, lw=3.4, zorder=4)
            if i == len(PANELS) - 1:
                ax.text(152, lab[1], "head start", color=SHIPPED, fontweight="bold", ha="left", va="center",
                        clip_on=False)  # fmt: skip
    return fig


def main() -> int:
    _check()
    # Saved at the canvas's declared bounds, so every page frames the plane identically.
    save(_head_start(4), OUT, "head-start.webp", column=FULL_BLEED, tight=False)
    for stage in (1, 2, 3):
        save(_head_start(stage), OUT, f"head-start.build{stage}.webp", column=FULL_BLEED, tight=False)
    curves = read_curves(CURVES)
    save(_running_start(curves, new=True), OUT, "running-start.webp", column=FULL_BLEED, tight=False)
    save(_running_start(curves, new=False), OUT, "running-start.build1.webp", column=FULL_BLEED, tight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
