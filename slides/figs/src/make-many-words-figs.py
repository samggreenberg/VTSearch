#!/usr/bin/env python
"""A typed-query detector's line, before and after it started from the typed words too (#4384, #4859).

Run from the repo root:

    python slides/figs/src/make-many-words-figs.py

**In So Many Words** (section XII, after Running Start) is Head Start's schematic again, drawn for a
detector made from typed text. The user typed "a chair"; the opening voted the text sort's top three
Good and four from its middle Bad. Both panels show the same plane of photos, the same votes:

- *Before*, the head started from the Goods' average (Head Start). The three Goods are the three chairs
  the words ranked first, which all sit at one end of the chairs, so their average points there: the
  line keeps that end and misses most of the chairs.
- *After*, the head started from the typed words as well. The words point at the chairs as a whole, so
  the line starts out facing all of them and turns only where the votes pay for it.

It is a drawing, not a fit, for Head Start's reason: in two dimensions three points and a word barely
disagree; the trouble needs the room hundreds of dimensions give it. So the panels draw the mechanism
each head reduces to with a few votes, on planted points, and `_check` asserts what the notes say: the
Goods are the words' top three; the before line misses most of the chairs; the after line keeps every
chair and nothing else; neither keeps anything that is not a chair. Pages: a, the votes; b, the line from the Goods' average; c, the
line from the words too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, INK, SOFT, save  # noqa: E402

SRC = Path(__file__).resolve().parent
OUT = SRC.parent

#: Head Start's palette (`make-head-start-figs.py`): blue the detector's line, ghost a line as it was,
#: green the Good side, red the Bad side.
BLUE = "#0b5fa5"
BAND = "#dce7f2"
GHOST = "#aab3c0"
GHOST_BAND = "#eef0f3"
RED = "#b91c1c"
GREEN = "#0d8a5f"
WIRE = "#98a2b0"

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 17,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)

#: The canvas: the whole slide, at 100 slide pixels to the inch.
FIG_W, FIG_H = 12.8, 7.2
#: A panel's plane, in its own units (aspect matches the axes box below), as Head Start's.
PLANE_W, PLANE_H = 10.0, 8.4
#: Item radius, in plane units.
R = 0.21
#: The middle of everything: where the Goods' average and the words are measured from.
MIDDLE = np.array([4.6, 3.9])


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


#: The typed words' direction, "a chair": at the chairs as a whole.
WORDS = _unit(np.array([1.0, 1.0]))
_ACROSS = np.array([-WORDS[1], WORDS[0]])
_CENTRE = MIDDLE + 3.95 * WORDS
#: The chairs, along an arc across the words' direction. Three near the lower-right end sit a little
#: further along the words, so they are the text sort's top three: those are the Goods.
CHAIRS = np.array(
    [
        _CENTRE + t * _ACROSS + s * WORDS
        for t, s in (
            (-2.6, 0.5),
            (-1.95, 1.15),
            (-1.35, 0.4),
            (-3.15, 0.05),
            (0.85, 0.0),
            (1.3, -0.04),
            (1.75, -0.1),
            (2.2, -0.2),
        )
    ]
)
GOODS, HIDDEN = CHAIRS[:3], CHAIRS[3:]
#: Four from the middle of the text sort, voted Bad.
BADS = np.array([[5.3, 6.15], [7.95, 3.4], [3.5, 6.3], [7.05, 4.3]])
#: Everything else, never voted.
OTHERS = np.array(
    [
        [1.0, 7.4],
        [2.2, 6.9],
        [3.4, 7.7],
        [1.6, 5.8],
        [2.9, 5.5],
        [4.6, 7.6],
        [3.9, 4.8],
        [1.0, 4.2],
        [1.9, 3.2],
        [1.5, 2.3],
        [3.0, 2.2],
        [0.9, 0.9],
        [2.5, 0.8],
        [4.1, 1.1],
        [3.7, 3.1],
        [6.4, 3.95],
        [5.85, 6.25],
        [5.6, 2.35],
        [7.4, 1.3],
        [8.1, 0.7],
        [1.9, 8.1],
        [8.9, 1.2],
        [7.0, 2.2],
        [8.3, 2.6],
        [4.3, 8.05],
    ]
)


def _line(direction: np.ndarray) -> tuple[np.ndarray, float]:
    """``(u, c)``: the line ``x · u = c`` facing *direction*, midway between the lowest Good and the highest Bad."""
    u = _unit(direction)
    return u, float((GOODS @ u).min() + (BADS @ u).max()) / 2


BEFORE = _line(GOODS.mean(0) - MIDDLE)
AFTER = _line(WORDS)


def _kept(line: tuple[np.ndarray, float], pts: np.ndarray) -> np.ndarray:
    u, c = line
    return pts @ u > c


def _check() -> None:
    """The drawing says what the notes say."""
    rank = np.argsort(-((np.vstack([CHAIRS, BADS, OTHERS]) - MIDDLE) @ WORDS))
    if set(rank[:3].tolist()) != {0, 1, 2}:
        raise SystemExit("In So Many Words: the Goods should be the words' top three")
    if _kept(BEFORE, HIDDEN).sum() * 2 > len(HIDDEN) or _kept(BEFORE, OTHERS).any():
        raise SystemExit("In So Many Words: the before line should miss most of the chairs, and keep nothing else")
    if not _kept(AFTER, HIDDEN).all() or _kept(AFTER, OTHERS).any() or _kept(AFTER, BADS).any():
        raise SystemExit("In So Many Words: the after line should keep every chair and nothing else")


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
    ax.plot(*MIDDLE, marker="+", markersize=20, mew=3.0, color=INK, zorder=6)
    ax.text(*(MIDDLE + np.array([-0.4, 0.0])), "middle", ha="right", va="center", fontsize=18, color=INK)


def _mean_mark(ax, p: np.ndarray) -> None:
    ax.plot(*p, marker="D", markersize=13, color=GREEN, markeredgecolor="white", markeredgewidth=1.5, zorder=6)


def _arrow(ax, a: np.ndarray, b: np.ndarray, color: str, lw: float = 3.0) -> None:
    ax.add_patch(
        FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=26, lw=lw, color=color, zorder=5, shrinkA=9, shrinkB=11)
    )


def _words(ax) -> None:
    """The typed words' arrow, named at its tail, where nothing else is."""
    tip = MIDDLE + 3.45 * WORDS
    _arrow(ax, MIDDLE, tip, INK, lw=3.4)
    ax.text(*(MIDDLE + np.array([0.15, -0.45])), "“a chair”", ha="left", va="top", fontsize=19, fontweight="bold",
            color=INK, zorder=6)  # fmt: skip


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


def _many_words(stage: int) -> Figure:
    """Stage 1: the votes; 2: the line from the Goods' average; 3: the line from the words too."""
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    left = _panel(fig, [0.035, 0.085, 0.455, 0.665], "Before: from the Goods' average")
    _items(left)
    if stage >= 2:
        u, c = BEFORE
        left.add_patch(Polygon(_kept_side(u, c), closed=True, facecolor=GHOST_BAND, edgecolor="none", zorder=0))
        a, b = _clip_line(u, c)
        left.plot([a[0], b[0]], [a[1], b[1]], color=GHOST, lw=4.0, ls=(0, (6, 3)), zorder=2)
        g = GOODS.mean(0)
        _mean_mark(left, g)
        _arrow(left, MIDDLE, g, SOFT)
    if stage >= 3:
        right = _panel(fig, [0.51, 0.085, 0.455, 0.665], "After: from the words too")
        _items(right)
        u, c = AFTER
        right.add_patch(Polygon(_kept_side(u, c), closed=True, facecolor=BAND, edgecolor="none", zorder=0))
        a, b = _clip_line(u, c)
        right.plot([a[0], b[0]], [a[1], b[1]], color=BLUE, lw=4.0, zorder=2)
        _mean_mark(right, GOODS.mean(0))
        _arrow(right, MIDDLE, GOODS.mean(0), SOFT)
        _words(right)
    _key(fig)
    return fig


def _key(fig: Figure) -> None:
    """One key for both panels, on every page, in the band right of the title notch."""
    ax = fig.add_axes([0.305, 0.80, 0.66, 0.08])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y, x, ms = 0.5, 0.0, 21
    entries = [("good", "Good"), ("bad", "Bad"), ("hidden", "chair"), ("other", "something else"),
               ("mean", "Goods' average")]  # fmt: skip
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]
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
            ax.plot(x + 0.01, y, marker="D", markersize=13, color=GREEN, mec="white", transform=ax.transAxes)
            x += 0.035
        t = ax.text(x, y, label, ha="left", va="center", fontsize=17, color=INK, transform=ax.transAxes)
        bb = t.get_window_extent(renderer)
        x = float(to_ax.transform((bb.x1, bb.y0))[0]) + 0.04


def main() -> int:
    _check()
    # Saved at the canvas's declared bounds, so every page frames the plane identically.
    save(_many_words(3), OUT, "many-words.webp", column=FULL_BLEED, tight=False)
    for stage in (1, 2):
        save(_many_words(stage), OUT, f"many-words.build{stage}.webp", column=FULL_BLEED, tight=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
