#!/usr/bin/env python
"""Coverage-atlas figures, for the appendix slides that explain it.

Run from the repo root:

    python slides/figs/src/make-atlas-figs.py

Four figures, and they are deliberately three different kinds of thing.

`atlas-blindspot` and `atlas-cells` are drawn on **the deck's own plane** — the
same blue-noise field, the same votes and the same fitted boundary that
`make-intro-figs.py` builds for *Pics on a Plane*. That is the whole reason
these two exist as figures rather than as bullets: the room has already been
taught to read this picture, so the atlas arrives as a second thing done to a
plane they know rather than as a new diagram to decode. The field is imported
from that generator rather than copied, and imported by *path* because the
module's name is not an identifier.

Both are drawings and not plots, exactly as the slide they extend says of
itself. The partition drawn on them is a real recursive k-means with the
atlas's own splitting rule (k = 3, stop under `MIN_NODE`), but it runs on the
two-dimensional positions. The shipped atlas centres and renormalises first and
partitions *directions* — a detail the deck deliberately does not teach, since
nothing the room has to follow turns on it.

`atlas-depth` is a schematic: a wireframe room whose floor is the space the
votes explored, and a detector boundary extruded up it because the shipped
head is a single linear layer and cannot do anything else. An earlier version
of this slide drew the centred sphere instead and was cut for being contrived
and hard to parse — a fair verdict, and the diagnosis worth keeping is that
points on a sphere give the eye nothing to judge position against. A floor,
four posts and a lid do.

`atlas-pvalues` is the only plot: it re-plots published numbers from the #3329
fit-quality study (`docs/experiments/2026-08-30-fit-quality-3329/`), read from
that report's aggregate CSVs at build time so the figure cannot drift from the
numbers the notes quote.
"""

import functools
import importlib.util
import math
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patheffects
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, save  # noqa: E402

OUT = Path(__file__).resolve().parent.parent
STUDY = _REPO_ROOT / "docs/experiments/2026-08-30-fit-quality-3329/agg"


def _intro():
    """The *Pics on a Plane* generator, imported by path.

    `make-intro-figs.py` is not an importable name, and renaming it would
    rewrite the history of a file whose figures are tuned by hand. Loading it
    by path costs four lines and touches nothing: its own figure writing is
    behind a `__main__` guard, so importing it draws nothing.
    """
    path = Path(__file__).resolve().parent / "make-intro-figs.py"
    spec = importlib.util.spec_from_file_location("intro_figs", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["intro_figs"] = module
    spec.loader.exec_module(module)
    return module


INTRO = _intro()
INK, SOFT, RULE = INTRO.INK, INTRO.SOFT, "#d8dee6"
#: The cell outlines. `RULE` is the theme's hairline grey and is meant to sit
#: under text on a screen; a partition drawn in it disappears off a projector,
#: and the partition is the figure.
CELL_LINE = "#aeb8c6"
BLUE, RED, GREEN = INTRO.BLUE, INTRO.RED, INTRO.GREEN
CANVAS, UNIT_PT, R = INTRO.CANVAS, INTRO.UNIT_PT, INTRO.R
#: The two figures drawn on the deck's plane share that plane's slide slot and
#: its headline shape, so they take its measured one-line notch rather than the
#: deck's two-line reserve — see `slides/STYLE.md`. Both headlines are one line;
#: gaining a second would mean re-measuring, and `build.py --check` would not
#: catch it, because a figure under a two-line headline is legal and merely
#: wrong. The other two figures here are drawings of my own with room to spare,
#: so they keep clear of the full reserve and take the standard.
NOTCH = INTRO.VOTE_NOTCH_PX

#: Type sizes, in points at `UNIT_PT` per drawing unit. The floor is 20 slide
#: pixels and this canvas renders at about 1.74 px/pt, so nothing may go below
#: ~11.5pt; `slide_figure.save` refuses a figure that does.
LABEL_PT = 15
NOTE_PT = 13

#: A caption on the plane figures sits *over* a field of items — there is no
#: margin to put it in, because the field is the slide. A white halo lets it
#: cross a circle without either of them becoming unreadable, which a plain
#: label does not: the deck's own rule is that a label binds to the thing it is
#: nearest, and a caption tangled in three circles binds to all of them.
HALO = [patheffects.withStroke(linewidth=6.0, foreground="white")]

#: The atlas's own splitting rule, at the scale this drawing works on. The
#: shipped values are k = 3 and `min_node_size = 20` against tens of thousands
#: of items; the field here holds ninety-eight, so the floor comes down with it
#: and the branching factor — which is what the picture is about — does not.
#:
#: The floor is what sets how many cells the drawing ends up with, and that
#: number has to keep the *ratio* the slide is about: the shipped atlas builds
#: thousands of cells and a session casts tens of votes, so almost every cell
#: is unexplored. At a floor of eight this field gave nine cells against eleven
#: votes and the picture said the opposite — one white cell in a grey page.
SPLIT_K = 3
MIN_NODE = 4


# ──────────────────────────────────────────────────────────────────────────────
# The plane, and a partition of it
# ──────────────────────────────────────────────────────────────────────────────


def _canvas() -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


#: One level of the partition: which cell each item is in, where each cell's
#: centroid is, and which cell of the level above each one came out of.
class _Level:
    def __init__(self, cells: np.ndarray, centres: np.ndarray, parents: np.ndarray) -> None:
        self.cells, self.centres, self.parents = cells, centres, parents


@functools.lru_cache(maxsize=1)
def _partition() -> tuple[_Level, ...]:
    """Recursive k-means over the field, one `_Level` per depth.

    The atlas's own rule, at this drawing's scale: split every cell into
    `SPLIT_K`, stop splitting one that could not give each child `MIN_NODE`
    items, stop entirely when no cell split. Depth 0 is the whole field.
    """
    pts, *_ = INTRO._scene()
    levels = [_Level(np.zeros(len(pts), dtype=int), pts.mean(axis=0)[None, :], np.array([-1]))]
    while True:
        cells = np.full(len(pts), -1)
        centres: list[np.ndarray] = []
        parents: list[int] = []
        for cell in range(len(levels[-1].centres)):
            members = np.flatnonzero(levels[-1].cells == cell)
            if len(members) < MIN_NODE * SPLIT_K:
                cells[members] = len(centres)
                centres.append(pts[members].mean(axis=0))
                parents.append(cell)
                continue
            fit = KMeans(n_clusters=SPLIT_K, n_init=10, random_state=0).fit(pts[members])
            for child in range(SPLIT_K):
                cells[members[fit.labels_ == child]] = len(centres)
                centres.append(fit.cluster_centers_[child])
                parents.append(cell)
        if len(centres) == len(levels[-1].centres):
            return tuple(levels)
        levels.append(_Level(cells, np.array(centres), np.array(parents)))


def _cell_raster(depth: int, steps: int = 760) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Which cell every point of the canvas falls in, at *depth*.

    Rasterised rather than solved, because the cells of a hierarchical k-means
    are nested perpendicular-bisector regions and their exact polygons are
    fiddly for a picture that only needs their outlines. Routing a canvas point
    is the same walk the atlas does with a query vector: descend into the
    nearest child, level by level, never reconsidering the branch.
    """
    xs = np.linspace(0, CANVAS[0], steps)
    ys = np.linspace(0, CANVAS[1], round(steps * CANVAS[1] / CANVAS[0]))
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    live = np.zeros(len(grid), dtype=int)
    for level in _partition()[1 : depth + 1]:
        nxt = np.empty(len(grid), dtype=int)
        for parent in np.unique(live):
            here = np.flatnonzero(live == parent)
            span = np.flatnonzero(level.parents == parent)
            distance = np.linalg.norm(grid[here][:, None, :] - level.centres[span][None, :, :], axis=2)
            nxt[here] = span[distance.argmin(axis=1)]
        live = nxt
    return xs, ys, live.reshape(len(ys), len(xs))


def _draw_cells(ax: plt.Axes, depth: int, *, color: str = CELL_LINE, width: float = 1.5) -> None:
    xs, ys, cells = _cell_raster(depth)
    for cell in np.unique(cells):
        ax.contour(xs, ys, (cells == cell).astype(float), levels=[0.5], colors=[color], linewidths=width, zorder=1)


def _leaf_depth() -> int:
    """The deepest level the splitting rule reached."""
    return len(_partition()) - 1


def _cells_at(depth: int) -> np.ndarray:
    """Each item's cell id at *depth*, clamped to the deepest level built."""
    return _partition()[min(depth, _leaf_depth())].cells


# ──────────────────────────────────────────────────────────────────────────────
# 1. The blind spot — what asking at the line cannot reach
# ──────────────────────────────────────────────────────────────────────────────

BLINDSPOT_STAGES = 4
#: How wide the ring of never-asked items is drawn, in canvas units.
RING_R = 1.55


@functools.lru_cache(maxsize=1)
def _ring() -> tuple[np.ndarray, tuple[int, ...]]:
    """Where the never-asked pocket goes, and which items fall in it.

    Chosen rather than placed: the disc is swept over the canvas and scored by
    how many items it holds, subject to holding no vote, keeping clear of the
    boundary the loop is asking along, and staying out of the title's corner.
    Picking it by hand would invite the suspicion that the figure's whole claim
    was arranged, and the constraint that it contain no vote is the claim.
    """
    pts, _first, second, curve, *_ = INTRO._scene()
    seed_good, seed_bad = INTRO._seed_votes()
    voted = pts[list(seed_good + seed_bad)]
    best, best_count = None, -1
    margin = RING_R + 1.15  # room for the ring's own caption under it
    for x in np.arange(margin, CANVAS[0] - margin, 0.1):
        for y in np.arange(margin, CANVAS[1] - margin, 0.1):
            centre = np.array([x, y])
            if x - RING_R < 4.2 and y + RING_R > 7.1:
                continue  # the headline's corner
            if np.hypot(*(curve - centre).T).min() < RING_R + 0.9:
                continue  # must be nowhere near the line the loop asks along
            if np.hypot(*(voted - centre).T).min() < RING_R + 0.5:
                continue  # and must hold no vote, which is the whole point
            count = int((np.hypot(*(pts - centre).T) < RING_R).sum())
            if count > best_count:
                best, best_count = centre, count
    if best is None or best_count < 4:
        raise SystemExit("no unexplored pocket clear of the boundary — the field moved")
    inside = tuple(int(i) for i in np.flatnonzero(np.hypot(*(pts - best).T) < RING_R))
    return best, inside


def _blindspot_stage(stage: int) -> plt.Figure:
    pts, _first, second, _curve, _after, asked, _again, labeled = INTRO._scene()
    seed_good, seed_bad = INTRO._seed_votes()
    fig, ax = _canvas()

    if stage >= 2:
        INTRO._band(ax, second, INTRO._band_width(second, pts, labeled + (asked,)))
    INTRO._boundary(ax, second)

    centre, inside = _ring()
    if stage >= 3:
        ax.add_patch(
            plt.Circle(
                tuple(centre),
                RING_R,
                facecolor="none",
                edgecolor=GREEN if stage >= 4 else SOFT,
                linewidth=2.4,
                linestyle=(0, (6, 5)),
                zorder=2,
            )
        )

    voted = {**{i: "good" for i in seed_good + (asked,)}, **{i: "bad" for i in seed_bad}}
    for i, p in enumerate(pts):
        if voted.get(i) == "good":
            INTRO._check(ax, p)
        elif voted.get(i) == "bad":
            INTRO._cross(ax, p)
        elif stage >= 4 and i in inside:
            # A match nobody has voted on and the detector rejects: the same
            # hollow circle as its neighbours, filled in the colour the deck
            # reserves for the Good side. Not a check — nobody has said so.
            ax.add_patch(plt.Circle(tuple(p), R, facecolor=GREEN, edgecolor=GREEN, linewidth=1.7, zorder=4))
        else:
            INTRO._circle(ax, p)

    if stage >= 2:
        ax.text(
            CANVAS[0] - 0.45,
            0.5,
            "every question the loop asks comes from this strip",
            color=BLUE,
            fontsize=LABEL_PT,
            ha="right",
            va="baseline",
            zorder=7,
            path_effects=HALO,
        )
    if stage >= 3:
        ax.text(
            centre[0],
            centre[1] - RING_R - 0.52,
            "and nothing has ever asked in here" if stage == 3 else "which is full of books",
            color=SOFT if stage == 3 else GREEN,
            fontsize=LABEL_PT,
            fontweight="normal" if stage == 3 else "bold",
            ha="center",
            va="baseline",
            zorder=7,
            path_effects=HALO,
        )
    return fig


def blindspot_fig() -> None:
    """Why coverage is a separate question from uncertainty.

    Four pages: the loop as *Pics on a Plane* left it; the strip it asks in;
    a region of the same collection it has never asked in; and what is in
    there. The last page is the argument — the detector is not uncertain about
    that pocket, it is confidently wrong about it, and no amount of asking
    along the boundary will ever produce a question there.
    """
    for stage in range(1, BLINDSPOT_STAGES):
        save(
            _blindspot_stage(stage),
            OUT,
            f"atlas-blindspot.build{stage}.png",
            column=FULL_BLEED,
            tight=False,
            notch=NOTCH,
        )
    save(_blindspot_stage(BLINDSPOT_STAGES), OUT, "atlas-blindspot.png", column=FULL_BLEED, tight=False, notch=NOTCH)


# ──────────────────────────────────────────────────────────────────────────────
# 2. The atlas — a partition that remembers where you have been
# ──────────────────────────────────────────────────────────────────────────────

CELLS_STAGES = 6


def _scores() -> np.ndarray:
    """A sort score per item: the toy detector's own signed distance.

    The probe below is the shipped rule, and the shipped rule reads the sort
    scores the app is showing. Here that is the second fit — the detector as
    the loop left it — so the picture's probe is picked from the same numbers
    its boundary is drawn from rather than from a second invented quantity.
    """
    pts, _first, second, *_ = INTRO._scene()
    return second.decision_function(pts)


def _next_cell() -> int:
    """The cell the walk stops at: the largest one carrying no vote.

    Breadth-first from the root, biggest sibling first, first cell with no
    evidence wins — which at one level of leaves is just "largest unvoted".
    """
    seed_good, seed_bad = INTRO._seed_votes()
    _pts, _first, _second, _curve, _after, asked, _again, _labeled = INTRO._scene()
    voted = set(seed_good + seed_bad + (asked,))
    cells = _cells_at(_leaf_depth())
    order = sorted(set(cells.tolist()), key=lambda c: -int((cells == c).sum()))
    for cell in order:
        if not (voted & set(np.flatnonzero(cells == cell).tolist())):
            return cell
    raise SystemExit("every cell carries a vote — the drawing has nothing to explore")


def _probe(cell: int) -> int:
    """Which item of *cell* the atlas asks about: the shipped surprise probe.

    The region's median score decides what the model presumes about it, and
    the probe is the element that would most cheaply prove the presumption
    wrong: the lowest-scored item of a region it calls good, the highest-scored
    item of one it calls bad. Informative whichever way the answer lands.
    """
    scores = _scores()
    members = np.flatnonzero(_cells_at(_leaf_depth()) == cell)
    presumed_good = float(np.median(scores[members])) >= 0.0
    return int(members[scores[members].argmin() if presumed_good else scores[members].argmax()])


def _cells_stage(stage: int) -> plt.Figure:
    pts, _first, _second, _curve, _after, asked, _again, _labeled = INTRO._scene()
    seed_good, seed_bad = INTRO._seed_votes()
    fig, ax = _canvas()

    if stage >= 2:
        _draw_cells(ax, 1 if stage == 2 else _leaf_depth())

    cell = _next_cell()
    probe = _probe(cell)

    if stage >= 4:
        # The evidence channels: a cell somebody has voted in is a cell the
        # walk will not stop at, so the wash is the whole state the atlas keeps.
        voted_cells = {int(_cells_at(_leaf_depth())[i]) for i in seed_good + seed_bad + (asked,)}
        xs, ys, raster = _cell_raster(_leaf_depth())
        covered = np.isin(raster, sorted(voted_cells)).astype(float)
        ax.contourf(xs, ys, covered, levels=[0.5, 1.5], colors=["#eef1f5"], zorder=0)

    if stage >= 5:
        xs, ys, raster = _cell_raster(_leaf_depth())
        chosen = (raster == cell).astype(float)
        ax.contourf(xs, ys, chosen, levels=[0.5, 1.5], colors=[INTRO.BAND], zorder=0)
        ax.contour(xs, ys, chosen, levels=[0.5], colors=[BLUE], linewidths=3.0, zorder=3)

    voted = {**{i: "good" for i in seed_good + (asked,)}, **{i: "bad" for i in seed_bad}}
    for i, p in enumerate(pts):
        if stage >= 4 and voted.get(i) == "good":
            INTRO._check(ax, p)
        elif stage >= 4 and voted.get(i) == "bad":
            INTRO._cross(ax, p)
        elif stage >= 6 and i == probe:
            INTRO._circle(ax, p, glyph="?")
        else:
            INTRO._circle(ax, p)

    caption_xy = (CANVAS[0] - 0.45, CANVAS[1] - 0.72)
    captions = {
        2: "split the collection in three",
        3: "and again, until a cell is small",
        4: "every vote marks its cell, and every cell above it",
        5: "the walk stops at the biggest cell nobody has voted in",
        6: "and asks the one item most likely to prove that cell wrong",
    }
    if stage in captions:
        ax.text(
            *caption_xy,
            captions[stage],
            color=INK if stage >= 5 else SOFT,
            fontsize=LABEL_PT,
            fontweight="bold" if stage >= 5 else "normal",
            ha="right",
            va="baseline",
            zorder=7,
            path_effects=HALO,
        )
    return fig


def cells_fig() -> None:
    """The atlas itself, drawn on the plane the deck already taught.

    Six pages: the collection; one split; the recursion; the votes landing in
    cells; the walk stopping at the biggest cell with no evidence in it; and
    the surprise probe inside that cell.
    """
    for stage in range(1, CELLS_STAGES):
        save(_cells_stage(stage), OUT, f"atlas-cells.build{stage}.png", column=FULL_BLEED, tight=False, notch=NOTCH)
    save(_cells_stage(CELLS_STAGES), OUT, "atlas-cells.png", column=FULL_BLEED, tight=False, notch=NOTCH)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Domain shift — the direction your votes never varied in
# ──────────────────────────────────────────────────────────────────────────────

DEPTH_STAGES = 5
#: The scene, in world units: a box `BOX_U` x `BOX_V` on the floor and `BOX_H`
#: tall. The floor is the space the votes explored; the height is a direction
#: they never varied in.
BOX_U, BOX_V, BOX_H = 8.4, 5.0, 5.0
#: Cavalier projection: the floor's receding axis goes up and to the right, and
#: height goes straight up. No perspective — a vanishing point would make two
#: items the same size only by accident, and this figure counts items.
PROJ_X0, PROJ_Y0 = 4.95, 1.35
PROJ_U, PROJ_V, PROJ_VY = 0.93, 0.55, 0.38
#: Where the detector's boundary sits on the floor. The region it admits runs
#: to the lid and is drawn dashed there, because it does not stop at the lid —
#: the box stops. A tube with a top on it would say the model had an opinion
#: about how high is too high, which is the one thing it does not have.
CURVE_U, CURVE_V, CURVE_RU, CURVE_RV = 5.5, 2.5, 2.05, 1.40
TUBE_H = BOX_H
#: How high the second corpus floats, and how high the alternative truth stops.
NEW_H = 3.55
DOME_H = 1.95


def proj(u: float, v: float, h: float) -> np.ndarray:
    """One world point on the page. See `PROJ_*`."""
    return np.array([PROJ_X0 + u * PROJ_U + v * PROJ_V, PROJ_Y0 + v * PROJ_VY + h])


def _spaced(count: int, ulim: tuple[float, float], vlim: tuple[float, float], gap: float) -> np.ndarray:
    """Blue-noise-ish floor positions.

    Uniform sampling clumps, and on a figure whose subject is *where the items
    are* a clump reads as a cluster that means something. Rejection sampling
    is enough at these counts.
    """
    rng = np.random.default_rng(5)
    out: list[np.ndarray] = []
    for _ in range(8000):
        if len(out) == count:
            break
        q = np.array([rng.uniform(*ulim), rng.uniform(*vlim)])
        if all(float(np.hypot(*(q - o))) > gap for o in out):
            out.append(q)
    return np.array(out)


def _inside(u: float, v: float) -> bool:
    return ((u - CURVE_U) / CURVE_RU) ** 2 + ((v - CURVE_V) / CURVE_RV) ** 2 < 1.0


def _wireframe(ax: plt.Axes) -> None:
    """The box, as twelve edges and a floor.

    Every edge is drawn, none hidden. Hidden-line removal would be more correct
    and less useful: the tube inside is translucent on purpose, so an edge
    vanishing behind it reads as a mistake rather than as depth, and the box is
    scaffolding — its whole job is to say "this is a volume" and then recede.
    """
    corners = {(u, v, h): proj(u * BOX_U, v * BOX_V, h * BOX_H) for u in (0, 1) for v in (0, 1) for h in (0, 1)}
    ax.add_patch(
        plt.Polygon(
            [corners[0, 0, 0], corners[1, 0, 0], corners[1, 1, 0], corners[0, 1, 0]],
            closed=True,
            facecolor="#f7f9fb",
            edgecolor="none",
            zorder=0,
        )
    )
    for a, b in (
        # the floor, then the lid, then the four posts
        ((0, 0, 0), (1, 0, 0)),
        ((1, 0, 0), (1, 1, 0)),
        ((1, 1, 0), (0, 1, 0)),
        ((0, 1, 0), (0, 0, 0)),
        ((0, 0, 1), (1, 0, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (0, 1, 1)),
        ((0, 1, 1), (0, 0, 1)),
        ((0, 0, 0), (0, 0, 1)),
        ((1, 0, 0), (1, 0, 1)),
        ((1, 1, 0), (1, 1, 1)),
        ((0, 1, 0), (0, 1, 1)),
    ):
        wide = a[2] == 0 and b[2] == 0  # the floor is the one edge loop items sit on
        ax.plot(
            *zip(corners[a], corners[b]),
            color=CELL_LINE if wide else RULE,
            linewidth=1.6 if wide else 1.2,
            zorder=1,
        )


def _tube(ax: plt.Axes, base: np.ndarray, top: np.ndarray, angles: np.ndarray) -> None:
    """The region the detector admits, extruded to the lid of the room."""
    for i in range(len(angles) - 1):
        ax.add_patch(
            plt.Polygon(
                [base[i], base[i + 1], top[i + 1], top[i]],
                closed=True,
                facecolor=INTRO.BAND,
                edgecolor="none",
                alpha=0.5,
                zorder=2,
            )
        )
    ax.plot(top[:, 0], top[:, 1], color=BLUE, linewidth=1.8, linestyle=(0, (6, 4)), zorder=4)
    for k in range(0, len(angles) - 1, 22):
        ax.plot(*zip(base[k], top[k]), color=BLUE, linewidth=1.2, alpha=0.75, zorder=3)


def _item(ax: plt.Axes, point: np.ndarray, zorder: int) -> None:
    ax.add_patch(plt.Circle(tuple(point), 0.155, facecolor="none", edgecolor=INK, linewidth=1.7, zorder=zorder))


def _floor_items(ax: plt.Axes) -> None:
    """The corpus the votes came from, lying on the floor of the room."""
    for index, (u, v) in enumerate(_spaced(46, (0.4, BOX_U - 0.4), (0.4, BOX_V - 0.4), 0.78)):
        point = proj(u, v, 0.0)
        if _inside(u, v) and index % 3 == 0:
            INTRO._check(ax, point)
        elif not _inside(u, v) and index % 9 == 4:
            INTRO._cross(ax, point)
        else:
            _item(ax, point, 6)


#: What each page says, and in which colour. The last two are the argument, so
#: they are the two drawn bold.
DEPTH_CAPTIONS = {
    1: ("every item you have ever voted on is on this floor", "INK"),
    2: ("the detector, drawn where it cuts", "BLUE"),
    3: ("and it says the same thing at every height", "BLUE"),
    4: ("so this corpus comes back Good, confidently", "BLUE"),
    5: ("unless the concept stops here, and nothing on the floor says", "GREEN"),
}


def _depth_stage(stage: int) -> plt.Figure:
    fig, ax = _canvas()
    _wireframe(ax)

    angles = np.linspace(0, 2 * np.pi, 180)
    curve = np.stack([CURVE_U + CURVE_RU * np.cos(angles), CURVE_V + CURVE_RV * np.sin(angles)], axis=1)
    base = np.array([proj(u, v, 0.0) for u, v in curve])

    if stage >= 3:
        _tube(ax, base, np.array([proj(u, v, TUBE_H) for u, v in curve]), angles)
    if stage >= 5:
        # The alternative truth: same footprint, but it stops.
        dome = np.array([proj(CURVE_U + CURVE_RU * math.cos(a), CURVE_V, DOME_H * math.sin(a)) for a in angles[:91]])
        ax.plot(dome[:, 0], dome[:, 1], color=GREEN, linewidth=2.4, linestyle=(0, (7, 5)), zorder=6)
    if stage >= 2:
        ax.plot(base[:, 0], base[:, 1], color=BLUE, linewidth=2.6, zorder=5)

    _floor_items(ax)

    if stage >= 4:
        for u, v in _spaced(11, (CURVE_U - 1.8, CURVE_U + 1.8), (CURVE_V - 1.0, CURVE_V + 1.0), 0.70):
            _item(ax, proj(u, v, NEW_H), 7)

    text, colour = DEPTH_CAPTIONS[stage]
    ax.text(
        CANVAS[0] - 0.45,
        0.52,
        text,
        color={"INK": INK, "BLUE": BLUE, "GREEN": GREEN}[colour],
        fontsize=LABEL_PT,
        fontweight="bold" if stage >= 4 else "normal",
        ha="right",
        va="baseline",
        zorder=8,
        path_effects=HALO,
    )
    return fig


def depth_fig() -> None:
    """Domain shift, as a floor in a room.

    Five pages. The corpus the votes came from lies on the floor of a box; the
    detector cuts it; that cut has no lid, because the shipped head is a single
    `Linear(D, 1)` and a linear score is *exactly* constant along every
    direction its weight vector does not point in (`vtscore/training/mlp.py`,
    the `LINEAR_SVM_HEAD` sentinel). So the boundary extrudes to a tube, and a
    second corpus floating a long way up it is scored Good with no hint that
    anything is being extrapolated. The dashed dome is the alternative the
    votes cannot rule out — and cannot confirm, which is the point.

    The box is the reason this reads at all. The earlier attempt at this slide
    drew the same idea on a sphere and the room had nothing to judge position
    against; a floor, four posts and a lid give every item a place.
    """
    for stage in range(1, DEPTH_STAGES):
        save(_depth_stage(stage), OUT, f"atlas-depth.build{stage}.png", column=FULL_BLEED, tight=False)
    save(_depth_stage(DEPTH_STAGES), OUT, "atlas-depth.png", column=FULL_BLEED, tight=False)


# ──────────────────────────────────────────────────────────────────────────────
# 4. The p-values — the second job, and what is wrong with it
# ──────────────────────────────────────────────────────────────────────────────

PVALUES_STAGES = 3
#: Printed names for the combiners the study compared, in the CSV's own order.
COMBINER_LABELS = {
    "median": "median",
    "mean": "mean — shipped",
    "deepest": "deepest node only",
    "fisher": "Fisher",
    "min": "min",
}


@functools.lru_cache(maxsize=1)
def _combiners() -> list[dict[str, float | str | bool]]:
    """The #3329 combiner comparison, read from the study's aggregate CSV."""
    path = STUDY / "b2_combiner_comparison.csv"
    if not path.exists():
        raise SystemExit(f"missing study data: {path}")
    header, *rows = path.read_text().strip().splitlines()
    keys = header.split(",")
    out = []
    for row in rows:
        record = dict(zip(keys, row.split(",")))
        out.append(
            {
                "name": record["combiner"],
                "ks": float(record["ks_uniform_median"]),
                "frac": float(record["frac_below_05_median"]),
                "nominal": float(record["nominal_frac_below_05"]),
                "shipped": record["is_shipped"] == "True",
            }
        )
    return out


def _pvalues_stage(stage: int) -> plt.Figure:
    rows = _combiners()
    nominal = rows[0]["nominal"]
    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0.36, right=0.955, bottom=0.17, top=0.80)

    ax.set_xlim(0, 0.36)
    ax.set_ylim(-0.012, 0.36)
    ax.set_xlabel("of its own held-out data, the share it calls atypical", fontsize=LABEL_PT, color=INK, labelpad=9)
    ax.set_ylabel("distance from a calibrated p-value", fontsize=LABEL_PT, color=INK, labelpad=9)
    ax.tick_params(labelsize=NOTE_PT, colors=SOFT, length=4)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(RULE)

    ax.axvline(nominal, color=BLUE, linewidth=2.0, linestyle=(0, (5, 4)), zorder=1)
    ax.text(
        nominal + 0.006,
        0.352,
        f"what it promises: {nominal:.0%}",
        color=BLUE,
        fontsize=NOTE_PT,
        ha="left",
        va="top",
        zorder=4,
    )
    ax.axhline(0.0, color=GREEN, linewidth=2.0, zorder=1)
    # Left end of the line it names: the right end is where the patch-embedder
    # note goes, and two labels in one corner is one label nobody reads.
    ax.text(0.004, 0.006, "calibrated", color=GREEN, fontsize=NOTE_PT, ha="left", va="bottom", zorder=4)

    drawn = [r for r in rows if r["shipped"]] if stage == 1 else rows
    for record in drawn:
        shipped = bool(record["shipped"])
        ax.plot(
            record["frac"],
            record["ks"],
            marker="o",
            markersize=13 if shipped else 9,
            markerfacecolor=INK if shipped else "white",
            markeredgecolor=INK,
            markeredgewidth=2.0,
            zorder=5,
        )
        ax.text(
            record["frac"],
            record["ks"] + 0.0135,
            COMBINER_LABELS[str(record["name"])],
            color=INK if shipped else SOFT,
            fontsize=NOTE_PT,
            fontweight="bold" if shipped else "normal",
            ha="center",
            va="baseline",
            zorder=6,
        )

    if stage >= 3:
        ax.text(
            0.352,
            0.062,
            "and on a patch embedder it calls 12% of its own\ndata strange, so the route refuses one",
            color=RED,
            fontsize=LABEL_PT,
            ha="right",
            va="top",
            linespacing=1.45,
            zorder=6,
        )
    return fig


def pvalues_fig() -> None:
    """The domain-shift verdict, against the two things it should satisfy.

    Across is the number anybody checks — what share of the atlas's own
    held-out data it calls atypical, which should be the alpha it promises.
    Up is how far the whole p-value distribution sits from the uniform a
    calibrated one would be. The shipped combiner is alone in landing on the
    first and is nowhere near the floor of the second, which is exactly how a
    spot check passes it for as long as anybody spot-checks.
    """
    for stage in range(1, PVALUES_STAGES):
        save(_pvalues_stage(stage), OUT, f"atlas-pvalues.build{stage}.png", column=FULL_BLEED, tight=False)
    save(_pvalues_stage(PVALUES_STAGES), OUT, "atlas-pvalues.png", column=FULL_BLEED, tight=False)


if __name__ == "__main__":
    blindspot_fig()
    cells_fig()
    depth_fig()
    pvalues_fig()
    print("wrote figures to", OUT)
