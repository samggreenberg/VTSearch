#!/usr/bin/env python
"""Coverage-atlas figures, for the appendix slides that explain it.

Run from the repo root:

    python slides/figs/src/make-atlas-figs.py

Four figures, and they are deliberately two different kinds of thing.

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

`atlas-far` is drawn in that same room: the shipped atlas built on its floor,
and the distance from each item to its cell's centre, which is what the atlas's
typicality score ranks. A plot of how calibrated that score is (`atlas-pvalues`)
was cut from the deck as too technical for a feature rarely used (#4872).
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
from matplotlib.path import Path as MplPath
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import FULL_BLEED, save  # noqa: E402

OUT = Path(__file__).resolve().parent.parent


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
#: All four figures sit under one-line headlines, so all four take the deck
#: plane's measured one-line notch rather than the deck's two-line reserve —
#: see `slides/STYLE.md`. A headline gaining a second line would mean
#: re-measuring, and `build.py --check` would not catch it, because a figure
#: under a two-line headline is legal and merely wrong.
NOTCH = INTRO.VOTE_NOTCH_PX

#: Type sizes, in points at `UNIT_PT` per drawing unit. The floor is 20 slide
#: pixels and this canvas renders at about 1.74 px/pt, so nothing may go below
#: ~11.5pt; `slide_figure.save` refuses a figure that does.
LABEL_PT = 15
NOTE_PT = 13

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


def _canvas(width: float = CANVAS[0], height: float = CANVAS[1]) -> tuple[plt.Figure, plt.Axes]:
    """The plane's canvas; *width* or *height* past the plane's own leaves room to its right or above it."""
    fig, ax = plt.subplots(figsize=(width * UNIT_PT / 72, height * UNIT_PT / 72))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
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
#: The radii the never-asked ring may take, in canvas units. Swept rather than
#: fixed, because the radius is half of what keeps the ring off the items: a
#: circle of one size that happens to pass through four of them is a picture
#: the room reads as "which side are these on?" when the slide's claim is that
#: they are on the inside.
RING_RADII = np.arange(1.3, 2.05, 0.05)
#: How far every item's edge must sit from the ring, in canvas units, so that
#: each one is plainly in or plainly out of it. The ring is drawn 3pt wide —
#: under 0.04 units either side of its centre line — so this leaves a gap the
#: back of a room can see.
RING_CLEAR = 0.15
#: How far the ring must sit outside the loose cut of the band, and how far
#: from any vote, in canvas units.
RING_BAND_GAP = 0.35
RING_VOTE_GAP = 0.4


@functools.lru_cache(maxsize=1)
def _ring() -> tuple[np.ndarray, float, tuple[int, ...]]:
    """Where the never-asked pocket goes, how big it is, and which items fall in it.

    Chosen rather than placed: the disc is swept over the canvas, at every
    radius in `RING_RADII`, and scored by how many items it holds, subject to
    holding no vote, keeping clear of the band the loop is asking along,
    staying out of the title's corner, and passing through no item. Ties go to
    the ring with the most room between it and the nearest item. Picking it by
    hand would invite the suspicion that the figure's whole claim was arranged,
    and the constraint that it contain no vote is the claim.
    """
    pts, _first, second, curve, _after, asked, _again, labeled = INTRO._scene()
    seed_good, seed_bad = INTRO._seed_votes()
    voted = pts[list(seed_good + seed_bad)]
    band = INTRO._band_width(second, pts, labeled + (asked,))
    best, best_key = None, (-1, 0.0)
    for radius in RING_RADII:
        margin = radius + 0.3
        for x in np.arange(margin, CANVAS[0] - margin, 0.05):
            for y in np.arange(margin, CANVAS[1] - margin, 0.05):
                centre = np.array([x, y])
                if x - radius < 4.2 and y + radius > 7.1:
                    continue  # the headline's corner
                if np.hypot(*(curve - centre).T).min() < radius + band + RING_BAND_GAP:
                    continue  # must be nowhere near the band the loop asks along
                if np.hypot(*(voted - centre).T).min() < radius + RING_VOTE_GAP:
                    continue  # and must hold no vote, which is the whole point
                dist = np.hypot(*(pts - centre).T)
                clear = float(np.abs(dist - radius).min()) - R
                if clear < RING_CLEAR:
                    continue  # and must not run through an item
                key = (int((dist < radius).sum()), clear)
                if key > best_key:
                    best, best_key = (centre, float(radius)), key
    if best is None or best_key[0] < 4:
        raise SystemExit("no unexplored pocket clear of the boundary — the field moved")
    centre, radius = best
    inside = tuple(int(i) for i in np.flatnonzero(np.hypot(*(pts - centre).T) < radius))
    return centre, radius, inside


def _blindspot_stage(stage: int) -> plt.Figure:
    pts, _first, second, _curve, _after, asked, _again, labeled = INTRO._scene()
    seed_good, seed_bad = INTRO._seed_votes()
    fig, ax = _canvas()

    if stage >= 2:
        INTRO._band(ax, second, INTRO._band_width(second, pts, labeled + (asked,)))
    INTRO._boundary(ax, second)

    # The ring is solid, never dashed: a dashed line on this plane is one of the
    # band's looser and tighter cuts, and the ring is not a cut. On the last page
    # it is drawn exactly as the detector's own curve is — the same blue, the
    # same weight — because that is what it is: a second piece of the boundary
    # between Good and Bad, the piece the detector never drew.
    centre, radius, inside = _ring()
    if stage >= 3:
        ax.add_patch(
            plt.Circle(
                tuple(centre),
                radius,
                facecolor="none",
                edgecolor=BLUE if stage >= 4 else SOFT,
                linewidth=3.0 if stage >= 4 else 2.4,
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
            # circle as its neighbours, hatched the way the deck hatches the
            # Good side of a block of media (#4533). Not a check, since nobody
            # has said so, and not a solid dot either, which reads as a vote.
            ax.add_patch(
                plt.Circle(tuple(p), R, facecolor="white", edgecolor=GREEN, hatch="//////", linewidth=1.7, zorder=4)
            )
        else:
            INTRO._circle(ax, p)

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
            f"atlas-blindspot.build{stage}.webp",
            column=FULL_BLEED,
            tight=False,
            notch=NOTCH,
        )
    save(_blindspot_stage(BLINDSPOT_STAGES), OUT, "atlas-blindspot.webp", column=FULL_BLEED, tight=False, notch=NOTCH)


# ──────────────────────────────────────────────────────────────────────────────
# 2. The atlas — a partition that remembers where you have been
# ──────────────────────────────────────────────────────────────────────────────

CELLS_STAGES = 7
#: The atlas's tree beside the plane (owner, #4872): the walk goes "from the
#: top, biggest first", so the top has to be on the slide. The panel's width
#: and the rows its first three levels stand on, in plane units. The plane
#: keeps its own shape and the slide fits both, so the plane draws smaller
#: here than on the slides either side of it: the price of seeing the walk.
TREE_W = 5.4
TREE_GAP = 0.45
#: Room over the plane. The wider canvas fits the slide at a smaller scale and
#: centres in it, which brings the plane's top-left items up into the title's
#: notch; a band of headroom moves the plane down out of it, and costs nothing,
#: since the slide letterboxes this canvas top and bottom anyway.
TREE_HEADROOM = 1.4
TREE_LEVEL_Y = (7.3, 5.1, 2.9)
#: The stubs under a cell that splits again, as on Span's tree: a cell the
#: picture's deeper split reaches, drawn as where the tree goes on.
TREE_STUB = 0.55
#: Gaps between neighbouring nodes on a row, and the extra between two
#: parents' children, in plane units.
TREE_NODE_GAP, TREE_GROUP_GAP = 0.14, 0.3
#: The fade over the walk's cell on page f (owner, #4872): light where the
#: detector is least sure the cell is Bad, toward the checks, and darker red
#: away from them. Washes of the deck's Bad red, not the red itself, so the
#: items over them still read.
FADE_LIGHT, FADE_DARK = "#fdeaea", "#e9a1a1"


def _scores() -> np.ndarray:
    """A sort score per item: how near it sits to the checks, the Goods voted so far.

    The probe below is the shipped rule, and the shipped rule reads the sort
    scores the app is showing. The toy detector's own decision function is no
    good for that here: it is an RBF fit to a dozen votes, which far from them
    all drifts back to its intercept, so the far corner of the plane would
    outscore the photos beside the checks. Nearness to the checks is what the
    room reads off the picture, and it is what page f's fade draws (owner,
    #4872: "light pink toward the checks and darker red away from the checks").
    """
    return _goodness(INTRO._scene()[0])


def _goodness(xy: np.ndarray) -> np.ndarray:
    """Minus each point's distance to the nearest Good vote: higher toward the checks."""
    seed_good, _seed_bad = INTRO._seed_votes()
    pts, _first, _second, _curve, _after, asked, _again, _labeled = INTRO._scene()
    goods = pts[list(seed_good + (asked,))]
    near = np.full(len(xy), np.inf)
    for chunk in range(0, len(xy), 4096):
        part = xy[chunk : chunk + 4096]
        near[chunk : chunk + 4096] = np.hypot(*(part[:, None, :] - goods[None, :, :]).transpose(2, 0, 1)).min(axis=1)
    return -near


def _members(depth: int, cell: int) -> np.ndarray:
    return np.flatnonzero(_cells_at(depth) == cell)


def _children(depth: int, cell: int) -> list[int]:
    """The cells *cell* split into, biggest first, or none if it did not split."""
    if depth + 1 > _leaf_depth():
        return []
    level = _partition()[depth + 1]
    kids = [int(k) for k in np.flatnonzero(level.parents == cell)]
    if len(kids) < 2:
        return []
    return sorted(kids, key=lambda k: -len(_members(depth + 1, k)))


def _voted() -> set[int]:
    seed_good, seed_bad = INTRO._seed_votes()
    _pts, _first, _second, _curve, _after, asked, _again, _labeled = INTRO._scene()
    return set(seed_good + seed_bad + (asked,))


def _next_cell() -> tuple[int, int]:
    """``(depth, cell)`` the walk stops at: breadth first from the top, biggest sibling first.

    The shipped `CoverageAtlas.next_sample` walk: the first cell carrying no
    vote. The tree beside the plane is laid out in that order, so the walk is
    the tree read a row at a time, left to right.
    """
    voted = _voted()
    queue = [(0, 0)]
    while queue:
        depth, cell = queue.pop(0)
        if not voted & set(_members(depth, cell).tolist()):
            return depth, cell
        queue += [(depth + 1, k) for k in _children(depth, cell)]
    raise SystemExit("every cell carries a vote — the drawing has nothing to explore")


def _probe(depth: int, cell: int) -> int:
    """Which item of the cell the atlas asks about: the shipped surprise probe.

    The region's median score decides what the model presumes about it, and
    the probe is the element that would most cheaply prove the presumption
    wrong: the lowest-scored item of a region it calls good, the highest-scored
    item of one it calls bad. Informative whichever way the answer lands.
    """
    pts, _first, second, *_ = INTRO._scene()
    scores = _scores()
    members = _members(depth, cell)
    presumed_good = float(np.median(second.decision_function(pts[members]))) >= 0.0
    return int(members[scores[members].argmin() if presumed_good else scores[members].argmax()])


def _tree_radius(depth: int, cell: int) -> float:
    """A node's radius: area for the cell's count of items, so "biggest first" can be seen."""
    return 0.07 + 0.03 * math.sqrt(len(_members(depth, cell)))


@functools.lru_cache(maxsize=1)
def _tree_xy() -> dict[tuple[int, int], tuple[float, float]]:
    """Each node's centre: the third row laid out left to right in the walk's order, parents over their children."""
    row: list[tuple[int, int]] = []
    groups: list[int] = []
    for k1 in _children(0, 0):
        for k2 in _children(1, k1):
            row.append((2, k2))
            groups.append(k1)
    xs: list[float] = []
    for i, node in enumerate(row):
        r = _tree_radius(*node)
        if not xs:
            xs.append(r)
            continue
        gap = TREE_NODE_GAP + (TREE_GROUP_GAP if groups[i] != groups[i - 1] else 0.0)
        xs.append(xs[-1] + _tree_radius(*row[i - 1]) + gap + r)
    span = xs[-1] + _tree_radius(*row[-1])
    x0 = CANVAS[0] + TREE_GAP + (TREE_W - span) / 2
    xy = {node: (x0 + x, TREE_LEVEL_Y[2]) for node, x in zip(row, xs, strict=True)}
    for k1 in _children(0, 0):
        kids = [xy[(2, k2)][0] for k2 in _children(1, k1)]
        xy[(1, k1)] = (sum(kids) / len(kids), TREE_LEVEL_Y[1])
    tops = [xy[(1, k1)][0] for k1 in _children(0, 0)]
    xy[(0, 0)] = (sum(tops) / len(tops), TREE_LEVEL_Y[0])
    assert max(x + _tree_radius(*n) for n, (x, _y) in xy.items()) <= CANVAS[0] + TREE_GAP + TREE_W + 1e-9
    return xy


def _draw_tree(ax: plt.Axes, stage: int) -> None:
    """The atlas's first three levels, as Span draws its tree: a level a page, then the votes, then the walk's stop."""
    xy = _tree_xy()
    shown = {0: 1, 1: 2, 2: 3}
    voted = _voted()
    stop = _next_cell()
    for (depth, cell), (x, y) in xy.items():
        if stage < shown[depth]:
            continue
        if depth:
            parent = int(_partition()[depth].parents[cell])
            px, py = xy[(depth - 1, parent)]
            ax.plot([px, x], [py, y], color=RULE, linewidth=1.5, zorder=2)
        if depth == 2 and _children(2, cell):
            for dx in (-0.12, 0.0, 0.12):
                ax.plot([x, x + dx], [y, y - TREE_STUB], color=RULE, linewidth=1.2, zorder=1)
        covered = stage >= 4 and bool(voted & set(_members(depth, cell).tolist()))
        ax.add_patch(plt.Circle((x, y), _tree_radius(depth, cell), facecolor=INK if covered else "white",
                                edgecolor=SOFT, linewidth=1.4, zorder=4))  # fmt: skip
        if stage >= 5 and (depth, cell) == stop:
            ax.add_patch(plt.Circle((x, y), _tree_radius(depth, cell) + 0.12, facecolor="none", edgecolor=BLUE,
                                    linewidth=3.0, zorder=5))  # fmt: skip


#: The asked item's mark on Cover Story. The plane draws smaller there (see
#: `TREE_W`), and the deck's 13pt "?" would land under the type floor, so the
#: mark is drawn a size up: the same circle and glyph, scaled together.
ASK_PT = 16


def _ask(ax: plt.Axes, p: np.ndarray) -> None:
    """`INTRO._circle(..., glyph="?")` at `ASK_PT`: the item the app asks about."""
    scale = ASK_PT / INTRO.QUERY_PT
    ax.add_patch(plt.Circle(tuple(p), R * scale, facecolor="white", edgecolor=INK, linewidth=2.6, zorder=5))
    ax.text(p[0], p[1] + INTRO._glyph_drop("?") * scale, "?", color=INK, fontsize=ASK_PT, fontweight="bold",
            ha="center", va="baseline", zorder=6)  # fmt: skip


def _fade(ax: plt.Axes, chosen: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> None:
    """Page f's wash over the walk's cell: light toward the checks, where the detector is least sure, darker away."""
    grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    inside = chosen.reshape(-1) > 0.5
    score = np.full(len(grid), np.nan)
    score[inside] = _goodness(grid[inside])
    lo, hi = np.nanmin(score), np.nanmax(score)
    t = ((score - lo) / (hi - lo)).reshape(chosen.shape)
    light, dark = (np.array(matplotlib.colors.to_rgb(c)) for c in (FADE_LIGHT, FADE_DARK))
    rgba = np.zeros((*chosen.shape, 4))
    rgba[..., :3] = dark + (light - dark) * np.nan_to_num(t)[..., None]
    rgba[..., 3] = (chosen > 0.5).astype(float)
    ax.imshow(rgba, extent=(xs[0], xs[-1], ys[0], ys[-1]), origin="lower", interpolation="bilinear", zorder=0)


def _cells_stage(stage: int) -> plt.Figure:
    pts, _first, _second, _curve, _after, asked, _again, _labeled = INTRO._scene()
    seed_good, seed_bad = INTRO._seed_votes()
    fig, ax = _canvas(CANVAS[0] + TREE_GAP + TREE_W, CANVAS[1] + TREE_HEADROOM)

    if stage >= 2:
        _draw_cells(ax, 1 if stage == 2 else _leaf_depth())
    _draw_tree(ax, stage)

    depth, cell = _next_cell()
    probe = _probe(depth, cell)

    if stage >= 4:
        # The evidence channels: a cell somebody has voted in is a cell the
        # walk will not stop at, so the wash is the whole state the atlas keeps.
        voted_cells = {int(_cells_at(_leaf_depth())[i]) for i in seed_good + seed_bad + (asked,)}
        xs, ys, raster = _cell_raster(_leaf_depth())
        covered = np.isin(raster, sorted(voted_cells)).astype(float)
        ax.contourf(xs, ys, covered, levels=[0.5, 1.5], colors=["#eef1f5"], zorder=0)

    if stage >= 5:
        xs, ys, raster = _cell_raster(depth)
        chosen = (raster == cell).astype(float)
        if stage >= 6:
            _fade(ax, chosen, xs, ys)
        else:
            ax.contourf(xs, ys, chosen, levels=[0.5, 1.5], colors=[INTRO.BAND], zorder=0)
        ax.contour(xs, ys, chosen, levels=[0.5], colors=[BLUE], linewidths=3.0, zorder=3)

    voted = {**{i: "good" for i in seed_good + (asked,)}, **{i: "bad" for i in seed_bad}}
    for i, p in enumerate(pts):
        if stage >= 4 and voted.get(i) == "good":
            INTRO._check(ax, p)
        elif stage >= 4 and voted.get(i) == "bad":
            INTRO._cross(ax, p)
        elif stage >= 7 and i == probe:
            _ask(ax, p)
        else:
            INTRO._circle(ax, p)

    return fig


def cells_fig() -> None:
    """The atlas itself, drawn on the plane the deck already taught, with its tree beside it.

    Seven pages: the collection, one cell at the top of the tree; one split;
    the recursion; the votes landing in cells; the walk, top down and biggest
    first, stopping at the first cell with no evidence in it; the detector's
    certainty across that cell; and the surprise probe it picks there. The
    cell is presumed Bad, so the probe is its highest scorer, the one nearest
    the checks (owner, #4872: "the NE item, since it's the least red").
    """
    depth, cell = _next_cell()
    pts = INTRO._scene()[0]
    members = _members(depth, cell)
    probe = _probe(depth, cell)
    assert probe == members[np.argmax(pts[members, 0] + pts[members, 1])], "the probe should be the cell's NE-most item"
    for stage in range(1, CELLS_STAGES):
        save(_cells_stage(stage), OUT, f"atlas-cells.build{stage}.webp", column=FULL_BLEED, tight=False, notch=NOTCH)
    save(_cells_stage(CELLS_STAGES), OUT, "atlas-cells.webp", column=FULL_BLEED, tight=False, notch=NOTCH)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Domain shift — the direction your votes never varied in
# ──────────────────────────────────────────────────────────────────────────────

DEPTH_STAGES = 5
#: The scene, in world units: a prism `BOX_U` x `BOX_V` on the floor and `BOX_H`
#: tall. The floor is the space the votes explored; the height is a direction
#: they never varied in.
#:
#: `BOX_H` is what buys the rest of the drawing its size, which is why it is as
#: small as the picture can bear. The headline's notch sits in the slide's
#: top-left, and this projection's top face reaches its highest point almost
#: immediately — a fifth of the way across — so a room drawn tall enough to
#: reach past the notch's height has to start to the *right* of it, and four
#: units of slide go unused. Squat and wide clears the notch underneath
#: instead, and the whole scene grows by half to fill what that frees. The
#: height still has to read as a direction nobody voted along: at this scale it
#: is thirteen item-diameters of empty air between the two collections.
BOX_U, BOX_V, BOX_H = 8.4, 4.6, 3.0
#: Cavalier projection: the floor's receding axis goes up and to the right, and
#: height goes straight up. No perspective — a vanishing point would make two
#: items the same size only by accident, and this figure counts items.
#:
#: `PROJ_SCALE` sizes the whole drawing at once. It is set as large as the two
#: things boxing it in allow: the slide's own edges, and the headline's notch
#: above. Nothing else competes for the space — this figure carries no text,
#: like every other figure in the deck.
PROJ_X0, PROJ_Y0 = 0.45, 0.30
PROJ_SCALE = 1.45
PROJ_U, PROJ_V, PROJ_VY = 0.93, 0.55, 0.38
#: Where the detector's boundary sits on the floor. The region it admits runs
#: all the way to the lid, because it does not stop at the lid — the box stops.
#: A tube with a top on it would say the model had an opinion about how high is
#: too high, which is the one thing it does not have.
CURVE_U, CURVE_V, CURVE_RU, CURVE_RV = 5.5, 2.5, 2.05, 1.40
TUBE_H = BOX_H
#: The second collection lies on the ceiling exactly as the first lies on the
#: floor — same count, same spread, same relationship to its own plane. It is a
#: *domain*, not a handful of outliers, and a scatter of a dozen items floating
#: mid-air said the opposite: that domain shift is a few strange items rather
#: than a whole collection the detector has never seen.
NEW_H = BOX_H
NEW_SEED = 17
#: How many items each collection holds. Not a round number and not a constant
#: anybody chose: it is whatever keeps the floor as densely covered as it was
#: before the scene grew, so that enlarging the drawing enlarges the drawing
#: rather than thinning the corpus. The items themselves stay the size they are
#: everywhere else in the deck — they are the same items — so the only way to
#: hold the density is to draw more of them.
N_ITEMS = 76
#: How high the alternative truth stops. Short of half the room, so the dome
#: visibly does not reach the collection it is being asked about.
DOME_H = 1.45


def proj(u: float, v: float, h: float) -> np.ndarray:
    """One world point on the page. See `PROJ_*`."""
    return np.array([PROJ_X0 + PROJ_SCALE * (u * PROJ_U + v * PROJ_V), PROJ_Y0 + PROJ_SCALE * (v * PROJ_VY + h)])


def _spaced(count: int, ulim: tuple[float, float], vlim: tuple[float, float], gap: float, seed: int = 5) -> np.ndarray:
    """Blue-noise-ish positions in a plane, spaced by how far apart they *look*.

    Uniform sampling clumps, and on a figure whose subject is *where the items
    are* a clump reads as a cluster that means something. Rejection sampling is
    enough at these counts.

    `gap` is a distance on the page and not in the world, which is the whole
    reason for measuring it here. This projection squashes the receding axis to
    about a third, so two items a world-unit apart along it end up a third as
    far apart on screen as two a world-unit apart across it: a world-space gap
    that looks generous in one direction draws items nearly tangent in the
    other, which is exactly what it did.
    """
    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for _ in range(40000):
        if len(out) == count:
            return np.array(out)
        q = np.array([rng.uniform(*ulim), rng.uniform(*vlim)])
        here = proj(q[0], q[1], 0.0)
        if all(float(np.hypot(*(here - proj(o[0], o[1], 0.0)))) > gap for o in out):
            out.append(q)
    # Loud, because the failure is invisible in the output: a sampler that gives
    # up early just draws a thinner domain, and "the same number of items on the
    # ceiling as on the floor" is the whole claim of the picture.
    raise SystemExit(
        f"could not place {count} items at gap {gap} in {ulim} x {vlim} (got {len(out)}) — "
        f"lower the gap or enlarge the plane"
    )


def _inside(u: float, v: float, margin: float = 1.0) -> bool:
    """Whether (u, v) is inside the detector's boundary, at *margin* of its size.

    The margin is what keeps a vote off the line. An item the curve passes
    through reads as one the detector has cut in two rather than one it has
    called, which is a lesson `make-intro-figs.py` learned the hard way and
    spends a settling pass on; here it costs one parameter, because this
    figure is free to choose which items carry a mark.
    """
    return ((u - CURVE_U) / (CURVE_RU * margin)) ** 2 + ((v - CURVE_V) / (CURVE_RV * margin)) ** 2 < 1.0


def _wireframe(ax: plt.Axes) -> dict[int, plt.Polygon]:
    """The box, as twelve edges and two planes. Returns the planes by level.

    Every edge is drawn, none hidden. Hidden-line removal would be more correct
    and less useful: the tube inside is translucent on purpose, so an edge
    vanishing behind it reads as a mistake rather than as depth, and the box is
    scaffolding — its whole job is to say "this is a volume" and then recede.

    The two planes come back because the items are drawn *through* them: see
    `_item`, which clips a sphere's submerged half to the plane it sits in.
    """
    corners = {(u, v, h): proj(u * BOX_U, v * BOX_V, h * BOX_H) for u in (0, 1) for v in (0, 1) for h in (0, 1)}
    planes: dict[int, plt.Polygon] = {}
    for h in (0, 1):
        planes[h] = plt.Polygon(
            [corners[0, 0, h], corners[1, 0, h], corners[1, 1, h], corners[0, 1, h]],
            closed=True,
            facecolor=PLANE_FILL,
            edgecolor="none",
            zorder=0,
        )
        ax.add_patch(planes[h])
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
            zorder=Z_WIREFRAME,
        )
    return planes


#: How much each of the pillar's walls takes out of what is behind it. Low,
#: because it is paid twice: a sphere inside the pillar is behind one wall and
#: one beyond it is behind two, so the figure's darkest item keeps (1 - a)^2 of
#: itself. At a half — the obvious first guess — that is a quarter, and the
#: items behind the pillar stopped being items.
PILLAR_ALPHA = 0.26


#: Where the boundary's own outline runs up the page: the two angles at which
#: the footprint's tangent is vertical on screen, and so the two places a
#: surface standing on it turns away from the eye. Everything drawn between
#: them is glass; the silhouette is the only line either solid needs, because
#: it is the only line either solid has.
SILHOUETTE_ANGLE = math.atan2(CURVE_RV * PROJ_V, CURVE_RU * PROJ_U)
#: How wide every line belonging to the boundary is drawn — the foot, the mouth
#: and the two silhouettes are one outline around one solid, so they are one
#: weight.
OUTLINE_LW = 2.6
#: And how far the glass stops short of the outlines that cap it, in world
#: height. A wall running the whole way from the foot to the mouth ends *on*
#: them, so it tints the outer half of each stroke and the near arcs — the ones
#: with nothing standing in front of them — come out lighter than the far ones:
#: the tint exactly backwards. Stopping half a stroke short puts both arcs
#: wholly on the correct side, and the sliver of un-glassed wall it leaves is
#: narrower than the stroke drawn over it.
GLASS_INSET = 0.5 * OUTLINE_LW / UNIT_PT / PROJ_SCALE


def _footprint(angles: np.ndarray, height: float) -> np.ndarray:
    return np.array([proj(CURVE_U + CURVE_RU * math.cos(a), CURVE_V + CURVE_RV * math.sin(a), height) for a in angles])


def _tube(ax: plt.Axes, base: np.ndarray, top: np.ndarray, angles: np.ndarray) -> None:
    """The pillar's body. Its mouth and its foot are drawn by the caller.

    Both of those lie *in* a plane — the foot in the floor, the mouth in the
    ceiling — so both have to be painted over the submerged halves of the
    spheres in that plane, which means waiting until those spheres exist. What
    is left is the glass between them and the two edges it ends at, which close
    the mouth and the foot into one outline.
    """
    glass = (_footprint(angles, GLASS_INSET), _footprint(angles, TUBE_H - GLASS_INSET))
    for i in range(len(angles) - 1):
        ax.add_patch(
            plt.Polygon(
                [glass[0][i], glass[0][i + 1], glass[1][i + 1], glass[1][i]],
                closed=True,
                facecolor=INTRO.BAND,
                edgecolor="none",
                alpha=PILLAR_ALPHA,
                zorder=Z_PILLAR,
            )
        )
    for side in (SILHOUETTE_ANGLE, SILHOUETTE_ANGLE + math.pi):
        edge = np.array([_footprint(np.array([side]), h)[0] for h in (0.0, TUBE_H)])
        ax.plot(edge[:, 0], edge[:, 1], color=BLUE, linewidth=OUTLINE_LW, zorder=Z_PILLAR + 0.1)


@functools.lru_cache(maxsize=1)
def _dome_outline() -> np.ndarray:
    """The dome's silhouette: the arc over the top, from one edge to the other.

    A parallel projection is linear, so it takes the half-ellipsoid's shadow to
    an exact half-ellipse and there is nothing to trace or approximate. An
    ellipsoid `x' Q^-1 x = 1` projected by a linear `L` lands inside the ellipse
    whose matrix is `L Q L'`, and the square root of that 2x2 is the map from
    the unit circle onto its boundary. Half of that boundary is the dome: the
    contour meets the floor exactly where the footprint's own tangent goes
    vertical — `SILHOUETTE_ANGLE`, the same two places the pillar's edges stand
    — so the arc wanted is whichever half of the ellipse runs over the top.
    """
    projected = PROJ_SCALE**2 * np.array(
        [
            [PROJ_U**2 * CURVE_RU**2 + PROJ_V**2 * CURVE_RV**2, PROJ_V * PROJ_VY * CURVE_RV**2],
            [PROJ_V * PROJ_VY * CURVE_RV**2, PROJ_VY**2 * CURVE_RV**2 + DOME_H**2],
        ]
    )
    values, vectors = np.linalg.eigh(projected)
    root = vectors @ np.diag(np.sqrt(values)) @ vectors.T
    centre = proj(CURVE_U, CURVE_V, 0.0)
    start = math.atan2(root[0, 1], root[0, 0])
    halves = [
        np.stack([np.cos(t), np.sin(t)], axis=1) @ root.T + centre
        for t in (np.linspace(start, start + math.pi, 160), np.linspace(start + math.pi, start + 2 * math.pi, 160))
    ]
    return max(halves, key=lambda arc: arc[:, 1].mean())


#: How many gores the dome's glass is cut into. Far more than the shape needs —
#: its outline is drawn analytically and its curvature is settled by a few
#: dozen — and the number is about the apex, where every gore converges. Each
#: one is antialiased against its neighbours, so where they are narrower than a
#: pixel their coverage does not sum back to one and the top of the dome comes
#: out a shade light. Wider gores concentrate that into a visible notch; these
#: spread it under the outline. Measured at slide resolution, not guessed: 45
#: leaves a notch, 180 a smudge, this none.
DOME_GORES = 1200


def _dome(ax: plt.Axes) -> None:
    """The alternative pillar: same footprint, same glass, but it closes.

    Drawn exactly as `_tube` is, and for the same reason — it is the same claim
    about the same boundary, differing only in whether it has a lid, so a
    picture that drew it in another hand would be answering a different
    question. The surface is cut into gores rather than wall quads, one per
    angle step and each running the whole way up, so the near and far halves
    overlap in projection and the compositor tints what is behind them once and
    twice on its own. The gores are fill only: what is drawn is the silhouette,
    as it is for the pillar.
    """
    angles = np.linspace(0, 2 * np.pi, DOME_GORES)
    meridians = [
        np.array(
            [
                proj(
                    CURVE_U + CURVE_RU * math.cos(a) * math.cos(t),
                    CURVE_V + CURVE_RV * math.sin(a) * math.cos(t),
                    DOME_H * math.sin(t),
                )
                for t in np.linspace(math.asin(GLASS_INSET / DOME_H), math.pi / 2, 24)
            ]
        )
        for a in angles
    ]
    # Every gore ends at the same point, and `cos(pi/2)` is not quite zero: left
    # to the arithmetic they end a hairline apart, and 180 of them converging
    # leaves a seam at the top. Snap them together.
    for meridian in meridians:
        meridian[-1] = proj(CURVE_U, CURVE_V, DOME_H)
    for i in range(len(angles) - 1):
        ax.add_patch(
            plt.Polygon(
                np.vstack([meridians[i], meridians[i + 1][::-1]]),
                closed=True,
                facecolor=INTRO.BAND,
                edgecolor="none",
                alpha=PILLAR_ALPHA,
                zorder=Z_PILLAR,
            )
        )
    outline = _dome_outline()
    ax.plot(outline[:, 0], outline[:, 1], color=BLUE, linewidth=OUTLINE_LW, zorder=Z_PILLAR + 0.1)


#: The item radius, on the page.
ITEM_R = 0.165
#: The plane's own fill, and so also the colour of the half of a sphere that is
#: under it. Nudged up from the old near-white so the submerged crescent reads
#: at this size; it is the theme's `--wash`.
PLANE_FILL = "#e7ecf2"
#: How flat a circle drawn *in* a plane comes out on the page — the vertical
#: squash of this projection, and therefore the shape of a waterline. Derived
#: rather than chosen, so it stays right if the projection is re-angled.
WATERLINE_K = PROJ_VY / math.hypot(PROJ_U, PROJ_V)


#: How far the clip used for in-plane markings runs past the crescent it is
#: taken from, in canvas units. The crescent's boundary is the *centreline* of
#: the sphere's rim, so a clip taken from it exactly leaves the rim's outer half
#: standing on top of whatever is drawn through it — and a grey rim arc lying
#: over the boundary curve says the same wrong thing the crescent did before it:
#: that the sphere sits on the plane rather than in it. Half the rim's width,
#: rounded up: 1.7pt at 46pt to the unit.
CLIP_PAD = 0.03


@functools.lru_cache(maxsize=2)
def _submerged_outline(pad: float = 0.0) -> np.ndarray:
    """The half of a sphere that is under the plane it sits in, as a polygon.

    An item centred on a plane is cut by it at the equator. From above, the
    near half of that equator bulges downward, so what is left visible of the
    lower hemisphere is a crescent: bounded above by the waterline and below by
    the sphere's own silhouette. Drawn in the plane's colour, it reads as the
    sphere seen *through* the plane rather than as a circle sitting on top of
    one, which is the whole difference between a surface and a backdrop.
    """
    theta = np.linspace(0, math.pi, 48)
    waterline = np.stack([(ITEM_R + pad) * np.cos(theta), -WATERLINE_K * ITEM_R * np.sin(theta) + pad], axis=1)
    phi = np.linspace(math.pi, 2 * math.pi, 64)
    silhouette = np.stack([(ITEM_R + pad) * np.cos(phi), (ITEM_R + pad) * np.sin(phi)], axis=1)
    return np.vstack([waterline, silhouette])


def _item(
    ax: plt.Axes, point: np.ndarray, zorder: float, plane: plt.Polygon | None = None, fill: str = "white"
) -> np.ndarray | None:
    """One item: a sphere, half of it under *plane*.

    With no plane it is a plain circle — the first page's items lie on a plane
    seen face-on, where there is no near side to be cut by. With one, the
    submerged crescent is drawn in the plane's own colour, which reads as the
    sphere seen *through* the surface rather than as a disc sitting on it. The
    crescent is clipped to the plane all the same, though `_plane_margin` now
    keeps every item inside its quad: the clip costs nothing and it is the only
    thing standing between a re-spaced field and a sphere hanging in the void.

    The outline is drawn three times, which is one more than looks necessary
    and one fewer than it takes to get wrong. Black everywhere; then grey over
    the plane's whole area, which is a region on the page and not a half of the
    sphere; then black again above the waterline. What survives is a sphere
    outlined in black where it stands clear and in grey where it is seen
    through the plane, *including* the rim of one hanging over the plane's edge,
    which is clear of the plane and so stays black.

    *fill* is the sphere's face.
    """
    ax.add_patch(plt.Circle(tuple(point), ITEM_R, facecolor=fill, edgecolor="none", zorder=zorder))
    if plane is None:
        ax.add_patch(
            plt.Circle(tuple(point), ITEM_R, facecolor="none", edgecolor=INK, linewidth=1.7, zorder=zorder + 0.3)
        )
        return None

    submerged = _submerged_outline() + point
    crescent = plt.Polygon(submerged, closed=True, facecolor=PLANE_FILL, edgecolor="none", zorder=zorder + 0.05)
    ax.add_patch(crescent)
    crescent.set_clip_path(plane)
    # The fill alone is a few percent off white and vanishes at slide size; the
    # waterline is what makes the cut legible from the back of a room. Clipped
    # too, so the half of a sphere hanging over the plane's edge has no
    # waterline drawn across it.
    theta = np.linspace(0, math.pi, 48)
    water = ax.plot(
        point[0] + ITEM_R * np.cos(theta),
        point[1] - WATERLINE_K * ITEM_R * np.sin(theta),
        color=CELL_LINE,
        linewidth=1.1,
        zorder=zorder + 0.1,
    )[0]
    water.set_clip_path(plane)

    for colour, offset, clip in (
        (INK, 0.15, None),
        (OUTLINE_SUNK, 0.2, plane),
        (INK, 0.3, _above_water(ax, point, ITEM_R * 1.02)),
    ):
        ring = plt.Circle(
            tuple(point), ITEM_R, facecolor="none", edgecolor=colour, linewidth=1.7, zorder=zorder + offset
        )
        ax.add_patch(ring)
        if clip is not None:
            ring.set_clip_path(clip)
    return _submerged_outline(CLIP_PAD) + point


#: The closest two items may come on the page — the item diameter plus a little
#: air, so neighbours never read as one blob.
ITEM_GAP = 2 * ITEM_R + 0.10


def _plane_margin(radius: float) -> tuple[float, float]:
    """How far from a plane's edges a mark of *radius* has to sit to fit inside.

    The plane *is* the space, in the two directions it spans: everything the
    corpus covers is on it and nothing is beside it. So a mark hanging over its
    southern edge is not an item with a bottom, it is an item outside the world,
    and at this angle it reads as one leaning out of the picture toward the
    viewer. Every mark stays inside the quad.

    Returned per axis, because the two edges are not the same distance away on
    the page. A step in `v` moves a mark `PROJ_VY` up the page and the southern
    edge is horizontal, so clearing it is a matter of that one component; a step
    in `u` moves it straight across while the western edge runs off at the
    `v` axis's own slope, so what has to clear the edge is the part of that step
    perpendicular to it. Derived rather than measured, so the margins follow the
    projection if it is ever re-angled or resized.
    """
    across = math.hypot(PROJ_V, PROJ_VY)
    return (radius * across / (PROJ_SCALE * PROJ_U * PROJ_VY), radius / (PROJ_SCALE * PROJ_VY))


#: The floor and the ceiling, as the region their items' centres may occupy.
ITEM_LIMITS = (
    (_plane_margin(ITEM_R)[0], BOX_U - _plane_margin(ITEM_R)[0]),
    (_plane_margin(ITEM_R)[1], BOX_V - _plane_margin(ITEM_R)[1]),
)


#: The submerged shades of the two vote colours: the deck's own red and green,
#: darkened. A mark standing in the plane is cut by it exactly as a sphere is,
#: and the half underneath is seen through the same wash — which darkens what
#: is under it, as the spheres show by going from white to grey. A shade of a
#: pinned hue is a shade, not a second identity, so this stays inside the
#: theme's rule that colour means one thing.
RED_SUNK = "#6f1010"
GREEN_SUNK = "#07533a"
#: And the sphere rim, where the plane is over it. Dark grey rather than
#: black, for the same reason and by the same rule.
OUTLINE_SUNK = "#5b6472"


def _above_water(ax: plt.Axes, point: np.ndarray, reach: float) -> plt.Polygon:
    """An invisible patch covering everything above the waterline at *point*.

    The same curve the spheres are cut by, widened to *reach* so it spans a
    glyph rather than a circle. Used as a clip, so one mark can be drawn twice —
    dark underneath, bright above the line — without either copy knowing the
    shape of the other.
    """
    theta = np.linspace(math.pi, 0, 48)
    arc = np.stack([reach * np.cos(theta), -WATERLINE_K * reach * np.sin(theta)], axis=1) + point
    corners = np.array([[point[0] + reach, point[1] + 2 * reach], [point[0] - reach, point[1] + 2 * reach]])
    patch = plt.Polygon(np.vstack([arc, corners]), closed=True, facecolor="none", edgecolor="none", zorder=0)
    ax.add_patch(patch)
    return patch


def _sunk_glyph(ax: plt.Axes, point: np.ndarray, draw, sunk: str, zorder: float) -> None:
    """One vote mark, half of it under the plane.

    The glyph is drawn by the deck's own `_check` / `_cross`, then duplicated in
    the submerged shade underneath and the original clipped to the water's
    surface. Reading the geometry back off the artists rather than restating it
    is what keeps this from drifting away from the marks every other slide
    uses.
    """
    first = len(ax.lines)
    draw(ax, point)
    surface = _above_water(ax, point, INTRO.R * 1.35)
    for stroke in ax.lines[first:]:
        ax.plot(
            stroke.get_xdata(),
            stroke.get_ydata(),
            color=sunk,
            linewidth=stroke.get_linewidth(),
            solid_capstyle=stroke.get_solid_capstyle(),
            solid_joinstyle=stroke.get_solid_joinstyle(),
            zorder=zorder,
        )
        stroke.set_zorder(zorder + 0.1)
        stroke.set_clip_path(surface)


#: How many of the floor's items carry a vote. A handful: the slide is not about
#: the voting, it is about what the votes could not reach, and the corpus has to
#: look voted-on without looking laboured over.
N_GOOD, N_BAD = 5, 4


def _votes(floor: np.ndarray) -> dict[int, str]:
    """Which floor items carry a mark, spread as widely as the plane allows.

    Farthest-point selection: start from the eligible item nearest the pool's
    middle, then repeatedly take whichever is farthest from everything chosen
    so far. Two earlier rules both failed for the same reason — they spread the
    marks over the *sampler's* list rather than over the picture. An index
    modulo left one check and three crosses after the field was re-spaced;
    picking evenly along the list put all four crosses in one corner.

    Eligibility carries a margin around the boundary either way, so no mark
    lands on the curve — an item a line passes through reads as one the
    detector cut in half rather than one it called — and a wider margin from
    the plane's edge, so no glyph hangs off into space the way a sphere is
    meant to.
    """
    pools = {
        "good": [i for i, (u, v) in enumerate(floor) if _inside(u, v, 0.80) and _clear_of_edge(u, v)],
        "bad": [i for i, (u, v) in enumerate(floor) if not _inside(u, v, 1.20) and _clear_of_edge(u, v)],
    }
    marks: dict[int, str] = {}
    for name, count in (("good", N_GOOD), ("bad", N_BAD)):
        pool = pools[name]
        if len(pool) < count:
            raise SystemExit(f"only {len(pool)} items eligible for {count} {name} votes — respace the field")
        screen = {i: proj(floor[i][0], floor[i][1], 0.0) for i in pool}
        centre = np.mean([screen[i] for i in pool], axis=0)
        chosen = [min(pool, key=lambda i: float(np.hypot(*(screen[i] - centre))))]
        while len(chosen) < count:
            chosen.append(max(pool, key=lambda i: min(float(np.hypot(*(screen[i] - screen[j]))) for j in chosen)))
        marks.update({i: name for i in chosen})
    return marks


#: How far from a plane's edge an item must sit to be allowed a vote glyph — the
#: same containment the spheres get, for a mark that reaches further than they
#: do. `INTRO.R` is the glyph's own radius and the rest is the overshoot its
#: strokes are drawn with.
VOTE_MARGIN_U, VOTE_MARGIN_V = _plane_margin(INTRO.R * 1.35)


def _clear_of_edge(u: float, v: float) -> bool:
    return VOTE_MARGIN_U < u < BOX_U - VOTE_MARGIN_U and VOTE_MARGIN_V < v < BOX_V - VOTE_MARGIN_V


#: The paint order, as bands rather than as numbers. Anything the pillar stands
#: in front of is painted *under* it, which is what makes the blue shading right
#: without computing any of it: each of the pillar's wall quads is
#: half-transparent, a point inside the pillar is behind one wall, and a point
#: beyond it is behind two, so the compositor tints them once and twice on its
#: own. The alternative — drawing everything on top and tinting by hand — was
#: what made the pillar look like it stood behind every sphere it should have
#: hidden.
#:
#: Each band of items is 1.2 wide: `_painted_back_to_front` spends 0.9 of it on
#: depth and `_item` another 0.3 on one sphere's own fill, waterline and rims.
#: The `*_OVERLAY` levels sit just above their band and the pillar's glass just
#: above the first of them, so a behind-the-glass item is tinted from its fill
#: to its rim and so is the piece of boundary showing through it. Getting that
#: wrong is not subtle — the widest item's rim used to poke through the glass —
#: but it is invisible, so the arithmetic is written down rather than eyeballed.
Z_WIREFRAME = 0.3
Z_BEHIND_PILLAR, Z_BEHIND_OVERLAY = 1.0, 2.4
Z_PILLAR = 3.0
Z_IN_FRONT, Z_FRONT_OVERLAY = 6.0, 7.4
Z_CEILING, Z_CEILING_OVERLAY = 8.0, 9.4
#: And under every item, for the first stroke of anything lying in a plane.
#: Above the plane's own fill and the box's edges, below the glass — so the far
#: half of the pillar's foot, which is genuinely seen through the near wall,
#: takes the wall's tint without anything being computed for it.
Z_IN_PLANE = 0.6

#: One band's submerged regions and the level a marking lying in their plane has
#: to be repeated at to show through them.
Layer = tuple[list[np.ndarray], float]


def _occluded_by_pillar(u: float, v: float, h: float) -> bool:
    """Whether the pillar stands between this item and the eye.

    The projection is a shear, so "toward the camera" is one fixed direction in
    the world: the displacement that leaves a point where it is on the page.
    Walk it from the item and count crossings of the pillar's wall, keeping
    only those that happen while still inside the room's height — a ray that
    has already left through the ceiling is not being blocked by anything.

    Ceiling items come out false by construction, which is correct: from up
    there the walk leaves the room immediately, so nothing is in the way.
    """
    toward = np.array([PROJ_V / PROJ_U, -1.0, PROJ_VY])
    # ((u + a t - CU) / RU)^2 + ((v + b t - CV) / RV)^2 = 1, solved for t.
    du, dv = (u - CURVE_U) / CURVE_RU, (v - CURVE_V) / CURVE_RV
    au, av = toward[0] / CURVE_RU, toward[1] / CURVE_RV
    quad = (au * au + av * av, 2 * (du * au + dv * av), du * du + dv * dv - 1.0)
    disc = quad[1] ** 2 - 4 * quad[0] * quad[2]
    if disc < 0:
        return False
    root = math.sqrt(disc)
    return any(
        t > 1e-9 and 0.0 <= h + toward[2] * t <= BOX_H
        for t in ((-quad[1] - root) / (2 * quad[0]), (-quad[1] + root) / (2 * quad[0]))
    )


def _occluded_by_dome(u: float, v: float, h: float = 0.0) -> bool:
    """Whether the dome stands between this point (a floor item, or a rim of one) and the eye.

    Same walk as `_occluded_by_pillar`, against the half-ellipsoid instead of
    the cylinder — so an item inside the footprint is behind the near surface,
    an item just beyond it may be behind both, and one in front of it is behind
    neither. The walk only climbs, so the one height check is that it leaves
    the solid above the floor: a rim under the floor (the floor is glass too,
    and a sphere's lower half shows through it) can cross the full ellipsoid's
    mirrored lower half, which is not there.
    """
    toward = np.array([PROJ_V / PROJ_U, -1.0, PROJ_VY])
    du, dv, dh = (u - CURVE_U) / CURVE_RU, (v - CURVE_V) / CURVE_RV, h / DOME_H
    au, av, ah = toward[0] / CURVE_RU, toward[1] / CURVE_RV, toward[2] / DOME_H
    a = au * au + av * av + ah * ah
    b = 2 * (du * au + dv * av + dh * ah)
    disc = b * b - 4 * a * (du * du + dv * dv + dh * dh - 1.0)
    if disc < 0:
        return False
    leaves = (-b + math.sqrt(disc)) / (2 * a)
    return leaves > max(1e-9, -h / toward[2])


#: The directions round a sphere's page outline its rims are walked from, every
#: 45°. The lower half counts too: it is under the floor, and the floor is
#: glass, so that half shows through it and can stand behind a solid as well.
RIM_ANGLES = tuple(range(0, 360, 45))


def _partly_behind(occluded, u: float, v: float) -> bool:
    """Whether any of this floor item stands behind the solid, not just its centre.

    `occluded` walks one ray, from a point toward the eye, and a sphere is
    wider than a ray: one whose centre sits just outside the solid's silhouette
    can still overhang its outline, and if it stands *behind* the solid that
    overhang is behind the glass. Tested on the centre alone, those spheres were
    painted in front and sat on top of the solid's edge. So the walk is repeated
    from points round the sphere's page outline (`RIM_ANGLES`), each at the
    centre's depth: a step along `u` is purely sideways in this projection, and
    a step in height is purely upward. Testing only the two side rims missed a
    sphere whose lower edge overhung the dome's outline (#4517). Any hit sends the
    whole item under the glass, where only the part the solid covers is tinted.
    A sphere in *front* of the solid is unaffected: no ray from it toward the
    eye meets the solid, rim or centre.
    """
    if occluded(u, v, 0.0):
        return True
    radius = ITEM_R * 1.02
    return any(
        occluded(
            u + radius * math.cos(math.radians(angle)) / (PROJ_SCALE * PROJ_U),
            v,
            radius * math.sin(math.radians(angle)) / PROJ_SCALE,
        )
        for angle in RIM_ANGLES
    )


def _painted_back_to_front(points: np.ndarray, base_z: float) -> list[tuple[int, float]]:
    """Item indices in paint order, far first, with the z-order each gets.

    Two spheres whose circles overlap have to resolve, and the one nearer the
    eye has to win. In this projection nearer is simply smaller v, so sorting on
    it is the whole of the depth test.
    """
    order = sorted(range(len(points)), key=lambda i: -points[i][1])
    return [(i, base_z + 0.9 * k / max(len(order) - 1, 1)) for k, i in enumerate(order)]


def _floor_items(ax: plt.Axes, floor: np.ndarray, plane: plt.Polygon, behind: set[int]) -> list[Layer]:
    """The corpus the votes came from, lying in the floor of the room.

    Returns the submerged regions, grown by `CLIP_PAD`, split into the two
    bands: anything lying *in* the floor has to be drawn over them, rim
    included, and a piece of boundary showing through a sphere that stands
    behind the glass has to be drawn *under* the glass or it comes out brighter
    than the sphere around it.
    """
    voted = _votes(floor)
    crescents: dict[bool, list[np.ndarray]] = {True: [], False: []}
    for index, depth_z in _painted_back_to_front(floor, 0.0):
        u, v = floor[index]
        point = proj(u, v, 0.0)
        zorder = (Z_BEHIND_PILLAR if index in behind else Z_IN_FRONT) + depth_z
        if voted.get(index) == "good":
            _sunk_glyph(ax, point, INTRO._check, GREEN_SUNK, zorder)
        elif voted.get(index) == "bad":
            _sunk_glyph(ax, point, INTRO._cross, RED_SUNK, zorder)
        else:
            crescent = _item(ax, point, zorder, plane)
            if crescent is not None:
                crescents[index in behind].append(crescent)
    return [(crescents[True], Z_BEHIND_OVERLAY), (crescents[False], Z_FRONT_OVERLAY)]


def _draw_in_plane(ax: plt.Axes, xy: np.ndarray, layers: list[Layer], base_z: float = Z_IN_PLANE, **style) -> None:
    """Draw a curve that lies *in* a plane, over the spheres' submerged halves.

    A sphere's crescent is the part of it under the plane, so anything painted
    on the plane passes in front of it — and a curve that stops at the edge of
    every crescent it meets says the opposite: that the crescents are sitting on
    the plane rather than cut into it.

    But the plane is also what everything standing *in* it rises out of, so the
    same curve has to pass behind every part of every item that is above the
    waterline: a sphere's white cap, and the bright half of a vote mark. Those
    two rules are the same rule stated from either side of the surface, and the
    way to satisfy both is to draw the curve under all of it and then put it
    back exactly where the plane is what you are looking at.

    So the base stroke goes below every item, and one clipped copy per band goes
    just above that band, clipped to the union of its crescents — which is the
    only place a copy can show. Per band rather than once over everything,
    because a crescent behind the pillar's glass is tinted and the boundary
    showing through it has to be tinted with it.

    `base_z` is where the base stroke goes, and it is below the glass by
    default because that is where the pillar's foot belongs: look at the far
    half of it and you are looking through the near wall. The pillar's *mouth*
    is the exception and says so at the call site — walk from any point on it
    toward the eye and you leave through the open top, near half and far half
    alike, so no part of it is seen through anything.
    """
    ax.plot(xy[:, 0], xy[:, 1], **{**style, "zorder": base_z})
    for crescents, zorder in layers:
        if not crescents:
            continue
        over = ax.plot(xy[:, 0], xy[:, 1], **{**style, "zorder": zorder})[0]
        over.set_clip_path(
            MplPath.make_compound_path(*(MplPath(np.vstack([c, c[:1]]), closed=True) for c in crescents)),
            ax.transData,
        )


#: The flat version of the floor, for the first page. Sized to the tallest the
#: slide allows under the headline, which fixes the width by aspect and leaves
#: it centred — there is no room to lean into here, because a plan view has no
#: receding axis to lean along.
PLAN_SCALE = 1.5
PLAN_X0, PLAN_Y0 = 0.5 * (CANVAS[0] - PLAN_SCALE * BOX_U), 0.30


def plan(u: float, v: float) -> np.ndarray:
    """One floor point on the page, seen from straight above."""
    return np.array([PLAN_X0 + PLAN_SCALE * u, PLAN_Y0 + PLAN_SCALE * v])


def _plan_stage() -> plt.Figure:
    """The first page: what the deck has already shown, and nothing else.

    The same corpus, the same votes, the same boundary, flat — so the page after
    it adds exactly one thing, a room, and not one new item to look at. That is
    the whole job of this page: the surprise on the pages that follow has to be
    the *height*, and it cannot be if the audience is still working out whether
    these are the items they were looking at a minute ago.
    """
    fig, ax = _canvas()
    ax.add_patch(
        plt.Rectangle(
            (PLAN_X0, PLAN_Y0),
            PLAN_SCALE * BOX_U,
            PLAN_SCALE * BOX_V,
            facecolor=PLANE_FILL,
            edgecolor=CELL_LINE,
            linewidth=1.6,
            zorder=0,
        )
    )
    angles = np.linspace(0, 2 * np.pi, 180)
    curve = np.array([plan(CURVE_U + CURVE_RU * math.cos(a), CURVE_V + CURVE_RV * math.sin(a)) for a in angles])
    ax.plot(curve[:, 0], curve[:, 1], color=BLUE, linewidth=OUTLINE_LW, zorder=Z_IN_PLANE)
    floor = _spaced(N_ITEMS, *ITEM_LIMITS, ITEM_GAP)
    voted = _votes(floor)
    for index in range(len(floor)):
        point = plan(*floor[index])
        if voted.get(index) == "good":
            INTRO._check(ax, point)
        elif voted.get(index) == "bad":
            INTRO._cross(ax, point)
        else:
            _item(ax, point, Z_IN_FRONT)
    return fig


def _depth_stage(stage: int) -> plt.Figure:
    if stage == 1:
        return _plan_stage()
    fig, ax = _canvas()
    planes = _wireframe(ax)

    angles = np.linspace(0, 2 * np.pi, 180)
    base = _footprint(angles, 0.0)
    floor = _spaced(N_ITEMS, *ITEM_LIMITS, ITEM_GAP)
    if stage == 4:
        behind = {i for i, (u, v) in enumerate(floor) if _partly_behind(_occluded_by_pillar, u, v)}
    elif stage == 5:
        behind = {i for i, (u, v) in enumerate(floor) if _partly_behind(_occluded_by_dome, u, v)}
    else:
        behind = set()

    # The floor goes down first, because what lies *in* the floor has to be
    # drawn over its spheres' submerged halves and needs their outlines to do
    # it. Call order is not paint order — every artist here carries a z.
    crescents = _floor_items(ax, floor, planes[0], behind)

    top = _footprint(angles, TUBE_H)
    if stage == 4:
        _tube(ax, base, top, angles)
    if stage == 5:
        _dome(ax)
    _draw_in_plane(ax, base, crescents, color=BLUE, linewidth=OUTLINE_LW)

    lids: list[np.ndarray] = []
    if stage >= 3:
        ceiling = _spaced(N_ITEMS, *ITEM_LIMITS, ITEM_GAP, seed=NEW_SEED)
        for index, depth_z in _painted_back_to_front(ceiling, 0.0):
            crescent = _item(ax, proj(*ceiling[index], NEW_H), Z_CEILING + depth_z, planes[1])
            if crescent is not None:
                lids.append(crescent)
    if stage == 4:
        _draw_in_plane(ax, top, [(lids, Z_CEILING_OVERLAY)], base_z=Z_PILLAR + 0.05, color=BLUE, linewidth=OUTLINE_LW)

    return fig


def depth_fig() -> None:
    """Domain shift, as a floor in a room.

    Five pages, and the first two are one picture. It opens flat, on exactly
    what the deck has already drawn — the same corpus, the same votes, the same
    boundary — and then that plane becomes the floor of a room with no item
    added or moved. Spending a page on a picture the audience has seen is what
    buys the rest of them: the surprise here is the *height*, and it cannot land
    while anyone is still working out whether these are the items from a minute
    ago.

    Then a second collection arrives on the ceiling — the same count and spread
    as the first, because it is a domain and not a handful of outliers — and the
    last two pages are the rival readings of what the cut says about it. Never
    together: a page holding both asks the room to compare where it should be
    surprised.

    The pillar goes first, because it is what ships. The head is a
    `Linear(D, 1)` and a linear score is *exactly* constant along every
    direction its weight vector does not point in (`vtscore/training/mlp.py`,
    the `LINEAR_SVM_HEAD` sentinel), so the boundary has no lid: it extrudes,
    and the new collection is *sorted* by it, into matches and non-matches, with
    no hint that anything is being extrapolated. The dome goes second: the
    concept stops, which is what anybody drawing a boundary through a corpus
    assumes without saying so. It is cut from the same glass and drawn in the
    same hand as the pillar, because it is the same claim about the same
    boundary and only the lid is at issue; a dome sketched in another colour or
    another dash would be answering a different question. Every vote in the
    picture is on the floor, so nothing in the picture chooses between them.

    The room is the reason this reads at all. The earlier attempt at this slide
    drew the same idea on a sphere and there was nothing to judge position
    against; a floor, four posts and a lid give every item a place.
    """
    for stage in range(1, DEPTH_STAGES):
        save(_depth_stage(stage), OUT, f"atlas-depth.build{stage}.webp", column=FULL_BLEED, tight=False, notch=NOTCH)
    save(_depth_stage(DEPTH_STAGES), OUT, "atlas-depth.webp", column=FULL_BLEED, tight=False, notch=NOTCH)


# ──────────────────────────────────────────────────────────────────────────────
# 3b. Far Out — the atlas finding the room's second collection (#4517)
# ──────────────────────────────────────────────────────────────────────────────

FAR_STAGES = 4
#: How many floor points the atlas is built on. Denser than the drawing's
#: `N_ITEMS`, because the shipped atlas splits only nodes of twenty and more
#: and calibrates only those: built on the drawn seventy-six it would have a
#: handful of calibrated nodes and say little. The drawn floor items are then
#: *held out* — the atlas never saw them — which is the honest test.
FAR_BUILD = 2000
FAR_SEED = 23
#: The threshold an item is flagged at: the domain-shift report's own alpha.
FAR_ALPHA = 0.05
#: What the notes say the atlas makes of each collection, flagged of drawn.
FAR_FLAGGED = (2, 68)
#: The cells whose centres the slide draws: the atlas's own rule (three at a
#: time) two levels down, nine cells, on the floor it is built on. Like the
#: plane's partition above, a k-means on positions rather than on the centred
#: directions the shipped atlas splits: the picture is "how far from your
#: cell's centre", and that is what the shipped score ranks (owner, #4872).
FAR_LEVELS = 2
#: A centre's mark: a white cube, a shape no item has, half sunk in the floor as
#: every item is (owner, #4884: not a black diamond laid over everything). Its
#: half-side in world units, sized so the cube stands as big on the page as a
#: sphere does.
FAR_CUBE = 0.11
#: The cube's faces, lit from above: the top white, the near side a shade
#: down, the right side another. Under the floor each face is seen through the
#: floor's own wash, as a sphere's crescent is.
CUBE_FACES = {"top": "white", "front": "#f3f5f8", "right": "#e2e7ee"}
CUBE_FACES_SUNK = {"front": PLANE_FILL, "right": "#d6dde6"}
#: The spokes from each item to its cell's centre.
FAR_SPOKE = {"color": SOFT, "linewidth": 1.2}
#: How finely a spoke is cut to be tested against what stands in front of it,
#: in page units, before each change between seen and hidden is refined.
SPOKE_STEP = 0.01
#: Every object is painted far to near, two z-steps apiece (a sphere spends 0.3
#: of its own), the ceiling's after the floor's; the parts of the spokes the eye
#: can see go over all of them, and the box's three near edges over those.
Z_FAR_FLOOR, Z_FAR_CEILING, Z_FAR_SPOKES, Z_FAR_NEAR_EDGES = 1.0, 200.0, 400.0, 410.0


@functools.lru_cache(maxsize=1)
def _far_build() -> np.ndarray:
    rng = np.random.default_rng(FAR_SEED)
    (u0, u1), (v0, v1) = ITEM_LIMITS
    return np.column_stack([rng.uniform(u0, u1, FAR_BUILD), rng.uniform(v0, v1, FAR_BUILD)])


@functools.lru_cache(maxsize=1)
def _far_pvalues() -> tuple[np.ndarray, np.ndarray]:
    """The shipped atlas, built on the floor, scoring the drawn floor items and the ceiling.

    Real code on drawing coordinates: `CoverageAtlas` and its
    `typicality_pvalues`, the same pair the domain-shift route calls. Each item
    is its room position `(u, v, h)` as a three-number vector.
    """
    from vtscore.coverage.atlas import CoverageAtlas  # noqa: PLC0415

    build = _far_build()
    build3 = np.column_stack([build, np.zeros(len(build))])
    atlas = CoverageAtlas({i: row for i, row in enumerate(build3.astype(np.float32))})
    floor = _spaced(N_ITEMS, *ITEM_LIMITS, ITEM_GAP)
    ceiling = _spaced(N_ITEMS, *ITEM_LIMITS, ITEM_GAP, seed=NEW_SEED)

    def scored(points: np.ndarray, h: float) -> np.ndarray:
        return atlas.typicality_pvalues(np.column_stack([points, np.full(len(points), h)]).astype(np.float32))

    return scored(floor, 0.0), scored(ceiling, NEW_H)


@functools.lru_cache(maxsize=1)
def _far_tree() -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """The floor's first split and each part's own split: ``[(centre, [child centres])]``."""
    build = _far_build()
    top = KMeans(n_clusters=SPLIT_K, n_init=10, random_state=0).fit(build)
    out = []
    for k in range(SPLIT_K):
        part = build[top.labels_ == k]
        sub = KMeans(n_clusters=SPLIT_K, n_init=10, random_state=0).fit(part)
        out.append((top.cluster_centers_[k], list(sub.cluster_centers_)))
    return out


def _far_centre(u: float, v: float) -> np.ndarray:
    """The centre of the cell an item routes to: the nearest part, then that part's nearest child, as the atlas walks."""
    q = np.array([u, v])
    centre, kids = min(_far_tree(), key=lambda node: float(np.hypot(*(node[0] - q))))
    return min(kids, key=lambda c: float(np.hypot(*(c - q))))


def far_fig() -> None:
    """The atlas finding Out of Its Depth's second collection, drawn in the same room (#4517).

    Four pages (owner, #4872: "show the centers in a), the distance lines in
    b), the second domain in c), and the (longer) distance lines in d)"): the
    floor with the centres of the atlas's cells; each floor item's spoke to its
    cell's centre, short; the second collection on the ceiling, as Out of Its
    Depth showed it; and its spokes, each down to the nearest centre, long.
    How far an item is from its cell's centre, against how far the cell's own
    items are, is what the shipped score ranks, and the counts the notes give
    are that score's (`_far_pvalues`).
    """
    floor_p, ceiling_p = _far_pvalues()
    flagged = (int((floor_p < FAR_ALPHA).sum()), int((ceiling_p < FAR_ALPHA).sum()))
    assert flagged == FAR_FLAGGED, f"the notes say {FAR_FLAGGED} flagged, the atlas says {flagged}"
    for stage in range(1, FAR_STAGES):
        save(_far_stage(stage), OUT, f"atlas-far.build{stage}.webp", column=FULL_BLEED, tight=False, notch=NOTCH)
    save(_far_stage(FAR_STAGES), OUT, "atlas-far.webp", column=FULL_BLEED, tight=False, notch=NOTCH)


#: The way toward the eye, in world units: the displacement that leaves a point
#: where it is on the page. Nearer is south, and up (see `_occluded_by_pillar`).
TOWARD = np.array([PROJ_V / PROJ_U, -1.0, PROJ_VY])
#: A sphere's radius in the world, as its page radius is drawn.
ITEM_R_WORLD = ITEM_R / PROJ_SCALE


def _sunk(colour: str) -> str:
    """*colour* as seen through a plane: the map that takes white to the plane's fill and black to a sunk rim.

    It is the one the spheres already obey — their white face goes to
    `PLANE_FILL` under the floor and their black rim to `OUTLINE_SUNK` — taken
    per channel and applied to anything else seen through the same glass.
    """
    hexes = (colour, PLANE_FILL, OUTLINE_SUNK)
    c, white, black = (np.array([int(h[i : i + 2], 16) for i in (1, 3, 5)]) / 255.0 for h in hexes)
    out = black + c * (white - black)
    return "#" + "".join(f"{round(255 * v):02x}" for v in out)


def _cube(ax: plt.Axes, u: float, v: float, zorder: float) -> None:
    """A cell's centre: a cube half sunk in the floor, drawn as a sphere is.

    The camera sees the top, the south face and the east face. The floor cuts
    the two side faces at half height: above it they are white-ish with ink
    edges, below it they are seen through the floor in its wash, edged in the
    sunk grey, and the cut itself is the waterline, in the waterline's grey.
    """
    a = FAR_CUBE

    def at(du: float, dv: float, dh: float) -> np.ndarray:
        return proj(u + du * a, v + dv * a, dh * a)

    sunk_edge = {"color": OUTLINE_SUNK, "linewidth": 1.4, "solid_capstyle": "round"}
    ink_edge = {"color": INK, "linewidth": 1.7, "solid_capstyle": "round", "solid_joinstyle": "round"}
    faces = {
        "front": lambda h0, h1: [at(-1, -1, h0), at(1, -1, h0), at(1, -1, h1), at(-1, -1, h1)],
        "right": lambda h0, h1: [at(1, -1, h0), at(1, 1, h0), at(1, 1, h1), at(1, -1, h1)],
    }
    for name, face in faces.items():
        ax.add_patch(plt.Polygon(face(-1, 0), closed=True, facecolor=CUBE_FACES_SUNK[name], edgecolor="none",
                                 zorder=zorder))  # fmt: skip
        ax.add_patch(plt.Polygon(face(0, 1), closed=True, facecolor=CUBE_FACES[name], edgecolor="none",
                                 zorder=zorder + 0.1))  # fmt: skip
    top = [at(-1, -1, 1), at(1, -1, 1), at(1, 1, 1), at(-1, 1, 1)]
    ax.add_patch(plt.Polygon(top, closed=True, facecolor=CUBE_FACES["top"], edgecolor="none", zorder=zorder + 0.1))
    # Under the floor: the bottom edges and the lower halves of the three near verticals.
    for path in (
        [at(-1, -1, 0), at(-1, -1, -1), at(1, -1, -1), at(1, 1, -1), at(1, 1, 0)],
        [at(1, -1, 0), at(1, -1, -1)],
    ):
        ax.plot(*np.array(path).T, zorder=zorder + 0.05, **sunk_edge)
    # The waterline, where the floor cuts the two faces.
    ax.plot(*np.array([at(-1, -1, 0), at(1, -1, 0), at(1, 1, 0)]).T, color=CELL_LINE, linewidth=1.1,
            zorder=zorder + 0.15)  # fmt: skip
    # Above it: the upper halves of the verticals and the top's four edges.
    for path in (
        [at(-1, -1, 0), at(-1, -1, 1), at(-1, 1, 1), at(1, 1, 1), at(1, 1, 0)],
        [at(1, -1, 0), at(1, -1, 1)],
        [at(-1, -1, 1), at(1, -1, 1), at(1, 1, 1)],
    ):
        ax.plot(*np.array(path).T, zorder=zorder + 0.2, **ink_edge)


def _in_polygon(q: np.ndarray, polygon: np.ndarray) -> bool:
    return bool(MplPath(polygon).contains_point(tuple(q)))


class _Scene:
    """What stands between a point on a spoke and the eye.

    The figure is painted far to near, which settles every sphere and cube
    against every other; a spoke cannot be, because it runs from one depth to
    another. So each spoke is cut fine and every piece asks the scene whether
    anything covers it, and only the pieces nothing covers are drawn, over
    everything. That one rule is all of the owner's (#4884): a spoke passes
    behind whichever end is nearer the eye and over the other, a spoke from the
    ceiling passes in front of the floor's items it crosses, and the piece of
    one below the ceiling is seen through the ceiling.
    """

    def __init__(self, floor: np.ndarray, centres: list[np.ndarray], ceiling: np.ndarray | None) -> None:
        self.floor = [np.array([u, v, 0.0]) for u, v in floor]
        self.cubes = [np.array([u, v, 0.0]) for u, v in centres]
        self.ceiling = [] if ceiling is None else [np.array([u, v, NEW_H]) for u, v in ceiling]
        corners = [(0, 0), (BOX_U, 0), (BOX_U, BOX_V), (0, BOX_V)]
        self.lid = np.array([proj(u, v, BOX_H) for u, v in corners])
        self.crescent = _submerged_outline(CLIP_PAD)

    def _behind_sphere(self, point: np.ndarray, q: np.ndarray, centre: np.ndarray) -> bool:
        c = proj(*centre)
        if float(np.hypot(*(q - c))) > ITEM_R + CLIP_PAD:
            return False
        if point[2] == centre[2]:
            # In the sphere's own plane: under the cap, unless it is in front of the sunk half.
            return not _in_polygon(q - c, self.crescent)
        behind = float((centre - point) @ TOWARD) > 0.0
        return behind or float(np.linalg.norm(point - centre)) < ITEM_R_WORLD

    @staticmethod
    def _behind_cube(point: np.ndarray, centre: np.ndarray) -> bool:
        """Whether the ray from *point* toward the eye meets the cube: the slab test, from t = 0."""
        half = FAR_CUBE + CLIP_PAD / PROJ_SCALE
        lo, hi = 0.0, math.inf
        for k in range(3):
            near, far = centre[k] - half, centre[k] + half
            if abs(TOWARD[k]) < 1e-12:
                if not near <= point[k] <= far:
                    return False
                continue
            t0, t1 = sorted(((near - point[k]) / TOWARD[k], (far - point[k]) / TOWARD[k]))
            lo, hi = max(lo, t0), min(hi, t1)
            if lo > hi:
                return False
        return True

    def look(self, point: np.ndarray) -> str | None:
        """``None`` if something covers *point*, else how it is seen: ``"clear"``, or ``"sunk"`` under the lid."""
        q = proj(*point)
        if any(self._behind_cube(point, c) for c in self.cubes):
            return None
        if any(self._behind_sphere(point, q, c) for c in self.floor + self.ceiling):
            return None
        if point[2] < NEW_H - 1e-9 and _in_polygon(q, self.lid):
            return "sunk"
        return "clear"


def _spoke(ax: plt.Axes, scene: _Scene, start: np.ndarray, end: np.ndarray) -> None:
    """One spoke, drawn wherever nothing stands in front of it, tinted where the lid is over it."""
    length = float(np.hypot(*(proj(*end) - proj(*start))))
    ts = np.linspace(0.0, 1.0, max(2, math.ceil(length / SPOKE_STEP) + 1))

    def at(t: float) -> np.ndarray:
        return start + t * (end - start)

    looks = [scene.look(at(t)) for t in ts]
    # Each change between one look and the next, refined to a hair, so a spoke
    # ends on an object's edge rather than on the nearest step.
    cuts = [0.0]
    for i in range(1, len(ts)):
        if looks[i] != looks[i - 1]:
            lo, hi = ts[i - 1], ts[i]
            for _ in range(20):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if scene.look(at(mid)) == looks[i - 1] else (lo, mid)
            cuts.append(0.5 * (lo + hi))
    cuts.append(1.0)
    for t0, t1 in zip(cuts, cuts[1:], strict=False):
        look = scene.look(at(0.5 * (t0 + t1)))
        if look is None:
            continue
        colour = FAR_SPOKE["color"] if look == "clear" else _sunk(FAR_SPOKE["color"])
        ax.plot(*np.array([proj(*at(t0)), proj(*at(t1))]).T, color=colour, linewidth=FAR_SPOKE["linewidth"],
                solid_capstyle="butt", zorder=Z_FAR_SPOKES)  # fmt: skip


def _near_edges(ax: plt.Axes) -> None:
    """The three edges of the box that stand in front of something: the near right post, and the lid's near and right edges.

    Every other edge is behind everything it crosses on the page, so the
    wireframe's own strokes under the items are right for them. These three
    are nearer than the floor's items behind the post and the spokes under the
    lid (owner, #4884: the post in front of the lower spheres), so they are
    drawn again on top, in the same weight.
    """
    for a, b in (((1, 0, 0), (1, 0, 1)), ((0, 0, 1), (1, 0, 1)), ((1, 0, 1), (1, 1, 1))):
        ends = [proj(u * BOX_U, v * BOX_V, h * BOX_H) for u, v, h in (a, b)]
        ax.plot(*np.array(ends).T, color=RULE, linewidth=1.2, zorder=Z_FAR_NEAR_EDGES)


def _far_stage(stage: int) -> plt.Figure:
    fig, ax = _canvas()
    planes = _wireframe(ax)
    floor = _spaced(N_ITEMS, *ITEM_LIMITS, ITEM_GAP)
    centres = [c for _centre, kids in _far_tree() for c in kids]
    # The floor's spheres and cubes, far to near: whichever is further south is nearer and covers the other.
    things = [("item", tuple(p)) for p in floor] + [("cube", tuple(c)) for c in centres]
    for k, (kind, (u, v)) in enumerate(sorted(things, key=lambda thing: -thing[1][1])):
        zorder = Z_FAR_FLOOR + 0.5 * k
        if kind == "item":
            _item(ax, proj(u, v, 0.0), zorder, planes[0])
        else:
            _cube(ax, u, v, zorder)
    ceiling = _spaced(N_ITEMS, *ITEM_LIMITS, ITEM_GAP, seed=NEW_SEED) if stage >= 3 else None
    if ceiling is not None:
        for index, depth_z in _painted_back_to_front(ceiling, 0.0):
            _item(ax, proj(*ceiling[index], NEW_H), Z_FAR_CEILING + 100 * depth_z, planes[1])
    scene = _Scene(floor, centres, ceiling)
    if stage >= 2:
        for u, v in floor:
            _spoke(ax, scene, np.array([u, v, 0.0]), np.array([*_far_centre(u, v), 0.0]))
    if stage >= 4:
        for u, v in ceiling:
            _spoke(ax, scene, np.array([u, v, NEW_H]), np.array([*_far_centre(u, v), 0.0]))
    _near_edges(ax)
    return fig


if __name__ == "__main__":
    blindspot_fig()
    cells_fig()
    depth_fig()
    far_fig()
    print("wrote figures to", OUT)
