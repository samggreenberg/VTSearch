#!/usr/bin/env python
"""Introduction figures for the VTSearch decks.

Run from the repo root:

    python slides/figs/src/make-intro-figs.py

One figure lives here: `vote-boundary`, the picture of what a vote actually
*does*. The deck's other mechanism figures are drawn in score space — a number
line with a cut on it — which is the right space for a talk about the cut, and
the wrong one for a talk about the loop, because it cannot show why the item
the user is asked about next is the one it is.

So this figure works in *item* space: every media item is a point, and the
detector is the closed curve around the ones it currently calls a match. That
is a genuine 2D analogue rather than a picture of the shipped model — VTSearch
trains a linear SVM in embedding space, where the boundary is a hyperplane, and
a hyperplane in two dimensions is a straight line that cannot enclose anything.
An RBF SVM on two dimensions is the same object with the curvature the audience
would otherwise have to imagine, so that is what is fitted here: the boundary
on every stage is a real `sklearn` decision contour over the votes shown, not a
hand-drawn oval, and the item selected next is really the unlabeled point
nearest the boundary — the app's own `Hard` rule.

**On the build rule.** `slides/STYLE.md` says a reveal adds ink and nothing
moves or restyles between pages. This figure breaks that in exactly two places,
both because the restyle *is* the mechanism: the item under consideration turns
from a hollow circle into a solid disc and then into a check or a cross, and
the boundary moves once the retrain has happened (the previous boundary stays
on the slide, faded, so what the audience sees is where it went, not a curve
teleporting). Everything else — the canvas, the crop, every point's position —
is pinned across all six pages.
"""

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
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slide_figure import SIDEBAR_WIDE, save, tight_box  # noqa: E402

OUT = Path(__file__).resolve().parent.parent

INK = "#14181f"
SOFT = "#5b6472"
BLUE = "#0b5fa5"  # the detector's boundary — the shipped decision
RUST = "#b45309"  # the Bad side
GREEN = "#0d8a5f"  # the Good side
GHOST = "#aab3c0"  # the boundary as it was before the retrain

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "font.size": 17,
        "text.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    }
)

#: The figure is drawn in a 10x10 unit square at this many printed points per
#: unit, which fixes the size everything else is expressed in.
UNIT_PT = 46.0
CANVAS = (10.0, 10.0)

#: Item glyph radius, in figure units. Checks and crosses are drawn to the same
#: half-width, so a voted item occupies the same patch of the field as the
#: hollow circle it replaces and the field's texture does not change density
#: as votes accumulate.
R = 0.19

#: How far inside the boundary an unlabeled item has to sit before the figure
#: calls it obvious, in figure units — a little over two glyph radii, so the
#: gap is visible as a gap at slide size rather than inferred from the maths.
OBVIOUS_GAP = 0.55

#: Minimum separation between two highlighted items, in figure units — wide
#: enough that their halos cannot touch.
OBVIOUS_APART = 1.25

VOTE_BOUNDARY_STAGES = 6


# ──────────────────────────────────────────────────────────────────────────────
# The field of items
# ──────────────────────────────────────────────────────────────────────────────


def _field() -> np.ndarray:
    """A fixed field of items in 2D, spread with a minimum separation.

    Poisson-ish rather than uniform: a uniform draw clumps, and a clump reads
    as one blurred object at slide size rather than as several items.
    """
    rng = np.random.default_rng(7)
    pts: list[np.ndarray] = []
    while len(pts) < 54:
        p = rng.uniform([0.9, 0.9], [9.1, 9.1])
        if all(np.hypot(*(p - q)) > 0.86 for q in pts):
            pts.append(p)
    return np.array(pts)


#: The concept the user is actually looking for, as the ground truth the field
#: was drawn against: items inside this disc are matches. Nothing in the figure
#: draws it — the whole point is that only the user knows where it is.
TRUE_CENTRE = np.array([6.35, 6.15])
TRUE_RADIUS = 2.55


def _truth(pts: np.ndarray) -> np.ndarray:
    return np.hypot(*(pts - TRUE_CENTRE).T) < TRUE_RADIUS


#: The votes the user has already cast when the slide opens, as indices into
#: `_field()`. Chosen by hand for legibility — three Goods spread across the
#: concept and four Bads spread around the rest of the field — but checked
#: against `_truth` below, so a change to the field cannot silently make the
#: opening votes disagree with the concept they are supposed to describe.
SEED_GOOD = (0, 7, 13, 34, 46)
SEED_BAD = (2, 6, 14, 28, 49)


def _fit(pts: np.ndarray, good: tuple[int, ...], bad: tuple[int, ...]) -> SVC:
    """The detector, trained on the votes cast so far.

    An RBF SVM standing in for the shipped linear one — see the module
    docstring. `gamma` is fixed rather than scaled off the data so that the
    boundary's curvature is the same object before and after the extra vote;
    with `gamma="scale"` the retrain would change the kernel as well as the
    fit, and the audience would be watching two things move at once.
    """
    idx = list(good) + list(bad)
    y = [1] * len(good) + [0] * len(bad)
    model = SVC(kernel="rbf", gamma=0.34, C=12.0)
    model.fit(pts[idx], y)
    return model


def _next_question(pts: np.ndarray, model: SVC, labeled: tuple[int, ...]) -> int:
    """The unlabeled item the app would ask about next: nearest the boundary.

    This is the `Hard` selection rule, which is the one the slide is about —
    the item whose answer the detector cannot currently guess.
    """
    d = np.abs(model.decision_function(pts))
    d[list(labeled)] = np.inf
    return int(np.argmin(d))


def _obvious(pts: np.ndarray, model: SVC, labeled: tuple[int, ...], n: int = 3) -> list[int]:
    """The unlabeled items sitting furthest *inside* the boundary.

    The foil for the slide's argument: these are the ones a ranked result list
    puts on top, and asking about them is how a session spends twenty votes
    learning nothing it did not already know.

    Ranked by distance to the drawn curve rather than by decision value,
    because those are not the same picture. Decision value is what the model is
    confident about; the audience reads confidence off the *gap* between an item
    and the line, so an item the model scores highly but that happens to sit
    near the curve would be highlighted here and look like exactly the
    borderline case the next stage is about.
    """
    axis = np.linspace(0.0, CANVAS[0], 240)
    xx, yy = np.meshgrid(axis, axis)
    cells = np.c_[xx.ravel(), yy.ravel()]
    inside = (model.decision_function(cells) > 0).reshape(xx.shape)
    # The curve, as the inside cells that have an outside neighbour. Close
    # enough at this resolution to measure a gap in figure units.
    rim = inside & ~(np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1))
    curve = cells[rim.ravel()]
    depth = np.where(
        model.decision_function(pts) > 0,
        np.min(np.hypot(*(curve[:, None, :] - pts[None, :, :]).transpose(2, 0, 1)), axis=0),
        -1.0,
    )
    depth[list(labeled)] = -1.0
    # A gap, not a rank: an item is only "obvious" if it is clear of the curve
    # by more than the eye reads as touching it. Taking a fixed top-N instead
    # highlights whatever is least-near the line even when everything inside is
    # crowded against it — which draws a halo round exactly the borderline item
    # the next stage exists to contrast with.
    picked: list[int] = []
    for i in np.argsort(depth)[::-1]:
        if len(picked) == n or depth[i] < OBVIOUS_GAP:
            break
        # Keep them apart: two overlapping halos read as one smudged mark, and
        # the point being made is that there are *several* such items.
        if all(np.hypot(*(pts[i] - pts[j])) > OBVIOUS_APART for j in picked):
            picked.append(int(i))
    return picked


# ──────────────────────────────────────────────────────────────────────────────
# Glyphs
# ──────────────────────────────────────────────────────────────────────────────


def _circle(ax: plt.Axes, p: np.ndarray, *, filled: bool = False) -> None:
    """An item: hollow while unlabeled, solid black while it is being asked."""
    ax.add_patch(
        plt.Circle(
            tuple(p),
            R,
            facecolor=INK if filled else "none",
            edgecolor=INK,
            linewidth=2.6 if filled else 1.7,
            zorder=5 if filled else 3,
        )
    )


def _check(ax: plt.Axes, p: np.ndarray) -> None:
    x, y = p
    ax.plot(
        [x - R, x - 0.28 * R, x + R],
        [y + 0.05 * R, y - 0.85 * R, y + R],
        color=GREEN,
        linewidth=3.4,
        solid_capstyle="round",
        solid_joinstyle="miter",
        zorder=6,
    )


def _cross(ax: plt.Axes, p: np.ndarray) -> None:
    x, y = p
    for dx in (-1, 1):
        ax.plot(
            [x - dx * R, x + dx * R],
            [y - R, y + R],
            color=RUST,
            linewidth=3.2,
            solid_capstyle="round",
            zorder=6,
        )


def _boundary(ax: plt.Axes, model: SVC, *, ghost: bool = False) -> None:
    """The detector, drawn as the closed curve around what it calls a match."""
    grid = np.linspace(-0.6, 10.6, 420)
    xx, yy = np.meshgrid(grid, grid)
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contour(
        xx,
        yy,
        zz,
        levels=[0.0],
        colors=[GHOST if ghost else BLUE],
        linewidths=2.0 if ghost else 3.0,
        linestyles="--" if ghost else "-",
        zorder=1 if ghost else 2,
    )


def _halo(ax: plt.Axes, p: np.ndarray) -> None:
    """A second, wider ring: "the detector is already sure about this one".

    Drawn rather than written because the figure carries no text at all — the
    slide's own bullets narrate it, which is what keeps every mark on the field
    at a size the back row can resolve.
    """
    ax.add_patch(plt.Circle(tuple(p), R + 0.22, facecolor="none", edgecolor=SOFT, linewidth=1.6, zorder=2))
    ax.add_patch(plt.Circle(tuple(p), R + 0.42, facecolor="none", edgecolor=SOFT, linewidth=1.1, zorder=2))


# ──────────────────────────────────────────────────────────────────────────────
# The figure
# ──────────────────────────────────────────────────────────────────────────────


def vote_boundary_fig() -> None:
    """One vote, drawn in the space the items live in.

    Six pages: the corpus; the votes so far and the boundary they imply; the
    items the detector is already sure about, which is why a ranked list is the
    wrong thing to ask the user about; the item it is *not* sure about, which is
    the one the user actually gets; that item answered and the boundary redrawn;
    and the next question, which exists only because the boundary moved.
    """
    final = _vote_boundary_stage(VOTE_BOUNDARY_STAGES)
    box = tight_box(final)
    for stage in range(1, VOTE_BOUNDARY_STAGES):
        save(_vote_boundary_stage(stage), OUT, f"vote-boundary.build{stage}.png", column=SIDEBAR_WIDE, box=box)
    save(final, OUT, "vote-boundary.png", column=SIDEBAR_WIDE, box=box)


def _vote_boundary_stage(stage: int) -> plt.Figure:
    """Draw the first *stage* steps (1-based, cumulative)."""
    pts = _field()
    truth = _truth(pts)
    assert all(truth[i] for i in SEED_GOOD), "a seeded Good sits outside the concept"
    assert not any(truth[i] for i in SEED_BAD), "a seeded Bad sits inside the concept"

    first = _fit(pts, SEED_GOOD, SEED_BAD)
    labeled = SEED_GOOD + SEED_BAD
    asked = _next_question(pts, first, labeled)
    assert truth[asked], "the item the app asks about is not a match — the Good branch would be a lie"

    second = _fit(pts, SEED_GOOD + (asked,), SEED_BAD)
    asked_again = _next_question(pts, second, labeled + (asked,))

    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.set_aspect("equal")
    ax.set_axis_off()

    # ── stage 1: the corpus — one hollow circle per item, none of them known ──
    voted = {
        **({i: "good" for i in SEED_GOOD} if stage >= 2 else {}),
        **({i: "bad" for i in SEED_BAD} if stage >= 2 else {}),
        **({asked: "good"} if stage >= 5 else {}),
    }
    # The solid disc is the item currently in front of the user: filled at the
    # moment it is asked about, and gone the moment it is answered.
    filled = {asked} if stage == 4 else ({asked_again} if stage >= 6 else set())
    for i, p in enumerate(pts):
        mark = voted.get(i)
        if mark == "good":
            _check(ax, p)
        elif mark == "bad":
            _cross(ax, p)
        else:
            _circle(ax, p, filled=i in filled)

    # ── stage 2: the votes so far, and the detector they imply ───────────────
    if 2 <= stage <= 5:
        _boundary(ax, first, ghost=stage == 5)

    # ── stage 3: the items it is already sure about — the wrong ones to ask ──
    if stage == 3:
        for i in _obvious(pts, first, labeled):
            _halo(ax, pts[i])

    # ── stage 4: nothing but the solid disc above — the one it cannot guess ──
    # ── stage 5: the answer, and the boundary the retrain draws instead ──────
    if stage >= 5:
        _boundary(ax, second)

    # ── stage 6: which puts a different item on the new line. Repeat. ────────
    return fig


if __name__ == "__main__":
    vote_boundary_fig()
    print("wrote figures to", OUT)
