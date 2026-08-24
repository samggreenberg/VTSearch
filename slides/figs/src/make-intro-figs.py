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

**Where the items sit is the one thing the figure chooses.** Which items are
matches, and where the curve goes, follow from the data and the fit; the 2D
coordinates are arbitrary, so `_scene` spends them on making the drawing say
what is true. It settles the field until no boundary passes *through* an item —
a curve crossing a circle draws an item the detector cut in half, where the
claim is that the item sits near the line — and it places the two items the
figure singles out just outside the curve they are nearest, with everything
else given half again as much room, so that "this is the one it cannot call" is
readable and not merely true. Outside matters twice over: an item that is
already inside the boundary is one the model already calls a match, so
answering it Good would teach it nothing and the retrained curve would not
move. Assertions cover every claim that survives the settling.

**On the build rule.** `slides/STYLE.md` says a reveal adds ink and nothing
moves or restyles between pages. This figure breaks that in exactly two places,
both because the restyle *is* the mechanism: the item under consideration turns
from a hollow circle into a solid disc and then into a check or a cross, and
the boundary moves once the retrain has happened (the previous boundary stays
on the slide, faded, so what the audience sees is where it went, not a curve
teleporting). Everything else — the window, the crop, every item's position —
is pinned across all six pages.
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

#: Where the item the app is asking about sits, relative to the boundary, in
#: figure units. A curve that passes *through* a circle draws an item the
#: detector has cut in half; what the slide claims is that the item sits
#: **near** the line, which is a different picture and one the audience reads
#: off the gap. Wide enough to be a gap at slide size, narrow enough that "that
#: one is right on the line" is still the obvious reading.
CURVE_CLEAR = R + 0.15

#: How much room every *other* item gets, in figure units. Strictly more than
#: `CURVE_CLEAR`, because "this is the one it cannot call" is a claim about the
#: item being nearest the line, and a field where everything sits at the same
#: distance makes that claim unreadable however true it is.
CURVE_ROOM = R + 0.34

#: Minimum centre-to-centre distance between any two items, in figure units.
#: The field is drawn with more room than this; it is here because pushing
#: items off a curve can push two of them together.
ITEM_APART = 0.78

#: How far past the minimum a nudge lands, in figure units. Settling *onto* the
#: limit never terminates: the curve is sampled, so the measured distance
#: wobbles either side of the target and every pass finds the same item a
#: hair short of it again.
OVERSHOOT = 0.03

#: The movement below which the field counts as settled, in figure units —
#: about two rendered pixels in the deck, so nothing that survives it is
#: visible. A pinned item never stops moving entirely: pinning it moves the fit
#: it is a training point for, which moves the curve it is pinned to.
SETTLED = 0.006

#: How much of a pin's correction is applied per pass. Applying all of it makes
#: that same feedback loop ring instead of converging.
PIN_DAMPING = 0.55

#: Spacing of the sampled points that stand in for a drawn boundary, in figure
#: units — well under `SETTLED`, so sampling is never what a distance turns on.
CONTOUR_STEP = 0.01

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


@functools.lru_cache(maxsize=1)
def _truth() -> np.ndarray:
    """Which items are matches — decided once, on the field as first drawn.

    Read off the concept disc, and then *frozen*, because an item either is or
    is not what the user is looking for and where the figure chooses to draw it
    has no say in that. Recomputing it from the settled positions instead makes
    the drawing's one arbitrary choice — where to put a dot — silently relabel
    the data it is a drawing of.
    """
    return np.hypot(*(_field() - TRUE_CENTRE).T) < TRUE_RADIUS


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


def _contour(model: SVC) -> np.ndarray:
    """The detector's boundary, as a dense set of points lying on it.

    Everything the figure needs to know about the curve is *geometric* — which
    items it passes near, which it encloses with room to spare — and none of
    that is legible from the decision function, whose units are the model's
    and not the page's. So the curve is extracted once, at the level matplotlib
    would draw, and measured against in figure units.
    """
    axis = np.linspace(-1.0, CANVAS[0] + 1.0, 300)
    xx, yy = np.meshgrid(axis, axis)
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots()
    try:
        segments = ax.contour(xx, yy, zz, levels=[0.0]).allsegs[0]
    finally:
        plt.close(fig)
    # Resampled fine, because every distance in this figure is measured to
    # these points rather than to the polyline between them: at the grid's own
    # spacing the measurement is short by up to half a segment, which is enough
    # to argue about at the tolerances the settling loop works to.
    dense = []
    for seg in segments:
        step = np.hypot(*np.diff(seg, axis=0).T)
        walked = np.concatenate([[0.0], np.cumsum(step)])
        if walked[-1] <= 0:
            continue
        even = np.arange(0.0, walked[-1], CONTOUR_STEP)
        dense.append(np.column_stack([np.interp(even, walked, seg[:, axis]) for axis in (0, 1)]))
    return np.vstack(dense) if dense else np.empty((0, 2))


def _gap(curve: np.ndarray, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each item's distance to `curve`, and the point on the curve it is near."""
    if not len(curve):
        return np.full(len(pts), np.inf), np.zeros_like(pts)
    d = np.hypot(*(curve[:, None, :] - pts[None, :, :]).transpose(2, 0, 1))
    nearest = np.argmin(d, axis=0)
    return d[nearest, np.arange(len(pts))], curve[nearest]


def _next_question(
    curve: np.ndarray, pts: np.ndarray, labeled: tuple[int, ...], among: np.ndarray | None = None
) -> int:
    """The unlabeled item the app would ask about next: nearest the boundary.

    This is the `Hard` selection rule, which is the one the slide is about —
    the item whose answer the detector cannot currently guess. Measured to the
    drawn curve rather than by |decision value| so that the item the figure
    singles out is the one an audience picking by eye would also point at.

    `among` narrows the candidates, and is how the drawing picks the branch it
    is going to follow: the deck shows the user answering Good, so the item it
    hands them has to be one. That is a choice about which of two true stories
    to tell, not a fudge — the item really does end up nearest the line, because
    the field is then settled around that choice.
    """
    d, _ = _gap(curve, pts)
    d[list(labeled)] = np.inf
    if among is not None:
        keep = np.full(len(pts), np.inf)
        keep[among] = 0.0
        d = d + keep
    return int(np.argmin(d))


def _obvious(curve: np.ndarray, model: SVC, pts: np.ndarray, labeled: tuple[int, ...], n: int = 3) -> list[int]:
    """The unlabeled items sitting furthest *inside* the boundary.

    The foil for the slide's argument: these are the ones a ranked result list
    puts on top, and asking about them is how a session spends twenty votes
    learning nothing it did not already know.
    """
    d, _ = _gap(curve, pts)
    depth = np.where(model.decision_function(pts) > 0, d, -1.0)
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
# Settling the field
# ──────────────────────────────────────────────────────────────────────────────


def _side(model: SVC, p: np.ndarray) -> int:
    """+1 if `p` is inside the model's boundary, -1 if outside."""
    return 1 if model.decision_function([p])[0] > 0 else -1


def _place(model: SVC, near: np.ndarray, p: np.ndarray, distance: float, side: int | None) -> np.ndarray:
    """The point `distance` off the curve at `near`, on the wanted side of it.

    `side=None` keeps whichever side `p` is already on, which is the rule for a
    shove: moving an item across the boundary to tidy up a drawing would change
    it from a match to a non-match, and that is a different figure.
    """
    away = p - near
    length = float(np.hypot(*away))
    # An item sitting exactly on the curve has no side to be pushed to. Give it
    # one arbitrarily; the next pass has a direction to work with.
    unit = away / length if length > 1e-9 else np.array([1.0, 0.0])
    candidate = near + unit * distance
    if side is not None and _side(model, candidate) != side:
        candidate = near - unit * distance
    return candidate


def _relax(
    pts: np.ndarray,
    curves: list[tuple[SVC, np.ndarray]],
    pins: dict[int, tuple[int, int]],
) -> tuple[np.ndarray, int]:
    """Nudge items off the curves, and off each other, once.

    `pins` maps an item to `(which curve, which side)` and is how the two items
    the figure singles out get *placed* rather than merely cleared: they are set
    to `CURVE_CLEAR` on the named side of the named curve on every pass, so the
    item the app asks about ends up demonstrably nearer the line than anything
    else, and on the outside of it — which is what makes answering it Good move
    the boundary far enough to see.
    """
    moved = pts.copy()
    violations = 0
    for k, (model, curve) in enumerate(curves):
        gaps, nearest = _gap(curve, moved)
        for i in range(len(moved)):
            pinned_curve, pinned_side = pins.get(i, (None, None))
            if pinned_curve == k:
                target = _place(model, nearest[i], moved[i], CURVE_CLEAR, pinned_side)
                correction = target - moved[i]
                if float(np.hypot(*correction)) > SETTLED:
                    violations += 1
                moved[i] = moved[i] + correction * PIN_DAMPING
                continue
            # A pinned item owes the *other* curve only the same clearance
            # everything else owes the one it is pinned to. Asking it for the
            # full room makes the two rules fight wherever the two boundaries
            # run close together, and the field never settles.
            room = CURVE_CLEAR if pinned_curve is not None else CURVE_ROOM
            if gaps[i] < room - SETTLED:
                violations += 1
                moved[i] = _place(model, nearest[i], moved[i], room + OVERSHOOT, None)

    # Then the items against each other, because a shove off a curve can shove
    # two of them together.
    for i in range(len(moved)):
        for j in range(i + 1, len(moved)):
            offset = moved[j] - moved[i]
            distance = float(np.hypot(*offset))
            if distance >= ITEM_APART - SETTLED:
                continue
            violations += 1
            step = (offset / max(distance, 1e-9)) * (ITEM_APART + OVERSHOOT - distance) / 2
            moved[i] -= step
            moved[j] += step
    return moved, violations


def _enclosed(model: SVC) -> float:
    """The fraction of the canvas the model currently calls a match."""
    axis = np.linspace(-1.0, CANVAS[0] + 1.0, 200)
    xx, yy = np.meshgrid(axis, axis)
    return float((model.decision_function(np.c_[xx.ravel(), yy.ravel()]) > 0).mean())


def _window(pts: np.ndarray, curves: list[np.ndarray]) -> tuple[float, float, float, float]:
    """A square window holding the field and every curve, with room to spare.

    Derived rather than fixed, because the retrained boundary reaches further
    than the one it replaces — sometimes past where the field ends. Clipped at
    the frame it stops being a closed curve around the matches and becomes a
    line running off the page, which is a picture of something else.
    """
    ink = np.vstack([pts, *[c for c in curves if len(c)]])
    low = ink.min(axis=0) - (R + 0.45)
    high = ink.max(axis=0) + (R + 0.45)
    side = float(max(high - low))
    centre = (low + high) / 2
    return (centre[0] - side / 2, centre[0] + side / 2, centre[1] - side / 2, centre[1] + side / 2)


@functools.lru_cache(maxsize=1)
def _scene() -> tuple[
    np.ndarray, SVC, SVC, np.ndarray, np.ndarray, int, int, tuple[int, ...], tuple[float, float, float, float]
]:
    """The whole drawing, settled: positions, both fits, and every role.

    Settled, because the field as first drawn has the boundary running through
    half a dozen items, and an item a curve passes through reads as one the
    detector has cut in two rather than one it cannot call. Item positions are
    arbitrary — they are the one thing this figure is free to choose — so they
    are moved until nothing is touched, and the fits are redone at every step
    because moving a voted item moves the boundary that was fitted to it.

    Two items are *placed* rather than merely cleared. The item the app asks
    about is pinned just **outside** the first boundary, and the one it asks
    about next just outside the second. Outside matters: settling that item to
    wherever it happened to drift put it comfortably inside the curve, where
    the model already called it a match — so answering it Good taught the model
    nothing and the retrained boundary barely moved, which is the one thing
    this figure exists to show.

    The roles are re-derived from the settled positions in the outer pass, so
    the figure cannot end up singling out an item that was nearest the line
    before everything moved and is no longer.
    """
    pts = _field()
    labeled = SEED_GOOD + SEED_BAD
    matches = np.flatnonzero(_truth())
    first = _fit(pts, SEED_GOOD, SEED_BAD)
    asked = _next_question(_contour(first), pts, labeled, among=matches)
    second = _fit(pts, SEED_GOOD + (asked,), SEED_BAD)
    asked_again = _next_question(_contour(second), pts, labeled + (asked,))

    for _ in range(12):
        pins = {asked: (0, -1), asked_again: (1, -1)}
        for _ in range(120):
            first = _fit(pts, SEED_GOOD, SEED_BAD)
            second = _fit(pts, SEED_GOOD + (asked,), SEED_BAD)
            curve, curve_after = _contour(first), _contour(second)
            pts, violations = _relax(pts, [(first, curve), (second, curve_after)], pins)
            if not violations:
                break
        else:
            raise SystemExit("the field would not settle: items still touch a boundary after 120 passes")
        settled = (
            _next_question(curve, pts, labeled, among=matches),
            _next_question(curve_after, pts, labeled + (asked,)),
        )
        if settled == (asked, asked_again):
            break
        asked, asked_again = settled
    else:
        raise SystemExit("the item the app asks about keeps changing as the field settles")

    truth = _truth()
    assert all(truth[i] for i in SEED_GOOD), "a seeded Good sits outside the concept"
    assert not any(truth[i] for i in SEED_BAD), "a seeded Bad sits inside the concept"
    assert truth[asked], "the item the app asks about is not a match — the Good branch would be a lie"
    assert _side(first, pts[asked]) == -1, "the item the app asks about is already inside the boundary"
    assert _side(second, pts[asked]) == 1, "the Good vote did not bring its own item inside the new boundary"
    # The votes have to read right against the curve they trained: a check mark
    # drawn outside the blue line, or a cross drawn inside it, is a picture of
    # a detector that ignored its own training data.
    for model in (first, second):
        assert all(_side(model, pts[i]) == 1 for i in SEED_GOOD), "a Good is drawn outside the boundary"
        assert all(_side(model, pts[i]) == -1 for i in SEED_BAD), "a Bad is drawn inside the boundary"
    for drawn in (curve, curve_after):
        gaps, _ = _gap(drawn, pts)
        assert gaps.min() >= CURVE_CLEAR - SETTLED, "a boundary still runs through an item"
    gaps, _ = _gap(curve, pts)
    gaps[list(labeled)] = np.inf
    assert int(np.argmin(gaps)) == asked, "some other item ended up nearer the line than the one it asks about"
    growth = _enclosed(second) / _enclosed(first)
    assert growth >= 1.06, f"the retrained boundary barely moved (x{growth:.2f}) — nothing to see"
    window = _window(pts, [curve, curve_after])
    return pts, first, second, curve, curve_after, asked, asked_again, labeled, window


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
    pts, first, second, curve, _curve_after, asked, asked_again, labeled, window = _scene()

    fig, ax = plt.subplots(figsize=tuple(c * UNIT_PT / 72 for c in CANVAS))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_xlim(*window[:2])
    ax.set_ylim(*window[2:])
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
        for i in _obvious(curve, first, pts, labeled):
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
