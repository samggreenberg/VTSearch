"""Fit a face space's ball from a labelled face set its provider embedded (#5497).

The face crops' ball (#5478) keeps every face whose cosine to the Goods' centroid clears a floor
per balance, moved by how spread out the Goods sit.  The floors are cosines in one embedder's
space, so each space is fitted from its own faces: a labelled set (FHIBE is the practice set)
embedded by the server's face provider, whose people are the categories.  This is #5474's dump
and #5478's ``fit`` (``scripts/experiments/fhibe/face_balls_5474.py``, ``face_ball_line_5478.py``)
as one step that needs no eval run:

1. **Cells.**  For each person and seed, the review's split (:func:`~vtscore.eval.example_opening.stratified_split`
   on ``RandomState(seed)``) and its opening draw (:func:`~vtscore.eval.example_opening.choose_examples`);
   for K = 2, 3, 4 Goods and every Good of the voting half, the withheld half's cosines to the Goods'
   centroid, its labels, and the Goods' spread (:func:`~vtscore.detectors.centroid_head.goods_spread`).
2. **Floors.**  Per balance, the floor and slope (on a grid) with the best mean F-beta over every cell,
   the spread centred on its median (#5474's ``adaptive``).  Fitted on all the people for the entry,
   and on each half of them (two folds by person) for the check.
3. **Check (go/no-go).**  Each fold's floors scored on the other fold's people against the face crops'
   line from before the ball (:data:`~vtscore.detectors.centroid_head.FACE_BALL_BELOW`) on the same
   cosines, and the share of people whose own photos all sit above every other face at K = 4 (a
   clean gap).  The ball is used in the space only when it is level with that line or better at
   every balance (:data:`LEVEL`) and enough people have a clean gap (:data:`MIN_CLEAN_GAP`).

   The check is static, on the cosines alone: it prices the ball against the line a centroid is
   cut by under the quota, where a session past the quota would have the trained head.  On FHIBE
   it reads buffalo_s level at beta 1 (-0.003) and +0.055 at beta 4, where the session replay
   gained +0.010 and +0.016 (#5478); buffalo_l +0.03 and +0.09; FaceNet -0.07 and -0.02 at every K
   and crop size, where its sessions lost 0.08 and 0.04 (#5486).

The output is a :class:`~vtscore.media.face.face_ball.FaceBall` with its check and provenance;
:func:`~vtscore.media.face.face_ball.write_entry` puts it in ``face_ball.json``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

#: The balances the ball is fitted at: it is used from beta 1 up, read off a line in log beta between these.
FIT_BETAS = (1.0, 4.0)
#: Goods per cell: two, three, four, and every Good of the voting half (0).  One Good has no spread.
FIT_KS = (2, 3, 4, 0)
#: The K a clean gap is read at: the review's four starting photos.
GAP_K = 4
#: The floors a fit tries, and the slopes (#5474's grid).
GRID = np.round(np.arange(0.0, 0.9001, 0.0025), 4)
SLOPES = (0.0, 0.25, 0.5, 0.75, 1.0)
_STEP = float(GRID[1] - GRID[0])
#: How far below the pre-ball line the cross-fitted ball may score, at any balance, and still count as level: the
#: size of buffalo_s's static beta-1 reading (-0.003, per K -0.009 to +0.002) against its session gain (+0.010).
LEVEL = 0.01
#: The share of people with a clean gap at K = 4 below which the ball is not used, whatever it gains.
#: buffalo_s has 86%, buffalo_l 95%, FaceNet 56% (#5486), where FaceNet's own floors lose 0.04 to
#: 0.08 F over a session: its people are not balls.
MIN_CLEAN_GAP = 0.75
#: The review's voting share of each person's photos (``CALIB_SIM_FRACTION``'s default).
SIM_FRACTION = 0.5


def fold_of(person: str) -> int:
    """#5474's fold of a person: half the people fit the floors the other half is scored by."""
    return int(hashlib.sha256(person.encode()).hexdigest(), 16) % 2


@dataclass
class Cell:
    """One person, seed and K: the withheld half's cosines to the Goods' centroid, and what the fit reads."""

    corpus: str
    person: str
    seed: int
    k: int
    spread: float
    scores: np.ndarray
    labels: np.ndarray
    kept_g: np.ndarray
    tp_g: np.ndarray
    separable: bool

    @property
    def n_pos(self) -> int:
        return int(self.labels.sum())

    def f_grid(self, beta: float) -> np.ndarray:
        return _fbeta(self.tp_g, self.kept_g, self.n_pos, beta)


def _fbeta(tp: Any, kept: Any, npos: int, beta: float) -> Any:
    p = np.where(kept > 0, tp / np.maximum(kept, 1), 0.0)
    r = tp / npos
    b2 = beta * beta
    return np.where((p > 0) | (r > 0), (1 + b2) * p * r / np.maximum(b2 * p + r, 1e-12), 0.0)


def _f_at(cell: Cell, floor: float, beta: float) -> float:
    """F-beta of keeping every withheld face at or above *floor*."""
    keep = cell.scores >= floor
    return float(_fbeta(np.float64(cell.labels[keep].sum()), np.float64(keep.sum()), cell.n_pos, beta))


def _cell(corpus: str, person: str, seed: int, k: int, spread: float, s: np.ndarray, y: np.ndarray) -> Cell:
    s = s.astype(np.float64)
    order = np.argsort(-s, kind="stable")
    ss, ys = s[order], y[order].astype(np.int64)
    kept_g = np.searchsorted(-ss, -GRID, side="right")
    tp_g = np.concatenate([[0], np.cumsum(ys)])[kept_g]
    return Cell(corpus, person, seed, k, spread, s, y, kept_g, tp_g, bool(s[y == 1].min() > s[y == 0].max()))


def build_cells(
    corpora: Mapping[str, Mapping[Any, dict[str, Any]]],
    people: Sequence[str],
    seeds: Sequence[int] = (0,),
    *,
    sim_fraction: float = SIM_FRACTION,
    ks: Sequence[int] = FIT_KS,
) -> list[Cell]:
    """Every cell of *people* in each corpus of *corpora* (``{name: {id: media}}``), at each seed and K.

    The split and the Goods' order are the review's (#5474's ``dump``), so a fit on the review's people
    and seeds is the fit #5478 shipped.  A person with no Good in the voting half, or none withheld, has
    no cell at that seed; K above the person's Goods has none either.
    """
    from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415
    from vtscore.eval.example_opening import choose_examples, stratified_split  # noqa: PLC0415
    from vtscore.eval.labels import evaluable_pool, media_is_positive  # noqa: PLC0415

    cells: list[Cell] = []
    for corpus, given in corpora.items():
        medias: dict[Any, dict[str, Any]] = dict(given)
        ids = list(medias)
        row = {cid: i for i, cid in enumerate(ids)}
        u = np.stack([np.asarray(media_embedding(medias[c]), dtype=np.float64) for c in ids])
        u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-12)
        for person in people:
            pool = evaluable_pool(medias, person)
            for seed in seeds:
                rng = np.random.RandomState(seed)
                sim_ids, test_ids = stratified_split(pool, person, sim_fraction, rng)
                n_vote_pos = sum(1 for c in sim_ids if media_is_positive(pool[c], person))
                if n_vote_pos == 0:
                    continue
                order = choose_examples(sim_ids, pool, person, n_vote_pos, seed)
                t = np.asarray([row[c] for c in test_ids])
                y = np.asarray([1 if media_is_positive(pool[c], person) else 0 for c in test_ids], dtype=np.uint8)
                if not y.any():
                    continue
                for k in ks:
                    if k > n_vote_pos:
                        continue
                    ex = order if k == 0 else order[:k]
                    if len(ex) < 2:
                        continue
                    g = u[[row[c] for c in ex]]
                    q = g.mean(axis=0)
                    q /= max(np.linalg.norm(q), 1e-12)
                    rest = (g.sum(axis=0)[None, :] - g) / (len(ex) - 1)
                    rest /= np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
                    spread = float(np.mean(np.sum(g * rest, axis=1)))
                    # float32, as #5474's dumps stored them: the shipped floors were fitted on these
                    scores = (u[t] @ q).astype(np.float32)
                    cells.append(_cell(corpus, person, seed, k, spread, scores, y))
    # #5474's file order (corpus, then K, then person and seed), so every mean sums in the order it did
    cells.sort(key=lambda c: (c.corpus, "all" if c.k == 0 else str(c.k)))
    return cells


def fit_floor(cells: Sequence[Cell], beta: float) -> tuple[float, float, float]:
    """``(floor, slope, centre)`` with the best mean F-beta over *cells* (#5474's ``fit_adaptive``).

    Each cell's floor is ``floor + slope * (spread - centre)``, on the grid; *centre* is the median spread.
    """
    centre = float(np.median([c.spread for c in cells]))
    f = np.stack([c.f_grid(beta) for c in cells])
    best = (-1.0, 0.0, 0.0)
    for b in SLOPES:
        shift = np.rint(np.asarray([b * (c.spread - centre) for c in cells]) / _STEP).astype(int)
        idx = np.clip(np.arange(GRID.size)[None, :] + shift[:, None], 0, GRID.size - 1)
        m = np.take_along_axis(f, idx, axis=1).mean(axis=0)
        j = int(np.argmax(m))
        if m[j] > best[0]:
            best = (float(m[j]), float(GRID[j]), b)
    return best[1], best[2], centre


def check(cells: Sequence[Cell], betas: Sequence[float] = FIT_BETAS) -> dict[str, Any]:
    """The go/no-go check: each fold's ball scored on the other fold's people against the pre-ball line.

    ``gain[beta]`` is the mean over cells of the ball's F-beta less the line's, ``gain_se`` its standard
    error, ``clean_gap`` the share of K = 4 cells whose own photos all clear every other face, and
    ``use`` whether the ball goes: level or better at every balance (no more than :data:`LEVEL` below the
    line) and a clean gap at :data:`MIN_CLEAN_GAP` or more.
    """
    from vtscore.detectors.centroid_head import FACE_BALL_BELOW, centroid_cut  # noqa: PLC0415

    gain: dict[str, float] = {}
    gain_se: dict[str, float] = {}
    ball_f: dict[str, float] = {}
    line_f: dict[str, float] = {}
    for beta in betas:
        fits = {f: fit_floor([c for c in cells if fold_of(c.person) != f], beta) for f in (0, 1)}
        d, fb, fl = [], [], []
        for c in cells:
            floor, slope, centre = fits[fold_of(c.person)]
            b = _f_at(c, floor + slope * (c.spread - centre), beta)
            line = _f_at(c, centroid_cut(c.scores, rule=FACE_BALL_BELOW, beta=beta), beta)
            d.append(b - line)
            fb.append(b)
            fl.append(line)
        arr = np.asarray(d)
        key = f"{beta:g}"
        gain[key] = round(float(arr.mean()), 4)
        gain_se[key] = round(float(arr.std(ddof=1) / np.sqrt(arr.size)), 4) if arr.size > 1 else 0.0
        ball_f[key] = round(float(np.mean(fb)), 4)
        line_f[key] = round(float(np.mean(fl)), 4)
    gap = [c.separable for c in cells if c.k == GAP_K]
    clean = round(float(np.mean(gap)), 4) if gap else 0.0
    use = all(g >= -LEVEL for g in gain.values()) and clean >= MIN_CLEAN_GAP
    return {
        "use": use,
        "rule": (
            f"the ball is used when its cross-fitted F-beta is no more than {LEVEL:g} below {FACE_BALL_BELOW} "
            f"at every balance and at least {MIN_CLEAN_GAP:g} of K={GAP_K} cells have a clean gap"
        ),
        "gain": gain,
        "gain_se": gain_se,
        "ball_f": ball_f,
        "line_f": line_f,
        "clean_gap": clean,
        "cells": len(cells),
        "gap_cells": len(gap),
    }


def fit_face_ball(
    compat: str,
    corpora: Mapping[str, Mapping[Any, dict[str, Any]]],
    people: Sequence[str],
    seeds: Sequence[int] = (0,),
    *,
    sim_fraction: float = SIM_FRACTION,
    provenance: Mapping[str, Any] | None = None,
):
    """*compat*'s :class:`~vtscore.media.face.face_ball.FaceBall`, fitted on *people* in *corpora*, checked.

    The floors are fitted on every person; the check (:func:`check`) on each half scored on the other.
    *provenance* is merged into the entry's (corpora, people, seeds and sim fraction are always there).
    """
    from vtscore.media.face.face_ball import FaceBall  # noqa: PLC0415

    cells = build_cells(corpora, people, seeds, sim_fraction=sim_fraction)
    if not cells:
        raise ValueError("no cells: no person has two Goods in the voting half and a withheld photo")
    betas: dict[float, tuple[float, float]] = {}
    centres = set()
    for beta in FIT_BETAS:
        floor, slope, centre = fit_floor(cells, beta)
        betas[beta] = (floor, slope)
        centres.add(round(centre, 4))
    (centre,) = centres  # one median spread: every balance is fitted on the same cells
    verdict = check(cells)
    return FaceBall(
        compat=compat,
        use=bool(verdict["use"]),
        betas=betas,
        centre=centre,
        check=verdict,
        provenance={
            "corpora": {name: len(m) for name, m in corpora.items()},
            "people": len(people),
            "people_with_cells": len({c.person for c in cells}),
            "seeds": list(seeds),
            "sim_fraction": sim_fraction,
            **dict(provenance or {}),
        },
    )
