"""The autopilot vote-order simulation for the voting-iterations harness.

The voting-iterations evaluator (:mod:`vtscore.eval.voting_iterations`) simulates
a user labelling one item at a time and measures how fast the detector's cost
falls.  *Which* item the simulated user is shown next is decided here, by
reproducing what a real VTSearch user actually does — the **Autopilot** flow —
rather than any academic active-learning acquisition rule.  There is exactly one
strategy, ``autopilot``; the eval only cares about how the tool itself would
function.

The flow mirrors the app's Autopilot phases (see
``frontend/src/app/services/autopilot-state.service.ts``):

1. **Seed / Good** — gather the first :data:`_GOOD_TARGET` positives.  If the
   dataset can be text-sorted (the caller supplies ``seed_scores``, a per-item
   similarity to the typed query), the user votes top-down on that ranking, where
   positives cluster.  Otherwise the user hands the tool a few known-good
   examples: :data:`_GOOD_TARGET` random ground-truth positives ("3 random
   examples pulled from the Good").
2. **Bad** — gather the first :data:`_BAD_TARGET` negatives.  Once a detector
   exists (≥1 good and ≥1 bad), its lowest-scored items are the likely bads the
   tool surfaces at the bottom of learned sort; before that, the bottom of the
   text ranking (or, with no text, the item least similar to the good centroid —
   an example sort) is voted bad.
3. **Hard / New** — with the quorum reached, the tool cycles between *refining
   the boundary* (Hard: the item nearest the decision threshold, i.e. what the
   detector is least sure about) and *exploring diversity* (New: the coverage
   atlas's next under-explored region).  The two are interleaved on step parity.

An :class:`ALContext` bundles the per-step state; the selector is a callable
``ALContext -> int`` returning the chosen pool id.  :func:`select_next` dispatches
by name and :data:`STRATEGIES` is the (single-entry) registry, both kept so the
harness interface is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import torch

    from vtscore.state.coverage_atlas import CoverageAtlas


# Initial-phase vote targets, matching the app's Autopilot defaults
# (``INITIAL_STATE.goodToStart`` / ``badToStart``): three positives to teach the
# detector what "good" looks like, then four negatives so it has both sides of
# the cutoff.
_GOOD_TARGET = 3
_BAD_TARGET = 4

_EPS = 1e-9


@dataclass
class ALContext:
    """Per-step state the autopilot selector needs to pick the next item.

    Attributes:
        pool_ids: Candidate ids the selector chooses among (unlabelled so far),
            in a stable order.
        embeddings: ``{id: vector}`` for every id (pool and labelled) — the
            whole-image embedding, used for the example-sort good centroid.
        labeled: ``{id: 1.0 | 0.0}`` for every item labelled so far.  Empty
            before the first vote; its good/bad counts drive the phase.
        scores: ``{pool_id: p}`` sigmoid probabilities from the current detector,
            used by the Bad (lowest-scored) and Hard (nearest-threshold) picks.
            Empty before the first model exists.
        model: The current trained MLP (``None`` before the first trainable
            step, i.e. before one Good and one Bad vote coexist).
        threshold: The current decision threshold ``t`` (Hard measures
            ``|p - t|`` against it).
        atlas: The dataset's coverage atlas, driving the New (diversity) pick
            (``None`` disables it — the interleave then always picks Hard).
        rng: Seeded RNG driving the random good-seed pick, so a run is
            reproducible from its seed.
        pool_labels: Ground-truth ``{pool_id: 1.0 | 0.0}`` for the pool, used to
            draw the random known-good seed examples when no text sort is
            available.  This is not an oracle *ranking* — it stands in for the
            handful of positives a real user supplies by hand to bootstrap.
            ``None`` degrades the good seed to a uniform-random pick.
        seed_scores: Optional ``{id: similarity}`` text-sort ranking (cosine of
            each item to the typed query).  When present the seed/bad phases
            follow the text ranking (top for goods, bottom for bads); ``None``
            means the dataset has no text sort, so the flow seeds from random
            known-good examples instead.
    """

    pool_ids: list[int]
    embeddings: dict[int, np.ndarray]
    labeled: dict[int, float]
    scores: dict[int, float]
    model: Optional["torch.nn.Sequential"]
    threshold: float
    atlas: Optional["CoverageAtlas"]
    rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(0))
    pool_labels: Optional[dict[int, float]] = None
    seed_scores: Optional[dict[int, float]] = None


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _uniform_pick(ctx: ALContext, candidates: list[int]) -> int:
    """Return a uniform-random id from *candidates* using the seeded RNG."""
    util = ctx.rng.random(len(candidates))
    return candidates[int(np.argmax(util))]


def _good_centroid(ctx: ALContext) -> Optional[np.ndarray]:
    """Return the L2-normalised mean of the good-voted embeddings, or ``None``.

    Mirrors the example-sort centroid the live tool builds from multiple Good
    examples (each vector L2-normalised before averaging, so every example
    contributes equally regardless of its norm).
    """
    goods = [i for i, v in ctx.labeled.items() if v == 1.0]
    if not goods:
        return None
    normed = []
    for i in goods:
        v = np.asarray(ctx.embeddings[i], dtype=np.float32)
        n = float(np.linalg.norm(v))
        normed.append(v / n if n > _EPS else v)
    return np.mean(np.stack(normed), axis=0)


def _least_similar_to_goods(ctx: ALContext) -> int:
    """Return the pool item least similar to the good centroid (an example sort).

    The bottom of an example sort by the current positives is the most-likely
    bad, exactly what the tool would surface for a Bad vote before any detector
    exists.  Falls back to a uniform pick when no goods are labelled yet.
    """
    centroid = _good_centroid(ctx)
    if centroid is None:
        return _uniform_pick(ctx, ctx.pool_ids)
    cnorm = float(np.linalg.norm(centroid))

    def sim(i: int) -> float:
        v = np.asarray(ctx.embeddings[i], dtype=np.float32)
        n = float(np.linalg.norm(v)) * cnorm
        return float(np.dot(v, centroid) / n) if n > _EPS else 0.0

    return min(ctx.pool_ids, key=sim)


def _atlas_next(ctx: ALContext) -> Optional[int]:
    """Return the coverage atlas's next under-explored pool item, or ``None``.

    Delegates to :meth:`CoverageAtlas.next_sample` (the same call the app's New
    phase makes), passing the current pool scores + threshold so the probe lands
    on a representative surprise in the first evidence-free region.  Returns
    ``None`` when the atlas is exhausted or the pick has somehow left the pool,
    so the caller can fall back to a Hard pick.
    """
    atlas = ctx.atlas
    if atlas is None:
        return None
    pick = atlas.next_sample(ctx.scores or None, ctx.threshold)
    if pick is None or pick not in set(ctx.pool_ids):
        return None
    return pick


# ------------------------------------------------------------------
# The autopilot selector
# ------------------------------------------------------------------


def _select_autopilot(ctx: ALContext) -> int:
    """Pick the next pool id by reproducing the app's Autopilot flow."""
    n_good = sum(1 for v in ctx.labeled.values() if v == 1.0)
    n_bad = len(ctx.labeled) - n_good
    pool = ctx.pool_ids

    # --- Seed / Good phase: gather the initial positives. ---
    if n_good < _GOOD_TARGET:
        if ctx.seed_scores is not None:
            # Text sort available: vote top-down on the query ranking, where the
            # positives cluster — exactly what a user does after typing a query.
            return max(pool, key=lambda i: ctx.seed_scores.get(i, -np.inf))
        # No text sort: seed with random known-good examples the user supplies
        # by hand ("3 random examples pulled from the Good").
        positives = [i for i in pool if ctx.pool_labels is not None and ctx.pool_labels.get(i) == 1.0]
        return _uniform_pick(ctx, positives or pool)

    # --- Bad phase: gather the initial negatives. ---
    if n_bad < _BAD_TARGET:
        if ctx.model is not None and ctx.scores:
            # A detector exists: its lowest-scored items are the likely bads the
            # tool surfaces at the bottom of learned sort.
            return min(pool, key=lambda i: ctx.scores.get(i, 0.5))
        if ctx.seed_scores is not None:
            # No detector yet, but text sort exists: the bottom of the ranking
            # (least similar to the query) is the most-likely bad.
            return min(pool, key=lambda i: ctx.seed_scores.get(i, np.inf))
        # No detector, no text: example-sort by the good centroid, vote the
        # least-similar item bad.
        return _least_similar_to_goods(ctx)

    # --- Hard / New interleave: quorum reached.  Cycle refine-boundary and
    # explore-diversity the way Autopilot does past the initial phases. ---
    total = n_good + n_bad
    if total % 2 == 1:
        pick = _atlas_next(ctx)
        if pick is not None:
            return pick
    if ctx.model is not None and ctx.scores:
        # Hard = margin: the item nearest the decision threshold, i.e. the one
        # the detector is least sure about.
        return min(pool, key=lambda i: abs(ctx.scores.get(i, 0.5) - ctx.threshold))
    # No detector to rank uncertainty by yet (only reachable on a degenerate
    # pool): fall back to a uniform pick.
    return _uniform_pick(ctx, pool)


#: Registry of vote-order strategies.  The eval simulates only the real user
#: flow, so ``autopilot`` is the sole entry.
STRATEGIES: dict[str, Callable[[ALContext], int]] = {"autopilot": _select_autopilot}


def select_next(strategy: str, ctx: ALContext) -> int:
    """Pick the next pool id to query using the named *strategy*.

    Raises ``KeyError`` for an unknown name so a typo fails loudly instead of
    silently defaulting.
    """
    try:
        selector = STRATEGIES[strategy]
    except KeyError:
        raise KeyError(
            f"Unknown strategy {strategy!r}; available: {', '.join(available_strategies())}"
        ) from None
    if not ctx.pool_ids:
        raise ValueError("select_next called with an empty pool")
    return selector(ctx)


def available_strategies() -> list[str]:
    """Return the sorted list of registered strategy names (just ``autopilot``)."""
    return sorted(STRATEGIES)
