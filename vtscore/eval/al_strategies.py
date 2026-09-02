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
3. **Hard / New** — with the quorum reached, the tool *refines the boundary*
   (Hard: the item nearest the decision cutoff, i.e. what the detector is least
   sure about) until the detector reads smart and stable, then switches to
   *exploring diversity* (New: the coverage atlas's next under-explored
   region).

Phases are supplied per step by :class:`vtscore.eval.autopilot_flow.AutopilotFlow`,
which ports the app's own phase machine.  Passing ``ALContext.phase`` selects
that faithful flow; leaving it ``None`` keeps the older approximation (train
from the first vote pair, interleave Hard/New on step parity) so previously
published studies remain reproducible.  ``docs/EVAL.md`` documents the
difference and why it matters.

An :class:`ALContext` bundles the per-step state; the selector is a callable
``ALContext -> int`` returning the chosen pool id.  :func:`select_next` dispatches
by name and :data:`STRATEGIES` is the (single-entry) registry, both kept so the
harness interface is unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from vtscore.eval.startup_schedule import is_startup_phase

if TYPE_CHECKING:
    from vtscore.coverage.atlas import CoverageAtlas


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
        model: The current trained ranker (``None`` before the first trainable
            step, i.e. before one Good and one Bad vote coexist).  Typed
            loosely because the voting simulation is trainer-pluggable (MLP or
            SVM); the selector only checks whether it ``is not None`` — the
            actual per-item scores come from :attr:`scores`, never from calling
            the model here.
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
            follow the text ranking (top for goods, the cutoff for bads);
            ``None`` means the dataset has no text sort, so the flow seeds from
            random known-good examples instead.
        phase: The app's Autopilot phase for this step, from
            :class:`vtscore.eval.autopilot_flow.AutopilotFlow`.  When set, the
            selector reproduces the app's per-phase Sort + Select pairing
            exactly — including *not* consulting the detector before the first
            learned sort.  ``None`` selects the legacy behaviour (train from
            the first vote pair, parity-interleaved Hard/New), kept so
            published studies stay reproducible; see ``docs/EVAL.md``.
        startup_cut: The cut a parameterised opening's current round samples
            against (issue #3267), from
            :func:`vtscore.eval.startup_schedule.round_cut`.  Read only while
            :attr:`phase` names a schedule round (``s0``, ``s1``, ...);
            ``None`` everywhere else, including on every default-arm run.
    """

    pool_ids: list[int]
    embeddings: dict[int, np.ndarray]
    labeled: dict[int, float]
    scores: dict[int, float]
    model: Optional[Any]
    threshold: float
    atlas: Optional["CoverageAtlas"]
    rng: np.random.RandomState = field(default_factory=lambda: np.random.RandomState(0))
    pool_labels: Optional[dict[int, float]] = None
    seed_scores: Optional[dict[int, float]] = None
    phase: Optional[str] = None
    startup_cut: Optional[float] = None


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


def _centroid_similarities(ctx: ALContext, ids: list[int]) -> Optional[dict[int, float]]:
    """Cosine similarity of each id to the good centroid, or ``None``.

    The example sort the app falls back to when the user supplied media
    examples instead of a text query.
    """
    centroid = _good_centroid(ctx)
    if centroid is None:
        return None
    cnorm = float(np.linalg.norm(centroid))

    def sim(i: int) -> float:
        v = np.asarray(ctx.embeddings[i], dtype=np.float32)
        n = float(np.linalg.norm(v)) * cnorm
        return float(np.dot(v, centroid) / n) if n > _EPS else 0.0

    return {i: sim(i) for i in ids}


def _hard_pick_by_index(ctx: ALContext, ranking: dict[int, float], threshold: float) -> Optional[int]:
    """The app's ``hard`` select: the unlabeled item nearest the cutoff *by rank*.

    Mirrors the app's ``autoSelectNext`` pick rule
    (``frontend/src/app/utils/auto-select-next.ts``, pinned by the
    ``autopilot.auto_select_next`` mirror in
    ``scripts/check-eval-app-sync.py``): rank every item in the current sort
    descending, find the first position whose score is at or below *threshold*,
    and take the unlabeled item whose **index** is closest to it.

    Distance is measured in rank space, not score space, and that is deliberate
    in the app — "avoids biasing toward one side when scores cluster unevenly".
    A score-space ``argmin |p - t|`` picks from whichever side happens to be
    denser, which on a saturated cold-start model is the whole point of the
    difference.  The ranking spans labeled and unlabeled items alike (the app
    ranks the full sort and merely skips voted rows), so the index of the
    cutoff does not shift as votes accumulate.

    Returns ``None`` when no unlabeled item is rankable, so callers can fall
    back.
    """
    ordered = sorted(ranking, key=lambda i: ranking[i], reverse=True)
    threshold_index = len(ordered)
    for idx, cid in enumerate(ordered):
        if ranking[cid] <= threshold:
            threshold_index = idx
            break

    pool = set(ctx.pool_ids)
    best: Optional[int] = None
    best_dist = float("inf")
    for idx, cid in enumerate(ordered):
        if cid not in pool:
            continue
        dist = abs(idx - threshold_index)
        if dist < best_dist:
            best_dist = dist
            best = cid
    return best


def _sort_threshold(scores: dict[int, float]) -> float:
    """The cutoff a text / example sort would show for *scores*.

    Every cosine sort in the app draws its line with
    :func:`~vtscore.training.thresholds.calculate_gmm_threshold` over the full
    score distribution, and the ``hard`` select measures against that line — so
    the Bad phase's pick lands in the *middle* of the text ranking, not at its
    bottom.
    """
    from vtscore.training.thresholds import calculate_gmm_threshold  # noqa: PLC0415

    return calculate_gmm_threshold(list(scores.values()))


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


def _pick_good_phase(ctx: ALContext) -> int:
    """Good phase: the top of the text sort, or a known-good example by hand.

    Sort ``text``/``load`` + Select ``top`` — the user typing a query and
    voting down the ranking, or handing the tool a few examples it already
    knows are good.
    """
    pool = ctx.pool_ids
    if ctx.seed_scores is not None:
        seed = ctx.seed_scores
        return max(pool, key=lambda i: seed.get(i, -np.inf))
    positives = [i for i in pool if ctx.pool_labels is not None and ctx.pool_labels.get(i) == 1.0]
    return _uniform_pick(ctx, positives or pool)


def _pick_on_seed_sort(ctx: ALContext, cut: Optional[float]) -> int:
    """Rank-space ``hard`` select on the text/example sort, against *cut*.

    The one operation both pre-detector phases are made of: rank the seed sort,
    find *cut*'s rank position, take the nearest unlabelled item.  ``None``
    means the sort's own GMM line - the app's cutoff for every cosine sort.
    """
    ranking = ctx.seed_scores
    if ranking is None:
        ranking = _centroid_similarities(ctx, list(ctx.embeddings))
    if ranking:
        line = _sort_threshold(ranking) if cut is None else cut
        pick = _hard_pick_by_index(ctx, ranking, line)
        if pick is not None:
            return pick
    return _uniform_pick(ctx, ctx.pool_ids)


def _pick_bad_phase(ctx: ALContext) -> int:
    """Bad phase: the cutoff of the text/example sort — never the detector.

    Sort ``text``/``load`` + Select ``hard``.  The app has not run a learned
    sort yet at this point, so the ranking here is the *query* similarity and
    the cutoff is that sort's GMM line: the pick lands in the middle of the
    text ranking, where the ambiguous items are, rather than at the bottom.
    Finding negatives is easy; the ones worth spending a vote on are the ones
    the query ranking cannot separate.
    """
    return _pick_on_seed_sort(ctx, None)


def _pick_startup_round(ctx: ALContext) -> int:
    """A parameterised opening's round (issue #3267): the same seed-sort
    ``hard`` select, against the cut its round names.

    ``top`` arrives here as ``+inf`` and reproduces the Good phase's ``top``
    select exactly; ``mid`` arrives as the sort's GMM line and reproduces the
    Bad phase.  Everything between and beyond is the study's territory.
    """
    return _pick_on_seed_sort(ctx, ctx.startup_cut)


def _select_phase_faithful(ctx: ALContext, phase: str) -> int:
    r"""Pick the next pool id using the app's Sort + Select pairing for *phase*.

    Each branch below is one row of the mapping ``LabelViewComponent`` applies
    when the Autopilot phase changes (``restoreAutopilotSortSelect`` /
    the phase subscription):

    ==========  =====================  =============
    phase       Sort                   Select
    ==========  =====================  =============
    ``good``    text / example         ``top``
    ``bad``     text / example         ``hard``
    ``hard``    learned                ``hard``
    ``new``     learned                ``new``
    ``s``\ *i*   text / example         ``hard`` @ the round's cut
    ==========  =====================  =============

    The ``s``\ *i* row is a parameterised opening's round (issue #3267), which
    exists only when the harness was given a
    :class:`~vtscore.eval.startup_schedule.StartupState`.  It generalises the
    two rows above it: ``good`` is that select against a cut above every score,
    ``bad`` against the sort's own GMM line.

    The load-bearing detail is that ``bad`` is still on the *text* sort: no
    detector is trained until the quorum is met, so the harness must not
    consult ``ctx.scores`` here even when a model exists for measurement.
    """
    pool = ctx.pool_ids

    if is_startup_phase(phase):
        return _pick_startup_round(ctx)
    if phase == "good":
        return _pick_good_phase(ctx)
    if phase == "bad":
        return _pick_bad_phase(ctx)

    # --- new: explore the atlas; fall through to hard when exhausted. ---
    if phase == "new":
        pick = _atlas_next(ctx)
        if pick is not None:
            return pick

    # --- hard (and the new-phase fallback): refine the boundary. ---
    if ctx.model is not None and ctx.scores:
        pick = _hard_pick_by_index(ctx, ctx.scores, ctx.threshold)
        if pick is not None:
            return pick
    return _uniform_pick(ctx, pool)


def _select_autopilot(ctx: ALContext) -> int:
    """Pick the next pool id by reproducing the app's Autopilot flow.

    With ``ctx.phase`` set, dispatches to :func:`_select_phase_faithful`, which
    follows the app's phase machine.  Without it, the legacy approximation
    below runs: it trains from the first ``(good, bad)`` pair and interleaves
    Hard/New on step parity.  See ``docs/EVAL.md``.
    """
    if ctx.phase is not None:
        return _select_phase_faithful(ctx, ctx.phase)

    n_good = sum(1 for v in ctx.labeled.values() if v == 1.0)
    n_bad = len(ctx.labeled) - n_good
    pool = ctx.pool_ids

    # --- Seed / Good phase: gather the initial positives. ---
    if n_good < _GOOD_TARGET:
        if ctx.seed_scores is not None:
            # Text sort available: vote top-down on the query ranking, where the
            # positives cluster — exactly what a user does after typing a query.
            seed = ctx.seed_scores
            return max(pool, key=lambda i: seed.get(i, -np.inf))
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
            seed = ctx.seed_scores
            return min(pool, key=lambda i: seed.get(i, np.inf))
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
        raise KeyError(f"Unknown strategy {strategy!r}; available: {', '.join(available_strategies())}") from None
    if not ctx.pool_ids:
        raise ValueError("select_next called with an empty pool")
    return selector(ctx)


def available_strategies() -> list[str]:
    """Return the sorted list of registered strategy names (just ``autopilot``)."""
    return sorted(STRATEGIES)
