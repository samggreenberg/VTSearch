"""Conformal cross-calibration: held-out quantiles, pooled cuts, the cached folds.

The estimator the fold-anchored path replaced and still falls back to.  The
labelled set is split into Train/Calibrate folds with a head trained per fold
(:mod:`~vtscore.training.thresholds.fold_orderings`, which
:func:`calibration_folds` drives and :func:`calibration_folds_cached`
memoizes), and a split-conformal quantile is read off the pooled held-out
scores (:func:`conformal_threshold`,
:func:`calculate_cross_calibration_threshold`).  It is independent of
:mod:`vtscore.training.thresholds.anchored`: the two are rival estimators, not
layers.

The fold module's public names are re-exported here, so
``thresholds.conformal.compute_fold_orderings`` and its siblings still import.
A stub for :func:`compute_fold_orderings` placed here reaches this module's
callers of it; a stub for anything the fold code reads itself (its clock, the
split sizing) goes on ``fold_orderings``, where that code looks it up.
"""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple

import numpy as np

from vtscore.training.thresholds.fold_orderings import (
    CALIBRATION_SPLIT_SEED,
    compute_fold_orderings,
)
from vtscore.training.thresholds.fold_orderings import (
    compute_grouped_fold_node_scores as compute_grouped_fold_node_scores,
)
from vtscore.training.thresholds.fold_orderings import nearest_holdout as nearest_holdout
from vtscore.training.thresholds.gmm import scored_ordering, snap_cut_to_sample
from vtscore.training.thresholds.knobs import NO_GOOD_THRESHOLD

# False-negative budget of the conformal inclusion rule at inclusion 0; each
# +1 step of inclusion halves it (see :func:`conformal_threshold`).  0.25 means
# the default cutoff may sacrifice at most ~25% of true matches to the
# false-positive guard - a cap, spent only when class overlap forces it.
CONFORMAL_BASE_BUDGET = 0.25

# Positive-score quantile the inclusion = -10 end of the knob walks to: at -10
# only the region scoring above the 75th percentile of held-out positives is
# included - "just the most confident matches".
CONFORMAL_QPOS_MAX = 0.75


def _score_rows_digest(score_rows_by_group: dict | None) -> bytes | None:
    """Digest of the per-bag **inference** row stacks, for the calibration key.

    ``None`` when no override is in play, so a call that pools each bag over
    its training rows keeps a distinct cache key from one that pools over the
    scorer's rows - otherwise a cached ordering computed under the old geometry
    would be served after the wiring changed.
    """
    if score_rows_by_group is None:
        return None
    h = hashlib.blake2b()
    for g in sorted(score_rows_by_group, key=repr):
        h.update(repr(g).encode())
        h.update(np.asarray(score_rows_by_group[g], dtype=np.float32).tobytes())
    return h.digest()


def _calibration_cache_key(
    X_list: list,
    y_list: list[float],
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
) -> tuple:
    """Build a deterministic cache key for the calibration **fold orderings**.

    The orderings (per-fold held-out scores + labels) are a deterministic
    function of these inputs (the split RNG is seeded with
    :data:`CALIBRATION_SPLIT_SEED` whether or not the caller supplies one)
    and are **inclusion-independent** - ``inclusion`` is deliberately *not* in
    the key, so an Inclusion change hits the cache and only re-runs the cheap
    conformal quantile rule.  The key encodes a hash of the raw training vectors (not
    just label IDs) so a labelset re-resolved to different embeddings - e.g.
    after the embedder changes - invalidates the cache automatically.
    """
    # Hash the raw training vectors rather than embedding them in the key.
    # The full ``(N_labels x D x 4)``-byte string reaches ~150 MB at 100k
    # labels and would live in the calibration cache until the next call
    # invalidates it. blake2b (fast, 128-bit digest) keeps the key tiny while
    # still changing whenever the labelset re-resolves to different embeddings
    # (e.g. after the embedder changes); the cache is already invalidated by any
    # vote change, so the hash is purely for collision resistance.
    X_hash = hashlib.blake2b(np.stack(X_list).astype(np.float32, copy=False).tobytes()).digest()
    y_hash = hashlib.blake2b(np.asarray(y_list, dtype=np.float32).tobytes()).digest()
    # Bag membership changes the fold split and per-group max-pool, so a change
    # in grouping must invalidate the cached orderings even when X/y are equal.
    groups_key = tuple(str(g) for g in groups) if groups is not None else None
    # The fold models of a typed-query detector start from its query (#4859): a pass
    # under another query, or none, must not be served this one's folds.
    # Nor may a pass that weighs another set of hard-view Goods (#5413).
    from vtscore.training.svm_prior import active_query_prior_key, hard_view_good_weights  # noqa: PLC0415

    weights = hard_view_good_weights(np.stack(X_list), np.asarray(y_list))

    return (
        X_hash,
        y_hash,
        int(calibrate_count),
        float(calibration_fraction),
        int(hidden_dim),
        groups_key,
        _score_rows_digest(score_rows_by_group),
        active_query_prior_key(),
        None if weights is None else weights.tobytes(),
    )


class CalibrationFolds(NamedTuple):
    """The K calibration folds: held-out orderings, fallback sentinel, models.

    *orderings* are the per-fold ``(scores, labels)`` the conformal rule pools;
    *fallback* is the sentinel threshold to return outright when calibration
    was impossible (``None`` when the folds are real); *models* are the trained
    fold models in fold order, which the fold-anchored threshold
    (:func:`fold_anchored_gmm_threshold`) scores the haystack with so the
    anchors and the population it fits share one scale.

    Which training rows each fold held out is not a field here - adding one
    would break every caller that unpacks the three - but a ``holdout_sink``
    on :func:`calibration_folds` / :func:`calibration_folds_cached` receives it
    (see :func:`compute_fold_orderings`).
    """

    orderings: list[tuple[list[float], list[float]]]
    fallback: float | None
    models: list


def calibration_folds(
    X_list: list,
    y_list: list[float],
    input_dim: int,
    *,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
    rng: "np.random.RandomState | None" = None,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
    holdout_sink: list | None = None,
) -> CalibrationFolds:
    """Train the K calibration folds, keeping their models (uncached).

    Deterministic without a caller-supplied *rng*: the splits then come from a
    fresh ``RandomState(CALIBRATION_SPLIT_SEED)``, matching
    :func:`calibration_folds_cached`, so an uncached call (``det_ctx is None``)
    and a cached one produce the same folds for the same labelset.

    *holdout_sink* receives, per fold, the training row behind each held-out
    score (see :func:`compute_fold_orderings`).
    """
    models: list = []
    orderings, fallback = compute_fold_orderings(
        X_list,
        y_list,
        input_dim,
        rng=rng,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        groups=groups,
        score_rows_by_group=score_rows_by_group,
        model_sink=models,
        holdout_sink=holdout_sink,
    )
    return CalibrationFolds(orderings, fallback, models)


def calibration_folds_cached(
    X_list: list,
    y_list: list[float],
    input_dim: int,
    *,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
    det_ctx: Any = None,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
    holdout_sink: list | None = None,
) -> CalibrationFolds:
    """Memoized :func:`calibration_folds` keyed on the calibration inputs.

    When *det_ctx* is provided, caches the inclusion-independent folds on
    ``det_ctx.calibration_cache`` as ``(key, folds, holdout_rows)`` and reuses them whenever
    the (labels, calibrate settings) key matches.  This is the common case
    during interactive sorting: the user toggles ``inclusion`` or loads a new
    media item, the labels stay the same, and the only work left is re-running
    the cheap threshold rule over the cached folds - no ~200-epoch fold fits.

    A real label change produces a different cache key and falls through to a
    fresh calibration - no explicit invalidation needed.  *score_rows_by_group*
    (see :func:`compute_fold_orderings`) enters the key too, so a change in the
    rows a bag is scored over can never be served from a stale ordering.

    The trained fold *models* are cached alongside the orderings because the
    shipped threshold needs them on every retrain, cache hit or miss: the
    fold-anchored estimator scores the haystack through each fold model.  They
    are process-scoped in-memory state like ``DetectorContext.model`` and are
    never serialised.  So are the held-out rows *holdout_sink* receives, which
    a cache hit hands back as a fresh computation would.
    """
    key = None
    if det_ctx is not None:
        key = _calibration_cache_key(
            X_list,
            y_list,
            calibrate_count,
            calibration_fraction,
            hidden_dim,
            groups,
            score_rows_by_group,
        )
        cached = getattr(det_ctx, "calibration_cache", None)
        if cached is not None and cached[0] == key:
            if holdout_sink is not None:
                holdout_sink.extend(list(rows) for rows in (cached[2] if len(cached) > 2 else ()))
            return cached[1]

    holdouts: list[list[int]] = []
    folds = calibration_folds(
        X_list,
        y_list,
        input_dim,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        rng=np.random.RandomState(CALIBRATION_SPLIT_SEED),
        groups=groups,
        score_rows_by_group=score_rows_by_group,
        holdout_sink=holdouts,
    )
    if det_ctx is not None and key is not None:
        det_ctx.calibration_cache = (key, folds, tuple(tuple(rows) for rows in holdouts))
    if holdout_sink is not None:
        holdout_sink.extend(holdouts)
    return folds


def threshold_from_folds(folds: CalibrationFolds, inclusion_value: float) -> float:
    """The cross-calibration threshold *folds* implies at *inclusion_value*."""
    if folds.fallback is not None:
        return folds.fallback
    return threshold_from_fold_orderings(folds.orderings, inclusion_value)


def conformal_threshold(
    scores: list[float],
    labels: list[float],
    inclusion_value: float = 0,
) -> float:
    """Split-conformal quantile threshold over held-out calibration scores.

    Maps ``inclusion_value`` to a decision threshold via quantiles of the
    calibration score distributions rather than a min-cost search over
    observed cuts.  The min-cost argmin this replaced had exactly as many
    distinct optima as the calibration set had ranking errors, so on
    well-separated votes (the common case) the threshold never moved with
    inclusion; quantiles move whenever the scores have any spread (see
    docs/experiments/2026-07-27-inclusion-knob/REPORT.md and issue #2693).

    The rule, for ``k = inclusion_value`` (``BASE = CONFORMAL_BASE_BUDGET``):

    * A **false-negative cap** ``alpha(k) = min(1, BASE * 2^-k)``: the
      threshold never exceeds the ``alpha``-quantile of the calibration
      *positive* scores, so an estimated ``1 - alpha`` of true matches land
      at or above the cut.  ``+k`` therefore has a portable, user-facing
      meaning - "the fraction of true matches I'm willing to miss, halving
      per step" - independent of the dataset or detector.  The cap is an
      upper bound, not a target: when the classes separate cleanly the cut
      drops to the lowest calibration positive and the budget goes unspent
      (no match is sacrificed that the negatives don't force).
    * A **false-positive guard** for ``k <= 0``: the threshold stays at or
      above the ``1 - BASE * 2^k`` quantile of the calibration *negative*
      scores, so overlap-heavy tasks keep FPR control, and above a walk *up*
      toward the positive score distribution.  The walk interpolates linearly
      in score space from the **gap midpoint** at ``k = 0`` to the
      ``QPOS_MAX`` quantile of positives at ``k = -10`` (at -10 only the
      top-quartile-of-positives region remains - "just the surest matches").

    The **gap midpoint** is what keeps the default cut usable.  When the
    classes separate cleanly there is an empty band between the top of the
    negatives (``fp_guard``) and the bottom of the positives; *every* cut
    inside that band has identical empirical error on the calibration set, so
    the band's top edge - the single lowest calibration positive - is an
    arbitrary choice among equals, and it is the worst one available:

    * It is an **extreme order statistic** over a handful of held-out votes,
      so it moves violently from one vote to the next (issue #2781's "the
      threshold jumps to the top, then it's normal again one click later").
    * It is measured on the **fold models'** score scale but applied to the
      **final** model's scores.  The fold models train on half the votes and
      saturate, so their lowest held-out positive routinely lands above every
      score the final model produces - a cut that admits nothing at all, not
      even the items the user personally voted Good.

    Sitting in the middle of the band is the max-margin choice among cuts the
    calibration data cannot distinguish, and it costs nothing in FN budget:
    the midpoint is strictly below every calibration positive, so a cleanly
    separated task still spends none of its miss budget.

    Every component is monotone non-increasing in ``k``, so their min/max
    composition is too: the threshold is monotone non-increasing in
    inclusion **by construction**.  Raising inclusion can only grow the
    included set, and the sets are nested (everything included at ``k`` stays
    included at ``k + 1``) - which is what makes "cut off at Inclusion 1,
    verify up to Inclusion 4" a well-defined workflow.

    Args:
        scores: Held-out calibration scores, one per example.  Must come from
            data the scoring model did *not* train on; scores on the training
            votes themselves are optimistically separated and yield a
            too-tight band.
        labels: True binary labels (1.0 for good, 0.0 for bad),
            corresponding to ``scores``.
        inclusion_value: Number in ``[-10, 10]``; higher includes more.  The
            reporting knob is an integer, but the rule is continuous in ``k``
            and the acquisition cut sweeps fractional values (issue #3319).

    Returns:
        A float threshold, always realizable within the calibration score
        range (the rule never abstains).  Defaults to 0.5 when the score
        list is empty or single-class (no quantiles to take).
    """
    if not scores:
        return 0.5

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    pos = scores_arr[labels_arr == 1.0]
    neg = scores_arr[labels_arr != 1.0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    def _threshold_at(k: float) -> float:
        fn_cap = float(np.quantile(pos, min(1.0, CONFORMAL_BASE_BUDGET * 2.0**-k)))
        if k > 0:
            # The k=0 floor keeps the seam monotone: q_pos(alpha) can sit
            # above the k=0 cut when the budget goes unspent there.
            return min(fn_cap, _threshold_at(0))
        fp_guard = float(np.quantile(neg, 1.0 - CONFORMAL_BASE_BUDGET * 2.0**k))
        # Midpoint of the band the calibration data cannot resolve: from the
        # top of the negatives up to the lowest positive.  Collapses to
        # ``fp_guard`` under class overlap (no band), so the FPR-controlled
        # regime is untouched - only the cleanly-separated case moves.
        gap_mid = (fp_guard + max(fp_guard, float(np.min(pos)))) / 2.0
        # Walk from that midpoint at k=0 up to the QPOS_MAX positive quantile
        # at k=-10, linearly in score space.  Interpolating on values rather
        # than on quantile positions keeps the knob's stops evenly spaced even
        # when only a handful of calibration positives exist (a quantile walk
        # over 4 points has 4 stops; this one always has 11).
        top = float(np.quantile(pos, CONFORMAL_QPOS_MAX))
        walk = gap_mid + (-k / 10.0) * max(0.0, top - gap_mid)
        return min(fn_cap, max(fp_guard, walk))

    return _threshold_at(inclusion_value)


def threshold_from_fold_orderings(
    fold_orderings: list[tuple[list[float], list[float]]],
    inclusion_value: float,
) -> float:
    """Apply the conformal inclusion rule to the pooled fold orderings.

    Cheap: pools every fold's cached held-out ``(scores, labels)`` and runs
    :func:`conformal_threshold` once - no fold refits.

    **Pooling is deliberate, and only one of its two historical justifications
    survived measurement** (issue #3115, run 2026-08-25,
    ``docs/experiments/2026-08-25-calibration-fold-combine/REPORT.md``):

    * *Resolution* - "the knob's resolution is bounded by the number of
      calibration scores the quantiles are taken over, and per-fold quantiles
      on a handful of votes each would waste the other folds' scores" - **holds,
      and it is what keeps this rule shipped.**  Below roughly 45 votes on the
      run's single-vector cell (roughly 15 on its patch cell), averaging the
      folds' own cuts is *worse* than pooling, by as much as +0.03 paired regret
      around 20 votes.  Each fold's cut is then a quantile over a handful of
      held-out points, and averaging K coarse cuts loses to one quantile over
      the pool.
    * *Exchangeability* - "all folds' scores live on the same sigmoid scale, so
      the pool is exchangeable enough for the quantile rule" - **is refuted in
      the deep regime.**  Past ~100 votes, averaging the per-fold conformal cuts
      in score space beats pooling by −0.0078 ± 0.0015 and −0.016 ± 0.003 on the
      run's two cells.

    The two regimes point opposite ways and the rule is not switched, because
    every path that still reaches this function is a *haystack-less* one - the
    Inclusion slider's re-cut for a detector whose anchored fit degenerated, and
    the library-tier re-derivation entry points - none of which has any reason to
    sit in the deep regime.  A detector with a haystack is cut by
    :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold` and never
    arrives here.  :func:`combined_fold_conformal_threshold` is the measured
    challenger, kept eval-only for exactly this reason.

    Callers must pass a non-empty ``fold_orderings`` (the empty case is
    handled via the ``fallback`` from :func:`compute_fold_orderings`);
    an empty list returns :data:`NO_GOOD_THRESHOLD` defensively.  Held-out
    items the fold model could not score are dropped from the pool by
    :func:`scored_ordering`; if that leaves nothing to take a quantile over,
    the answer is again :data:`NO_GOOD_THRESHOLD` - no calibration evidence
    means admit nothing, never admit everything.
    """
    if not fold_orderings:
        return NO_GOOD_THRESHOLD
    scored = [scored_ordering(ordering) for ordering in fold_orderings]
    pooled_scores = [s for scores, _ in scored for s in scores]
    pooled_labels = [lb for _, labels in scored for lb in labels]
    if not pooled_scores:
        return NO_GOOD_THRESHOLD
    return conformal_threshold(pooled_scores, pooled_labels, inclusion_value)


#: Combine rules for the cross-calibration fold cuts (issue #3115), eval-only.
#:
#: Two functions in this module disagree about the same empirical fact.
#: :func:`threshold_from_fold_orderings` **pools** every fold's held-out scores
#: and takes one conformal quantile, justified by "all folds' scores live on the
#: same sigmoid scale".  :meth:`FoldAnchoredCut._combined_fold_quantile` takes
#: one cut per fold and averages them in **quantile** space specifically so that
#: no cross-scale averaging of raw cuts ever happens - i.e. it is built on the
#: premise that fold scores are *not* directly comparable.
#:
#: **Both were measured, and each is right in a different regime** (#3115, run
#: 2026-08-25; see :func:`threshold_from_fold_orderings` for the numbers and
#: ``docs/experiments/2026-08-25-calibration-fold-combine/REPORT.md`` for the
#: run).  Averaging beats pooling past ~100 votes in both of the run's cells and
#: loses to it in the cold start; the *quantile*-space step flips sign between
#: the cells outright.  That sign flip is **not attributable to the voting
#: mode**: the run's two cells confounded mode with the embedder, and #3310
#: later filled the missing corner on the same grid and found the neighbouring
#: fold-count effect follows the **embedder**, not the mode (as #3287's
#: calibration-fraction optimum also did).  So there is no measured licence for a
#: mode-dependent rule, and issue #3258, which proposed one, was closed unbuilt.
#:
#: The four rules here are the challengers, and they factor the disagreement
#: rather than confounding it.  Against the pooled control they decompose as:
#:
#: * ``pooled -> tmean``  - pooling vs **averaging**, held in one score space.
#: * ``tmean  -> qmean``  - score space vs **quantile** space, i.e. exactly the
#:   comparability premise the two docstrings disagree on, with the combine held
#:   fixed.
#: * ``*mean  -> *median`` - **contamination**: a degenerate fold pours its
#:   scores straight into a pooled quantile, gets 1/K weight under a mean, and
#:   ~none under a median.
#:
#: There is no ``qpooled``: a pooled cut has no single fold haystack to read a
#: quantile in, so that cell of the 2x2 does not exist.  That is why the total
#: ``pooled -> qmean`` contrast the issue asks for has to be read through the
#: two legs above rather than attributed to either on its own.
FOLD_CONFORMAL_COMBINES: tuple[str, ...] = ("tmean", "tmedian", "qmean", "qmedian")


def per_fold_conformal_cuts(
    fold_orderings: list[tuple[list[float], list[float]]],
    inclusion_value: int,
) -> list[tuple[int, float]]:
    """``(fold index, conformal cut)`` for every fold that can produce one.

    A fold whose scored held-out set is empty or **single-class** is skipped
    rather than cut: :func:`conformal_threshold` answers 0.5 there, which is a
    "no calibration evidence" sentinel and not a threshold, and averaging it in
    would move the combined cut toward the middle of the sigmoid for a reason
    that has nothing to do with where the classes sit.  Skipping is also what
    makes the contamination question *measurable* - the caller reports how many
    folds it dropped, so a row where the mean and the pooled quantile disagree
    can be attributed to the drop or exonerated of it.
    """
    cuts: list[tuple[int, float]] = []
    for i, ordering in enumerate(fold_orderings):
        scores, labels = scored_ordering(ordering)
        if not scores:
            continue
        arr = np.asarray(labels, dtype=np.float64)
        if not (bool(np.any(arr == 1.0)) and bool(np.any(arr != 1.0))):
            continue
        cuts.append((i, conformal_threshold(scores, labels, inclusion_value)))
    return cuts


def combined_fold_conformal_threshold(
    fold_orderings: list[tuple[list[float], list[float]]],
    inclusion_value: int,
    *,
    combine: str,
    fold_haystacks: "list[np.ndarray] | None" = None,
    final_scores: "list[float] | np.ndarray | None" = None,
) -> tuple[float, str]:
    """Combine the folds' *own* conformal cuts instead of pooling their scores.

    The challenger to :func:`threshold_from_fold_orderings` (issue #3115); see
    :data:`FOLD_CONFORMAL_COMBINES` for what each rule isolates.

    **Still eval-only, now by verdict rather than by default.**  The run this
    exists for happened (2026-08-25) and did resolve the combine leg above its
    pre-registered margin - but in the *deep* regime only, and every path that
    still reaches the pooled rule is a haystack-less fallback with no reason to
    be deep.  :func:`threshold_from_fold_orderings` carries the regimes and the
    numbers; the short version is that promoting ``tmean`` would trade a
    −0.008/−0.016 deep-regime gain for a cold-start loss up to +0.03 on exactly
    the paths that remain.

    ``"tmean"`` / ``"tmedian"`` average the per-fold cuts in **score** space.
    This is the rule that presumes the folds' sigmoid scales are comparable, and
    it is the one with an exact control: at ``K == 1`` there is a single cut to
    average, so both reproduce the pooled cut *bit for bit* - including the
    conformal rule's gap midpoint, which is a specific point inside an empty
    band rather than an order statistic.

    ``"qmean"`` / ``"qmedian"`` carry each fold's cut to the final model as a
    quantile of **that fold's own haystack** (:func:`rank_transfer`'s argument),
    combine the quantiles, then realize the result on *final_scores* and
    :func:`snap_cut_to_sample` it - the same chain
    :meth:`FoldAnchoredCut.threshold_at` runs, so the two paths differ in what
    is being cut and not in how the cut travels.  Note that this **cannot**
    reproduce the pooled cut even at ``K == 1``: a quantile records which
    observed scores a cut sits between and not where inside that gap it sat, so
    the conformal midpoint is destroyed by the round trip.  That is a real
    property of quantile-space combining and not an implementation wart, which
    is why the ``tmean`` leg exists to separate it from the combine itself.

    Reading each fold's quantile in its own haystack, rather than in its handful
    of held-out votes, also answers the resolution objection
    :func:`threshold_from_fold_orderings`' docstring raises: per-fold quantiles
    are coarse only when taken over the votes.  Taken over the sim set they are
    finer than the pooled rule's, not coarser.

    Args:
        fold_orderings: The fold prefix's cached ``(scores, labels)`` holdouts.
        inclusion_value: Passed through to :func:`conformal_threshold`.
        combine: One of :data:`FOLD_CONFORMAL_COMBINES`.
        fold_haystacks: Per-fold sim-set score arrays, index-aligned with
            *fold_orderings*.  Required by the ``q*`` rules, ignored by ``t*``.
        final_scores: The final model's sim-set scores, the array a ``q*``
            result is realized on.

    Returns:
        ``(threshold, provenance)``.  Provenance is
        ``"fold_conformal_{combine}[a/k]"`` with *a* the folds that contributed
        of the *k* offered, or ``"fold_conformal_fallback_pooled"`` when no fold
        could contribute one and the pooled rule answers instead - which keeps
        the arm defined on exactly the steps the control is defined on, so the
        contrast never silently drops rows.
    """
    if combine not in FOLD_CONFORMAL_COMBINES:
        raise ValueError(f"unknown fold conformal combine {combine!r}; expected one of {FOLD_CONFORMAL_COMBINES}")
    n_offered = len(fold_orderings)
    cuts = per_fold_conformal_cuts(fold_orderings, inclusion_value)

    if combine in ("tmean", "tmedian"):
        values = [c for _i, c in cuts]
    else:
        if fold_haystacks is None or final_scores is None:
            raise ValueError(f"combine {combine!r} needs fold_haystacks and final_scores")
        values = []
        for i, cut in cuts:
            if i >= len(fold_haystacks):
                continue
            src = np.sort(np.asarray(fold_haystacks[i], dtype=np.float64).ravel())
            if src.size == 0:
                continue
            values.append(float(np.searchsorted(src, cut, side="left")) / float(src.size))

    if not values:
        return threshold_from_fold_orderings(fold_orderings, inclusion_value), "fold_conformal_fallback_pooled"

    agg = float(np.mean(values)) if combine.endswith("mean") else float(np.median(values))
    provenance = f"fold_conformal_{combine}[{len(values)}/{n_offered}]"

    if combine in ("tmean", "tmedian"):
        return agg, provenance

    target = np.asarray(final_scores, dtype=np.float64).ravel()
    if target.size == 0:
        return threshold_from_fold_orderings(fold_orderings, inclusion_value), "fold_conformal_fallback_pooled"
    realized = float(np.quantile(target, min(1.0, max(0.0, agg))))
    return snap_cut_to_sample(realized, np.sort(target)), provenance


def calculate_cross_calibration_threshold(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    inclusion_value: int = 0,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
) -> float:
    """Estimate a decision threshold using k-fold calibration.

    Performs ``calibrate_count`` independent random Train/Calibrate splits.
    For each split, trains a model on the Train portion and scores the
    held-out Calibrate portion.  The pooled held-out scores then feed the
    conformal inclusion rule via :func:`threshold_from_fold_orderings`.

    Algorithm:
        For each of *k* = ``calibrate_count`` rounds:
        1. Stratified random split into Train (``1 - calibration_fraction``)
           and Calibrate (``calibration_fraction``).  Stratification guarantees
           the Train side has at least one of each class, so the per-fold MLP
           fit always has both-class supervision.
        2. Train a model on Train.
        3. Score the Calibrate portion.
        Pool the *k* rounds' held-out (score, label) pairs and apply
        :func:`conformal_threshold` at *inclusion_value*.

    Args:
        X_list: List of embedding arrays (one per labelled example).
        y_list: List of binary labels (1.0 for good, 0.0 for bad),
            aligned with ``X_list``.
        input_dim: Dimensionality of the embeddings.
        inclusion_value: Integer in ``[-10, 10]`` passed to
            :func:`conformal_threshold` to control the miss/false-alarm
            trade-off.  It does **not** enter model training (the fold models
            are inclusion-independent), so the same fold scores can be
            re-thresholded at any inclusion.
        rng: Optional RandomState for the Train/Calibrate splits.  When
            ``None`` a fresh ``RandomState(CALIBRATION_SPLIT_SEED)`` is used,
            so the splits are reproducible and the global ``np.random`` state
            is never read or advanced.
        calibrate_count: Number of random Train/Calibrate splits (default 2).
        calibration_fraction: Fraction of data used for calibration in each
            split (default 0.5).  For example, 0.2 means 80% Train / 20%
            Calibrate.  If the fraction is so extreme that a valid split
            cannot be formed (fewer than 2 training or 1 calibration
            examples), returns :data:`NO_GOOD_THRESHOLD` so that nothing
            is predicted as Good.
        hidden_dim: Force a specific hidden-layer width for the fold models.
            When ``None`` (default), each fold model auto-sizes based on its
            own training-set size.  Pass the full-data hidden dim to ensure
            fold models match the final model's architecture.
        score_rows_by_group: Per-group inference row stacks; see
            :func:`compute_fold_orderings`.  Grouped path only.

    Returns:
        A float threshold. Returns 0.5 when calibration is not possible:
        fewer than 4 examples total, or fewer than 2 of either class
        (stratified splitting needs at least one of each class on both
        the train and calibrate sides).  Returns :data:`NO_GOOD_THRESHOLD`
        (a finite sentinel above the sigmoid range) if
        ``calibration_fraction`` makes a valid split impossible.
    """
    orderings, fallback = compute_fold_orderings(
        X_list,
        y_list,
        input_dim,
        rng=rng,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
        groups=groups,
        score_rows_by_group=score_rows_by_group,
    )
    if fallback is not None:
        return fallback
    return threshold_from_fold_orderings(orderings, inclusion_value)
