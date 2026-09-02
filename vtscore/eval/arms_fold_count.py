"""The fold-count arms (issues #2897 / #3115).

How many calibration folds the cut should be read over, and how their per-fold
cuts should be combined.  Unlike the other arm families these need no simulation
scores of their own - they re-cut fold orderings the step already trained - so
they run whether or not safe thresholds are on; pooled simulation scores, when
present, only add the blended arm.

:func:`parse_fold_count_schedule` is the public half: it turns a study's
``"0-40:3,40-:5"``-style spec into the ``step -> k`` callable the loop threads.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

import numpy as np

from vtscore.eval.row_metrics import folds_used, operating_metrics, round6
from vtscore.training.blend_schedules import BlendContext


def _fold_schedule_segment(part: str) -> tuple[int, int]:
    """One ``"K@N"`` segment as ``(cut, count)``, or a ValueError naming it.

    Split out of :func:`parse_fold_count_schedule` so the validation reads as a
    list of the four ways a spec can be wrong rather than as branches inside the
    parse loop.  The error says which SEGMENT was bad, because a schedule is
    typed into a launcher and "invalid schedule" would leave a reader diffing
    the string by eye.
    """
    k_str, sep, n_str = part.strip().partition("@")
    if not sep:
        raise ValueError(f"fold-count schedule segment {part!r} is not 'K@N'")
    try:
        k, n = int(k_str), int(n_str)
    except ValueError as exc:
        raise ValueError(f"fold-count schedule segment {part!r} is not 'K@N': {exc}") from exc
    if k < 1:
        raise ValueError(f"fold-count schedule segment {part!r}: K must be >= 1")
    if n < 1:
        raise ValueError(f"fold-count schedule segment {part!r}: N must be >= 1")
    return n, k


def parse_fold_count_schedule(spec: str | None, base: int) -> "Callable[[int], int] | None":
    """Eval-only (#3314): resolve ``calibrate_count`` per step from the vote count.

    ``"K@N"`` means ``K(n_votes) = K while n_votes < N, else base`` - the family
    pre-registered in ``docs/experiments/2026-08-28-calibration-fold-count-3310/PLAN.md``,
    written the issue's way round: *more folds while the labelset is small,
    decaying to production's count*.  Several segments may be chained
    (``"8@25,4@60"``), and they are read in ascending ``N`` order, so the first
    cut a vote count falls under wins and the ordering in the string does not
    matter.

    This is a **harness knob and not a shipped setting**.  ``CALIBRATE_COUNT``
    and the app's own constant are untouched, so no other study's arm moves:
    a run that does not set it resolves to *base* at every step, which is
    exactly what the constant did before.  If the schedule ever ships, the
    right shape is a ``production_fold_count_for(n_votes)`` beside
    ``production_split_for``, with ``scripts/check-eval-app-sync.py`` gaining
    the mirror - the same discipline #3287's split fraction went through.

    Returns ``None`` for an empty spec, so the caller's fast path is the
    unscheduled one.
    """
    if not spec or not spec.strip():
        return None
    segments = sorted(_fold_schedule_segment(part) for part in spec.split(",") if part.strip())
    if not segments:
        return None

    def resolve(n_votes: int) -> int:
        for cut, k in segments:
            if n_votes < cut:
                return k
        return base

    return resolve


def _stop(timings: dict[str, float] | None, started: float) -> None:
    """Record ``anchored_seconds`` since *started*, when the caller wants timings."""
    if timings is not None:
        timings["anchored_seconds"] = time.monotonic() - started


def _fold_count_arms(
    prefix: list[tuple[list[float], list[float]]],
    xcal: float,
    inclusion: int,
    haystacks: "list[np.ndarray] | None",
    final_fit_scores: list[float] | None,
    ctx: BlendContext | None,
    gmm_cut: float | None,
    gmm_fit: Any,
    schedule: str | None,
    timings: dict[str, float] | None = None,
) -> list[tuple[str, float, str, float]]:
    """``(arm, threshold, provenance, blend weight)`` for one fold prefix.

    *timings*, when given, collects the wall clock of the parts a **live** run
    at this K would actually pay for (#3314) - currently ``anchored_seconds``,
    the fit of production's fold-anchored mixture over this prefix.  The other
    arms here are counterfactual re-cuts of the same folds and cost a live run
    nothing, so they are deliberately not timed: a cost model built off the
    whole function would price the study's own instrumentation as if the user
    waited through it.

    Split out of :func:`_fold_count_variant_rows` so the arm table is one
    readable list rather than a branch pile inside the K loop; every arm here is
    a different *rule* applied to the **same** already-trained folds, so adding
    one costs arithmetic and no fits.

    ``haystacks`` is ``None`` when the step does not carry one sim-set score
    array per fold in the prefix, which gates every arm that has to read a cut
    in the fold's own distribution (:func:`~vtscore.training.thresholds.rank_transfer`).
    ``final_fit_scores`` is the final model's haystack the quantile-realizing
    arms cut on - under the #3308 convention the caller passes it with the
    voted items already dropped, matching the fold ``haystacks``; the ``blend``
    arm's GMM inputs (*gmm_cut*, *gmm_fit*) are fitted upstream on the full
    distribution, as production's fallback is.
    """
    import dataclasses  # noqa: PLC0415

    from vtscore.training.thresholds import (  # noqa: PLC0415
        FOLD_CONFORMAL_COMBINES,
        blend_gmm_threshold,
        combined_fold_conformal_threshold,
        fit_fold_anchored_cut,
        fold_anchored_gmm_threshold,
        safe_blend_weight,
    )

    nan = float("nan")
    arms: list[tuple[str, float, str, float]] = [("xcal", xcal, "conformal", nan)]
    if ctx is not None and gmm_cut is not None:
        weight = safe_blend_weight(ctx, schedule)
        blended = blend_gmm_threshold(xcal, gmm_cut, ctx, schedule=schedule, fit=gmm_fit)
        arms.append(("blend", blended, "gmm_blend" if weight < 1.0 else "conformal", weight))

    # #3115, the combine rule.  ``xcal`` above IS the pooled control - it calls
    # `threshold_from_fold_orderings` verbatim - so these are challengers rather
    # than a re-emission of it.  The score-space pair needs nothing the pooled
    # arm does not already have; the quantile-space pair needs a haystack per
    # fold, the same condition the anchored arms carry.
    #
    # `transferable` binds that condition once - "can this arm read a cut in a
    # fold's own scale?" - so it is asked in one place instead of re-derived at
    # each use site, and so both `None` cases narrow for the type checker.
    final_scores = final_fit_scores if final_fit_scores else None
    transferable = haystacks if (haystacks is not None and final_scores is not None) else None
    for combine in FOLD_CONFORMAL_COMBINES:
        quantile_space = combine.startswith("q")
        if quantile_space and transferable is None:
            continue
        value, prov = combined_fold_conformal_threshold(
            prefix,
            inclusion,
            combine=combine,
            fold_haystacks=transferable if quantile_space else None,
            final_scores=final_scores if quantile_space else None,
        )
        arms.append((combine, value, prov, nan))

    # The arms for the rule users actually get.  ``anchored`` is production
    # (`FOLD_ANCHOR_COMBINE`); ``anchored_qmedian`` re-cuts the *same* fit under
    # the robust combine, which puts #3115's contamination question on the
    # shipped path rather than only on the retired blend.
    #
    # Fitted **once** and re-cut, not fitted twice.  `FoldAnchoredCut` exists to
    # separate the fit from the cut, and the combine rule is read at cut time, so
    # a second `fold_anchored_gmm_threshold` call would re-run one anchored EM
    # per fold to reach the same mixtures - doubling the study's dominant cost
    # (sum over the K grid, so 52 EM fits per step at K<=16, not 16).  It is also
    # the stronger contrast: the two rows differ in the combine and in *nothing
    # else*, including the fits' own numerical noise.
    if transferable is not None and final_scores is not None:
        # Timed with the fallback inside it (#3314): a step whose anchored fit
        # degenerates still pays for the attempt *and* for the ladder the
        # shipped helper walks afterwards, and that is what a user waits
        # through.  Charging those steps only for the branch they did not take
        # would under-price exactly the cold-start regime this study reads.
        t_anchored = time.monotonic()
        cut = fit_fold_anchored_cut(transferable, prefix, final_scores)
        if cut is None:
            # Both arms land on the same terminal fallback; take it from the
            # shipped helper rather than duplicating its ladder here.
            value, prov = fold_anchored_gmm_threshold(transferable, prefix, final_scores, inclusion)
            _stop(timings, t_anchored)
            arms.extend([("anchored", value, prov, nan), ("anchored_qmedian", value, prov, nan)])
        else:
            shipped = cut.threshold_at(inclusion)
            # The clock stops HERE, on production's own cut and before the
            # `qmedian` re-cut: that arm is a counterfactual read of the same
            # fitted mixture and costs a live run nothing, so timing it would
            # charge K for the harness's own instrumentation.  (Sharing the fit
            # is exactly why the two arms are cheap; see the note above.)
            _stop(timings, t_anchored)
            arms.append(("anchored", shipped, cut.provenance, nan))
            robust = dataclasses.replace(cut, combine="qmedian")
            arms.append(("anchored_qmedian", robust.threshold_at(inclusion), robust.provenance, nan))
    return arms


def _fold_count_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    inclusion: int,
    n_pool_rows: float,
    counts: list[int],
    sim_pooled_scores: list[float] | None,
    schedule: str | None,
    sim_fit_scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    """One metric row per calibration **fold count** K (issue #2897).

    *sim_fit_scores* is the sim haystack under the #3308 population convention
    (voted items dropped); the ``anchored`` / quantile-space arms realize their
    cuts on it, mirroring the shipped rule, while the retired ``blend`` arm
    keeps fitting *sim_pooled_scores* - production's fallback blend also keeps
    the full distribution.  ``None`` falls back to *sim_pooled_scores*.

    The study's screen for "does more cross-calibration buy anything, and what
    does it cost".  It is exact rather than approximate, because the folds are
    *nested*: :func:`~vtscore.training.thresholds.compute_fold_orderings` draws
    each fold as an independent stratified split off one ``RandomState(42)``
    stream, at a per-fold size that does not depend on the count, so the K folds
    a live ``calibrate_count=K`` run would train are byte-for-byte the first K
    of the Kmax folds trained here.  Slicing the prefix therefore reproduces
    each K's threshold exactly, and the arm at ``K == calibrate_count``
    reproduces this step's own pre-blend conformal cut - the control that
    licenses the rest of the table.

    Arms per K, because the fold count and the shipped threshold are different
    questions - and, since #3115, because *how* the folds are combined is a
    third one.  Every arm below re-reads the same already-trained fold prefix,
    so the whole table costs arithmetic:

    * ``folds_k{K}_xcal`` - the raw cross-calibration cut, the thing K is
      actually a knob on.
    * ``folds_k{K}_blend`` - that cut after the ``cap50`` safe-threshold mix-in.
      The blend weight depends only on the vote counts, so it is identical
      across K and this arm isolates how much of K's benefit survives being
      averaged with the GMM cut.  Emitted only when the step has the pooled sim
      scores the blend fits.
    * ``folds_k{K}_anchored`` - **production's rule** (#3116).  The blend above
      was retired by the 2026-08-05 population-anchored run; the shipped path is
      :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold`, which
      fits one anchored mixture *per fold* and combines them in quantile space.
      K therefore moves the shipped threshold through a path the other two arms
      do not exercise at all - the blend's GMM half is a single unanchored fit
      on the sim haystack and is K-independent by construction, so ``blend``
      varies only in its x-cal half.  Without this arm every conclusion about
      what ``calibrate_count`` does to the threshold users actually get is
      partial.  Emitted when the step carries at least K fold haystacks, which
      :func:`_safe_threshold_for_step` supplies (so: under safe thresholds).
    * ``folds_k{K}_{tmean,tmedian,qmean,qmedian}`` - **the combine rule**
      (#3115).  ``xcal`` above *is* the pooled control: it calls
      :func:`~vtscore.training.thresholds.threshold_from_fold_orderings`
      verbatim, which pools every fold's held-out scores into one bag and takes
      a single conformal quantile.  These four take one cut per fold and combine
      them instead, in score space (``t*``) or in fold-quantile space (``q*``);
      see :data:`~vtscore.training.thresholds.FOLD_CONFORMAL_COMBINES` for what
      each leg of the contrast isolates.  ``q*`` needs a haystack per fold and
      so shares the ``anchored`` arm's condition.
    * ``folds_k{K}_anchored_qmedian`` - production's rule re-cut under the
      robust combine, which puts the contamination question on the **shipped**
      path and not only on the retired blend.

    Note what is *not* a defect here: the unanchored ``fit_gmm_threshold`` is
    hoisted out of the K loop, and that is correct rather than a frozen
    reference - it reads only ``sim_pooled_scores``, which no fold count
    touches, so re-fitting it per K would return the same cut at K times the
    price.  The gap #3116 identified is the missing arm above, not the hoist.

    **The cost columns.**  ``fold_seconds`` is what #2897 and #3116 read: the
    measured fit time of this K's own folds plus the count-independent overhead
    of the threshold rule.  It is *not* the whole price of K, and #3314 found
    that reading it as one under-states the exchange rate badly, because two
    other K-proportional pieces of the shipped calibration are paid outside it
    and outside every other timing column the frame carries:

    * ``fold_score_seconds`` - one scoring pass over the sim set per fold, so
      the shipped rule can anchor each fold's mixture on that fold's own
      haystack (measured in :func:`_safe_threshold_for_step`).
    * ``anchored_seconds`` - production's ``fit_fold_anchored_cut`` over the
      prefix: one anchored EM per fold.

    ``cal_seconds`` sums all of it (``fold_fit_seconds + overhead +
    fold_score_seconds + anchored_seconds``) and is the column an affordability
    ceiling has to read.  ``train_seconds``, ``xcal_seconds``,
    ``pool_score_seconds`` and ``test_score_seconds`` cover the *rest* of the
    step and none of them covers the safe-threshold block, so
    ``cal_seconds + train_seconds + pool_score_seconds + test_score_seconds`` is
    the per-step wall clock a live run at K would have.

    All of it is measured inside the one Kmax run, so every K's timing shares
    one machine, one process and one cache state - the *ratios* are the
    load-bearing part, not the absolute seconds.

    This is the study's screen, not its verdict, for the usual reason (see
    :func:`_schedule_variant_rows`): K also steers acquisition through the
    threshold Autopilot's Hard pick ranks around, and a screen that holds the
    trajectory fixed cannot see the votes a different K would have collected.
    The live A/B runs exist to measure exactly that.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.training.thresholds import (  # noqa: PLC0415
        fit_gmm_threshold,
        threshold_from_fold_orderings,
    )

    fold_data = details.get("fold_count_data")
    if not fold_data:
        return []
    orderings = fold_data["orderings"]
    seconds = fold_data["seconds"]
    overhead = float(fold_data.get("overhead_seconds") or 0.0)
    haystacks = fold_data.get("haystacks") or []
    haystack_seconds = fold_data.get("haystack_seconds") or []
    if not orderings:
        return []

    ctx = None
    gmm_cut = gmm_fit = None
    if sim_pooled_scores:
        ctx = BlendContext(
            n_labels=int(details["n_votes"]),
            n_good=int(details["n_good"]),
            n_bad=int(details["n_bad"]),
        )
        gmm_cut, gmm_fit = fit_gmm_threshold(sim_pooled_scores)

    rows: list[dict[str, Any]] = []
    for k in counts:
        if k < 1 or k > len(orderings):
            continue
        prefix = orderings[:k]
        cal_scores = np.array([s for scores, _ in prefix for s in scores])
        cal_labels = np.array([lb for _, labels_ in prefix for lb in labels_])
        xcal = threshold_from_fold_orderings(prefix, inclusion)
        fold_fit_seconds = float(sum(seconds[:k]))
        fold_seconds = round6(fold_fit_seconds + overhead)
        # The rest of what a live run at K pays for, beside the fold fits
        # (#3314).  Scoring the sim set once per fold is K-proportional and is
        # measured in `_safe_threshold_for_step`; the anchored fit over the
        # prefix is measured below, inside `_fold_count_arms`.
        fold_score_seconds = float(sum(haystack_seconds[:k])) if len(haystack_seconds) >= k else float("nan")
        timings: dict[str, float] = {}

        arms = _fold_count_arms(
            prefix,
            xcal,
            inclusion,
            haystacks[:k] if len(haystacks) >= k else None,
            sim_fit_scores if sim_fit_scores is not None else sim_pooled_scores,
            ctx,
            gmm_cut,
            gmm_fit,
            schedule,
            timings=timings,
        )
        anchored_seconds = float(timings.get("anchored_seconds", float("nan")))
        # The FULL calibration wall clock at K: fold fits + one haystack scoring
        # pass per fold + production's anchored fit over the prefix + the
        # count-independent overhead of the conformal rule.  `fold_seconds` is
        # deliberately left as it was (#2897/#3116 read it, and archived runs
        # are compared against it), but it is only the first of those four
        # terms, so an affordability rule must read THIS column.
        cal_seconds = round6(fold_fit_seconds + overhead + fold_score_seconds + anchored_seconds)

        for arm, threshold, provenance, weight in arms:
            row = operating_metrics(
                base_scores,
                base_labels,
                threshold,
                inclusion,
                cal_scores,
                cal_labels,
                pool_variant="max",
                provenance=provenance,
                n_pool_rows=n_pool_rows,
            )
            row["gmm_variant"] = f"folds_k{k}_{arm}"
            row["schedule"] = schedule or ""
            row["xcal_threshold"] = round6(xcal)
            row["gmm_cut"] = round6(gmm_cut) if gmm_cut is not None else float("nan")
            row["blend_weight"] = round6(weight)
            row["fold_count"] = k
            row["fold_seconds"] = fold_seconds
            row["fold_fit_seconds"] = round6(fold_fit_seconds)
            row["fold_score_seconds"] = round6(fold_score_seconds)
            row["anchored_seconds"] = round6(anchored_seconds)
            row["cal_seconds"] = cal_seconds
            row["n_cal_scores"] = int(cal_scores.size)
            row["n_folds_used"] = folds_used(provenance, k)
            rows.append(row)
    return rows
