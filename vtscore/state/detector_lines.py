"""A detector's line, re-derived from what its last retrain cached: re-cuts, balance state, acquisition cut.

Everything here reads a :class:`~vtscore.state.detector_context.DetectorContext`
and the estimators parked on it; nothing retrains.  :func:`recompute_detector_thresholds`
is the one writer: it re-cuts every loaded detector at its own balance.
"""

from __future__ import annotations

import math
import warnings
from typing import Any, overload

from vtscore.state.detector_context import DetectorContext
from vtscore.state.detector_registry import loaded_detector_contexts
from vtscore.state.primitives import _state_lock


def _recut_balance(ctx: "DetectorContext", beta: float) -> float | None:
    """The balance's line for *ctx* at *beta*, or ``None`` for the caller's fallbacks.

    The labels' line (#4452) when the last retrain fitted one: its class model
    cut at *beta* and the prevalence it was cut at, reading nothing from the
    ranking, so it moves the same in Train and in Find.  A trained head with no
    class model (too few votes, one class) draws no count on the corpus: the
    fallbacks answer, as the retrain's did.  Only a context with no head at
    all still reads the count rule over its ranking (#4413).
    """
    from vtscore.detectors.query_head import is_quota_head  # noqa: PLC0415
    from vtscore.training.thresholds import balance_line  # noqa: PLC0415

    if ctx.labels_line is not None:
        return float(ctx.labels_line.threshold(beta))
    if ctx.centroid_line is not None and is_quota_head(ctx.model):
        # The Goods' centroid's line, or a typed query's (#4758), at this balance, on the
        # cosines it was drawn on (#4732).
        return ctx.centroid_line.threshold_at(beta)
    if ctx.model is not None:
        return None
    return balance_line(
        ctx.line_ranking,
        beta,
        ctx.precision_check,
        human_voted_ids(ctx),
        proposal=detector_balance_proposal(ctx, beta),
    )


def recut_detector_threshold(
    ctx: "DetectorContext",
    inclusion_value: float | None = None,
    *,
    beta: float | None = None,
) -> float | None:
    """The threshold *ctx*'s cached estimators cut at an operating point, or ``None``.

    The one place a detector's cut is re-derived without a retrain.  A balance
    change (:func:`recompute_detector_thresholds`), the acquisition cut
    (:func:`detector_acquisition_threshold`) and Smart's pricing all read it.

    The operating point is a balance (*beta*) or, when none is given, an
    inclusion (*inclusion_value*) - the internal unit, not a user preference
    (#4269).

    **Under a balance** (#4413) the line is :func:`_recut_balance`'s: the
    labels' line (#4452) when the last retrain fitted one, or the Goods'
    centroid's or a typed query's own line (#4732, #4758).  A trained head
    with no class model (too few votes, one class) draws no count on the
    corpus: the fallbacks below answer, as the retrain's did.  Only a context
    with no head at all keeps a count (#4272): the top *count* unvoted items
    of the ranking the last retrain scored (``ctx.line_ranking``), where
    *count* is the mixture's F-beta argmax under the balance's cap
    (:func:`~vtscore.training.thresholds.balance_line`, shared with the eval
    harness; a finished walk's end replaces it only under the ``trim`` check
    shape, which only the harness forces).  The unvoted remainder is read
    against the live votes, so that line follows the ranking at the same
    count as votes come in.  With no ranking to read (never trained against a
    haystack, or a structural detector) the balance has no line and the
    fallbacks below answer at the inclusion given.

    With no balance, or a balance that leaves the line to them, in order:

    * **The fold-anchored estimator** (``ctx.anchored_cut_cache``), when training
      fitted one, re-cut at *inclusion_value*.  Arithmetic on fitted state, so
      the result is exactly what a retrain at this operating point would have
      stored.
    * **The conformal rule over the cached fold orderings**, when there is no
      estimator (a degenerate fit, or no haystack to fit one on), at that
      inclusion.  That drops the schedule blend a retrain would have mixed in,
      which needs state the cache does not keep (the comprehensive-audit-2026-07
      "skip blend on slides" ruling).
    * **``None`` - leave the threshold alone** - in every other case: no cached
      folds, or folds that never split (``folds.fallback`` set: too few votes,
      or one class).  The stored threshold there came from an inclusion-blind
      rule - the schedule blend, or the fallback itself - so no operating
      point moves it.  Returning the fallback instead would replace a fitted
      blend cut with the bare 0.5 sentinel, which can admit *fewer* items at
      a more lenient operating point.
    """
    from vtscore.training.thresholds import reporting_line

    if beta is None and inclusion_value is None:
        raise ValueError("an operating point needs an inclusion or a balance")
    if beta is not None:
        kept = _recut_balance(ctx, beta)
        if kept is not None:
            return kept
    line = reporting_line(
        ctx.anchored_cut_cache,
        None,
        inclusion_value=inclusion_value if inclusion_value is not None else 0.0,
        min_precision=None,
    )
    if line.threshold is not None:
        return line.threshold
    cache = ctx.calibration_cache
    if cache is None or line.inclusion is None:
        return None
    folds = cache[1]
    if folds.fallback is not None or not folds.orderings:
        return None
    from vtscore.training.thresholds import threshold_from_fold_orderings

    return float(threshold_from_fold_orderings(folds.orderings, line.inclusion))


def detector_precision_floor(ctx: "DetectorContext", min_precision: float) -> Any:
    """Deprecated (#4362): always an ``insufficient_evidence`` :class:`~vtscore.training.thresholds.PrecisionFloorCut`.

    It read the #4220 estimate a retrain parked on ``ctx.precision_floor_cache``.
    A retrain no longer builds that estimate, so there is never evidence to
    read: the verdict is ``insufficient_evidence`` with zero calibration
    positives, as it always was for a detector with no estimate.  The spot
    check decides the line (:func:`detector_balance_state`).  A caller that
    wants the estimator's own reading builds a
    :class:`~vtscore.training.thresholds.PrecisionFloorEstimate` itself.
    """
    from vtscore.training.thresholds import unpromised

    warnings.warn(
        "detector_precision_floor() is deprecated: a retrain no longer builds the #4220 precision estimate "
        "(#4362), so its verdict is always insufficient_evidence. Read detector_balance_state() instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return unpromised(min_precision)


def human_voted_ids(ctx: "DetectorContext") -> set[int]:
    """The items a person has voted on in *ctx*'s current session: never a spot check's candidate.

    In Find mode every scored item carries a machine label, so the human's
    votes are the verified ones; elsewhere they are the vote dicts themselves.
    """
    if ctx.find_mode:
        return set(ctx.verified_ids)
    return set(ctx.good_votes) | set(ctx.bad_votes)


def detector_line_labels(ctx: "DetectorContext") -> dict[int, bool]:
    """The human votes as anchors for the line's mixture: ``{media id: is Good}`` over :func:`human_voted_ids`."""
    return {cid: cid in ctx.good_votes for cid in human_voted_ids(ctx)}


def detector_balance_proposal(ctx: "DetectorContext", beta: float) -> int | None:
    """The mixture's F-beta argmax on *ctx*'s ranking at *beta* (#4413), or ``None``: the unchecked balance line's proposal."""
    from vtscore.training.thresholds import fbeta_count

    return fbeta_count(ctx.line_ranking, beta, detector_line_labels(ctx), human_voted_ids(ctx))


def detector_balance_positives(ctx: "DetectorContext") -> float | None:
    """The mixture's count of positives among *ctx*'s unvoted ranking (#4413): a balance walk's recall denominator."""
    from vtscore.training.thresholds import mixture_positives

    return mixture_positives(ctx.line_ranking, detector_line_labels(ctx), human_voted_ids(ctx))


def detector_walk_positives(ctx: "DetectorContext", beta: float) -> float:
    """The positives a balance walk over *ctx*'s ranking reads recall against (#4419): the mixture's count, else the cap."""
    from vtscore.training.thresholds import walk_positives

    return walk_positives(ctx.line_ranking, beta, detector_line_labels(ctx), human_voted_ids(ctx))


@overload
def detector_balance_state(ctx: "DetectorContext", beta: float) -> dict[str, Any]: ...
@overload
def detector_balance_state(ctx: "DetectorContext", beta: None) -> None: ...
@overload
def detector_balance_state(ctx: "DetectorContext", beta: float | None) -> dict[str, Any] | None: ...
def detector_balance_state(ctx: "DetectorContext", beta: float | None) -> dict[str, Any] | None:
    """What the balance says about *ctx*'s current line (#4413), for a response that carries the line.

    Every place a detector's threshold leaves the process - a sort result, a
    Find pass, a balance change, a headless export - reports it beside the
    threshold (#4247, #4272).  ``status`` is ``unchecked`` (no spot check has
    finished at this balance) or ``checked`` (one has; it audits the line and
    never moves it, #4452); ``count`` is how many unvoted items of the ranking
    scored last the line keeps (the balance's cap when no ranking is kept);
    ``precision`` and ``recall`` are the check's likely ranges for the set the
    line kept when it started (#4483) and ``fbeta`` its estimate, with
    ``stale`` once the ranking under them moved; ``schedule`` is what a check
    at this balance costs.  On a structural detector's line ``status`` is
    ``gate`` and ``count`` is what the verification gate passes, unvoted
    (#4505).  ``None`` when no balance is set (a library caller's choice; the
    app always sets one).
    """
    from vtscore.training.thresholds import balance_state, gate_balance_state

    if beta is None:
        return None
    voted = human_voted_ids(ctx)
    if ctx.line_ranking is None and ctx.gate_passed is not None:
        # The line is the verification gate's boundary: no ranking, no check.
        state = gate_balance_state(beta, ctx.gate_passed, voted).as_dict()
    else:
        # Wherever there is a head, the count is what its threshold keeps of the
        # ranking scored last - possibly none - not the count rule's: under the
        # labelset's line (#4452), the Goods' centroid's (#4643) or a typed
        # query's head's (#4758), and under a trained head with no class model,
        # whose line is the retrain's fallback cut (:func:`_recut_balance`).
        # Only a context with no head draws its line by the count rule.
        drawn = ctx.labels_line is not None or ctx.model is not None
        state = balance_state(
            beta,
            ctx.precision_check,
            ctx.line_ranking,
            voted,
            proposal=None if drawn else detector_balance_proposal(ctx, beta),
            threshold=ctx.threshold if drawn else None,
        ).as_dict()
    state.update(detector_check_prompt(ctx, len(voted)))
    return state


def detector_check_prompt(ctx: "DetectorContext", n_votes: int) -> dict[str, Any]:
    """Whether Autopilot should run a spot check on *ctx*'s line: mid-run for weak separation (#4496), or before Done.

    ``separation`` is the labels line's d' (``None`` before a retrain has drawn
    one, or when it is unbounded); ``check_due`` is :func:`weak_check_due` on
    it, never once a check has ended (prompted once, #4909), and
    ``done_check_due`` is :func:`done_check_due` from the same count: whether
    the check Autopilot runs once its lights go green should run, or skip
    because one ended moments ago.  Neither is due in Find, which runs no
    check (#4317), nor while a check is running, nor on a line a check cannot
    walk (#4489): no ranking - a structural detector's line is the
    verification gate's boundary - or nothing in it unvoted, where a check
    would be refused.  ``check_due`` is never due on face crops either
    (:func:`weak_check_prompts`, #4799): there the check's votes cost the
    session and find nothing.  ``separation`` is still reported.
    """
    from vtscore.training.thresholds import done_check_due, weak_check_due, weak_check_prompts  # noqa: PLC0415

    sep = ctx.labels_line.separation if ctx.labels_line is not None else None
    ranking = ctx.line_ranking
    checkable = (
        not ctx.find_mode
        and ctx.precision_check_run is None
        and ranking is not None
        and ranking.unvoted_ids(human_voted_ids(ctx)).size > 0
    )
    due = checkable and weak_check_prompts(ctx.media_type) and weak_check_due(sep, n_votes, ctx.check_ended_votes)
    done_due = checkable and done_check_due(n_votes, ctx.check_ended_votes)
    shown = round(float(sep), 4) if sep is not None and math.isfinite(sep) else None
    return {"separation": shown, "check_due": bool(due), "done_check_due": bool(done_due)}


def detector_acquisition_threshold(
    ctx: "DetectorContext", inclusion_value: float | None = None, *, beta: float | None = None
) -> float:
    """The cut Autopilot's ``hard`` / ``new`` picks should sample around.

    **Not the decision line.**  ``ctx.threshold`` is what the user sees and what
    Find calls a match; the picks read a threshold as a rank position rather
    than a boundary, so they sample at a second cut.  The first rule that
    answers, in order:

    * **The Goods' centroid or a typed query's head** samples at its cosines'
      midpoint, whatever line it shows (#4136, #4732).
    * **Under a balance** (*beta*) it is the score where the labels line's
      corpus posterior falls below
      :data:`~vtscore.training.thresholds.ACQUISITION_TARGET_PRECISION` (0.5:
      the items the line's model thinks are even odds; #3546), read off
      ``ctx.labels_line``.  Failing that it can be a rank: the score at
      :data:`~vtscore.training.thresholds.ACQUISITION_ARGMAX_FACTOR` of the
      depth of the mixture's F-beta argmax over the unvoted ranking
      (:func:`~vtscore.training.thresholds.acquisition_threshold`, #4409).
      That factor is ``None`` (the #4427 revert: the rank cut harvested the
      user's unvoted top and few fresh images cleared the threshold), so the
      balance falls through to the offset below; a mixture with no estimate
      does too.
    * **The offset**, the fallback: a cut taken from the fold-anchored
      estimator,
      :data:`~vtscore.training.thresholds.ACQUISITION_INCLUSION_OFFSET`
      inclusion steps below the reporting cut, so further *up* the ranking.
      Decoupling the two measured 4.5x the positives per 100 votes at lower
      cost - see ``docs/experiments/2026-08-07-acquisition-inclusion/REPORT.md``
      (PR #2876).

    The offset is relative to **the inclusion the reporting cut sits at**.
    *inclusion_value* names it when the caller knows it in those units (the
    Inclusion 0 cut of a detector with no balance).  ``None`` recovers it from ``ctx.threshold`` through
    :meth:`~vtscore.training.thresholds.FoldAnchoredCut.inclusion_for_threshold`,
    which is the path a cut chosen by another rule takes: under a balance the
    reporting cut is wherever the set the balance keeps ends, and acquisition
    samples four steps stricter than *that*.

    Derived on demand rather than stored beside ``ctx.threshold``: there are
    four places that write a threshold onto a detector context, and a second
    field would be one more thing for each of them to forget.  Re-cutting is
    arithmetic on already-fitted Gaussians, so it is cheap enough to do per
    request.

    Falls back to ``ctx.threshold`` when there is no fold-anchored estimator to
    re-cut (no haystack to fit one on, a degenerate fit that fell back to the
    schedule blend, which has no inclusion-aware form, or a structural detector
    whose threshold is not on the estimator's scale): the two jobs coincide
    there.
    """
    from vtscore.detectors.query_head import is_quota_head  # noqa: PLC0415

    if ctx.centroid_line is not None and is_quota_head(ctx.model):
        # The Goods' centroid (and a typed query's head, #4758) samples at its cosines'
        # midpoint whatever line it shows, as a typed query's sort does (#4136, #4732).
        return ctx.centroid_line.acquisition_threshold()
    if beta is not None:
        from vtscore.training.thresholds import (
            ACQUISITION_TARGET_PRECISION,
            acquisition_threshold,
            target_precision_threshold,
        )

        # #3546: sample where the labels line's own corpus posterior says even
        # odds.  The line - 4 re-cut below had saturated into a rank pin under
        # the balance; this follows the class model instead.
        if ACQUISITION_TARGET_PRECISION is not None and ctx.labels_line is not None:
            at_target = target_precision_threshold(ctx.labels_line, ACQUISITION_TARGET_PRECISION)
            if at_target is not None and math.isfinite(at_target):
                return at_target

        at_argmax = acquisition_threshold(ctx.line_ranking, beta, detector_line_labels(ctx), human_voted_ids(ctx))
        if at_argmax is not None:
            return at_argmax
    cut = ctx.anchored_cut_cache
    if cut is None:
        return ctx.threshold
    from vtscore.training.thresholds import acquisition_inclusion

    reporting = inclusion_value if inclusion_value is not None else cut.inclusion_for_threshold(ctx.threshold)
    # A line with no balance is drawn at the fallback inclusion; the caller
    # names it (see ``detector_line_inclusion``) so acquisition samples exactly
    # its offset below it rather than below the strict end of whatever plateau
    # that cut shares.
    if reporting is None:
        return ctx.threshold
    candidate = float(cut.threshold_at(acquisition_inclusion(reporting)))
    return candidate if math.isfinite(candidate) else ctx.threshold


def detector_line_inclusion(ctx: "DetectorContext", beta: float | None) -> float | None:
    """The inclusion *ctx*'s reporting line sits at, for Autopilot's acquisition offset.

    Under a balance (#4413) the line keeps a set rather than an inclusion,
    so it is ``None`` and :func:`detector_acquisition_threshold`
    recovers it from the line itself (owner, 2026-09-28: acquisition sits at
    *X - 4*, where *X* is the derived inclusion of the production cut).  With
    no preference it is Inclusion 0, the line's own cut (#4269).
    """
    from vtscore.training.thresholds import PRECISION_FLOOR_FALLBACK_INCLUSION

    if beta is not None:
        return None
    return float(PRECISION_FLOOR_FALLBACK_INCLUSION)


def recompute_detector_thresholds(beta: float | None) -> None:
    """Re-derive each loaded detector's threshold at its own balance, leaving the head in place.

    The balance is a pure cutoff knob: a change must not drop the model or
    re-score the haystack - only move the threshold over already-computed
    scores (:func:`recut_detector_threshold`).  Detectors with nothing cached
    to re-cut are left untouched; the next training pass computes the
    threshold at the new balance.

    **Each detector is re-cut at its own balance.**  The balance is per
    detector (``DetectorContext.beta``, seeded on first read from the balance
    the detector keeps on its JSON, else the user's setting; see #3416,
    #4665), and :func:`vtscore.state.set_beta` writes only the active
    detector's.  A detector that has not read its own balance yet is cut at
    the one it keeps, else at *beta* - the user's setting, which is what that
    first read will seed it with.  A detector already holding its own balance
    keeps its cut.  With no balance at all (*beta* ``None`` and none seeded
    or kept, a library caller's choice) the line is the Inclusion 0 cut
    (#4269).
    """
    from vtscore.detectors.balance import detector_stored_beta
    from vtscore.training.thresholds import PRECISION_FLOOR_FALLBACK_INCLUSION

    contexts = loaded_detector_contexts()
    # Read outside the lock: what an unseeded detector keeps is a read of its JSON.
    kept = {id(ctx): detector_stored_beta(ctx.detector_id) for ctx in contexts if not ctx.beta_seeded}
    with _state_lock:
        for ctx in contexts:
            if ctx.beta_seeded:
                own = ctx.beta
            else:
                stored = kept.get(id(ctx))
                own = beta if stored is None else stored
            threshold = recut_detector_threshold(ctx, PRECISION_FLOOR_FALLBACK_INCLUSION, beta=own)
            if threshold is not None:
                ctx.threshold = threshold


def user_beta() -> float | None:
    """The user's balance: what a detector that has not read its own yet will be seeded with.

    The ``beta`` setting: ``None`` - no balance - only when it is set so
    (``CoreConfig(beta=None)``), and :data:`~vtscore.config.DEFAULT_BETA` in a
    library-only process that never configured one.
    """
    from vtscore.config import get_setting

    beta = get_setting("beta")
    return None if beta is None else float(beta)


def recompute_detector_thresholds_for_inclusion(inclusion_value: int) -> None:
    """Deprecated: accepts only ``0``, and re-cuts every loaded detector at its balance.

    Inclusion is no longer a user preference (#4269), so there is no Inclusion
    change to follow.  ``0`` is accepted with a ``DeprecationWarning`` and
    delegates to :func:`recompute_detector_thresholds`; any other value raises
    ``ValueError``.
    """
    from vtscore.config.core_config import _retired_inclusion

    _retired_inclusion("vtscore.state.core.recompute_detector_thresholds_for_inclusion()", inclusion_value)
    recompute_detector_thresholds(user_beta())
