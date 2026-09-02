"""The retired safe-threshold blend, and the thin shim over the blend schedules.

Combines a cross-calibration cut and a GMM cut under a mix-in schedule
(:mod:`vtscore.training.blend_schedules`).  No longer the shipped safe
threshold - :func:`~vtscore.training.thresholds.anchored.fold_anchored_gmm_threshold`
is - but it survives as that path's fallback for label counts too small to form
folds at all, and for the harness arms that still measure it.
"""

from __future__ import annotations

import math

from vtscore.training.blend_schedules import BlendContext, BlendSchedule, get_schedule
from vtscore.training.thresholds.gmm import GmmFit1D, fit_gmm_threshold
from vtscore.utils.scores import scored_only


def calculate_safe_threshold(
    xcal_threshold: float,
    all_scores: list[float],
    ctx: "BlendContext | int",
    schedule: "str | BlendSchedule | None" = None,
) -> float:
    """Combine the cross-calibration and GMM thresholds under a mix-in schedule.

    When few labels are available the cross-calibration threshold can be
    unreliable, so a GMM cut fitted on the full score distribution stands in for
    it.  *How much* it stands in - and for how long - is the schedule's job
    (:mod:`vtscore.training.blend_schedules`, issue #2841).  Passing ``None``
    takes :data:`~vtscore.training.blend_schedules.PRODUCTION_SCHEDULE`
    (``cap50``); callers that know the voting mode should resolve the name
    through :func:`~vtscore.training.blend_schedules.production_schedule_for`
    instead, because the two modes want different curves.

    **No longer the shipped safe threshold.**  The 2026-08-05 population-
    anchored run (docs/experiments/2026-08-05-population-anchored-calibration/REPORT.md)
    retired the schedule in favour of *fusing* the two estimators rather than
    averaging them as rivals: :func:`fold_anchored_gmm_threshold` is the
    production path now.  This blend survives as its fallback, for the label
    counts too small to form calibration folds at all (where it degenerates to
    the pure GMM cut anyway) and for the harness arms that still measure it.

    Args:
        xcal_threshold: The cross-calibrated threshold.
        all_scores: Model output scores for all medias (used for GMM fitting).
        ctx: A :class:`~vtscore.training.blend_schedules.BlendContext` carrying
            the vote counts.  A bare ``int`` is accepted as the total label
            count for callers that have no class breakdown; schedules that ramp
            on the rarer class then see a degenerate split and are not
            meaningful, so pass a real context wherever the labels are known.
        schedule: Registry name or instance; ``None`` selects production.

    Returns:
        A finite threshold float. If either candidate is non-finite, falls back
        to the other; if both are, returns ``0.5``.  The result is guaranteed
        finite so it can be safely stored on ``DetectorContext.threshold``
        without breaking ``score >= threshold`` comparisons.

    Media the head could not score are dropped from *all_scores* before the
    GMM sees them (:func:`~vtscore.utils.scores.scored_only`); they are
    recorded a unit below the sigmoid range and would otherwise pull a fitted
    component - and the blend with it - under zero (issue #3180).  When that
    leaves *no* population at all, there is nothing for the GMM to stand in
    for and the x-cal cut ships alone rather than being blended against
    :func:`fit_gmm_threshold`'s "too few scores" 0.5.
    """
    population = scored_only(all_scores)
    if population.size == 0:
        return xcal_threshold if math.isfinite(xcal_threshold) else 0.5
    cut, fit = fit_gmm_threshold(population.tolist())
    return blend_gmm_threshold(xcal_threshold, cut, ctx, schedule=schedule, fit=fit)


def _as_context(ctx: "BlendContext | int") -> BlendContext:
    """Normalise the ``BlendContext | int`` argument into a context.

    An ``int`` carries no class breakdown, so the split is left degenerate
    rather than guessed: schedules reading ``n_good``/``n_rare`` would be
    fabricating an answer, and a caller that wants them must supply real counts.
    """
    if isinstance(ctx, BlendContext):
        return ctx
    return BlendContext(n_labels=int(ctx), n_good=0, n_bad=int(ctx))


def safe_blend_weight(ctx: "BlendContext | int", schedule: "str | BlendSchedule | None" = None) -> float:
    """The weight the schedule puts on the **x-cal** cut; 0 means pure GMM."""
    sched = schedule if isinstance(schedule, BlendSchedule) else get_schedule(schedule)
    return sched.weight(_as_context(ctx))


def blend_gmm_threshold(
    xcal_threshold: float,
    gmm_threshold: float,
    ctx: "BlendContext | int",
    schedule: "str | BlendSchedule | None" = None,
    fit: GmmFit1D | None = None,
) -> float:
    """Combine a pre-computed x-cal and GMM cut under *schedule*.

    The combining core of :func:`calculate_safe_threshold`, split out so a
    caller with a pre-computed GMM cut (the #2799/#2841 measurement harness
    re-cuts one fitted GMM under several rules and schedules) applies the
    identical schedule and finite-guards without re-fitting.  *fit* is the
    :class:`GmmFit1D` behind *gmm_threshold* when the caller has it; schedules
    that need the component geometry (the corridors) fall back to a plain
    weighted blend without it.
    """
    # Defend against non-finite inputs from either side: an upstream
    # ``calculate_cross_calibration_threshold`` can theoretically still
    # surface inf/NaN, and ``calculate_gmm_threshold`` returns NaN when
    # the model produced non-finite scores. Without these guards the blend
    # below would store NaN on ``DetectorContext.threshold`` and silently
    # break every ``score >= threshold`` comparison downstream.
    xcal_finite = math.isfinite(xcal_threshold)
    gmm_finite = math.isfinite(gmm_threshold)
    if not xcal_finite and not gmm_finite:
        return 0.5
    if not xcal_finite:
        return gmm_threshold
    if not gmm_finite:
        return xcal_threshold

    sched = schedule if isinstance(schedule, BlendSchedule) else get_schedule(schedule)
    blended = sched.combine(xcal_threshold, gmm_threshold, _as_context(ctx), fit)
    if not math.isfinite(blended):
        return 0.5
    return blended
