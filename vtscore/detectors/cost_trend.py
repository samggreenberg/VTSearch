"""The Smart indicator's arithmetic, shared by the app and the eval harness.

Smart asks one question: *is the detector still getting better?*  It is
answered by re-scoring the last :data:`SMART_WINDOW` cached models against the
current labelset and looking at the trend of their error costs.  This module
holds every constant and every line of that arithmetic, and both consumers call
it rather than copying it:

* :func:`vtscore.detectors.labeling_progress._compute_smart_status` (the app's
  ``/api/labeling-status`` indicator, which gates Autopilot's ``hard`` phase),
* :func:`vtscore.eval.autopilot_flow.smart_status` (the harness's simulated
  user, which has to stop where the app stops for a study to mean anything).

Two conditions, not one, and the second is the one from issue #3832
----------------------------------------------------------------------

Green needs the trend to be **flat**, and the rule has always measured that as
a least-squares slope normalised by the window's mean cost, green above
:data:`SMART_FLAT_THRESHOLD`.  That half is unchanged: a decline shallower than
1.5% of the mean per step is not worth another click.

What was missing is any notion of *how noisy the window is*.  On a category the
embedding cannot separate - the #3831 dice pool, where the label is a die's
roll and the embedding only knows its number of sides - the cost is flat on
average but jumps step to step, because every new label moves the labelset the
whole window is re-scored against.  A slope fitted through ten such points is
below the flatness threshold a large fraction of the time with nothing having
changed about the detector: measured on flat synthetic series, the slope test
alone reads "still falling" on 8.6% of windows at a coefficient of variation of
0.1, 24% at 0.2 and 36% at 0.4.  The light was reporting the noise, and
Autopilot's phase display bounced between Done and Boundary with it.

So a decline must also be **larger than the window's own scatter** before it
counts: the regression slope's t-statistic (slope over its standard error,
``n - 2`` degrees of freedom) must be at or below ``-``:data:`SMART_SLOPE_T`.
That makes the false-yellow rate a property of the *test* rather than of the
pool: about 4% at every noise scale, since it is the one-tailed tail area of
the t distribution at 8 degrees of freedom.  A genuine decline is still caught
- a run losing 5% of its cost per step reads yellow on 99% of windows at cv 0.1
- because a real trend shrinks the residuals it is measured against.  A
perfectly straight decline has no residuals at all and is infinitely
significant, which is why the synthetic ``[0.9, 0.7, 0.5, 0.3]`` window in the
tests stays yellow.

Hysteresis - "once green, stay green unless the slope falls below twice the
threshold" - was the other candidate in #3832 and is deliberately not what
happened here.  Both callers compute this status as a pure function of a cost
window (the app recomputes it from its step cache on every 2s poll, and
memoises on the *inputs*), so a rule that depended on the previous answer would
make the light depend on when the page was loaded and would have to be threaded
through the harness's per-step trajectory as well.  A noise model gets the same
flapping away without a hidden variable.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from vtscore.detectors.stability import MIN_PER_CLASS

Status = Literal["red", "yellow", "green"]

#: How many of the most recent models' error costs the trend is fitted over,
#: and the fewest points that make a trend mean anything.  The app windows its
#: last :data:`SMART_WINDOW` *models* rather than label steps, because most of
#: its steps never trained one.
SMART_WINDOW = 10
SMART_MIN_POINTS = 3

#: Relative-slope cutoff: the least-squares slope divided by the window's mean
#: cost.  Below this the cost is falling fast enough to be worth more labels.
SMART_FLAT_THRESHOLD = -0.015

#: How many standard errors below zero the slope must sit before that decline
#: is believed.  2.0 is a one-tailed tail area of about 4% at the 8 degrees of
#: freedom a full window has - see the module docstring for why the rule needs
#: a noise model at all (issue #3832).
SMART_SLOPE_T = 2.0


@dataclass(frozen=True)
class CostTrend:
    """The fitted trend of one error-cost window.

    *relative_slope* is the least-squares slope normalised by the window's mean
    cost (so it reads as "fraction of the cost lost per step"), and *t_stat*
    that slope in units of its own standard error.  ``t_stat`` is ``-inf`` for
    a perfectly straight decline and ``+inf`` for a perfectly straight rise:
    a line through the points with no residual is as significant as a trend
    gets, and the sign keeps the comparison in
    :func:`smart_status_from_costs` honest either way.
    """

    relative_slope: float
    t_stat: float


def cost_trend(costs: Sequence[float]) -> CostTrend:
    """Fit :data:`CostTrend` over *costs*, oldest first.

    Needs at least three points; :func:`smart_status_from_costs` is what
    enforces that, so callers that want the raw numbers should check first.
    """
    n_pts = len(costs)
    x_vals = list(range(n_pts))
    x_mean = sum(x_vals) / n_pts
    y_mean = sum(costs) / n_pts

    numer = sum((x_vals[i] - x_mean) * (costs[i] - y_mean) for i in range(n_pts))
    denom = sum((x - x_mean) ** 2 for x in x_vals)
    slope = numer / denom if denom != 0 else 0.0
    relative_slope = slope / y_mean if y_mean > 0 else slope

    # Residual standard error of the slope: how much of the window's spread the
    # fitted line does *not* explain, per unit of x spread.
    resid_sq = sum((costs[i] - (y_mean + slope * (x_vals[i] - x_mean))) ** 2 for i in range(n_pts))
    if n_pts > 2 and denom > 0 and resid_sq > 0:
        std_err = math.sqrt((resid_sq / (n_pts - 2)) / denom)
    else:
        std_err = 0.0
    # A dead straight window leaves a residual of rounding error rather than of
    # exactly zero, so "no doubt at all" is a scale-relative test: below this
    # the standard error is float noise, not scatter the costs actually have.
    if std_err > 1e-9 * max(abs(y_mean), 1e-12):
        t_stat = slope / std_err
    else:
        t_stat = 0.0 if slope == 0 else math.copysign(math.inf, slope)

    return CostTrend(relative_slope=relative_slope, t_stat=t_stat)


def smart_status_from_costs(costs: Sequence[float], good: int, bad: int) -> dict[str, Any]:
    """The Smart indicator over one window of error costs, oldest first.

    Returns the ``/api/labeling-status`` sub-object: ``status`` (red / yellow /
    green) and ``reason``, plus ``slope`` (the relative slope), ``slope_t``
    (that slope in standard errors) and ``drift_within_noise`` once there is
    enough history to fit them.  ``drift_within_noise`` is true only on green,
    and means the cost does drift down but by less than the window's own
    scatter - the UI says so rather than claiming the run has converged.

    Only the last :data:`SMART_WINDOW` costs are read, so a caller may hand in
    a longer history.
    """
    if good < MIN_PER_CLASS or bad < MIN_PER_CLASS:
        return {
            "status": "red",
            "reason": f"Need at least {MIN_PER_CLASS} good and {MIN_PER_CLASS} bad. Currently {good}g, {bad}b.",
        }

    window = [float(c) for c in costs][-SMART_WINDOW:]
    if len(window) < SMART_MIN_POINTS:
        return {"status": "yellow", "reason": "Not enough valid model steps in recent history to assess trend."}

    trend = cost_trend(window)
    extras: dict[str, Any] = {
        "slope": round(trend.relative_slope, 4),
        "slope_t": round(trend.t_stat, 2) if math.isfinite(trend.t_stat) else trend.t_stat,
        "drift_within_noise": False,
    }

    if trend.relative_slope >= SMART_FLAT_THRESHOLD:
        return {"status": "green", "reason": "Error cost has leveled off. You can likely stop labeling.", **extras}
    if trend.t_stat > -SMART_SLOPE_T:
        # Falling on the fit, but by less than the window's own step-to-step
        # scatter: the trend is not distinguishable from noise, and calling it
        # a decline is what flapped the light on a plateaued category (#3832).
        # ``drift_within_noise`` is that distinction, reported so the UI can say
        # which kind of green this is without re-deriving the rule.
        extras["drift_within_noise"] = True
        return {
            "status": "green",
            "reason": (
                "Error cost has leveled off: it drifts down, but by less than it bounces around "
                "between retrains. You can likely stop labeling."
            ),
            **extras,
        }
    return {"status": "yellow", "reason": "Error cost is still declining. Keep labeling.", **extras}
