"""The Smart indicator's arithmetic, shared by the app and the eval harness.

Smart asks one question: *is the detector still getting better?*  It is
answered by re-scoring the last :data:`SMART_WINDOW` cached models against the
current labelset and looking at the trend of their F-beta, at the session's
balance: green once it stops rising (issue #5134).  This module holds every
constant and every line of that arithmetic, and both consumers call it rather
than copying it:

* :func:`vtscore.detectors.labeling_progress._compute_smart_status` (the app's
  ``/api/labeling-status`` indicator, which gates Autopilot's ``hard`` phase),
* :func:`vtscore.eval.autopilot_flow.smart_status` (the harness's simulated
  user, which has to stop where the app stops for a study to mean anything).

What a point is (issue #5134)
-----------------------------

Each point is one cached model's **best F-beta on the current votes**
(:func:`best_fbeta`): its scores over every voted media, every cut through them,
the best F-beta at the session's beta (:data:`SMART_NO_BALANCE_BETA` for a
caller with no balance).  Cut-free, so it reads the ranking alone and cannot
move when the served line switches rule.  Until #5134 a point was the model's
``FPR + FNR`` at its own Inclusion 0 cut, green once it stopped *falling*: the
last place the app priced a detector off the objective.  The A/B that retired it
(``docs/experiments/2026-10-20-smart-fbeta-5134``) found the two lights green at
the same clicks; the owner chose F-beta so that every light speaks the
objective's units.  The cost rule (:func:`smart_status_from_costs`,
:func:`smart_cut`) stays as the eval's ``cost`` arm.

Two conditions, not one, and the second is the one from issue #3832
----------------------------------------------------------------------

Green needs the trend to be **flat**, and the rule has always measured that as
a least-squares slope normalised by the window's mean, green below
:data:`SMART_RISE_THRESHOLD` (the cost rule: above :data:`SMART_FLAT_THRESHOLD`).
A rise shallower than 1.5% of the mean per step is not worth another click.
The rest of this section was written for the cost; it holds for F-beta with
every sign turned over.

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

What a cost was priced at (issue #4243; the eval's ``cost`` arm since #5134)
--------------------------------------------------------------------------

Every cost in the window is ``FPR + FNR`` at :data:`SMART_INCLUSION`, measured
at each model's own cut for that inclusion (:func:`smart_cut`), whatever
Inclusion or precision floor the user has set.  Both callers take the weight and
the cut from here, so what Smart measures cannot depend on which tier asked.

The A/B's other readings (issue #5134)
--------------------------------------

:func:`average_precision` and :func:`fbeta_at_cut` score the same window the
other ways #5134 priced (``vtscore.eval.autopilot_flow.SMART_SIGNALS``), read by
the same rising rule, :func:`rising_status_from_scores`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

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

#: The Inclusion every Smart cost is priced at, whatever line the user is shown:
#: ``inclusion_cost_weights(0)``, i.e. FPR + FNR, at each model's own cut for
#: this inclusion (:func:`smart_cut`), not at the line it was served with
#: (issue #4243).
#:
#: Smart asks whether the *detector* is still improving, and the reporting line
#: is not the detector.  Under a precision floor (#4224) the served line is no
#: inclusion's cut: it keeps a set (#4272), the floor's starting candidate
#: until a spot check ends and the set the check ended on after it, so it moves
#: with the floor and with every check.  Nothing clears the progress cache when
#: that happens, so a window scored at the served lines would mix models cut
#: different ways.  Moving off the Inclusion 0 cut raises FPR + FNR, and a
#: rising cost reads green, so the mix could end Autopilot's ``hard`` phase
#: early.  One weight and one cut rule for every model keep the window's points
#: comparable; the floor only moves the line.
SMART_INCLUSION = 0


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


def smart_cut(
    served_threshold: float,
    served_inclusion: float | None,
    recut: Callable[[float], float | None],
) -> float:
    """The threshold Smart scores one model at: its own cut at :data:`SMART_INCLUSION`.

    *served_threshold* is the line the model was served with, cut at
    *served_inclusion*.  *recut* re-derives the same model's cut at another
    inclusion from its fitted estimator; it returns ``None`` when there is no
    inclusion-aware cut to re-derive (a fallback or schedule-blend cut).  The
    served line is then the answer, because an inclusion-blind cut is the same
    at every inclusion.

    A model served at :data:`SMART_INCLUSION` is scored at its served line
    exactly, without a re-cut, so a user at the default Inclusion sees the
    indicator they always had.  *served_inclusion* is ``None`` when no
    inclusion drew the served line - a precision floor that promised its own
    cut (#4245) - and the model is then always re-cut.  The app passes its re-cut seam
    (``vtscore.state.core.recut_detector_threshold``) and the eval harness its
    step's fold-anchored fit; the rule for choosing between them lives here, so
    the two cannot drift.
    """
    if served_inclusion is not None and served_inclusion == SMART_INCLUSION:
        return served_threshold
    cut = recut(SMART_INCLUSION)
    if cut is None or not math.isfinite(cut):
        return served_threshold
    return float(cut)


#: The flatness cutoff with its sign turned over, for a score that should rise
#: (issue #5134): a window whose least-squares slope gains less than 1.5% of its
#: mean per model has stopped going up.
SMART_RISE_THRESHOLD = -SMART_FLAT_THRESHOLD

#: The beta Smart scores at for a caller with no balance (the app always sets
#: one, #4413): F1.
SMART_NO_BALANCE_BETA = 1.0


def smart_beta(beta: float | None) -> float:
    """The beta Smart's F-beta is read at: the session's balance, or :data:`SMART_NO_BALANCE_BETA` with none."""
    return SMART_NO_BALANCE_BETA if beta is None else float(beta)


def rising_status_from_scores(scores: Sequence[float], good: int, bad: int) -> dict[str, Any]:
    """The Smart indicator over one window of F-beta points, oldest first (issue #5134).

    The app's rule since #5134, on :func:`best_fbeta` points; the eval's other
    readings (#5134's AP and cut F-beta) go through it too.  The window,
    per-class minimum and trend-and-scatter test of :func:`smart_status_from_costs`
    (#3832) with the sign turned over: green once the window's relative slope
    is below :data:`SMART_RISE_THRESHOLD`, or its rise is within
    :data:`SMART_SLOPE_T` standard errors of nothing.  ``drift_within_noise``
    marks the second kind of green.  Non-finite entries (a model that could not
    be scored) are dropped before the window is cut.
    """
    if good < MIN_PER_CLASS or bad < MIN_PER_CLASS:
        return {
            "status": "red",
            "reason": f"Need at least {MIN_PER_CLASS} good and {MIN_PER_CLASS} bad. Currently {good}g, {bad}b.",
        }

    window = [float(s) for s in scores if math.isfinite(float(s))][-SMART_WINDOW:]
    if len(window) < SMART_MIN_POINTS:
        return {"status": "yellow", "reason": "Not enough valid model steps in recent history to assess trend."}

    trend = cost_trend(window)
    extras: dict[str, Any] = {
        "slope": round(trend.relative_slope, 4),
        "slope_t": round(trend.t_stat, 2) if math.isfinite(trend.t_stat) else trend.t_stat,
        "drift_within_noise": False,
    }
    if trend.relative_slope <= SMART_RISE_THRESHOLD:
        return {"status": "green", "reason": "F-beta has stopped rising. You can likely stop labeling.", **extras}
    if trend.t_stat < SMART_SLOPE_T:
        extras["drift_within_noise"] = True
        return {
            "status": "green",
            "reason": (
                "F-beta has stopped rising: it drifts up, but by less than it bounces around "
                "between retrains. You can likely stop labeling."
            ),
            **extras,
        }
    return {"status": "yellow", "reason": "F-beta is still rising. Keep labeling.", **extras}


def _scored(scores: Any, labels: Any) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel() >= 0.5
    n = min(s.size, y.size)
    s, y = s[:n], y[:n]
    ok = np.isfinite(s)
    return s[ok], y[ok]


def average_precision(scores: Any, labels: Any) -> float:
    """Average precision of *scores* against *labels* (``1.0`` Good); NaN without both classes.

    Best first, ties broken by position (a stable sort): the mean, over the
    Goods, of the precision of the set down to each one.
    """
    s, y = _scored(scores, labels)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return math.nan
    order = np.argsort(-s, kind="stable")
    hits = y[order]
    tp = np.cumsum(hits)
    precision = tp / np.arange(1, hits.size + 1)
    return float(precision[hits].sum() / n_pos)


def fbeta_at_cut(scores: Any, labels: Any, cut: float, beta: float) -> float:
    """F-*beta* of the items at or above *cut* (the codebase's ``>=``); NaN without both classes."""
    s, y = _scored(scores, labels)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return math.nan
    kept = s >= cut
    tp = float((kept & y).sum())
    b2 = float(beta) ** 2
    return (1.0 + b2) * tp / (b2 * n_pos + float(kept.sum()))


def best_fbeta(scores: Any, labels: Any, beta: float) -> float:
    """The best F-*beta* any cut through *scores* reaches; NaN without both classes.

    Cut-free, so it reads the ranking alone: every set from the top down, at
    the session's balance.
    """
    s, y = _scored(scores, labels)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == y.size:
        return math.nan
    order = np.argsort(-s, kind="stable")
    tp = np.cumsum(y[order])
    k = np.arange(1, s.size + 1)
    # A cut falls only between distinct scores: keep the last item of each run of ties.
    sorted_s = s[order]
    ends = np.append(sorted_s[1:] != sorted_s[:-1], True)
    b2 = float(beta) ** 2
    f = (1.0 + b2) * tp[ends] / (b2 * n_pos + k[ends])
    return float(f.max())
