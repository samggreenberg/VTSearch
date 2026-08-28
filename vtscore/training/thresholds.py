"""Decision-threshold computation for learned-sort scores.

GMM-, cross-calibration-, and safe-threshold helpers. These are media-
agnostic: they take score lists and label lists and return a single
float threshold. Detector-specific glue (sourcing ``X_list`` / ``y_list``
from votes, caching on ``DetectorContext``) lives in
:mod:`vtscore.detectors`.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from vtscore.training.blend_schedules import BlendContext, BlendSchedule, get_schedule
from vtscore.utils.scores import scored_mask, scored_only

# Sentinel threshold meaning "predict nothing as Good". Sigmoid scores are
# in [0, 1], so any value > 1.0 makes every ``score >= threshold`` check
# evaluate to False. Kept finite (vs. ``float("inf")``) so it cannot poison
# downstream blends - ``0.0 * inf`` evaluates to NaN, which would then be
# stored on ``DetectorContext.threshold`` and break every comparison.
NO_GOOD_THRESHOLD = 2.0

# Seed of the Train/Calibrate fold splits when a caller passes no ``rng``.
# The splits must be reproducible: the same labelset has to yield the same fold
# models, hence the same conformal threshold, hence the same Good/Bad verdicts
# on every run and every reload of a detector.  Falling back to the *global*
# ``np.random`` would break that twice over - the splits would differ run to
# run, and the request/background threads that run Find scoring would mutate
# shared global RNG state, making results order-dependent under concurrency.
CALIBRATION_SPLIT_SEED = 42

# Rows sampled from the training matrix when seeding the split-size dither (see
# :func:`_split_dither_rng`).  Small on purpose: this only has to separate two
# labelsets of equal size, not summarise them.
_DITHER_SAMPLE_ROWS = 32

# False-negative budget of the conformal inclusion rule at inclusion 0; each
# +1 step of inclusion halves it (see :func:`conformal_threshold`).  0.25 means
# the default cutoff may sacrifice at most ~25% of true matches to the
# false-positive guard - a cap, spent only when class overlap forces it.
CONFORMAL_BASE_BUDGET = 0.25

# Positive-score quantile the inclusion = -10 end of the knob walks to: at -10
# only the region scoring above the 75th percentile of held-out positives is
# included - "just the most confident matches".
CONFORMAL_QPOS_MAX = 0.75

# Bounds of the Inclusion knob.  Every threshold rule keyed on inclusion is
# defined over this closed range, and every sweep of it (the UI slider's stops,
# the Find Stats chart) runs over exactly these values.
INCLUSION_MIN = -10
INCLUSION_MAX = 10

# Above this many scores, fit the GMM on a random subsample instead of the full
# set. A 2-component, 1-D GMM only needs to recover the two clusters' means and
# variances, which 50k samples estimate as accurately as the full population -
# so the threshold is statistically indistinguishable while the EM fit stays
# O(50k) instead of O(N). This matters because ``calculate_gmm_threshold`` runs
# on the *full* score distribution on every cosine/text sort (sorting.py) and in
# the safe-threshold blend, where N reaches ~250k (GUI Find) to 2M+ (CLI Find).
_GMM_MAX_SAMPLES = 50_000

#: The shipped Train/Calibrate split of each calibration fold, per the **space
#: the detector learns in** (issue #3287 measured them separately; see
#: ``docs/experiments/calibration-fraction-3287/REPORT.md``).  The value is the
#: **Calibrate** share, so ``0.3`` means 70% Train / 30% Calibrate.
#:
#: * ``single_vector`` - one embedding vector per media.  Spending more votes
#:   on fitting each fold's model and fewer on reading its threshold is worth
#:   −0.012 to −0.013 ± 0.003 in cost on both single-vector embedders
#:   measured, winning in every vote band; the gain is largest when votes are
#:   scarce and decays toward 150 clicks.
#: * ``patch`` - a patch-grid embedder (whatever style it currently votes in).
#:   Nothing measured beats the incumbent 0.5 here, and 0.3 is +0.015 ± 0.005
#:   *worse* on ``dinov3_patch/whole_image`` - which is why the key is the
#:   embedder's capability rather than the voting mode: the same row-wise
#:   calibrator wants opposite splits on ``siglip/whole`` vs ``dinov3/whole``,
#:   while both ``dinov3`` styles agree on 0.5.
PRODUCTION_SPLIT_BY_SPACE: dict[str, float] = {
    "single_vector": 0.3,
    "patch": 0.5,
}

#: Fallback when the space is unknown.  0.5 is the incumbent and the
#: never-harmful choice: it is not significantly worse than any arm measured,
#: on any geometry.
PRODUCTION_SPLIT = 0.5


def production_split_for(*, patch_space: bool | None) -> float:
    """The shipped ``calibration_fraction`` for the space a detector learns in.

    *patch_space* says whether the detector's embedder produces a patch grid
    (its capability, not what it is doing in the current configuration -
    ``dinov3_patch`` wants 0.5 in both its styles, including the boxless
    fallback that emits no patches at all).  ``None`` means "unknown", which
    takes :data:`PRODUCTION_SPLIT` rather than guessing - the same three-state
    contract as :func:`vtscore.training.blend_schedules.production_schedule_for`.

    An explicit user setting always wins over this table; callers resolve that
    precedence via
    :func:`vtscore.detectors.training.resolve_calibration_fraction`.
    """
    if patch_space is None:
        return PRODUCTION_SPLIT
    return PRODUCTION_SPLIT_BY_SPACE["patch" if patch_space else "single_vector"]


def classify_threshold_provenance(fallback: float | None) -> str:
    """Name the code path a trained threshold came from, from its *fallback*.

    :func:`compute_fold_orderings` returns a ``fallback`` that fully discriminates
    which branch produced the threshold: ``None`` means the conformal quantile
    rule ran on real fold orderings; :data:`NO_GOOD_THRESHOLD` (2.0) means the
    "no valid Train/Calibrate split" sentinel; ``0.5`` means a too-few-labels
    early return.  Used by the calibration study (issue #2781) to attribute the
    runaway-threshold bug; the safe-threshold GMM blend is a separate caller and
    is tagged ``"gmm_blend"`` at that site, not here.
    """
    if fallback is None:
        return "conformal"
    if fallback == NO_GOOD_THRESHOLD:
        return "no_good_sentinel"
    if fallback == 0.5:
        return "too_few_default"
    return "unknown"


def inclusion_cost_weights(inclusion_value: int) -> tuple[float, float]:
    """``(fpr_weight, fnr_weight)`` - the rate loss the Inclusion knob names.

    Inclusion is defined as a trade-off between the two error *rates*:
    ``cost = fpr_weight * FPR + fnr_weight * FNR``.  Each ``+1`` step doubles
    the price of a miss (matching :func:`conformal_threshold`'s halving
    false-negative budget) and each ``-1`` step doubles the price of a false
    alarm, so the knob means the same thing to every rule that reads it - the
    conformal quantile, the rate-optimal GMM cut
    (:meth:`GmmFit1D.rate_crossing`), and the eval harness's scoring.

    This is the single definition; :mod:`vtscore.eval.calibration_metrics` and
    :mod:`vtscore.eval.voting_iterations` delegate here so a measured arm and
    the shipped path can never disagree about what an inclusion value costs.
    """
    if inclusion_value >= 0:
        return 1.0, 2.0**inclusion_value
    return 2.0 ** (-inclusion_value), 1.0


#: How far *below* the reporting inclusion the **acquisition** cut sits.
#:
#: The threshold does two unrelated jobs.  Reporting is the decision line the
#: user sees and every metric is scored at.  Acquisition is what Autopilot's
#: ``hard`` and ``new`` picks consume - and those read the threshold as a **rank
#: position** in the descending ranking, not as a decision boundary, so they want
#: the opposite thing from it.
#:
#: The direction is therefore the opposite of the intuition from the cost
#: weights: a *negative* offset prices false alarms higher, *raises* the cut,
#: moves it *up* the ranking, and so returns *more* positives.
#:
#: ``-1`` is the only value that passes the pre-registered ship rule in **all
#: **two** binary environments measured, and this constant is deliberately **not**
#: gated by voting mode.  The history is worth keeping, because the first answer
#: was bigger and did not survive:
#:
#: * ``coco_val x siglip2`` (binary, PR #2876) found an interior optimum at
#:   ``-3``: positives per 100 votes 4 -> 18, final cost 0.137 -> 0.129 (95% CI
#:   [-0.025, -0.005]), average precision 0.696 -> 0.817.  #2878 shipped it.
#: * ``visual_genome_m x siglip`` (binary, PR #2891) **rejected** ``-3``: cost CI
#:   [+0.003, +0.022] against a +0.01 tolerance.  Only ``-1`` passed.
#:
#: So the disagreement runs along the *environment*, not the voting mode: the
#: largest split (``-3`` ships on COCO, fails on VG) is **within** binary voting,
#: which no mode gate can reach - and that leg alone is what sets this value.
#: ``-1`` is the value with no measured harm in either environment.  Do not raise
#: it without a further environment; do not gate it by mode without evidence that
#: mode - and not label supply - is the axis.
#:
#: **The region-voting check is still OUTSTANDING.**  It was run (PR #2909) and
#: its result is **void**: that run predates #2943, which fixed the harness
#: scoring the acquisition pool by each media's whole-image vector while cutting
#: the threshold on region max-pooled scores.  On a patch dataset that put the
#: cut above the entire pool - pinned on 39% of ``k=-3`` steps against 1.5% of
#: the ``k=+2`` falsifier - so the aggressive arms were clamped and the lever was
#: partly inert exactly where the decision needed it live.  The two binary
#: environments are unaffected (``patch_grid`` on 0/4193, so they scored and cut
#: in one space).  Read the banner on
#: ``docs/experiments/acquisition-inclusion/REPORT_REGION_VOTING.md`` before
#: citing anything from that run, and re-run it before concluding anything about
#: voting mode.
#:
#: **The known cost of this conservatism**: on a starved COCO-like environment
#: ``-1`` finds 6 positives per 100 votes where ``-3`` finds 18.  Under binary
#: voting the benefit is sharply concentrated in *starved* cells and turns
#: negative in well-supplied ones (measured on arm-independent axes: AP response
#: slope -0.0207 on log prevalence, CI [-0.0259, -0.0159]).  A **supply-dependent**
#: offset - aggressive while positives are scarce, relaxing as they accumulate -
#: is the way to recover COCO's gain without charging the other environments'
#: tails, and it subsumes the voting-mode question entirely (#2910).
#:
#: See ``docs/experiments/acquisition-inclusion/REPORT.md`` (COCO) and
#: ``REPORT_SECOND_ENVIRONMENT.md`` (VG binary) for the two live environments,
#: and ``REPORT_REGION_VOTING.md`` for the voided region run and how to redo it.
ACQUISITION_INCLUSION_OFFSET = -1


def acquisition_inclusion(inclusion_value: int, offset: int = ACQUISITION_INCLUSION_OFFSET) -> int:
    """The inclusion the **selector's** cut is taken at, given the reporting one.

    One definition, shared by the app and the eval harness, so a measured arm
    and the shipped path cannot disagree about where acquisition samples - the
    same discipline :func:`inclusion_cost_weights` follows.  *offset* exists for
    the harness's arms; production always takes the default.

    An *offset*, not an absolute value.  The run that measured ``-3`` held
    reporting at inclusion 0, where the two readings coincide; away from 0 only
    the offset preserves what was measured, because the mechanism is the *gap*
    between where the line is drawn and where sampling happens.  Reading ``-3``
    absolutely would collapse the gap to nothing at reporting inclusion -3 and
    invert it below that - the direction the ``acq_p2`` arm falsified.

    Deliberately unclamped.  The reporting inclusion is clamped to ``[-10, 10]``
    at the API edge, so this can reach -13; the cost weights are exponential but
    finite there, and :meth:`FoldAnchoredCut.threshold_at` clamps the quantile it
    realizes to ``[0, 1]`` anyway.  Clamping here would instead silently switch
    the mechanism off at the bottom of the slider, which is the failure mode that
    is hard to notice.
    """
    return inclusion_value + offset


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of ``a*x^2 + b*x + c``, degenerating gracefully to the linear case.

    Uses the cancellation-free ("citardauq") pairing ``q = -(b + sign(b)*sqrt(D))/2``,
    ``x = {q/a, c/q}`` rather than the textbook formula.  That matters here because
    the near-equal-variance case drives ``a`` toward 0, where ``(-b + sqrt(D)) /
    (2a)`` is catastrophic cancellation over a vanishing denominator while ``c/q``
    stays accurate and converges smoothly to the linear root ``-c/b``.
    """
    if a == 0.0:
        return [] if b == 0.0 else [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    q = -0.5 * (b + math.copysign(math.sqrt(disc), b))
    if q == 0.0:
        # Only reachable with b == 0 and disc == 0, i.e. ``a*x^2 = 0``.
        return [0.0]
    return [q / a, c / q]


def _weighted_gaussian_crossing(
    w_lo: float,
    mu_lo: float,
    var_lo: float,
    w_hi: float,
    mu_hi: float,
    var_hi: float,
    *,
    lam: float = 1.0,
) -> float | None:
    """Score between the two means where the weighted component densities cross.

    Solves ``w_lo * N(x; mu_lo, var_lo) == lam * w_hi * N(x; mu_hi, var_hi)``.
    Taking logs makes the difference a quadratic ``f(x) = a x^2 + b x + c``
    (``f > 0`` means the Bad component owns that score), so the crossing is a
    root of that quadratic - at ``lam == 1``, the Bayes decision boundary between
    the two fitted components **with the mixture weights as class priors**, i.e.
    the cut that minimises expected misclassification *count*.

    *lam* tilts that boundary, and which value is correct depends on the loss
    (issue #2836).  Minimising a weighted sum of **rates** - ``fpr_weight * FPR +
    fnr_weight * FNR``, what the Inclusion knob is defined in terms of and what
    this repo scores - instead puts the cut where ``fnr_weight * f_pos ==
    fpr_weight * f_neg``, which carries no priors at all; that is
    ``lam = (fnr_weight / fpr_weight) * (w_lo / w_hi)``, dividing the prior-odds
    factor back out.  See :meth:`GmmFit1D.rate_crossing`.

    **Not the shipped cut.**  Production cuts at the midpoint between the means
    (:func:`calculate_gmm_threshold`); this solver stays live only as an eval
    variant (``*_cross`` in :data:`vtscore.eval.voting_iterations._SAFE_GMM_VARIANTS`).
    #2798 shipped it on the geometry argument below and #2799 measured it as a
    small net loss (+0.0036 cost at 6-20 votes, +0.0059 at 2-5), so #2833 reverted
    it.  The geometry argument was right in *direction* - the crossing does sit
    above the midpoint under max-pooling - but the exchange rate is unfavourable:
    it buys ~1 FPR for ~1.3 FNR, and for a needle-finding tool the missed positive
    is the worse error.  #2836 is the open question of why (leading hypothesis:
    we score a *rate* loss, so the prior-odds term in this crossing is the bias).

    The crossing and the midpoint agree **exactly** when the components are
    equal-weight and equal-variance.  They diverge precisely where region voting
    lives: a media's score is the max over ~24 region nodes, so the Bad mode is an
    extreme-value statistic - wider, right-skewed, and far heavier than the Good
    mode.  A wider/heavier low component pushes the crossing *above* the midpoint
    (with equal variances the offset is ``var * ln(w_lo/w_hi) / (mu_hi - mu_lo)``).

    Returns ``None`` - meaning "the caller should fall back to the midpoint" -
    whenever the crossing is not a well-defined boundary: non-positive weights or
    variances, non-ordered/degenerate means, a complex-root fit, no root strictly
    between the means (near-equal variances with an extreme weight ratio push the
    linear root outside the interval), or a fit in which the Bad component still
    out-densities the Good one at the Good mean.  When two roots land inside the
    interval the larger one is taken: above it the Good component dominates all
    the way to its own mean, which is the boundary a threshold wants.
    """
    if not (w_lo > 0.0 and w_hi > 0.0 and var_lo > 0.0 and var_hi > 0.0 and lam > 0.0):
        return None
    if not (mu_hi > mu_lo):
        return None

    # Solve in ``u = x - mu_lo`` so the interval is ``(0, d)``.  Shifting keeps
    # the roots exact while dropping the ``mu^2 / var`` terms that would dominate
    # the coefficients (and their cancellation) for score scales far from zero.
    d = mu_hi - mu_lo
    offset = math.log(w_lo / (lam * w_hi)) + 0.5 * math.log(var_hi / var_lo)
    a = 0.5 / var_hi - 0.5 / var_lo
    b = -d / var_hi
    c = 0.5 * d * d / var_hi + offset

    # The Good mode must actually be Good-dominated, else "the score above which
    # Good wins" is not something this fit expresses.  Evaluated in closed form
    # rather than as ``a d^2 + b d + c`` (the same value, without the cancellation).
    if offset - 0.5 * d * d / var_lo >= 0.0:
        return None

    inside = [u for u in _quadratic_roots(a, b, c) if math.isfinite(u) and 0.0 < u < d]
    if not inside:
        return None
    return mu_lo + max(inside)


#: ``cut_fallback_kind`` when the rule found an interior stationary point, i.e.
#: nothing was substituted or continued and the cut *is* the root.
CUT_KIND_INTERIOR: str = ""
#: ``cut_fallback_kind`` when the crossing ran off the inter-mean interval and
#: the cut was continued past that edge at the rule's own first-order slope.
#: The cut still moves with the cost tilt; it is simply no longer a stationary
#: point of the rate loss.
CUT_KIND_CONTINUED: str = "continued"
#: ``cut_fallback_kind`` when the fit is too degenerate to express a boundary at
#: all (non-positive weights/variances, non-ordered means) and the rule returned
#: the plain midpoint.  Distinct from :data:`CUT_KIND_CONTINUED` because the cut
#: is then *constant* in the cost tilt - the failure mode issue #2896 removed
#: everywhere it could be removed.
CUT_KIND_DEGENERATE_MIDPOINT: str = "degenerate_midpoint"


def _rate_cut(
    w_lo: float,
    mu_lo: float,
    var_lo: float,
    w_hi: float,
    mu_hi: float,
    var_hi: float,
    *,
    lam: float,
) -> tuple[float, str]:
    """The rate-optimal cut: a **sup** over the inter-mean interval, continued
    past the edges at the rule's own first-order slope so it is **strictly**
    monotone in *lam* everywhere.

    ``(cut, kind)``, where *kind* is one of :data:`CUT_KIND_INTERIOR`,
    :data:`CUT_KIND_CONTINUED` or :data:`CUT_KIND_DEGENERATE_MIDPOINT` - empty
    exactly when an interior stationary point existed, so ``bool(kind)`` is the
    "no interior stationary point" flag, and the non-empty values say *how* the
    cut was produced instead.  The distinction matters to anything auditing the
    rule: a continued cut still answers the Inclusion knob, a degenerate
    midpoint does not (issue #2900).  Inside the interval the cut is

        ``sup { x in [mu_lo, mu_hi] : w_lo*N_lo(x) >= lam*w_hi*N_hi(x) }``

    - the highest score at which the Bad component still out-densities the Good
    one under the cost tilt.  Raising *lam* (pricing misses higher, i.e.
    raising Inclusion) shrinks that set pointwise, so the sup can only fall:
    the rule is **monotone in the cost ratio by construction**, which is what
    the Inclusion knob's nesting contract needs.  Where the densities genuinely
    cross inside the interval this returns exactly
    :func:`_weighted_gaussian_crossing`'s root, so the shipped cut is the
    stationary point of the rate loss wherever one exists.  Picking the
    interval's *midpoint* when no root exists instead - the obvious-looking
    fallback - is what broke monotonicity: with a Good component wider than the
    Bad one the root enters and leaves the interval non-monotonically, and a
    midpoint fallback let a *more* exclusive inclusion cut *lower* than a less
    exclusive one.

    **Past the edges the cut keeps moving** (issue #2896).  Returning the bare
    edge once the crossing runs off the interval - the previous behaviour -
    made the cut *constant* in *lam* there, and that flat step propagated all
    the way up: the composed ``mid_tilt`` quantile plateaued over whole bands
    of the Inclusion slider, and the acquisition offset
    (:data:`ACQUISITION_INCLUSION_OFFSET`), which lives entirely inside such a
    band whenever the tilt saturates, silently collapsed to a no-op - Autopilot
    degraded to sampling at the reporting line with nothing surfacing it.  So
    when Bad still out-densities Good at ``mu_hi`` the cut continues *above*
    the Good mean, and when Good owns the whole interval it continues *below*
    the Bad mean, each by the log-cost excess beyond the edge times ``var/d``
    (the mixture-weighted variance over the mean gap) - the exact slope of the
    equal-variance crossing (:meth:`GmmFit1D.equal_var_offset`), so for
    equal-variance fits the continuation extends the interior crossing line
    *seamlessly*, and for unequal variances it is the rule's own first-order
    slope.  The continuation is strictly decreasing in ``ln(lam)`` and stays on
    the far side of its edge, so overall monotonicity is preserved; the only
    plateau left downstream is the honest one, where the cut runs off the end
    of the haystack's support and the empirical quantile pins at 0 or 1.

    Returns the midpoint (flagged) for a fit too degenerate to express a
    boundary at all: non-positive weights/variances or non-ordered means.
    """
    mid = (mu_lo + mu_hi) / 2.0
    if not (w_lo > 0.0 and w_hi > 0.0 and var_lo > 0.0 and var_hi > 0.0 and lam > 0.0):
        return mid, CUT_KIND_DEGENERATE_MIDPOINT
    if not (mu_hi > mu_lo):
        return mid, CUT_KIND_DEGENERATE_MIDPOINT

    # Same shifted quadratic as ``_weighted_gaussian_crossing`` (u = x - mu_lo,
    # interval (0, d)): ``g(u) > 0`` means the Bad component owns that score.
    d = mu_hi - mu_lo
    offset = math.log(w_lo / (lam * w_hi)) + 0.5 * math.log(var_hi / var_lo)
    a = 0.5 / var_hi - 0.5 / var_lo
    b = -d / var_hi
    c = 0.5 * d * d / var_hi + offset

    # Slope of the out-of-interval continuation: the equal-variance crossing
    # moves by var/d per nat of log-cost, evaluated at the mixture-weighted
    # variance (the same variance ``equal_var_offset`` uses).
    slope = (w_lo * var_lo + w_hi * var_hi) / d

    # g(d), in closed form (the same value as ``a d^2 + b d + c``, without the
    # cancellation): Bad still ahead at the Good mean means the whole interval
    # belongs to Bad.  The excess is the log-cost margin by which it is still
    # ahead - 0 exactly when the crossing sits at ``mu_hi``, growing linearly
    # as ``lam`` falls - so the continuation leaves the edge without a step.
    excess = offset - 0.5 * d * d / var_lo
    if excess >= 0.0:
        return mu_hi + slope * excess, CUT_KIND_CONTINUED

    inside = [u for u in _quadratic_roots(a, b, c) if math.isfinite(u) and 0.0 <= u < d]
    if not inside:
        # g < 0 across the whole interval: Good owns it from mu_lo up.  Here
        # ``c = g(0) <= 0`` (a positive g(0) with g(d) < 0 forces a root
        # inside), and ``-c`` is the log-cost margin by which Good is ahead at
        # the Bad mean - the mirror-image continuation below ``mu_lo``.
        return mu_lo - slope * max(0.0, -c), CUT_KIND_CONTINUED
    return mu_lo + max(inside), CUT_KIND_INTERIOR


@dataclass(frozen=True)
class GmmFit1D:
    """The two components of a fitted 1-D, 2-component GMM, ordered by mean.

    Carries exactly the parameters the two candidate cut rules need, so one EM
    fit can be re-cut under both rules (the safe-threshold measurement study,
    issue #2799, and its #2836 follow-up) instead of re-fitting per rule.  ``lo``
    is the Bad (low-mean) component, ``hi`` the Good one.
    """

    w_lo: float
    mu_lo: float
    var_lo: float
    w_hi: float
    mu_hi: float
    var_hi: float

    def midpoint(self) -> float:
        """The production cut: the midpoint between the two component means."""
        return (self.mu_lo + self.mu_hi) / 2.0

    def crossing(self, lam: float = 1.0) -> float | None:
        """Root of ``w_lo*N_lo(x) == lam*w_hi*N_hi(x)``; ``None`` when undefined."""
        return _weighted_gaussian_crossing(
            self.w_lo, self.mu_lo, self.var_lo, self.w_hi, self.mu_hi, self.var_hi, lam=lam
        )

    def crossing_or_midpoint(self) -> float:
        """Eval-only cut (#2798, reverted by #2833): equal-density crossing, midpoint when none exists."""
        crossing = self.crossing()
        return self.midpoint() if crossing is None else crossing

    def rate_crossing(self, fpr_weight: float = 1.0, fnr_weight: float = 1.0) -> float | None:
        """The cut that minimises ``fpr_weight*FPR + fnr_weight*FNR`` (issue #2836).

        A weighted sum of **rates** normalises each error by its own class, so it
        is prevalence-free by construction - that is what makes the Inclusion knob
        portable across datasets.  Differentiating it gives the stationarity
        condition ``fnr_weight * f_pos(x) == fpr_weight * f_neg(x)``: the
        **prior-free** density crossing, with no mixture weights in it.  Under the
        identification ``f_neg = N_lo``, ``f_pos = N_hi`` that is
        :meth:`crossing` at ``lam = (fnr_weight/fpr_weight) * (w_lo/w_hi)``,
        i.e. exactly :meth:`crossing_or_midpoint`'s ``lam = 1`` rule with the
        prior-odds factor divided back out.

        With equal variances this lands at ``midpoint() +
        var*ln(fpr_weight/fnr_weight)/(mu_hi-mu_lo)`` - so at equal cost weights
        it *is* the midpoint-of-means, which is why the historical heuristic is
        better than it looks.  With unequal variances the two separate, and the
        gap is the part of this rule the midpoint cannot express.
        """
        if not (fpr_weight > 0.0 and fnr_weight > 0.0 and self.w_hi > 0.0):
            return None
        return self.crossing(lam=(fnr_weight / fpr_weight) * (self.w_lo / self.w_hi))

    def equal_var_offset(self, lam: float = 1.0) -> float:
        """Closed-form ``crossing(lam) - midpoint()`` **if** the variances were equal.

        Evaluates ``var * ln(w_lo/(lam*w_hi)) / (mu_hi - mu_lo)`` at the
        mixture-weighted variance.  Exact when ``var_lo == var_hi``; elsewhere it
        is the first-order prediction issue #2836 checks the realised offset
        against, and the size of the prior-odds bias it attributes to ``lam = 1``.
        """
        d = self.mu_hi - self.mu_lo
        if not (d > 0.0 and self.w_hi > 0.0 and lam > 0.0):
            return float("nan")
        var = self.w_lo * self.var_lo + self.w_hi * self.var_hi
        return var * math.log(self.w_lo / (lam * self.w_hi)) / d


def gmm_fit_array(scores: "list[float] | np.ndarray") -> np.ndarray:
    """The (possibly subsampled) float64 array a score-GMM is fitted on.

    Above :data:`_GMM_MAX_SAMPLES` scores, takes a deterministic (seed-42)
    random subsample; below, returns the scores unchanged.  Exposed separately
    from :func:`fit_score_gmm` so a caller that needs the fit's *input* too
    (e.g. for the median fallback, or to transform the same sample into logit
    space) subsamples exactly once.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape[0] > _GMM_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=_GMM_MAX_SAMPLES, replace=False)
    return arr


def scored_ordering(
    ordering: tuple[list[float], list[float]],
) -> tuple[list[float], list[float]]:
    """A fold's ``(scores, labels)`` ordering with the unscored items dropped.

    A held-out item whose fold model produced a non-finite logit is recorded at
    :data:`~vtscore.utils.scores.NON_FINITE_SCORE_SENTINEL` (``-1.0``) - the
    orderings are cached and swept, so they cannot hold ``NaN``.  As a
    *calibration anchor* that sentinel is worse than useless: the rules here
    read the anchors as positions on the sigmoid scale, and a labelled item
    sitting a unit below the scale drags both the conformal quantile and the
    anchored mixture down with it.  The item carries no information about where
    the cut belongs, so it is dropped - together with its label, which is why
    this filters the pair rather than the score list alone.
    """
    scores, labels = ordering
    keep = scored_mask(scores)
    if bool(keep.all()):
        return scores, labels
    return (
        [s for s, ok in zip(scores, keep.tolist(), strict=True) if ok],
        [lb for lb, ok in zip(labels, keep.tolist(), strict=True) if ok],
    )


def snap_cut_to_sample(cut: float, sorted_scores: np.ndarray) -> float:
    """Canonicalise *cut* to the empty interval of *sorted_scores* it falls in.

    A threshold that lands strictly between two adjacent observed scores is
    **unidentifiable**: every value in that open interval admits exactly the
    same items, so which one a fit happens to produce carries no information -
    only float noise.  This maps the whole interval to its midpoint, so the
    returned cut is a function of the two bracketing *data* values rather than
    of the last bits of an EM fit.

    That is the fix for issue #3166.  On a **saturated** score distribution -
    positives near 1.0, negatives near 0, nothing in between - the interval is
    enormous, and the un-canonicalised cut slides freely across it: the
    fold-anchored estimator realises its combined quantile with
    ``np.quantile``, which interpolates *linearly* between adjacent order
    statistics, so a difference of ``dq`` in the quantile moves the threshold by
    ``dq * (n - 1) * gap``.  With ``n`` ~ 9k and a unit-wide gap that is a gain
    of ~9000x: a sub-part-per-million wobble in the quantile - well within what
    differing BLAS kernels or thread counts produce between two machines -
    became the 0.026 threshold difference issue #3166 measured between two runs
    that agreed bit-for-bit on every other column.  Snapping removes the gain
    entirely: the threshold cannot move until the quantile crosses a whole
    order statistic, at which point the admitted set really has changed.

    **Decision-exact by construction.**  The snap only ever moves a cut inside
    an interval containing no observed score, so ``score >= threshold`` gives
    the identical verdict on every element of *sorted_scores*.  A cut that
    coincides with an observed score is already canonical and is returned
    unchanged - moving it would flip that score's (and its duplicates') verdict,
    which is the one case where the exact value does carry information.

    **Applied only where the chain has gain**, i.e. at
    :meth:`FoldAnchoredCut.threshold_at`, whose ``np.quantile`` interpolation is
    the amplifier.  :func:`calculate_gmm_threshold` and :func:`fit_gmm_threshold`
    deliberately do *not* snap: their cut is ``fit.midpoint()``, a smooth
    function of the fitted means, so an ulp of wobble in the fit buys an ulp of
    wobble in the cut and there is nothing to amplify.  Snapping them would also
    cost more than it bought - it would break the ``calculate_gmm_threshold(s) ==
    fit_score_gmm(gmm_fit_array(s)).midpoint()`` recomposition identity the eval
    harness relies on to re-cut one fit and reproduce the app, and above
    :data:`_GMM_MAX_SAMPLES` it would snap against a *subsample*, whose empty
    intervals are not empty in the full population.  ``threshold_at`` has
    neither problem: it snaps against exactly the array its own quantile was
    realised on.

    Args:
        cut: The candidate threshold.
        sorted_scores: The sample the threshold will be applied to, **sorted
            ascending**.  Empty or non-finite inputs are passed through.

    Returns:
        The canonical representative of *cut*'s admitted set.  Out-of-support
        cuts (below every score, or above every score) have no bracketing pair
        to snap to and are returned unchanged.
    """
    if sorted_scores.size == 0 or not math.isfinite(cut):
        return float(cut)
    i = int(np.searchsorted(sorted_scores, cut, side="left"))
    # ``i == 0`` / ``i == size``: the cut sits off the end of the support, so
    # there is no empty *interval* bracketing it - only an unbounded ray, which
    # has no midpoint.  ``sorted_scores[i] == cut``: the cut is exactly on an
    # observed score, where the value is identifiable (it decides that score's
    # own verdict) and must be left alone.
    if i == 0 or i == sorted_scores.size:
        return float(cut)
    if float(sorted_scores[i]) == cut:
        return float(cut)
    return (float(sorted_scores[i - 1]) + float(sorted_scores[i])) / 2.0


def fit_score_gmm(arr: np.ndarray) -> GmmFit1D | None:
    """Fit a deterministic 2-component GMM to a 1-D score array.

    Returns ``None`` when the fit fails (fewer than 2 scores, or an EM
    failure), leaving the fallback policy to the caller -
    :func:`calculate_gmm_threshold` falls back to the median.
    """
    if arr.shape[0] < 2:
        return None

    from sklearn.mixture import GaussianMixture  # noqa: PLC0415

    try:
        gmm: GaussianMixture = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(arr.reshape(-1, 1))

        # The stubs type these ``np.ndarray | None``; all are set after ``fit``.
        assert gmm.means_ is not None
        assert gmm.covariances_ is not None
        assert gmm.weights_ is not None
        means = np.ravel(gmm.means_)
        # ``covariances_`` is (n_components, 1, 1) under the default "full"
        # covariance type; ravel gives the two scalar variances.
        variances = np.ravel(gmm.covariances_)
        weights = np.ravel(gmm.weights_)

        low_idx = 0 if means[0] < means[1] else 1
        high_idx = 1 - low_idx
        return GmmFit1D(
            w_lo=float(weights[low_idx]),
            mu_lo=float(means[low_idx]),
            var_lo=float(variances[low_idx]),
            w_hi=float(weights[high_idx]),
            mu_hi=float(means[high_idx]),
            var_hi=float(variances[high_idx]),
        )
    except Exception:
        return None


# --- Anchored (semi-supervised) mixture estimation: issue #2852 ---------------
#
# The label-anchored mixture fits the same 2-component 1-D Gaussian mixture as
# ``fit_score_gmm`` but on a *partially labelled* sample: the haystack scores
# are free, while each voted item's score has its component membership fixed by
# its label (Good -> high component, Bad -> low component).  This is classical
# semi-supervised ML estimation for a mixture: EM where the E-step clamps the
# labelled points' responsibilities to one-hot and the M-step counts each
# labelled point ``anchor_weight`` times.  ``anchor_weight`` is therefore the
# single fusion knob: the anchors' M-step mass is ``anchor_weight * n_labels``
# against the haystack's ``N``, so with few labels the population dominates
# (the GMM regime) and as labels accumulate the labelled class-conditionals
# take over smoothly - the schedule the safe-blend hand-tunes, derived instead
# from relative likelihood mass.

#: Default per-anchor multiplicity for the anchored EM: each labelled score
#: counts as this many haystack scores in the M-step.  Chosen so a handful of
#: votes is already visible against the ``_GMM_MAX_SAMPLES``-sized haystack
#: sample without drowning it; the #2852 experiment sweeps this.
ANCHOR_WEIGHT_DEFAULT = 10.0

#: Relative variance floor for the anchored EM, as a fraction of the total
#: sample variance.  A component pinned to (near-)duplicate anchor scores would
#: otherwise collapse its variance to 0 and take the likelihood to infinity.
_ANCHOR_VAR_FLOOR_FRAC = 1e-6

#: Minimum mixture weight either anchored component may end with; below this
#: the fit has effectively deleted a component and the anchored path must fall
#: back to the unanchored fit.
_ANCHOR_MIN_WEIGHT = 1e-6


def _anchored_em(
    x: np.ndarray,
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    init: GmmFit1D,
    anchor_weight: float,
    max_iter: int,
    tol: float,
) -> GmmFit1D | None:
    """Run the anchored EM iterations; ``None`` on numerical failure.

    *x* is the free (unlabelled) sample, *a_lo* / *a_hi* the anchor scores
    clamped to the low / high component.  Pure numpy, log-domain E-step, fixed
    iteration order, and no BLAS calls (see the M-step comment) - deterministic
    for fixed inputs, and reproducible across machines with it.  Only
    numerical failure (non-finite parameters) returns ``None`` here; semantic
    degeneracy (inverted means, collapsed component) is judged by the caller so
    it can name the reason.
    """
    lam = float(anchor_weight)
    n = float(x.size)
    n_lo, n_hi = float(a_lo.size), float(a_hi.size)
    total_mass = n + lam * (n_lo + n_hi)

    w = np.array([init.w_lo, init.w_hi], dtype=np.float64)
    mu = np.array([init.mu_lo, init.mu_hi], dtype=np.float64)
    var = np.array([init.var_lo, init.var_hi], dtype=np.float64)

    pooled = np.concatenate([x, a_lo, a_hi])
    var_floor = max(1e-12, _ANCHOR_VAR_FLOOR_FRAC * float(np.var(pooled)))
    var = np.maximum(var, var_floor)

    sum_a_lo, sum_a_hi = float(a_lo.sum()), float(a_hi.sum())

    for _ in range(max_iter):
        # E-step over the free sample only (anchors are clamped one-hot).
        # Log-domain: log w_c - 0.5*log(2*pi*var_c) - (x-mu_c)^2 / (2*var_c).
        log_p = (
            np.log(np.maximum(w, 1e-300))[None, :]
            - 0.5 * np.log(2.0 * math.pi * var)[None, :]
            - (x[:, None] - mu[None, :]) ** 2 / (2.0 * var[None, :])
        )
        log_p -= log_p.max(axis=1, keepdims=True)
        r = np.exp(log_p)
        r /= r.sum(axis=1, keepdims=True)

        # M-step with the anchors folded in at weight ``lam`` each.
        m_lo = float(r[:, 0].sum()) + lam * n_lo
        m_hi = float(r[:, 1].sum()) + lam * n_hi
        if not (m_lo > 0.0 and m_hi > 0.0):
            return None
        # ``np.sum`` rather than ``@``: a dot product is dispatched to BLAS,
        # whose accumulation order depends on the kernel the CPU selects and on
        # the thread count, so the same input can differ in its last bits
        # between two machines.  numpy's own pairwise reduction does not
        # (issue #3166).  The extra temporaries are one ``x``-sized array each.
        mu_new = np.array(
            [
                (float(np.sum(r[:, 0] * x)) + lam * sum_a_lo) / m_lo,
                (float(np.sum(r[:, 1] * x)) + lam * sum_a_hi) / m_hi,
            ]
        )
        var_new = np.array(
            [
                (float(np.sum(r[:, 0] * (x - mu_new[0]) ** 2)) + lam * float(((a_lo - mu_new[0]) ** 2).sum())) / m_lo,
                (float(np.sum(r[:, 1] * (x - mu_new[1]) ** 2)) + lam * float(((a_hi - mu_new[1]) ** 2).sum())) / m_hi,
            ]
        )
        var_new = np.maximum(var_new, var_floor)
        w_new = np.array([m_lo, m_hi]) / total_mass

        if not (np.all(np.isfinite(mu_new)) and np.all(np.isfinite(var_new)) and np.all(np.isfinite(w_new))):
            return None
        delta = max(
            float(np.max(np.abs(mu_new - mu))),
            float(np.max(np.abs(var_new - var))),
            float(np.max(np.abs(w_new - w))),
        )
        mu, var, w = mu_new, var_new, w_new
        if delta < tol:
            break

    return GmmFit1D(
        w_lo=float(w[0]),
        mu_lo=float(mu[0]),
        var_lo=float(var[0]),
        w_hi=float(w[1]),
        mu_hi=float(mu[1]),
        var_hi=float(var[1]),
    )


def fit_anchored_score_gmm(
    arr: np.ndarray,
    anchor_scores: "list[float] | np.ndarray",
    anchor_labels: "list[float] | np.ndarray",
    *,
    anchor_weight: float = ANCHOR_WEIGHT_DEFAULT,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> tuple[GmmFit1D | None, str]:
    """Fit the label-anchored 2-component mixture (issue #2852).

    *arr* is the (possibly :func:`gmm_fit_array`-subsampled) haystack score
    sample; *anchor_scores* / *anchor_labels* are the voted items' scores and
    binary labels (1.0 Good -> high component, else low).  Initialised from the
    **unanchored** :func:`fit_score_gmm` fit (deterministic, seed-42 EM), then
    refined by anchored EM (see :func:`_anchored_em`).

    Returns ``(fit, provenance)``.  On success the provenance is ``"anchored"``
    and the fit's components are class-identified by construction (``hi`` is
    the Good-anchored component).  On failure the fit is ``None`` and the
    provenance names the reason (``"no_anchors"``, ``"too_few_scores"``,
    ``"unanchored_init_failed"``, ``"em_failed"``, ``"inverted_means"``,
    ``"component_collapse"``) - the caller decides the fallback policy
    (:func:`anchored_gmm_fit` falls back to the unanchored fit, never to 0.5).

    Anchors force a component ordering rather than inherit one: if the labelled
    scores contradict the population modes (Good votes living in the low mode),
    the anchored means invert or a component collapses, and that is reported as
    a degeneracy instead of silently shipping a backwards cut.
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    a = np.asarray(anchor_scores, dtype=np.float64).ravel()
    z = np.asarray(anchor_labels, dtype=np.float64).ravel()
    if x.size < 2:
        return None, "too_few_scores"
    if a.size == 0 or a.size != z.size:
        return None, "no_anchors"
    if not (anchor_weight > 0.0):
        return None, "no_anchors"

    init = fit_score_gmm(x)
    if init is None:
        return None, "unanchored_init_failed"

    a_hi = a[z == 1.0]
    a_lo = a[z != 1.0]
    fit = _anchored_em(x, a_lo, a_hi, init, anchor_weight, max_iter, tol)
    if fit is None:
        return None, "em_failed"
    if not (fit.mu_hi > fit.mu_lo):
        return None, "inverted_means"
    if fit.w_lo < _ANCHOR_MIN_WEIGHT or fit.w_hi < _ANCHOR_MIN_WEIGHT:
        return None, "component_collapse"
    return fit, "anchored"


def anchored_gmm_fit(
    all_scores: "list[float] | np.ndarray",
    anchor_scores: "list[float] | np.ndarray",
    anchor_labels: "list[float] | np.ndarray",
    *,
    anchor_weight: float = ANCHOR_WEIGHT_DEFAULT,
) -> tuple[GmmFit1D | None, str]:
    """Production-shaped anchored fit with the #2852 fallback policy applied.

    Subsamples via :func:`gmm_fit_array`, attempts the anchored fit, and on any
    anchored degeneracy falls back to the **unanchored** GMM fit of the same
    sample - never to 0.5.  Returns ``(fit, provenance)`` where provenance is
    ``"anchored"`` or ``"unanchored:<reason>"``; the fit is ``None`` only when
    the unanchored fit fails too (provenance ``"gmm_failed:<reason>"``, caller
    falls back to its median rule).
    """
    arr = gmm_fit_array(all_scores)
    fit, provenance = fit_anchored_score_gmm(arr, anchor_scores, anchor_labels, anchor_weight=anchor_weight)
    if fit is not None:
        return fit, provenance
    fallback = fit_score_gmm(arr)
    if fallback is not None:
        return fallback, f"unanchored:{provenance}"
    return None, f"gmm_failed:{provenance}"


def gmm_cut_from_fit(fit: GmmFit1D, rule: str, fpr_weight: float = 1.0, fnr_weight: float = 1.0) -> tuple[float, str]:
    """Apply a named cut *rule* to *fit*; ``(cut, kind)``.

    *kind* is the ``cut_fallback_kind`` vocabulary above - empty exactly when
    the rule found an interior stationary point, so ``bool(kind)`` is the "no
    interior stationary point" flag and the value names *how* the cut was
    produced otherwise.

    ``"mid"`` is the historical midpoint; ``"rate"`` is the rate-optimal
    crossing at the given cost weights (see :meth:`GmmFit1D.rate_crossing`).

    ``"rate"`` goes through :func:`_rate_cut`, which reads the cut as a
    **sup** rather than a bare root so it stays monotone in the cost ratio
    even on fits where the root enters and leaves the inter-mean interval, and
    continues past the interval edges at the rule's first-order slope so it
    never flattens (issue #2896); the kind marks the cases with no interior
    stationary point.  The value is the stationary point wherever one exists,
    so this is the rule the #2836 / #2852 measurements scored.

    ``"mid"`` never reports a kind: the midpoint of two means is defined for
    every fit, so it has no fallback branch to distinguish.

    ``"cross_tilt"`` is **eval-only** (issue #2865's candidate 2, as literally
    specified): the same solve at ``lam = fnr/fpr``, i.e. the Bayes
    misclassification-*count* boundary - mixture weights kept as class priors -
    tilted by the cost ratio.  It exists because ``"rate"``, despite the
    surrounding prose, does *not* read the mixture weights: the prior-odds
    factor in its ``lam`` cancels the ``w_lo/w_hi`` inside :func:`_rate_cut`'s
    ``offset`` exactly, leaving the cut invariant to the weights at every
    inclusion (only the out-of-interval continuation *slope* still averages the
    variances by them).  So "drop the mixture-weight factor from ``rate``" -
    #2865's stated candidate - is a no-op, and the rule that genuinely *does*
    read the acquisition-biased weights is this one.  Do not ship it without a
    measurement: retaining the prior odds is precisely what #2836 identified as
    the bias in the ``cross`` rule #2833 reverted.

    ``"mid_tilt"`` (the shipped fold-level rule, :data:`FOLD_ANCHOR_CUT_RULE`)
    and ``"q_tilt"`` (eval-only) are deliberately *not* accepted here: both are
    defined in fold-quantile space over a :class:`FoldAnchoredCut`'s combined
    folds (:meth:`FoldAnchoredCut._quantile_at`), so they have no per-fit,
    score-space form for this function to apply.
    """
    if rule == "mid":
        return fit.midpoint(), CUT_KIND_INTERIOR
    if rule in ("rate", "cross_tilt"):
        if not (fpr_weight > 0.0 and fnr_weight > 0.0 and fit.w_hi > 0.0):
            return fit.midpoint(), CUT_KIND_DEGENERATE_MIDPOINT
        lam = fnr_weight / fpr_weight
        if rule == "rate":
            lam *= fit.w_lo / fit.w_hi
        return _rate_cut(fit.w_lo, fit.mu_lo, fit.var_lo, fit.w_hi, fit.mu_hi, fit.var_hi, lam=lam)
    raise ValueError(f"unknown cut rule {rule!r}; expected 'mid', 'rate' or 'cross_tilt'")


def rank_transfer(
    cut: float, source_scores: "list[float] | np.ndarray", target_scores: "list[float] | np.ndarray"
) -> float:
    """Carry *cut* from one score distribution to another as a quantile.

    Reads the empirical quantile of *cut* in *source_scores* (the fraction of
    source scores strictly below it) and realizes that quantile on
    *target_scores* via linear interpolation.  A cut is a point on a score
    *scale*; two models scoring the same haystack are related by an
    approximately monotone map, and a quantile is invariant under any monotone
    map - so this is the scale-transfer step that lets a cut measured on a fold
    model's distribution be applied to the final model's (deficit 2 of
    ``docs/plans/population-anchored-calibration.md``).
    """
    src = np.sort(np.asarray(source_scores, dtype=np.float64).ravel())
    tgt = np.asarray(target_scores, dtype=np.float64).ravel()
    if src.size == 0 or tgt.size == 0:
        return float(cut)
    q = float(np.searchsorted(src, cut, side="left")) / float(src.size)
    return float(np.quantile(tgt, min(1.0, max(0.0, q))))


#: Production anchor mass for the fold-anchored threshold: **κ = 0.3**, i.e.
#: each vote counts as three tenths of a haystack point among the ~50k the
#: mixture is fitted on.  The 2026-08-05 deep-regime run swept κ ∈ {1 … 100}
#: and found performance degrading monotonically as κ grew, leaving the optimum
#: on the grid's bottom edge; the 2026-08-06 anchor-mass sweep extended the grid
#: to κ ∈ {0.01 … 3} across six environments and found the optimum **interior**
#: at κ=0.3 (docs/experiments/population-anchored-calibration/REPORT.md).
FOLD_ANCHOR_WEIGHT = 0.3

#: Production cut rule for the fold-anchored threshold: the **midpoint
#: anchored at inclusion 0, tilted by the rate rule** (issue #2865).
#:
#: The anchor-mass sweep picked the plain midpoint at κ=0.3, and the mechanism
#: is that ``mid`` ignores the mixture weights while ``rate`` reads them: at
#: light anchoring the weights carry the votes' acquisition-biased prevalence
#: out of proportion to their honesty, so the rule that never looks at them
#: wins.  But every arm of both calibration runs was scored at inclusion 0,
#: and a bare midpoint also ignores the *cost* weights the Inclusion knob
#: arrives as - shipping it verbatim made the knob a no-op for every detector
#: with usable folds.
#:
#: ``mid_tilt`` keeps the measured winner exactly where it was measured and
#: restores the knob everywhere else.  In fold-quantile space,
#: ``q(k) = q_mid + (q_rate(k) - q_rate(0))``: the midpoint's combined fold
#: quantile, shifted by however far the rate-optimal cut's own quantile moves
#: from its inclusion-0 position (see :meth:`FoldAnchoredCut._quantile_at`).
#: At inclusion 0 the shift is identically zero, so the threshold is
#: bit-for-bit the measured ``κ=0.3, mid`` arm; away from 0 it inherits
#: ``rate``'s monotone tilt without inheriting ``rate``'s weight-biased
#: *location*.
#:
#: **The tilt is measured** (issue #2865, 336 cells over four environments and
#: thirteen stops of the knob, on the shipped head; see
#: ``docs/experiments/inclusion-cut-rule/REPORT.md``).  It held: no candidate
#: both delivered more of the knob and stayed within the pre-registered 0.01
#: regret tolerance at every stop.  Two numbers worth keeping here - the ``mid``
#: cut this replaced admitted **one** set for the whole slider in every one of
#: 65,671 measured cell-steps and cost up to +0.18 regret away from inclusion 0;
#: and because ``mid_tilt`` differs from ``rate`` by the *constant*
#: ``q_mid - q_rate(0)`` in fold-quantile space, that sweep re-priced the
#: inclusion-0 choice under thirteen cost weightings and ``mid``'s location
#: survived all of them.
FOLD_ANCHOR_CUT_RULE = "mid_tilt"

#: Production fold-combine rule: mean of the per-fold quantiles.  With the
#: shipped ``calibrate_count=2`` the mean and the median coincide.
FOLD_ANCHOR_COMBINE = "qmean"

#: Minimum unlabeled-remainder size for the voted-media haystack exclusion
#: (issue #3308).  The fold-anchored estimator drops the voted media from its
#: haystacks - their scores under the models trained on them are optimistically
#: shifted - but only while the remainder is still big enough to *be* a
#: population estimate.  Below this many remaining scores the exclusion is
#: switched off entirely and the full (contaminated) haystack is used, because
#: two failure modes take over at once: the empirical quantiles the transfer
#: runs on lose resolution (1/n per order statistic), and after deep Autopilot
#: voting the leftover items are exactly the ones acquisition never found
#: interesting, so the remainder is a *selection-biased* sample of the corpus.
#: Measured on the overlapping-data eval environment (60-item sim set, votes
#: driven to exhaustion, 8 seeds paired step-for-step): exclusion is *neutral*
#: at remainder 50-56 (-0.004 +/- 0.007 direct cost, and only trajectory noise
#: downstream) and harmful below - +0.025 at remainder 40-49, +0.057 at 30-39,
#: +0.18 under 10 (the cut collapses onto a handful of drained leftovers,
#: FNR 0.7-0.9).  A 200-item synthetic single-vector environment is clearly
#: *positive* (-0.03 to -0.06) at every measured remainder >=60, even with 70%
#: of the corpus voted.  60 is therefore the smallest floor with a measured
#: win above it and nothing but measured neutrality or harm below it.  The
#: switch is all-or-nothing, never partial, so the fold haystacks and the
#: final realization sample always cover one identical population.
EXCLUSION_MIN_REMAINDER = 60

#: Step size of the **eval-only** ``"q_tilt"`` cut rule (issue #2865's candidate
#: 3), in units of combined-fold quantile per inclusion step.
#:
#: ``q_tilt`` decouples the Inclusion knob from the mixture entirely: it takes
#: the measured midpoint's combined fold quantile and shifts it by a fixed
#: amount per step of the knob, so the admitted fraction moves by construction
#: rather than by whatever the fitted Gaussians happen to imply.  That makes it
#: the simplest rule that cannot be inclusion-blind - and its price is this free
#: parameter, which has no principled value and must be *fitted*.
#:
#: **The step size has now been swept, and the rule lost at every value of it**
#: (issue #2865: {0.005, 0.01, 0.02, 0.04, 0.08} x four environments; see
#: ``docs/experiments/inclusion-cut-rule/REPORT.md``).  Small steps keep the
#: knob and cannot move far enough at large ``|k|``; large steps run the
#: quantile past 1.0 and admit nothing at the ends.  There is no value at which
#: ``q_tilt`` is not worse than the shipped ``mid_tilt``, so 0.02 remains what
#: it always was - an arbitrary constant on an arm that exists to be beaten -
#: and nothing should ship at it.
FOLD_ANCHOR_QTILT_STEP = 0.02


@dataclass(frozen=True, eq=False)
class FoldAnchoredCut:
    """A fitted fold-anchored ("cross-LabeledGMM") estimator, ready to re-cut.

    Holds everything the threshold depends on *except* inclusion: the per-fold
    anchored mixtures, each fold's sorted haystack sample (to read a cut's
    quantile in the scale it was measured on), and the final model's sorted
    haystack sample (to realize the combined quantile on the scale the
    threshold is applied on).

    Splitting the fit from the cut is what makes the Inclusion knob cheap
    *and* faithful under this estimator: re-cutting at another inclusion is
    arithmetic on the fitted Gaussians plus two array lookups - no EM, no
    scoring pass - so an Inclusion slide reproduces exactly what a fresh
    retrain at that inclusion would have stored (see
    :func:`vtscore.state.core.recompute_detector_thresholds_for_inclusion`).

    Under the shipped :data:`FOLD_ANCHOR_CUT_RULE` (``"mid_tilt"``) a re-cut
    answers the knob: inclusion 0 reproduces the measured midpoint cut
    bit-for-bit, and every other inclusion shifts the midpoint's combined
    quantile by the rate rule's own displacement from *its* inclusion-0
    position (issue #2865; see :meth:`_quantile_at`).
    """

    fits: tuple[GmmFit1D, ...]
    fold_haystacks: tuple[np.ndarray, ...]
    final_haystack: np.ndarray
    n_anchored: int
    cut_rule: str = FOLD_ANCHOR_CUT_RULE
    combine: str = FOLD_ANCHOR_COMBINE
    #: Only read by the eval-only ``"q_tilt"`` rule; see
    #: :data:`FOLD_ANCHOR_QTILT_STEP`.
    qtilt_step: float = FOLD_ANCHOR_QTILT_STEP

    @property
    def provenance(self) -> str:
        """``"fold_anchored[a/k]"`` - *a* of the *k* used folds fitted anchored."""
        return f"fold_anchored[{self.n_anchored}/{len(self.fits)}]"

    def _combined_fold_quantile(self, rule: str, fpr_weight: float, fnr_weight: float) -> float:
        """Combined fold quantile of *rule*'s per-fold cuts at these cost weights."""
        quantiles = []
        for fit, src in zip(self.fits, self.fold_haystacks, strict=True):
            cut, _kind = gmm_cut_from_fit(fit, rule, fpr_weight, fnr_weight)
            quantiles.append(float(np.searchsorted(src, cut, side="left")) / float(src.size))
        if self.combine == "qmean":
            return float(np.mean(quantiles))
        if self.combine == "qmedian":
            return float(np.median(quantiles))
        raise ValueError(f"unknown fold combine {self.combine!r}; expected 'qmean' or 'qmedian'")

    def _quantile_at(self, fpr_weight: float, fnr_weight: float) -> float:
        """Combined fold quantile at these cost weights, under ``cut_rule``.

        ``"mid"`` and ``"rate"`` are per-fit rules (:func:`gmm_cut_from_fit`)
        read straight through.  ``"mid_tilt"`` is composed here rather than per
        fit because it is defined in fold-quantile space:
        ``q = q_mid + (q_rate(weights) - q_rate(equal weights))``.  At equal
        cost weights - inclusion 0 - the parenthesised shift is *identically*
        zero (both terms are the same computation on the same fits), so the
        rule is bit-for-bit the plain midpoint exactly where the calibration
        runs measured it.  Elsewhere it moves the admitted fraction by however
        much the rate-optimal cut would have moved its own, so it inherits
        ``rate``'s monotonicity in the cost ratio without inheriting ``rate``'s
        weight-biased inclusion-0 location.  A fold too degenerate for a rate
        cut contributes a zero shift (its ``rate`` cut falls back to the
        midpoint at every weight), degrading that fold to plain ``mid`` rather
        than poisoning the tilt.

        ``"q_tilt"`` (**eval-only**, issue #2865's candidate 3) is the same
        shape with the mixture taken out of the tilt: ``q = q_mid - step *
        log2(fnr/fpr)``.  The log-cost ratio *is* the inclusion value for
        weights from :func:`inclusion_cost_weights`, so this reads "shift the
        admitted fraction by :attr:`qtilt_step` per step of the knob" - also
        identically ``q_mid`` at inclusion 0, also monotone, but moving the
        admitted set *by construction* rather than by whatever the fitted
        Gaussians happen to imply.  Its step size is a free parameter with no
        principled value (:data:`FOLD_ANCHOR_QTILT_STEP`), which is the whole of
        what it trades away for that guarantee.
        """
        if self.cut_rule == "mid_tilt":
            q_mid = self._combined_fold_quantile("mid", 1.0, 1.0)
            q_rate = self._combined_fold_quantile("rate", fpr_weight, fnr_weight)
            q_rate_zero = self._combined_fold_quantile("rate", *inclusion_cost_weights(0))
            return q_mid + (q_rate - q_rate_zero)
        if self.cut_rule == "q_tilt":
            q_mid = self._combined_fold_quantile("mid", 1.0, 1.0)
            return q_mid - self.qtilt_step * math.log2(fnr_weight / fpr_weight)
        return self._combined_fold_quantile(self.cut_rule, fpr_weight, fnr_weight)

    def quantile_at(self, inclusion_value: int) -> float:
        """The combined fold quantile this estimator admits at *inclusion_value*.

        :meth:`threshold_at` realizes this quantile on the final model's
        haystack; reading it directly separates "did the rule move the cut" from
        "did the haystack have anything there to move past", which is the
        distinction the #2865 sweep turns on - on a cleanly separated haystack a
        whole band of the knob can move the quantile while realizing the same
        threshold and the same admitted set.
        """
        return self._quantile_at(*inclusion_cost_weights(inclusion_value))

    def threshold_at(self, inclusion_value: int) -> float:
        """The threshold this estimator cuts at *inclusion_value*.

        Inclusion reaches the cut only as the rate weights it optimises
        (:func:`inclusion_cost_weights`).  Under the shipped ``"mid_tilt"``
        rule the midpoint's combined quantile is shifted by the rate rule's
        displacement from its own inclusion-0 position (see
        :meth:`_quantile_at`), so raising inclusion lowers the threshold and
        admits more - the same direction the conformal rule moves - while
        inclusion 0 remains exactly the measured midpoint cut.  ``"rate"``
        tilts the crossing itself; plain ``"mid"`` ignores the weights and is
        constant in inclusion.

        The realized value is finally canonicalised by
        :func:`snap_cut_to_sample`, so a cut landing between two adjacent
        haystack scores reports the midpoint of that empty interval rather than
        wherever ``np.quantile``'s linear interpolation put it.  That is
        decision-exact - the admitted set is identical either way - and it is
        what makes the threshold **reproducible**: without it, interpolating
        across an empty interval multiplies any wobble in *q* by
        ``(n - 1) * gap``, which on a saturated distribution turned a
        sub-part-per-million cross-machine difference into a visible threshold
        move (issue #3166).  The cost is that the threshold now steps rather
        than slides *within* one order-statistic band, which is honest: the
        admitted set was constant across that band all along.

        **Monotone by construction**, so the included sets stay nested
        (everything included at ``k`` stays included at ``k + 1``) - the
        contract that makes "cut off at Inclusion 1, verify up to Inclusion 4"
        well defined.  Every link in the chain is monotone: the per-fold cut in
        the cost weights (:func:`gmm_cut_from_fit`, continued past the
        inter-mean interval at its first-order slope rather than clamped, so
        the exits stay monotone *and strictly moving* - issue #2896), the
        empirical quantile of that cut in its fold's haystack, the mean/median
        across folds, the ``mid_tilt`` composition (a constant plus a
        monotone-in-inclusion shift), realizing a quantile on the final
        haystack, and the snap (which maps each order-statistic band to its own
        midpoint, so it is non-decreasing).  The plateaus in the chain are the
        honest ones - a band of the knob over which the *admitted set* does not
        change, including the boundary where a cut runs off its haystack's
        support and the quantile pins at 0 or 1 - crucially, the acquisition offset
        (:data:`ACQUISITION_INCLUSION_OFFSET`) therefore stays a real gap
        across the slider instead of silently collapsing wherever the tilt
        used to saturate.
        """
        if self.final_haystack.size == 0:
            return 0.5
        q = self._quantile_at(*inclusion_cost_weights(inclusion_value))
        realized = float(np.quantile(self.final_haystack, min(1.0, max(0.0, q))))
        return snap_cut_to_sample(realized, self.final_haystack)


def fit_fold_anchored_cut(
    fold_haystack_scores: "list[np.ndarray]",
    fold_anchor_orderings: list[tuple[list[float], list[float]]],
    final_scores: "list[float] | np.ndarray",
    *,
    anchor_weight: float = FOLD_ANCHOR_WEIGHT,
    cut_rule: str = FOLD_ANCHOR_CUT_RULE,
    combine: str = FOLD_ANCHOR_COMBINE,
) -> FoldAnchoredCut | None:
    """Fit the fold-anchored mixtures; ``None`` when no fold yielded a fit.

    Per calibration fold *k*: fit the anchored mixture on the **fold model's**
    haystack scores (``fold_haystack_scores[k]``) with anchors from that fold's
    *held-out* labelled scores (``fold_anchor_orderings[k]``, the same
    ``(scores, labels)`` orderings the conformal rule pools).  Anchors and
    population then share one scale and the anchors are honest - the labelled
    items were not in that fold model's training set, so their scores carry no
    train-set optimism.

    A fold whose anchored fit degenerates falls back to that fold's
    **unanchored** GMM fit; a fold that fails both is dropped.  ``None`` means
    every fold failed (or there were none), leaving the terminal fallback to
    the caller - :func:`fold_anchored_gmm_threshold` cuts the final model's own
    distribution instead, never 0.5.

    Every score that reaches a fit - haystack and anchor alike - is first put
    through :func:`~vtscore.utils.scores.scored_only`, so an item the head
    could not score (:data:`~vtscore.utils.scores.NON_FINITE_SCORE_SENTINEL`,
    ``-1.0``) is *absent* rather than treated as an observation a full unit
    below the sigmoid range.  Without that, a couple of unreadable media drag
    the fitted cut - and, through the quantile realised on the final haystack,
    the shipped threshold - below zero, where every real score clears it
    (issue #3180).
    """
    final_arr = gmm_fit_array(scored_only(final_scores))
    if final_arr.size == 0:
        return None
    fits: list[GmmFit1D] = []
    haystacks: list[np.ndarray] = []
    n_anchored = 0
    for hay, ordering in zip(fold_haystack_scores, fold_anchor_orderings, strict=True):
        a_scores, a_labels = scored_ordering(ordering)
        arr = gmm_fit_array(scored_only(hay))
        fit, provenance = fit_anchored_score_gmm(arr, a_scores, a_labels, anchor_weight=anchor_weight)
        if fit is None:
            fit = fit_score_gmm(arr)
            if fit is None:
                continue
        elif provenance == "anchored":
            n_anchored += 1
        fits.append(fit)
        haystacks.append(np.sort(arr))
    if not fits:
        return None
    return FoldAnchoredCut(
        fits=tuple(fits),
        fold_haystacks=tuple(haystacks),
        final_haystack=np.sort(final_arr),
        n_anchored=n_anchored,
        cut_rule=cut_rule,
        combine=combine,
    )


def fold_anchored_gmm_threshold(
    fold_haystack_scores: "list[np.ndarray]",
    fold_anchor_orderings: list[tuple[list[float], list[float]]],
    final_scores: "list[float] | np.ndarray",
    inclusion_value: int = 0,
    *,
    anchor_weight: float = FOLD_ANCHOR_WEIGHT,
    cut_rule: str = FOLD_ANCHOR_CUT_RULE,
    combine: str = FOLD_ANCHOR_COMBINE,
) -> tuple[float, str]:
    """The fold-anchored ("cross-LabeledGMM") mixture threshold (#2852 comment).

    **This is the shipped threshold path**: the 2026-08-06 anchor-mass sweep
    measured it at the defaults above as the best rule this harness has seen,
    and the best single global setting available - pooled over six environments
    it cuts −0.0437 paired regret vs pure cross-calibration in the deep regime,
    it beats the previously shipped ``κ=1, rate`` head to head in 6 of 6
    environments, and forcing it everywhere leaves each environment within
    0.0067 of its own optimum.  See
    ``docs/experiments/population-anchored-calibration/REPORT.md``.  The eval
    harness calls this same function for its default arm, so a measured
    baseline cannot drift from the app.

    Two limits of that recommendation are on record rather than fixed here.
    The gain tracks *positive*-anchor count, so on binary-voting detectors with
    few positives this only reaches a dead heat with the ``cap50`` blend it
    replaced (open work in ``docs/plans/population-anchored-calibration.md``).
    And every arm was scored at inclusion 0 - which the shipped ``mid_tilt``
    rule reproduces bit-for-bit while restoring the Inclusion tilt away from
    it; the tilt itself is the part issue #2865's inclusion sweep still owes a
    measurement.

    Fits via :func:`fit_fold_anchored_cut`, then carries each fold's cut to the
    final model as a quantile of that fold's haystack distribution
    (:func:`rank_transfer`'s argument: two models scoring the same haystack are
    related by an approximately monotone map, and quantiles are invariant under
    monotone maps), combines the folds in **quantile** space so no cross-scale
    averaging of raw cuts ever happens, and realizes the result on
    *final_scores*.

    Degeneracy policy per the issue: a fold whose anchored fit degenerates
    falls back to that fold's unanchored GMM fit; if every fold fails both
    fits, the final model's own unanchored GMM midpoint is returned (and its
    median if even that fails) - never 0.5.  Returns ``(threshold,
    provenance)`` with provenance ``"fold_anchored[a/k]"`` (*a* folds anchored
    of *k* used) or ``"fold_fallback_final_unanchored"`` /
    ``"fold_fallback_final_median"`` on the terminal fallbacks.
    """
    cut = fit_fold_anchored_cut(
        fold_haystack_scores,
        fold_anchor_orderings,
        final_scores,
        anchor_weight=anchor_weight,
        cut_rule=cut_rule,
        combine=combine,
    )
    if cut is not None:
        return cut.threshold_at(inclusion_value), cut.provenance

    final_arr = gmm_fit_array(final_scores)
    fallback_fit = fit_score_gmm(final_arr) if final_arr.size >= 2 else None
    if fallback_fit is not None:
        return fallback_fit.midpoint(), "fold_fallback_final_unanchored"
    if final_arr.size:
        return float(np.median(final_arr)), "fold_fallback_final_median"
    return 0.5, "fold_fallback_final_median"


def calculate_gmm_threshold(scores: list[float]) -> float:
    """Use a Gaussian Mixture Model to find a threshold between two score distributions.

    Fits a 2-component GMM to the provided scores, assuming a bimodal distribution
    representing Bad (low) and Good (high) classes, and returns the **midpoint
    between the two fitted component means**.

    #2798 replaced this midpoint with the components' equal-density crossing (see
    :func:`_weighted_gaussian_crossing`) on the geometry argument that max-pooling
    fattens the Bad mode, so the midpoint cuts inside Bad mass.  #2799 measured the
    two as paired within-step variants and the crossing lost on cost in every
    max-pooled window (report ``docs/experiments/safe-thresholds/REPORT.md``), so
    #2833 reverted to the midpoint.  The crossing solver is retained for the eval
    variant family and for #2836, which is looking for a third, better-founded cut.

    For score sets larger than :data:`_GMM_MAX_SAMPLES`, fits on a deterministic
    (seed-42) random subsample - the two-Gaussian fit is unchanged in practice
    but the cost no longer grows with the dataset size.

    Args:
        scores: List of model confidence scores, expected to follow a bimodal distribution.

    Returns:
        A float threshold. Scores at or above this value are classified as Good.
        Returns ``0.5`` when fewer than 2 scores are provided; falls back to
        the median of scores if GMM fitting fails.
    """
    if len(scores) < 2:
        return 0.5

    arr = gmm_fit_array(scores)
    fit = fit_score_gmm(arr)
    if fit is None:
        # If GMM fails, return median (of the subsample when one was taken -
        # representative of the full distribution and keeps this path bounded).
        return float(np.median(arr))
    return fit.midpoint()


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
    return (
        X_hash,
        y_hash,
        int(calibrate_count),
        float(calibration_fraction),
        int(hidden_dim),
        groups_key,
        _score_rows_digest(score_rows_by_group),
    )


class CalibrationFolds(NamedTuple):
    """The K calibration folds: held-out orderings, fallback sentinel, models.

    *orderings* are the per-fold ``(scores, labels)`` the conformal rule pools;
    *fallback* is the sentinel threshold to return outright when calibration
    was impossible (``None`` when the folds are real); *models* are the trained
    fold models in fold order, which the fold-anchored threshold
    (:func:`fold_anchored_gmm_threshold`) scores the haystack with so the
    anchors and the population it fits share one scale.
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
) -> CalibrationFolds:
    """Train the K calibration folds, keeping their models (uncached).

    Deterministic without a caller-supplied *rng*: the splits then come from a
    fresh ``RandomState(CALIBRATION_SPLIT_SEED)``, matching
    :func:`calibration_folds_cached`, so an uncached call (``det_ctx is None``)
    and a cached one produce the same folds for the same labelset.
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
) -> CalibrationFolds:
    """Memoized :func:`calibration_folds` keyed on the calibration inputs.

    When *det_ctx* is provided, caches the inclusion-independent folds on
    ``det_ctx.calibration_cache`` as ``(key, folds)`` and reuses them whenever
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
    never serialised.
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
            return cached[1]

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
    )
    if det_ctx is not None and key is not None:
        det_ctx.calibration_cache = (key, folds)
    return folds


def threshold_from_folds(folds: CalibrationFolds, inclusion_value: int) -> float:
    """The cross-calibration threshold *folds* implies at *inclusion_value*."""
    if folds.fallback is not None:
        return folds.fallback
    return threshold_from_fold_orderings(folds.orderings, inclusion_value)


def conformal_threshold(
    scores: list[float],
    labels: list[float],
    inclusion_value: int = 0,
) -> float:
    """Split-conformal quantile threshold over held-out calibration scores.

    Maps ``inclusion_value`` to a decision threshold via quantiles of the
    calibration score distributions rather than a min-cost search over
    observed cuts.  The min-cost argmin this replaced had exactly as many
    distinct optima as the calibration set had ranking errors, so on
    well-separated votes (the common case) the threshold never moved with
    inclusion; quantiles move whenever the scores have any spread (see
    docs/experiments/inclusion-knob/REPORT.md and issue #2693).

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
        inclusion_value: Integer in ``[-10, 10]``; higher includes more.

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

    def _threshold_at(k: int) -> float:
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


def _per_bag_fit_weights(
    y_rows: np.ndarray,
    group_rows: list,
) -> np.ndarray:
    """Per-row loss weights that balance Good votes against Bad **bags**.

    Every Good row weighs ``n_bad_bags / n_good``; every Bad row weighs
    ``1 / (rows in its bag)`` so each Bad image contributes exactly one image's
    worth of negative mass regardless of how many region nodes it flooded in.
    Total Good mass (``n_bad_bags``) equals total Bad mass (``n_bad_bags``), so
    the classes are balanced at the same magnitude as :func:`train_model`'s
    default inverse-frequency weights - only the *unit* changes from row to bag.
    """
    from collections import Counter  # noqa: PLC0415

    n_good = sum(1 for lbl in y_rows if lbl == 1.0)
    bad_groups = {g for g, lbl in zip(group_rows, y_rows, strict=True) if lbl == 0.0}
    n_bad_bags = len(bad_groups)
    bag_sizes = Counter(g for g, lbl in zip(group_rows, y_rows, strict=True) if lbl == 0.0)
    good_w = (n_bad_bags / n_good) if n_good else 1.0
    w = np.ones(len(y_rows), dtype=np.float32)
    for i, (g, lbl) in enumerate(zip(group_rows, y_rows, strict=True)):
        w[i] = good_w if lbl == 1.0 else (1.0 / bag_sizes[g])
    return w


def _pooled_group_scores(
    model: Any,
    cal_groups: list,
    rows_by_group: dict,
    X_np: np.ndarray,
    score_rows_by_group: dict | None,
) -> list[float]:
    """Collapse each calibration group to one max-pooled sigmoid score.

    With *score_rows_by_group* each group pools over the rows the scorer will
    max-pool at **inference**; otherwise it pools over the rows it trained on
    (the historical behaviour every production caller takes).
    """
    import torch  # noqa: PLC0415

    from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

    device = next(model.parameters()).device
    if score_rows_by_group is not None:
        blocks = [np.asarray(score_rows_by_group[g], dtype=np.float32) for g in cal_groups]
        sizes = [b.shape[0] for b in blocks]
        with torch.no_grad():
            X_cal = torch.tensor(np.concatenate(blocks, axis=0), dtype=torch.float32).to(device)
            flat = sigmoid_to_finite_scores(model(X_cal))
        out: list[float] = []
        offset = 0
        for size in sizes:
            out.append(max(flat[offset : offset + size]))
            offset += size
        return out

    cal_idx = [i for g in cal_groups for i in rows_by_group[g]]
    with torch.no_grad():
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32).to(device)
        row_scores = sigmoid_to_finite_scores(model(X_cal))
    by_row = dict(zip(cal_idx, row_scores, strict=True))
    return [max(by_row[i] for i in rows_by_group[g]) for g in cal_groups]


def _group_node_blocks(
    model: Any,
    cal_groups: list,
    rows_by_group: dict,
    X_np: np.ndarray,
    score_rows_by_group: dict | None,
) -> list[np.ndarray]:
    """Per calibration group, the array of that group's per-node sigmoid scores.

    The un-pooled counterpart of :func:`_pooled_group_scores`: it returns each
    group's full node-score vector rather than its max, so a caller can re-pool
    the bag under an alternative rule (top-k, extreme-value) while reusing the
    exact fold model and node scores.  ``max`` over each returned block
    reproduces :func:`_pooled_group_scores` value-for-value.
    """
    import torch  # noqa: PLC0415

    from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

    device = next(model.parameters()).device
    if score_rows_by_group is not None:
        blocks = [np.asarray(score_rows_by_group[g], dtype=np.float32) for g in cal_groups]
        sizes = [b.shape[0] for b in blocks]
        with torch.no_grad():
            X_cal = torch.tensor(np.concatenate(blocks, axis=0), dtype=torch.float32).to(device)
            flat = sigmoid_to_finite_scores(model(X_cal))
        out: list[np.ndarray] = []
        offset = 0
        for size in sizes:
            out.append(np.asarray(flat[offset : offset + size], dtype=np.float64))
            offset += size
        return out

    cal_idx = [i for g in cal_groups for i in rows_by_group[g]]
    with torch.no_grad():
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32).to(device)
        row_scores = sigmoid_to_finite_scores(model(X_cal))
    by_row = dict(zip(cal_idx, row_scores, strict=True))
    return [np.asarray([by_row[i] for i in rows_by_group[g]], dtype=np.float64) for g in cal_groups]


def _split_dither_rng(X_np: np.ndarray, y_np: np.ndarray) -> "np.random.RandomState":
    """A tie-break RNG for the Train/Calibrate split sizes, seeded from the labelset.

    Deliberately **not** :data:`CALIBRATION_SPLIT_SEED`.  That seed is a
    constant, so a draw from it is the same number on every call and would
    replace one deterministic function of the vote count with another - which
    is exactly the failure this dither exists to fix (issue #3286).  Seeding
    from a digest of the training vectors and their labels instead gives a
    draw that is *stable for a given labelset* - so the threshold stays a pure
    function of the votes, and :func:`_calibration_cache_key` (which hashes the
    same two arrays) stays valid - while differing between two labelsets that
    merely happen to be the same size.

    The digest is taken over the labels in full plus a **strided sample** of the
    training rows, rather than the whole matrix.  Two reasons, and both are
    requirements rather than optimisations:

    * *Bounded cost.*  A flooded patch labelset reaches tens of thousands of
      rows, so hashing all of it would add a ~100 MB pass to every step.  The
      sample is capped at :data:`_DITHER_SAMPLE_ROWS` rows.
    * *Sensitivity to the whole labelset.*  A fixed prefix would be useless: the
      rows are laid out Good-then-Bad and the earliest votes never move, so a
      prefix digest would barely change as a session accumulates votes and the
      dither would freeze into a constant - a coherent pattern again, just a
      different one.  Striding by ``len // k`` re-samples different rows at
      every size, and the labels change length and composition on every vote.

    Only slicing and ``tobytes`` are involved - no arithmetic over the
    embeddings - so the digest is byte-exact across machines.  A reduction like
    a column sum would not be: SIMD width changes the summation order, which is
    how #3166 turned a sub-part-per-million difference into a moved threshold.
    """
    stride = max(1, len(X_np) // _DITHER_SAMPLE_ROWS)
    h = hashlib.blake2b(np.ascontiguousarray(X_np[::stride], dtype=np.float32).tobytes(), digest_size=8)
    h.update(np.ascontiguousarray(y_np, dtype=np.float32).tobytes())
    return np.random.RandomState(int.from_bytes(h.digest()[:4], "little"))


def _dithered_count(exact: float, rng: "np.random.RandomState") -> int:
    """Round *exact* to an integer, breaking a fractional part at random.

    Stochastic rounding: ``P(round up) = frac(exact)``, so the count is
    **unbiased** (its expectation is *exact*) instead of being pinned to
    whichever side ``round`` picks.  A whole number is returned unchanged and
    draws nothing, so at the shipped ``calibration_fraction = 0.5`` this fires
    on odd vote counts only - the exact ties, where "nearest" has no answer.

    Why this is not just cosmetic (issue #3286).  ``round`` is round-half-to-
    **even**, so at a 50/50 split the tie-break alternates with the vote count:
    the odd vote joins Train at ``n % 4 == 1`` and Calibrate at ``n % 4 == 3``,
    and ``n_train`` climbs 4, 5, 5, 5, 6, 7, 7, 7, 8 - stalling for two votes,
    then jumping twice.  The fold models see a labelset share that seesaws with
    period 4, and every threshold read off them inherits it.  One user never
    notices; but the eval simulates one vote per step, so ``n`` tracks the step
    index in *every* run and the seesaw is phase-locked across all of them.
    Averaging hundreds of trajectories then cancels the noise and leaves the
    artifact: a visible 4-vote ripple on the learning curves, big enough to
    read as a real effect (see the #3286 investigation).  Randomising the tie
    decoheres the runs, so the ripple averages away like the noise it is.
    """
    low = math.floor(exact)
    frac = exact - low
    if frac <= 0.0:
        return int(low)
    return int(low) + (1 if rng.random_sample() < frac else 0)


def _grouped_folds(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
    seconds_sink: list[float] | None = None,
) -> tuple[list[tuple[Any, list]], float | None, np.ndarray, dict, dict]:
    """Train the bag-aware calibration folds; return the trained fold models.

    The shared core of :func:`_compute_fold_orderings_grouped` and
    :func:`compute_grouped_fold_node_scores`: both need identical fold splits and
    fold models, differing only in how they collapse each calibration group
    (max-pool vs. keep every node).  Returns
    ``(folds, fallback, X_np, rows_by_group, label_by_group)`` where *folds* is a
    list of ``(model, cal_groups)`` and *fallback* is a sentinel threshold when
    calibration is impossible (empty *folds* then).

    *seconds_sink*, when given, receives each fold's split-and-fit wall clock in
    fold order — the per-fold marginal cost of ``calibrate_count`` (issue #2897).
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    _rng = rng if rng is not None else np.random.RandomState(CALIBRATION_SPLIT_SEED)
    X_np = np.array(X_list)
    y_np = np.array(y_list)
    grp = list(groups)

    # Rows per group, and each group's (single) label.
    order_groups: list = []
    rows_by_group: dict = {}
    label_by_group: dict = {}
    for i, g in enumerate(grp):
        if g not in rows_by_group:
            rows_by_group[g] = []
            order_groups.append(g)
            label_by_group[g] = y_np[i]
        rows_by_group[g].append(i)

    pos_groups = [g for g in order_groups if label_by_group[g] == 1.0]
    neg_groups = [g for g in order_groups if label_by_group[g] == 0.0]
    n = len(order_groups)
    if n < 4:
        return [], 0.5, X_np, rows_by_group, label_by_group
    if len(pos_groups) < 2 or len(neg_groups) < 2:
        return [], 0.5, X_np, rows_by_group, label_by_group

    # Split sizes are dithered, not rounded, so a half-case does not resolve the
    # same way for every labelset of the same size (issue #3286).
    dither = _split_dither_rng(X_np, y_np)
    n_cal = max(1, _dithered_count(n * calibration_fraction, dither))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return [], NO_GOOD_THRESHOLD, X_np, rows_by_group, label_by_group

    def _per_class_n_train(class_total: int) -> int:
        target = _dithered_count(class_total * n_train / n, dither)
        return max(1, min(class_total - 1, target))

    n_train_pos = _per_class_n_train(len(pos_groups))
    n_train_neg = _per_class_n_train(len(neg_groups))

    # Index the plain group lists by position - group ids are tuples, and
    # ``np.array(list_of_tuples)`` would build a 2-D array and mangle them.
    folds: list[tuple[Any, list]] = []
    for _ in range(max(1, calibrate_count)):
        t_fold = time.monotonic()
        pos_perm = _rng.permutation(len(pos_groups))
        neg_perm = _rng.permutation(len(neg_groups))
        train_groups = [pos_groups[i] for i in pos_perm[:n_train_pos]] + [neg_groups[i] for i in neg_perm[:n_train_neg]]
        cal_groups = [pos_groups[i] for i in pos_perm[n_train_pos:]] + [neg_groups[i] for i in neg_perm[n_train_neg:]]

        train_idx = [i for g in train_groups for i in rows_by_group[g]]
        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        fold_w = torch.tensor(_per_bag_fit_weights(y_np[train_idx], [grp[i] for i in train_idx]), dtype=torch.float32)
        model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim, sample_weights=fold_w)
        folds.append((model, cal_groups))
        if seconds_sink is not None:
            seconds_sink.append(time.monotonic() - t_fold)

    return folds, None, X_np, rows_by_group, label_by_group


def _compute_fold_orderings_grouped(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
    score_rows_by_group: dict | None = None,
    model_sink: list | None = None,
    seconds_sink: list[float] | None = None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Bag-aware variant of :func:`compute_fold_orderings`.

    Splits by *group* (a voted image) instead of by row so a Bad bag's flooded
    region negatives never straddle the Train/Calibrate boundary, sizes the
    split over votes not rows, weight-balances each fold fit per-bag, and
    collapses every calibration group to a single max-pooled score (an image
    scores by its best region, as at inference).

    *score_rows_by_group* overrides which rows a calibration group collapses
    over - see :func:`compute_fold_orderings`.
    """
    folds, fallback, X_np, rows_by_group, label_by_group = _grouped_folds(
        X_list, y_list, input_dim, groups, rng, calibrate_count, calibration_fraction, hidden_dim, seconds_sink
    )
    if fallback is not None:
        return [], fallback
    if model_sink is not None:
        model_sink.extend(model for model, _cal in folds)

    orderings: list[tuple[list[float], list[float]]] = []
    for model, cal_groups in folds:
        # Collapse each calibration group to one max-pooled score, so a Good
        # bag and a Bad bag are pooled the same way the scorer pools an image.
        group_scores = _pooled_group_scores(model, cal_groups, rows_by_group, X_np, score_rows_by_group)
        group_labels = [float(label_by_group[g]) for g in cal_groups]
        orderings.append((group_scores, group_labels))

    return orderings, None


def compute_grouped_fold_node_scores(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    score_rows_by_group: dict | None = None,
    model_sink: list | None = None,
    seconds_sink: list[float] | None = None,
) -> tuple[list[tuple[list[np.ndarray], list[float]]], float | None]:
    """Bag-aware calibration folds, returning each held-out group's node scores.

    Like :func:`_compute_fold_orderings_grouped` but instead of max-pooling every
    calibration group it returns the group's **full node-score vector**, so a
    caller (the #2781 calibration study) can re-pool the same fold models' scores
    under alternative rules (top-k mean, extreme-value ``pnorm``) to recalibrate
    a threshold for a pooling variant without retraining.  ``max`` over each
    returned block reproduces this arm's production threshold exactly.

    Returns ``(fold_node_data, fallback)`` where *fold_node_data* is a list, one
    entry per fold, of ``(group_node_scores, group_labels)`` - *group_node_scores*
    being a list of 1-D float arrays (one per held-out calibration group).

    *model_sink*, when given, receives each trained fold model in fold order -
    the #2852 fold-anchored eval arm scores the haystack with the same fold
    models the orderings came from, so the anchors and the population it fits
    share one score scale without a retrain.
    """
    folds, fallback, X_np, rows_by_group, label_by_group = _grouped_folds(
        X_list, y_list, input_dim, groups, rng, calibrate_count, calibration_fraction, hidden_dim, seconds_sink
    )
    if fallback is not None:
        return [], fallback
    if model_sink is not None:
        model_sink.extend(model for model, _cal in folds)

    fold_node_data: list[tuple[list[np.ndarray], list[float]]] = []
    for model, cal_groups in folds:
        blocks = _group_node_blocks(model, cal_groups, rows_by_group, X_np, score_rows_by_group)
        group_labels = [float(label_by_group[g]) for g in cal_groups]
        fold_node_data.append((blocks, group_labels))
    return fold_node_data, None


def compute_fold_orderings(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    groups: list | None = None,
    score_rows_by_group: dict | None = None,
    model_sink: list | None = None,
    seconds_sink: list[float] | None = None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Train the K calibration folds and return their held-out orderings.

    Each ordering is a ``(cal_scores, cal_labels)`` pair: the fold model's
    sigmoid scores on its held-out calibration split, and that split's true
    labels.  Because :func:`train_model` is inclusion-independent, these
    orderings do **not** depend on ``inclusion`` - so they can be cached once
    and re-thresholded at any inclusion via :func:`threshold_from_fold_orderings`
    (and swept across all inclusions for the Stats chart).

    Returns ``(orderings, fallback)``.  When calibration is not possible the
    orderings are empty and ``fallback`` is the sentinel threshold the public
    wrapper must return (mirrors :func:`calculate_cross_calibration_threshold`'s
    historical early-returns); otherwise ``fallback`` is ``None``.

    *groups* activates the **bag-aware** path used when Bad votes are flooded
    into their region nodes: rows sharing a ``groups`` id belong to one voted
    image (a Bad bag's region negatives, or a single Good row) and
    are kept together on one side of every Train/Calibrate split, split counts
    are taken over *groups* (votes) not rows, fold fits are *weights*-balanced,
    and each calibration group collapses to one max-pooled score - matching how
    inference scores an image by its best region.  When *groups* is ``None``
    (every non-flooded caller) the historical row-wise path runs unchanged.

    *score_rows_by_group* (grouped path only) maps each group id to the row
    stack that group should be **scored** over, decoupling "what the fold model
    trains on" from "what a calibration bag collapses to".  It exists because
    the two are not the same whenever a Good vote contributes fewer rows than a
    Bad vote floods: the Good bag then collapses to a max over 1 row while the
    Bad bag - and every image at inference - collapses to a max over N, and
    ``max`` is an upward-biased order statistic, so the calibrated cut
    lands systematically high and the threshold over-rejects positives.  Passing
    each vote's inference rows here puts both classes in the geometry and at the
    width the scorer will actually use.  The production vote / labelset paths
    supply each voted image's full region-node stack
    (:func:`vtscore.detectors.training.inference_score_rows`); ``None`` keeps
    the "collapse over the training rows" behaviour for callers that have no
    inference geometry to offer.

    *model_sink*, when given, receives each trained fold model in fold order
    (see :func:`compute_grouped_fold_node_scores`); production callers pass
    nothing and the models stay fold-local as before.  *seconds_sink* likewise
    receives each fold's wall clock, which is what makes the *cost* half of the
    fold-count question (issue #2897) measurable without a second run.

    The folds are **independent repeated splits**, not a partition: every fold
    re-draws a stratified ``calibration_fraction`` holdout from the same labels,
    so raising ``calibrate_count`` averages more draws at a *fixed* per-fold
    calibration size rather than shrinking each holdout.  Two consequences the
    fold-count study rests on: the per-fold work is flat in K (total cost is
    linear in K), and the folds at ``calibrate_count=k`` are exactly the first
    *k* folds at any larger count drawn from the same ``rng`` - the splits are
    nested, so one run at Kmax yields every smaller K's calibration for free.
    """
    if groups is not None:
        return _compute_fold_orderings_grouped(
            X_list,
            y_list,
            input_dim,
            groups,
            rng=rng,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
            score_rows_by_group=score_rows_by_group,
            model_sink=model_sink,
            seconds_sink=seconds_sink,
        )
    n = len(X_list)
    if n < 4:
        return [], 0.5

    _rng = rng if rng is not None else np.random.RandomState(CALIBRATION_SPLIT_SEED)
    X_np = np.array(X_list)
    y_np = np.array(y_list)

    # Split sizes are dithered, not rounded, so a half-case does not resolve the
    # same way for every labelset of the same size (issue #3286).
    dither = _split_dither_rng(X_np, y_np)
    n_cal = max(1, _dithered_count(n * calibration_fraction, dither))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return [], NO_GOOD_THRESHOLD

    pos_idx = np.where(y_np == 1.0)[0]
    neg_idx = np.where(y_np == 0.0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return [], 0.5

    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    calibrate_count = max(1, calibrate_count)

    def _per_class_n_train(class_total: int) -> int:
        target = _dithered_count(class_total * n_train / n, dither)
        return max(1, min(class_total - 1, target))

    n_train_pos = _per_class_n_train(len(pos_idx))
    n_train_neg = _per_class_n_train(len(neg_idx))

    orderings: list[tuple[list[float], list[float]]] = []
    for _ in range(calibrate_count):
        t_fold = time.monotonic()
        pos_perm = _rng.permutation(pos_idx)
        neg_perm = _rng.permutation(neg_idx)
        train_idx = np.concatenate([pos_perm[:n_train_pos], neg_perm[:n_train_neg]])
        cal_idx = np.concatenate([pos_perm[n_train_pos:], neg_perm[n_train_neg:]])

        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32)

        model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim)
        if model_sink is not None:
            model_sink.append(model)

        with torch.no_grad():
            from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

            X_cal = X_cal.to(next(model.parameters()).device)
            # Sanitize non-finite sigmoids (destabilised fold model): the
            # orderings are cached, swept for the Stats chart, and averaged
            # into ``DetectorContext.threshold`` - a NaN here would silently
            # break every downstream ``score >= threshold`` comparison and
            # leak NaN into JSON responses.
            scores = sigmoid_to_finite_scores(model(X_cal))
        orderings.append((scores, y_np[cal_idx].tolist()))
        if seconds_sink is not None:
            seconds_sink.append(time.monotonic() - t_fold)

    return orderings, None


def threshold_from_fold_orderings(
    fold_orderings: list[tuple[list[float], list[float]]],
    inclusion_value: int,
) -> float:
    """Apply the conformal inclusion rule to the pooled fold orderings.

    Cheap: pools every fold's cached held-out ``(scores, labels)`` and runs
    :func:`conformal_threshold` once - no fold refits.  Pooling (rather than
    averaging per-fold thresholds) is deliberate: the knob's resolution is
    bounded by the number of calibration scores the quantiles are taken over,
    and per-fold quantiles on a handful of votes each would waste the other
    folds' scores.  All folds' scores live on the same sigmoid scale, so the
    pool is exchangeable enough for the quantile rule.

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
#: premise that fold scores are *not* directly comparable.  Both cannot be right,
#: and nobody has measured which.
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
    :data:`FOLD_CONFORMAL_COMBINES` for what each rule isolates.  **Eval-only**
    - nothing in the app calls this, and the run it exists for is what would
    license changing that.

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


def fit_gmm_threshold(scores: list[float]) -> tuple[float, GmmFit1D | None]:
    """The GMM cut of *scores* **and** the fit behind it.

    :func:`calculate_gmm_threshold` discards the fit; the corridor schedules
    (issue #2841) need the component means, so this returns both from one EM
    fit.  ``None`` accompanies the 0.5 / median fallbacks, where there is no
    fit to speak of and a schedule must degrade to the plain blend.
    """
    if len(scores) < 2:
        return 0.5, None
    arr = gmm_fit_array(scores)
    fit = fit_score_gmm(arr)
    if fit is None:
        return float(np.median(arr)), None
    return fit.midpoint(), fit


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
    anchored run (docs/experiments/population-anchored-calibration/REPORT.md)
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
