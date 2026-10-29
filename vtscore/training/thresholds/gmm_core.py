"""The 1-D two-component mixture's fit type, and the cuts read off a fit.

The bottom layer of the score mixture: :class:`GmmFit1D` (the two fitted
components, ordered by mean), the density-crossing solvers that read a cut off
one (:func:`_weighted_gaussian_crossing`, and :func:`_rate_cut` with its
``CUT_KIND_*`` vocabulary), and the sample handling every fit shares - the
subsample a fit is made on (:func:`gmm_fit_array`), the unscored-item filter
(:func:`scored_ordering`) and the cut canonicalisation
(:func:`snap_cut_to_sample`).

Nothing is fitted here: :mod:`~vtscore.training.thresholds.gmm_em` estimates a
:class:`GmmFit1D`, and :mod:`~vtscore.training.thresholds.gmm_threshold` names
the rules that cut one.  This module imports nothing else from
:mod:`vtscore.training.thresholds`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vtscore.utils.scores import scored_mask

# Above this many scores, fit the GMM on a random subsample instead of the full
# set. A 2-component, 1-D GMM only needs to recover the two clusters' means and
# variances, which 50k samples estimate as accurately as the full population -
# so the threshold is statistically indistinguishable while the EM fit stays
# O(50k) instead of O(N). This matters because ``calculate_gmm_threshold`` runs
# on the *full* score distribution on every cosine/text sort (sorting.py) and in
# the safe-threshold blend, where N reaches ~250k (GUI Find) to 2M+ (CLI Find).
_GMM_MAX_SAMPLES = 50_000


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
    variant (``*_cross`` in :data:`vtscore.eval.arms_safe_gmm._SAFE_GMM_VARIANTS`).
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
