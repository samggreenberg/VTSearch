"""The 1-D two-component Gaussian mixture, its cuts, and its anchored variant.

The self-contained estimation layer: fit two Gaussians to a list of scores
(:func:`fit_score_gmm`), optionally clamping voted items' component membership
(:func:`fit_anchored_score_gmm`), and read a cut off the fit
(:func:`gmm_cut_from_fit`, :func:`_rate_cut`, :func:`_weighted_gaussian_crossing`).

This module imports nothing else from :mod:`vtscore.training.thresholds` - it
has no notion of folds, conformal quantiles or blends, and the conformal layer
has no notion of it beyond the plain fit.
"""

from __future__ import annotations

import math
import os
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


#: Iteration cap for the unanchored EM (:func:`fit_score_gmm`), and the
#: parameter-delta tolerance it would stop on if the likelihood rule below were
#: switched off.  Both are sklearn's own values for the fit this replaced, so
#: "how long may it run" did not change with the implementation.  The tolerance
#: is inert on the shipped path - :func:`_plain_em` passes
#: :data:`_EM_LOGLIK_TOL`, and :func:`_anchored_em` reads one rule or the other,
#: never both - and is kept so the parameter rule stays reachable as a measured
#: arm (``scripts/experiments/gmm_init/arms_3585.py``).
_EM_MAX_ITER = 100
_EM_TOL = 1e-8

#: Convergence tolerance for the unanchored fit, on the **mean log-likelihood**
#: rather than on the parameters: stop when an iteration improves the objective
#: by less than this.  Set to sklearn's own default, so the fit this replaced
#: and the fit that replaced it stop at the same place in the same sense -
#: which is what lets the two be compared as implementations of one estimator
#: instead of as two estimators.  See :func:`_anchored_em` for why the
#: parameter-delta rule is the wrong one here (a barely bimodal sort crawls a
#: flat ridge for 200 iterations to buy 0.004 nats).
_EM_LOGLIK_TOL = 1e-3

#: Iteration cap for the 2-means init below.  Lloyd's on a sorted 1-D sample
#: converges in a handful of boundary moves and each one costs a binary search,
#: so this is a runaway guard rather than a budget.
_KMEANS_MAX_ITER = 100

#: The empty anchor arrays :func:`_plain_em` hands :func:`_anchored_em`.  Module
#: level so a fit does not allocate two throwaway arrays; never written to (the
#: loop only reads ``.sum()`` and ``(a - mu) ** 2`` off them).
_NO_ANCHORS = np.empty(0, dtype=np.float64)


def _two_means_init(xs: np.ndarray) -> GmmFit1D | None:
    """Deterministic 2-means split of the **sorted** sample *xs*, as EM's start.

    Exact Lloyd's algorithm for ``k = 2`` in one dimension.  A 2-means
    assignment in 1-D is an *interval* split, so an iteration is a
    :func:`numpy.searchsorted` for the midpoint of the two centres followed by
    two O(1) prefix-sum means - not a pass over the data.  The whole
    initialiser is therefore one sort and one pair of cumulative sums, and its
    cost does not grow with the iteration count the way sklearn's
    ``init_params="kmeans"`` does, where a third of the initialiser's time went
    at 50k scores (issue #3585).

    Started from the sample's **min and max** rather than from sampled points,
    so it is deterministic without an RNG - no ``random_state`` to thread
    through, and the same scores give the same init on every machine.  Scores
    are bounded (a sigmoid or a cosine), so the pair is not an outlier artefact,
    and it is the start that survives a *saturated* distribution: quartiles
    coincide as soon as one class holds under 25% of the sample, which on the
    #3166 shape is exactly the case the split has to find.

    Returns ``None`` only when there is nothing to fit at all: fewer than two
    scores, or a non-finite sample.  A **constant** sample has no split to find,
    but it does have an answer - two identical components sitting on the value -
    and that is what it gets, because the anchored fit initialises from here and
    its anchors are what pull such a haystack apart.  (sklearn's initialiser
    answered a constant sample with a phantom second component at zero, so
    ``calculate_gmm_threshold`` on a haystack of 0.3s returned 0.15.)
    """
    n = int(xs.size)
    if n < 2:
        return None
    lo_c, hi_c = float(xs[0]), float(xs[-1])
    if not (math.isfinite(lo_c) and math.isfinite(hi_c)) or not math.isfinite(float(xs.sum())):
        return None
    if hi_c <= lo_c:
        return GmmFit1D(w_lo=0.5, mu_lo=lo_c, var_lo=1e-12, w_hi=0.5, mu_hi=lo_c, var_hi=1e-12)

    csum = np.concatenate(([0.0], np.cumsum(xs)))
    csq = np.concatenate(([0.0], np.cumsum(xs * xs)))
    var_all = max(float(csq[n] / n - (csum[n] / n) ** 2), 0.0)
    floor = max(1e-12, _ANCHOR_VAR_FLOOR_FRAC * var_all)

    split = 0
    for _ in range(_KMEANS_MAX_ITER):
        # Everything below the midpoint of the two centres belongs to the low
        # cluster.  ``side="left"`` ties with the E-step's ``>=`` convention.
        new_split = int(np.searchsorted(xs, 0.5 * (lo_c + hi_c), side="left"))
        # A boundary that empties a cluster is not a 2-means solution; keep the
        # previous centres and stop rather than divide by zero.
        if new_split <= 0 or new_split >= n or new_split == split:
            split = new_split if 0 < new_split < n else split
            break
        split = new_split
        lo_c = float(csum[split] / split)
        hi_c = float((csum[n] - csum[split]) / (n - split))
    if not 0 < split < n:
        # Lloyd's never found a non-degenerate boundary (two distinct values,
        # one of them a single point, and similar edges).  Split the sample in
        # half instead: the EM below is what has to find the components, and an
        # init that names two non-empty groups is all it needs.
        split = n // 2

    def _moments(a: int, b: int) -> tuple[float, float]:
        """``(mean, variance)`` of ``xs[a:b]`` from the prefix sums."""
        m = float(b - a)
        mean = (csum[b] - csum[a]) / m
        # Cancellation in ``E[x^2] - E[x]^2`` can go slightly negative on a
        # tight cluster far from zero; the floor is what the EM applies anyway.
        return mean, max((csq[b] - csq[a]) / m - mean * mean, floor)

    mu_lo, var_lo = _moments(0, split)
    mu_hi, var_hi = _moments(split, n)
    return GmmFit1D(
        w_lo=split / n,
        mu_lo=mu_lo,
        var_lo=var_lo,
        w_hi=(n - split) / n,
        mu_hi=mu_hi,
        var_hi=var_hi,
    )


def _plain_em(x: np.ndarray, init: GmmFit1D) -> GmmFit1D | None:
    """The unanchored EM: :func:`_anchored_em` with no anchors at all.

    With both anchor arrays empty every anchor term in that loop vanishes
    identically - the M-step masses are the responsibility sums, the total mass
    is ``n``, and the anchored sums are zero - so it *is* the plain
    two-component EM, run in the same preallocated-buffer, BLAS-free form.
    Sharing the body is the point: the unanchored fit and the anchored refit it
    initialises cannot drift into two different estimators of the same mixture.
    """
    return _anchored_em(x, _NO_ANCHORS, _NO_ANCHORS, init, 1.0, _EM_MAX_ITER, _EM_TOL, _EM_LOGLIK_TOL)


def fit_score_gmm(arr: np.ndarray) -> GmmFit1D | None:
    """Fit a deterministic 2-component GMM to a 1-D score array.

    Returns ``None`` when the fit fails - fewer than 2 scores, a non-finite
    sample, or an EM failure - leaving the fallback policy to the caller
    (:func:`calculate_gmm_threshold` falls back to the median).  A **constant**
    sample is not a failure: it comes back as two identical components sitting
    on the value, so the cut is that value (see :func:`_two_means_init`).

    **The estimator, and why it is ours** (issue #3585).  Two Gaussians over one
    dimension, fitted by EM from a deterministic 2-means init
    (:func:`_two_means_init`) with the same loop the anchored fit uses
    (:func:`_plain_em`).  Until #3585 this was sklearn's
    ``GaussianMixture(n_components=2, random_state=42)``, which is the same
    estimator of the same model - and was measured at **48% of a whole
    cosine/text sort** (10% after this change, and the sort as a whole 1.7x
    faster), because it pays ``covariance_type="full"`` machinery and a k-means
    init per call on a problem whose covariance is a scalar.  The fit itself is
    **7.3x** cheaper at the sizes a fold sees and **12x** at
    :data:`_GMM_MAX_SAMPLES`.  It is
    kept, unused by production, as :func:`fit_score_gmm_sklearn`: the swap
    *moves the fit* (a different init lands EM in a different place within its
    tolerance, and can land it in a different basin), so it is licensed by a
    measured equivalence rather than by argument, and the reference has to stay
    in the tree for that measurement to be repeatable.  What the study measured
    is in ``docs/experiments/2026-09-13-gmm-init-3585/REPORT.md``.
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    if x.size < 2:
        return None
    xs = np.sort(x)
    init = _two_means_init(xs)
    if init is None:
        return None
    fit = _plain_em(x, init)
    if fit is None:
        return None
    # EM keeps the init's component order in every case measured, but nothing in
    # the loop enforces it, and every caller reads ``lo``/``hi`` as an ordering.
    if fit.mu_hi < fit.mu_lo:
        return GmmFit1D(
            w_lo=fit.w_hi, mu_lo=fit.mu_hi, var_lo=fit.var_hi, w_hi=fit.w_lo, mu_hi=fit.mu_lo, var_hi=fit.var_lo
        )
    return fit


def fit_score_gmm_sklearn(arr: np.ndarray) -> GmmFit1D | None:
    """The pre-#3585 sklearn fit, kept as the reference the swap is measured against.

    **Not on any production path.**  This is what :func:`fit_score_gmm` was
    until #3585 replaced it with the native EM above, and it stays so that the
    equivalence can be re-measured rather than believed: the gate is an
    admitted-set comparison over real fold haystacks
    (``scripts/experiments/gmm_init/``), and a gate whose baseline lives only in
    a git history cannot be re-run against the next candidate.
    ``tests_lib/sorting/test_gmm_native_fit.py`` holds the standing form of it.
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

#: Convergence tolerance for the **anchored refit**, on the log-likelihood
#: rather than on the parameters (issue #3825).  Set to ``None`` to take the
#: parameter rule at :data:`_ANCHORED_EM_TOL` instead, which is what shipped
#: until #3825.
#:
#: **Not** the unanchored fit's :data:`_EM_LOGLIK_TOL`, and the difference is
#: the whole finding of #3825.  1e-3 is sklearn's tolerance, it is right for
#: :func:`fit_score_gmm`, and transferring it here is a **regression**: on 114
#: paired trajectory cells it costs +0.026 +- 0.006, because a barely
#: identified fold mixture stopped that early leaves the components too close
#: together and the midpoint cut lands low (FPR +0.044 to buy FNR -0.018).
#: What ports is the *criterion*, not the number.  1e-8 is numerically the
#: incumbent's own tolerance applied to the likelihood instead of the
#: parameters: 1.6x cheaper, the admitted set moves by a median of zero and a
#: p90 of one media in two thousand, and the share of folds exiting on
#: ``max_iter`` falls from 5.7% to 1.4%.  Measured in
#: ``docs/experiments/2026-09-13-anchored-em-stop-3825/REPORT.md``.
#:
#: Read inside :func:`fit_anchored_score_gmm` rather than bound as a default
#: argument, so a study can install a stopping rule for a whole run the way
#: ``scripts/experiments/gmm_init/arms_3825.py`` does.
_ANCHORED_EM_LOGLIK_TOL: "float | None" = 1e-8

#: Iteration cap and parameter-delta tolerance for the anchored refit - the
#: signature defaults of :func:`fit_anchored_score_gmm`, named so the loop's
#: budget is one fact rather than two literals.  The tolerance is inert while
#: :data:`_ANCHORED_EM_LOGLIK_TOL` is set (:func:`_anchored_em` reads one rule
#: or the other, never both).
_ANCHORED_EM_MAX_ITER = 200
_ANCHORED_EM_TOL = 1e-8


def _anchor_loglik_terms(
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    mu: np.ndarray,
    const: np.ndarray,
    two_var: np.ndarray,
    lam: float,
) -> float:
    """The clamped points' contribution to :func:`_anchored_em`'s objective.

    Their responsibilities are fixed one-hot, so each anchor simply adds its own
    component's log density at the multiplicity *lam* the M-step counts it with;
    *const* already holds ``log w_c - 0.5*log(2*pi*var_c)`` per component.  Zero
    when there are no anchors, which is what makes the anchored and free
    objectives the same arithmetic for :func:`_plain_em`.
    """
    total = 0.0
    if a_lo.size:
        total += lam * (a_lo.size * const[0] - float(np.sum((a_lo - mu[0]) ** 2)) / two_var[0])
    if a_hi.size:
        total += lam * (a_hi.size * const[1] - float(np.sum((a_hi - mu[1]) ** 2)) / two_var[1])
    return total


def _record_stopping(
    stats: "dict[str, float] | None",
    n_iter: int,
    stopped: bool,
    loglik: float,
    loglik_tol: "float | None",
) -> None:
    """Fill an :func:`_anchored_em` caller's *stats* sink, if it passed one.

    ``converged`` is 1.0 only when the loop met its tolerance; a fit that left
    on ``max_iter`` is not the estimator anyone specified, and before #3825
    nothing anywhere reported the difference.  ``loglik`` appears only when
    there was a log-likelihood rule to record - the objective at the parameters
    the final iteration started from, i.e. the value the stopping decision was
    taken on rather than a re-evaluation.
    """
    if stats is None:
        return
    stats["n_iter"] = float(n_iter)
    stats["converged"] = float(stopped)
    if loglik_tol is not None:
        stats["loglik"] = float(loglik)


def _anchored_em(
    x: np.ndarray,
    a_lo: np.ndarray,
    a_hi: np.ndarray,
    init: GmmFit1D,
    anchor_weight: float,
    max_iter: int,
    tol: float,
    loglik_tol: "float | None" = None,
    stats: "dict[str, float] | None" = None,
    anchored_objective: bool = True,
) -> GmmFit1D | None:
    """Run the anchored EM iterations; ``None`` on numerical failure.

    *x* is the free (unlabelled) sample, *a_lo* / *a_hi* the anchor scores
    clamped to the low / high component.  Pure numpy, log-domain E-step, fixed
    iteration order, and no BLAS calls (see the M-step comment) - deterministic
    for fixed inputs, and reproducible across machines with it.  Only
    numerical failure (non-finite parameters) returns ``None`` here; semantic
    degeneracy (inverted means, collapsed component) is judged by the caller so
    it can name the reason.

    **This loop is the shipped threshold's dominant cost, which is why it is
    written the way it is** (issue #3558).  The 2026-08-28 fold-count study
    priced a calibration fold at 0.148 s and put **86% of that in this
    function's caller**, :func:`fit_anchored_score_gmm`; that single term is
    the whole reason ``calibrate_count`` ships at 2 when the study measured
    K=6 as worth −0.0057 ± 0.0012 cost early
    (``docs/experiments/2026-08-28-calibration-fold-count-3310/REPORT.md``).
    At the study's 7,747-media haystack the loop is **overhead-bound, not
    FLOP-bound**: two components over one dimension is ~15k doubles an
    iteration, so what it costs is numpy dispatches and allocation, not
    arithmetic.  So the body works in **preallocated 1-D buffers, one per
    component**, instead of allocating a fresh ``(n, 2)`` array (and half a
    dozen ``(n, 2)`` temporaries) per iteration - measured **5-6x faster** at
    1k-50k free scores, and the gap widens with *n*.

    Every operation is deliberately the *same* operation on the *same* values
    as the two-column form it replaces, so the fit is **bit-for-bit**
    identical, not merely close: the arithmetic is elementwise either way, an
    ``axis=1`` max over two columns is :func:`numpy.maximum` of the two, an
    ``axis=1`` sum over two columns is their sum, and each reduction still
    sees a contiguous array of the same values in the same order.  That
    equivalence is the point - a faster fit that moved the threshold would be
    a calibration change needing a study, not an optimisation - and it is
    pinned by ``TestAnchoredEmEquivalence`` in
    ``tests_lib/sorting/test_anchored_gmm.py``, which runs the pre-#3558
    two-column loop side by side and asserts exact equality of all six
    parameters.  **Keep that test passing rather than "tidying" this body**;
    the readable form is right there in the test to diff against.

    **Two stopping rules, and which one you want depends on the sample.**  By
    default (``loglik_tol=None``) the loop stops when no parameter moved by more
    than *tol* - the anchored path's rule, and the right one there because an
    anchored refit starts from an already-converged unanchored fit and has a
    short way to go.  Passing *loglik_tol* switches to "stop when the mean
    log-likelihood stops improving by that much", which is sklearn's rule and
    what :func:`fit_score_gmm` uses.  *anchored_objective* then says which
    likelihood: the **weighted semi-supervised** one this loop ascends, anchors
    included at weight *anchor_weight* and divided by the total mass (the
    default, and the only one that is the estimator's own objective), or the
    free sample's alone.  They coincide exactly when there are no anchors, so
    the unanchored fit cannot tell the difference; on an anchored refit they do
    not, and #3825 measured which one to stop on.

    The difference is not a detail: on a **barely bimodal** sample - a cosine
    text sort, where the query's matches are a shoulder on one broad mode rather
    than a second mode - the likelihood surface has a flat ridge, and EM crawls
    along it forever.  Measured on real 13k-score sorts, the parameter rule at
    1e-8 runs the full 200 iterations to buy ~0.004 nats, while the likelihood
    rule stops in ~25 with the same cut.  A criterion that cannot tell "still
    converging" from "converged, and now drifting" spends all of its time in the
    second case.  Computing the objective costs an extra reduction and an
    in-place log per iteration, which is why it is opt-in rather than always on:
    the anchored path would pay it for a rule it does not need, and it must stay
    bit-for-bit what #3558 pinned.

    **Which rule the anchored refit uses, and why that changed** (issue #3825).
    Until #3825 the anchored path passed nothing and therefore took the
    parameter rule - and #3585, which made the *initialiser* 5x cheaper, left
    that refit as ~90% of a fold's fit: measured on the captured fold corpus it
    ran **97-200 iterations** where the init took ~15, and on 2 of 5 sampled
    folds it never converged at all, exiting on *max_iter*.  The diagnosis is
    the same one the sort path gave: a barely-identified mixture crawls a flat
    ridge, and the parameter criterion cannot tell that it has stopped learning
    anything.  So the anchored path now passes
    :data:`_ANCHORED_EM_LOGLIK_TOL`, and the choice of that value is a measured
    one - ``docs/experiments/2026-09-13-anchored-em-stop-3825/REPORT.md``.

    Pass *stats* to learn **how** a fit ended: the dict comes back with
    ``n_iter`` (iterations run) and ``converged`` (1.0 when the loop met its
    tolerance, 0.0 when it exhausted *max_iter*).  Nothing reported that before
    #3825, which is the only reason a shipped estimator could spend a month
    exiting on its iteration cap without anyone knowing.  It is filled only on
    the success path; a numerical failure returns ``None`` and leaves it alone.

    The initialiser every anchored fit runs first used to be sklearn's
    ``GaussianMixture`` - the larger half of the remaining cost, and ~5x slower
    per EM iteration than this loop on the same estimation problem.  #3585
    replaced it with :func:`fit_score_gmm`, which is now **this same loop with
    no anchors** (:func:`_plain_em`), so the two halves of a fold's fit no
    longer differ by a factor of five for no reason.  That swap moves the fit
    and was licensed by measurement, not by argument; see
    ``docs/experiments/2026-09-13-gmm-init-3585/REPORT.md``.
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

    # Reused every iteration: ``r_lo`` / ``r_hi`` hold the two columns the
    # two-column form kept in one ``(n, 2)`` array, and ``scratch`` stands in
    # for its per-iteration temporaries.  Allocated once because the loop is
    # dispatch- and allocation-bound at these sizes (see the docstring).
    r_lo = np.empty_like(x)
    r_hi = np.empty_like(x)
    scratch = np.empty_like(x)
    prev_loglik: float | None = None
    loglik = float("nan")
    n_iter = 0
    stopped = False

    for n_iter in range(1, max_iter + 1):
        # E-step over the free sample only (anchors are clamped one-hot).
        # Log-domain: log w_c - 0.5*log(2*pi*var_c) - (x-mu_c)^2 / (2*var_c).
        # ``const`` is that first pair of terms, which do not depend on x.
        const = np.log(np.maximum(w, 1e-300)) - 0.5 * np.log(2.0 * math.pi * var)
        two_var = 2.0 * var
        np.subtract(x, mu[0], out=r_lo)
        np.square(r_lo, out=r_lo)
        np.divide(r_lo, two_var[0], out=r_lo)
        np.subtract(const[0], r_lo, out=r_lo)
        np.subtract(x, mu[1], out=r_hi)
        np.square(r_hi, out=r_hi)
        np.divide(r_hi, two_var[1], out=r_hi)
        np.subtract(const[1], r_hi, out=r_hi)
        # Subtract the per-row max (an ``axis=1`` max over two columns is just
        # their elementwise maximum), exponentiate, and normalise by the row
        # sum (likewise, their elementwise sum).
        np.maximum(r_lo, r_hi, out=scratch)
        # The shift the log-sum-exp is taken around; its own sum is half the
        # log-likelihood, and it is about to be overwritten.
        max_sum = float(np.sum(scratch)) if loglik_tol is not None else 0.0
        np.subtract(r_lo, scratch, out=r_lo)
        np.subtract(r_hi, scratch, out=r_hi)
        np.exp(r_lo, out=r_lo)
        np.exp(r_hi, out=r_hi)
        np.add(r_lo, r_hi, out=scratch)
        np.divide(r_lo, scratch, out=r_lo)
        np.divide(r_hi, scratch, out=r_hi)
        if loglik_tol is not None:
            # ``scratch`` still holds the row sums the responsibilities were
            # normalised by, so the objective costs no second pass over the
            # data: a log-sum-exp is the shift plus the log of that sum.  It is
            # the log-likelihood at the parameters this iteration *started*
            # from, which is exactly the quantity sklearn compares between
            # iterations - so "converged" means the same thing in both, and the
            # stopping decision is taken below, after the M-step, as it is there.
            #
            # ``anchored_objective`` decides *whose* likelihood that is: the
            # free sample's alone, or the weighted semi-supervised objective
            # this EM actually ascends, anchors included (#3825).  With no
            # anchors the two are the same arithmetic on the same values -
            # ``total_mass`` is ``n`` and both anchor terms are skipped - so
            # :func:`_plain_em` is unaffected either way, bit for bit.
            np.log(scratch, out=scratch)
            loglik = max_sum + float(np.sum(scratch))
            if anchored_objective:
                loglik = (loglik + _anchor_loglik_terms(a_lo, a_hi, mu, const, two_var, lam)) / total_mass
            else:
                loglik /= n

        # M-step with the anchors folded in at weight ``lam`` each.
        m_lo = float(r_lo.sum()) + lam * n_lo
        m_hi = float(r_hi.sum()) + lam * n_hi
        if not (m_lo > 0.0 and m_hi > 0.0):
            return None
        # ``np.sum`` rather than ``@``: a dot product is dispatched to BLAS,
        # whose accumulation order depends on the kernel the CPU selects and on
        # the thread count, so the same input can differ in its last bits
        # between two machines.  numpy's own pairwise reduction does not
        # (issue #3166), and it reduces in the same order whatever the stride,
        # which is what lets these run over the 1-D buffers instead of over
        # two strided column views.
        np.multiply(r_lo, x, out=scratch)
        sum_lo_x = float(np.sum(scratch))
        np.multiply(r_hi, x, out=scratch)
        sum_hi_x = float(np.sum(scratch))
        mu_new = np.array(
            [
                (sum_lo_x + lam * sum_a_lo) / m_lo,
                (sum_hi_x + lam * sum_a_hi) / m_hi,
            ]
        )
        np.subtract(x, mu_new[0], out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, r_lo, out=scratch)
        sum_lo_sq = float(np.sum(scratch))
        np.subtract(x, mu_new[1], out=scratch)
        np.square(scratch, out=scratch)
        np.multiply(scratch, r_hi, out=scratch)
        sum_hi_sq = float(np.sum(scratch))
        var_new = np.array(
            [
                (sum_lo_sq + lam * float(((a_lo - mu_new[0]) ** 2).sum())) / m_lo,
                (sum_hi_sq + lam * float(((a_hi - mu_new[1]) ** 2).sum())) / m_hi,
            ]
        )
        var_new = np.maximum(var_new, var_floor)
        w_new = np.array([m_lo, m_hi]) / total_mass

        if not (np.all(np.isfinite(mu_new)) and np.all(np.isfinite(var_new)) and np.all(np.isfinite(w_new))):
            return None
        if loglik_tol is None:
            delta = max(
                float(np.max(np.abs(mu_new - mu))),
                float(np.max(np.abs(var_new - var))),
                float(np.max(np.abs(w_new - w))),
            )
            mu, var, w = mu_new, var_new, w_new
            stopped = delta < tol
        else:
            mu, var, w = mu_new, var_new, w_new
            stopped = prev_loglik is not None and abs(loglik - prev_loglik) < loglik_tol
            prev_loglik = loglik
        if stopped:
            break

    _record_stopping(stats, n_iter, stopped, loglik, loglik_tol)

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
    max_iter: int = _ANCHORED_EM_MAX_ITER,
    tol: float = _ANCHORED_EM_TOL,
    stats: "dict[str, float] | None" = None,
) -> tuple[GmmFit1D | None, str]:
    """Fit the label-anchored 2-component mixture (issue #2852).

    *arr* is the (possibly :func:`gmm_fit_array`-subsampled) haystack score
    sample; *anchor_scores* / *anchor_labels* are the voted items' scores and
    binary labels (1.0 Good -> high component, else low).  Initialised from the
    **unanchored** :func:`fit_score_gmm` fit (deterministic: a 2-means init and
    this same EM loop with no anchors), then refined by anchored EM (see
    :func:`_anchored_em`).

    Returns ``(fit, provenance)``.  On success the provenance is ``"anchored"``
    and the fit's components are class-identified by construction (``hi`` is
    the Good-anchored component).  On failure the fit is ``None`` and the
    provenance names the reason (``"no_anchors"``, ``"too_few_scores"``,
    ``"unanchored_init_failed"``, ``"em_failed"``, ``"inverted_means"``,
    ``"component_collapse"``) - the caller decides the fallback policy
    (:func:`anchored_gmm_fit` falls back to the unanchored fit, never to 0.5).

    Pass *stats* to receive the refit's ``n_iter`` / ``converged`` (see
    :func:`_anchored_em`); the loop is stopped on
    :data:`_ANCHORED_EM_LOGLIK_TOL`, read here at call time rather than bound as
    a default so a study can install another rule (#3825).

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
    fit = _anchored_em(x, a_lo, a_hi, init, anchor_weight, max_iter, tol, _ANCHORED_EM_LOGLIK_TOL, stats)
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


def calculate_gmm_threshold(scores: list[float]) -> float:
    """Use a Gaussian Mixture Model to find a threshold between two score distributions.

    Fits a 2-component GMM to the provided scores, assuming a bimodal distribution
    representing Bad (low) and Good (high) classes, and returns the **midpoint
    between the two fitted component means**.

    #2798 replaced this midpoint with the components' equal-density crossing (see
    :func:`_weighted_gaussian_crossing`) on the geometry argument that max-pooling
    fattens the Bad mode, so the midpoint cuts inside Bad mass.  #2799 measured the
    two as paired within-step variants and the crossing lost on cost in every
    max-pooled window (report ``docs/experiments/2026-08-03-safe-thresholds/REPORT.md``), so
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

    The two-Gaussian fit is refit on whatever distribution it is handed, so this
    is scale-adaptive: it is the reason every cosine sort draws an embedder-
    appropriate line without a per-embedder constant (#3347).  The ``0.5``
    literal is the sole exception - a sigmoid-scale sentinel returned on a
    haystack of 0 or 1 items, where there is no distribution to fit.  On a
    cosine sort it is a cosine-scale number that was never fitted on cosines;
    with at most one item ranked it changes nothing, which is why it stands.

    This is :func:`fit_gmm_threshold` with the fit discarded, and is kept as the
    one-value entry point every cosine/text sort calls.  The two used to carry
    byte-identical bodies; delegating is what stops them drifting apart.
    """
    return fit_gmm_threshold(scores)[0]


# ----------------------------------------------------------------------------
# The typed-query (text / cosine) sort's line (issue #3826)
# ----------------------------------------------------------------------------

#: The two rules a cosine sort's line can be drawn with.
#:
#: * ``"gmm_midpoint"`` - :func:`calculate_gmm_threshold`, the midpoint of a
#:   two-Gaussian fit.  The shipped default.
#: * ``"guarded_tail"`` - :func:`guarded_text_sort_threshold`.  Keep the mixture
#:   (fitted to convergence) only when its two components are separated, and
#:   otherwise cut the upper tail of the one broad mode.
#:
#: Which one :func:`text_sort_threshold` uses is :data:`TEXT_SORT_CUT_RULE`.
TEXT_SORT_CUT_RULES = ("gmm_midpoint", "guarded_tail")


#: The rule every cosine/text sort draws its line with, and Autopilot's opening
#: reads, from the ``VTSEARCH_TEXT_SORT_CUT`` environment variable.  **Off by
#: default**: #3826 chose the guarded rule, and its trajectory A/B failed the
#: pre-registered ship rule.  Autopilot's Bad phase votes near this line, and at
#: the guarded line it samples the top of the ranking, so the first detectors see
#: only near-miss negatives (Δcost +0.016 ± 0.005 over 100 clicks, +0.071 at 6-20
#: votes; ``docs/experiments/2026-09-23-text-cut-ab-3826/REPORT.md``).  A
#: display-only version, with the midpoint kept as the acquisition cut, is #4136.  An
#: unrecognised value falls back to the default instead of raising, so a typo
#: cannot take the sort route down.
def resolve_text_sort_cut_rule(value: str | None) -> str:
    """Normalise a ``VTSEARCH_TEXT_SORT_CUT`` value to a rule name; unknown or unset -> the default."""
    rule = (value or "").strip().lower()
    return rule if rule in TEXT_SORT_CUT_RULES else "gmm_midpoint"


TEXT_SORT_CUT_RULE = resolve_text_sort_cut_rule(os.environ.get("VTSEARCH_TEXT_SORT_CUT"))

#: Ashman's D at or above which the two fitted components count as separated
#: and the mixture's midpoint is kept.  2 is the textbook bimodality criterion
#: for a two-Gaussian mixture.  On 1,120 real text sorts, 9% cleared it, almost
#: all of them single-object images (``caltech101_m``, 60%).  Elsewhere the two
#: components are two halves of one mode (#3826).
TEXT_SORT_SEPARATION_D = 2.0

#: How many bulk sigmas above the median the tail cut sits.  Chosen as the
#: F1-best constant on three environments and tested on the fourth.  It was the
#: pick in 3 of 4 folds (3.5 in the other).  This is a tuned constant, not a
#: p-value: real negative tails are heavier than Gaussian, and at k = 3 a
#: Gaussian bulk under-predicts the negatives above the cut 2.7-6x.
TEXT_SORT_TAIL_K = 3.0

#: The converged fit's stopping rule, for the separated branch: the shipped
#: EM continued to 1e-10 in mean log-likelihood.  At the shipped 1e-3 the fit
#: still moves 5% of the haystack under a random start.  On separated data the
#: continuation is short.
_TEXT_SORT_CONVERGED_TOL = 1e-10
_TEXT_SORT_CONVERGED_MAX_ITER = 5000


def ashman_d(fit: GmmFit1D) -> float:
    """``|mu_hi - mu_lo| * sqrt(2 / (var_lo + var_hi))``: component separation in pooled sigmas."""
    pooled = fit.var_lo + fit.var_hi
    if not pooled > 0.0:
        return math.inf if fit.mu_hi != fit.mu_lo else 0.0
    return abs(fit.mu_hi - fit.mu_lo) * math.sqrt(2.0 / pooled)


def bulk_location_scale(scores: "list[float] | np.ndarray") -> tuple[float, float]:
    """The median of *scores*, and a sigma read from the half **below** it.

    The sigma is 1.4826 x the median distance of the lower half from the median,
    the MAD's Gaussian consistency factor applied to one side.  A typed query's
    matches are a right shoulder on one broad mode, so the lower half is the
    side they do not reach.  A heavy shoulder of positives therefore moves
    neither number much, which a symmetric MAD or a standard deviation would
    not survive.  Non-finite scores are ignored.  Returns ``(nan, nan)`` for an
    empty sample and a zero sigma for a constant one.
    """
    x = np.asarray(scores, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    mu = float(np.median(x))
    below = mu - x[x <= mu]
    return mu, 1.4826 * float(np.median(below))


def converge_score_gmm(arr: np.ndarray, start: GmmFit1D) -> GmmFit1D | None:
    """Continue the shipped unanchored EM from *start* until the likelihood stops moving.

    This is the "converged" fit of #3826: the same loop and the same objective as
    :func:`fit_score_gmm`, at :data:`_TEXT_SORT_CONVERGED_TOL`.  The stopping
    point is then no longer a parameter of the answer, and on separated data
    that is what makes the midpoint reproducible.  Returns ``None`` on
    numerical failure.
    """
    x = np.asarray(arr, dtype=np.float64).ravel()
    fit = _anchored_em(
        x,
        _NO_ANCHORS,
        _NO_ANCHORS,
        start,
        1.0,
        _TEXT_SORT_CONVERGED_MAX_ITER,
        _EM_TOL,
        _TEXT_SORT_CONVERGED_TOL,
    )
    if fit is None:
        return None
    if fit.mu_hi < fit.mu_lo:
        return GmmFit1D(
            w_lo=fit.w_hi, mu_lo=fit.mu_hi, var_lo=fit.var_hi, w_hi=fit.w_lo, mu_hi=fit.mu_lo, var_hi=fit.var_lo
        )
    return fit


def guarded_text_sort_threshold(scores: list[float]) -> tuple[float, str]:
    """The guarded line for a typed-query sort, and which branch drew it (issue #3826).

    A cosine sort of a typed query is usually **one** broad mode, with the
    query's matches as a thin right shoulder.  A two-Gaussian fit then splits
    the mode itself, and its midpoint lands in the densest region of the scores.
    On 1,120 real sorts the midpoint admitted a median 43% of the haystack, 12-54x
    the true matches outside single-object images.  That placement is also why
    a re-initialised or re-toleranced fit moved 6% of the verdicts: any
    perturbation of a line drawn through the mode moves thousands of medias.

    So:

    * fit the shipped mixture (:func:`fit_score_gmm`), and if its components are
      **separated** (:func:`ashman_d` >= :data:`TEXT_SORT_SEPARATION_D`),
      converge it (:func:`converge_score_gmm`) and cut at the midpoint.  Branch
      ``"gmm"``.  There the two components are real and the midpoint is well
      posed.
    * otherwise cut the tail: :func:`bulk_location_scale`'s median +
      :data:`TEXT_SORT_TAIL_K` sigmas.  Branch ``"tail"``.  The line admits
      what the bulk cannot explain.  It has no optimiser, so nothing but the
      data can move it.

    Measured against the midpoint on those 1,120 labelled sorts: F1 0.17 -> 0.39,
    and median admitted / true matches 17x -> 1.0x.  A bootstrap resample moves
    0.32% of the haystack instead of 1.2%.  The line is **worse** on the
    Inclusion-0 rate cost (FPR+FNR, +0.053), which prices a missed match at
    1/prevalence false alarms.  That trade was the decision #3826 asked for.
    See ``docs/experiments/2026-09-22-text-cut-3826/REPORT.md``.

    **Known failure: a majority-class query.**  When the matches are a large
    share of the haystack ("a person" in COCO is 54%) and the mixture is *not*
    separated, the median sits inside the matches, the tail is only the top of
    them, and the line admits a handful where the midpoint was roughly right.
    On the 28 measured sorts above 20% prevalence, F1 was 0.31 against the
    midpoint's 0.62.  Nothing label-free distinguishes that sort from a
    query the embedder barely separates, so this is documented and pinned by a
    test rather than guessed at.  A *separated* majority class takes the
    ``"gmm"`` branch and is unaffected.

    The tail cut can sit above every score, which paints nothing green.  That
    is "nothing here stands out", not an error.  It happened on 0.3% of the
    measured sorts.

    Fallbacks mirror :func:`calculate_gmm_threshold`: ``0.5`` with fewer than two
    scores (branch ``"fallback"``), and the shipped midpoint when the bulk has no
    spread to measure (a constant or two-valued sample, branch ``"gmm"``).
    """
    if len(scores) < 2:
        return 0.5, "fallback"
    arr = gmm_fit_array(scores)
    fit = fit_score_gmm(arr)
    if fit is None:
        return float(np.median(arr)), "fallback"
    if ashman_d(fit) >= TEXT_SORT_SEPARATION_D:
        converged = converge_score_gmm(arr, fit)
        return (fit if converged is None else converged).midpoint(), "gmm"
    mu, sigma = bulk_location_scale(scores)
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0.0):
        return fit.midpoint(), "gmm"
    return mu + TEXT_SORT_TAIL_K * sigma, "tail"


def text_sort_threshold(scores: list[float], rule: str | None = None) -> float:
    """The line a cosine/text sort draws, under *rule* (default :data:`TEXT_SORT_CUT_RULE`).

    The single entry point for "where does a typed-query sort's green region
    end".  The app's sort routes call it through
    :func:`vtscore.training.query_sort.cosine_sort_active`, and the eval
    harness's Autopilot opening calls it too (the Bad phase votes at this line).
    That way a study of the rule and the app cannot disagree about which line
    was drawn.  With the default rule it *is* :func:`calculate_gmm_threshold`.
    """
    chosen = TEXT_SORT_CUT_RULE if rule is None else rule
    if chosen == "guarded_tail":
        return guarded_text_sort_threshold(scores)[0]
    if chosen != "gmm_midpoint":
        raise ValueError(f"unknown text-sort cut rule {chosen!r}; expected one of {TEXT_SORT_CUT_RULES}")
    return calculate_gmm_threshold(scores)
