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
