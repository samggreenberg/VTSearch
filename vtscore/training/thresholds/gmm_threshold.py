"""The score mixture's cut: a named rule applied to a fit, and the one-call threshold.

:func:`gmm_cut_from_fit` reads a cut off a
:class:`~vtscore.training.thresholds.gmm_core.GmmFit1D` under a named rule;
:func:`calculate_gmm_threshold` fits the scores and returns the midpoint of the
two component means, with its ``0.5`` / median fallbacks - the line every
cosine sort and the safe-threshold blend draw - and :func:`fit_gmm_threshold`
returns that cut together with the fit behind it.

Reads :mod:`~vtscore.training.thresholds.gmm_core` and
:mod:`~vtscore.training.thresholds.gmm_em`.
"""

from __future__ import annotations

import numpy as np

from vtscore.training.thresholds.gmm_core import (
    CUT_KIND_DEGENERATE_MIDPOINT,
    CUT_KIND_INTERIOR,
    GmmFit1D,
    _rate_cut,
    gmm_fit_array,
)
from vtscore.training.thresholds.gmm_em import fit_score_gmm


def gmm_cut_from_fit(fit: GmmFit1D, rule: str, fpr_weight: float = 1.0, fnr_weight: float = 1.0) -> tuple[float, str]:
    """Apply a named cut *rule* to *fit*; ``(cut, kind)``.

    *kind* is :mod:`~vtscore.training.thresholds.gmm_core`'s
    ``cut_fallback_kind`` vocabulary (``CUT_KIND_*``) - empty exactly when
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


def _fit_gmm_threshold_with_array(scores: list[float]) -> tuple[float, np.ndarray | None, GmmFit1D | None]:
    """:func:`fit_gmm_threshold`, also handing back the array the fit was made on.

    ``(cut, arr, fit)``.  *arr* is :func:`gmm_fit_array`'s subsample (``None``
    under two scores, where nothing is fitted), so a caller that continues the
    fit (:func:`converge_score_gmm`) works the same sample the shipped cut was
    read off.  The one body behind :func:`fit_gmm_threshold`,
    :func:`calculate_gmm_threshold` and :func:`text_sort_cuts`: the typed-query
    sort's acquisition cut is this midpoint, fallbacks included, and delegating
    is what keeps it bit-identical to the example sort's line.
    """
    if len(scores) < 2:
        return 0.5, None, None
    arr = gmm_fit_array(scores)
    fit = fit_score_gmm(arr)
    if fit is None:
        return float(np.median(arr)), arr, None
    return fit.midpoint(), arr, fit


def fit_gmm_threshold(scores: list[float]) -> tuple[float, GmmFit1D | None]:
    """The GMM cut of *scores* **and** the fit behind it.

    :func:`calculate_gmm_threshold` discards the fit; the corridor schedules
    (issue #2841) need the component means, so this returns both from one EM
    fit.  ``None`` accompanies the 0.5 / median fallbacks, where there is no
    fit to speak of and a schedule must degrade to the plain blend.
    """
    cut, _arr, fit = _fit_gmm_threshold_with_array(scores)
    return cut, fit


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
