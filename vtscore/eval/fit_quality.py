"""Absolute goodness-of-fit diagnostics for the score-mixture fits (issue #3329).

Every fit diagnostic already in this tree is **relative**: ``evt_loglik_gain``
prices a Gumbel+Normal mixture against a 2-Gaussian one, and the #2836
decomposition prices one cut against another.  Both answer "which of these two
is better?".  Neither can answer "is either of them any good?" - a
misspecification both families share is invisible to every number the harness
currently emits, because it cancels in the comparison.

This module supplies the missing half: statistics of a fitted mixture against
**the data it was fitted to**, plus - where a simulation knows them - against
the **true class-conditional** distributions the mixture claims to be modelling.

Four families, and each exists because the obvious cheaper statistic is
misleading in a specific way:

* **Distance to the fitted CDF** (:func:`ecdf_distances`).  Kolmogorov-Smirnov,
  Cramer-von Mises and Anderson-Darling against the fitted mixture.  A p-value
  is deliberately *not* returned: these fits see up to ``_GMM_MAX_SAMPLES``
  (50k) points, where every test rejects every model and "p < 1e-300" says only
  that the sample is large.  The statistics are reported as **effect sizes**,
  and AD is carried beside KS because the cut lives in a tail while KS is
  dominated by the bulk.

* **Tail calibration at the cut** (:func:`tail_calibration`).  The
  decision-relevant one, and the only one denominated in something a user
  feels: how many items the fitted model *thinks* sit above the threshold
  against how many actually do.  A mixture can track a histogram beautifully in
  the middle and still be wrong by a factor of three where the line is drawn,
  and that factor is exactly the false-positive rate the detector will surprise
  someone with.

* **Class-conditional shape** (:func:`class_shape`).  Skewness, excess
  kurtosis and an Anderson-Darling normality statistic per **true** class,
  computed on the logit axis - which is where the extreme-value limit lives
  (see :mod:`vtscore.eval.cut_rules`) and therefore where a max-pooled Bad mode
  should look least Gaussian.  The harness already records ``s_mu_neg`` /
  ``s_var_neg``; those are the first two moments, and the first two moments are
  precisely the statistics that *cannot* see the skew a max over ~24 region
  nodes is predicted to induce.

* **Identification** (:func:`identification`).  Whether the fitted split *is*
  the class split, which the mixture assumes when it calls its low component
  Bad.  Reported as balanced accuracy and adjusted Rand of the responsibility
  argmax against the true labels, plus the signed component-mean errors.  The
  #2836 chain prices this as one number inside a telescoping sum of cut
  displacements; here it is measured directly, in the partition itself.

**This module reads labels and is therefore eval-only.**  It never places a cut
and never feeds one back into a trajectory: it delegates every fit to the app's
own :func:`~vtscore.training.thresholds.fit_score_gmm` /
:func:`~vtscore.training.thresholds.anchored_gmm_fit` rather than re-deriving
one, so there is no app surface mirrored here and nothing for
``scripts/check-eval-app-sync.py`` to pin.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np

from vtscore.training.thresholds import GmmFit1D

# Reused rather than re-derived: the shape statistics below have to be taken on
# exactly the axis the EVT fit is taken on, or a "the Bad mode is skewed" result
# is a statement about two different transforms rather than about the data.
from vtscore.eval.cut_rules import _to_logit as to_logit

#: Sentinel for a statistic that could not be computed (too few points, a
#: degenerate fit, a class with no members).  Kept as NaN rather than omitted so
#: every diagnostic row has the same columns and an analyzer can count what it
#: dropped instead of silently ranging over a ragged frame.
NAN = float("nan")

#: Minimum sample size for a distributional statistic.  Below this the order
#: statistics are too coarse for AD's tail weighting to mean anything, and the
#: result would be dominated by which side of a step a single point fell.
MIN_GOF_N = 25

#: Minimum per-class count for a class-conditional shape statistic.  Skewness
#: and kurtosis are third and fourth moments; at small n their sampling
#: variance dwarfs any real asymmetry, and reporting them anyway is how a noisy
#: cell becomes a "finding".
MIN_CLASS_N = 30


def mixture_cdf(fit: GmmFit1D, x: np.ndarray) -> np.ndarray:
    """CDF of the fitted 2-component Gaussian mixture at *x*.

    ``w_lo * Phi((x - mu_lo)/sd_lo) + w_hi * Phi((x - mu_hi)/sd_hi)``, evaluated
    through :func:`math.erf` so the module carries no scipy dependency.  The
    weights are used as they were fitted; they are not renormalised, so a
    degenerate fit whose weights do not sum to one produces a CDF that does not
    reach one, which is visible rather than silently repaired.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros_like(x)
    for w, mu, var in ((fit.w_lo, fit.mu_lo, fit.var_lo), (fit.w_hi, fit.mu_hi, fit.var_hi)):
        if not (var > 0.0):
            continue
        z = (x - mu) / (math.sqrt(2.0 * var))
        out += w * 0.5 * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(z))
    return out


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[np.float64])(z / math.sqrt(2.0)))


def _ad_from_uniform(u: np.ndarray) -> float:
    """Anderson-Darling statistic ``A^2`` from PIT values *u*, sorted ascending.

    ``A^2 = -n - (1/n) * sum (2i-1) * [ln u_i + ln(1 - u_{n+1-i})]``.  Clipped
    away from 0 and 1 because a single PIT value that rounds to either end sends
    the statistic to infinity, which would report a *perfect* fit's rounding as
    an infinitely bad one.
    """
    n = u.size
    if n < 2:
        return NAN
    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    i = np.arange(1, n + 1, dtype=np.float64)
    s = np.sum((2.0 * i - 1.0) * (np.log(u) + np.log1p(-u[::-1])))
    return float(-n - s / n)


def ecdf_distances(sample: np.ndarray, fit: GmmFit1D) -> dict[str, float]:
    """Distance between the empirical distribution of *sample* and *fit*'s CDF.

    Returns ``ks``, ``cvm`` and ``ad``.  All three are computed from the
    probability-integral transform of the sample through the fitted mixture, so
    they share one evaluation and cannot disagree about which points were used.

    **No p-values, by design.**  At the 50k samples these fits routinely see,
    any of the three rejects any model that is not exactly right, so a p-value
    would report the sample size rather than the misfit.  ``ks`` is a
    probability-scale distance and is directly comparable across cells of
    different size; ``ad`` weights the tails, which is where the cut lives and
    where ``ks`` is least sensitive.
    """
    x = np.sort(np.asarray(sample, dtype=np.float64).ravel())
    n = x.size
    if n < MIN_GOF_N:
        return {"ks": NAN, "cvm": NAN, "ad": NAN}
    u = np.clip(mixture_cdf(fit, x), 0.0, 1.0)
    i = np.arange(1, n + 1, dtype=np.float64)
    ks = float(np.max(np.maximum(i / n - u, u - (i - 1.0) / n)))
    cvm = float(np.sum((u - (2.0 * i - 1.0) / (2.0 * n)) ** 2) + 1.0 / (12.0 * n))
    return {"ks": ks, "cvm": cvm, "ad": _ad_from_uniform(u)}


def tail_calibration(sample: np.ndarray, fit: GmmFit1D, cut: float) -> dict[str, float]:
    """How much mass the fit puts above *cut*, against how much is really there.

    The statistic the decision actually rests on.  ``predicted`` is the fitted
    mixture's survival at the cut, ``empirical`` the sample fraction above it,
    and ``ratio`` is ``empirical / predicted`` - so ``ratio > 1`` means the fit
    **under-predicts** how many items clear the line, which is the direction a
    Gaussian is expected to err in when the true Bad mode has a heavy right tail.

    ``lo_ratio`` is the same comparison restricted to the fitted **low**
    component: the fitted Bad mass above the cut against nothing observable, so
    it is reported as the predicted quantity alone (``lo_predicted``) for the
    supervised counterpart in :func:`class_shape` to be compared against.

    A ratio is returned rather than a difference because the quantity is a rate
    that ranges over orders of magnitude across cells; a difference of 0.002
    means something completely different at prevalence 0.5 and at 0.005.
    """
    x = np.asarray(sample, dtype=np.float64).ravel()
    out = {"predicted": NAN, "empirical": NAN, "ratio": NAN, "lo_predicted": NAN}
    if x.size < MIN_GOF_N or not math.isfinite(cut):
        return out
    predicted = float(1.0 - mixture_cdf(fit, np.array([cut]))[0])
    empirical = float(np.mean(x >= cut))
    out["predicted"] = predicted
    out["empirical"] = empirical
    # A predicted mass of exactly zero is not a ratio of infinity, it is a fit
    # that has placed no mass where the data demonstrably has some - reported as
    # NaN so an analyzer bands it as "undefined" rather than averaging an inf.
    if predicted > 0.0:
        out["ratio"] = empirical / predicted
    if fit.var_lo > 0.0:
        z = (cut - fit.mu_lo) / math.sqrt(2.0 * fit.var_lo)
        out["lo_predicted"] = float(fit.w_lo * 0.5 * (1.0 - math.erf(z)))
    return out


def _moments(u: np.ndarray) -> tuple[float, float]:
    """``(skewness, excess kurtosis)`` of *u*, or NaNs when undefined."""
    n = u.size
    if n < MIN_CLASS_N:
        return NAN, NAN
    mu = float(np.mean(u))
    d = u - mu
    m2 = float(np.mean(d**2))
    if not (m2 > 0.0):
        return NAN, NAN
    m3 = float(np.mean(d**3))
    m4 = float(np.mean(d**4))
    return m3 / m2**1.5, m4 / m2**2 - 3.0


def class_shape(scores: np.ndarray, labels: np.ndarray, *, logit: bool = True) -> dict[str, float]:
    """Shape of each **true** class-conditional score distribution.

    For each class, the skewness, excess kurtosis, and an Anderson-Darling
    statistic against the best-fitting Normal (moment MLE) - i.e. "if we
    insisted this class were Gaussian, how badly would we be wrong?".  This is
    the direct test of the mixture's structural assumption, and it needs labels,
    which is why it is eval-only.

    Taken on the **logit** axis by default.  A score is a squashed logit, and the
    squash pulls both tails in: a Bad mode that is a max over ~24 region nodes is
    an extreme-value statistic on the logit axis, and measuring its skew after a
    sigmoid would report the sigmoid's compression rather than the data's shape.

    Note ``ad_*`` here is a normality statistic for **one** class, and is not
    comparable to :func:`ecdf_distances`'s ``ad``, which scores the two-component
    mixture against the pooled sample.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    out: dict[str, float] = {}
    if s.size != y.size:
        raise ValueError(f"scores/labels length mismatch: {s.size} != {y.size}")
    axis = to_logit(s) if logit else s
    for name, mask in (("neg", y != 1.0), ("pos", y == 1.0)):
        u = axis[mask]
        u = u[np.isfinite(u)]
        skew, kurt = _moments(u)
        out[f"n_{name}"] = float(u.size)
        out[f"skew_{name}"] = skew
        out[f"kurt_{name}"] = kurt
        out[f"ad_normal_{name}"] = NAN
        if u.size >= MIN_CLASS_N:
            sd = float(np.std(u))
            if sd > 0.0:
                z = np.sort((u - float(np.mean(u))) / sd)
                out[f"ad_normal_{name}"] = _ad_from_uniform(_normal_cdf(z))
    return out


def identification(scores: np.ndarray, labels: np.ndarray, fit: GmmFit1D) -> dict[str, float]:
    """Does the fitted component split coincide with the true class split?

    The mixture's whole contract is that its low component is Bad and its high
    component is Good - every cut rule in the tree reads it that way.  This
    measures the claim: assign each score to the component that owns the larger
    responsibility, and score that partition against the true labels.

    Returns balanced accuracy (``bal_acc``, 0.5 = the split carries no class
    information), adjusted Rand (``ari``, 0 = chance), and the signed errors of
    each fitted component mean against its class's true mean (``mu_lo_err`` =
    ``mu_lo - mean(negatives)``).  The signed errors are what say whether a
    failure is a *rotation* of the split or a *shift* of one component.

    Balanced accuracy is used rather than raw accuracy because prevalence here
    runs to a few per cent, where a partition that calls everything Bad scores
    97% and is worth nothing.
    """
    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels, dtype=np.float64).ravel()
    if s.size != y.size:
        raise ValueError(f"scores/labels length mismatch: {s.size} != {y.size}")
    out = {"bal_acc": NAN, "ari": NAN, "mu_lo_err": NAN, "mu_hi_err": NAN, "n_ident": float(s.size)}
    pos = y == 1.0
    neg = ~pos
    if pos.sum() >= 1 and fit.var_hi > 0.0:
        out["mu_hi_err"] = float(fit.mu_hi - np.mean(s[pos]))
    if neg.sum() >= 1 and fit.var_lo > 0.0:
        out["mu_lo_err"] = float(fit.mu_lo - np.mean(s[neg]))
    if s.size < MIN_GOF_N or pos.sum() == 0 or neg.sum() == 0:
        return out
    if not (fit.var_lo > 0.0 and fit.var_hi > 0.0 and fit.w_lo > 0.0 and fit.w_hi > 0.0):
        return out

    # Responsibility argmax, in the log domain so a component whose density
    # underflows at a far-tail point does not become a 0/0 responsibility.
    log_lo = math.log(fit.w_lo) - 0.5 * math.log(2 * math.pi * fit.var_lo) - (s - fit.mu_lo) ** 2 / (2 * fit.var_lo)
    log_hi = math.log(fit.w_hi) - 0.5 * math.log(2 * math.pi * fit.var_hi) - (s - fit.mu_hi) ** 2 / (2 * fit.var_hi)
    pred_hi = log_hi > log_lo

    tp = float(np.sum(pred_hi & pos))
    fn = float(np.sum(~pred_hi & pos))
    tn = float(np.sum(~pred_hi & neg))
    fp = float(np.sum(pred_hi & neg))
    tpr = tp / (tp + fn) if (tp + fn) > 0 else NAN
    tnr = tn / (tn + fp) if (tn + fp) > 0 else NAN
    if math.isfinite(tpr) and math.isfinite(tnr):
        out["bal_acc"] = 0.5 * (tpr + tnr)
    out["ari"] = _adjusted_rand(tp, fp, fn, tn)
    return out


def _adjusted_rand(tp: float, fp: float, fn: float, tn: float) -> float:
    """Adjusted Rand index of a 2x2 contingency table.

    Written out for the binary case rather than pulled from sklearn: it is four
    lines here, and it keeps this module importable without dragging sklearn in
    for a statistic that has a closed form.
    """
    n = tp + fp + fn + tn
    if n < 2:
        return NAN

    def c2(x: float) -> float:
        return x * (x - 1.0) / 2.0

    index = c2(tp) + c2(fp) + c2(fn) + c2(tn)
    a = c2(tp + fn) + c2(fp + tn)  # true-class pair counts
    b = c2(tp + fp) + c2(fn + tn)  # predicted-class pair counts
    total = c2(n)
    if total <= 0.0:
        return NAN
    expected = a * b / total
    denom = 0.5 * (a + b) - expected
    if abs(denom) < 1e-12:
        return NAN
    return float((index - expected) / denom)


def anchor_mass_fraction(n_haystack: int, n_anchors: int, anchor_weight: float) -> float:
    """Share of the anchored EM's M-step mass contributed by the labels.

    ``kappa*v / (N + kappa*v)`` - the quantity that decides whether "anchored"
    is a description of the fit or only of its guards.  At the shipped
    ``FOLD_ANCHOR_WEIGHT`` of 0.3 with 20 votes against a 50k haystack sample
    this is 1.2e-4, and the E-step never sees the anchors at all (they are
    clamped one-hot), so the labels move the fitted means by at most that share
    of the distance between the anchor mean and the component mean.

    Reported per step so a "the anchors identify the components" claim can be
    checked against the mass that would have to carry it.
    """
    if n_haystack < 0 or n_anchors < 0 or not (anchor_weight >= 0.0):
        return NAN
    mass = anchor_weight * float(n_anchors)
    total = float(n_haystack) + mass
    if total <= 0.0:
        return NAN
    return float(mass / total)


#: Column order for one goodness-of-fit diagnostic row.  Kept as an explicit
#: tuple - rather than "whatever keys the dict happens to have" - so a row
#: written by an older run and one written by a newer run line up in the same
#: frame, and a missing statistic reads as NaN instead of shifting every column
#: after it.
FIT_QUALITY_COLUMNS: tuple[str, ...] = (
    # Distance to the fitted mixture (label-free; computable in the app too).
    "gof_ks",
    "gof_cvm",
    "gof_ad",
    # Tail calibration at the production cut (label-free).
    "tail_predicted",
    "tail_empirical",
    "tail_ratio",
    "tail_lo_predicted",
    # True class-conditional shape (labels; logit axis).
    "shape_n_neg",
    "shape_skew_neg",
    "shape_kurt_neg",
    "shape_ad_normal_neg",
    "shape_n_pos",
    "shape_skew_pos",
    "shape_kurt_pos",
    "shape_ad_normal_pos",
    # Component-to-class identification (labels).
    "ident_bal_acc",
    "ident_ari",
    "ident_mu_lo_err",
    "ident_mu_hi_err",
    "ident_n",
    # Anchoring: how much mass the labels actually carry.
    "anchor_mass_frac",
    "anchor_n",
    "anchor_kappa",
    # Anchored-vs-unanchored parameter drift (the H3 statistic).
    "anchored_dmu_lo",
    "anchored_dmu_hi",
    "anchored_dw_lo",
)


def fit_quality_row(
    sample: np.ndarray,
    fit: GmmFit1D | None,
    *,
    cut: float | None = None,
    labels: Optional[np.ndarray] = None,
    label_scores: Optional[np.ndarray] = None,
    anchored_fit: GmmFit1D | None = None,
    n_anchors: int = 0,
    anchor_weight: float = 0.0,
) -> dict[str, Any]:
    """Assemble one :data:`FIT_QUALITY_COLUMNS` row.

    *sample* is the haystack score array the mixture was fitted to (already
    subsampled by :func:`~vtscore.training.thresholds.gmm_fit_array`, so the
    statistics are taken against exactly the points the EM saw).  *labels* /
    *label_scores* are the **ground-truth** labelled scores for the shape and
    identification statistics - in a simulation that is the sim set, not the
    user's votes, because the question is whether the fit matches the *classes*
    rather than whether it matches the handful of items someone clicked.

    Every field defaults to NaN, so a caller that has no labels (or no fit) still
    produces a complete, aligned row.
    """
    row: dict[str, Any] = dict.fromkeys(FIT_QUALITY_COLUMNS, NAN)
    row["anchor_n"] = float(n_anchors)
    row["anchor_kappa"] = float(anchor_weight)
    row["anchor_mass_frac"] = anchor_mass_fraction(int(np.size(sample)), n_anchors, anchor_weight)

    if fit is None:
        return row

    d = ecdf_distances(sample, fit)
    row["gof_ks"], row["gof_cvm"], row["gof_ad"] = d["ks"], d["cvm"], d["ad"]

    if cut is not None:
        t = tail_calibration(sample, fit, cut)
        row["tail_predicted"] = t["predicted"]
        row["tail_empirical"] = t["empirical"]
        row["tail_ratio"] = t["ratio"]
        row["tail_lo_predicted"] = t["lo_predicted"]

    if labels is not None and label_scores is not None:
        sh = class_shape(label_scores, labels)
        for k, v in sh.items():
            row[f"shape_{k}"] = v
        ident = identification(label_scores, labels, fit)
        row["ident_bal_acc"] = ident["bal_acc"]
        row["ident_ari"] = ident["ari"]
        row["ident_mu_lo_err"] = ident["mu_lo_err"]
        row["ident_mu_hi_err"] = ident["mu_hi_err"]
        row["ident_n"] = ident["n_ident"]

    if anchored_fit is not None:
        row["anchored_dmu_lo"] = float(anchored_fit.mu_lo - fit.mu_lo)
        row["anchored_dmu_hi"] = float(anchored_fit.mu_hi - fit.mu_hi)
        row["anchored_dw_lo"] = float(anchored_fit.w_lo - fit.w_lo)

    return row
