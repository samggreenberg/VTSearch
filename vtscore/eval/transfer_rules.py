"""Estimators for the #2883 transfer study: is ``transfer`` a bias or a variance?

The #2836 chain ends at ``sim_oracle -> test oracle``, named "finite-sim-set
estimation / transfer".  On the production arm's ramp window it is the dominant
term - ``+0.037`` of a ``+0.056`` total, 67 %, on the corrected (#3187)
decomposition of the #3130 cells.  Two things about that term are not what its
name suggests, and this module exists to measure both.

**There is no distribution to transfer across.**  ``D_sim`` and ``D_test`` are
one random partition of a single pool
(:func:`~vtscore.eval.voting_iterations._split_media_ids`) scored by one model,
so the two score samples are draws from the same distribution and nothing in
this term is a shift: it is estimation error end to end.  The corrected table
already says so in threshold units - on that arm and window the term's *mean* is
``+0.0003`` against a mean *absolute* of ``0.0168``, while every other term in
the chain carries a mean within 10-30 % of its own absolute size.  A
perturbation that is symmetric about zero and still costs ``+0.037`` is a
**variance**, not a bias, and variance has different remedies from
misspecification.

**The reference point is optimistic.**  ``oracle_cost`` is the minimum of the
empirical cost over the **test sample itself**
(:func:`~vtscore.eval.calibration_metrics.oracle_cut`, whose own docstring calls
it "a lower bound on achievable cost, not a rule").  Subtracting a sample
minimum inflates every gap measured against it, so some of the ``+0.037`` is the
reference overfitting rather than anything a better fit could recover.  #3116
raises exactly this against the sibling ``rule_inefficiency`` /
``calibration_shift`` split and it has never been applied here.

Three estimators bracket the one population quantity ``C(tau*)`` that the
decomposition needs and has never had:

======================================  =========================================
:func:`~vtscore.eval.calibration_metrics.oracle_cut`  the sample minimum - a **lower** bound
:func:`honest_test_oracle`              cross-fitted: the cut is chosen and scored
                                        on disjoint folds, so it is honest, and
                                        it is an **upper** bound because the cut
                                        is picked on ``(K-1)/K`` of the sample
the learning-curve intercept            ``a`` in ``a + b/m`` fitted over
                                        :func:`sim_oracle_subsample_cut` - a
                                        third estimate that uses neither bound
======================================  =========================================

and two variance-reduced readings of the *sim* side test whether
``pooled_sim_oracle`` is the bound ``decisions.family_headroom_exhausted``
treats it as.  It is the empirical rate-loss minimiser over the sim scores, so
it bounds every rule's loss **on the sim set**.  Nothing makes it a bound on
**test** loss, which is what every table in this line reports, and an estimator
that trades a little bias for less variance can beat it out of sample.  Two
observations in the repo already show that happening: #3116 records a
calibration-set oracle beaten on test by the trained cut in every row of #2897,
and this study's own ``cost_identification`` is **negative** (-0.0057) - the
label-reading ``supervised`` cut losing on test to the unsupervised
``priorfree``.

**Every estimator here is deterministic given its input.**  A re-analysis has to
reproduce a run's numbers, and a resampling estimator that draws from global RNG
state does not; the draw is seeded from a digest of the score array itself, so
it is reproducible from a dump without plumbing a step index through.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any

import numpy as np

from vtscore.eval.calibration_metrics import operating_cost, oracle_cut
from vtscore.training.thresholds import fit_score_gmm, gmm_fit_array

#: Sim-set fractions for the learning curve.  Roughly geometric, so ``1/m`` -
#: the axis the ``a + b/m`` fit is linear in - is spread evenly rather than
#: bunched at the large-sample end where the curve is flat and says least.
#:
#: Every level is an exact multiple of 0.001, because :func:`subsample_rule`
#: encodes it as a milli-fraction and the analyzer reads the level back out of
#: the *name*.  A grid point that does not round-trip (0.0625 -> "f062" -> 0.062)
#: would put a 0.8 % error into the very x-axis the ``a + b/m`` fit is taken
#: over, which is a silently wrong slope rather than a visible failure.
TRANSFER_SUBSAMPLE_GRID: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50)

#: Bootstrap replicates for :func:`sim_oracle_bagged_cut`.
TRANSFER_BAG_REPLICATES: int = int(os.environ.get("VTS_TRANSFER_BAG_B", "32"))

#: Bootstrap replicates for :func:`bagged_gaussian_fit_cuts` - the label-free
#: arm, where each replicate costs an EM fit rather than a sort.
TRANSFER_BAGFIT_REPLICATES: int = int(os.environ.get("VTS_TRANSFER_BAGFIT_B", "16"))

#: Grid resolution for the smoothed cost curve.
TRANSFER_SMOOTH_GRID: int = 512

#: Folds for :func:`honest_test_oracle`.  Five, not two: the bound is only as
#: tight as the cut is good, and a cut picked on 80 % of the sample is closer to
#: the full-sample cut than one picked on half of it.
HONEST_ORACLE_FOLDS: int = 5

#: Study seed, mixed into every draw so this module's randomness is independent
#: of the harness's split/vote-order stream.
TRANSFER_SEED: int = 2883


def subsample_rule(frac: float) -> str:
    """``0.0625 -> "sim_oracle_f062"`` - the rule name for one curve level.

    Milli-fraction, matching the ``tail_a<milli-alpha>`` convention #2881 set, so
    the analyzer can parse a level back out of a variant name in the *data*
    rather than trusting that the grid constant has not moved since the run.
    """
    return f"sim_oracle_f{round(frac * 1000):03d}"


def subsample_fraction_of(rule: str) -> float | None:
    """``"sim_oracle_f062" -> 0.0625``-ish (``0.062``); ``None`` if not a level."""
    if not rule.startswith("sim_oracle_f"):
        return None
    tail = rule[len("sim_oracle_f") :]
    return int(tail) / 1000.0 if tail.isdigit() else None


#: One rule per curve level.
SUBSAMPLE_RULES: tuple[str, ...] = tuple(subsample_rule(f) for f in TRANSFER_SUBSAMPLE_GRID)

#: The two variance-reduced readings of the same sim set.  ``bag`` resamples,
#: ``smooth`` regularises the cost curve instead - two different ways to trade
#: bias for variance, so a null on one is not a null on the idea.
VARIANCE_REDUCED_RULES: tuple[str, ...] = ("sim_oracle_bag", "sim_oracle_smooth")

#: Label-reading, like ``sim_oracle`` itself: never shippable, reported to locate
#: the error.
TRANSFER_ORACLE_RULES: tuple[str, ...] = (*SUBSAMPLE_RULES, *VARIANCE_REDUCED_RULES)

#: The **label-free** arm: the same variance-reduction idea applied to the
#: unsupervised mixture fit, which is the only side of this that could ship.
#: Exploratory - see ``docs/experiments/2026-08-24-transfer-2883/PREREG.md``; #2883 item 1
#: asks for the characterisation *before* a remedy, so these are measured and
#: excluded from the ship gate rather than allowed to win on this run.
BAGGED_FIT_RULES: tuple[str, ...] = ("bagfit_mid", "bagfit_priorfree")

#: Which :func:`~vtscore.eval.cut_rules.gaussian_cuts` key each bagged arm
#: averages.  Deliberately that namespace and not production's
#: ``gmm_cut_from_fit`` one (``mid`` / ``rate`` / ``*_tilt``): the point of the
#: arm is the paired contrast ``pooled_bagfit_X - pooled_X``, which isolates
#: bagging only if both sides are the *same rule* on the same fit.


def _rng(scores: np.ndarray, salt: int) -> np.random.Generator:
    """A generator seeded from *scores* themselves, so the draw is reproducible.

    Keying on the data rather than on a step index means an offline re-analysis
    of a dumped score array reproduces the run's own estimate exactly, with no
    identifier to thread through three call layers and get wrong.
    """
    payload = np.ascontiguousarray(scores, dtype=np.float64).tobytes()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return np.random.default_rng([TRANSFER_SEED, salt, int.from_bytes(digest, "little")])


def _finite(x: float) -> float:
    xf = float(x)
    return xf if np.isfinite(xf) else float("nan")


def sim_oracle_subsample_cut(
    scores: np.ndarray,
    labels: np.ndarray,
    frac: float,
    fpr_weight: float,
    fnr_weight: float,
) -> float:
    """:func:`oracle_cut` over a seeded random ``frac`` of the sim pairs.

    The learning curve's x-axis.  Subsampling **at cut time** - rather than
    re-running the simulation with a smaller ``sim_fraction`` - is what makes the
    curve readable: the test set, the vote trajectory and the per-step model are
    bit-identical across levels, so the only thing that moves is the number of
    labelled sim scores the cut is estimated from.  Changing ``sim_fraction``
    would move the test set in the opposite direction at the same time and
    confound the reference point with the estimator.

    Drawn **without** replacement: this level is meant to be an oracle on a
    smaller sample, not a bootstrap of the full one.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    n = scores.size
    if n == 0 or labels.size != n:
        return float("nan")
    m = max(2, int(round(frac * n)))
    if m >= n:
        return _finite(oracle_cut(scores, labels, fpr_weight, fnr_weight)[0])
    idx = _rng(scores, salt=round(frac * 1000)).choice(n, size=m, replace=False)
    return _finite(oracle_cut(scores[idx], labels[idx], fpr_weight, fnr_weight)[0])


def sim_oracle_bagged_cut(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_weight: float,
    fnr_weight: float,
    replicates: int | None = None,
) -> float:
    """Mean of :func:`oracle_cut` over bootstrap resamples of the sim pairs.

    The empirical minimiser is a step function of the sample: it lands *on* an
    observed score, and which score wins can swing on one label.  Averaging the
    argmin over resamples keeps the same target and cuts the jitter, which is the
    textbook trade of a little bias for less variance - and the direct test of
    whether ``pooled_sim_oracle`` bounds *test* loss (it does not; it bounds
    sim-set loss, which is a different claim - see the module docstring).

    The mean, not the median: the quantity being averaged is a threshold on a
    bounded score axis with no heavy tail, and the mean is what makes this the
    bagged estimator rather than a robustified one.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    n = scores.size
    b = TRANSFER_BAG_REPLICATES if replicates is None else int(replicates)
    if n == 0 or labels.size != n or b <= 0:
        return float("nan")
    rng = _rng(scores, salt=1)
    cuts = np.empty(b, dtype=np.float64)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        ys = labels[idx]
        # A resample with one class missing has a degenerate cost curve; its
        # argmin is the "predict nothing" endpoint, which would drag the mean to
        # an edge for a reason that is an artefact of the draw, not the data.
        if ys.min() == ys.max():
            cuts[i] = np.nan
            continue
        cuts[i] = oracle_cut(scores[idx], ys, fpr_weight, fnr_weight)[0]
    good = cuts[np.isfinite(cuts)]
    return _finite(float(good.mean())) if good.size else float("nan")


def _silverman_bandwidth(scores: np.ndarray) -> float:
    """Silverman's rule, guarded so a degenerate sample yields no smoothing."""
    n = scores.size
    if n < 2:
        return 0.0
    sd = float(np.std(scores))
    q75, q25 = np.percentile(scores, [75.0, 25.0])
    iqr = float(q75 - q25)
    sigma = min(sd, iqr / 1.349) if iqr > 0.0 else sd
    if not np.isfinite(sigma) or sigma <= 0.0:
        return 0.0
    return 0.9 * sigma * float(n) ** -0.2


def sim_oracle_smoothed_cut(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_weight: float,
    fnr_weight: float,
) -> float:
    """Argmin of a **kernel-smoothed** empirical cost curve over the sim pairs.

    The second variance-reduced reading, and deliberately not a resampling one:
    it replaces each class's empirical CDF with a Gaussian-kernel-smoothed one
    (bandwidth by Silverman) and minimises the resulting continuous curve, so it
    can land *between* observed scores.  ``bag`` and ``smooth`` regularise the
    same estimator through different doors, which is why both are here - a null
    on one alone would not settle whether the idea works.

    Falls back to the plain :func:`oracle_cut` when the sample is too degenerate
    to give a positive bandwidth, so this rule never has "no root".
    """
    from scipy.special import ndtr  # noqa: PLC0415

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    n = scores.size
    if n == 0 or labels.size != n:
        return float("nan")
    pos = scores[labels == 1.0]
    neg = scores[labels != 1.0]
    h = _silverman_bandwidth(scores)
    if h <= 0.0 or pos.size == 0 or neg.size == 0:
        return _finite(oracle_cut(scores, labels, fpr_weight, fnr_weight)[0])
    lo, hi = float(scores.min()), float(scores.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return _finite(oracle_cut(scores, labels, fpr_weight, fnr_weight)[0])
    grid = np.linspace(lo - h, hi + h, TRANSFER_SMOOTH_GRID)
    # P(score >= tau) under the smoothed negative CDF -> smoothed FPR; the
    # mirrored quantity over positives -> smoothed FNR.
    fpr = ndtr((neg[None, :] - grid[:, None]) / h).mean(axis=1)
    fnr = ndtr((grid[:, None] - pos[None, :]) / h).mean(axis=1)
    cost = fpr_weight * fpr + fnr_weight * fnr
    return _finite(float(grid[int(np.argmin(cost))]))


def transfer_oracle_cuts(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_weight: float,
    fnr_weight: float,
) -> dict[str, float]:
    """Every :data:`TRANSFER_ORACLE_RULES` cut from one sim sample."""
    out: dict[str, float] = {}
    for frac in TRANSFER_SUBSAMPLE_GRID:
        out[subsample_rule(frac)] = sim_oracle_subsample_cut(scores, labels, frac, fpr_weight, fnr_weight)
    out["sim_oracle_bag"] = sim_oracle_bagged_cut(scores, labels, fpr_weight, fnr_weight)
    out["sim_oracle_smooth"] = sim_oracle_smoothed_cut(scores, labels, fpr_weight, fnr_weight)
    return out


def bagged_gaussian_fit_cuts(
    scores: np.ndarray,
    cuts_of_fit: "Callable[[Any, float, float], dict[str, float]]",
    fpr_weight: float,
    fnr_weight: float,
    replicates: int | None = None,
) -> dict[str, float]:
    """The label-free arm: bag the **mixture fit**, not the labelled cost curve.

    Resample the haystack, refit the two-component Gaussian, take the rule's cut,
    average.  This reads no labels, so unlike everything else in this module it
    could ship - it is the remedy the variance diagnosis implies, measured in the
    same run but pre-registered as exploratory and kept out of the ship gate
    (#2883 item 1 asks for the characterisation before the remedy).

    *cuts_of_fit* is injected rather than imported - it is
    :func:`~vtscore.eval.cut_rules.gaussian_cuts`, and ``cut_rules`` imports this
    module, so importing it back would be a cycle.

    Returns NaN for a rule where no replicate produced a finite cut, which is the
    same "no root" signal :func:`~vtscore.eval.cut_rules.gaussian_cuts` gives, so
    the caller's existing fallback handling applies unchanged.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = scores.size
    b = TRANSFER_BAGFIT_REPLICATES if replicates is None else int(replicates)
    out: dict[str, float] = dict.fromkeys(BAGGED_FIT_RULES, float("nan"))
    if n < 2 or b <= 0:
        return out
    rng = _rng(scores, salt=2)
    collected: dict[str, list[float]] = {name: [] for name in BAGGED_FIT_RULES}
    for _ in range(b):
        fit = fit_score_gmm(gmm_fit_array(scores[rng.integers(0, n, size=n)]))
        if fit is None:
            continue
        replicate = cuts_of_fit(fit, fpr_weight, fnr_weight)
        for name in BAGGED_FIT_RULES:
            cut = float(replicate.get(name[len("bagfit_") :], float("nan")))
            if np.isfinite(cut):
                collected[name].append(cut)
    for name, vals in collected.items():
        if vals:
            out[name] = _finite(float(np.mean(vals)))
    return out


def honest_test_oracle(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_weight: float,
    fnr_weight: float,
    folds: int = HONEST_ORACLE_FOLDS,
) -> tuple[float, float]:
    """Cross-fitted test oracle: ``(cost, mean threshold)``.

    The decomposition's last link is measured against ``min_tau`` of the
    empirical cost on the **test sample itself**, which is a sample minimum and
    therefore biased low - so ``transfer`` is biased high by however much the
    reference overfits.  This is the same quantity computed honestly: partition
    the test set, choose the cut on ``K-1`` folds, pay for it on the held-out
    one, and pool the held-out costs weighted by fold size.

    Read it as an **upper** bound on the population optimum ``C(tau*)``: the cut
    is estimated from ``(K-1)/K`` of the sample, so it is a slightly worse cut
    than the full-sample one, and the naive minimum is the matching lower bound.
    ``transfer`` therefore lives in a bracket rather than at a point, which is
    the honest way to report a gap whose reference is itself estimated.

    Returns ``(nan, nan)`` when the split cannot be made honestly - fewer than
    *folds* items, or a fold whose complement carries only one class.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    n = scores.size
    nan = float("nan")
    if n < folds or labels.size != n or folds < 2:
        return nan, nan
    order = _rng(scores, salt=3).permutation(n)
    parts = np.array_split(order, folds)
    total_cost = 0.0
    total_w = 0.0
    thresholds: list[float] = []
    for part in parts:
        if part.size == 0:
            continue
        mask = np.ones(n, dtype=bool)
        mask[part] = False
        fit_labels = labels[mask]
        if fit_labels.size == 0 or fit_labels.min() == fit_labels.max():
            return nan, nan
        thr, _c, _f, _fn = oracle_cut(scores[mask], fit_labels, fpr_weight, fnr_weight)
        if not np.isfinite(thr):
            return nan, nan
        cost, _fpr, _fnr = operating_cost(scores[part], labels[part], thr, fpr_weight, fnr_weight)
        if not np.isfinite(cost):
            return nan, nan
        total_cost += cost * part.size
        total_w += part.size
        thresholds.append(float(thr))
    if total_w <= 0.0:
        return nan, nan
    return _finite(total_cost / total_w), _finite(float(np.mean(thresholds)))
