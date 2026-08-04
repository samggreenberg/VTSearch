"""Score-cut rules and their oracle decomposition (issue #2836).

The safe-threshold path fits a 2-component mixture to the *unlabelled* sim-set
score distribution and cuts it somewhere.  Which "somewhere" is correct is a
question about the loss being minimised, and this module makes the candidate
answers - and the ways each one can be wrong - explicit and measurable.

**The rule family.**  Every Gaussian rule here is the same solve at a different
tilt, ``w_lo*N_lo(x) == lam*w_hi*N_hi(x)``:

===============  ==========================  ===================================
rule             ``lam``                     minimises
===============  ==========================  ===================================
``cross``        ``1``                       misclassification **count** (Bayes,
                                             mixture weights as priors) - #2798,
                                             reverted by #2833
``priorfree``    ``w_lo/w_hi``               ``FPR + FNR`` (equal-weight rates)
``rate``         ``(wn/wf)*(w_lo/w_hi)``     ``wf*FPR + wn*FNR`` - the scored
                                             loss, at the live Inclusion setting
``mid``          -                            (midpoint of means; equals
                                             ``priorfree`` iff the variances are
                                             equal) - what production ships
===============  ==========================  ===================================

The ``gumbel_*`` rules are the same three tilts against a
:class:`~vtscore.training.evt_mixture.GumbelNormalFit1D` - a low component with
the right *shape* for a max over region nodes.

**The decomposition.**  A simulation knows the sim set's true labels, so the gap
between what a rule cuts and where the rate loss is actually minimised can be
split into named terms rather than reported as one number
(:func:`decomposition_cuts`).  Each successive pair differs in exactly one
assumption, so the chain telescopes:

``cross`` → ``priorfree``    the **loss/prior** term: dividing out the prior odds
``priorfree`` → ``supervised``  the **identification** term: the unsupervised
                             components are not the classes
``supervised`` → ``sim_oracle``  the **misspecification** term: the classes are
                             not Gaussian (a max-pooled Bad mode is an EVD)
``sim_oracle`` → test oracle  the **estimation/transfer** term: a finite sim set
                             is not the test set

Whichever term dominates names the repair; if the first one does, the leading
#2836 hypothesis is right and the fix is a one-parameter change to the tilt.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from vtscore.eval.calibration_metrics import oracle_cut
from vtscore.training.evt_mixture import (
    GumbelNormalFit1D,
    fit_gumbel_normal_mixture,
    gaussian_mixture_mean_loglik,
)
from vtscore.training.thresholds import GmmFit1D, fit_score_gmm, gmm_fit_array

#: Cut rules backed by the Gaussian mixture, in the order they are reported.
GAUSSIAN_RULES: tuple[str, ...] = ("mid", "cross", "priorfree", "rate")
#: Cut rules backed by the Gumbel(low) + Normal(high) mixture.
EVT_RULES: tuple[str, ...] = ("gumbel_cross", "gumbel_priorfree", "gumbel_rate")
#: Label-reading diagnostics.  **Not rules** - they read the sim set's true
#: labels, so they are upper bounds on what an unsupervised cut could achieve,
#: reported to locate the error rather than to be shipped.
ORACLE_RULES: tuple[str, ...] = ("supervised", "sim_oracle")

ALL_RULES: tuple[str, ...] = (*GAUSSIAN_RULES, *EVT_RULES, *ORACLE_RULES)


def _finite(x: float | None) -> float:
    """``float(x)`` with ``None``/non-finite collapsed to NaN."""
    if x is None:
        return float("nan")
    xf = float(x)
    return xf if math.isfinite(xf) else float("nan")


def gaussian_cuts(fit: GmmFit1D, fpr_weight: float, fnr_weight: float) -> dict[str, float]:
    """Every Gaussian-family cut from one fit.  NaN where the rule has no root.

    No midpoint fallback is applied here: the caller decides whether a missing
    root means "fall back to the midpoint" (what a shippable rule must do) or
    "record a miss" (what the measurement wants).  Conflating the two would
    silently score the midpoint under another rule's name.
    """
    return {
        "mid": _finite(fit.midpoint()),
        "cross": _finite(fit.crossing()),
        "priorfree": _finite(fit.rate_crossing(1.0, 1.0)),
        "rate": _finite(fit.rate_crossing(fpr_weight, fnr_weight)),
    }


def evt_cuts(fit: GumbelNormalFit1D, fpr_weight: float, fnr_weight: float) -> dict[str, float]:
    """Every EVT-family cut from one Gumbel+Normal fit.  NaN where undefined."""
    return {
        "gumbel_cross": _finite(fit.crossing()),
        "gumbel_priorfree": _finite(fit.rate_crossing(1.0, 1.0)),
        "gumbel_rate": _finite(fit.rate_crossing(fpr_weight, fnr_weight)),
    }


def supervised_cut(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_weight: float,
    fnr_weight: float,
) -> tuple[float, dict[str, float]]:
    """Rate-optimal cut between **label-supervised** Gaussians, and their moments.

    Fits one Gaussian per true class (moment MLE) and solves the same rate
    stationarity condition ``wn*f_pos == wf*f_neg``.  Compared against the
    unsupervised ``priorfree`` cut this isolates the *identification* error: how
    much of the gap is the EM split failing to be the class split, as opposed to
    the tilt being wrong.  The class weights are set to the cost weights (not the
    prevalence), which is what makes it the rate solve rather than the count one.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    pos = labels == 1.0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    nan = float("nan")
    stats = {"s_mu_neg": nan, "s_var_neg": nan, "s_mu_pos": nan, "s_var_pos": nan, "s_prevalence": nan}
    if n_pos < 2 or n_neg < 2:
        return nan, stats

    mu_neg, var_neg = float(scores[~pos].mean()), float(scores[~pos].var())
    mu_pos, var_pos = float(scores[pos].mean()), float(scores[pos].var())
    stats = {
        "s_mu_neg": mu_neg,
        "s_var_neg": var_neg,
        "s_mu_pos": mu_pos,
        "s_var_pos": var_pos,
        "s_prevalence": n_pos / (n_pos + n_neg),
    }
    if not (var_neg > 0.0 and var_pos > 0.0 and mu_pos > mu_neg):
        return nan, stats

    # w_lo := fpr_weight, w_hi := fnr_weight, lam := 1 solves wf*f_neg == wn*f_pos.
    supervised = GmmFit1D(
        w_lo=fpr_weight,
        mu_lo=mu_neg,
        var_lo=var_neg,
        w_hi=fnr_weight,
        mu_hi=mu_pos,
        var_hi=var_pos,
    )
    return _finite(supervised.crossing()), stats


def sim_oracle_cut(
    scores: np.ndarray,
    labels: np.ndarray,
    fpr_weight: float,
    fnr_weight: float,
) -> float:
    """The empirical rate-loss minimiser over the sim scores, using true labels.

    The end of the chain on the sim side: no parametric form at all, so the gap
    between this and :func:`supervised_cut` is exactly what assuming Gaussian
    classes costs.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if scores.size == 0 or labels.size != scores.size:
        return float("nan")
    thr, _cost, _fpr, _fnr = oracle_cut(scores, labels, fpr_weight, fnr_weight)
    return _finite(float(thr))


def fit_both_mixtures(scores: list[float] | np.ndarray) -> tuple[GmmFit1D | None, GumbelNormalFit1D | None, dict]:
    """Fit the Gaussian and the Gumbel+Normal mixture to the same sample.

    Both see the identical (possibly subsampled) array and the EVT fit is seeded
    from the Gaussian one's midpoint, so ``evt_loglik_gain`` - the per-point log
    likelihood difference - is a like-for-like comparison of the two low-component
    shapes rather than an initialisation artefact.  A positive gain is direct
    evidence for the misspecification hypothesis.
    """
    arr = gmm_fit_array(scores)
    gmm = fit_score_gmm(arr)
    evt = fit_gumbel_normal_mixture(arr, init_split=None if gmm is None else gmm.midpoint())
    nan = float("nan")
    params: dict[str, Any] = {
        "sim_n": float(arr.size),
        "gmm_ok": 1 if gmm is not None else 0,
        "evt_ok": 1 if evt is not None else 0,
        "w_lo": nan,
        "mu_lo": nan,
        "var_lo": nan,
        "w_hi": nan,
        "mu_hi": nan,
        "var_hi": nan,
        "evt_w_lo": nan,
        "evt_loc_lo": nan,
        "evt_scale_lo": nan,
        "evt_mu_hi": nan,
        "evt_var_hi": nan,
        "gmm_loglik": nan,
        "evt_loglik": nan,
        "evt_loglik_gain": nan,
        "pred_offset_equal_var": nan,
    }
    if gmm is not None:
        params.update(
            w_lo=gmm.w_lo,
            mu_lo=gmm.mu_lo,
            var_lo=gmm.var_lo,
            w_hi=gmm.w_hi,
            mu_hi=gmm.mu_hi,
            var_hi=gmm.var_hi,
            gmm_loglik=gaussian_mixture_mean_loglik(arr, gmm),
            pred_offset_equal_var=gmm.equal_var_offset(),
        )
    if evt is not None:
        params.update(
            evt_w_lo=evt.w_lo,
            evt_loc_lo=evt.loc_lo,
            evt_scale_lo=evt.scale_lo,
            evt_mu_hi=evt.mu_hi,
            evt_var_hi=evt.var_hi,
            evt_loglik=evt.mean_loglik,
        )
    if gmm is not None and evt is not None:
        params["evt_loglik_gain"] = params["evt_loglik"] - params["gmm_loglik"]
    return gmm, evt, params


def decomposition_cuts(
    sim_scores: list[float] | np.ndarray,
    sim_labels: np.ndarray,
    fpr_weight: float,
    fnr_weight: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """All cut rules plus the label-reading diagnostics, from one sim sample.

    Returns ``(cuts, params)``: *cuts* maps every name in :data:`ALL_RULES` to a
    score (NaN where that rule has no root on this fit), *params* carries the
    fitted mixture parameters, the fit-quality comparison, and the supervised
    class moments - everything the analyzer needs to test the #2836 predictions
    offline without re-running the simulation.
    """
    gmm, evt, params = fit_both_mixtures(sim_scores)
    nan = float("nan")
    cuts: dict[str, float] = dict.fromkeys(ALL_RULES, nan)

    if gmm is not None:
        cuts.update(gaussian_cuts(gmm, fpr_weight, fnr_weight))
    if evt is not None:
        cuts.update(evt_cuts(evt, fpr_weight, fnr_weight))

    sup, sup_stats = supervised_cut(sim_scores, sim_labels, fpr_weight, fnr_weight)
    params.update(sup_stats)
    cuts["supervised"] = sup
    cuts["sim_oracle"] = sim_oracle_cut(sim_scores, sim_labels, fpr_weight, fnr_weight)

    # Where the true optimum sits in the *fitted* Bad component's upper tail.  If
    # this is stable across steps and categories it is itself a shippable rule
    # ("cut the Bad tail at alpha"), which is the fallback answer if no crossing
    # rule wins outright.
    params["oracle_lo_sf_gauss"] = nan
    params["oracle_lo_sf_evt"] = nan
    tau_star = cuts["sim_oracle"]
    if math.isfinite(tau_star):
        if gmm is not None and gmm.var_lo > 0.0:
            z = (tau_star - gmm.mu_lo) / math.sqrt(gmm.var_lo)
            params["oracle_lo_sf_gauss"] = float(0.5 * math.erfc(z / math.sqrt(2.0)))
        if evt is not None:
            params["oracle_lo_sf_evt"] = evt.lo_survival(tau_star)
    return cuts, params
