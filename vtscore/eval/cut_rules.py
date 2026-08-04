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

#: Sigmoid scores are clipped into ``[eps, 1-eps]`` before the logit transform so
#: saturated scores stay finite.
_LOGIT_EPS = 1e-6


def _to_logit(x: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(x, dtype=np.float64), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p) - np.log1p(-p)


def _from_logit(u: float) -> float:
    return float(1.0 / (1.0 + math.exp(-u)))


def _mean_log_jacobian(x: np.ndarray) -> float:
    """``mean log |du/dx|`` for ``u = logit(x)``, i.e. ``-mean log(x(1-x))``.

    A density fitted in logit space cannot be compared to one fitted in score
    space without this: ``f_x(x) = f_u(u) * du/dx``.  Adding it converts a
    logit-space log likelihood into the score-space one, which is what makes the
    Gaussian-vs-Gumbel comparison a like-for-like model comparison rather than an
    artefact of the axis each was fitted on.
    """
    p = np.clip(np.asarray(x, dtype=np.float64), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return float(-np.mean(np.log(p) + np.log1p(-p)))


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
    """Every EVT-family cut from one Gumbel+Normal fit, mapped back to score space.

    The fit lives in **logit** space (see :func:`fit_both_mixtures`), but a
    density crossing is invariant under a monotone reparametrisation: both sides
    of ``w_lo*f_lo == lam*w_hi*f_hi`` pick up the same Jacobian, which cancels.
    So the root can be solved on the logit axis and squashed back, and it is the
    same point as solving in score space would have given.
    """
    out = {}
    for name, cut in (
        ("gumbel_cross", fit.crossing()),
        ("gumbel_priorfree", fit.rate_crossing(1.0, 1.0)),
        ("gumbel_rate", fit.rate_crossing(fpr_weight, fnr_weight)),
    ):
        out[name] = float("nan") if cut is None else _from_logit(cut)
    return out


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

    **The EVT fit is done in logit space, and that is not a detail.**  The
    extreme-value limit applies to the max of the *region logits*; the score is
    that max pushed through a sigmoid, which is a strongly nonlinear squash, so a
    Gumbel fitted to sigmoid scores is fitting the wrong axis and measurably
    loses to a 2-Gaussian mixture.  On the logit axis - where the maximum is
    actually taken - the family is the right one.  (This is a different claim
    from #2799's dead logit-space *Gaussian* variant: the cut is invariant to a
    monotone reparametrisation, so moving a Gaussian across spaces changes little;
    changing the *family* to the one the limit theorem names is the hypothesis.)

    Both fits see the identical (possibly subsampled) sample and the EVT fit is
    seeded from the Gaussian one's midpoint, so ``evt_loglik_gain`` is not an
    initialisation artefact.  It is reported in **score-space** units: the
    logit-space log likelihoods have the change-of-variable Jacobian added back
    (:func:`_mean_log_jacobian`), without which a cross-space comparison is
    meaningless.  ``gmm_logit_loglik`` is the same Gaussian mixture fitted on the
    logit axis, so a gain can be attributed to the *family* rather than the axis.
    """
    arr = gmm_fit_array(scores)
    gmm = fit_score_gmm(arr)
    u = _to_logit(arr)
    jac = _mean_log_jacobian(arr)
    gmm_logit = fit_score_gmm(u)
    init = None if gmm is None else float(_to_logit(np.array([gmm.midpoint()]))[0])
    evt = fit_gumbel_normal_mixture(u, init_split=init)
    nan = float("nan")
    params: dict[str, Any] = {
        "sim_n": float(arr.size),
        # What ``calculate_gmm_threshold`` returns when the fit fails.
        "fallback_median": float(np.median(arr)) if arr.size else nan,
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
        "gmm_logit_loglik": nan,
        "evt_loglik": nan,
        "evt_loglik_gain": nan,
        "pred_offset_equal_var": nan,
    }
    if gmm_logit is not None:
        params["gmm_logit_loglik"] = gaussian_mixture_mean_loglik(u, gmm_logit) + jac
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
        # These four are in logit units; the loglik is converted to score space.
        params.update(
            evt_w_lo=evt.w_lo,
            evt_loc_lo=evt.loc_lo,
            evt_scale_lo=evt.scale_lo,
            evt_mu_hi=evt.mu_hi,
            evt_var_hi=evt.var_hi,
            evt_loglik=evt.mean_loglik + jac,
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
    else:
        # Mirror :func:`calculate_gmm_threshold`'s fallback exactly, so the
        # ``mid`` rule reproduces the shipped threshold even when EM fails - and
        # so it stays a usable fallback for the rules that have no root.
        cuts["mid"] = params["fallback_median"]
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
            # The EVT fit lives on the logit axis, so the tail level is read there.
            params["oracle_lo_sf_evt"] = evt.lo_survival(float(_to_logit(np.array([tau_star]))[0]))
    return cuts, params
