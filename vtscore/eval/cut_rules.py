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
:class:`~vtscore.eval.evt_mixture.GumbelNormalFit1D` - a component with the
right *shape* for a max over region nodes.  ``gumbel_any_*`` are those three
again without #2836's assumption that the Gumbel is necessarily the *low*
component; that assumption is what the Gumbel arm's fallback rate turned out to
be made of (issue #2846).

The ``tail_a*`` rules are a different animal from all of the above: not a
crossing at any tilt, but the fitted **Bad** component's own upper quantile -
"cut where alpha of the Bad mass is still above the cut" (issue #2881).  There is
no ``lam``, no orientation question, and no boundary that might not exist; it is
one constant against one fitted tail.

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
from vtscore.eval.evt_mixture import (
    GumbelNormalFit1D,
    fit_gumbel_normal_mixture_state,
    gaussian_mixture_mean_loglik,
)
from vtscore.eval.transfer_rules import (
    BAGGED_FIT_RULES,
    TRANSFER_ORACLE_RULES,
    bagged_gaussian_fit_cuts,
    transfer_oracle_cuts,
)
from vtscore.training.thresholds import (
    CUT_KIND_CONTINUED,
    CUT_KIND_DEGENERATE_MIDPOINT,
    CUT_KIND_INTERIOR,
    GmmFit1D,
    fit_score_gmm,
    gmm_fit_array,
)

#: ``cut_fallback_kind`` when *this* module's decomposition family substituted
#: the fit's own midpoint for a rule that has no root (issue #2900).  The
#: substitution is deliberately **rule-independent**: the family's job is to
#: compare tilts against each other on one fit, so every rule that misses gets
#: the same neutral stand-in rather than each rule's own extrapolation.  That is
#: what keeps ``rate`` comparable to its ``cross``/``priorfree`` siblings - and
#: at inclusion 0, where the cost weights are ``(1, 1)``, it is what keeps
#: ``rate`` *identical* to ``priorfree`` by construction, an identity every
#: report in ``docs/experiments/2026-08-04-gmm-cut/`` reads its ``*_rate`` rows through.
#:
#: Production does not do this.  It is not a bug on either side; the two answer
#: different questions, and this value in the emitted rows is what lets an
#: analyzer tell them apart instead of pooling both under ``cut_fallback == 1``.
CUT_KIND_MIDPOINT: str = "midpoint"

#: The whole ``cut_fallback_kind`` vocabulary, across both families.  Empty
#: means the rule found an interior stationary point and nothing was
#: substituted; the rest name which path produced the cut:
#:
#: ================================  =====================  ====================
#: value                             emitted by             cut is
#: ================================  =====================  ====================
#: ``""``                            both                   the rule's own root
#: ``"midpoint"``                    decomposition family   that fit's midpoint
#: ``"continued"``                   production rule        continued past an
#:                                                          inter-mean edge
#: ``"degenerate_midpoint"``         production rule        that fit's midpoint
#: ================================  =====================  ====================
#:
#: ``"midpoint"`` and ``"degenerate_midpoint"`` are both midpoints but are not
#: the same event: the first is a measurement policy applied to a sound fit, the
#: second is a fit too degenerate for any rule to cut.
CUT_FALLBACK_KINDS: tuple[str, ...] = (
    CUT_KIND_INTERIOR,
    CUT_KIND_MIDPOINT,
    CUT_KIND_CONTINUED,
    CUT_KIND_DEGENERATE_MIDPOINT,
)

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
#: Cut rules backed by the Gumbel + Normal mixture, at the same three tilts.
#:
#: The ``gumbel_*`` family is #2836's: it requires the Gumbel to be the *low*
#: component and declines the fit otherwise.  The ``gumbel_any_*`` family is
#: #2846's repair, solving in whichever orientation EM converged to.  Both are
#: measured because the difference between them **is** the #2846 question, and it
#: has to be answered on real scores rather than on the synthetic bench (which
#: was what mismeasured it the first time).
EVT_RULES: tuple[str, ...] = (
    "gumbel_cross",
    "gumbel_priorfree",
    "gumbel_rate",
    "gumbel_any_cross",
    "gumbel_any_priorfree",
    "gumbel_any_rate",
)
#: The pre-registered constant for the one-constant tail rule (issue #2881): the
#: median survival level at which the *true* rate optimum sat in the fitted Gumbel
#: low component, over #2846's 511 cells.  It is a median over one dataset and one
#: geometry, which is why the grid below sweeps around it rather than trusting it.
TAIL_ALPHA_PREREGISTERED: float = 0.158
#: Tail levels measured, in increasing alpha (so in *decreasing* cut).  Chosen so
#: the cut moves in near-even steps rather than alpha: for a Gumbel the cut is
#: ``loc - scale*ln(-ln(1-alpha))``, i.e. logarithmic in alpha, so an evenly
#: spaced alpha grid would bunch every point on one side of the optimum.  These
#: seven sit at ``loc + {3.20, 2.48, 2.15, 1.76, 1.39, 1.03, 0.67}*scale`` - even
#: 0.36-wide steps except the bottom rung, which reaches twice as far to bracket
#: a genuinely conservative cut without spending two levels down there.
#:
#: The grid exists because "the cost curve is flat near 0.158" is the actual
#: claim the stability finding makes, and a single hardcoded constant cannot test
#: it.  Only :data:`TAIL_ALPHA_PREREGISTERED` is a ship candidate; the rest are
#: measured to show the *shape* of the curve.  See ``analyze_cut.SWEEP_ONLY``.
TAIL_ALPHA_GRID: tuple[float, ...] = (0.04, 0.08, 0.11, TAIL_ALPHA_PREREGISTERED, 0.22, 0.30, 0.40)


def tail_alpha_rule(alpha: float) -> str:
    """Rule name for a tail level, e.g. ``0.158 -> "tail_a158"`` (units: milli-alpha).

    Three digits rather than a decimal point because these names become CSV
    column values and variant ids, and a ``.`` in either reads as a path
    separator to half the tooling that touches them.
    """
    return f"tail_a{round(alpha * 1000):03d}"


#: One rule per swept tail level.  Backed by the same EVT fit as
#: :data:`EVT_RULES`, but **not** a crossing: it inverts the fitted low
#: component's survival function instead of looking for a boundary between the
#: modes.  That is what makes the family worth another run after #2846 - a
#: quantile exists for every non-degenerate fit, so the 20-25 % midpoint-fallback
#: rate that diluted every crossing contrast to nothing should collapse to the
#: EVT fit-failure rate alone.
TAIL_RULES: tuple[str, ...] = tuple(tail_alpha_rule(a) for a in TAIL_ALPHA_GRID)

#: Every rule read off the EVT fit, so a failed fit can be attributed to all of
#: them at once.  ``EVT_RULES`` and ``TAIL_RULES`` stay separate above because
#: they answer different questions and only one of them declines for orientation.
EVT_FIT_RULES: tuple[str, ...] = (*EVT_RULES, *TAIL_RULES)

#: Label-reading diagnostics.  **Not rules** - they read the sim set's true
#: labels, so they are upper bounds on what an unsupervised cut could achieve,
#: reported to locate the error rather than to be shipped.
ORACLE_RULES: tuple[str, ...] = ("supervised", "sim_oracle")

#: #2883's readings of the **same** sim set as ``sim_oracle``: four subsample
#: levels (the learning curve in sim-set size) and two variance-reduced
#: estimators.  Label-reading like ``ORACLE_RULES``, and kept separate from them
#: because these do not sit on the decomposition chain - they measure the last
#: link's *shape* rather than adding a link to it.  See
#: :mod:`vtscore.eval.transfer_rules`.
#:
#: ``BAGGED_FIT_RULES`` is the label-free counterpart and does belong with the
#: Gaussian family: same fit, same rules, averaged over bootstrap refits.

ALL_RULES: tuple[str, ...] = (
    *GAUSSIAN_RULES,
    *BAGGED_FIT_RULES,
    *EVT_RULES,
    *TAIL_RULES,
    *ORACLE_RULES,
    *TRANSFER_ORACLE_RULES,
)


def _finite(x: float | None) -> float:
    """``float(x)`` with ``None``/non-finite collapsed to NaN."""
    if x is None:
        return float("nan")
    xf = float(x)
    return xf if math.isfinite(xf) else float("nan")


def gaussian_cuts(fit: GmmFit1D, fpr_weight: float, fnr_weight: float) -> dict[str, float]:
    """Every Gaussian-family cut from one fit.  NaN where the rule has no root.

    No fallback is applied here: the caller decides whether a missing root means
    "substitute something shippable" or "record a miss" (what the measurement
    wants).  Conflating the two would silently score a fallback under another
    rule's name.  Note the two answers have genuinely diverged - production's
    ``rate`` rule (:func:`~vtscore.training.thresholds.gmm_cut_from_fit`) neither
    returns the midpoint nor declines here: it continues past the inter-mean
    interval at the rule's own first-order slope, so it never stops moving with
    the cost tilt.  This function keeps reporting NaN because the decomposition
    is measuring *where the stationary point sits*, and "there is none" is the
    honest answer to that question.  The divergence is deliberate and is
    recorded per row in ``cut_fallback_kind`` (:data:`CUT_KIND_MIDPOINT` vs
    :data:`CUT_KIND_CONTINUED`), so an analysis that needs the shipped path can
    exclude the substituted steps rather than mistake them for it (#2900).
    """
    return {
        "mid": _finite(fit.midpoint()),
        "cross": _finite(fit.crossing()),
        "priorfree": _finite(fit.rate_crossing(1.0, 1.0)),
        "rate": _finite(fit.rate_crossing(fpr_weight, fnr_weight)),
    }


def evt_cuts(fit: GumbelNormalFit1D, fpr_weight: float, fnr_weight: float) -> tuple[dict[str, float], dict[str, str]]:
    """``(cuts, reasons)`` for every EVT-family rule, mapped back to score space.

    The fit lives in **logit** space (see :func:`fit_both_mixtures`), but a
    density crossing is invariant under a monotone reparametrisation: both sides
    of ``w_lo*f_lo == lam*w_hi*f_hi`` pick up the same Jacobian, which cancels.
    So the root can be solved on the logit axis and squashed back, and it is the
    same point as solving in score space would have given.

    *reasons* names why each rule produced no cut (one of
    :data:`~vtscore.eval.evt_mixture.CROSSING_REASONS`), which is what makes
    a fallback auditable rather than invisible — the #2846 diagnosis turned
    entirely on being able to tell ``modes_swapped`` (the fit is sound, the
    orientation assumption was not) from ``hi_owns_lo_mode`` (the two components
    genuinely collapsed onto each other).
    """
    cuts: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for name, (reason, cut) in (
        ("gumbel_cross", fit.crossing_state()),
        ("gumbel_priorfree", fit.rate_crossing_state(1.0, 1.0)),
        ("gumbel_rate", fit.rate_crossing_state(fpr_weight, fnr_weight)),
        ("gumbel_any_cross", fit.crossing_state(allow_swapped=True)),
        ("gumbel_any_priorfree", fit.rate_crossing_state(1.0, 1.0, allow_swapped=True)),
        ("gumbel_any_rate", fit.rate_crossing_state(fpr_weight, fnr_weight, allow_swapped=True)),
    ):
        cuts[name] = float("nan") if cut is None else _from_logit(cut)
        reasons[name] = reason
    return cuts, reasons


def tail_cuts(
    fit: GumbelNormalFit1D,
    alphas: tuple[float, ...] = TAIL_ALPHA_GRID,
) -> tuple[dict[str, float], dict[str, str]]:
    """``(cuts, reasons)`` for every tail-alpha rule, mapped back to score space.

    Each rule cuts where the fitted **Bad** component still has *alpha* of its
    mass above the cut - "cut the Bad tail at alpha", one constant, no crossing
    and no ``lam``.  Solved on the logit axis, where the fit lives, and squashed
    back, which is sound for a different reason than :func:`evt_cuts`': a crossing
    survives the change of variable because both sides pick up the same Jacobian
    and it cancels, whereas a quantile survives because the sigmoid is *monotone*,
    so it carries a tail probability through unchanged.  Either way the answer is
    the point solving in score space would have given.

    Nothing here consults the orientation: :meth:`GumbelNormalFit1D.lo_quantile`
    reads whichever component came out low, so the ``modes_swapped`` axis that
    #2836 and #2846 spent themselves on simply does not arise.  A rule declines
    only for a degenerate fit, which is why the reason vocabulary is a two-element
    subset of :data:`~vtscore.eval.evt_mixture.CROSSING_REASONS` rather than
    the full set.
    """
    cuts: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for alpha in alphas:
        cut = fit.lo_quantile(alpha)
        name = tail_alpha_rule(alpha)
        cuts[name] = float("nan") if cut is None else _from_logit(cut)
        reasons[name] = "degenerate_params" if cut is None else "ok"
    return cuts, reasons


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
    evt_fail, evt = fit_gumbel_normal_mixture_state(u, init_split=init)
    nan = float("nan")
    params: dict[str, Any] = {
        "sim_n": float(arr.size),
        # What ``calculate_gmm_threshold`` returns when the fit fails.
        "fallback_median": float(np.median(arr)) if arr.size else nan,
        "gmm_ok": 1 if gmm is not None else 0,
        "evt_ok": 1 if evt is not None else 0,
        "evt_fit_fail": evt_fail,
        # 1 when the Gumbel landed on the low mode, which is what #2836 assumed
        # it always would; the rate at which this is 0 is the #2846 finding.
        "evt_gumbel_is_low": nan if evt is None else int(evt.gumbel_is_low),
        "w_lo": nan,
        "mu_lo": nan,
        "var_lo": nan,
        "w_hi": nan,
        "mu_hi": nan,
        "var_hi": nan,
        "evt_w_gumbel": nan,
        "evt_loc": nan,
        "evt_scale": nan,
        "evt_mu": nan,
        "evt_var": nan,
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
        # Reported per *component* rather than per mode, because which one is the
        # low mode is an outcome of the fit (``evt_gumbel_is_low``) rather than a
        # property of the family.
        params.update(
            evt_w_gumbel=evt.w_gumbel,
            evt_loc=evt.loc,
            evt_scale=evt.scale,
            evt_mu=evt.mu,
            evt_var=evt.var,
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
) -> tuple[dict[str, float], dict[str, Any], dict[str, str]]:
    """All cut rules plus the label-reading diagnostics, from one sim sample.

    Returns ``(cuts, params, reasons)``: *cuts* maps every name in
    :data:`ALL_RULES` to a score (NaN where that rule has no root on this fit),
    *params* carries the fitted mixture parameters, the fit-quality comparison,
    and the supervised class moments, and *reasons* names why each EVT rule
    declined to fire - everything the analyzer needs to test the #2836/#2846
    predictions offline without re-running the simulation.
    """
    # The label-reading rules below index scores against labels, so normalize the
    # caller's sequence once here rather than at each use site.
    scores = np.asarray(sim_scores, dtype=float)
    gmm, evt, params = fit_both_mixtures(scores)
    nan = float("nan")
    cuts: dict[str, float] = dict.fromkeys(ALL_RULES, nan)
    # A rule whose fit never existed still needs a reason; "the fit failed" is a
    # different diagnosis from "the fit had no crossing" and the two must not be
    # pooled into one fallback count.
    reasons: dict[str, str] = dict.fromkeys(EVT_FIT_RULES, f"fit_{params['evt_fit_fail']}")

    # Label-free, so it is computed whether or not the single full-sample fit
    # succeeded: a bootstrap refit can find a mixture where one EM run on the
    # whole haystack did not, and vice versa.
    cuts.update(bagged_gaussian_fit_cuts(scores, gaussian_cuts, fpr_weight, fnr_weight))
    if gmm is not None:
        cuts.update(gaussian_cuts(gmm, fpr_weight, fnr_weight))
    else:
        # Mirror :func:`calculate_gmm_threshold`'s fallback exactly, so the
        # ``mid`` rule reproduces the shipped threshold even when EM fails - and
        # so it stays a usable fallback for the rules that have no root.
        cuts["mid"] = params["fallback_median"]
    if evt is not None:
        evt_cut_map, evt_reasons = evt_cuts(evt, fpr_weight, fnr_weight)
        cuts.update(evt_cut_map)
        reasons.update(evt_reasons)
        tail_cut_map, tail_reasons = tail_cuts(evt)
        cuts.update(tail_cut_map)
        reasons.update(tail_reasons)

    sup, sup_stats = supervised_cut(scores, sim_labels, fpr_weight, fnr_weight)
    params.update(sup_stats)
    cuts["supervised"] = sup
    cuts["sim_oracle"] = sim_oracle_cut(scores, sim_labels, fpr_weight, fnr_weight)
    # #2883: the same target read off less data, and off the same data with two
    # different regularisers.  Together these say whether the last link of the
    # chain is a bias the fit could remove or a variance the sample size sets.
    cuts.update(transfer_oracle_cuts(scores, sim_labels, fpr_weight, fnr_weight))

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
    return cuts, params, reasons
