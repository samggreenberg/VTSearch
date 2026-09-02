"""The two inclusion-budget sweeps.

:func:`_inclusion_sweep_rows` sweeps the *conformal* rule's budget and asks
whether its ``alpha(k)`` guarantee holds; :func:`_cut_inclusion_rows` (issue
#2865) sweeps the **fold-anchored** estimator's cut *rules* and asks which one
should answer the knob at all.  They write to separate side frames
(:data:`~vtscore.eval.voting_columns.INCLUSION_SWEEP_COLUMNS` and
:data:`~vtscore.eval.voting_columns.CUT_INCLUSION_COLUMNS`) because they answer
different questions about the same knob.

:func:`_cut_inclusion_arms` is pinned against
``vtscore.training.thresholds.fold_anchored_gmm_threshold`` by
`scripts/check-eval-app-sync.py`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from vtscore.eval.row_metrics import round6
from vtscore.training.thresholds import threshold_from_fold_orderings


def _inclusion_sweep_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    inclusion_sweep_ks: list[int],
) -> list[dict[str, Any]]:
    """Re-threshold the base fold orderings at each inclusion *k* and measure test FNR.

    Near-free (no refits): pools the cached fold orderings once and applies the
    conformal rule at each ``k``, then measures the realised test FPR/FNR at that
    cut.  Checks the Inclusion budget ``alpha(k) = 0.25 * 2^-k`` against the
    measured FNR under the **grouped** calibration path (issue #2781 / the
    grouped-arm follow-up in inclusion-calibration-bias.md).  Returns ``[]`` when
    the base threshold was a fallback (no real orderings to sweep).
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost  # noqa: PLC0415

    fold_orderings = details.get("fold_orderings") or []
    if not fold_orderings:
        return []
    base_scores = np.asarray(base_scores, dtype=np.float64)
    base_labels = np.asarray(base_labels, dtype=np.float64)
    out: list[dict[str, Any]] = []
    for k in inclusion_sweep_ks:
        thr_k = threshold_from_fold_orderings(fold_orderings, k)
        wf, wn = inclusion_weights(k)
        cost_k, fpr_k, fnr_k = operating_cost(base_scores, base_labels, thr_k, wf, wn)
        alpha_k = 0.25 * 2.0 ** (-k)
        out.append(
            {
                "inclusion_k": k,
                "alpha": round6(alpha_k),
                "sweep_threshold": round6(float(thr_k)),
                "sweep_fpr": round6(fpr_k),
                "sweep_fnr": round6(fnr_k),
                "excess_fnr": round6(fnr_k - alpha_k),
            }
        )
    return out


def _cut_inclusion_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    fold_haystacks: list,
    sim_scores: list[float],
    ks: list[int],
    weights: list[float],
    rules: list[str],
    fold_combines: list[str],
    qtilt_steps: list[float],
) -> list[dict[str, Any]]:
    """Sweep the fold-anchored cut *rules* across the whole Inclusion knob (#2865).

    The shipped ``mid`` cut was picked by two calibration runs that scored every
    arm at inclusion 0, and a bare midpoint ignores the cost weights inclusion
    arrives as - so it made the knob a no-op for every detector with usable
    folds.  ``mid_tilt`` was shipped to restore the tilt while reproducing the
    measured arm exactly at 0, but the *tilt* has never been priced against its
    alternatives.  This frame is that measurement.

    **Nearly free, and faithful for the same reason.**  The expensive part of a
    fold-anchored threshold is the per-fold anchored EM, and it does not depend
    on the cut rule, the combine, or the inclusion - so the fit is taken **once
    per anchor weight** via the app's own :func:`fit_fold_anchored_cut`, and
    every (rule, combine, k) point re-cuts it through the app's own
    :meth:`~vtscore.training.thresholds.FoldAnchoredCut.threshold_at`.  That is
    not merely cheaper than calling
    :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold` per point,
    it is exactly what production does when the user drags the slider
    (:func:`vtscore.state.core.recompute_detector_thresholds_for_inclusion`
    re-cuts a cached estimator with no refit), so the sweep measures the object
    the app actually re-cuts rather than a chain of independent retrains.

    A weight whose fit fails outright contributes no rows.  The terminal
    fallbacks :func:`fold_anchored_gmm_threshold` applies in that case
    (final-model unanchored midpoint, then its median) are deliberately *not*
    reproduced here: both are inclusion-blind by construction, so they would
    enter this frame as arms that trivially lose the knob-liveness comparison
    while saying nothing about the rule under test.

    Every row is scored under the cost weights of **its own** ``k`` and against
    the oracle cut at that same ``k`` - the run's reporting inclusion does not
    enter - so regret is comparable along the knob as well as across arms.

    *sim_scores* arrives under the #3308 population convention (the caller
    passes the voted-items-dropped haystack when it has one), matching the
    already-filtered *fold_haystacks*, so every re-cut here realizes its
    quantile on the same population the shipped cut does.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost, oracle_cut  # noqa: PLC0415

    fold_orderings = details.get("fold_orderings") or []
    if not fold_orderings or not fold_haystacks or not ks:
        return []

    n_folds = min(len(fold_haystacks), len(fold_orderings))
    fold_hay, orderings = fold_haystacks[:n_folds], fold_orderings[:n_folds]
    final_scores = np.asarray(sim_scores, dtype=np.float64)
    base_scores = np.asarray(base_scores, dtype=np.float64)
    base_labels = np.asarray(base_labels, dtype=np.float64)
    n_test = int(base_scores.size)

    # The oracle depends only on k, not on the arm - hoist it out of the arm
    # loop so a 5-rule x 8-weight grid pays for it once per k rather than 40x.
    oracle_by_k = {}
    for k in ks:
        wf, wn = inclusion_weights(k)
        o_thr, o_cost, _o_fpr, _o_fnr = oracle_cut(base_scores, base_labels, wf, wn)
        oracle_by_k[k] = (wf, wn, o_thr, o_cost)

    arms = _cut_inclusion_arms(fold_hay, orderings, final_scores, weights, rules, fold_combines, qtilt_steps)

    out: list[dict[str, Any]] = []
    for arm, arm_cut in arms:
        for k in ks:
            wf, wn, o_thr, o_cost = oracle_by_k[k]
            thr = arm_cut.threshold_at(k)
            if not np.isfinite(thr):
                continue
            cost, fpr, fnr = operating_cost(base_scores, base_labels, thr, wf, wn)
            n_admitted = int(np.count_nonzero(base_scores >= thr))
            out.append(
                {
                    **arm,
                    "inclusion_k": k,
                    "fold_quantile": round6(arm_cut.quantile_at(k)),
                    "cut_threshold": round6(float(thr)),
                    "cut_cost": round6(cost),
                    "cut_fpr": round6(fpr),
                    "cut_fnr": round6(fnr),
                    "k_oracle_threshold": round6(float(o_thr)),
                    "k_oracle_cost": round6(o_cost),
                    "cut_regret": round6(cost - o_cost),
                    "admitted_frac": round6(n_admitted / n_test if n_test else 0.0),
                    "n_admitted": n_admitted,
                    "n_test": n_test,
                }
            )
    return out


def _cut_inclusion_arms(
    fold_hay: list,
    orderings: list[tuple[list[float], list[float]]],
    final_scores: "np.ndarray",
    weights: list[float],
    rules: list[str],
    fold_combines: list[str],
    qtilt_steps: list[float],
) -> "list[tuple[dict[str, Any], Any]]":
    """The ``(identity columns, re-cut estimator)`` pairs the #2865 sweep scores.

    One anchored fit per *weight* - the only expensive step, and the only one
    that depends on none of the swept axes - re-cut into every (rule, combine,
    step) arm by :func:`dataclasses.replace`.  A weight whose fit fails
    contributes no arms.
    """
    from vtscore.training.thresholds import fit_fold_anchored_cut  # noqa: PLC0415

    arms: list[tuple[dict[str, Any], Any]] = []
    for weight in weights:
        cut = fit_fold_anchored_cut(fold_hay, orderings, final_scores, anchor_weight=weight)
        if cut is None:
            continue
        for rule in rules:
            # ``q_tilt`` alone carries a free step size, so it is the only rule
            # that expands over ``qtilt_steps``; the others take a NaN step so
            # the whole frame keeps one row shape.
            steps = list(qtilt_steps) if rule == "q_tilt" else [float("nan")]
            for combine in fold_combines:
                for step in steps:
                    arm_cut = replace(cut, cut_rule=rule, combine=combine)
                    name = f"fold_anchored_w{weight:g}_{rule}_{combine}"
                    if rule == "q_tilt":
                        arm_cut = replace(arm_cut, qtilt_step=step)
                        name += f"_s{step:g}"
                    ident = {
                        "arm": name,
                        "cut_rule": rule,
                        "anchor_weight": weight,
                        "combine": combine,
                        "qtilt_step": round6(step),
                    }
                    arms.append((ident, arm_cut))
    return arms
