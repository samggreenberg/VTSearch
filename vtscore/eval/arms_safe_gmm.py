"""The safe-threshold cut-variant arms (issues #2799 / #2836).

One extra result row per cut variant per step, all scored against the same
held-out test scores as the shipped row beside them, plus the per-(step,
geometry) cut-decomposition frame that telescopes today's rule into its
assumptions.  :data:`_SAFE_GMM_VARIANTS` is the arm list, tagged in each row's
``gmm_variant`` column; :data:`_ORACLE_VARIANTS` names the ones that are oracles
rather than rules, which are read differently.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vtscore.eval.row_metrics import operating_metrics, round6
from vtscore.training.blend_schedules import BlendContext
from vtscore.training.thresholds import CUT_KIND_INTERIOR

#: Safe-threshold cut variants (issues #2799, #2836): ``(name, fit_scores, rule)``.
#: ``fit_scores`` picks which sim-set score distribution the mixture is fitted on
#: ("pooled" = the style's inference max-pool, what production fits post-#2797;
#: "image" = the whole-image vector scores, the historical pre-#2797 geometry).
#: ``rule`` names a cut in :mod:`vtscore.eval.cut_rules` - the ``lam``-tilt
#: family over the Gaussian mixture ("mid" is production; "cross" is #2798's
#: count-optimal crossing, reverted by #2833; "priorfree"/"rate" are #2836's
#: rate-optimal tilts), the same tilts over a Gumbel-low mixture ("gumbel_*") and
#: over one whose Gumbel may land on either mode ("gumbel_any_*", #2846's repair
#: to the first family's fallback rate), and the two label-reading diagnostics
#: ("supervised", "sim_oracle") that locate the error rather than compete to
#: ship.  ``xcal_only`` is the no-blend control: the conformal threshold at the
#: same step, or - on a step whose folds fell back - the same sentinel the
#: shipped blend feeds its x-cal side (:func:`_blend_xcal_input`), so the
#: control is the blend's own input with the mix-in removed rather than a cut
#: nobody computed.  ``pooled_mid`` must reproduce the production blend exactly.
#: The ``tail_a*`` sweep is #2881's one-constant rule at seven tail levels - the
#: fitted Bad component's own quantile rather than a crossing of any kind.
#:
#: **The rule names here are the ones in :mod:`vtscore.eval.cut_rules`**, spelled
#: out rather than derived, because this module is deliberately import-light (no
#: numpy at import time) and importing the rule tables to build the list would
#: undo that.  ``test_cut_rules`` asserts the two agree, so the duplication
#: cannot drift silently - which matters more than usual here, since a rule that
#: is defined but never emitted produces a table with a missing row rather than
#: an error.
#:
#: The #2798 logit-space variants are gone: #2799 measured them at +0.0006 cost
#: (dead) and each extra fit costs a step's CPU that the #2836 arms need.
_SAFE_GMM_VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("xcal_only", "", ""),
    ("image_mid", "image", "mid"),
    ("image_cross", "image", "cross"),
    ("image_priorfree", "image", "priorfree"),
    ("image_rate", "image", "rate"),
    ("pooled_mid", "pooled", "mid"),
    ("pooled_cross", "pooled", "cross"),
    ("pooled_priorfree", "pooled", "priorfree"),
    ("pooled_rate", "pooled", "rate"),
    ("pooled_gumbel_cross", "pooled", "gumbel_cross"),
    ("pooled_gumbel_priorfree", "pooled", "gumbel_priorfree"),
    ("pooled_gumbel_rate", "pooled", "gumbel_rate"),
    ("pooled_gumbel_any_cross", "pooled", "gumbel_any_cross"),
    ("pooled_gumbel_any_priorfree", "pooled", "gumbel_any_priorfree"),
    ("pooled_gumbel_any_rate", "pooled", "gumbel_any_rate"),
    ("pooled_tail_a040", "pooled", "tail_a040"),
    ("pooled_tail_a080", "pooled", "tail_a080"),
    ("pooled_tail_a110", "pooled", "tail_a110"),
    ("pooled_tail_a158", "pooled", "tail_a158"),
    ("pooled_tail_a220", "pooled", "tail_a220"),
    ("pooled_tail_a300", "pooled", "tail_a300"),
    ("pooled_tail_a400", "pooled", "tail_a400"),
    ("pooled_supervised", "pooled", "supervised"),
    ("pooled_sim_oracle", "pooled", "sim_oracle"),
    # #2883: the last link's shape.  Four subsample levels give the learning
    # curve in sim-set size (the test set and the trajectory are identical
    # across them - only the number of labelled sim scores moves), and two
    # variance-reduced estimators of the same target test whether the empirical
    # minimiser is the bound `family_headroom_exhausted` treats it as.
    ("pooled_sim_oracle_f050", "pooled", "sim_oracle_f050"),
    ("pooled_sim_oracle_f100", "pooled", "sim_oracle_f100"),
    ("pooled_sim_oracle_f250", "pooled", "sim_oracle_f250"),
    ("pooled_sim_oracle_f500", "pooled", "sim_oracle_f500"),
    ("pooled_sim_oracle_bag", "pooled", "sim_oracle_bag"),
    ("pooled_sim_oracle_smooth", "pooled", "sim_oracle_smooth"),
    # The label-free counterpart: bag the mixture fit rather than the labelled
    # cost curve.  Exploratory, not ship-gated - see PREREG.
    ("pooled_bagfit_mid", "pooled", "bagfit_mid"),
    ("pooled_bagfit_priorfree", "pooled", "bagfit_priorfree"),
)


#: Variants that read the sim set's true labels.  Reported for the decomposition,
#: never eligible to ship - a rule cannot see these labels in the app.
_ORACLE_VARIANTS: frozenset[str] = frozenset(
    {
        "pooled_supervised",
        "pooled_sim_oracle",
        # #2883's readings of the same sim set - label-reading for the same
        # reason and, like the two above, emitting NaN rather than falling back
        # to a midpoint under another rule's name.
        "pooled_sim_oracle_f050",
        "pooled_sim_oracle_f100",
        "pooled_sim_oracle_f250",
        "pooled_sim_oracle_f500",
        "pooled_sim_oracle_bag",
        "pooled_sim_oracle_smooth",
    }
)


def _safe_gmm_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    sim_scores_by_geometry: dict[str, list[float]],
    sim_labels_by_geometry: dict[str, "np.ndarray"],
    inclusion: int,
    n_pool_rows: float,
    schedule: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Metric rows per cut variant, plus the per-geometry decomposition rows.

    Every variant re-cuts the *same* per-step model: each candidate sim-set score
    distribution is fitted once (both mixture families), every cut rule reads that
    one fit, and each cut is blended with the step's conformal threshold on the
    production label ramp.  All variants are evaluated against the same held-out
    test scores (*base_scores*, the inference max-pool), so the rows are paired
    within a step by construction.

    Two costs are recorded per variant: ``cost`` at the *blended* threshold (what
    a user would get, and what the ship decision reads) and ``raw_cut_cost`` at
    the unblended cut (what the *rule* is worth, undamped by the conformal
    threshold it is averaged with).  On the ramp the blend can shrink a large cut
    difference to a small cost difference, so the rule comparison belongs on the
    raw column and the ship comparison on the blended one.

    A rule whose root does not exist on a given fit falls back to that fit's
    midpoint and is flagged in ``cut_fallback`` so the analyzer can exclude
    fallen-back steps from a rule's own contrast rather than silently scoring
    the midpoint under another name.  **The midpoint is this family's fallback,
    not production's** - the shipped ``rate`` rule
    (:func:`~vtscore.training.thresholds.gmm_cut_from_fit`) continues past the
    inter-mean interval at its own first-order slope instead, so on the fits
    where this flag fires these arms are measuring a different rule than the
    app runs.  That divergence is **kept on purpose** (issue #2900): this family
    compares tilts against each other on one fit, and a rule-independent
    stand-in is what keeps ``rate`` commensurable with the ``cross`` and
    ``priorfree`` siblings it is differenced against - at inclusion 0 it is what
    keeps ``rate`` bit-identical to ``priorfree``, which is how every report in
    ``docs/experiments/2026-08-04-gmm-cut/`` reads those rows.  It is no longer *invisible*
    though: ``cut_fallback_kind`` carries
    :data:`~vtscore.eval.cut_rules.CUT_KIND_MIDPOINT` on exactly these steps,
    against the production family's ``continued`` / ``degenerate_midpoint``, so
    an analysis that wants the shipped path can filter for it instead of reading
    a substituted midpoint as "what the app would have done".  The fold-anchored
    family below calls the production function directly and so does not have
    that gap at all.  For the EVT rules
    ``cut_fail_reason`` additionally names *which* guard declined, because the
    repairs those guards want are different and the counts alone cannot tell them
    apart (issue #2846).  The oracle variants do not fall back; they emit NaN cuts
    and are dropped by the analyzer's joins.

    Returns ``(variant_rows, diagnostic_rows)``; the diagnostic rows carry the
    fitted mixture parameters and every cut in the decomposition chain, one row
    per (step, geometry), and still need the caller's identifying columns.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost, oracle_cut  # noqa: PLC0415
    from vtscore.eval.cut_rules import CUT_KIND_MIDPOINT, decomposition_cuts  # noqa: PLC0415
    from vtscore.eval.transfer_rules import honest_test_oracle  # noqa: PLC0415
    from vtscore.training.thresholds import blend_gmm_threshold, safe_blend_weight  # noqa: PLC0415

    xcal = float(details["xcal_threshold"])
    n_votes = int(details["n_votes"])
    pre_blend_provenance = str(details.get("pre_blend_provenance", "conformal"))
    fold_orderings = details.get("fold_orderings") or []
    cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None
    wf, wn = inclusion_weights(inclusion)
    nan = float("nan")

    # One fit pass per geometry; every rule below reads these.
    cuts_by_geometry: dict[str, dict[str, float]] = {}
    reasons_by_geometry: dict[str, dict[str, str]] = {}
    diag_rows: list[dict[str, Any]] = []
    geometries = sorted({fit for _n, fit, round6 in _SAFE_GMM_VARIANTS if fit})
    for geometry in geometries:
        scores = sim_scores_by_geometry[geometry]
        labels = sim_labels_by_geometry[geometry]
        if len(scores) < 2:
            # Mirrors calculate_gmm_threshold's "too few scores" default so the
            # production-blend sanity check holds at every step.
            cuts_by_geometry[geometry] = dict.fromkeys((r for _n, f, r in _SAFE_GMM_VARIANTS if f == geometry), 0.5)
            reasons_by_geometry[geometry] = {}
            continue
        cuts, params, reasons = decomposition_cuts(scores, labels, wf, wn)
        cuts_by_geometry[geometry] = cuts
        reasons_by_geometry[geometry] = reasons
        diag = {
            "geometry": geometry,
            "sim_prevalence": round6(float(np.mean(labels))) if len(labels) else nan,
            # The count, not just the rate: a threshold estimated from labelled
            # scores is limited by the *rarer* class, so #2883's scaling claim is
            # about positives and prevalence alone cannot express it.
            "sim_n_pos": round6(float(np.sum(labels == 1.0))) if len(labels) else nan,
        }
        # ``evt_fit_fail`` is a reason string; everything else in params is numeric.
        diag.update({k: v if isinstance(v, str) else round6(float(v)) for k, v in params.items()})
        diag.update({f"tau_{name}": round6(float(value)) for name, value in cuts.items()})
        diag["tau_test_oracle"] = nan  # filled below, once the test oracle is known
        diag_rows.append(diag)

    ctx = BlendContext(
        n_labels=n_votes,
        n_good=int(details.get("n_good", 0)),
        n_bad=int(details.get("n_bad", n_votes)),
    )
    weight = safe_blend_weight(ctx, schedule)
    rows: list[dict[str, Any]] = []
    for name, geometry, rule in _SAFE_GMM_VARIANTS:
        fallback = 0
        fallback_kind = CUT_KIND_INTERIOR
        fail_reason = ""
        if name == "xcal_only":
            threshold = xcal
            gmm_cut = nan
            provenance = pre_blend_provenance
        else:
            gmm_cut = cuts_by_geometry[geometry][rule]
            if not np.isfinite(gmm_cut) and name not in _ORACLE_VARIANTS:
                gmm_cut = cuts_by_geometry[geometry].get("mid", nan)
                fallback = 1
                fallback_kind = CUT_KIND_MIDPOINT
                # Empty for the Gaussian rules, which have no reason vocabulary;
                # the EVT rules name the guard that declined so a fallback can be
                # attributed rather than merely counted (issue #2846).
                fail_reason = reasons_by_geometry[geometry].get(rule, "")
            threshold = blend_gmm_threshold(xcal, gmm_cut, ctx, schedule=schedule) if np.isfinite(gmm_cut) else nan
            provenance = "gmm_blend"
        if not np.isfinite(threshold):
            continue
        row = operating_metrics(
            base_scores,
            base_labels,
            threshold,
            inclusion,
            cal_scores,
            cal_labels,
            pool_variant="max",
            provenance=provenance,
            n_pool_rows=n_pool_rows,
        )
        row["gmm_variant"] = name
        row["schedule"] = ""
        row["xcal_threshold"] = round6(xcal)
        row["gmm_cut"] = round6(gmm_cut)
        row["blend_weight"] = round6(weight)
        row["cut_fallback"] = fallback
        row["cut_fallback_kind"] = fallback_kind
        row["cut_fail_reason"] = fail_reason
        if np.isfinite(gmm_cut):
            raw_cost, raw_fpr, raw_fnr = operating_cost(base_scores, base_labels, gmm_cut, wf, wn)
            row["raw_cut_cost"] = round6(raw_cost)
            row["raw_cut_fpr"] = round6(raw_fpr)
            row["raw_cut_fnr"] = round6(raw_fnr)
        rows.append(row)

    # The last link in the chain: the best cut on the held-out test set.  Read off
    # any emitted row (all share the same base_scores/base_labels oracle).
    if rows and diag_rows:
        # #2883: that cut is the argmin of the empirical cost on the test sample
        # *itself*, so its cost is a sample minimum - biased low, which biases
        # `transfer` high by however much the reference overfits.  Record the
        # cross-fitted version beside it (cut and cost on disjoint folds) so the
        # last link can be reported as a bracket instead of a point.
        honest_cost, honest_tau = honest_test_oracle(base_scores, base_labels, wf, wn)
        # From `oracle_cut` itself, NOT by re-scoring at `rows[0]["oracle_threshold"]`:
        # that column is rounded on the way out, and re-evaluating a cost at a
        # rounded threshold moves items across the boundary.  With ~55 test
        # positives one FNR step is 1/55 = 0.018 - half the size of the term this
        # study is measuring - so the rounding is not a rounding error here.
        _naive_tau, naive_cost, _nfpr, _nfnr = oracle_cut(base_scores, base_labels, wf, wn)
        n_test = float(np.asarray(base_labels).size)
        n_test_pos = float(np.sum(np.asarray(base_labels) == 1.0))
        for diag in diag_rows:
            diag["tau_test_oracle"] = rows[0]["oracle_threshold"]
            diag["tau_test_oracle_honest"] = round6(honest_tau)
            diag["cost_test_oracle_naive"] = round6(naive_cost)
            diag["cost_test_oracle_honest"] = round6(honest_cost)
            diag["test_n"] = round6(n_test)
            diag["test_n_pos"] = round6(n_test_pos)
    return rows, diag_rows
