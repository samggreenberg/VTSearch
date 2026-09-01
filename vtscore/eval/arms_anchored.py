"""The anchored-mixture arms (issue #2852).

Cuts fitted with the fold anchor pulling on the mixture, swept over anchor
weight, cut rule and fold-combining rule, and paired against the same held-out
test scores as the shipped row (and as the ``pooled_mid`` / ``xcal_only``
variants in :mod:`vtscore.eval.arms_safe_gmm`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from vtscore.eval.row_metrics import operating_metrics, round6
from vtscore.training.thresholds import (
    CUT_KIND_INTERIOR,
    anchored_gmm_fit,
    fold_anchored_gmm_threshold,
    gmm_cut_from_fit,
    rank_transfer,
)

#: Default sweep grid for the anchored-mixture eval arms (issue #2852).  Each
#: (anchor_weight, cut rule[, fold combine]) combination is one paired
#: within-step variant; the GRID run overrides these via
#: ``simulate_voting_iterations``'s ``anchored_*`` parameters to exhaust the
#: grid registered in ``docs/plans/population-anchored-calibration.md``.
_ANCHORED_WEIGHTS: tuple[float, ...] = (1.0, 10.0, 100.0)


_ANCHORED_RULES: tuple[str, ...] = ("mid", "rate")


_ANCHORED_FOLD_COMBINES: tuple[str, ...] = ("qmean",)


def _anchored_variant_rows(
    details: dict[str, Any],
    base_scores: "np.ndarray",
    base_labels: "np.ndarray",
    sim_scores: list[float],
    sim_ids: list[int],
    good_ids: list[int],
    bad_ids: list[int],
    fold_haystacks: list,
    inclusion: int,
    n_pool_rows: float,
    weights: list[float],
    rules: list[str],
    fold_combines: list[str],
    fold_anchored: bool,
    sim_fit_scores: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Metric rows for the anchored-mixture threshold arms (issue #2852).

    Three families, all evaluated against the same held-out test scores as the
    #2799 variants so every row is step-paired with the shipped blend
    (``pooled_mid``) and pure x-cal (``xcal_only``):

    * ``anchored_w{W}_{rule}`` - the **label-anchored** mixture: anchored EM on
      the final model's sim-set (haystack) scores with the voted items' own
      final-model scores clamped to their labelled component.  One EM per
      anchor weight; each cut rule re-cuts the same fit.
    * ``fold_anchored_w{W}_{rule}_{combine}`` - the **fold-anchored**
      ("cross-LabeledGMM") repair: per calibration fold, anchored EM on that
      fold model's haystack scores with that fold's held-out labelled scores
      as anchors (honest anchors, one shared scale), each fold's cut carried
      to the final model by rank transfer and combined in quantile space.
    * ``rank_transfer`` - the conformal x-cal cut carried from the pooled fold
      haystack distribution to the final model's as a quantile: the
      scale-transfer-only arm that attributes H1 (see the plan).

    Anchored thresholds are used **raw** - the estimator replaces the blend
    rather than feeding it - so ``blend_weight`` is NaN and ``raw_cut_*``
    equals the headline cost columns.  The estimator path actually taken
    (anchored / unanchored fallback / fold tally) is recorded in
    ``threshold_provenance``.

    *fold_haystacks* are the per-fold sim-set score arrays the shipped
    threshold already computed (:func:`_safe_threshold_for_step`); the fold
    family re-cuts them rather than paying the scoring passes twice.  The grid
    point at the production ``(κ, rule, combine)`` therefore reproduces the
    step's own shipped cut exactly - the grid's *other* points are the
    deviation under test.
    """
    import numpy as np  # noqa: PLC0415

    from vtscore.eval.calibration_metrics import inclusion_weights, operating_cost  # noqa: PLC0415

    xcal = float(details["xcal_threshold"])
    fold_orderings = details.get("fold_orderings") or []
    cal_scores = np.array([s for scores, _ in fold_orderings for s in scores]) if fold_orderings else None
    cal_labels = np.array([lb for _, labels_ in fold_orderings for lb in labels_]) if fold_orderings else None
    wf, wn = inclusion_weights(inclusion)
    nan = float("nan")

    score_by_id = dict(zip(sim_ids, sim_scores, strict=True))
    anchor_scores = [score_by_id[cid] for cid in (*good_ids, *bad_ids) if cid in score_by_id]
    anchor_labels = [1.0] * sum(1 for cid in good_ids if cid in score_by_id) + [0.0] * sum(
        1 for cid in bad_ids if cid in score_by_id
    )
    # The #3308 population convention: the voted items anchor the fits, so they
    # are dropped from the free haystack sample instead of sitting in it twice
    # (once free, once clamped) - matching the shipped cut and the (already
    # filtered) *fold_haystacks* the fold family re-cuts.  *sim_fit_scores*
    # carries the caller's floor decision (resolve_exclusion_floor), so this
    # family can never diverge from the shipped population convention.
    final_scores = np.asarray(sim_fit_scores if sim_fit_scores is not None else sim_scores, dtype=np.float64)

    rows: list[dict[str, Any]] = []

    def emit(name: str, threshold: float, provenance: str, cut_kind: str) -> None:
        """Record one anchored arm.  *cut_kind* is the production rule's own
        ``cut_fallback_kind`` (:func:`~vtscore.training.thresholds.gmm_cut_from_fit`),
        so these rows say ``continued`` / ``degenerate_midpoint`` where the
        decomposition family says ``midpoint`` - the two substitute different
        values on the same fits (issue #2900).
        """
        if not np.isfinite(threshold):
            return
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
        row["xcal_threshold"] = round6(xcal)
        row["gmm_cut"] = round6(threshold)
        row["blend_weight"] = nan
        row["cut_fallback"] = int(bool(cut_kind))
        row["cut_fallback_kind"] = cut_kind
        raw_cost, raw_fpr, raw_fnr = operating_cost(base_scores, base_labels, threshold, wf, wn)
        row["raw_cut_cost"] = round6(raw_cost)
        row["raw_cut_fpr"] = round6(raw_fpr)
        row["raw_cut_fnr"] = round6(raw_fnr)
        rows.append(row)

    # --- Label-anchored family: one anchored EM per weight, re-cut per rule. ---
    for weight in weights:
        fit, provenance = anchored_gmm_fit(final_scores, anchor_scores, anchor_labels, anchor_weight=weight)
        if fit is None:
            continue
        for rule in rules:
            if rule in ("mid_tilt", "q_tilt"):
                # Fold-level rules, defined in fold-quantile space: a single
                # label-anchored fit has no folds to tilt across.  The fold
                # family below sweeps them; here they are skipped rather than
                # fed to gmm_cut_from_fit, which (correctly) rejects them.
                continue
            cut, cut_kind = gmm_cut_from_fit(fit, rule, wf, wn)
            emit(f"anchored_w{weight:g}_{rule}", cut, provenance, cut_kind)

    # --- Fold-anchored family + the rank-transfer attribution arm. ---
    if fold_anchored and fold_haystacks and fold_orderings:
        _emit_fold_anchored_rows(
            emit,
            xcal,
            fold_haystacks,
            fold_orderings,
            final_scores,
            inclusion,
            weights,
            rules,
            fold_combines,
        )

    return rows


def _emit_fold_anchored_rows(
    emit: Callable[[str, float, str, str], None],
    xcal: float,
    fold_haystacks: list,
    fold_orderings: list[tuple[list[float], list[float]]],
    final_scores: "np.ndarray",
    inclusion: int,
    weights: list[float],
    rules: list[str],
    fold_combines: list[str],
) -> None:
    """Emit the fold-family arm rows over pre-computed per-fold sim scores.

    The scoring pass per fold model is the fold arms' whole marginal cost, and
    the shipped threshold already paid it (see :func:`_safe_threshold_for_step`),
    so the grid re-cuts those arrays rather than re-scoring per grid point.
    ``rank_transfer`` reuses them too: the conformal cut carried from the pooled
    fold haystack distribution to the final model's - the scale-transfer-only
    attribution arm of the plan.

    Every grid point goes through the same
    :func:`~vtscore.training.thresholds.fold_anchored_gmm_threshold` the app
    ships; the grid *is* the deviation under test, so the arm at the production
    (κ, rule, combine) reproduces the shipped cut exactly.

    These rows carry no ``cut_fallback_kind``: a fold-anchored threshold is
    composed from one per-fit cut *per fold* in quantile space, so there is no
    single fit whose fallback branch the row could name.  A per-fold breakdown
    would be a different column with a different unit of observation, and the
    ``mid_tilt`` rule the app ships already degrades a rate-less fold to plain
    ``mid`` rather than to a substituted value (see
    :meth:`~vtscore.training.thresholds.FoldAnchoredCut._quantile_at`).
    """
    import numpy as np  # noqa: PLC0415

    n_folds = min(len(fold_haystacks), len(fold_orderings))
    fold_hay, orderings = fold_haystacks[:n_folds], fold_orderings[:n_folds]

    emit(
        "rank_transfer",
        rank_transfer(xcal, np.concatenate(fold_hay), final_scores),
        "rank_transfer",
        CUT_KIND_INTERIOR,
    )
    for weight in weights:
        for rule in rules:
            for combine in fold_combines:
                threshold, provenance = fold_anchored_gmm_threshold(
                    fold_hay,
                    orderings,
                    final_scores,
                    inclusion,
                    anchor_weight=weight,
                    cut_rule=rule,
                    combine=combine,
                )
                emit(f"fold_anchored_w{weight:g}_{rule}_{combine}", threshold, provenance, CUT_KIND_INTERIOR)
