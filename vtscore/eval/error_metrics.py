"""Inclusion-weighted FPR/FNR error metrics, shared across the eval subsystem.

These are the exact quantities the detector's threshold search
(:func:`vtscore.training.thresholds.find_optimal_threshold`) and the
voting-iterations evaluator already use, factored out so the region-curve
sweep (:mod:`vtscore.eval.region_curve`) reports the *same* metric and so
there is one implementation of the FP/FN → weighted-cost computation.

Convention (matches ``find_optimal_threshold`` / ``_inclusion_weights``):

    cost = fpr_weight * FPR + fnr_weight * FNR
    FPR  = FP / N_negatives      FNR = FN / N_positives      (RATES, not counts)

with the weights set by an integer ``inclusion`` (powers of two):

    inclusion == 0 -> (1, 1)                 => cost = FPR + FNR   (default, balanced)
    inclusion  > 0 -> (1, 2**inclusion)      => up-weight FNR  (favor recall / include more)
    inclusion  < 0 -> (2**(-inclusion), 1)   => up-weight FPR  (favor precision / exclude more)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Scores/labels may arrive as Python lists or numpy arrays (the region-curve
# max-pool returns arrays); accept either.
FloatArray = Sequence[float] | np.ndarray

_NAN_RESULT = {"cost": float("nan"), "fpr": float("nan"), "fnr": float("nan")}


def inclusion_weights(inclusion: int) -> tuple[float, float]:
    """Return ``(fpr_weight, fnr_weight)`` for a given inclusion value."""
    if inclusion >= 0:
        return 1.0, 2.0**inclusion
    return 2.0 ** (-inclusion), 1.0


def weighted_error(
    scores: FloatArray,
    labels: FloatArray,
    threshold: float,
    inclusion: int = 0,
) -> dict[str, float]:
    """Confusion-based ``{cost, fpr, fnr}`` for ``scores >= threshold`` vs ``labels``.

    ``labels`` are ``1.0`` (positive) / ``0.0`` (negative). Empty input yields a
    NaN result. Rates degrade gracefully to ``0.0`` when a class is absent (a
    class with no members contributes no error), matching the historical
    behaviour of ``voting_iterations._evaluate_on_test``.
    """
    if len(scores) == 0:
        return dict(_NAN_RESULT)

    total_pos = sum(1 for lbl in labels if lbl == 1.0)
    total_neg = len(labels) - total_pos

    fp = fn = 0
    for score, label in zip(scores, labels, strict=True):
        predicted = 1 if score >= threshold else 0
        if predicted == 1 and label == 0.0:
            fp += 1
        elif predicted == 0 and label == 1.0:
            fn += 1

    fpr = fp / total_neg if total_neg > 0 else 0.0
    fnr = fn / total_pos if total_pos > 0 else 0.0

    fpr_weight, fnr_weight = inclusion_weights(inclusion)
    cost = fpr_weight * fpr + fnr_weight * fnr
    return {"cost": round(cost, 6), "fpr": round(fpr, 6), "fnr": round(fnr, 6)}


def f1_at(scores: FloatArray, labels: FloatArray, threshold: float) -> float:
    """F1 of ``scores >= threshold`` vs ``labels`` (1/0). ``F1 = 2TP/(2TP+FP+FN)``.

    NaN when undefined (no predicted or actual positives). Same operating point
    as :func:`weighted_error`, so this is the F1 at the cross-calibrated threshold.
    """
    if len(scores) == 0:
        return float("nan")
    tp = fp = fn = 0
    for s, y in zip(scores, labels, strict=True):
        pred = s >= threshold
        if pred and y == 1.0:
            tp += 1
        elif pred and y == 0.0:
            fp += 1
        elif not pred and y == 1.0:
            fn += 1
    denom = 2 * tp + fp + fn
    return round(2 * tp / denom, 6) if denom > 0 else float("nan")


def min_weighted_cost(
    scores: FloatArray,
    labels: FloatArray,
    inclusion: int = 0,
) -> float:
    """Oracle best-case weighted cost: min over all thresholds of ``weighted_error``.

    Sweeps every candidate threshold (each unique score) plus the "predict all
    negative" endpoint, and returns the minimum inclusion-weighted cost. This
    "peeks" at the labels to place the threshold, so it is an optimistic
    upper-bound reference for the cross-calibrated (realistic) cost, never a
    production metric. Returns NaN when either class is absent (both rates
    undefined). Vectorised, mirroring ``find_optimal_threshold``.
    """
    if len(scores) == 0:
        return float("nan")

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)

    total_positives = int(np.sum(labels_arr == 1.0))
    total_negatives = int(labels_arr.size - total_positives)
    if total_positives == 0 or total_negatives == 0:
        return float("nan")

    order = np.argsort(-scores_arr)
    sorted_labels = labels_arr[order]

    fpr_weight, fnr_weight = inclusion_weights(inclusion)

    # At position i (threshold = i-th highest score) items 0..i are predicted
    # positive: FP = negatives seen so far, FN = positives not yet seen.
    cum_pos = np.cumsum(sorted_labels == 1.0)
    cum_neg = np.cumsum(sorted_labels == 0.0)
    fpr = cum_neg / total_negatives
    fnr = (total_positives - cum_pos) / total_positives
    costs = fpr_weight * fpr + fnr_weight * fnr

    # Endpoint: predict everything negative (threshold above the max score) ->
    # FPR = 0, FNR = 1. The cumulative sweep starts at "1 item positive", so add
    # this explicitly to cover the high-threshold extreme.
    predict_none_cost = fnr_weight * 1.0
    return float(round(min(float(costs.min()), predict_none_cost), 6))


def max_f1(scores: FloatArray, labels: FloatArray) -> float:
    """Oracle best-case F1: max over all thresholds of ``f1_at`` (peeks at labels).

    Sweeps every candidate threshold (each unique score) and returns the highest
    achievable F1. Unlike ``min_weighted_cost`` (which optimizes rate-based FPR+FNR),
    this optimizes F1 directly, so at extreme class imbalance the two pick very
    different thresholds — the min-cost point can have poor precision (many false
    positives in count) and thus poor F1. This is the true F1 **ceiling**: the
    cross-calibrated F1 can never exceed it. NaN when no positives exist. An
    optimistic reference, never a production metric.
    """
    if len(scores) == 0:
        return float("nan")

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    total_positives = int(np.sum(labels_arr == 1.0))
    if total_positives == 0:
        return float("nan")

    order = np.argsort(-scores_arr)
    sorted_labels = labels_arr[order]
    # At position i (threshold = i-th highest score) items 0..i are predicted
    # positive: TP = positives seen, FP = negatives seen, FN = positives not yet seen.
    cum_tp = np.cumsum(sorted_labels == 1.0)
    cum_fp = np.cumsum(sorted_labels == 0.0)
    denom = cum_tp + cum_fp + total_positives  # = 2*TP + FP + FN
    f1 = np.where(denom > 0, 2.0 * cum_tp / denom, 0.0)
    return float(round(float(f1.max()), 6))
