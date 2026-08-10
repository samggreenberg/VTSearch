"""Inclusion-weighted FPR/FNR error metrics, shared across the eval subsystem.

These are the inclusion-weighted FP/FN quantities the region-curve sweep
(:mod:`vtscore.eval.region_curve`) and the oracle (:func:`min_weighted_cost`)
report, factored out so there is one implementation of the FP/FN →
weighted-cost computation. (Production's threshold *search* is the
split-conformal :func:`vtscore.training.thresholds.conformal_threshold`, which
picks a quantile rather than minimizing this cost; these metrics score whatever
operating point it lands on.)

Convention (matches :func:`inclusion_weights`):

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


def roc_auc(scores: FloatArray, labels: FloatArray) -> float:
    """Area under the ROC curve: P(score of a random positive > a random negative).

    Threshold-free, so unlike ``cost``/``f1`` it measures the **ranking** alone and is
    immune to where the cut lands - the companion metric for reading a run whose oracle
    is good but whose calibrated cut is not.

    Computed as the Mann-Whitney U statistic over **midranks**, which is exact under
    ties. Ties are not an edge case here: an image's score is a max-pool over its
    regions, so identical scores are common (and a degenerate head can make every score
    identical, which must read as 0.5 rather than 0 or 1). NaN when either class is
    absent, matching :func:`f1_at`'s "undefined" convention.
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.size == 0 or s.size != y.size:
        return float("nan")
    finite = np.isfinite(s)
    s, y = s[finite], y[finite]
    n_pos = int((y == 1.0).sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Midranks: average rank within each group of tied scores (scipy.rankdata "average").
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < sorted_s.size:
        j = i
        while j + 1 < sorted_s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based midrank of the tie block
        i = j + 1
    rank_sum_pos = float(ranks[y == 1.0].sum())
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return round(float(auc), 6)


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
    undefined). Vectorised.
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


def _cut_sweep(scores: FloatArray, labels: FloatArray) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    """Cumulative ``(TP, FP, P, N)`` at every **realizable** cut of ``scores``.

    Sorts descending and returns, for each cut point, the positives and negatives at or
    above it. Unlike :func:`min_weighted_cost` / :func:`max_f1` — which evaluate at every
    array position — cuts that would split a block of tied scores are dropped, since
    ``score >= threshold`` cannot separate equal scores. That matters because these
    scores are max-pools over regions, so ties are common: a degenerate head that emits
    one identical score everywhere has exactly ONE realizable cut ("predict all"), and a
    position-wise sweep would instead report a perfect ceiling it can never reach.

    Returns ``None`` when either class is absent (every rate-based metric is undefined).
    """
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if s.size == 0 or s.size != y.size:
        return None
    n_pos = int(np.sum(y == 1.0))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    cum_tp = np.cumsum(y_sorted == 1.0)
    cum_fp = np.cumsum(y_sorted == 0.0)
    # Keep position i only when the next score differs (or i is the last): otherwise the
    # cut falls inside a tie block and is not achievable by any threshold.
    keep = np.ones(s_sorted.size, dtype=bool)
    keep[:-1] = s_sorted[:-1] != s_sorted[1:]
    return cum_tp[keep], cum_fp[keep], n_pos, n_neg


def max_accuracy(scores: FloatArray, labels: FloatArray) -> float:
    """Oracle best-case accuracy: max over all realizable thresholds (peeks at labels).

    ``(TP + TN) / (P + N)``, including the predict-all-negative endpoint (accuracy
    ``N / (P + N)``), which is the one that wins under heavy imbalance. A true ceiling:
    the cross-calibrated accuracy can never exceed it. NaN when either class is absent.
    """
    sweep = _cut_sweep(scores, labels)
    if sweep is None:
        return float("nan")
    cum_tp, cum_fp, n_pos, n_neg = sweep
    correct = cum_tp + (n_neg - cum_fp)  # TP + TN
    best = max(float(correct.max()), float(n_neg))  # endpoint: predict everything negative
    return float(round(best / (n_pos + n_neg), 6))


def max_balanced_accuracy(scores: FloatArray, labels: FloatArray) -> float:
    """Oracle best-case balanced accuracy: max over all realizable thresholds.

    ``(TPR + TNR) / 2``, i.e. ``1 - (FPR + FNR) / 2``. Unlike :func:`max_accuracy` this is
    immune to prevalence, so the degenerate predict-all-negative endpoint scores 0.5 and
    never wins outright. Related to but distinct from :func:`min_weighted_cost`: they
    coincide only at ``inclusion == 0``, where ``max_balanced_accuracy == 1 - cost / 2``.
    NaN when either class is absent.
    """
    sweep = _cut_sweep(scores, labels)
    if sweep is None:
        return float("nan")
    cum_tp, cum_fp, n_pos, n_neg = sweep
    balanced = 0.5 * (cum_tp / n_pos + (n_neg - cum_fp) / n_neg)
    best = max(float(balanced.max()), 0.5)  # endpoint: predict everything negative
    return float(round(best, 6))


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
