"""Decision-threshold computation for learned-sort scores.

GMM-, cross-calibration-, and safe-threshold helpers. These are media-
agnostic: they take score lists and label lists and return a single
float threshold. Detector-specific glue (sourcing ``X_list`` / ``y_list``
from votes, caching on ``DetectorContext``) lives in
:mod:`vtscore.detectors`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Sentinel threshold meaning "predict nothing as Good". Sigmoid scores are
# in [0, 1], so any value > 1.0 makes every ``score >= threshold`` check
# evaluate to False. Kept finite (vs. ``float("inf")``) so it cannot poison
# downstream blends — ``0.0 * inf`` evaluates to NaN, which would then be
# stored on ``DetectorContext.threshold`` and break every comparison.
NO_GOOD_THRESHOLD = 2.0


def calculate_gmm_threshold(scores: list[float]) -> float:
    """Use a Gaussian Mixture Model to find a threshold between two score distributions.

    Fits a 2-component GMM to the provided scores, assuming a bimodal distribution
    representing Bad (low) and Good (high) classes. Returns the midpoint between the
    two component means as the decision threshold.

    Args:
        scores: List of model confidence scores, expected to follow a bimodal distribution.

    Returns:
        A float threshold. Scores at or above this value are classified as Good.
        Falls back to the median of scores if GMM fitting fails or fewer than 2 scores
        are provided.
    """
    if len(scores) < 2:
        return 0.5

    from sklearn.mixture import GaussianMixture  # noqa: PLC0415

    # Reshape for sklearn
    X = np.array(scores).reshape(-1, 1)

    try:
        # Fit a 2-component GMM
        gmm: GaussianMixture = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(X)

        # Get the means of the two components.  The stub types `means_`
        # as `np.ndarray | None`; after `fit` it's always set.
        assert gmm.means_ is not None
        means = np.ravel(gmm.means_)

        # Identify which component is "low" (Bad) and which is "high" (Good)
        low_idx = 0 if means[0] < means[1] else 1
        high_idx = 1 - low_idx

        # Threshold is at the intersection of the two Gaussians
        # For simplicity, use the midpoint between means
        threshold = (means[low_idx] + means[high_idx]) / 2.0

        return float(threshold)
    except Exception:
        # If GMM fails, return median
        return float(np.median(scores))


def _calibration_cache_key(
    X_list: list,
    y_list: list[float],
    inclusion_value: int,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
) -> tuple:
    """Build a deterministic cache key for cross-calibration inputs.

    ``calculate_cross_calibration_threshold`` is a deterministic function of
    these inputs (the RNG is seeded with 42 at every call site that uses the
    cache), so two calls with matching keys must produce the same threshold.
    The key encodes the raw training vectors (not just the label IDs) so that
    a labelset re-resolved to different embeddings — e.g. after the embedder
    changes — invalidates the cache automatically.
    """
    X_bytes = np.stack(X_list).astype(np.float32, copy=False).tobytes()
    y_bytes = np.asarray(y_list, dtype=np.float32).tobytes()
    return (
        X_bytes,
        y_bytes,
        int(inclusion_value),
        int(calibrate_count),
        float(calibration_fraction),
        int(hidden_dim),
    )


def cross_calibration_threshold_cached(
    X_list: list,
    y_list: list[float],
    input_dim: int,
    inclusion_value: int,
    *,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
    det_ctx: Any = None,
) -> float:
    """Memoized wrapper around :func:`calculate_cross_calibration_threshold`.

    When *det_ctx* is provided, stores the last computed threshold on
    ``det_ctx.calibration_cache`` and reuses it on the next call when every
    input matches.  This is the common case during interactive sorting: the
    user toggles ``inclusion`` or loads a new media item, the labels stay
    the same, and recomputing two ~200-epoch fold fits would produce the
    same number we computed last time.

    A real label change produces a different cache key and falls through
    to a fresh calibration — no explicit invalidation needed.
    """
    if det_ctx is not None:
        key = _calibration_cache_key(
            X_list,
            y_list,
            inclusion_value,
            calibrate_count,
            calibration_fraction,
            hidden_dim,
        )
        cached = getattr(det_ctx, "calibration_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
    else:
        key = None

    rng = np.random.RandomState(42)
    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        inclusion_value,
        rng=rng,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
    )

    if det_ctx is not None and key is not None:
        det_ctx.calibration_cache = (key, threshold)

    return threshold


def find_optimal_threshold(
    scores: list[float],
    labels: list[float],
    inclusion_value: int = 0,
) -> float:
    """Find the score threshold that best separates good (1) from bad (0) examples.

    Iterates over all candidate thresholds (each unique score value) and picks the
    one that minimises a weighted combination of false-positive rate (FPR) and
    false-negative rate (FNR). The relative weight of FPR vs. FNR is governed by
    ``inclusion_value``.

    Args:
        scores: List of model output scores, one per example.
        labels: List of true binary labels (1.0 for good, 0.0 for bad),
            corresponding to ``scores``.
        inclusion_value: Integer in ``[-10, 10]`` controlling the FPR/FNR trade-off.
            - 0: minimise ``fpr + fnr`` (equal weight).
            - Positive: minimise ``fpr + 2^inclusion_value * fnr`` (prefer recall,
              i.e., include more items).
            - Negative: minimise ``2^(-inclusion_value) * fpr + fnr`` (prefer
              precision, i.e., exclude more items).

    Returns:
        The float threshold that achieves the lowest weighted cost.
        Defaults to 0.5 if the score list is empty.
    """
    if not scores:
        return 0.5

    # Vectorized O(n log n) threshold search using cumulative sums
    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    # Sort by score descending
    order = np.argsort(-scores_arr)
    sorted_scores = scores_arr[order]
    sorted_labels = labels_arr[order]

    total_positives = int(np.sum(sorted_labels == 1))
    total_negatives = len(sorted_labels) - total_positives

    if total_positives == 0 or total_negatives == 0:
        return 0.5

    # Calculate weights based on inclusion
    if inclusion_value >= 0:
        fpr_weight = 1.0
        fnr_weight = 2.0**inclusion_value
    else:
        fpr_weight = 2.0 ** (-inclusion_value)
        fnr_weight = 1.0

    # Cumulative counts as we move the threshold down the sorted list.
    # At position i, threshold = sorted_scores[i], so items 0..i are predicted positive.
    cum_positives = np.cumsum(sorted_labels == 1)  # TP at each threshold
    cum_negatives = np.cumsum(sorted_labels == 0)  # FP at each threshold

    # FP = cum_negatives, FN = total_positives - cum_positives
    fp = cum_negatives
    fn = total_positives - cum_positives

    fpr = fp / total_negatives
    fnr = fn / total_positives

    costs = fpr_weight * fpr + fnr_weight * fnr

    best_idx = int(np.argmin(costs))
    return float(sorted_scores[best_idx])


def calculate_cross_calibration_threshold(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    inclusion_value: int = 0,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
) -> float:
    """Estimate a decision threshold using k-fold calibration.

    Performs ``calibrate_count`` independent random Train/Calibrate splits.
    For each split, trains a model on the Train portion and finds the
    optimal threshold on the Calibrate portion. Returns the mean of all
    thresholds.

    Algorithm:
        For each of *k* = ``calibrate_count`` rounds:
        1. Stratified random split into Train (``1 - calibration_fraction``)
           and Calibrate (``calibration_fraction``).  Stratification guarantees
           the Train side has at least one of each class, so the per-fold MLP
           fit always has both-class supervision.
        2. Train a model on Train.
        3. Find optimal threshold on Calibrate.
        Return mean of all *k* thresholds.

    Args:
        X_list: List of embedding arrays (one per labelled example).
        y_list: List of binary labels (1.0 for good, 0.0 for bad),
            aligned with ``X_list``.
        input_dim: Dimensionality of the embeddings.
        inclusion_value: Integer in ``[-10, 10]`` passed to :func:`train_model`
            and :func:`find_optimal_threshold` to control the FPR/FNR trade-off.
        rng: Optional seeded RandomState for reproducible splits. Falls back
            to the global ``np.random`` state when ``None``.
        calibrate_count: Number of random Train/Calibrate splits (default 2).
        calibration_fraction: Fraction of data used for calibration in each
            split (default 0.5).  For example, 0.2 means 80% Train / 20%
            Calibrate.  If the fraction is so extreme that a valid split
            cannot be formed (fewer than 2 training or 1 calibration
            examples), returns :data:`NO_GOOD_THRESHOLD` so that nothing
            is predicted as Good.
        hidden_dim: Force a specific hidden-layer width for the fold models.
            When ``None`` (default), each fold model auto-sizes based on its
            own training-set size.  Pass the full-data hidden dim to ensure
            fold models match the final model's architecture.

    Returns:
        A float threshold. Returns 0.5 when calibration is not possible:
        fewer than 4 examples total, or fewer than 2 of either class
        (stratified splitting needs at least one of each class on both
        the train and calibrate sides).  Returns :data:`NO_GOOD_THRESHOLD`
        (a finite sentinel above the sigmoid range) if
        ``calibration_fraction`` makes a valid split impossible.
    """
    n = len(X_list)
    if n < 4:
        return 0.5

    _rng = rng if rng is not None else np.random
    X_np = np.array(X_list)
    y_np = np.array(y_list)

    # Split sizes: calibration_fraction of n goes to calibrate, rest to train
    n_cal = max(1, round(n * calibration_fraction))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return NO_GOOD_THRESHOLD

    # Stratify by class so every fold's train side has both classes — an
    # unstratified random split could produce a single-class y_train on
    # small or skewed labelsets, and ``train_model`` (correctly) refuses
    # to fit BCE on that.  Needs at least two of each class to guarantee
    # the train side stays mixed; below that we cannot calibrate
    # reliably and fall back to the neutral 0.5 sentinel.
    pos_idx = np.where(y_np == 1.0)[0]
    neg_idx = np.where(y_np == 0.0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return 0.5

    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    calibrate_count = max(1, calibrate_count)
    thresholds: list[float] = []

    # Per-class train counts proportional to ``n_train / n`` but clamped to
    # ``[1, class_total - 1]`` so the train side keeps at least one of each
    # class (the H6 invariant) and the cal side is non-empty for at least
    # one class.
    def _per_class_n_train(class_total: int) -> int:
        target = round(class_total * n_train / n)
        return max(1, min(class_total - 1, target))

    n_train_pos = _per_class_n_train(len(pos_idx))
    n_train_neg = _per_class_n_train(len(neg_idx))

    for _ in range(calibrate_count):
        pos_perm = _rng.permutation(pos_idx)
        neg_perm = _rng.permutation(neg_idx)
        train_idx = np.concatenate([pos_perm[:n_train_pos], neg_perm[:n_train_neg]])
        cal_idx = np.concatenate([pos_perm[n_train_pos:], neg_perm[n_train_neg:]])

        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32)

        model = train_model(X_train, y_train, input_dim, inclusion_value, hidden_dim=hidden_dim)

        with torch.no_grad():
            X_cal = X_cal.to(next(model.parameters()).device)
            scores = torch.sigmoid(model(X_cal)).squeeze(1).cpu().tolist()
        t = find_optimal_threshold(scores, y_np[cal_idx].tolist(), inclusion_value)
        thresholds.append(t)

    return sum(thresholds) / len(thresholds)


def calculate_safe_threshold(
    xcal_threshold: float,
    all_scores: list[float],
    n_labels: int,
) -> float:
    """Blend cross-calibration and GMM thresholds for robustness with small label counts.

    When few labels are available the cross-calibration threshold can be unreliable.
    This function computes a GMM-based threshold on the full score distribution and
    returns a weighted average of the two, where the weight assigned to x-cal grows
    linearly with the number of labels.

    Blending rules:
        * ``n_labels < 6``  → pure GMM threshold.
        * ``n_labels >= 20`` → pure x-cal threshold.
        * In between → linear interpolation.

    Args:
        xcal_threshold: The cross-calibrated threshold.
        all_scores: Model output scores for all medias (used for GMM fitting).
        n_labels: Total number of labelled examples (good + bad).

    Returns:
        A finite blended threshold float. If either input is non-finite,
        falls back to the other; if both are non-finite, returns ``0.5``.
        The result is guaranteed finite so it can be safely stored on
        ``DetectorContext.threshold`` without breaking ``score >= threshold``
        comparisons.
    """
    import math  # noqa: PLC0415

    gmm_threshold = calculate_gmm_threshold(all_scores)

    # Defend against non-finite inputs from either side: an upstream
    # ``calculate_cross_calibration_threshold`` can theoretically still
    # surface inf/NaN, and ``calculate_gmm_threshold`` returns NaN when
    # the model produced non-finite scores. Without these guards the blend
    # below would store NaN on ``DetectorContext.threshold`` and silently
    # break every ``score >= threshold`` comparison downstream.
    xcal_finite = math.isfinite(xcal_threshold)
    gmm_finite = math.isfinite(gmm_threshold)
    if not xcal_finite and not gmm_finite:
        return 0.5
    if not xcal_finite:
        return gmm_threshold
    if not gmm_finite:
        return xcal_threshold

    # Linear ramp: 0 at 6 labels, 1 at 20 labels
    MIN_LABELS = 6
    MAX_LABELS = 20
    label_weight = max(0.0, min(1.0, (n_labels - MIN_LABELS) / (MAX_LABELS - MIN_LABELS)))

    blended = label_weight * xcal_threshold + (1.0 - label_weight) * gmm_threshold
    if not math.isfinite(blended):
        return 0.5
    return blended
