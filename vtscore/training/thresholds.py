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
# downstream blends - ``0.0 * inf`` evaluates to NaN, which would then be
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
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
) -> tuple:
    """Build a deterministic cache key for the calibration **fold orderings**.

    The orderings (per-fold held-out scores + labels) are a deterministic
    function of these inputs (RNG seeded with 42 at every cached call site)
    and are **inclusion-independent** - ``inclusion`` is deliberately *not* in
    the key, so an Inclusion change hits the cache and only re-runs the cheap
    min-cost search.  The key encodes the raw training vectors (not just label
    IDs) so a labelset re-resolved to different embeddings - e.g. after the
    embedder changes - invalidates the cache automatically.  See
    docs/plans/find-verification-workflow.md.
    """
    X_bytes = np.stack(X_list).astype(np.float32, copy=False).tobytes()
    y_bytes = np.asarray(y_list, dtype=np.float32).tobytes()
    return (
        X_bytes,
        y_bytes,
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

    When *det_ctx* is provided, caches the inclusion-independent **fold
    orderings** on ``det_ctx.calibration_cache`` as ``(key, (orderings,
    fallback))`` and reuses them whenever the (labels, calibrate settings)
    key matches.  This is the common case during interactive sorting: the
    user toggles ``inclusion`` or loads a new media item, the labels stay the
    same, and the only work left is re-running the cheap min-cost search over
    the cached orderings - no ~200-epoch fold fits.

    A real label change produces a different cache key and falls through to a
    fresh calibration - no explicit invalidation needed.
    """
    payload: tuple[list[tuple[list[float], list[float]]], float | None] | None = None
    key = None
    if det_ctx is not None:
        key = _calibration_cache_key(X_list, y_list, calibrate_count, calibration_fraction, hidden_dim)
        cached = getattr(det_ctx, "calibration_cache", None)
        if cached is not None and cached[0] == key:
            payload = cached[1]

    if payload is None:
        rng = np.random.RandomState(42)
        payload = compute_fold_orderings(
            X_list,
            y_list,
            input_dim,
            rng=rng,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
        )
        if det_ctx is not None and key is not None:
            det_ctx.calibration_cache = (key, payload)

    orderings, fallback = payload
    if fallback is not None:
        return fallback
    return threshold_from_fold_orderings(orderings, inclusion_value)


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


def compute_fold_orderings(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Train the K calibration folds and return their held-out orderings.

    Each ordering is a ``(cal_scores, cal_labels)`` pair: the fold model's
    sigmoid scores on its held-out calibration split, and that split's true
    labels.  Because :func:`train_model` is inclusion-independent, these
    orderings do **not** depend on ``inclusion`` - so they can be cached once
    and re-thresholded at any inclusion via :func:`threshold_from_fold_orderings`
    (and swept across all inclusions for the Stats chart).  See
    docs/plans/find-verification-workflow.md.

    Returns ``(orderings, fallback)``.  When calibration is not possible the
    orderings are empty and ``fallback`` is the sentinel threshold the public
    wrapper must return (mirrors :func:`calculate_cross_calibration_threshold`'s
    historical early-returns); otherwise ``fallback`` is ``None``.
    """
    n = len(X_list)
    if n < 4:
        return [], 0.5

    _rng = rng if rng is not None else np.random
    X_np = np.array(X_list)
    y_np = np.array(y_list)

    n_cal = max(1, round(n * calibration_fraction))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return [], NO_GOOD_THRESHOLD

    pos_idx = np.where(y_np == 1.0)[0]
    neg_idx = np.where(y_np == 0.0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return [], 0.5

    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

    calibrate_count = max(1, calibrate_count)

    def _per_class_n_train(class_total: int) -> int:
        target = round(class_total * n_train / n)
        return max(1, min(class_total - 1, target))

    n_train_pos = _per_class_n_train(len(pos_idx))
    n_train_neg = _per_class_n_train(len(neg_idx))

    orderings: list[tuple[list[float], list[float]]] = []
    for _ in range(calibrate_count):
        pos_perm = _rng.permutation(pos_idx)
        neg_perm = _rng.permutation(neg_idx)
        train_idx = np.concatenate([pos_perm[:n_train_pos], neg_perm[:n_train_neg]])
        cal_idx = np.concatenate([pos_perm[n_train_pos:], neg_perm[n_train_neg:]])

        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32)

        model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim)

        with torch.no_grad():
            X_cal = X_cal.to(next(model.parameters()).device)
            scores = torch.sigmoid(model(X_cal)).squeeze(1).cpu().tolist()
        orderings.append((scores, y_np[cal_idx].tolist()))

    return orderings, None


def threshold_from_fold_orderings(
    fold_orderings: list[tuple[list[float], list[float]]],
    inclusion_value: int,
) -> float:
    """Average the per-fold min-cost thresholds at *inclusion_value*.

    Cheap: just re-runs :func:`find_optimal_threshold` over each fold's cached
    ``(scores, labels)``.  Callers must pass a non-empty ``fold_orderings``
    (the empty case is handled via the ``fallback`` from
    :func:`compute_fold_orderings`).
    """
    thresholds = [find_optimal_threshold(s, lbls, inclusion_value) for s, lbls in fold_orderings]
    return sum(thresholds) / len(thresholds)


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
        inclusion_value: Integer in ``[-10, 10]`` passed to
            :func:`find_optimal_threshold` to control the FPR/FNR trade-off.
            It does **not** enter model training (the fold models are
            inclusion-independent), so the same fold scores can be re-thresholded
            at any inclusion - see docs/plans/find-verification-workflow.md.
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
    orderings, fallback = compute_fold_orderings(
        X_list,
        y_list,
        input_dim,
        rng=rng,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
        hidden_dim=hidden_dim,
    )
    if fallback is not None:
        return fallback
    return threshold_from_fold_orderings(orderings, inclusion_value)


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
