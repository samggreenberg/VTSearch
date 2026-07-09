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

# Above this many scores, fit the GMM on a random subsample instead of the full
# set. A 2-component, 1-D GMM only needs to recover the two clusters' means and
# variances, which 50k samples estimate as accurately as the full population -
# so the threshold is statistically indistinguishable while the EM fit stays
# O(50k) instead of O(N). This matters because ``calculate_gmm_threshold`` runs
# on the *full* score distribution on every cosine/text sort (sorting.py) and in
# the safe-threshold blend, where N reaches ~250k (GUI Find) to 2M+ (CLI Find).
_GMM_MAX_SAMPLES = 50_000


def calculate_gmm_threshold(scores: list[float]) -> float:
    """Use a Gaussian Mixture Model to find a threshold between two score distributions.

    Fits a 2-component GMM to the provided scores, assuming a bimodal distribution
    representing Bad (low) and Good (high) classes. Returns the midpoint between the
    two component means as the decision threshold.

    For score sets larger than :data:`_GMM_MAX_SAMPLES`, fits on a deterministic
    (seed-42) random subsample - the two-Gaussian fit is unchanged in practice
    but the cost no longer grows with the dataset size.

    Args:
        scores: List of model confidence scores, expected to follow a bimodal distribution.

    Returns:
        A float threshold. Scores at or above this value are classified as Good.
        Returns ``0.5`` when fewer than 2 scores are provided; falls back to
        the median of scores if GMM fitting fails.
    """
    if len(scores) < 2:
        return 0.5

    from sklearn.mixture import GaussianMixture  # noqa: PLC0415

    arr = np.asarray(scores, dtype=np.float64)
    if arr.shape[0] > _GMM_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        arr = rng.choice(arr, size=_GMM_MAX_SAMPLES, replace=False)

    # Reshape for sklearn
    X = arr.reshape(-1, 1)

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
        # If GMM fails, return median (of the subsample when one was taken -
        # representative of the full distribution and keeps this path bounded).
        return float(np.median(arr))


def _calibration_cache_key(
    X_list: list,
    y_list: list[float],
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int,
    groups: list | None = None,
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
    # Bag membership changes the fold split and per-group max-pool, so a change
    # in grouping must invalidate the cached orderings even when X/y are equal.
    groups_key = tuple(str(g) for g in groups) if groups is not None else None
    return (
        X_bytes,
        y_bytes,
        int(calibrate_count),
        float(calibration_fraction),
        int(hidden_dim),
        groups_key,
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
    groups: list | None = None,
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
        key = _calibration_cache_key(X_list, y_list, calibrate_count, calibration_fraction, hidden_dim, groups)
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
            groups=groups,
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
        The float threshold that achieves the lowest weighted cost, or
        :data:`NO_GOOD_THRESHOLD` when predicting nothing positive is
        strictly cheaper than every realizable cut (e.g. a top-scored
        negative under a precision-biased inclusion).
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

    # Only positions where the score strictly drops afterwards are feasible
    # cut points: ``score >= threshold`` includes *every* item tied with the
    # threshold, so a mid-tie position advertises a TP/FP split the returned
    # threshold cannot realize (common with a saturated MLP emitting exact
    # 1.0/0.0 sigmoids).  The last position is always a feasible cut.
    feasible = np.append(sorted_scores[:-1] > sorted_scores[1:], True)
    costs = np.where(feasible, costs, np.inf)

    best_idx = int(np.argmin(costs))
    best_cost = float(costs[best_idx])

    # "Predict nothing positive" (a threshold above every observed score) is
    # a legitimate candidate the observed scores can't express: FP=0, FN=all
    # positives, so its cost is exactly ``fnr_weight``.  Observed thresholds
    # win ties so behaviour only changes when abstaining is strictly better.
    if fnr_weight < best_cost:
        return NO_GOOD_THRESHOLD

    return float(sorted_scores[best_idx])


def _per_bag_fit_weights(
    y_rows: np.ndarray,
    group_rows: list,
) -> np.ndarray:
    """Per-row loss weights that balance Good votes against Bad **bags**.

    Every Good row weighs ``n_bad_bags / n_good``; every Bad row weighs
    ``1 / (rows in its bag)`` so each Bad image contributes exactly one image's
    worth of negative mass regardless of how many region nodes it flooded in.
    Total Good mass (``n_bad_bags``) equals total Bad mass (``n_bad_bags``), so
    the classes are balanced at the same magnitude as :func:`train_model`'s
    default inverse-frequency weights - only the *unit* changes from row to bag.
    """
    from collections import Counter  # noqa: PLC0415

    n_good = sum(1 for lbl in y_rows if lbl == 1.0)
    bad_groups = {g for g, lbl in zip(group_rows, y_rows, strict=True) if lbl == 0.0}
    n_bad_bags = len(bad_groups)
    bag_sizes = Counter(g for g, lbl in zip(group_rows, y_rows, strict=True) if lbl == 0.0)
    good_w = (n_bad_bags / n_good) if n_good else 1.0
    w = np.ones(len(y_rows), dtype=np.float32)
    for i, (g, lbl) in enumerate(zip(group_rows, y_rows, strict=True)):
        w[i] = good_w if lbl == 1.0 else (1.0 / bag_sizes[g])
    return w


def _compute_fold_orderings_grouped(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Bag-aware variant of :func:`compute_fold_orderings`.

    Splits by *group* (a voted image) instead of by row so a Bad bag's flooded
    region negatives never straddle the Train/Calibrate boundary, sizes the
    split over votes not rows, weight-balances each fold fit per-bag, and
    collapses every calibration group to a single max-pooled score (an image
    scores by its best region, as at inference).
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415
    from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

    _rng = rng if rng is not None else np.random
    X_np = np.array(X_list)
    y_np = np.array(y_list)
    grp = list(groups)

    # Rows per group, and each group's (single) label.
    order_groups: list = []
    rows_by_group: dict = {}
    label_by_group: dict = {}
    for i, g in enumerate(grp):
        if g not in rows_by_group:
            rows_by_group[g] = []
            order_groups.append(g)
            label_by_group[g] = y_np[i]
        rows_by_group[g].append(i)

    pos_groups = [g for g in order_groups if label_by_group[g] == 1.0]
    neg_groups = [g for g in order_groups if label_by_group[g] == 0.0]
    n = len(order_groups)
    if n < 4:
        return [], 0.5
    if len(pos_groups) < 2 or len(neg_groups) < 2:
        return [], 0.5

    n_cal = max(1, round(n * calibration_fraction))
    n_train = n - n_cal
    if n_train < 2 or n_cal < 1:
        return [], NO_GOOD_THRESHOLD

    def _per_class_n_train(class_total: int) -> int:
        target = round(class_total * n_train / n)
        return max(1, min(class_total - 1, target))

    n_train_pos = _per_class_n_train(len(pos_groups))
    n_train_neg = _per_class_n_train(len(neg_groups))

    pos_arr = np.array(pos_groups, dtype=object)
    neg_arr = np.array(neg_groups, dtype=object)

    orderings: list[tuple[list[float], list[float]]] = []
    for _ in range(max(1, calibrate_count)):
        pos_perm = _rng.permutation(len(pos_arr))
        neg_perm = _rng.permutation(len(neg_arr))
        train_groups = [pos_arr[i] for i in pos_perm[:n_train_pos]] + [neg_arr[i] for i in neg_perm[:n_train_neg]]
        cal_groups = [pos_arr[i] for i in pos_perm[n_train_pos:]] + [neg_arr[i] for i in neg_perm[n_train_neg:]]

        train_idx = [i for g in train_groups for i in rows_by_group[g]]
        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        fold_w = torch.tensor(
            _per_bag_fit_weights(y_np[train_idx], [grp[i] for i in train_idx]), dtype=torch.float32
        )
        model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim, sample_weights=fold_w)

        # Score every calibration row, then max-pool per group to one score.
        cal_idx = [i for g in cal_groups for i in rows_by_group[g]]
        with torch.no_grad():
            X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32).to(next(model.parameters()).device)
            row_scores = sigmoid_to_finite_scores(model(X_cal))
        pos_in_cal = {i: s for i, s in zip(cal_idx, row_scores, strict=True)}
        group_scores: list[float] = []
        group_labels: list[float] = []
        for g in cal_groups:
            rows = rows_by_group[g]
            group_scores.append(max(pos_in_cal[i] for i in rows))
            group_labels.append(float(label_by_group[g]))
        orderings.append((group_scores, group_labels))

    return orderings, None


def compute_fold_orderings(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    groups: list | None = None,
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

    *groups* activates the **bag-aware** path used when Bad votes are flooded
    into their region nodes: rows sharing a ``groups`` id belong to one voted
    image (a Bad bag's region negatives, or a single Good row) and
    are kept together on one side of every Train/Calibrate split, split counts
    are taken over *groups* (votes) not rows, fold fits are *weights*-balanced,
    and each calibration group collapses to one max-pooled score - matching how
    inference scores an image by its best region.  When *groups* is ``None``
    (every non-flooded caller) the historical row-wise path runs unchanged.
    """
    if groups is not None:
        return _compute_fold_orderings_grouped(
            X_list,
            y_list,
            input_dim,
            groups,
            rng=rng,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
        )
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
            from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

            X_cal = X_cal.to(next(model.parameters()).device)
            # Sanitize non-finite sigmoids (destabilised fold model): the
            # orderings are cached, swept for the Stats chart, and averaged
            # into ``DetectorContext.threshold`` - a NaN here would silently
            # break every downstream ``score >= threshold`` comparison and
            # leak NaN into JSON responses.
            scores = sigmoid_to_finite_scores(model(X_cal))
        orderings.append((scores, y_np[cal_idx].tolist()))

    return orderings, None


def threshold_from_fold_orderings(
    fold_orderings: list[tuple[list[float], list[float]]],
    inclusion_value: int,
) -> float:
    """Aggregate the per-fold min-cost thresholds at *inclusion_value*.

    Cheap: just re-runs :func:`find_optimal_threshold` over each fold's cached
    ``(scores, labels)``.  Callers must pass a non-empty ``fold_orderings``
    (the empty case is handled via the ``fallback`` from
    :func:`compute_fold_orderings`).

    A fold whose min-cost cut is "predict nothing" returns
    :data:`NO_GOOD_THRESHOLD` (2.0), which is a *vote to abstain*, not a
    number to average.  Numerically averaging it dragged the mean above the
    sigmoid range whenever a single fold abstained (with the default
    ``calibrate_count=2`` any lone abstain forced the whole ensemble to
    abstain, while at 3+ folds the same lone abstain often did not - a
    fold-count-dependent artifact that also stored an ill-defined ~1.3 as the
    "threshold").  Instead the sentinel is tallied as a vote: the ensemble
    abstains only when a **strict majority** of folds abstain; otherwise the
    threshold is the mean of just the folds that produced a real cut.
    """
    per_fold = [find_optimal_threshold(s, lbls, inclusion_value) for s, lbls in fold_orderings]
    if not per_fold:
        return NO_GOOD_THRESHOLD
    finite = [t for t in per_fold if t != NO_GOOD_THRESHOLD]
    n_abstain = len(per_fold) - len(finite)
    # Strict majority abstains -> abstain overall.  ``not finite`` (every fold
    # abstained) is a strict majority for any non-empty ensemble, so it is
    # subsumed here and the ``sum(finite)`` below never divides by zero.
    if n_abstain * 2 > len(per_fold):
        return NO_GOOD_THRESHOLD
    return sum(finite) / len(finite)


def calculate_cross_calibration_threshold(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    inclusion_value: int = 0,
    rng: np.random.RandomState | None = None,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    hidden_dim: int | None = None,
    groups: list | None = None,
) -> float:
    """Estimate a decision threshold using k-fold calibration.

    Performs ``calibrate_count`` independent random Train/Calibrate splits.
    For each split, trains a model on the Train portion and finds the
    optimal threshold on the Calibrate portion. Aggregates the per-fold
    thresholds via :func:`threshold_from_fold_orderings` (mean of the folds
    that produced a real cut; abstain overall when a strict majority abstain).

    Algorithm:
        For each of *k* = ``calibrate_count`` rounds:
        1. Stratified random split into Train (``1 - calibration_fraction``)
           and Calibrate (``calibration_fraction``).  Stratification guarantees
           the Train side has at least one of each class, so the per-fold MLP
           fit always has both-class supervision.
        2. Train a model on Train.
        3. Find optimal threshold on Calibrate.
        Aggregate the *k* thresholds: a fold voting to abstain
        (:data:`NO_GOOD_THRESHOLD`) counts as a vote, not a value; the
        ensemble abstains only under a strict majority, otherwise it returns
        the mean of the non-abstaining folds.

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
        groups=groups,
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
        * ``n_labels <= 6``  → pure GMM threshold (the x-cal weight
          ``(n_labels - 6) / 14`` is 0 at exactly 6).
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
