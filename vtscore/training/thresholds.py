"""Decision-threshold computation for learned-sort scores.

GMM-, cross-calibration-, and safe-threshold helpers. These are media-
agnostic: they take score lists and label lists and return a single
float threshold. Detector-specific glue (sourcing ``X_list`` / ``y_list``
from votes, caching on ``DetectorContext``) lives in
:mod:`vtscore.detectors`.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

# Sentinel threshold meaning "predict nothing as Good". Sigmoid scores are
# in [0, 1], so any value > 1.0 makes every ``score >= threshold`` check
# evaluate to False. Kept finite (vs. ``float("inf")``) so it cannot poison
# downstream blends - ``0.0 * inf`` evaluates to NaN, which would then be
# stored on ``DetectorContext.threshold`` and break every comparison.
NO_GOOD_THRESHOLD = 2.0

# Split-conformal inclusion rule (issue #2784), backported from ``dev`` so this
# branch's calibration matches the shipped app. ``BASE`` is the ``k=0`` budget:
# the FN cap is the 0.25-quantile of calibration positives, the FP guard the
# 0.75-quantile of negatives.
CONFORMAL_BASE_BUDGET = 0.25
#: Positive-score quantile the ``k=-10`` end of the inclusion walk reaches.
CONFORMAL_QPOS_MAX = 0.75

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
    min-cost search.  The key encodes a hash of the raw training vectors (not
    just label IDs) so a labelset re-resolved to different embeddings - e.g.
    after the embedder changes - invalidates the cache automatically.  See
    docs/plans/find-verification-workflow.md.
    """
    # Hash the raw training vectors rather than embedding them in the key.
    # The full ``(N_labels x D x 4)``-byte string reaches ~150 MB at 100k
    # labels and would live in the calibration cache until the next call
    # invalidates it. blake2b (fast, 128-bit digest) keeps the key tiny while
    # still changing whenever the labelset re-resolves to different embeddings
    # (e.g. after the embedder changes); the cache is already invalidated by any
    # vote change, so the hash is purely for collision resistance.
    X_hash = hashlib.blake2b(np.stack(X_list).astype(np.float32, copy=False).tobytes()).digest()
    y_hash = hashlib.blake2b(np.asarray(y_list, dtype=np.float32).tobytes()).digest()
    # Bag membership changes the fold split and per-group max-pool, so a change
    # in grouping must invalidate the cached orderings even when X/y are equal.
    groups_key = tuple(str(g) for g in groups) if groups is not None else None
    return (
        X_hash,
        y_hash,
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


def _pooled_group_scores(
    model: Any,
    cal_groups: list,
    rows_by_group: dict,
    X_np: np.ndarray,
    score_rows_by_group: dict | None,
) -> list[float]:
    """Collapse each calibration group to one max-pooled sigmoid score.

    With *score_rows_by_group* each group pools over the rows the scorer will
    max-pool at **inference**; otherwise it pools over the rows it trained on
    (the historical behaviour every production caller takes).
    """
    import torch  # noqa: PLC0415

    from vtscore.utils.scores import sigmoid_to_finite_scores  # noqa: PLC0415

    device = next(model.parameters()).device
    if score_rows_by_group is not None:
        blocks = [np.asarray(score_rows_by_group[g], dtype=np.float32) for g in cal_groups]
        sizes = [b.shape[0] for b in blocks]
        with torch.no_grad():
            X_cal = torch.tensor(np.concatenate(blocks, axis=0), dtype=torch.float32).to(device)
            flat = sigmoid_to_finite_scores(model(X_cal))
        out: list[float] = []
        offset = 0
        for size in sizes:
            out.append(max(flat[offset : offset + size]))
            offset += size
        return out

    cal_idx = [i for g in cal_groups for i in rows_by_group[g]]
    with torch.no_grad():
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32).to(device)
        row_scores = sigmoid_to_finite_scores(model(X_cal))
    by_row = dict(zip(cal_idx, row_scores, strict=True))
    return [max(by_row[i] for i in rows_by_group[g]) for g in cal_groups]


def _compute_fold_orderings_grouped(
    X_list: list[np.ndarray],
    y_list: list[float],
    input_dim: int,
    groups: list,
    rng: np.random.RandomState | None,
    calibrate_count: int,
    calibration_fraction: float,
    hidden_dim: int | None,
    score_rows_by_group: dict | None = None,
) -> tuple[list[tuple[list[float], list[float]]], float | None]:
    """Bag-aware variant of :func:`compute_fold_orderings`.

    Splits by *group* (a voted image) instead of by row so a Bad bag's flooded
    region negatives never straddle the Train/Calibrate boundary, sizes the
    split over votes not rows, weight-balances each fold fit per-bag, and
    collapses every calibration group to a single max-pooled score (an image
    scores by its best region, as at inference).

    *score_rows_by_group* overrides which rows a calibration group collapses
    over - see :func:`compute_fold_orderings`.
    """
    import torch  # noqa: PLC0415

    from vtscore.training.mlp import train_model  # noqa: PLC0415

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

    # Index the plain group lists by position - group ids are tuples, and
    # ``np.array(list_of_tuples)`` would build a 2-D array and mangle them.
    orderings: list[tuple[list[float], list[float]]] = []
    for _ in range(max(1, calibrate_count)):
        pos_perm = _rng.permutation(len(pos_groups))
        neg_perm = _rng.permutation(len(neg_groups))
        train_groups = [pos_groups[i] for i in pos_perm[:n_train_pos]] + [neg_groups[i] for i in neg_perm[:n_train_neg]]
        cal_groups = [pos_groups[i] for i in pos_perm[n_train_pos:]] + [neg_groups[i] for i in neg_perm[n_train_neg:]]

        train_idx = [i for g in train_groups for i in rows_by_group[g]]
        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        fold_w = torch.tensor(_per_bag_fit_weights(y_np[train_idx], [grp[i] for i in train_idx]), dtype=torch.float32)
        model = train_model(X_train, y_train, input_dim, hidden_dim=hidden_dim, sample_weights=fold_w)

        # Collapse each calibration group to one max-pooled score, so a Good
        # bag and a Bad bag are pooled the same way the scorer pools an image.
        group_scores = _pooled_group_scores(model, cal_groups, rows_by_group, X_np, score_rows_by_group)
        group_labels = [float(label_by_group[g]) for g in cal_groups]
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
    score_rows_by_group: dict | None = None,
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

    *score_rows_by_group* (grouped path only) maps each group id to the row
    stack that group should be **scored** over, decoupling "what the fold model
    trains on" from "what a calibration bag collapses to".  It exists because
    the two are not the same whenever a Good vote contributes fewer rows than a
    Bad vote floods: the Good bag then collapses to a max over 1 row while the
    Bad bag - and every image at inference - collapses to a max over N, and
    ``max`` is an upward-biased order statistic, so the fold's min-cost cut
    lands systematically high and the threshold over-rejects positives.  Passing
    each vote's inference rows here puts both classes in the geometry and at the
    width the scorer will actually use.  ``None`` (every production caller)
    keeps the historical "collapse over the training rows" behaviour.
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
            score_rows_by_group=score_rows_by_group,
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


def conformal_threshold(
    scores: list[float],
    labels: list[float],
    inclusion_value: int = 0,
) -> float:
    """Split-conformal quantile threshold over held-out calibration scores (#2784).

    Backported verbatim from ``dev`` so this branch calibrates like the shipped
    app. Maps ``inclusion_value`` (``k``) to a cut via quantiles of the calibration
    score distributions rather than a min-cost search over observed cuts:

    * a **false-negative cap** ``alpha(k) = min(1, BASE * 2^-k)`` — the cut never
      exceeds the ``alpha``-quantile of the calibration *positives*;
    * a **false-positive guard** for ``k <= 0`` at the ``1 - BASE * 2^k`` quantile
      of the *negatives*, walking up toward the positives as ``k`` drops;
    * the **gap midpoint** at ``k=0`` — the midpoint of the band between the top of
      the negatives and the lowest positive. Every cut in that band has identical
      empirical error, so its top edge (the lowest calibration positive) is an
      arbitrary, high-variance choice; the midpoint is the max-margin one and is
      what stops a single boundary vote vaulting the cut over the positives.

    Monotone non-increasing in ``k`` by construction. Returns ``0.5`` when the
    score list is empty or single-class; never abstains otherwise.
    """
    if not scores:
        return 0.5

    scores_arr = np.asarray(scores, dtype=np.float64)
    labels_arr = np.asarray(labels, dtype=np.float64)
    pos = scores_arr[labels_arr == 1.0]
    neg = scores_arr[labels_arr != 1.0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    def _threshold_at(k: int) -> float:
        fn_cap = float(np.quantile(pos, min(1.0, CONFORMAL_BASE_BUDGET * 2.0**-k)))
        if k > 0:
            return min(fn_cap, _threshold_at(0))
        fp_guard = float(np.quantile(neg, 1.0 - CONFORMAL_BASE_BUDGET * 2.0**k))
        gap_mid = (fp_guard + max(fp_guard, float(np.min(pos)))) / 2.0
        top = float(np.quantile(pos, CONFORMAL_QPOS_MAX))
        walk = gap_mid + (-k / 10.0) * max(0.0, top - gap_mid)
        return min(fn_cap, max(fp_guard, walk))

    return _threshold_at(inclusion_value)


def threshold_from_fold_orderings(
    fold_orderings: list[tuple[list[float], list[float]]],
    inclusion_value: int,
) -> float:
    """Apply the conformal inclusion rule to the pooled fold orderings (#2784).

    Pools every fold's cached held-out ``(scores, labels)`` and runs
    :func:`conformal_threshold` once — no per-fold argmin (that was the pre-#2784
    behaviour whose lowest-positive anchor made a single boundary vote vault the
    cut over the calibration positives). Pooling (vs. averaging per-fold
    quantiles) is deliberate: the knob's resolution is bounded by how many
    calibration scores the quantile sees. Empty ``fold_orderings`` (handled via the
    ``fallback`` from :func:`compute_fold_orderings`) returns
    :data:`NO_GOOD_THRESHOLD` defensively.
    """
    if not fold_orderings:
        return NO_GOOD_THRESHOLD
    pooled_scores = [s for scores, _ in fold_orderings for s in scores]
    pooled_labels = [lb for _, labels in fold_orderings for lb in labels]
    return conformal_threshold(pooled_scores, pooled_labels, inclusion_value)


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
    score_rows_by_group: dict | None = None,
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
        score_rows_by_group: Per-group inference row stacks; see
            :func:`compute_fold_orderings`.  Grouped path only.

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
        score_rows_by_group=score_rows_by_group,
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
