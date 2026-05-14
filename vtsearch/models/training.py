"""ML training utilities for learned sorting."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from vtsearch import config
from vtsearch.config import MLP_DROPOUT, MLP_HIDDEN_MAX, MLP_HIDDEN_MIN

# ``TRAIN_EPOCHS`` and ``TRAIN_PATIENCE`` are read off ``config`` at call time
# (not import time) so env-var / monkey-patched overrides take effect.

if TYPE_CHECKING:
    import torch
    import torch.nn as nn


def _training_vec_for_vote(
    media: dict[str, Any],
    region_box: tuple[float, float, float, float] | None,
) -> np.ndarray:
    """Return the training vector for one vote on *media*.

    When *region_box* is set **and** *media* has a stored ``patch_grid``,
    pool the box on-the-fly via
    :func:`vtsearch.models.patch_regions.box_to_vote_vector`.  Otherwise
    fall back to ``media["embedding"]`` — the v1/legacy image-level vector.
    Patch-embedder v2.
    """
    if region_box is not None:
        grid = media.get("patch_grid")
        if grid is not None:
            from vtsearch.models.patch_regions import box_to_vote_vector  # noqa: PLC0415

            return box_to_vote_vector(np.asarray(grid), region_box)
    return media["embedding"]


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

        # Get the means of the two components
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


def _auto_hidden_dim(n_train: int) -> int:
    """Choose hidden-layer width based on training-set size.

    Keeps the model small when few votes are available to reduce
    overfitting, and grows (up to ``MLP_HIDDEN_MAX``) as more labels
    arrive.
    """
    return max(MLP_HIDDEN_MIN, min(MLP_HIDDEN_MAX, n_train // 3))


def build_model(
    input_dim: int,
    hidden_dim: int = 64,
    dropout: float = 0.0,
    generator: torch.Generator | None = None,
) -> nn.Sequential:
    """Construct the MLP architecture (untrained).

    The model outputs raw logits (no sigmoid).  Apply ``torch.sigmoid``
    to the output at inference time to obtain probabilities in [0, 1].

    Args:
        input_dim: Dimensionality of the input embeddings.
        hidden_dim: Number of neurons in the hidden layer (default 64).
        dropout: Dropout probability applied after the hidden layer
            (default 0.0 — no dropout).  Active only during training.
        generator: Optional local RNG for weight initialisation.  When
            provided the weights are re-initialised using this generator
            instead of PyTorch's global RNG, making construction
            thread-safe and deterministic.

    Returns:
        An ``nn.Sequential`` model with layers:
        ``Linear(input_dim, hidden_dim) -> ReLU -> Dropout -> Linear(hidden_dim, 1)``.
    """
    import torch.nn as nn  # noqa: PLC0415

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    )
    if generator is not None:
        for module in model.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), generator=generator)
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(module.bias, -bound, bound, generator=generator)
    return model


def build_model_from_weights(weights: dict[str, list]) -> nn.Sequential:
    """Reconstruct a trained model from exported weight lists.

    Handles both legacy (3-layer, no dropout) and current (4-layer with
    dropout) weight formats.  Legacy keys ``2.weight`` / ``2.bias`` are
    remapped to ``3.weight`` / ``3.bias`` automatically.

    Args:
        weights: Mapping of state-dict key names to nested Python lists
            (as produced by ``tensor.tolist()``).

    Returns:
        A model in eval mode ready for inference.
    """
    import torch  # noqa: PLC0415

    # Remap legacy 3-layer format (keys 0,2) → 4-layer format (keys 0,3)
    if "2.weight" in weights and "3.weight" not in weights:
        weights = dict(weights)
        weights["3.weight"] = weights.pop("2.weight")
        weights["3.bias"] = weights.pop("2.bias")

    input_dim = len(weights["0.weight"][0])
    hidden_dim = len(weights["0.bias"])

    model = build_model(input_dim, hidden_dim=hidden_dim, dropout=0.0)
    state_dict = {k: torch.tensor(v, dtype=torch.float32) for k, v in weights.items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def train_model(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    input_dim: int,
    inclusion_value: int = 0,
    seed: int = 42,
    hidden_dim: int | None = None,
) -> nn.Sequential:
    """Train a small MLP classifier and return the trained model.

    The hidden-layer width is chosen automatically based on the number of
    training examples (see :func:`_auto_hidden_dim`) unless explicitly
    provided via ``hidden_dim``.  Dropout is applied during training to
    reduce overfitting.

    Trains using weighted binary cross-entropy loss with logits
    (``BCEWithLogitsLoss``).  Class weights are adjusted based on
    ``inclusion_value`` to bias the classifier toward including more
    (positive) or fewer (positive) items.

    A local ``torch.Generator`` seeded with *seed* is used for model-weight
    initialisation, and ``torch.random.fork_rng`` isolates the global RNG
    (used by ``nn.Dropout``) so concurrent calls don't overwrite each
    other's seed (thread-safe and deterministic).

    Args:
        X_train: Float tensor of shape ``(N, input_dim)`` containing training embeddings.
        y_train: Float tensor of shape ``(N, 1)`` containing binary labels
            (1.0 for good, 0.0 for bad).
        input_dim: Dimensionality of the input embeddings.
        inclusion_value: Integer in ``[-10, 10]`` controlling class-weight bias.
            - 0: balance classes equally (weight_true = num_false / num_true).
            - Positive: increase weight for True samples by ``2 ** inclusion_value``,
              causing the model to include more items.
            - Negative: increase weight for False samples by ``2 ** (-inclusion_value)``,
              causing the model to exclude more items.
        seed: Seed for the local RNG used for weight initialisation (default 42).
        hidden_dim: Number of neurons in the hidden layer.  When ``None``
            (default) the width is chosen automatically via
            :func:`_auto_hidden_dim` based on the training-set size.

    Returns:
        A trained ``nn.Sequential`` model in eval mode.
        The model outputs raw logits — apply ``torch.sigmoid`` at inference.
    """
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    from vtsearch.models.loader import ensure_torch_configured

    ensure_torch_configured()

    n_train = len(X_train)
    if hidden_dim is None:
        hidden_dim = _auto_hidden_dim(n_train)

    # Use a local Generator for weight init and fork_rng for nn.Dropout
    # so that concurrent training calls don't overwrite each other's
    # global seed (thread-safe and deterministic).
    g = torch.Generator()
    g.manual_seed(seed)

    model = build_model(input_dim, hidden_dim=hidden_dim, dropout=MLP_DROPOUT, generator=g)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Calculate class weights based on inclusion
    num_true = y_train.sum().item()
    num_false = len(y_train) - num_true

    # Base weights for balanced classes
    if num_true > 0 and num_false > 0:
        weight_true = num_false / num_true
        weight_false = 1.0
    else:
        weight_true = 1.0
        weight_false = 1.0

    # Adjust weights based on inclusion
    if inclusion_value >= 0:
        # Increase weight for True samples
        weight_true *= 2.0**inclusion_value
    else:
        # Increase weight for False samples
        weight_false *= 2.0 ** (-inclusion_value)

    # Create sample weights
    weights = torch.where(y_train == 1, weight_true, weight_false).squeeze()
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    # Fork the global RNG so the Dropout seed is isolated per call —
    # concurrent training invocations each get their own RNG state.
    model.train()
    epochs = config.TRAIN_EPOCHS
    patience = config.TRAIN_PATIENCE
    best_loss = float("inf")
    epochs_since_improve = 0
    # Require at least this much absolute decrease to count as progress —
    # without it, float noise drifts the loss down forever and the early-stop
    # never fires.
    min_delta = 1e-4
    with torch.random.fork_rng(), torch.enable_grad():
        torch.manual_seed(seed)
        for _ in range(epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            losses = loss_fn(logits, y_train)
            weighted_loss = (losses.squeeze() * weights).mean()
            weighted_loss.backward()
            optimizer.step()
            cur = weighted_loss.item()
            if cur < best_loss - min_delta:
                best_loss = cur
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if patience > 0 and epochs_since_improve >= patience:
                    break

    model.eval()
    return model


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
        1. Randomly split data into Train (``1 - calibration_fraction``)
           and Calibrate (``calibration_fraction``).
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
            examples), returns ``float('inf')`` so that nothing is
            predicted as Good.
        hidden_dim: Force a specific hidden-layer width for the fold models.
            When ``None`` (default), each fold model auto-sizes based on its
            own training-set size.  Pass the full-data hidden dim to ensure
            fold models match the final model's architecture.

    Returns:
        A float threshold. Returns 0.5 if fewer than 4 examples are provided
        (insufficient data for calibration).  Returns ``float('inf')`` if
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
        return float("inf")

    import torch  # noqa: PLC0415

    calibrate_count = max(1, calibrate_count)
    thresholds: list[float] = []

    for _ in range(calibrate_count):
        indices = _rng.permutation(n)
        train_idx = indices[:n_train]
        cal_idx = indices[n_train:]

        X_train = torch.tensor(X_np[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y_np[train_idx], dtype=torch.float32).unsqueeze(1)
        X_cal = torch.tensor(X_np[cal_idx], dtype=torch.float32)

        model = train_model(X_train, y_train, input_dim, inclusion_value, hidden_dim=hidden_dim)

        with torch.no_grad():
            scores = torch.sigmoid(model(X_cal)).squeeze(1).tolist()
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
        A blended threshold float.
    """
    import math  # noqa: PLC0415

    gmm_threshold = calculate_gmm_threshold(all_scores)

    # If xcal_threshold is infinite (e.g. due to impossible fold split),
    # fall back to the GMM threshold entirely.
    if not math.isfinite(xcal_threshold):
        return gmm_threshold

    # Linear ramp: 0 at 6 labels, 1 at 20 labels
    MIN_LABELS = 6
    MAX_LABELS = 20
    label_weight = max(0.0, min(1.0, (n_labels - MIN_LABELS) / (MAX_LABELS - MIN_LABELS)))

    return label_weight * xcal_threshold + (1.0 - label_weight) * gmm_threshold


def train_and_score(
    clips_dict: dict[int, dict[str, Any]],
    good_votes: dict[int, None],
    bad_votes: dict[int, None],
    inclusion_value: int = 0,
    safe_thresholds: bool = False,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
    vote_region_boxes: dict[int, tuple[float, float, float, float]] | None = None,
) -> tuple[list[dict[str, Any]], float, nn.Sequential | None]:
    """Train a small MLP on voted media embeddings and score every media.

    Uses k-fold calibration to determine an appropriate decision threshold,
    then trains a final model on all labelled data and scores every media in
    ``clips_dict``.

    Args:
        clips_dict: Mapping of media ID to media data dict. Each value must contain
            an ``"embedding"`` key with a ``numpy.ndarray`` embedding vector.
        good_votes: Dict whose keys are media IDs labelled as good (values are ``None``).
        bad_votes: Dict whose keys are media IDs labelled as bad (values are ``None``).
        inclusion_value: Integer in ``[-10, 10]`` passed to the training and
            threshold-finding functions to control the inclusion/exclusion bias.
        safe_thresholds: When ``True``, blend the cross-calibration threshold with
            a GMM-based threshold for robustness when few labels are available.
        calibrate_count: Number of random Train/Calibrate splits for threshold
            calibration (default 2).
        calibration_fraction: Fraction of labelled data reserved for calibration
            in each split (default 0.5).  For example, 0.2 means 80% Train /
            20% Calibrate.
        vote_region_boxes: Optional ``media_id -> (x0, y0, x1, y1)`` map from
            yes-votes that designated a region.  When set and the source
            media has a stored ``patch_grid``, the training vector for that
            vote is pooled on-the-fly via
            :func:`vtsearch.models.patch_regions.box_to_vote_vector` instead
            of using ``media["embedding"]``.  Falls back to the full-image
            vector when the media lacks a patch grid (legacy datasets,
            single-vector embedders) or when the box is missing.  Patch-
            embedder v2.

    Returns:
        A tuple ``(results, threshold, model)`` where:

        - ``results`` is a list of ``{"id": int, "score": float}`` dicts, sorted
          by score in descending order (highest confidence first).
        - ``threshold`` is the decision boundary as a float (cross-calibrated,
          or blended with GMM when ``safe_thresholds`` is ``True``).
        - ``model`` is the trained ``nn.Sequential`` model (``None`` when
          training was not possible).
    """
    import torch  # noqa: PLC0415

    region_boxes = vote_region_boxes or {}

    X_list = []
    y_list = []
    for cid in good_votes:
        if cid in clips_dict:
            X_list.append(_training_vec_for_vote(clips_dict[cid], region_boxes.get(cid)))
            y_list.append(1.0)
    for cid in bad_votes:
        if cid in clips_dict:
            X_list.append(clips_dict[cid]["embedding"])
            y_list.append(0.0)

    # Guard against empty or single-class training data after filtering
    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return [], 0.5, None

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)

    input_dim = X.shape[1]

    # Compute hidden_dim once from the *full* label count so that fold
    # models use the same architecture as the final model.  This makes
    # cross-calibration thresholds directly comparable to final-model
    # scores (same capacity, same score distribution shape).
    hidden_dim = _auto_hidden_dim(len(X_list))

    # Calculate threshold using k-fold calibration.
    # Use a seeded RNG so that train/calibrate splits are deterministic —
    # without this, the global np.random state makes results non-reproducible.
    # When ``safe_thresholds`` is on and the label count is below the
    # ``calculate_safe_threshold`` ramp floor, the blended weight on the
    # cross-cal threshold is exactly 0 (pure GMM), so the calibration
    # trainings would be discarded.  Skip them and pass ``inf`` to signal
    # "use the GMM threshold" downstream.
    if safe_thresholds and len(X_list) < 6:
        threshold = float("inf")
    else:
        cal_rng = np.random.RandomState(42)
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            inclusion_value,
            rng=cal_rng,
            calibrate_count=calibrate_count,
            calibration_fraction=calibration_fraction,
            hidden_dim=hidden_dim,
        )

    # Train final model on all data.  Training is image-level in v1 — the
    # MLP only ever sees one vector per voted media (``media["embedding"]``,
    # which equals the patch-region full-image vector for patch datasets),
    # mirroring the v1 vote rule of "vote on whole images".  Region-level
    # training examples are a phase-2 concern.
    model = train_model(X, y, input_dim, inclusion_value, hidden_dim=hidden_dim)

    # Score every media — region-aware max-pool over regions when the
    # dataset is patch-region-aware, plain single-vector scoring when not.
    # We build one flat tensor of (media, region) rows so the MLP runs
    # in a single forward pass; the per-media max is computed after.
    all_ids = sorted(clips_dict.keys())
    flat_vecs: list[np.ndarray] = []
    media_index_per_row: list[int] = []
    region_index_per_row: list[int] = []
    for mi, cid in enumerate(all_ids):
        media = clips_dict[cid]
        regions = media.get("patch_regions")
        if regions:
            for ri, r in enumerate(regions):
                flat_vecs.append(np.asarray(r.vec, dtype=np.float32))
                media_index_per_row.append(mi)
                region_index_per_row.append(ri)
        else:
            flat_vecs.append(np.asarray(media["embedding"], dtype=np.float32))
            media_index_per_row.append(mi)
            region_index_per_row.append(0)

    X_all = torch.tensor(np.array(flat_vecs), dtype=torch.float32)
    with torch.no_grad():
        flat_scores = torch.sigmoid(model(X_all)).squeeze(1).cpu().numpy()

    # Max-pool per media; remember the winning region index so we can
    # surface ``best_region.box`` to the UI for patch-region media.
    scores: list[float] = [-1.0] * len(all_ids)
    best_region: list[int] = [0] * len(all_ids)
    for s, mi, ri in zip(flat_scores, media_index_per_row, region_index_per_row):
        if s > scores[mi]:
            scores[mi] = float(s)
            best_region[mi] = ri

    if safe_thresholds:
        n_labels = len(X_list)
        threshold = calculate_safe_threshold(threshold, scores, n_labels)

    # Sort by raw scores (full precision) so that tiny differences still
    # affect ordering.  Round only for the JSON response values.
    paired = sorted(
        zip(all_ids, scores, best_region),
        key=lambda t: t[1],
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for cid, s, bri in paired:
        entry: dict[str, Any] = {"id": cid, "score": round(s, 4)}
        media = clips_dict[cid]
        regions = media.get("patch_regions")
        if regions and 0 <= bri < len(regions):
            entry["best_region"] = list(regions[bri].box)
        results.append(entry)
    return results, threshold, model


# ---------------------------------------------------------------------------
# Origin-based helpers (for weight-free detector serialisation)
# ---------------------------------------------------------------------------


def collect_media_origins(
    media_ids: dict[int, None] | list[int],
    snap: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collect origin info for a set of media IDs from a medias snapshot.

    Each returned dict contains ``origin``, ``origin_name``, ``filename``,
    and ``md5`` — enough to re-resolve the original file later.

    Args:
        media_ids: Media IDs (keys of a votes dict, or a plain list).
        snap: Snapshot of all loaded medias (from :func:`snapshot_medias`).

    Returns:
        A list of origin dicts, one per matched media.
    """
    origins: list[dict[str, Any]] = []
    for cid in media_ids:
        if cid not in snap:
            continue
        media = snap[cid]
        origins.append(
            {
                "origin": media.get("origin"),
                "origin_name": media.get("origin_name", ""),
                "filename": media.get("filename", ""),
                "md5": media.get("md5", ""),
            }
        )
    return origins


def train_detector_from_origins(
    good_origins: list[dict[str, Any]],
    bad_origins: list[dict[str, Any]],
    inclusion: int,
    media_type: str,
    calibrate_count: int = 2,
    calibration_fraction: float = 0.5,
) -> tuple[dict[str, list] | None, float]:
    """Resolve origin entries to files, embed them, and train a detector MLP.

    This is the load-time counterpart of file-based detector export: given
    the origin lists that were saved to disk, it re-derives the MLP weights
    by resolving the original media files, embedding them, and training.

    Args:
        good_origins: Origin dicts for media labelled Good.
        bad_origins: Origin dicts for media labelled Bad.
        inclusion: The inclusion value to use for training.
        media_type: Media type string (e.g. ``"audio"``, ``"image"``).
        calibrate_count: Number of k-fold calibration splits.
        calibration_fraction: Fraction reserved for calibration.

    Returns:
        A ``(weights, threshold)`` tuple.  ``weights`` is ``None`` if
        resolution/embedding failed for too many entries (need at least
        one good and one bad).
    """
    import torch  # noqa: PLC0415

    from vtsearch.models.resolver import embed_file, resolve_file_context

    X_list: list = []
    y_list: list[float] = []

    for entry in good_origins:
        with resolve_file_context(
            entry.get("origin"),
            entry.get("origin_name", ""),
            entry.get("filename", ""),
        ) as file_path:
            if file_path is None:
                continue
            emb = embed_file(file_path, media_type)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(1.0)

    for entry in bad_origins:
        with resolve_file_context(
            entry.get("origin"),
            entry.get("origin_name", ""),
            entry.get("filename", ""),
        ) as file_path:
            if file_path is None:
                continue
            emb = embed_file(file_path, media_type)
        if emb is None:
            continue
        X_list.append(emb)
        y_list.append(0.0)

    num_good = sum(1 for v in y_list if v == 1.0)
    num_bad = len(y_list) - num_good
    if len(X_list) < 2 or num_good == 0 or num_bad == 0:
        return None, 0.5

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    threshold = calculate_cross_calibration_threshold(
        X_list,
        y_list,
        input_dim,
        inclusion,
        calibrate_count=calibrate_count,
        calibration_fraction=calibration_fraction,
    )
    model = train_model(X, y, input_dim, inclusion)

    state_dict = model.state_dict()
    weights = {k: v.tolist() for k, v in state_dict.items()}
    return weights, threshold
