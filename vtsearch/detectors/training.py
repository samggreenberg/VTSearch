"""Detector training helpers: validate, train, threshold, serialise.

Consolidates the repeated train → calibrate → safe-threshold → serialise
pipeline used by detector route handlers and test helpers.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def validate_good_bad_split(y_list: list[float]) -> tuple[int, int]:
    """Check that *y_list* contains at least one good and one bad label.

    Returns ``(num_good, num_bad)``.
    Raises ``ValueError`` when either count is zero.
    """
    num_good = sum(1 for y in y_list if y == 1.0)
    num_bad = len(y_list) - num_good
    if num_good == 0 or num_bad == 0:
        raise ValueError("Need at least one good and one bad labeled example")
    return num_good, num_bad


def train_and_threshold(
    X_list: list,
    y_list: list[float],
    snap: dict | None = None,
) -> tuple[Any, float]:
    """Train an MLP and compute a calibrated threshold.

    This is the canonical training pipeline used by all detector routes:

    1. Cross-calibration threshold (respects ``calibrate_count`` /
       ``calibration_fraction`` settings).
    2. Full-data model training (respects ``inclusion`` setting).
    3. Optional safe-threshold blending when ``get_safe_thresholds()`` is
       enabled and *snap* is provided.

    Args:
        X_list: Embedding vectors (list of numpy arrays).
        y_list: Binary labels (1.0 = good, 0.0 = bad).
        snap: Optional media snapshot for safe-threshold scoring.

    Returns:
        ``(model, threshold)``
    """
    import torch

    from vtsearch.models import (
        calculate_cross_calibration_threshold,
        calculate_safe_threshold,
        train_model,
    )
    from vtsearch.state import (
    get_calibrate_count,
    get_calibration_fraction,
    get_inclusion,
    get_safe_thresholds,
)

    X = torch.from_numpy(np.stack(X_list).astype(np.float32, copy=False))
    y = torch.tensor(y_list, dtype=torch.float32).unsqueeze(1)
    input_dim = X.shape[1]

    safe = bool(get_safe_thresholds() and snap)
    # Below the safe-threshold ramp floor the cross-cal output is blended
    # away (pure GMM), so don't pay for the fold trainings.
    if safe and len(X_list) < 6:
        threshold = float("inf")
    else:
        threshold = calculate_cross_calibration_threshold(
            X_list,
            y_list,
            input_dim,
            get_inclusion(),
            calibrate_count=get_calibrate_count(),
            calibration_fraction=get_calibration_fraction(),
        )

    model = train_model(X, y, input_dim, get_inclusion())

    if safe:
        from vtsearch.models.embedding_matrix import get_embedding_matrix_for_snap

        _all_ids, all_embs = get_embedding_matrix_for_snap(snap)
        X_all = torch.from_numpy(all_embs)
        with torch.no_grad():
            X_all = X_all.to(next(model.parameters()).device)
            all_scores = torch.sigmoid(model(X_all)).squeeze(1).cpu().tolist()
        threshold = calculate_safe_threshold(threshold, all_scores, len(y_list))

    return model, threshold


def serialize_weights(model) -> dict[str, list]:
    """Convert a PyTorch model's state dict to JSON-serialisable nested lists."""
    return {key: value.cpu().tolist() for key, value in model.state_dict().items()}
