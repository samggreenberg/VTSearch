"""Tests for the MLP trainer (vtscore.training.mlp)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vtscore.training.mlp import train_model


def _balanced_xy(n_per_class: int = 5, dim: int = 8, seed: int = 0):
    """Build a tiny balanced (X, y) pair suitable for a smoke train."""
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.standard_normal((2 * n_per_class, dim)).astype(np.float32))
    y = torch.tensor(
        [1.0] * n_per_class + [0.0] * n_per_class, dtype=torch.float32
    ).unsqueeze(1)
    return X, y


class TestSingleClassGuard:
    """``train_model`` must refuse single-class ``y`` instead of silently
    training a degenerate model (bug H6 in the logical-bug audit).

    BCE has no discriminative signal when every label is the same — the
    model would saturate to a single constant for every input.  The guard
    raises ``ValueError`` so callers can't accidentally produce a useless
    model.
    """

    def test_all_positive_raises(self):
        rng = np.random.default_rng(0)
        X = torch.tensor(rng.standard_normal((6, 8)).astype(np.float32))
        y = torch.ones((6, 1), dtype=torch.float32)
        with pytest.raises(ValueError, match="at least one positive"):
            train_model(X, y, input_dim=8, hidden_dim=4)

    def test_all_negative_raises(self):
        rng = np.random.default_rng(0)
        X = torch.tensor(rng.standard_normal((6, 8)).astype(np.float32))
        y = torch.zeros((6, 1), dtype=torch.float32)
        with pytest.raises(ValueError, match="at least one positive"):
            train_model(X, y, input_dim=8, hidden_dim=4)

    def test_empty_y_raises(self):
        X = torch.zeros((0, 8), dtype=torch.float32)
        y = torch.zeros((0, 1), dtype=torch.float32)
        with pytest.raises(ValueError, match="at least one positive"):
            train_model(X, y, input_dim=8, hidden_dim=4)

    def test_mixed_classes_trains(self):
        """Sanity check that the guard does not block the happy path."""
        X, y = _balanced_xy()
        model = train_model(X, y, input_dim=8, hidden_dim=4)
        assert next(model.parameters()).device.type in {"cpu", "cuda", "mps"}
