"""Tests for the MLP trainer (vtscore.training.mlp)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from vtscore.concurrency.async_jobs import AsyncJob, bind_job_cancellation
from vtscore.concurrency.progress import CancelledError
from vtscore.training.mlp import train_model


def _balanced_xy(n_per_class: int = 5, dim: int = 8, seed: int = 0):
    """Build a tiny balanced (X, y) pair suitable for a smoke train."""
    rng = np.random.default_rng(seed)
    X = torch.tensor(rng.standard_normal((2 * n_per_class, dim)).astype(np.float32))
    y = torch.tensor([1.0] * n_per_class + [0.0] * n_per_class, dtype=torch.float32).unsqueeze(1)
    return X, y


class TestSingleClassGuard:
    """``train_model`` must refuse single-class ``y`` instead of silently
    training a degenerate model (bug H6 in the logical-bug audit).

    BCE has no discriminative signal when every label is the same; the
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


class TestJobCancellation:
    """The epoch loop must honour a cancel of the background job that owns
    the worker thread, so cancelling a *running* learned-sort/eval job
    actually stops the GIL-bound training instead of merely being advisory.
    """

    def test_cancelled_bound_job_aborts_at_epoch_boundary(self):
        X, y = _balanced_xy()
        job = AsyncJob(job_id="j1")
        job.cancel()  # already cancelled before the first epoch runs
        with bind_job_cancellation(job):  # noqa: SIM117
            with pytest.raises(CancelledError):
                train_model(X, y, input_dim=8, hidden_dim=4)

    def test_uncancelled_bound_job_trains_normally(self):
        """A bound-but-not-cancelled job must not disturb the happy path."""
        X, y = _balanced_xy()
        job = AsyncJob(job_id="j2")
        with bind_job_cancellation(job):
            model = train_model(X, y, input_dim=8, hidden_dim=4)
        assert next(model.parameters()).device.type in {"cpu", "cuda", "mps"}

    def test_no_binding_trains_normally(self):
        """Outside any job the epoch-boundary check is a pure no-op."""
        X, y = _balanced_xy()
        model = train_model(X, y, input_dim=8, hidden_dim=4)
        assert sum(p.numel() for p in model.parameters()) > 0


class TestEarlyStopDeterminism:
    """The per-epoch ``weighted_loss.item()`` host-device sync is skipped on
    CUDA (checked only on the patience-check cadence).  On CPU the read is
    free, so the per-epoch early-stop path is kept unchanged.  These tests pin
    that the CPU training path stays bit-for-bit deterministic across runs, so
    the sync-cadence change can't silently perturb the early-stop decision on
    the path the whole test suite exercises.
    """

    def _params_flat(self, model):
        return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

    def test_same_seed_same_weights(self):
        """Two runs with the same seed produce identical weights."""
        X, y = _balanced_xy(seed=7)
        m1 = train_model(X, y, input_dim=8, hidden_dim=4, seed=123)
        m2 = train_model(X, y, input_dim=8, hidden_dim=4, seed=123)
        assert torch.equal(self._params_flat(m1), self._params_flat(m2))

    def test_early_stop_fires_deterministically(self, monkeypatch):
        """With a tiny patience the loss plateaus and early-stop fires the same
        way every run: identical trained weights across repeated calls.

        ``train_model`` reads ``TRAIN_EPOCHS`` / ``TRAIN_PATIENCE`` off
        ``config`` at call time, so patching the attributes forces the
        early-stop branch without an env-var reload.
        """
        from vtscore import config

        monkeypatch.setattr(config, "TRAIN_PATIENCE", 3, raising=False)
        monkeypatch.setattr(config, "TRAIN_EPOCHS", 500, raising=False)

        X, y = _balanced_xy(seed=3)
        m1 = train_model(X, y, input_dim=8, hidden_dim=4, seed=99)
        m2 = train_model(X, y, input_dim=8, hidden_dim=4, seed=99)
        assert torch.equal(self._params_flat(m1), self._params_flat(m2))
