"""Tests for runtime torch configuration.

Covers:
- ``VTSEARCH_TORCH_THREADS`` env-var override drives ``torch.set_num_threads``
  via :data:`vtsearch.config.TORCH_THREADS` (read by
  :func:`vtsearch.media.torch_setup.ensure_torch_configured`).
- ``get_torch_device()`` selection delegates to
  :func:`vtsearch.config.resolve_device`, honouring ``VTSEARCH_DEVICE``.
- ``train_model`` returning a model on the selected device.
"""

from __future__ import annotations

import importlib
from unittest import mock

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def reset_torch_configured_flag():
    """Force ``ensure_torch_configured`` to re-run on each test."""
    import vtsearch.media.torch_setup as torch_setup

    torch_setup._torch_configured = False
    yield
    torch_setup._torch_configured = False


def test_torch_threads_constant_default(monkeypatch):
    """``TORCH_THREADS`` defaults to 1 when the env var is unset."""
    monkeypatch.delenv("VTSEARCH_TORCH_THREADS", raising=False)
    import vtsearch.config as config

    config = importlib.reload(config)
    assert config.TORCH_THREADS == 1


def test_torch_threads_constant_honours_env(monkeypatch):
    monkeypatch.setenv("VTSEARCH_TORCH_THREADS", "4")
    import vtsearch.config as config

    config = importlib.reload(config)
    assert config.TORCH_THREADS == 4


def test_torch_threads_constant_clamps_to_one(monkeypatch):
    monkeypatch.setenv("VTSEARCH_TORCH_THREADS", "0")
    import vtsearch.config as config

    config = importlib.reload(config)
    assert config.TORCH_THREADS == 1


def test_ensure_torch_configured_applies_constant(monkeypatch):
    """``ensure_torch_configured`` passes ``TORCH_THREADS`` to torch."""
    import vtsearch.media.torch_setup as torch_setup

    monkeypatch.setattr(torch_setup, "TORCH_THREADS", 2)
    with mock.patch.object(torch, "set_num_threads") as set_threads:
        torch_setup.ensure_torch_configured()

    set_threads.assert_called_once_with(2)


def test_get_torch_device_falls_back_to_cpu_without_cuda(monkeypatch):
    """With no CUDA the resolved device is CPU."""
    from vtsearch.embedding import loader

    monkeypatch.setenv("VTSEARCH_DEVICE", "auto")
    import vtsearch.config as config

    importlib.reload(config)
    importlib.reload(loader)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_get_torch_device_honours_explicit_cpu(monkeypatch):
    """``VTSEARCH_DEVICE=cpu`` forces CPU even when CUDA is available."""
    monkeypatch.setenv("VTSEARCH_DEVICE", "cpu")
    import vtsearch.config as config
    import vtsearch.embedding.loader as loader

    importlib.reload(config)
    importlib.reload(loader)
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_get_torch_device_returns_cuda_when_available(monkeypatch):
    """``auto`` resolves to cuda when torch reports CUDA is available."""
    monkeypatch.setenv("VTSEARCH_DEVICE", "auto")
    import vtsearch.config as config
    import vtsearch.embedding.loader as loader

    importlib.reload(config)
    importlib.reload(loader)
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        dev = loader.get_torch_device()

    assert dev.type == "cuda"


def test_train_model_places_model_on_selected_device(monkeypatch):
    """train_model must move the returned model onto ``get_torch_device()``."""
    from vtsearch.embedding import loader
    from vtsearch.training.mlp import train_model

    fake_device = torch.device("cpu")
    monkeypatch.setattr(loader, "get_torch_device", lambda: fake_device)

    rng = np.random.default_rng(42)
    X = torch.tensor(rng.standard_normal((20, 16)).astype(np.float32))
    y = torch.tensor([1.0] * 10 + [0.0] * 10, dtype=torch.float32).unsqueeze(1)

    model = train_model(X, y, input_dim=16, inclusion_value=0, hidden_dim=8)
    param_device = next(model.parameters()).device
    assert param_device.type == fake_device.type


def test_imports_do_not_break():
    """Sanity check: reloading loader picks up changes without errors."""
    import vtsearch.embedding.loader as loader

    importlib.reload(loader)
