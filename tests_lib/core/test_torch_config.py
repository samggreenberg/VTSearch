"""Tests for runtime torch configuration.

Covers:
- ``VTSEARCH_TORCH_THREADS`` env-var override drives ``torch.set_num_threads``
  via :data:`vtscore.config.TORCH_THREADS` (read by
  :func:`vtscore.media.torch_setup.ensure_torch_configured`).
- ``get_torch_device()`` selection delegates to
  :func:`vtscore.config.resolve_device`, honouring ``VTSEARCH_DEVICE``.
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
    import vtscore.media.torch_setup as torch_setup

    torch_setup._torch_configured = False
    yield
    torch_setup._torch_configured = False


@pytest.fixture(autouse=True)
def reset_cuda_probe_cache():
    """Clear the per-process CUDA smoke-test cache so each test re-probes."""
    import vtscore.config as config

    config._cuda_runnable.clear()
    yield
    config._cuda_runnable.clear()


def test_torch_threads_constant_default(monkeypatch):
    """``TORCH_THREADS`` defaults to 1 when the env var is unset."""
    monkeypatch.delenv("VTSEARCH_TORCH_THREADS", raising=False)
    import vtscore.config as config

    config = importlib.reload(config)
    assert config.TORCH_THREADS == 1


def test_torch_threads_constant_honours_env(monkeypatch):
    monkeypatch.setenv("VTSEARCH_TORCH_THREADS", "4")
    import vtscore.config as config

    config = importlib.reload(config)
    assert config.TORCH_THREADS == 4


def test_torch_threads_constant_clamps_to_one(monkeypatch):
    monkeypatch.setenv("VTSEARCH_TORCH_THREADS", "0")
    import vtscore.config as config

    config = importlib.reload(config)
    assert config.TORCH_THREADS == 1


def test_max_upload_mb_default(monkeypatch):
    """``MAX_UPLOAD_MB`` defaults to a bounded 2 GiB cap, not unlimited."""
    monkeypatch.delenv("VTSEARCH_MAX_UPLOAD_MB", raising=False)
    import vtscore.config as config

    config = importlib.reload(config)
    assert config.MAX_UPLOAD_MB == 2048


def test_max_upload_mb_honours_env(monkeypatch):
    monkeypatch.setenv("VTSEARCH_MAX_UPLOAD_MB", "512")
    import vtscore.config as config

    config = importlib.reload(config)
    assert config.MAX_UPLOAD_MB == 512


def test_max_upload_mb_zero_disables_cap(monkeypatch):
    """``VTSEARCH_MAX_UPLOAD_MB=0`` opts back into Flask's unlimited body size."""
    monkeypatch.setenv("VTSEARCH_MAX_UPLOAD_MB", "0")
    import vtscore.config as config

    config = importlib.reload(config)
    assert config.MAX_UPLOAD_MB == 0


def test_max_upload_mb_negative_clamps_to_zero(monkeypatch):
    monkeypatch.setenv("VTSEARCH_MAX_UPLOAD_MB", "-5")
    import vtscore.config as config

    config = importlib.reload(config)
    assert config.MAX_UPLOAD_MB == 0


def test_ensure_torch_configured_applies_constant(monkeypatch):
    """``ensure_torch_configured`` passes ``TORCH_THREADS`` to torch."""
    import vtscore.media.torch_setup as torch_setup

    monkeypatch.setattr(torch_setup, "TORCH_THREADS", 2)
    with mock.patch.object(torch, "set_num_threads") as set_threads:
        torch_setup.ensure_torch_configured()

    set_threads.assert_called_once_with(2)


def test_get_torch_device_falls_back_to_cpu_without_cuda(monkeypatch):
    """With no CUDA the resolved device is CPU."""
    from vtscore.embedding import loader

    monkeypatch.setenv("VTSEARCH_DEVICE", "auto")
    import vtscore.config as config

    importlib.reload(config)
    importlib.reload(loader)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_get_torch_device_honours_explicit_cpu(monkeypatch):
    """``VTSEARCH_DEVICE=cpu`` forces CPU even when CUDA is available."""
    monkeypatch.setenv("VTSEARCH_DEVICE", "cpu")
    import vtscore.config as config
    import vtscore.embedding.loader as loader

    importlib.reload(config)
    importlib.reload(loader)
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_get_torch_device_returns_cuda_when_available(monkeypatch):
    """``auto`` resolves to cuda when CUDA is available AND a kernel can run."""
    monkeypatch.setenv("VTSEARCH_DEVICE", "auto")
    import vtscore.config as config
    import vtscore.embedding.loader as loader

    importlib.reload(config)
    importlib.reload(loader)
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(config, "_cuda_can_run", return_value=True),
    ):
        dev = loader.get_torch_device()

    assert dev.type == "cuda"


def test_auto_falls_back_to_cpu_when_kernel_cannot_run(monkeypatch):
    """CUDA visible but no runnable kernel image -> CPU under ``auto``.

    Reproduces the ``cudaErrorNoKernelImageForDevice`` case: the wheel was
    built without a kernel image for this GPU, so ``is_available()`` is True
    but the smoke-test launch raises. The host must degrade to CPU, not crash.
    """
    monkeypatch.setenv("VTSEARCH_DEVICE", "auto")
    import vtscore.config as config
    import vtscore.embedding.loader as loader

    importlib.reload(config)
    importlib.reload(loader)
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(config, "_cuda_can_run", return_value=False),
    ):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_explicit_cuda_pin_falls_back_when_kernel_cannot_run(monkeypatch):
    """Even an explicit ``VTSEARCH_DEVICE=cuda`` pin degrades to CPU if unusable."""
    monkeypatch.setenv("VTSEARCH_DEVICE", "cuda")
    import vtscore.config as config
    import vtscore.embedding.loader as loader

    importlib.reload(config)
    importlib.reload(loader)
    with mock.patch.object(config, "_cuda_can_run", return_value=False):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_cuda_can_run_returns_false_when_launch_raises(monkeypatch):
    """``_cuda_can_run`` swallows a kernel-launch error and caches False."""
    import vtscore.config as config

    importlib.reload(config)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no kernel image is available for execution on the device")

    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch, "zeros", side_effect=_boom),
    ):
        assert config._cuda_can_run("cuda") is False
        # Cached: a second call doesn't re-probe (would raise if it did, but
        # the cache short-circuits before touching torch).
        assert config._cuda_can_run("cuda") is False
    assert config._cuda_runnable["cuda"] is False


def test_cuda_can_run_warning_reports_device_and_arch_mismatch(monkeypatch, caplog):
    """The kernel-image warning names the GPU's compute capability and the
    arch list the installed build was compiled for, and steers users toward an
    older tag for old GPUs (cu128 dropped Volta) rather than just "the newest"."""
    import logging

    import vtscore.config as config

    importlib.reload(config)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no kernel image is available for execution on the device")

    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch, "zeros", side_effect=_boom),
        mock.patch.object(torch.cuda, "get_device_name", return_value="Tesla V100S-PCIE-32GB"),
        mock.patch.object(torch.cuda, "get_device_capability", return_value=(7, 0)),
        mock.patch.object(torch.cuda, "get_arch_list", return_value=["sm_75", "sm_80", "sm_90"]),
        caplog.at_level(logging.WARNING, logger="vtscore.config"),
    ):
        assert config._cuda_can_run("cuda") is False

    msg = caplog.text
    assert "Tesla V100S-PCIE-32GB" in msg
    assert "compute capability 7.0" in msg
    assert "sm_75" in msg
    # Must NOT tell a Volta owner to "just use the newest" tag (cu128 drops sm_70).
    assert "cu124" in msg
    assert "OLDER tag" in msg


def test_describe_cuda_mismatch_returns_empty_when_torch_unqueryable(monkeypatch):
    """The diagnostic suffix degrades to an empty string if torch raises."""
    import vtscore.config as config

    importlib.reload(config)
    with mock.patch.object(torch.cuda, "get_device_name", side_effect=RuntimeError("boom")):
        assert config._describe_cuda_mismatch("cuda") == ""


def test_cuda_can_run_false_without_cuda(monkeypatch):
    """No CUDA at all -> ``_cuda_can_run`` is False without launching anything."""
    import vtscore.config as config

    importlib.reload(config)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert config._cuda_can_run("cuda") is False


def test_train_model_places_model_on_selected_device(monkeypatch):
    """train_model must move the returned model onto ``get_torch_device()``."""
    from vtscore.embedding import loader
    from vtscore.training.mlp import train_model

    fake_device = torch.device("cpu")
    monkeypatch.setattr(loader, "get_torch_device", lambda: fake_device)

    rng = np.random.default_rng(42)
    X = torch.tensor(rng.standard_normal((20, 16)).astype(np.float32))
    y = torch.tensor([1.0] * 10 + [0.0] * 10, dtype=torch.float32).unsqueeze(1)

    model = train_model(X, y, input_dim=16, hidden_dim=8)
    param_device = next(model.parameters()).device
    assert param_device.type == fake_device.type


def test_imports_do_not_break():
    """Sanity check: reloading loader picks up changes without errors."""
    import vtscore.embedding.loader as loader

    importlib.reload(loader)
