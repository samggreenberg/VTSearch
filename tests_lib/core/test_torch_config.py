"""Tests for runtime torch configuration.

Covers:
- ``VTSEARCH_TORCH_THREADS`` env-var override drives ``torch.set_num_threads``
  via :data:`vtscore.config.TORCH_THREADS` (read by
  :func:`vtscore.media.torch_setup.ensure_torch_configured`).
- ``get_torch_device()`` selection delegates to
  :func:`vtscore.config.resolve_device`, honouring ``VTSEARCH_DEVICE``.
- ``train_model`` returning a model on the selected device.

``vtscore.config`` is a package, so the env vars are re-read with
``config._reload_all()`` rather than ``importlib.reload`` (which would only
re-run the package's re-exports, leaving the cached submodules untouched), and a
stub for something a *config* function calls - ``allocated_cpus``,
``_cuda_can_run`` - goes on the submodule that owns it, because the package
attribute is only a copy.  See the :mod:`vtscore.config` docstring.
"""

from __future__ import annotations

import importlib
import sys
from unittest import mock

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def restore_reloaded_modules():
    """Undo what ``importlib.reload`` does to the rest of the session.

    Every test here reloads ``vtscore.config`` (and often
    ``vtscore.embedding.loader``) to re-read an env var at import time.  A
    reload re-executes the module *in place*, so it resets every module-level
    value to its import-time default - including the ones the conftests install
    for the whole session, none of which the reloading test wants to change.
    Nothing restored them, so the wipe leaked into every later test sharing the
    worker process.

    That is issue #3101: ``TRAIN_EPOCHS`` reverted from the session's 30 to the
    production 200, so a *fully seeded* threshold fixture trained a different
    model depending on whether these tests happened to run before it on the
    same xdist worker - which read as a nondeterministic failure in the safe
    threshold subsystem rather than as test-state leakage.  Restoring the
    module dicts fixes the whole class of leak rather than the one constant
    that happened to be load-bearing; the conftests additionally re-assert the
    two values with the worst blast radius.

    Declared first so it tears down *last*, after the narrower fixtures below
    have cleared the caches they own on the reloaded module.

    The snapshot covers every ``vtscore.config`` submodule as well as the
    package: ``_reload_all`` re-executes all of them, so restoring the package's
    ``__dict__`` alone would leave a test's env var live in the submodule that
    actually holds it.
    """
    import vtscore.config as config
    import vtscore.embedding.loader as loader

    modules = [config, *(sys.modules[f"vtscore.config.{name}"] for name in config._RELOAD_ORDER), loader]
    snapshots = [(module, dict(module.__dict__)) for module in modules]
    yield
    for module, snapshot in snapshots:
        module.__dict__.clear()
        module.__dict__.update(snapshot)


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
    from vtscore.config import device as device_mod

    device_mod._cuda_runnable.clear()
    yield
    device_mod._cuda_runnable.clear()


def test_torch_threads_constant_default(monkeypatch):
    """``TORCH_THREADS`` defaults to 1 when the env var is unset."""
    monkeypatch.delenv("VTSEARCH_TORCH_THREADS", raising=False)
    import vtscore.config as config

    config = config._reload_all()
    assert config.TORCH_THREADS == 1


def test_torch_threads_constant_honours_env(monkeypatch):
    monkeypatch.setenv("VTSEARCH_TORCH_THREADS", "4")
    import vtscore.config as config

    config = config._reload_all()
    assert config.TORCH_THREADS == 4


def test_torch_threads_constant_clamps_to_one(monkeypatch):
    monkeypatch.setenv("VTSEARCH_TORCH_THREADS", "0")
    import vtscore.config as config

    config = config._reload_all()
    assert config.TORCH_THREADS == 1


def test_decode_workers_sized_from_allocation(monkeypatch):
    """The pool leaves one CPU for the calling thread."""
    from vtscore.config import runtime as config

    monkeypatch.delenv("VTSEARCH_DECODE_WORKERS", raising=False)
    monkeypatch.setattr(config, "allocated_cpus", lambda: 8)
    assert config.resolve_decode_workers() == 7


def test_decode_workers_capped(monkeypatch):
    """A fat node does not get a proportionally fat pool."""
    from vtscore.config import runtime as config

    monkeypatch.delenv("VTSEARCH_DECODE_WORKERS", raising=False)
    monkeypatch.setattr(config, "allocated_cpus", lambda: 96)
    assert config.resolve_decode_workers() == config.DEFAULT_DECODE_WORKER_CAP


def test_decode_workers_floor_of_one_on_a_single_cpu(monkeypatch):
    """One CPU still gets one worker: it overlaps decode with the forward."""
    from vtscore.config import runtime as config

    monkeypatch.delenv("VTSEARCH_DECODE_WORKERS", raising=False)
    monkeypatch.setattr(config, "allocated_cpus", lambda: 1)
    assert config.resolve_decode_workers() == 1


def test_decode_workers_env_override(monkeypatch):
    from vtscore.config import runtime as config

    monkeypatch.setattr(config, "allocated_cpus", lambda: 8)
    monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "3")
    assert config.resolve_decode_workers() == 3


def test_decode_workers_zero_disables_the_pool(monkeypatch):
    from vtscore.config import runtime as config

    monkeypatch.setattr(config, "allocated_cpus", lambda: 8)
    monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "0")
    assert config.resolve_decode_workers() == 0


def test_decode_workers_invalid_env_falls_back(monkeypatch):
    from vtscore.config import runtime as config

    monkeypatch.setattr(config, "allocated_cpus", lambda: 4)
    monkeypatch.setenv("VTSEARCH_DECODE_WORKERS", "lots")
    assert config.resolve_decode_workers() == 3


def test_allocated_cpus_is_positive():
    """Whatever the platform, the answer is a usable worker count."""
    import vtscore.config as config

    assert config.allocated_cpus() >= 1


def test_max_upload_mb_default(monkeypatch):
    """``MAX_UPLOAD_MB`` defaults to a bounded 2 GiB cap, not unlimited."""
    monkeypatch.delenv("VTSEARCH_MAX_UPLOAD_MB", raising=False)
    import vtscore.config as config

    config = config._reload_all()
    assert config.MAX_UPLOAD_MB == 2048


def test_max_upload_mb_honours_env(monkeypatch):
    monkeypatch.setenv("VTSEARCH_MAX_UPLOAD_MB", "512")
    import vtscore.config as config

    config = config._reload_all()
    assert config.MAX_UPLOAD_MB == 512


def test_max_upload_mb_zero_disables_cap(monkeypatch):
    """``VTSEARCH_MAX_UPLOAD_MB=0`` opts back into Flask's unlimited body size."""
    monkeypatch.setenv("VTSEARCH_MAX_UPLOAD_MB", "0")
    import vtscore.config as config

    config = config._reload_all()
    assert config.MAX_UPLOAD_MB == 0


def test_max_upload_mb_negative_clamps_to_zero(monkeypatch):
    monkeypatch.setenv("VTSEARCH_MAX_UPLOAD_MB", "-5")
    import vtscore.config as config

    config = config._reload_all()
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

    config._reload_all()
    importlib.reload(loader)
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_get_torch_device_honours_explicit_cpu(monkeypatch):
    """``VTSEARCH_DEVICE=cpu`` forces CPU even when CUDA is available."""
    monkeypatch.setenv("VTSEARCH_DEVICE", "cpu")
    import vtscore.config as config
    import vtscore.embedding.loader as loader

    config._reload_all()
    importlib.reload(loader)
    with mock.patch.object(torch.cuda, "is_available", return_value=True):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_get_torch_device_returns_cuda_when_available(monkeypatch):
    """``auto`` resolves to cuda when CUDA is available AND a kernel can run."""
    monkeypatch.setenv("VTSEARCH_DEVICE", "auto")
    import vtscore.config as config
    from vtscore.config import device as device_mod
    import vtscore.embedding.loader as loader

    config._reload_all()
    importlib.reload(loader)
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(device_mod, "_cuda_can_run", return_value=True),
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
    from vtscore.config import device as device_mod
    import vtscore.embedding.loader as loader

    config._reload_all()
    importlib.reload(loader)
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(device_mod, "_cuda_can_run", return_value=False),
    ):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_explicit_cuda_pin_falls_back_when_kernel_cannot_run(monkeypatch):
    """Even an explicit ``VTSEARCH_DEVICE=cuda`` pin degrades to CPU if unusable."""
    monkeypatch.setenv("VTSEARCH_DEVICE", "cuda")
    import vtscore.config as config
    from vtscore.config import device as device_mod
    import vtscore.embedding.loader as loader

    config._reload_all()
    importlib.reload(loader)
    with mock.patch.object(device_mod, "_cuda_can_run", return_value=False):
        dev = loader.get_torch_device()

    assert dev.type == "cpu"


def test_cuda_can_run_returns_false_when_launch_raises(monkeypatch):
    """``_cuda_can_run`` swallows a kernel-launch error and caches False."""
    import vtscore.config as config
    from vtscore.config import device as device_mod

    config._reload_all()

    def _boom(*_args, **_kwargs):
        raise RuntimeError("no kernel image is available for execution on the device")

    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch, "zeros", side_effect=_boom),
    ):
        assert device_mod._cuda_can_run("cuda") is False
        # Cached: a second call doesn't re-probe (would raise if it did, but
        # the cache short-circuits before touching torch).
        assert device_mod._cuda_can_run("cuda") is False
    assert device_mod._cuda_runnable["cuda"] is False


def test_cuda_can_run_warning_reports_device_and_arch_mismatch(monkeypatch, caplog):
    """The kernel-image warning names the GPU's compute capability and the
    arch list the installed build was compiled for, and steers users toward an
    older tag for old GPUs (cu128 dropped Volta) rather than just "the newest"."""
    import logging

    import vtscore.config as config
    from vtscore.config import device as device_mod

    config._reload_all()

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
        assert device_mod._cuda_can_run("cuda") is False

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
    from vtscore.config import device as device_mod

    config._reload_all()
    with mock.patch.object(torch.cuda, "get_device_name", side_effect=RuntimeError("boom")):
        assert device_mod._describe_cuda_mismatch("cuda") == ""


def test_cuda_can_run_false_without_cuda(monkeypatch):
    """No CUDA at all -> ``_cuda_can_run`` is False without launching anything."""
    import vtscore.config as config
    from vtscore.config import device as device_mod

    config._reload_all()
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        assert device_mod._cuda_can_run("cuda") is False


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
