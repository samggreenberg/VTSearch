"""Fallback behaviour of the cuML backend selectors (``vtscore.gpu_backends``).

These run on a CPU box: they inject a *fake* ``cuml`` whose estimators construct
cleanly but raise inside ``fit`` — the shape of the real-world failure where
cuML's lazy nvrtc kernel compile blows up on a mismatched CUDA toolchain (e.g. a
CUDA-12 nvrtc parsing CUDA-13 fp8 headers).  The fit-level entry points must swap
to the CPU library, return a correct result, and disable cuML for the rest of the
process so later calls don't re-pay the failure.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import vtscore.gpu_backends as gb


@pytest.fixture(autouse=True)
def _reset_kill_switch():
    """The cuML-failed flag is a process global; reset it around each test."""
    gb._cuml_runtime_failed = False
    yield
    gb._cuml_runtime_failed = False


def _install_fake_broken_cuml(monkeypatch):
    """Register a ``cuml`` whose UMAP/KMeans construct fine but raise in ``fit``."""

    class _BrokenUMAP:
        def __init__(self, **_kw):
            pass

        def fit_transform(self, _mat):
            raise RuntimeError("nvrtc: simulated cuda_fp8.hpp compile error")

    class _BrokenKMeans:
        def __init__(self, **_kw):
            pass

        def fit_predict(self, _vecs):
            raise RuntimeError("nvrtc: simulated cuda_fp8.hpp compile error")

    cuml = types.ModuleType("cuml")
    manifold = types.ModuleType("cuml.manifold")
    manifold.UMAP = _BrokenUMAP  # type: ignore[attr-defined]
    cluster = types.ModuleType("cuml.cluster")
    cluster.KMeans = _BrokenKMeans  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cuml", cuml)
    monkeypatch.setitem(sys.modules, "cuml.manifold", manifold)
    monkeypatch.setitem(sys.modules, "cuml.cluster", cluster)
    # Force the cuML branch regardless of whether a GPU resolves on this host.
    monkeypatch.setattr(gb, "cuml_enabled", lambda: True)


def test_umap_fit_transform_falls_back_when_cuml_fit_raises(monkeypatch, caplog):
    _install_fake_broken_cuml(monkeypatch)
    rng = np.random.default_rng(0)
    mat = rng.standard_normal((60, 16)).astype(np.float32)

    with caplog.at_level("WARNING"):
        coords = gb.umap_fit_transform(
            mat, n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42
        )

    # CPU umap-learn produced a real layout despite the cuML blowup.
    assert isinstance(coords, np.ndarray)
    assert coords.shape == (60, 2)
    assert np.isfinite(coords).all()
    # The failure was logged and the process-global kill switch flipped.
    assert gb._cuml_runtime_failed is True
    assert any("cuML UMAP failed" in r.message for r in caplog.records)


def test_kmeans_fit_predict_falls_back_when_cuml_fit_raises(monkeypatch, caplog):
    _install_fake_broken_cuml(monkeypatch)
    rng = np.random.default_rng(1)
    vecs = np.vstack([rng.standard_normal((30, 8)) + 5.0, rng.standard_normal((30, 8)) - 5.0]).astype(np.float32)

    with caplog.at_level("WARNING"):
        labels, inertia = gb.kmeans_fit_predict(vecs, n_clusters=2, random_state=42, n_init=1)

    # sklearn KMeans clustered the two blobs despite the cuML blowup.
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (60,)
    assert set(np.unique(labels).tolist()) <= {0, 1}
    assert inertia is not None
    assert gb._cuml_runtime_failed is True
    assert any("cuML KMeans failed" in r.message for r in caplog.records)


def test_kill_switch_disables_cuml_for_rest_of_process():
    # Once a runtime failure is recorded, cuml_enabled() short-circuits to False
    # *before* touching the device check, so subsequent calls skip cuML entirely.
    gb._cuml_runtime_failed = True
    assert gb.cuml_enabled() is False


def test_repeated_failure_warns_only_once(monkeypatch, caplog):
    _install_fake_broken_cuml(monkeypatch)
    rng = np.random.default_rng(2)
    vecs = rng.standard_normal((20, 8)).astype(np.float32)

    with caplog.at_level("WARNING"):
        # Two cuML attempts in a row; the kill switch makes the second a no-op,
        # but even forcing the branch again must not emit a second WARNING.
        gb.kmeans_fit_predict(vecs, n_clusters=2, random_state=0, n_init=1)
        monkeypatch.setattr(gb, "cuml_enabled", lambda: True)
        gb.kmeans_fit_predict(vecs, n_clusters=2, random_state=0, n_init=1)

    warnings = [r for r in caplog.records if r.levelname == "WARNING" and "cuML" in r.message]
    assert len(warnings) == 1
