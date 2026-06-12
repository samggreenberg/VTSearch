"""Tests for Stage-1 projection (``vtscore.projection.umap_projection``)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import vtscore.projection.umap_projection as up
from vtscore.projection.umap_projection import Projection, fit_projection


def _matrix(n: int, d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


def test_empty_dataset():
    proj = fit_projection(np.empty((0, 8), dtype=np.float32), [])
    assert isinstance(proj, Projection)
    assert proj.coords.shape == (0, 2)
    assert proj.method == "trivial"
    assert proj.ids == []
    assert proj.bounds == (0.0, 0.0, 0.0, 0.0)
    assert proj.projection_id  # minted regardless


def test_single_point_is_trivial_at_origin():
    proj = fit_projection(_matrix(1, 8), [5])
    assert proj.coords.shape == (1, 2)
    assert proj.method == "trivial"
    assert proj.ids == [5]
    np.testing.assert_array_equal(proj.coords, np.zeros((1, 2), dtype=np.float32))


def test_two_points_trivial():
    proj = fit_projection(_matrix(2, 8), [1, 2])
    assert proj.coords.shape == (2, 2)
    assert proj.method == "trivial"


def test_small_n_uses_pca_fallback():
    proj = fit_projection(_matrix(6, 8, seed=3), [10, 11, 12, 13, 14, 15], min_n_for_umap=10)
    assert proj.method == "pca"
    assert proj.coords.shape == (6, 2)
    assert np.isfinite(proj.coords).all()
    # PCA is deterministic: a second fit on the same matrix gives the same layout.
    proj2 = fit_projection(_matrix(6, 8, seed=3), [10, 11, 12, 13, 14, 15], min_n_for_umap=10)
    np.testing.assert_allclose(proj.coords, proj2.coords, atol=1e-5)


def test_scalar_embeddings_fall_back_to_trivial():
    # d < 2 can't support PCA-2; trivial layout instead of crashing.
    proj = fit_projection(_matrix(20, 1), list(range(20)))
    assert proj.method == "trivial"
    assert proj.coords.shape == (20, 2)


def test_ids_length_mismatch_raises():
    with pytest.raises(ValueError):
        fit_projection(_matrix(5, 8), [1, 2, 3])


def test_umap_fit_shape_and_metadata():
    # A real (seeded, reproducible) UMAP fit on a small synthetic set.
    n, d = 60, 12
    proj = fit_projection(_matrix(n, d, seed=11), list(range(n)), n_neighbors=10, random_state=7)
    assert proj.method == "umap"
    assert proj.coords.shape == (n, 2)
    assert proj.coords.dtype == np.float32
    assert np.isfinite(proj.coords).all()
    assert len(proj.ids) == n
    xmin, ymin, xmax, ymax = proj.bounds
    assert xmax >= xmin and ymax >= ymin


def test_n_neighbors_clamped_below_n():
    # n_neighbors larger than N-1 must not blow up: it's clamped.
    n, d = 15, 8
    proj = fit_projection(_matrix(n, d, seed=2), list(range(n)), n_neighbors=100, random_state=1)
    assert proj.method == "umap"
    assert proj.coords.shape == (n, 2)


def _clustered_matrix(per: int = 40, d: int = 12, seed: int = 0) -> np.ndarray:
    """Three well-separated high-dim Gaussian clusters so UMAP yields distinct
    islands and compaction has oceans to close."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((3, d)).astype(np.float32) * 12.0
    return np.concatenate([c + rng.standard_normal((per, d)).astype(np.float32) for c in centres])


def test_compaction_on_by_default_moves_clusters_together():
    mat = _clustered_matrix(seed=5)
    ids = list(range(mat.shape[0]))
    packed = fit_projection(mat, ids, n_neighbors=10, random_state=3, compact=True)
    loose = fit_projection(mat, ids, n_neighbors=10, random_state=3, compact=False)
    assert packed.method == "umap" and loose.method == "umap"
    assert packed.coords.shape == loose.coords.shape

    # Default compaction must actually change the layout (close the oceans):
    # the packed bbox is no larger than the raw UMAP bbox.
    def _area(c):
        span = c.max(axis=0) - c.min(axis=0)
        return float(span[0] * span[1])

    assert not np.allclose(packed.coords, loose.coords)
    assert _area(packed.coords) <= _area(loose.coords) + 1e-3


def test_compaction_preserves_finiteness_and_dtype():
    mat = _clustered_matrix(seed=6)
    proj = fit_projection(mat, list(range(mat.shape[0])), n_neighbors=10, random_state=2)
    assert proj.coords.dtype == np.float32
    assert np.isfinite(proj.coords).all()


def test_progress_callback_invoked():
    seen: list[tuple[str, str, int, int]] = []
    fit_projection(
        _matrix(6, 8),
        list(range(6)),
        min_n_for_umap=10,
        on_progress=lambda s, m, c, t: seen.append((s, m, c, t)),
    )
    assert seen  # at least one milestone reported
    assert all(evt[0] == "projecting" for evt in seen)


def test_umap_fit_emits_elapsed_heartbeats(monkeypatch):
    """While the (blocking) UMAP fit runs, fit_projection ticks elapsed-seconds
    heartbeats so the browse view never shows dead airtime.

    Synchronization is event-based (not sleep-based): a fake reducer blocks
    until the test has observed at least one heartbeat, then releases.
    """
    import umap

    release = threading.Event()  # lets the fake fit finish
    saw_heartbeat = threading.Event()

    class _FakeReducer:
        def __init__(self, **_kwargs):
            pass

        def fit_transform(self, mat):
            release.wait(timeout=5)
            return np.zeros((mat.shape[0], 2), dtype=np.float32)

    monkeypatch.setattr(umap, "UMAP", _FakeReducer)
    monkeypatch.setattr(up, "_HEARTBEAT_SECONDS", 0.02)

    seen: list[tuple[str, str, int, int]] = []

    def _on_progress(status, message, current, total):
        seen.append((status, message, current, total))
        if "elapsed" in message:
            saw_heartbeat.set()

    n, d = 12, 8
    result: list[Projection] = []

    def _run():
        result.append(fit_projection(_matrix(n, d), list(range(n)), on_progress=_on_progress))

    worker = threading.Thread(target=_run, name="fit-under-test")
    worker.start()
    try:
        assert saw_heartbeat.wait(timeout=5), "expected at least one elapsed heartbeat"
    finally:
        release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()

    proj = result[0]
    assert proj.method == "umap"
    assert proj.coords.shape == (n, 2)
    # Heartbeats are indeterminate (total == 0) so the frontend bar animates
    # instead of freezing at 0/N for the whole fit.
    heartbeats = [evt for evt in seen if "elapsed" in evt[1]]
    assert heartbeats
    assert all(total == 0 for *_rest, total in heartbeats)
