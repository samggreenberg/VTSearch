"""Tests for the VTSBrowse projection routes.

``POST /api/projection/build``, ``GET /api/projection/meta``, and
``GET /api/projection/tiles/<level>/<tx>/<ty>``.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np

import app as app_module  # noqa: F401
from vtscore.concurrency.async_jobs import projection_jobs
from vtscore.projection import Projection, build_pyramid
from vtscore.state.core import get_active_context


def _wait_projection(timeout: float = 30.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = projection_jobs.current()
        if job is None:
            return
        if job.status in ("running", "pending"):
            job.done_event.wait(timeout=0.05)
            continue
        time.sleep(0.01)
        follow = projection_jobs.current()
        if follow is None or follow.job_id == job.job_id:
            return
    raise TimeoutError(f"projection job did not finish within {timeout}s")


def _fake_fit_projection(matrix, ids, **kwargs):
    """PCA-like fast fake to avoid numba JIT from UMAP."""
    from vtscore.projection.umap_projection import Projection

    rng = np.random.default_rng(0)
    coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
    import uuid

    return Projection(uuid.uuid4().hex, list(ids), coords, "fake")


class TestProjectionMeta:
    """``GET /api/projection/meta`` status reporting."""

    def test_idle_when_no_projection(self, client):
        resp = client.get("/api/projection/meta")
        assert resp.status_code == 200
        assert resp.get_json()["status"] == "idle"

    def test_ready_after_cache_populated(self, client):
        ctx = get_active_context()
        rng = np.random.default_rng(42)
        ids = list(ctx.medias.keys())[:5]
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("test-proj-id", ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)
        ctx._projection = proj
        ctx._pyramid = pyr

        resp = client.get("/api/projection/meta")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ready"
        assert body["projection_id"] == "test-proj-id"
        assert body["point_count"] == len(ids)
        assert len(body["levels"]) == 2
        assert body["method"] == "pca"
        assert isinstance(body["media_type"], str)


class TestProjectionBuild:
    """``POST /api/projection/build`` background job lifecycle."""

    def test_empty_dataset_returns_409(self, client):
        ctx = get_active_context()
        saved = dict(ctx.medias)
        ctx.medias.clear()
        try:
            resp = client.post("/api/projection/build")
            assert resp.status_code == 409
        finally:
            ctx.medias.update(saved)

    @patch("vtscore.projection.umap_projection.fit_projection", side_effect=_fake_fit_projection)
    def test_build_and_poll(self, _mock_fit, client):
        resp = client.post("/api/projection/build")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] in ("building", "ready")

        _wait_projection()

        meta_resp = client.get("/api/projection/meta")
        assert meta_resp.status_code == 200
        meta = meta_resp.get_json()
        assert meta["status"] == "ready"
        assert "projection_id" in meta
        assert meta["point_count"] > 0
        assert len(meta["levels"]) >= 1

    def test_cached_projection_skips_rebuild(self, client):
        ctx = get_active_context()
        rng = np.random.default_rng(99)
        ids = list(ctx.medias.keys())[:3]
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("cached-id", ids, coords, "trivial")
        pyr = build_pyramid(proj, n_levels=1)
        ctx._projection = proj
        ctx._pyramid = pyr

        resp = client.post("/api/projection/build")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ready"
        assert body["projection_id"] == "cached-id"


class TestProjectionPersistence:
    """Sidecar save/load integration in the build route."""

    def test_build_loads_from_sidecar(self, client):
        """When a valid sidecar exists, build returns ready without computing."""
        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())
        rng = np.random.default_rng(55)
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("sidecar-pid", ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)

        from vtscore.projection.persistence import save_projection

        fake_pkl = "/tmp/test_sidecar_load.pkl"  # noqa: S108
        save_projection(fake_pkl, proj, pyr)

        with patch(
            "vtsearch.routes.projection._pkl_path_for",
            return_value=fake_pkl,
        ):
            resp = client.post("/api/projection/build")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["status"] == "ready"
            assert body["projection_id"] == "sidecar-pid"

        assert ctx._pyramid is not None
        assert ctx._pyramid.projection_id == "sidecar-pid"

        import os

        os.unlink(fake_pkl.replace(".pkl", ".projection"))

    def test_build_skips_stale_sidecar(self, client):
        """A sidecar with mismatched ids is ignored."""
        ctx = get_active_context()
        wrong_ids = [999990, 999991, 999992]
        rng = np.random.default_rng(66)
        coords = rng.standard_normal((3, 2)).astype(np.float32)
        proj = Projection("stale-pid", wrong_ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=1)

        from vtscore.projection.persistence import save_projection

        fake_pkl = "/tmp/test_sidecar_stale.pkl"  # noqa: S108
        save_projection(fake_pkl, proj, pyr)

        with (
            patch(
                "vtsearch.routes.projection._pkl_path_for",
                return_value=fake_pkl,
            ),
            patch(
                "vtscore.projection.umap_projection.fit_projection",
                side_effect=_fake_fit_projection,
            ),
        ):
            resp = client.post("/api/projection/build")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["status"] in ("building", "ready")
            if body["status"] == "building":
                _wait_projection()
            assert ctx._pyramid is not None
            assert ctx._pyramid.projection_id != "stale-pid"

        import os

        sidecar = fake_pkl.replace(".pkl", ".projection")
        if os.path.exists(sidecar):
            os.unlink(sidecar)

    @patch("vtscore.projection.umap_projection.fit_projection", side_effect=_fake_fit_projection)
    def test_build_saves_sidecar(self, _mock_fit, client, tmp_path):
        """After a fresh build, the projection is persisted to a sidecar."""
        fake_pkl = str(tmp_path / "persist_test.pkl")
        with patch(
            "vtsearch.routes.projection._pkl_path_for",
            return_value=fake_pkl,
        ):
            resp = client.post("/api/projection/build")
            assert resp.status_code == 200
            _wait_projection()

        from pathlib import Path

        sidecar = Path(fake_pkl).with_suffix(".projection")
        assert sidecar.exists(), "sidecar should be written after build"


class TestProjectionTiles:
    """``GET /api/projection/tiles/<level>/<tx>/<ty>``."""

    def test_404_when_not_built(self, client):
        resp = client.get("/api/projection/tiles/0/0/0")
        assert resp.status_code == 404

    def test_empty_tile_returns_empty_cells(self, client):
        ctx = get_active_context()
        rng = np.random.default_rng(7)
        ids = list(ctx.medias.keys())[:4]
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("tile-test", ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)
        ctx._projection = proj
        ctx._pyramid = pyr

        resp = client.get("/api/projection/tiles/0/999/999")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["cells"] == []

    def test_populated_tile(self, client):
        ctx = get_active_context()
        rng = np.random.default_rng(7)
        ids = list(ctx.medias.keys())[:4]
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("tile-test-2", ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)
        ctx._projection = proj
        ctx._pyramid = pyr

        found_cells = False
        for level, tx, ty in pyr.tiles:
            resp = client.get(f"/api/projection/tiles/{level}/{tx}/{ty}")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["level"] == level
            assert body["tx"] == tx
            assert body["ty"] == ty
            if body["cells"]:
                found_cells = True
                cell = body["cells"][0]
                assert "q" in cell
                assert "r" in cell
                assert "cx" in cell
                assert "cy" in cell
                assert "count" in cell
                assert "rep_id" in cell
            break

        assert found_cells, "Expected at least one tile with cells"
