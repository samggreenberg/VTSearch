"""Tests for the VTSBrowse projection routes.

``POST /api/projection/build``, ``GET /api/projection/meta``, and
``GET /api/projection/tiles/<shape>/<level>/<tx>/<ty>``.
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
        ctx._pyramids = {"hex": pyr}

        resp = client.get("/api/projection/meta")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ready"
        assert body["projection_id"] == "test-proj-id"
        assert body["bin_shape"] == "hex"
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

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
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
        ctx._pyramids = {"hex": pyr}

        resp = client.post("/api/projection/build")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ready"
        assert body["projection_id"] == "cached-id"


class TestProjectionPersistence:
    """Container-based projection persistence in the build route."""

    def test_build_loads_from_container(self, client, tmp_path):
        """When a valid projection exists in the container, build returns ready without computing."""
        import pickle as _pickle

        from vtscore.datasets.container import append_projection, write_container

        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())
        rng = np.random.default_rng(55)
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("container-pid", ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)

        fake_pkl = tmp_path / "test_container_load.pkl"
        pkl_bytes = _pickle.dumps({"medias": {}})
        write_container(fake_pkl, pkl_bytes, {"format_version": 1})
        append_projection(fake_pkl, proj, pyr)

        with patch(
            "vtsearch.routes.projection._pkl_path_for",
            return_value=str(fake_pkl),
        ):
            resp = client.post("/api/projection/build")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["status"] == "ready"
            assert body["projection_id"] == "container-pid"

        assert ctx._pyramids.get("hex") is not None
        assert ctx._pyramids["hex"].projection_id == "container-pid"

    def test_build_skips_stale_container_projection(self, client, tmp_path):
        """A container projection with mismatched ids is ignored."""
        import pickle as _pickle

        from vtscore.datasets.container import append_projection, write_container

        ctx = get_active_context()
        wrong_ids = [999990, 999991, 999992]
        rng = np.random.default_rng(66)
        coords = rng.standard_normal((3, 2)).astype(np.float32)
        proj = Projection("stale-pid", wrong_ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=1)

        fake_pkl = tmp_path / "test_container_stale.pkl"
        pkl_bytes = _pickle.dumps({"medias": {}})
        write_container(fake_pkl, pkl_bytes, {"format_version": 1})
        append_projection(fake_pkl, proj, pyr)

        with (
            patch(
                "vtsearch.routes.projection._pkl_path_for",
                return_value=str(fake_pkl),
            ),
            patch(
                "vtscore.projection.fit_projection",
                side_effect=_fake_fit_projection,
            ),
        ):
            resp = client.post("/api/projection/build")
            assert resp.status_code == 200
            body = resp.get_json()
            assert body["status"] in ("building", "ready")
            if body["status"] == "building":
                _wait_projection()
            assert ctx._pyramids.get("hex") is not None
            assert ctx._pyramids["hex"].projection_id != "stale-pid"

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_build_persists_to_container(self, _mock_fit, client, tmp_path):
        """After a fresh build, the projection is persisted into the container."""
        import pickle as _pickle

        from vtscore.datasets.container import read_projection, write_container

        fake_pkl = tmp_path / "persist_test.pkl"
        pkl_bytes = _pickle.dumps({"medias": {}})
        write_container(fake_pkl, pkl_bytes, {"format_version": 1})

        with patch(
            "vtsearch.routes.projection._pkl_path_for",
            return_value=str(fake_pkl),
        ):
            resp = client.post("/api/projection/build")
            assert resp.status_code == 200
            _wait_projection()

        loaded = read_projection(fake_pkl)
        assert loaded is not None, "projection should be persisted into the container"


class TestProjectionTiles:
    """``GET /api/projection/tiles/<shape>/<level>/<tx>/<ty>``."""

    def test_404_when_not_built(self, client):
        resp = client.get("/api/projection/tiles/hex/0/0/0")
        assert resp.status_code == 404

    def test_empty_tile_returns_empty_cells(self, client):
        ctx = get_active_context()
        rng = np.random.default_rng(7)
        ids = list(ctx.medias.keys())[:4]
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("tile-test", ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}

        resp = client.get("/api/projection/tiles/hex/0/999/999")
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
        ctx._pyramids = {"hex": pyr}

        found_cells = False
        for level, tx, ty in pyr.tiles:
            resp = client.get(f"/api/projection/tiles/hex/{level}/{tx}/{ty}")
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
                # Each cell carries the full member id list (for selection), and
                # it agrees with the cell's count and includes its representative.
                assert "member_ids" in cell
                assert len(cell["member_ids"]) == cell["count"]
                assert cell["rep_id"] in cell["member_ids"]
            break

        assert found_cells, "Expected at least one tile with cells"

    def test_tile_member_ids_cover_every_item(self, client):
        """Across a level, the served cells' member ids partition the dataset."""
        ctx = get_active_context()
        rng = np.random.default_rng(11)
        ids = list(ctx.medias.keys())[:6]
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("tile-members", ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}

        recovered: list[int] = []
        for level, tx, ty in pyr.tiles:
            if level != 0:
                continue
            resp = client.get(f"/api/projection/tiles/hex/{level}/{tx}/{ty}")
            assert resp.status_code == 200
            for cell in resp.get_json()["cells"]:
                recovered.extend(cell["member_ids"])
        assert sorted(recovered) == sorted(ids)

    def test_served_tile_is_cacheable(self, client):
        ctx = get_active_context()
        rng = np.random.default_rng(7)
        ids = list(ctx.medias.keys())[:4]
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("tile-cache-hdr", ids, coords, "pca")
        ctx._projection = proj
        ctx._pyramids = {"hex": build_pyramid(proj, n_levels=2)}

        resp = client.get("/api/projection/tiles/hex/0/999/999")
        assert resp.status_code == 200
        # Frozen tiles are immutable for the dataset's life, so the browser may
        # reuse them without a round-trip — but only keyed by the dataset header.
        assert "max-age=" in resp.headers["Cache-Control"]
        assert "immutable" in resp.headers["Cache-Control"]
        assert "X-Dataset-Id" in resp.headers["Vary"]

    def test_not_built_404_is_not_cached(self, client):
        # A projection that isn't built yet will exist later, so the negative
        # response must not be frozen into the browser's cache.
        resp = client.get("/api/projection/tiles/hex/0/0/0")
        assert resp.status_code == 404
        assert "immutable" not in resp.headers.get("Cache-Control", "")


class TestBinShapeToggle:
    """Hex/square bin-shape selection across build, meta, and tiles."""

    def _seed_hex(self, ctx, pid: str = "shape-pid"):
        """Cache a hex projection + pyramid on the context and return them.

        The projection must span the *whole* embedding matrix (every media id),
        since that's what the inline re-bin path checks before reusing the
        shared layout for the other shape.
        """
        rng = np.random.default_rng(13)
        ids = sorted(ctx.medias.keys())
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection(pid, ids, coords, "pca")
        pyr = build_pyramid(proj, bin_shape="hex", n_levels=2)
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}
        return proj, ids

    def test_build_square_reuses_layout(self, client):
        """Requesting square when only hex exists re-bins inline (no UMAP) and is ready."""
        ctx = get_active_context()
        proj, _ids = self._seed_hex(ctx)

        # No fit_projection patch: a re-fit here would be a bug (and slow), so
        # reaching ready proves the shared layout was reused.
        resp = client.post("/api/projection/build", json={"shape": "square"})
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ready"
        assert body["projection_id"] == proj.projection_id
        assert ctx._pyramids.get("square") is not None
        assert ctx._pyramids["square"].bin_shape == "square"

    def test_meta_per_shape(self, client):
        ctx = get_active_context()
        self._seed_hex(ctx)
        client.post("/api/projection/build", json={"shape": "square"})

        hex_meta = client.get("/api/projection/meta?shape=hex").get_json()
        sq_meta = client.get("/api/projection/meta?shape=square").get_json()
        assert hex_meta["bin_shape"] == "hex"
        assert sq_meta["bin_shape"] == "square"
        # Same frozen layout underlies both binnings.
        assert hex_meta["projection_id"] == sq_meta["projection_id"]

    def test_square_tiles_served(self, client):
        ctx = get_active_context()
        self._seed_hex(ctx)
        client.post("/api/projection/build", json={"shape": "square"})
        pyr = ctx._pyramids["square"]

        for level, tx, ty in pyr.tiles:
            resp = client.get(f"/api/projection/tiles/square/{level}/{tx}/{ty}")
            assert resp.status_code == 200
            assert resp.get_json()["level"] == level
            break

    def test_unknown_shape_rejected(self, client):
        ctx = get_active_context()
        self._seed_hex(ctx)
        assert client.get("/api/projection/meta?shape=triangle").status_code == 400
        assert client.get("/api/projection/tiles/triangle/0/0/0").status_code == 400
        assert client.post("/api/projection/build", json={"shape": "triangle"}).status_code == 400

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_square_persisted_alongside_hex(self, _mock_fit, client, tmp_path):
        """Building both shapes leaves both pyramids readable from the container."""
        import pickle as _pickle

        from vtscore.datasets.container import read_projection, write_container

        fake_pkl = tmp_path / "both_shapes.pkl"
        pkl_bytes = _pickle.dumps({"medias": {}})
        write_container(fake_pkl, pkl_bytes, {"format_version": 1})

        with patch(
            "vtsearch.routes.projection._pkl_path_for",
            return_value=str(fake_pkl),
        ):
            client.post("/api/projection/build", json={"shape": "hex"})
            _wait_projection()
            resp = client.post("/api/projection/build", json={"shape": "square"})
            assert resp.get_json()["status"] == "ready"

        hex_loaded = read_projection(fake_pkl, "hex")
        sq_loaded = read_projection(fake_pkl, "square")
        assert hex_loaded is not None
        assert sq_loaded is not None
        assert hex_loaded[1].bin_shape == "hex"
        assert sq_loaded[1].bin_shape == "square"


class TestProjectionSubset:
    """Subset projection: UMAP a handful of ids (the positives of a Find run).

    Driven by ``POST /api/projection/build`` with an ``ids`` body and the
    ``?subset=1`` selector on ``meta``/``tiles``.  The subset layout is
    ephemeral (never persisted) and lives alongside the full projection.
    """

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_subset_build_and_poll(self, _mock_fit, client):
        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())[:4]

        resp = client.post("/api/projection/build", json={"ids": ids})
        assert resp.status_code == 200
        assert resp.get_json()["status"] in ("building", "ready")
        _wait_projection()

        meta = client.get("/api/projection/meta?subset=1").get_json()
        assert meta["status"] == "ready"
        assert meta["point_count"] == len(ids)
        assert ctx._subset_projection is not None
        assert ctx._subset_ids == ids

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_subset_does_not_clobber_full(self, _mock_fit, client):
        """A full build and a subset build coexist in separate slots."""
        ctx = get_active_context()
        client.post("/api/projection/build")
        _wait_projection()
        full_proj = ctx._projection
        assert full_proj is not None

        ids = sorted(ctx.medias.keys())[:3]
        client.post("/api/projection/build", json={"ids": ids})
        _wait_projection()

        # Full projection is untouched; subset is its own (smaller) layout.
        assert ctx._projection is full_proj
        assert ctx._subset_projection is not None
        assert ctx._subset_projection is not full_proj
        assert len(ctx._subset_projection.ids) == len(ids)

        # Full meta still reports the full point count, not the subset's.
        full_meta = client.get("/api/projection/meta").get_json()
        assert full_meta["status"] == "ready"
        assert full_meta["point_count"] == len(ctx.medias)

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_subset_tiles_served_with_flag(self, _mock_fit, client):
        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())[:4]
        client.post("/api/projection/build", json={"ids": ids})
        _wait_projection()

        pyr = ctx._subset_pyramids["hex"]
        for level, tx, ty in pyr.tiles:
            resp = client.get(f"/api/projection/tiles/hex/{level}/{tx}/{ty}?subset=1")
            assert resp.status_code == 200
            assert resp.get_json()["level"] == level
            break
        # Without the subset flag the full pyramid isn't built, so 404.
        assert client.get("/api/projection/tiles/hex/0/0/0").status_code == 404

    def test_subset_empty_ids_returns_409(self, client):
        resp = client.post("/api/projection/build", json={"ids": []})
        assert resp.status_code == 409

    def test_subset_bad_ids_returns_400(self, client):
        assert client.post("/api/projection/build", json={"ids": "nope"}).status_code == 400
        assert client.post("/api/projection/build", json={"ids": ["a", "b"]}).status_code == 400

    def test_subset_unknown_ids_returns_409(self, client):
        """Ids that aren't in the dataset yield nothing to project."""
        resp = client.post("/api/projection/build", json={"ids": [999990, 999991]})
        assert resp.status_code == 409

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_subset_meta_idle_without_build(self, _mock_fit, client):
        """Subset meta is idle until a subset build runs (even if full is ready)."""
        client.post("/api/projection/build")
        _wait_projection()
        assert client.get("/api/projection/meta?subset=1").get_json()["status"] == "idle"


class TestSubsetRemove:
    """``POST /api/projection/subset/remove``: cull ids without re-fitting UMAP."""

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_remove_drops_ids_and_keeps_coords(self, _mock_fit, client):
        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())[:5]
        client.post("/api/projection/build", json={"ids": ids})
        _wait_projection()

        before = ctx._subset_projection
        proj_id_before = before.projection_id
        # Coordinates of an id we will KEEP must be byte-for-byte unchanged.
        keep_id = ids[-1]
        keep_coord = before.coords[before.ids.index(keep_id)].copy()

        removed = ids[:2]
        resp = client.post("/api/projection/subset/remove", json={"ids": removed})
        assert resp.status_code == 200
        meta = resp.get_json()

        # Same layout identity, bumped content version, smaller point count.
        assert meta["projection_id"] == proj_id_before
        assert meta["content_version"] == 1
        assert meta["point_count"] == len(ids) - len(removed)

        after = ctx._subset_projection
        assert set(after.ids) == set(ids) - set(removed)
        assert ctx._subset_ids == sorted(set(ids) - set(removed))
        # The kept point did not move — no re-fit happened.
        np.testing.assert_array_equal(after.coords[after.ids.index(keep_id)], keep_coord)

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_remove_shrinks_bounds_to_survivors(self, _mock_fit, client):
        """Served bounds track the surviving points so the client re-frames.

        ``rebin_like`` keeps the template's bounds (grid identity), but the
        route stamps the reduced projection's extent over them — otherwise the
        browse canvas's post-cull zoom-to-fit and the minimap keep framing
        dead space where the culled points used to be.
        """
        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())[:6]
        client.post("/api/projection/build", json={"ids": ids})
        _wait_projection()

        proj = ctx._subset_projection
        bounds_before = ctx._subset_pyramids["hex"].bounds
        # Remove the point at the extreme right of the layout so the surviving
        # extent provably shrinks along x.
        xmax_idx = int(np.argmax(proj.coords[:, 0]))
        removed = [proj.ids[xmax_idx]]

        meta = client.post("/api/projection/subset/remove", json={"ids": removed}).get_json()

        survivors = ctx._subset_projection
        assert tuple(meta["bounds"]) == survivors.bounds
        assert tuple(meta["bounds"]) != tuple(bounds_before)
        assert meta["bounds"][2] < bounds_before[2]  # xmax shrank
        # The stored pyramid serves the same shrunken bounds on later meta GETs.
        again = client.get("/api/projection/meta", query_string={"subset": "1"}).get_json()
        assert tuple(again["bounds"]) == survivors.bounds

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_remove_bumps_version_each_call(self, _mock_fit, client):
        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())[:6]
        client.post("/api/projection/build", json={"ids": ids})
        _wait_projection()

        m1 = client.post("/api/projection/subset/remove", json={"ids": [ids[0]]}).get_json()
        m2 = client.post("/api/projection/subset/remove", json={"ids": [ids[1]]}).get_json()
        assert m1["content_version"] == 1
        assert m2["content_version"] == 2
        assert m2["projection_id"] == m1["projection_id"]

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_remove_rebuilds_both_shapes(self, _mock_fit, client):
        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())[:6]
        client.post("/api/projection/build", json={"ids": ids})
        _wait_projection()
        # Build the square binning too, so both shapes are present.
        client.post("/api/projection/build", json={"ids": ids, "shape": "square"})
        _wait_projection()
        assert set(ctx._subset_pyramids) == {"hex", "square"}

        client.post("/api/projection/subset/remove", json={"ids": [ids[0]], "shape": "square"})
        # Both shapes re-binned over the reduced layout.
        assert ctx._subset_pyramids["hex"].point_count == len(ids) - 1
        assert ctx._subset_pyramids["square"].point_count == len(ids) - 1

    def test_remove_without_subset_returns_409(self, client):
        resp = client.post("/api/projection/subset/remove", json={"ids": [1]})
        assert resp.status_code == 409

    def test_remove_empty_ids_rejected(self, client):
        resp = client.post("/api/projection/subset/remove", json={"ids": []})
        assert resp.status_code == 422


class TestForceReproject:
    """``force`` re-fits UMAP over the displayed items into a fresh layout.

    Powers the Browser's "Re-project" button: unlike a plain rebuild (which
    short-circuits on the cached/persisted layout) it always runs a new fit and
    returns a new ``projection_id``.
    """

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_force_full_rebuild_makes_new_layout(self, _mock_fit, client):
        ctx = get_active_context()
        client.post("/api/projection/build")
        _wait_projection()
        first_id = ctx._pyramids["hex"].projection_id

        # A plain rebuild short-circuits on the cached pyramid (same id)...
        again = client.post("/api/projection/build").get_json()
        assert again["status"] == "ready"
        assert again["projection_id"] == first_id

        # ...but force re-fits into a brand-new layout.
        resp = client.post("/api/projection/build", json={"force": True})
        assert resp.status_code == 200
        assert resp.get_json()["status"] in ("building", "ready")
        _wait_projection()
        assert ctx._pyramids["hex"].projection_id != first_id

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_force_full_replaces_persisted(self, _mock_fit, client, tmp_path):
        import pickle as _pickle

        from vtscore.datasets.container import read_projection, write_container

        fake_pkl = tmp_path / "reproject_persist.pkl"
        write_container(fake_pkl, _pickle.dumps({"medias": {}}), {"format_version": 1})

        with patch("vtsearch.routes.projection._pkl_path_for", return_value=str(fake_pkl)):
            client.post("/api/projection/build")
            _wait_projection()
            first = read_projection(fake_pkl, "hex")
            assert first is not None
            first_id = first[0].projection_id

            client.post("/api/projection/build", json={"force": True})
            _wait_projection()
            second = read_projection(fake_pkl, "hex")
            assert second is not None
            assert second[0].projection_id != first_id

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_force_full_drops_stale_other_shape(self, _mock_fit, client, tmp_path):
        import pickle as _pickle

        from vtscore.datasets.container import read_projection, write_container

        fake_pkl = tmp_path / "reproject_shapes.pkl"
        write_container(fake_pkl, _pickle.dumps({"medias": {}}), {"format_version": 1})

        with patch("vtsearch.routes.projection._pkl_path_for", return_value=str(fake_pkl)):
            client.post("/api/projection/build", json={"shape": "hex"})
            _wait_projection()
            client.post("/api/projection/build", json={"shape": "square"})
            _wait_projection()
            assert read_projection(fake_pkl, "square") is not None

            # Force-rebuild hex: the stale square entry must be dropped so it
            # can't resurrect the old (now-superseded) shared coordinates.
            client.post("/api/projection/build", json={"shape": "hex", "force": True})
            _wait_projection()
            assert read_projection(fake_pkl, "square") is None
            assert read_projection(fake_pkl, "hex") is not None

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_force_subset_refits_same_ids(self, _mock_fit, client):
        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())[:5]
        client.post("/api/projection/build", json={"ids": ids})
        _wait_projection()
        first_id = ctx._subset_projection.projection_id

        # Same ids without force → cached, same layout id.
        again = client.post("/api/projection/build", json={"ids": ids}).get_json()
        assert again["status"] == "ready"
        assert again["projection_id"] == first_id

        # Same ids with force → fresh fit, new id, unchanged membership.
        resp = client.post("/api/projection/build", json={"ids": ids, "force": True})
        assert resp.status_code == 200
        _wait_projection()
        after = ctx._subset_projection
        assert after.projection_id != first_id
        assert after.ids == ids


class TestBrowseCompactSetting:
    """The per-media-type ``browse_compact`` preference flows into the UMAP fit.

    ``compact`` controls Stage-1 coordinates, so the build route reads the
    setting for the dataset's media type and threads it into ``fit_projection``.
    Defaults to on; the Settings → Browser toggle can turn it off per type.
    """

    @staticmethod
    def _capturing_fake(captured: dict):
        def fake(matrix, ids, **kwargs):
            captured["compact"] = kwargs.get("compact")
            return _fake_fit_projection(matrix, ids, **kwargs)

        return fake

    def _media_type(self, ctx) -> str:
        first = next(iter(ctx.medias.values()))
        return first.get("media_type", "audio")

    def test_compact_on_by_default(self, client):
        captured: dict = {}
        with patch("vtscore.projection.fit_projection", side_effect=self._capturing_fake(captured)):
            client.post("/api/projection/build")
            _wait_projection()
        assert captured["compact"] is True

    def test_compact_setting_disables_packing(self, client):
        ctx = get_active_context()
        media_type = self._media_type(ctx)
        resp = client.put("/api/settings", json={"browse_compact": {media_type: False}})
        assert resp.status_code == 200

        captured: dict = {}
        with patch("vtscore.projection.fit_projection", side_effect=self._capturing_fake(captured)):
            client.post("/api/projection/build")
            _wait_projection()
        assert captured["compact"] is False

    def test_compact_setting_applies_to_subset_build(self, client):
        ctx = get_active_context()
        media_type = self._media_type(ctx)
        client.put("/api/settings", json={"browse_compact": {media_type: False}})
        ids = sorted(ctx.medias.keys())[:4]

        captured: dict = {}
        with patch("vtscore.projection.fit_projection", side_effect=self._capturing_fake(captured)):
            client.post("/api/projection/build", json={"ids": ids})
            _wait_projection()
        assert captured["compact"] is False
