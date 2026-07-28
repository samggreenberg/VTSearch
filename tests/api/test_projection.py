"""Tests for the VTSBrowse projection routes.

``POST /api/projection/build``, ``GET /api/projection/meta``, and
``GET /api/projection/tiles/<level>/<tx>/<ty>``.

The bin shape is derived from the dataset's media type (squares for
browsable-thumbnail media, hexes otherwise), not requested by the client. The
generated fixtures are audio, which now tiles as *squares* (waveform
thumbnails); an autouse fixture defaults the route's resolved media type to
``text`` — the canonical hex type — so these lattice-agnostic tests stay on the
hex path unless a test overrides the media type.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

import app as app_module  # noqa: F401
from vtscore.concurrency.async_jobs import projection_jobs
from vtscore.projection import Projection, build_pyramid
from vtscore.state.core import get_active_context


@pytest.fixture(autouse=True)
def _default_hex_media_type():
    """Default the route's resolved media type to a hex-lattice type.

    The bin shape is a property of the dataset's media type. The generated test
    fixtures are audio, which now tiles as squares (waveform thumbnails), so
    resolve to ``text`` — the canonical hex type — by default, keeping the many
    lattice-agnostic tests below on the hex path. Tests that exercise a specific
    media type (image → square, audio → square) patch ``_media_type_for``
    themselves, which overrides this.
    """
    with patch("vtsearch.routes.projection._media_type_for", return_value="text"):
        yield


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

    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_building_while_parked_behind_another_job(self, _mock_fit, client):
        """A full build parked behind another projection job still reports
        ``building``, not ``idle``.

        The projection runner is a single app-wide slot shared across every
        dataset's full build and every subset build. When another job holds the
        runner, this dataset's build is parked in the pending slot. Meta must
        look up this context's own tracked job (``ctx._full_job_id``) and report
        its ``building`` status — *not* ask the runner for ``current()``, which
        returns the unrelated job hogging the slot and made meta fall back to
        ``idle``. That false ``idle`` made the frontend's poll loop re-issue
        ``build()`` forever, hanging the Browse spinner until the blocking job
        happened to free the runner (the reported "reload fixes it" bug).
        """
        import threading

        started = threading.Event()
        release = threading.Event()

        def _blocker(job):
            started.set()
            release.wait(timeout=30)

        # Occupy the single runner with an unrelated job (a different dataset id
        # so the build we fire next can't coalesce into it).
        projection_jobs.start(("blocker",), _blocker, dataset_id="__other__")
        assert started.wait(timeout=5)

        try:
            # The runner is busy, so this build parks in the pending slot.
            resp = client.post("/api/projection/build")
            assert resp.status_code == 200
            build_body = resp.get_json()
            assert build_body["status"] == "building"

            # Regression: meta must find the parked build and say "building".
            meta = client.get("/api/projection/meta").get_json()
            assert meta["status"] == "building"
        finally:
            release.set()

        # Freeing the runner promotes the parked build; it then finishes.
        build_job = projection_jobs.get(build_body["job_id"])
        assert build_job is not None
        assert build_job.done_event.wait(timeout=30)

        meta = client.get("/api/projection/meta").get_json()
        assert meta["status"] == "ready"
        assert meta["point_count"] > 0


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
        ctx._pyramids = {"hex": pyr}

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
        ctx._pyramids = {"hex": pyr}

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
            resp = client.get(f"/api/projection/tiles/{level}/{tx}/{ty}")
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

        resp = client.get("/api/projection/tiles/0/999/999")
        assert resp.status_code == 200
        # Frozen tiles are immutable for the dataset's life, so the browser may
        # reuse them without a round-trip — but only keyed by the dataset header.
        assert "max-age=" in resp.headers["Cache-Control"]
        assert "immutable" in resp.headers["Cache-Control"]
        assert "X-Dataset-Id" in resp.headers["Vary"]

    def test_not_built_404_is_not_cached(self, client):
        # A projection that isn't built yet will exist later, so the negative
        # response must not be frozen into the browser's cache.
        resp = client.get("/api/projection/tiles/0/0/0")
        assert resp.status_code == 404
        assert "immutable" not in resp.headers.get("Cache-Control", "")


class TestBinShapeByMediaType:
    """The bin shape is a fixed property of the dataset's media type.

    Browsable-thumbnail media (image / video / document, and audio via its
    waveform PNG) → square; text → hex. The client never sends a shape; the
    routes derive it from the active dataset via ``MediaType.has_thumbnail``.
    """

    @patch("vtsearch.routes.projection._media_type_for", return_value="audio")
    def test_audio_dataset_reports_square(self, _mock_mt, client):
        """Audio resolves to the square lattice (its waveform thumbnails tile
        as squares like image/video)."""
        ctx = get_active_context()
        rng = np.random.default_rng(13)
        ids = sorted(ctx.medias.keys())
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection("audio-pid", ids, coords, "pca")
        ctx._projection = proj
        ctx._pyramids = {"square": build_pyramid(proj, bin_shape="square", n_levels=2)}

        meta = client.get("/api/projection/meta").get_json()
        assert meta["bin_shape"] == "square"

    @patch("vtsearch.routes.projection._media_type_for", return_value="image")
    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_image_dataset_builds_and_serves_square(self, _mock_fit, _mock_mt, client):
        """An image dataset is tiled with squares end to end (build, meta, tiles)."""
        ctx = get_active_context()

        resp = client.post("/api/projection/build")
        assert resp.status_code == 200
        assert resp.get_json()["status"] in ("building", "ready")
        _wait_projection()

        assert ctx._pyramids.get("square") is not None
        assert ctx._pyramids["square"].bin_shape == "square"

        meta = client.get("/api/projection/meta").get_json()
        assert meta["status"] == "ready"
        assert meta["bin_shape"] == "square"
        assert meta["media_type"] == "image"

        # Tiles are served at the shape-agnostic URL; the route resolves the
        # square pyramid from the (image) media type.
        pyr = ctx._pyramids["square"]
        for level, tx, ty in pyr.tiles:
            tile_resp = client.get(f"/api/projection/tiles/{level}/{tx}/{ty}")
            assert tile_resp.status_code == 200
            assert tile_resp.get_json()["level"] == level
            break

    @patch("vtsearch.routes.projection._media_type_for", return_value="image")
    @patch("vtscore.projection.fit_projection", side_effect=_fake_fit_projection)
    def test_square_persisted_for_image_dataset(self, _mock_fit, _mock_mt, client, tmp_path):
        """An image dataset's build persists its square pyramid into the container."""
        import pickle as _pickle

        from vtscore.datasets.container import read_projection, write_container

        fake_pkl = tmp_path / "image_square.pkl"
        pkl_bytes = _pickle.dumps({"medias": {}})
        write_container(fake_pkl, pkl_bytes, {"format_version": 1})

        with patch(
            "vtsearch.routes.projection._pkl_path_for",
            return_value=str(fake_pkl),
        ):
            client.post("/api/projection/build")
            _wait_projection()

        sq_loaded = read_projection(fake_pkl, "square")
        assert sq_loaded is not None
        assert sq_loaded[1].bin_shape == "square"
        # Only the one shape the media type uses is written.
        assert read_projection(fake_pkl, "hex") is None


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
            resp = client.get(f"/api/projection/tiles/{level}/{tx}/{ty}?subset=1")
            assert resp.status_code == 200
            assert resp.get_json()["level"] == level
            break
        # Without the subset flag the full pyramid isn't built, so 404.
        assert client.get("/api/projection/tiles/0/0/0").status_code == 404

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
        """Re-project clears every persisted shape, including a legacy square one.

        Containers written when the bin shape was user-toggleable can carry both
        a hex and a square projection. A forced rebuild must drop all of them so
        a stale entry can't resurrect the old (now-superseded) shared
        coordinates.
        """
        import pickle as _pickle

        from vtscore.datasets.container import append_projection, read_projection, write_container

        ctx = get_active_context()
        ids = sorted(ctx.medias.keys())
        rng = np.random.default_rng(77)
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        legacy = Projection("legacy-pid", ids, coords, "pca")

        fake_pkl = tmp_path / "reproject_shapes.pkl"
        write_container(fake_pkl, _pickle.dumps({"medias": {}}), {"format_version": 1})
        # Seed a legacy container with both shapes persisted.
        append_projection(fake_pkl, legacy, build_pyramid(legacy, bin_shape="hex", n_levels=2))
        append_projection(fake_pkl, legacy, build_pyramid(legacy, bin_shape="square", n_levels=2))
        assert read_projection(fake_pkl, "square") is not None

        with patch("vtsearch.routes.projection._pkl_path_for", return_value=str(fake_pkl)):
            # Force-rebuild the (audio → hex) dataset: the stale square entry is
            # dropped, and the fresh hex layout replaces the legacy one.
            client.post("/api/projection/build", json={"force": True})
            _wait_projection()
            assert read_projection(fake_pkl, "square") is None
            hex_after = read_projection(fake_pkl, "hex")
            assert hex_after is not None
            assert hex_after[0].projection_id != "legacy-pid"

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

    @pytest.fixture(autouse=True)
    def _audio_media_type(self):
        """These tests set the compact preference for the fixtures' real media
        type (audio) and assert the route reads it, so keep the route resolving
        ``audio`` here — overriding the module-level hex default."""
        with patch("vtsearch.routes.projection._media_type_for", return_value="audio"):
            yield

    @staticmethod
    def _capturing_fake(captured: dict):
        def fake(matrix, ids, **kwargs):
            captured["compact"] = kwargs.get("compact")
            return _fake_fit_projection(matrix, ids, **kwargs)

        return fake

    def _media_type(self, ctx) -> str:
        first = next(iter(ctx.medias.values()))
        return first.get("media_type", "audio")

    def test_compact_off_by_default(self, client):
        # Compaction defaults OFF: the empirical sweep found it costs ~2% taxonomy
        # separability and ~5-6% neighbourhood structure on every dataset/embedder
        # (see PROJECTION_COMPACT_DEFAULT / docs/plans/vtsbrowse-empirical-tuning.md).
        captured: dict = {}
        with patch("vtscore.projection.fit_projection", side_effect=self._capturing_fake(captured)):
            client.post("/api/projection/build")
            _wait_projection()
        assert captured["compact"] is False

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


def test_projection_params_match(monkeypatch):
    """The persisted-projection guard invalidates a layout when UMAP params change."""
    import numpy as np

    from vtsearch.routes import projection as proj_route
    from vtscore.projection.umap_projection import Projection

    coords = np.zeros((3, 2), dtype=np.float32)
    ids = [0, 1, 2]
    umap_default = Projection("p", ids, coords, "umap", 15, 0.1)
    umap_changed = Projection("p", ids, coords, "umap", 30, 0.1)
    pca = Projection("p", ids, coords, "pca", None, None)
    legacy = Projection("p", ids, coords, "umap", None, None)

    # Active settings at the config defaults.
    monkeypatch.setattr(proj_route, "_umap_params", lambda ctx=None: (15, 0.1))
    assert proj_route._projection_params_match(umap_default) is True
    assert proj_route._projection_params_match(umap_changed) is False
    assert proj_route._projection_params_match(pca) is True
    # Legacy None params are assumed to be the config defaults.
    assert proj_route._projection_params_match(legacy) is True

    # Operator tuned the setting away from the default -> stale layouts recompute.
    monkeypatch.setattr(proj_route, "_umap_params", lambda ctx=None: (30, 0.1))
    assert proj_route._projection_params_match(legacy) is False
    assert proj_route._projection_params_match(umap_changed) is True
    assert proj_route._projection_params_match(pca) is True


class TestProjectionLabels:
    """``GET /api/projection/labels`` — the region signpost layer."""

    @staticmethod
    def _build_hex_pyramid(projection_id: str = "label-proj"):
        rng = np.random.default_rng(7)
        ctx = get_active_context()
        ids = list(ctx.medias.keys())[:5]
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection(projection_id, ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)
        return proj, pyr

    @staticmethod
    def _label_set(projection_id: str):
        from vtscore.projection import RegionLabel, make_label_set

        return make_label_set(
            projection_id,
            [
                RegionLabel(level=0, x=0.0, y=0.0, text="animal sounds", score=0.9, source="keyphrase"),
                RegionLabel(level=1, x=0.5, y=-0.5, text="dog barking", score=0.7, source="keyphrase"),
            ],
        )

    def test_idle_when_no_projection(self, client):
        resp = client.get("/api/projection/labels")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "idle"
        assert body["labels"] == []

    def test_ready_but_empty_when_no_labeler_has_run(self, client):
        ctx = get_active_context()
        proj, pyr = self._build_hex_pyramid()
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}

        resp = client.get("/api/projection/labels")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ready"
        assert body["projection_id"] == pyr.projection_id
        assert body["labels"] == []

        # And the meta advertises the absence, so the client can skip the fetch.
        assert client.get("/api/projection/meta").get_json()["has_labels"] is False

    def test_serves_labels_for_matching_layout(self, client):
        ctx = get_active_context()
        proj, pyr = self._build_hex_pyramid()
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}
        ctx._region_labels = self._label_set(pyr.projection_id)

        resp = client.get("/api/projection/labels")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["status"] == "ready"
        assert body["projection_id"] == pyr.projection_id
        assert len(body["labels"]) == 2
        first = body["labels"][0]
        assert first["text"] == "animal sounds"
        assert first["level"] == 0
        assert first["score"] == 0.9
        assert first["source"] == "keyphrase"
        # Terminal-neighbour flags default True (present but not pinned to a tree
        # in this hand-built set); the canvas fades such signs as before.
        assert first["has_coarser"] is True
        assert first["has_finer"] is True
        assert set(first) == {"level", "x", "y", "text", "score", "source", "has_coarser", "has_finer"}

        assert client.get("/api/projection/meta").get_json()["has_labels"] is True

    def test_stale_label_set_is_treated_as_absent(self, client):
        # Labels are pinned to the layout they were computed from: a set left
        # over from a previous (re-fit) layout must not be served over the new
        # coordinates.
        ctx = get_active_context()
        proj, pyr = self._build_hex_pyramid("new-layout")
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}
        ctx._region_labels = self._label_set("old-layout")

        body = client.get("/api/projection/labels").get_json()
        assert body["status"] == "ready"
        assert body["labels"] == []
        assert client.get("/api/projection/meta").get_json()["has_labels"] is False

    def test_subset_labels_are_independent(self, client):
        # The subset browse carries its own label set; the full-dataset one
        # must not bleed into it (and vice versa).
        ctx = get_active_context()
        proj, pyr = self._build_hex_pyramid("subset-layout")
        ctx._subset_projection = proj
        ctx._subset_pyramids = {"hex": pyr}
        ctx._subset_ids = list(proj.ids)
        ctx._region_labels = self._label_set("some-other-layout")

        body = client.get("/api/projection/labels?subset=1").get_json()
        assert body["status"] == "ready"
        assert body["labels"] == []

        ctx._subset_region_labels = self._label_set(pyr.projection_id)
        body = client.get("/api/projection/labels?subset=1").get_json()
        assert len(body["labels"]) == 2
        assert client.get("/api/projection/meta?subset=1").get_json()["has_labels"] is True

    def test_reset_full_projection_drops_labels(self, client):
        from vtsearch.routes.projection import _reset_full_projection

        ctx = get_active_context()
        proj, pyr = self._build_hex_pyramid()
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}
        ctx._region_labels = self._label_set(pyr.projection_id)

        _reset_full_projection(ctx)
        assert ctx._region_labels is None


class TestPersistedLabelRestore:
    """Labels persisted in the dataset container are restored on serve.

    The labeling pipeline persists the full-dataset ``RegionLabelSet`` next to
    the projection; a fresh process (no in-memory set) must serve it back —
    but only over the exact layout it was computed from, and only while its
    labeler signature matches the active pipeline's (a ``None`` active
    signature — no toponymy here — still serves: derived text pinned to the
    right layout beats nothing).
    """

    @staticmethod
    def _persist(tmp_path, projection_id: str, signature: str):
        import zipfile

        from vtscore.datasets.container import append_region_labels

        pkl = tmp_path / "container.pkl"
        if not pkl.exists():
            with zipfile.ZipFile(pkl, "w") as zf:
                zf.writestr("placeholder", b"")
        append_region_labels(pkl, TestProjectionLabels._label_set(projection_id), signature)
        return pkl

    def _serve(self, client, monkeypatch, tmp_path, *, stored_id, stored_sig, active_sig="__unset__"):
        ctx = get_active_context()
        proj, pyr = TestProjectionLabels._build_hex_pyramid("live-layout")
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}
        pkl = self._persist(tmp_path, stored_id, stored_sig)
        monkeypatch.setattr("vtsearch.routes.projection._pkl_path_for", lambda dataset_id: str(pkl))
        if active_sig != "__unset__":
            monkeypatch.setattr("vtscore.projection.signpost_prep.labeler_signature", lambda ctx: active_sig)
        return client.get("/api/projection/labels").get_json()

    def test_restores_matching_layout(self, client, monkeypatch, tmp_path):
        # Active signature is None here (the suite reports toponymy
        # unavailable), so the persisted set serves unconditionally.
        body = self._serve(client, monkeypatch, tmp_path, stored_id="live-layout", stored_sig="sig-v1")
        assert body["status"] == "ready"
        assert [lab["text"] for lab in body["labels"]] == ["animal sounds", "dog barking"]
        # And it lands in the context cache like a live-built set.
        assert get_active_context()._region_labels is not None

    def test_matching_signature_serves(self, client, monkeypatch, tmp_path):
        body = self._serve(
            client, monkeypatch, tmp_path, stored_id="live-layout", stored_sig="sig-v1", active_sig="sig-v1"
        )
        assert len(body["labels"]) == 2

    def test_stale_signature_treated_as_absent(self, client, monkeypatch, tmp_path):
        body = self._serve(
            client, monkeypatch, tmp_path, stored_id="live-layout", stored_sig="sig-old", active_sig="sig-new"
        )
        assert body["labels"] == []

    def test_wrong_layout_treated_as_absent(self, client, monkeypatch, tmp_path):
        body = self._serve(client, monkeypatch, tmp_path, stored_id="some-old-layout", stored_sig="sig-v1")
        assert body["labels"] == []

    def test_stale_signature_kicks_background_relabel(self, client, monkeypatch, tmp_path):
        # A persisted set whose signature no longer matches the active pipeline
        # is served absent, but a background rebuild is kicked so the signs
        # self-heal to the active labeler without a forced Re-project (#2404).
        import threading

        from vtscore.concurrency.async_jobs import signpost_relabel_jobs
        from vtscore.projection import RegionLabel, make_label_set

        ctx = get_active_context()
        rebuilt = make_label_set(
            "live-layout",
            [RegionLabel(level=0, x=0.0, y=0.0, text="rebuilt sign", score=0.5, source="keyphrase")],
        )

        started = threading.Event()
        proceed = threading.Event()

        def _fake_prep(c, proj, *, subset, on_progress=None):
            # Block until the test has observed the interim (unlettered) serve,
            # then install the rebuilt signs exactly as the real pipeline does.
            started.set()
            assert proceed.wait(timeout=5)
            c._region_labels = rebuilt
            return rebuilt

        monkeypatch.setattr("vtscore.projection.signpost_prep.prep_signposts", _fake_prep)

        # First serve: stale set detected → absent now, rebuild kicked.
        body = self._serve(
            client, monkeypatch, tmp_path, stored_id="live-layout", stored_sig="sig-old", active_sig="sig-new"
        )
        assert body["labels"] == []
        job_id = ctx._relabel_job_id
        assert job_id
        assert started.wait(timeout=5)

        # A second poll while the rebuild is in flight coalesces onto the same
        # job instead of queueing another.
        assert client.get("/api/projection/labels").get_json()["labels"] == []
        assert ctx._relabel_job_id == job_id
        assert len(signpost_relabel_jobs.active_jobs()) == 1

        # Let the rebuild finish; the signs self-heal on the next serve.
        proceed.set()
        job = signpost_relabel_jobs.get(job_id)
        assert job is not None
        assert job.done_event.wait(timeout=10)

        body = client.get("/api/projection/labels").get_json()
        assert [lab["text"] for lab in body["labels"]] == ["rebuilt sign"]
        assert client.get("/api/projection/meta").get_json()["has_labels"] is True

    def test_relabel_not_kicked_without_projection(self, client, monkeypatch, tmp_path):
        # The kick needs the frozen layout in hand; a stale set with no live
        # projection just serves absent (no crash, no job).
        from vtscore.concurrency.async_jobs import signpost_relabel_jobs

        ctx = get_active_context()
        pkl = self._persist(tmp_path, "live-layout", "sig-old")
        monkeypatch.setattr("vtsearch.routes.projection._pkl_path_for", lambda dataset_id: str(pkl))
        monkeypatch.setattr("vtscore.projection.signpost_prep.labeler_signature", lambda ctx: "sig-new")

        proj, pyr = TestProjectionLabels._build_hex_pyramid("live-layout")
        ctx._pyramids = {"hex": pyr}
        ctx._projection = None  # layout not resolved into the context

        body = client.get("/api/projection/labels").get_json()
        assert body["labels"] == []
        assert ctx._relabel_job_id is None
        assert signpost_relabel_jobs.active_jobs() == []


class TestDemoSignpostsLazyBuild:
    """Ground-truth signposts derived on demand from hierarchical categories.

    A dataset that ships ``/``-separated ``category`` paths (the synthetic
    world-map demo, Places365, …) lights up the signpost layer with no labeler
    run: :func:`_label_set_for` builds and caches the signs the first time meta
    or the labels endpoint is hit.
    """

    # Europe/Asia → 2 continents, 4 countries, 6 cities; 2 items per leaf city.
    _PATHS = [
        "Europe/France/Paris",
        "Europe/France/Paris",
        "Europe/France/Lyon",
        "Europe/France/Lyon",
        "Europe/Italy/Rome",
        "Europe/Italy/Rome",
        "Asia/Japan/Tokyo",
        "Asia/Japan/Tokyo",
        "Asia/Japan/Osaka",
        "Asia/Japan/Osaka",
        "Asia/China/Beijing",
        "Asia/China/Beijing",
    ]

    def _setup(self, ctx, projection_id="demo-signpost-layout"):
        rng = np.random.default_rng(11)
        ids = list(ctx.medias.keys())[: len(self._PATHS)]
        assert len(ids) == len(self._PATHS), "fixture needs at least 12 medias"
        saved = {i: ctx.medias[i].get("category") for i in ids}
        for path, mid in zip(self._PATHS, ids):
            ctx.medias[mid]["category"] = path
        coords = rng.standard_normal((len(ids), 2)).astype(np.float32)
        proj = Projection(projection_id, ids, coords, "pca")
        pyr = build_pyramid(proj, n_levels=2)
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}
        ctx._region_labels = None
        return ids, saved, pyr

    def test_meta_and_labels_build_lazily_from_categories(self, client):
        ctx = get_active_context()
        ids, saved, pyr = self._setup(ctx)
        try:
            # Meta advertises signs even though no labeler ran.
            meta = client.get("/api/projection/meta").get_json()
            assert meta["has_labels"] is True

            body = client.get("/api/projection/labels").get_json()
            assert body["status"] == "ready"
            assert body["projection_id"] == pyr.projection_id
            labels = body["labels"]
            # 2 continents + 4 countries + 6 cities = 12 signs across 3 levels.
            assert len(labels) == 12
            by_level: dict[float, list] = {}
            for lab in labels:
                by_level.setdefault(lab["level"], []).append(lab["text"])
            levels = sorted(by_level)
            assert len(levels) == 3
            assert levels[0] == 0.0
            assert set(by_level[levels[0]]) == {"Europe", "Asia"}
            assert len(by_level[levels[1]]) == 4  # countries
            assert len(by_level[levels[2]]) == 6  # cities
            assert all(lab["source"] == "ground-truth" for lab in labels)
        finally:
            for mid, cat in saved.items():
                if cat is None:
                    ctx.medias[mid].pop("category", None)
                else:
                    ctx.medias[mid]["category"] = cat

    def test_cached_and_pinned_to_layout(self, client):
        ctx = get_active_context()
        ids, saved, pyr = self._setup(ctx)
        try:
            client.get("/api/projection/labels")
            # The build is cached on the context, pinned to this layout.
            assert ctx._region_labels is not None
            assert ctx._region_labels.projection_id == pyr.projection_id
            assert len(ctx._region_labels.labels) == 12
        finally:
            for mid, cat in saved.items():
                if cat is None:
                    ctx.medias[mid].pop("category", None)
                else:
                    ctx.medias[mid]["category"] = cat

    def test_flat_categories_yield_no_signs(self, client):
        ctx = get_active_context()
        proj, pyr = TestProjectionLabels._build_hex_pyramid("flat-layout")
        ctx._projection = proj
        ctx._pyramids = {"hex": pyr}
        ctx._region_labels = None
        # The generated fixtures carry flat categories (e.g. "test-image"), so
        # the hierarchical probe declines and no signs are built.
        body = client.get("/api/projection/labels").get_json()
        assert body["status"] == "ready"
        assert body["labels"] == []
        assert client.get("/api/projection/meta").get_json()["has_labels"] is False
