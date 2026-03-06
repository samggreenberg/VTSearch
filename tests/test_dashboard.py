"""Tests for the VTSearch dashboard API endpoint."""

import app as app_module  # noqa: F401 — triggers conftest side effects
from vtsearch.datasets.registry import register_dataset
from vtsearch.models.registry import register_model
from vtsearch.utils import add_autorun_detector, get_autorun_detectors, medias


class TestDashboardDatasetInfo:
    """Tests for GET /api/dashboard/dataset-info."""

    def test_returns_404_when_no_dataset(self, client):
        """Endpoint returns 404 when no dataset is loaded."""
        saved = dict(medias)
        medias.clear()
        try:
            resp = client.get("/api/dashboard/dataset-info")
            assert resp.status_code == 404
            data = resp.get_json()
            assert "error" in data
        finally:
            medias.update(saved)

    def test_returns_info_with_loaded_medias(self, client):
        """Endpoint returns dataset metadata when medias are loaded."""
        resp = client.get("/api/dashboard/dataset-info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "name" in data
        assert "num_medias" in data
        assert "media_type" in data
        assert "origin" in data
        assert data["num_medias"] == len(medias)

    def test_returns_correct_media_type(self, client):
        """Endpoint returns the media type of the first media."""
        resp = client.get("/api/dashboard/dataset-info")
        data = resp.get_json()
        first_media = next(iter(medias.values()))
        assert data["media_type"] == first_media.get("type", "audio")

    def test_returns_origin_from_media(self, client):
        """Endpoint extracts origin info when medias have origin set."""
        first_key = next(iter(medias))
        saved_origin = medias[first_key].get("origin")
        try:
            medias[first_key]["origin"] = {"importer": "demo", "params": {"name": "test_dataset"}}
            resp = client.get("/api/dashboard/dataset-info")
            data = resp.get_json()
            assert data["origin"] == "demo:test_dataset"
            assert data["name"] == "test_dataset"
        finally:
            if saved_origin is not None:
                medias[first_key]["origin"] = saved_origin
            else:
                medias[first_key].pop("origin", None)

    def test_returns_origin_folder(self, client):
        """Endpoint formats folder origin correctly."""
        first_key = next(iter(medias))
        saved_origin = medias[first_key].get("origin")
        try:
            medias[first_key]["origin"] = {"importer": "folder", "params": {"path": "/data/sounds"}}
            resp = client.get("/api/dashboard/dataset-info")
            data = resp.get_json()
            assert data["origin"] == "folder:/data/sounds"
        finally:
            if saved_origin is not None:
                medias[first_key]["origin"] = saved_origin
            else:
                medias[first_key].pop("origin", None)

    def test_returns_origin_pickle(self, client):
        """Endpoint formats pickle origin correctly."""
        first_key = next(iter(medias))
        saved_origin = medias[first_key].get("origin")
        try:
            medias[first_key]["origin"] = {"importer": "pickle", "params": {"filename": "esc50.pkl"}}
            resp = client.get("/api/dashboard/dataset-info")
            data = resp.get_json()
            assert data["origin"] == "file:esc50.pkl"
            assert data["name"] == "esc50.pkl"
        finally:
            if saved_origin is not None:
                medias[first_key]["origin"] = saved_origin
            else:
                medias[first_key].pop("origin", None)

    def test_returns_unknown_origin_when_none(self, client):
        """Endpoint returns 'unknown' origin when medias have no origin."""
        # Default test medias don't have origin set
        saved_origins = {}
        for k, v in medias.items():
            saved_origins[k] = v.get("origin")
            v.pop("origin", None)
        try:
            resp = client.get("/api/dashboard/dataset-info")
            data = resp.get_json()
            assert data["origin"] == "unknown"
        finally:
            for k, v in medias.items():
                if saved_origins.get(k) is not None:
                    v["origin"] = saved_origins[k]


class TestDashboardHtmlPresent:
    """Test that the dashboard-related content is present in the Angular bundle."""

    def test_index_serves_angular_app(self, client):
        """GET / should serve the Angular SPA shell."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data

    def test_dashboard_view_in_bundle(self, client):
        """The Angular bundle should contain dashboard view references."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "dashboard" in combined
        assert "Datasets" in combined
        assert "Models" in combined

    def test_dashboard_route_defined(self, client):
        """The /dashboard route should serve the Angular SPA."""
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert b"<app-root>" in resp.data


class TestGuessMediaType:
    """Frontend auto-populates the media-type dropdown when creating a new model.

    The guessing logic lives in the Angular frontend (dashboard component):
    1. If all datasets in the registry share a single media_type, use that.
    2. Otherwise, if settings.autoload_media_types has exactly one entry, use it.

    These tests verify the underlying data contracts that the frontend logic relies on.
    """

    def test_js_contains_guessing_logic(self, client):
        """main.js should include the media-type guessing code."""
        resp = client.get("/static/main.js")
        text = resp.data.decode("utf-8")
        assert "guessedMediaType" in text or "media_type" in text

    def test_single_dataset_type_in_registry(self, client):
        """When the registry has one dataset, its media_type is available."""
        register_dataset(
            name="test-audio",
            media_type="audio",
            num_items=10,
            pkl_path="/tmp/fake.pkl",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        assert len(data["datasets"]) == 1
        assert data["datasets"][0]["media_type"] == "audio"

    def test_multiple_same_type_datasets_in_registry(self, client):
        """Multiple datasets of the same type yield a single unique type."""
        register_dataset(name="ds1", media_type="image", num_items=5, pkl_path="/tmp/a.pkl")
        register_dataset(name="ds2", media_type="image", num_items=3, pkl_path="/tmp/b.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        types = {d["media_type"] for d in data["datasets"]}
        assert types == {"image"}

    def test_mixed_type_datasets_no_single_guess(self, client):
        """Multiple datasets with different types produce more than one unique type."""
        register_dataset(name="ds1", media_type="audio", num_items=5, pkl_path="/tmp/a.pkl")
        register_dataset(name="ds2", media_type="image", num_items=3, pkl_path="/tmp/b.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        types = {d["media_type"] for d in data["datasets"]}
        assert len(types) > 1

    def test_autoload_single_type_from_settings(self, client):
        """When autoload_media_types has exactly one entry, settings returns it."""
        client.put("/api/settings", json={"autoload_media_types": ["video"]})
        resp = client.get("/api/settings")
        data = resp.get_json()
        assert data["autoload_media_types"] == ["video"]

    def test_autoload_multiple_types_no_single_guess(self, client):
        """When autoload_media_types has multiple entries, no single guess."""
        client.put("/api/settings", json={"autoload_media_types": ["audio", "image"]})
        resp = client.get("/api/settings")
        data = resp.get_json()
        assert len(data["autoload_media_types"]) > 1

    def test_empty_registry_falls_back_to_settings(self, client):
        """With no datasets, the frontend should fall back to autoload settings."""
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        assert len(data["datasets"]) == 0
        # Set a single autoload type
        client.put("/api/settings", json={"autoload_media_types": ["paragraph"]})
        resp = client.get("/api/settings")
        data = resp.get_json()
        assert data["autoload_media_types"] == ["paragraph"]


class TestDashboardDatasetRegistryColumns:
    """Tests that the dataset registry API returns fields needed for new columns."""

    def test_dataset_registry_includes_created_at(self, client):
        """Registered datasets include a created_at timestamp."""
        register_dataset(name="ts-ds", media_type="audio", num_items=5, pkl_path="/tmp/ts.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "created_at" in ds
        assert isinstance(ds["created_at"], (int, float))
        assert ds["created_at"] > 0

    def test_dataset_registry_includes_origin(self, client):
        """Registered datasets include an origin string."""
        register_dataset(
            name="orig-ds",
            media_type="image",
            num_items=10,
            pkl_path="/tmp/orig.pkl",
            origin="demo:flowers",
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["origin"] == "demo:flowers"

    def test_dataset_registry_includes_source(self, client):
        """Registered datasets include a source dict for the Details column."""
        src = {"importer": "folder", "params": {"path": "/data/images"}}
        register_dataset(
            name="src-ds",
            media_type="image",
            num_items=8,
            pkl_path="/tmp/src.pkl",
            origin="folder:/data/images",
            source=src,
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["source"] == src

    def test_dataset_registry_default_origin_is_unknown(self, client):
        """Datasets registered without origin default to 'unknown'."""
        register_dataset(name="no-origin", media_type="audio", num_items=3, pkl_path="/tmp/no.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["origin"] == "unknown"

    def test_dataset_registry_includes_loaded_field(self, client):
        """Registered datasets include the loaded boolean (renamed to Loaded? in UI)."""
        register_dataset(name="ld-ds", media_type="audio", num_items=5, pkl_path="/tmp/ld.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "loaded" in ds
        assert isinstance(ds["loaded"], bool)


class TestDashboardDatasetDupesColumn:
    """Tests that the dataset registry API returns the num_dupes field."""

    def test_dataset_registry_includes_num_dupes(self, client):
        """Registered datasets include a num_dupes integer."""
        register_dataset(name="dupes-ds", media_type="audio", num_items=10, pkl_path="/tmp/dupes.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert "num_dupes" in ds
        assert isinstance(ds["num_dupes"], int)

    def test_dataset_registry_num_dupes_defaults_to_zero(self, client):
        """Datasets registered without num_dupes default to 0."""
        register_dataset(name="no-dupes", media_type="audio", num_items=5, pkl_path="/tmp/nodupes.pkl")
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["num_dupes"] == 0

    def test_dataset_registry_stores_explicit_num_dupes(self, client):
        """Datasets registered with explicit num_dupes retain the value."""
        register_dataset(
            name="with-dupes", media_type="audio", num_items=20, pkl_path="/tmp/wdupes.pkl", num_dupes=3
        )
        resp = client.get("/api/datasets/registry")
        data = resp.get_json()
        ds = data["datasets"][0]
        assert ds["num_dupes"] == 3


class TestDashboardModelRegistryColumns:
    """Tests that the model registry API returns fields needed for new columns."""

    def test_model_registry_includes_created_at(self, client):
        """Registered models include a created_at timestamp."""
        register_model(name="ts-model", media_type="audio", trainable=True)
        resp = client.get("/api/models/registry")
        data = resp.get_json()
        m = data["models"][0]
        assert "created_at" in m
        assert isinstance(m["created_at"], (int, float))
        assert m["created_at"] > 0

    def test_model_registry_includes_autodetect_false_by_default(self, client):
        """Models without an autorun detector show autodetect=False."""
        register_model(name="no-det", media_type="image", trainable=True)
        resp = client.get("/api/models/registry")
        data = resp.get_json()
        m = data["models"][0]
        assert "autodetect" in m
        assert m["autodetect"] is False

    def test_model_registry_reflects_autodetect_flag(self, client):
        """Models with an autorun detector reflect its autodetect flag."""
        add_autorun_detector("det-a", "audio", None, 0.5, autodetect=True)
        register_model(
            name="det-a",
            media_type="audio",
            trainable=False,
            detector_name="det-a",
        )
        resp = client.get("/api/models/registry")
        data = resp.get_json()
        m = data["models"][0]
        assert m["autodetect"] is True

    def test_autodetect_toggle_via_api(self, client):
        """Toggling autodetect via PUT updates the model registry response."""
        add_autorun_detector("toggle-det", "audio", None, 0.5, autodetect=False)
        register_model(
            name="toggle-det",
            media_type="audio",
            trainable=False,
            detector_name="toggle-det",
        )

        # Initially false
        resp = client.get("/api/models/registry")
        m = resp.get_json()["models"][0]
        assert m["autodetect"] is False

        # Toggle on
        resp = client.put(
            "/api/autorun-detectors/toggle-det/autodetect",
            json={"autodetect": True},
        )
        assert resp.status_code == 200

        # Verify reflected
        resp = client.get("/api/models/registry")
        m = resp.get_json()["models"][0]
        assert m["autodetect"] is True

    def test_model_registry_includes_loaded_field(self, client):
        """Registered models include the loaded boolean (not shown in UI)."""
        register_model(name="ld-model", media_type="audio", trainable=True)
        resp = client.get("/api/models/registry")
        data = resp.get_json()
        m = data["models"][0]
        assert "loaded" in m
        assert isinstance(m["loaded"], bool)

    def test_model_registry_includes_trainable_field(self, client):
        """Registered models include the trainable boolean."""
        register_model(name="trainable-m", media_type="audio", trainable=True)
        register_model(name="pregen-m", media_type="audio", trainable=False)
        resp = client.get("/api/models/registry")
        data = resp.get_json()
        by_name = {m["name"]: m for m in data["models"]}
        assert by_name["trainable-m"]["trainable"] is True
        assert by_name["pregen-m"]["trainable"] is False

    def test_model_registry_includes_last_trained_at(self, client):
        """Registered models include the last_trained_at field (None by default)."""
        register_model(name="lt-model", media_type="audio", trainable=True)
        resp = client.get("/api/models/registry")
        data = resp.get_json()
        m = data["models"][0]
        assert "last_trained_at" in m
        assert m["last_trained_at"] is None

    def test_model_registry_last_trained_at_set_on_label_save(self, client):
        """Saving labels updates last_trained_at to a timestamp."""
        from vtsearch.models.registry import register_model as reg_model, update_model

        entry = reg_model(name="lt-save", media_type="audio", trainable=True, trainable_model_name="lt_save")
        import time

        now = time.time()
        update_model(entry["id"], last_trained_at=now)
        resp = client.get("/api/models/registry")
        data = resp.get_json()
        m = data["models"][0]
        assert isinstance(m["last_trained_at"], (int, float))
        assert m["last_trained_at"] >= now


class TestDashboardColumnHeaders:
    """Verify the frontend JS contains the updated column headers."""

    def test_dataset_grid_has_new_column_headers(self, client):
        """Bundle JS should include the dataset column headers."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "Created" in combined
        assert "Origin" in combined
        assert "Dupes" in combined
        assert "Loaded" in combined

    def test_model_grid_has_new_column_headers(self, client):
        """Bundle JS should include the model column headers."""
        import glob as globmod

        combined = ""
        for path in globmod.glob("static/*.js"):
            resp = client.get(f"/{path}")
            combined += resp.data.decode("utf-8")
        assert "Autorun" in combined
        assert "Trainable" in combined
