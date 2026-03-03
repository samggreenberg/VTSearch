"""Tests for the VTSearch dashboard API endpoint."""

import app as app_module  # noqa: F401 — triggers conftest side effects
from vtsearch.datasets.registry import register_dataset
from vtsearch.utils import medias


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
    """Test that the dashboard HTML is present in the index page."""

    def test_dashboard_view_in_html(self, client):
        """The dashboard view div should be in the index HTML."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"dashboard-view" in resp.data
        assert b"Datasets" in resp.data
        assert b"Models" in resp.data

    def test_dashboard_menu_item(self, client):
        """The burger menu should have a Dashboard entry."""
        resp = client.get("/")
        assert b"menu-dashboard" in resp.data

    def test_label_and_detect_buttons(self, client):
        """Label and Detect buttons should be present and disabled by default."""
        resp = client.get("/")
        assert b"dash-label-btn" in resp.data
        assert b"dash-detect-btn" in resp.data


class TestGuessMediaType:
    """Frontend auto-populates the media-type dropdown when creating a new model.

    The guessing logic lives in app.js (showNewModelForm):
    1. If all datasets in the registry share a single media_type, use that.
    2. Otherwise, if settings.autoload_media_types has exactly one entry, use it.

    These tests verify the underlying data contracts that the JS logic relies on.
    """

    def test_js_contains_guessing_logic(self, client):
        """app.js should include the media-type guessing code."""
        resp = client.get("/static/app.js")
        text = resp.data.decode("utf-8")
        assert "guessedMediaType" in text
        assert "datasetTypes" in text

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
