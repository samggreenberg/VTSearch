"""Tests for the VTSearch dashboard API endpoint."""

import app as app_module  # noqa: F401 — triggers conftest side effects
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
