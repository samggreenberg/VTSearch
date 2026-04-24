"""Detector export tests.

Covers server-side detector export and labelset export.
"""

import json

import pytest

import app as app_module


class TestServerDetectorExport:
    """Tests for the ServerFileProcessorExporter endpoints."""

    @pytest.fixture(autouse=True)
    def _setup_server_dir(self, tmp_path):
        """Point detectors_dir at a temp directory for each test."""
        from vtsearch import settings

        det_dir = tmp_path / "detectors"
        settings.set_detectors_dir(str(det_dir))
        self._det_dir = det_dir
        yield

    def _vote(self):
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19, 20]})

    # -- basic success --

    def test_export_server_creates_file(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={"name": "my_detector"})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["success"] is True
        assert data["name"] == "my_detector"
        assert (self._det_dir / "my_detector.json").exists()

    def test_export_server_file_contains_valid_json(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "valid_json"})
        content = json.loads((self._det_dir / "valid_json.json").read_text())
        # New format: origins instead of weights
        assert "good_origins" in content
        assert "bad_origins" in content
        assert "inclusion" in content
        assert "media_type" in content
        assert "name" in content
        # Weights must NOT be serialised to disk
        assert "weights" not in content

    def test_export_server_returns_name(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={"name": "path_check"})
        data = resp.get_json()
        assert data["success"] is True
        assert data["name"] == "path_check"

    # -- name validation --

    def test_export_server_missing_name_returns_400(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={})
        assert resp.status_code == 400
        assert "name" in resp.get_json()["error"].lower()

    def test_export_server_empty_name_returns_400(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={"name": "  "})
        assert resp.status_code == 400

    # -- vote validation --

    def test_export_server_no_votes_returns_400(self, client):
        resp = client.post("/api/detector/export-server", json={"name": "no_votes"})
        assert resp.status_code == 400
        assert "vote" in resp.get_json()["error"].lower()

    # -- overwrite logic --

    def test_export_server_returns_409_when_exists(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "dup"})
        # Second export without overwrite
        resp = client.post("/api/detector/export-server", json={"name": "dup"})
        assert resp.status_code == 409
        data = resp.get_json()
        assert data["exists"] is True
        assert "dup" in data["name"]

    def test_export_server_overwrite_succeeds(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "overwrite_me"})
        resp = client.post(
            "/api/detector/export-server",
            json={"name": "overwrite_me", "overwrite": True},
        )
        assert resp.status_code == 200
        assert resp.get_json()["success"] is True

    # -- list server files --

    def test_list_server_files_empty(self, client):
        resp = client.get("/api/detector/server-files")
        assert resp.status_code == 200
        assert resp.get_json()["files"] == []

    def test_list_server_files_after_export(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "listed"})
        resp = client.get("/api/detector/server-files")
        data = resp.get_json()
        names = [f["name"] for f in data["files"]]
        assert "listed" in names

    def test_list_server_files_has_filename_and_size(self, client):
        self._vote()
        client.post("/api/detector/export-server", json={"name": "sized"})
        resp = client.get("/api/detector/server-files")
        entry = resp.get_json()["files"][0]
        assert "filename" in entry
        assert "size_bytes" in entry
        assert entry["size_bytes"] > 0

    # -- name sanitisation --

    def test_export_server_sanitises_name(self, client):
        self._vote()
        resp = client.post("/api/detector/export-server", json={"name": "a/b\\c:d"})
        assert resp.status_code == 200
        data = resp.get_json()
        # Special characters should be stripped
        assert "/" not in data["name"]
        assert "\\" not in data["name"]
        assert ":" not in data["name"]


class TestDetectorLabelsetExport:
    """Test the labelset export flow from the detector export context.

    The frontend Export Detector modal offers labelset exporters alongside
    the traditional detector weight export.  The flow is:
      1. GET /api/exporters  (list available exporters)
      2. GET /api/labels/export  (build labelset from current votes)
      3. POST /api/exporters/export  (run the chosen exporter on the labelset)
    """

    def test_exporters_list_available(self, client):
        """GET /api/exporters returns labelset exporters usable from the detector modal."""
        resp = client.get("/api/exporters")
        assert resp.status_code == 200
        names = {e["name"] for e in resp.get_json()}
        # At least the server JSON exporter should be present
        assert "server_json_file" in names

    def test_labelset_export_server_json_full(self, client, tmp_path):
        """Full flow: votes -> labelset -> server_json_file exporter -> file on disk."""
        app_module.good_votes.update({k: None for k in [1, 2, 3]})
        app_module.bad_votes.update({k: None for k in [18, 19]})

        # Step 1: get the labelset
        labels_resp = client.get("/api/labels/export")
        assert labels_resp.status_code == 200
        labels_data = labels_resp.get_json()
        assert "labels" in labels_data
        assert len(labels_data["labels"]) == 5

        # Step 2: export through server_json_file exporter
        fpath = tmp_path / "exported_labels.json"
        export_resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(fpath)},
                "results": labels_data,
            },
        )
        assert export_resp.status_code == 200
        data = export_resp.get_json()
        assert data["success"] is True

        # The file should contain valid JSON with the labels
        downloaded = json.loads(fpath.read_text())
        assert "labels" in downloaded
        assert len(downloaded["labels"]) == 5

    def test_labelset_export_server_json(self, client, tmp_path):
        """Full flow: votes -> labelset -> server_json_file exporter -> file on disk."""
        app_module.good_votes.update({k: None for k in [1, 2]})
        app_module.bad_votes.update({k: None for k in [3]})

        labels_resp = client.get("/api/labels/export")
        labels_data = labels_resp.get_json()

        fpath = tmp_path / "detector_labels.json"
        export_resp = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(fpath)},
                "results": labels_data,
            },
        )
        assert export_resp.status_code == 200
        assert export_resp.get_json()["success"] is True
        assert fpath.exists()

        written = json.loads(fpath.read_text())
        assert "labels" in written
        assert len(written["labels"]) == 3

    def test_labelset_roundtrip_via_exporter(self, client, tmp_path):
        """Labels exported through an exporter can be re-imported to restore votes."""
        app_module.good_votes.update({k: None for k in [1, 3]})
        app_module.bad_votes.update({k: None for k in [2]})

        # Export labelset
        labels_resp = client.get("/api/labels/export")
        labels_data = labels_resp.get_json()

        # Save via server_json_file exporter
        fpath = tmp_path / "labels_roundtrip.json"
        client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(fpath)},
                "results": labels_data,
            },
        )

        # Clear votes and re-import
        app_module.good_votes.clear()
        app_module.bad_votes.clear()

        saved = json.loads(fpath.read_text())
        resp = client.post("/api/labels/import", json=saved)
        data = resp.get_json()
        assert data["applied"] == 3
        assert set(app_module.good_votes) == {1, 3}
        assert set(app_module.bad_votes) == {2}
