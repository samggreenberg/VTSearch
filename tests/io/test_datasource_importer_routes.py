"""API tests for the datasource-importer routes (list / run / field options)."""

import io

import pytest

from vtscore.datasource_importers import get_datasource_importer
from vtscore.datasource_importers.base import DataSourceImporter, FetchedMediaItem
from vtscore.plugins import PluginField
from vtsearch.routes.media.server import SERVER_MEDIA_DIR


@pytest.fixture
def example_media_cleanup():
    """Track and remove files this test adds to ``example_media/``."""
    before = set(SERVER_MEDIA_DIR.glob("*")) if SERVER_MEDIA_DIR.exists() else set()
    yield
    if SERVER_MEDIA_DIR.exists():
        for f in set(SERVER_MEDIA_DIR.glob("*")) - before:
            f.unlink(missing_ok=True)


class TestDatasourceImportersList:
    def test_lists_builtins_with_fields_and_tabs(self, client):
        resp = client.get("/api/datasource-importers")
        assert resp.status_code == 200
        data = resp.get_json()
        by_name = {imp["name"]: imp for imp in data["importers"]}
        assert by_name["server_file"]["category"] == "server"
        assert [f["key"] for f in by_name["server_file"]["fields"]] == ["path"]
        assert by_name["url_download"]["category"] == "services"
        assert {t["id"] for t in data["tabs"]} >= {"services", "server", "demo"}

    def test_hidden_plugins_filtered(self, client, monkeypatch):
        monkeypatch.setattr(
            "vtsearch.settings.get_effective_hidden_plugins",
            lambda: {"datasource_importers": {"url_download"}},
        )
        resp = client.get("/api/datasource-importers")
        names = [imp["name"] for imp in resp.get_json()["importers"]]
        assert "url_download" not in names
        assert "server_file" in names


class TestDatasourceImportRun:
    def test_server_file_fetches_into_example_media(self, client, tmp_path, example_media_cleanup):
        src = tmp_path / "bark.wav"
        src.write_bytes(b"RIFFxxxxWAVE")

        resp = client.post("/api/datasource-import/server_file", json={"path": str(src)})
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["original_name"] == "bark.wav"
        assert data["filename"].endswith(".wav")

        saved = SERVER_MEDIA_DIR / data["filename"]
        assert saved.read_bytes() == b"RIFFxxxxWAVE"

    def test_unknown_importer_404_lists_available(self, client):
        resp = client.post("/api/datasource-import/nope", json={})
        assert resp.status_code == 404
        assert "server_file" in str(resp.get_json())

    def test_missing_required_field_rejected(self, client):
        resp = client.post("/api/datasource-import/server_file", json={})
        assert resp.status_code in (400, 422)

    def test_missing_file_maps_to_400(self, client, tmp_path):
        resp = client.post(
            "/api/datasource-import/server_file",
            json={"path": str(tmp_path / "missing.wav")},
        )
        assert resp.status_code == 400
        assert "File not found" in resp.get_json()["message"]

    def test_url_ssrf_rejected(self, client):
        resp = client.post(
            "/api/datasource-import/url_download",
            json={"url": "http://127.0.0.1/x.wav"},
        )
        assert resp.status_code == 400
        assert "private/internal" in resp.get_json()["message"]

    def test_upstream_error_maps_to_502(self, client, tmp_path, monkeypatch):
        def _boom(field_values):
            raise RuntimeError("upstream fell over")

        monkeypatch.setattr(get_datasource_importer("server_file"), "fetch", _boom)
        src = tmp_path / "a.wav"
        src.write_bytes(b"x")
        resp = client.post("/api/datasource-import/server_file", json={"path": str(src)})
        assert resp.status_code == 502
        assert "upstream fell over" in resp.get_json()["message"]

    def test_file_field_importer_accepts_multipart(self, client, monkeypatch, example_media_cleanup):
        class UploadThingDataSourceImporter(DataSourceImporter):
            """Test-only importer with a file field."""

            fields = [PluginField(key="file", label="File", field_type="file", required=True)]

            def fetch(self, field_values):
                upload = field_values["file"]
                return FetchedMediaItem(data=upload.read(), filename=upload.filename)

        fake = UploadThingDataSourceImporter()
        monkeypatch.setattr(
            "vtsearch.routes.media.datasource.get_datasource_importer",
            lambda name: fake if name == "upload_thing" else None,
        )

        resp = client.post(
            "/api/datasource-import/upload_thing",
            data={"file": (io.BytesIO(b"\x01\x02"), "pick.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["original_name"] == "pick.png"
        assert (SERVER_MEDIA_DIR / data["filename"]).read_bytes() == b"\x01\x02"


class TestDatasourceImportFieldOptions:
    def test_unknown_field_400(self, client):
        resp = client.post(
            "/api/datasource-import/server_file/options",
            json={"field_key": "nope", "values": {}},
        )
        assert resp.status_code == 400

    def test_non_dynamic_field_400(self, client):
        resp = client.post(
            "/api/datasource-import/server_file/options",
            json={"field_key": "path", "values": {}},
        )
        assert resp.status_code == 400
        assert "not dynamic" in resp.get_json()["message"]

    def test_unknown_importer_404(self, client):
        resp = client.post(
            "/api/datasource-import/nope/options",
            json={"field_key": "x", "values": {}},
        )
        assert resp.status_code == 404

    def test_unimplemented_dynamic_options_501(self, client, monkeypatch):
        imp = get_datasource_importer("server_file")
        monkeypatch.setattr(imp.fields[0], "dynamic_options", True)
        resp = client.post(
            "/api/datasource-import/server_file/options",
            json={"field_key": "path", "values": {}},
        )
        assert resp.status_code == 501

    def test_options_normalised_to_value_label(self, client, monkeypatch):
        imp = get_datasource_importer("server_file")
        monkeypatch.setattr(imp.fields[0], "dynamic_options", True)
        monkeypatch.setattr(imp, "get_field_options", lambda key, values: ["plain", ("id1", "Label 1")])
        resp = client.post(
            "/api/datasource-import/server_file/options",
            json={"field_key": "path", "values": {}},
        )
        assert resp.status_code == 200
        assert resp.get_json()["options"] == [
            {"value": "plain", "label": "plain"},
            {"value": "id1", "label": "Label 1"},
        ]

    def test_plugin_error_maps_to_502(self, client, monkeypatch):
        imp = get_datasource_importer("server_file")
        monkeypatch.setattr(imp.fields[0], "dynamic_options", True)

        def _boom(key, values):
            raise RuntimeError("service auth failed")

        monkeypatch.setattr(imp, "get_field_options", _boom)
        resp = client.post(
            "/api/datasource-import/server_file/options",
            json={"field_key": "path", "values": {}},
        )
        assert resp.status_code == 502
        assert "service auth failed" in resp.get_json()["message"]
