"""Tests for the Settings Import/Export abstraction.

Covers:
- SettingsImporter and SettingsExporter base classes
- Auto-discovery registries
- GET /api/settings-importers, GET /api/settings-exporters endpoints
- Built-in plugins: local_json_file, server_json_file (both import and export)
- POST /api/settings-importers/import/<name> endpoint
- POST /api/settings-exporters/export endpoint
"""

from __future__ import annotations

import io
import json

import pytest


# ---------------------------------------------------------------------------
# SettingsImporter base class
# ---------------------------------------------------------------------------


class TestSettingsImporterBase:
    def test_run_raises_not_implemented(self):
        from vtsearch.settings_io.importers.base import SettingsImporter

        imp = SettingsImporter()
        with pytest.raises(NotImplementedError):
            imp.run({})

    def test_to_dict_contains_standard_keys(self):
        from vtsearch.settings_io.importers.base import SettingsImporter

        class Minimal(SettingsImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "Minimal importer."
            fields = []

            def run(self, field_values):
                return {}

        d = Minimal().to_dict()
        assert d["name"] == "minimal"
        assert d["display_name"] == "Minimal"
        assert "icon" in d
        assert "fields" in d

    def test_default_icon(self):
        from vtsearch.settings_io.importers.base import SettingsImporter

        assert SettingsImporter.icon == "\u2699\ufe0f"


# ---------------------------------------------------------------------------
# SettingsExporter base class
# ---------------------------------------------------------------------------


class TestSettingsExporterBase:
    def test_export_raises_not_implemented(self):
        from vtsearch.settings_io.exporters.base import SettingsExporter

        exp = SettingsExporter()
        with pytest.raises(NotImplementedError):
            exp.export({}, {})

    def test_to_dict_contains_standard_keys(self):
        from vtsearch.settings_io.exporters.base import SettingsExporter

        class Minimal(SettingsExporter):
            name = "minimal"
            display_name = "Minimal"
            description = "Minimal exporter."
            fields = []

            def export(self, settings_data, field_values):
                return {"message": "ok"}

        d = Minimal().to_dict()
        assert d["name"] == "minimal"
        assert d["display_name"] == "Minimal"
        assert "icon" in d
        assert "fields" in d

    def test_default_icon(self):
        from vtsearch.settings_io.exporters.base import SettingsExporter

        assert SettingsExporter.icon == "\U0001f4e4"


# ---------------------------------------------------------------------------
# Importer registry
# ---------------------------------------------------------------------------


class TestSettingsImporterRegistry:
    def test_list_returns_builtins(self):
        from vtsearch.settings_io.importers import list_settings_importers

        names = {imp.name for imp in list_settings_importers()}
        assert "local_json_file" in names
        assert "server_json_file" in names

    def test_get_known(self):
        from vtsearch.settings_io.importers import get_settings_importer

        for name in ("local_json_file", "server_json_file"):
            imp = get_settings_importer(name)
            assert imp is not None, f"Settings importer '{name}' not found"
            assert imp.name == name

    def test_get_unknown_returns_none(self):
        from vtsearch.settings_io.importers import get_settings_importer

        assert get_settings_importer("no_such") is None

    def test_each_has_display_name_and_icon(self):
        from vtsearch.settings_io.importers import list_settings_importers

        for imp in list_settings_importers():
            assert imp.display_name, f"{imp.name} missing display_name"
            assert imp.icon, f"{imp.name} missing icon"
            assert imp.description, f"{imp.name} missing description"


# ---------------------------------------------------------------------------
# Exporter registry
# ---------------------------------------------------------------------------


class TestSettingsExporterRegistry:
    def test_list_returns_builtins(self):
        from vtsearch.settings_io.exporters import list_settings_exporters

        names = {exp.name for exp in list_settings_exporters()}
        assert "local_json_file" in names
        assert "server_json_file" in names

    def test_get_known(self):
        from vtsearch.settings_io.exporters import get_settings_exporter

        for name in ("local_json_file", "server_json_file"):
            exp = get_settings_exporter(name)
            assert exp is not None, f"Settings exporter '{name}' not found"
            assert exp.name == name

    def test_get_unknown_returns_none(self):
        from vtsearch.settings_io.exporters import get_settings_exporter

        assert get_settings_exporter("no_such") is None


# ---------------------------------------------------------------------------
# LocalFileSettingsImporter
# ---------------------------------------------------------------------------


class TestLocalFileSettingsImporter:
    def _get_importer(self):
        from vtsearch.settings_io.importers.local_json_file import SETTINGS_IMPORTER

        return SETTINGS_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "local_json_file"

    def test_has_file_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "file" in fields
        assert fields["file"].field_type == "file"

    def test_run_parses_uploaded_file(self):
        imp = self._get_importer()
        settings_data = {"theme": "light", "volume": 0.8}
        raw = json.dumps(settings_data).encode()
        file_storage = type("FS", (), {"read": lambda self: raw})()
        result = imp.run({"file": file_storage})
        assert result["theme"] == "light"
        assert result["volume"] == 0.8

    def test_run_raises_on_no_file(self):
        with pytest.raises(ValueError, match="No file"):
            self._get_importer().run({"file": None})

    def test_run_raises_on_empty_file(self):
        file_storage = type("FS", (), {"read": lambda self: b""})()
        with pytest.raises(ValueError, match="empty"):
            self._get_importer().run({"file": file_storage})

    def test_run_raises_on_invalid_json(self):
        file_storage = type("FS", (), {"read": lambda self: b"not json"})()
        with pytest.raises(ValueError, match="JSON"):
            self._get_importer().run({"file": file_storage})

    def test_run_raises_on_non_dict_json(self):
        file_storage = type("FS", (), {"read": lambda self: b"[1,2,3]"})()
        with pytest.raises(ValueError, match="dict"):
            self._get_importer().run({"file": file_storage})


# ---------------------------------------------------------------------------
# ServerFileSettingsImporter
# ---------------------------------------------------------------------------


class TestServerFileSettingsImporter:
    def _get_importer(self):
        from vtsearch.settings_io.importers.server_json_file import SETTINGS_IMPORTER

        return SETTINGS_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "server_json_file"

    def test_has_filepath_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "filepath" in fields
        assert fields["filepath"].field_type == "server_path"

    def test_run_reads_server_file(self, tmp_path):
        settings_data = {"theme": "dark", "volume": 0.5}
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(settings_data))
        result = self._get_importer().run({"filepath": str(p)})
        assert result["theme"] == "dark"
        assert result["volume"] == 0.5

    def test_run_raises_on_missing_file(self):
        with pytest.raises(ValueError, match="not found"):
            self._get_importer().run({"filepath": "/nonexistent/settings.json"})

    def test_run_raises_on_empty_filepath(self):
        with pytest.raises(ValueError, match="file path"):
            self._get_importer().run({"filepath": ""})

    def test_run_raises_on_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(ValueError, match="JSON"):
            self._get_importer().run({"filepath": str(p)})

    def test_run_raises_on_non_dict_json(self, tmp_path):
        p = tmp_path / "array.json"
        p.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="dict"):
            self._get_importer().run({"filepath": str(p)})


# ---------------------------------------------------------------------------
# LocalFileSettingsExporter
# ---------------------------------------------------------------------------


class TestLocalFileSettingsExporter:
    def _get_exporter(self):
        from vtsearch.settings_io.exporters.local_json_file import SETTINGS_EXPORTER

        return SETTINGS_EXPORTER

    def test_name(self):
        assert self._get_exporter().name == "local_json_file"

    def test_no_fields(self):
        assert self._get_exporter().fields == []

    def test_export_returns_download_data(self):
        settings_data = {"theme": "dark", "volume": 0.7}
        result = self._get_exporter().export(settings_data, {})
        assert result["download"] is True
        assert result["data"] == settings_data
        assert "message" in result
        assert "filename" in result


# ---------------------------------------------------------------------------
# ServerFileSettingsExporter
# ---------------------------------------------------------------------------


class TestServerFileSettingsExporter:
    def _get_exporter(self):
        from vtsearch.settings_io.exporters.server_json_file import SETTINGS_EXPORTER

        return SETTINGS_EXPORTER

    def test_name(self):
        assert self._get_exporter().name == "server_json_file"

    def test_has_filepath_field(self):
        fields = {f.key: f for f in self._get_exporter().fields}
        assert "filepath" in fields
        assert fields["filepath"].field_type == "server_path"

    def test_export_writes_file(self, tmp_path):
        settings_data = {"theme": "light", "volume": 0.3}
        dest = tmp_path / "exported.json"
        result = self._get_exporter().export(settings_data, {"filepath": str(dest)})
        assert "message" in result
        assert dest.exists()
        written = json.loads(dest.read_text())
        assert written["theme"] == "light"
        assert written["volume"] == 0.3

    def test_export_creates_parent_dirs(self, tmp_path):
        dest = tmp_path / "sub" / "dir" / "settings.json"
        self._get_exporter().export({"theme": "dark"}, {"filepath": str(dest)})
        assert dest.exists()

    def test_export_raises_on_empty_path(self):
        with pytest.raises(ValueError, match="file path"):
            self._get_exporter().export({}, {"filepath": ""})


# ---------------------------------------------------------------------------
# API – GET /api/settings-importers
# ---------------------------------------------------------------------------


class TestGetSettingsImportersEndpoint:
    def test_returns_200(self, client):
        res = client.get("/api/settings-importers")
        assert res.status_code == 200

    def test_returns_list(self, client):
        data = client.get("/api/settings-importers").get_json()
        assert isinstance(data, list)

    def test_contains_builtin_importers(self, client):
        names = {e["name"] for e in client.get("/api/settings-importers").get_json()}
        assert "local_json_file" in names
        assert "server_json_file" in names

    def test_each_entry_has_required_keys(self, client):
        for entry in client.get("/api/settings-importers").get_json():
            assert "name" in entry
            assert "display_name" in entry
            assert "fields" in entry


# ---------------------------------------------------------------------------
# API – GET /api/settings-exporters
# ---------------------------------------------------------------------------


class TestGetSettingsExportersEndpoint:
    def test_returns_200(self, client):
        res = client.get("/api/settings-exporters")
        assert res.status_code == 200

    def test_returns_list(self, client):
        data = client.get("/api/settings-exporters").get_json()
        assert isinstance(data, list)

    def test_contains_builtin_exporters(self, client):
        names = {e["name"] for e in client.get("/api/settings-exporters").get_json()}
        assert "local_json_file" in names
        assert "server_json_file" in names


# ---------------------------------------------------------------------------
# API – POST /api/settings-importers/import/<name>
# ---------------------------------------------------------------------------


class TestSettingsImportEndpoint:
    def test_server_json_import(self, client, tmp_path):
        settings_data = {"theme": "light", "volume": 0.6}
        p = tmp_path / "settings.json"
        p.write_text(json.dumps(settings_data))
        res = client.post(
            "/api/settings-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "theme" in data["keys"]

    def test_local_json_import(self, client):
        settings_data = {"theme": "dark", "volume": 0.9}
        raw = json.dumps(settings_data).encode()
        data = {"file": (io.BytesIO(raw), "settings.json")}
        res = client.post(
            "/api/settings-importers/import/local_json_file",
            data=data,
            content_type="multipart/form-data",
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["success"] is True

    def test_unknown_importer_404(self, client):
        res = client.post("/api/settings-importers/import/nonexistent", json={})
        assert res.status_code == 404

    def test_missing_filepath_422(self, client):
        # Schema-level validation: the per-plugin marshmallow schema
        # rejects the empty ``filepath`` with the standard 422 envelope
        # (the field is declared ``required=True`` on the
        # server_json_file importer).
        res = client.post(
            "/api/settings-importers/import/server_json_file",
            json={"filepath": ""},
        )
        assert res.status_code == 422
        assert "filepath" in res.get_json().get("errors", {}).get("json", {})


# ---------------------------------------------------------------------------
# API – POST /api/settings-exporters/export
# ---------------------------------------------------------------------------


class TestSettingsExportEndpoint:
    def test_local_json_export(self, client):
        res = client.post(
            "/api/settings-exporters/export",
            json={"exporter_name": "local_json_file", "field_values": {}},
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert data["download"] is True
        assert isinstance(data["data"], dict)

    def test_server_json_export(self, client, tmp_path):
        dest = tmp_path / "exported.json"
        res = client.post(
            "/api/settings-exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": str(dest)},
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert dest.exists()

    def test_unknown_exporter_404(self, client):
        res = client.post(
            "/api/settings-exporters/export",
            json={"exporter_name": "nonexistent"},
        )
        assert res.status_code == 404

    def test_missing_exporter_name_422(self, client):
        # Schema-level validation: marshmallow rejects the missing required
        # ``exporter_name`` with the standard 422 ``errors`` envelope.
        res = client.post("/api/settings-exporters/export", json={})
        assert res.status_code == 422
        assert "exporter_name" in res.get_json().get("errors", {}).get("json", {})

    def test_missing_required_field_400(self, client):
        # Handler-level validation: plugin field check uses ``message``.
        res = client.post(
            "/api/settings-exporters/export",
            json={"exporter_name": "server_json_file", "field_values": {}},
        )
        assert res.status_code == 400
        assert "Missing required field" in res.get_json().get("message", "")
