"""Tests for the Labelset Exporter abstraction.

Covers:
- ExporterField and LabelsetExporter base classes
- Auto-discovery registry
- Built-in exporters: gui, server_json_file, server_csv_file, email_smtp
- Flask API routes: GET /api/exporters, POST /api/exporters/export
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

SAMPLE_RESULTS = {
    "media_type": "audio",
    "detectors_run": 2,
    "results": {
        "dog_bark": {
            "detector_name": "dog_bark",
            "threshold": 0.5,
            "total_hits": 3,
            "hits": [
                {"id": 1, "filename": "bark1.wav", "score": 0.9},
                {"id": 2, "filename": "bark2.wav", "score": 0.7},
                {"id": 3, "filename": "bark3.wav", "score": 0.6},
            ],
        },
        "cat_meow": {
            "detector_name": "cat_meow",
            "threshold": 0.6,
            "total_hits": 1,
            "hits": [
                {"id": 5, "filename": "meow.wav", "score": 0.8},
            ],
        },
    },
}

EMPTY_RESULTS = {
    "media_type": "audio",
    "detectors_run": 0,
    "results": {},
}


# ---------------------------------------------------------------------------
# ExporterField
# ---------------------------------------------------------------------------


class TestExporterField:
    def test_to_dict_contains_required_keys(self):
        from vtscore.exporters.base import ExporterField

        f = ExporterField(key="fp", label="File Path", field_type="text")
        d = f.to_dict()
        assert d["key"] == "fp"
        assert d["label"] == "File Path"
        assert d["field_type"] == "text"
        assert "description" in d
        assert "options" in d
        assert "default" in d
        assert "required" in d
        assert "placeholder" in d

    def test_defaults(self):
        from vtscore.exporters.base import ExporterField

        f = ExporterField(key="x", label="X", field_type="text")
        assert f.required is True
        assert f.default == ""
        assert f.placeholder == ""
        assert f.options == []
        assert f.description == ""

    def test_custom_values(self):
        from vtscore.exporters.base import ExporterField

        f = ExporterField(
            key="mode",
            label="Mode",
            field_type="select",
            options=["a", "b"],
            default="a",
            required=False,
            description="Choose mode",
            placeholder="Pick one",
        )
        d = f.to_dict()
        assert d["options"] == ["a", "b"]
        assert d["default"] == "a"
        assert d["required"] is False


# ---------------------------------------------------------------------------
# LabelsetExporter base class
# ---------------------------------------------------------------------------


class TestLabelsetExporterBase:
    def test_export_raises_not_implemented(self):
        from vtscore.exporters.base import LabelsetExporter

        exp = LabelsetExporter()
        with pytest.raises(NotImplementedError):
            exp.export({}, {})

    def test_to_dict_contains_standard_keys(self):
        from vtscore.exporters.base import ExporterField, LabelsetExporter

        class Dummy(LabelsetExporter):
            name = "dummy"
            display_name = "Dummy"
            description = "A test exporter."
            icon = "🧪"
            fields = [ExporterField(key="k", label="K", field_type="text")]

            def export(self, results, field_values):
                return {"message": "ok"}

        d = Dummy().to_dict()
        assert d["name"] == "dummy"
        assert d["display_name"] == "Dummy"
        assert d["description"] == "A test exporter."
        assert d["icon"] == "🧪"
        assert len(d["fields"]) == 1
        assert d["fields"][0]["key"] == "k"


# ---------------------------------------------------------------------------
# Registry (auto-discovery)
# ---------------------------------------------------------------------------


class TestExporterRegistry:
    def test_list_exporters_returns_all_builtins(self):
        from vtscore.exporters import list_exporters

        names = {e.name for e in list_exporters()}
        assert "gui" in names
        assert "server_json_file" in names
        assert "server_csv_file" in names
        assert "email_smtp" in names

    def test_get_exporter_known(self):
        from vtscore.exporters import get_exporter

        for name in ("gui", "server_json_file", "server_csv_file", "email_smtp"):
            exp = get_exporter(name)
            assert exp is not None, f"Exporter '{name}' not found"
            assert exp.name == name

    def test_get_exporter_unknown_returns_none(self):
        from vtscore.exporters import get_exporter

        assert get_exporter("no_such_exporter") is None

    def test_each_exporter_has_display_name_and_icon(self):
        from vtscore.exporters import list_exporters

        for exp in list_exporters():
            assert exp.display_name, f"{exp.name} missing display_name"
            assert exp.icon, f"{exp.name} missing icon"
            assert exp.description, f"{exp.name} missing description"

    def test_each_exporter_fields_are_valid(self):
        from vtscore.exporters import list_exporters

        for exp in list_exporters():
            for f in exp.fields:
                assert f.key, f"{exp.name} has a field without a key"
                assert f.label, f"{exp.name} field '{f.key}' has no label"
                assert f.field_type in (
                    "text",
                    "password",
                    "email",
                    "file",
                    "folder",
                    "select",
                    "server_path",
                    "url",
                    "number",
                ), f"{exp.name} field '{f.key}' has unknown type '{f.field_type}'"


# ---------------------------------------------------------------------------
# GUI exporter
# ---------------------------------------------------------------------------


class TestDisplayLabelsetExporter:
    def test_has_no_fields(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("gui")
        assert exp.fields == []

    def test_export_returns_message_and_display_results(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("gui")
        result = exp.export(SAMPLE_RESULTS, {})
        assert "message" in result
        assert "display_results" in result
        assert result["display_results"] is SAMPLE_RESULTS

    def test_export_counts_hits_in_message(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("gui")
        result = exp.export(SAMPLE_RESULTS, {})
        # 3 + 1 = 4 total hits
        assert "4" in result["message"]
        assert "2" in result["message"]  # 2 detectors

    def test_export_empty_results(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("gui")
        result = exp.export(EMPTY_RESULTS, {})
        assert "message" in result
        assert result["display_results"] is EMPTY_RESULTS

    def test_export_cli_prints_origins_and_names(self, capsys):
        from vtscore.exporters import get_exporter

        results_with_origin = {
            "media_type": "audio",
            "detectors_run": 1,
            "results": {
                "det1": {
                    "detector_name": "det1",
                    "threshold": 0.5,
                    "total_hits": 2,
                    "hits": [
                        {
                            "id": 1,
                            "filename": "bark1.wav",
                            "origin_name": "bark1.wav",
                            "origin": {"importer": "server_folder", "params": {"path": "/data"}},
                            "score": 0.9,
                            "category": "dog",
                        },
                        {
                            "id": 2,
                            "filename": "bark2.wav",
                            "origin_name": "bark2.wav",
                            "score": 0.7,
                            "category": "dog",
                        },
                    ],
                },
            },
        }
        exp = get_exporter("gui")
        result = exp.export_cli(results_with_origin, {})
        captured = capsys.readouterr()
        assert "message" in result
        # Should list origin and name, not scores or categories
        assert "folder(/data)" in captured.out
        assert "bark1.wav" in captured.out
        assert "bark2.wav" in captured.out
        assert "score" not in captured.out.lower()
        assert "category" not in captured.out.lower()
        assert "dog" not in captured.out

    def test_export_cli_no_hits(self, capsys):
        from vtscore.exporters import get_exporter

        exp = get_exporter("gui")
        result = exp.export_cli(EMPTY_RESULTS, {})
        captured = capsys.readouterr()
        assert "No items predicted as Good" in captured.out
        assert "message" in result

    def test_export_converts_labelset_to_display_format(self):
        """When results come from /api/labels/export (LabelSet format),
        the GUI exporter should convert them to the display format."""
        from vtscore.exporters import get_exporter

        labelset_data = {
            "labels": [
                {"md5": "aaa", "label": "good", "origin_name": "file1.wav", "filename": "file1.wav"},
                {"md5": "bbb", "label": "bad", "origin_name": "file2.wav", "filename": "file2.wav"},
                {"md5": "ccc", "label": "good", "origin_name": "file3.wav", "filename": "file3.wav"},
            ]
        }
        exp = get_exporter("gui")
        result = exp.export(labelset_data, {})
        assert "display_results" in result
        dr = result["display_results"]
        # Should have the autodetect-results structure
        assert "results" in dr
        assert "media_type" in dr
        assert "detectors_run" in dr
        # All 3 labels should appear as hits
        hits = dr["results"]["labels"]["hits"]
        assert len(hits) == 3
        # Good labels come first
        assert hits[0]["label"] == "good"
        assert hits[1]["label"] == "good"
        assert hits[2]["label"] == "bad"
        assert "3" in result["message"]


# ---------------------------------------------------------------------------
# Server JSON file exporter
# ---------------------------------------------------------------------------


class TestServerJsonLabelsetExporter:
    def test_has_filepath_field(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_json_file")
        keys = [f.key for f in exp.fields]
        assert "filepath" in keys

    def test_export_writes_json(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_json_file")
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "results.json"
            result = exp.export(SAMPLE_RESULTS, {"filepath": str(fpath)})
            assert "message" in result
            assert fpath.exists()
            written = json.loads(fpath.read_text())
            assert written["media_type"] == "audio"
            assert written["detectors_run"] == 2

    def test_export_creates_parent_dirs(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_json_file")
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "sub" / "dir" / "results.json"
            exp.export(SAMPLE_RESULTS, {"filepath": str(fpath)})
            assert fpath.exists()

    def test_export_message_contains_hit_count(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("server_json_file")
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "out.json"
            result = exp.export(SAMPLE_RESULTS, {"filepath": str(fpath)})
            assert "4" in result["message"]  # 3 + 1 hits

    def test_to_dict_has_all_keys(self):
        from vtscore.exporters import get_exporter

        d = get_exporter("server_json_file").to_dict()
        assert d["name"] == "server_json_file"
        assert "fields" in d
        assert len(d["fields"]) >= 1


# ---------------------------------------------------------------------------
# Email SMTP exporter
# ---------------------------------------------------------------------------


class TestEmailLabelsetExporter:
    def test_has_required_fields(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        keys = {f.key for f in exp.fields}
        assert "to" in keys

    def test_fields_are_from_and_to(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        keys = {f.key for f in exp.fields}
        assert keys == {"from", "to"}

    def test_export_raises_on_missing_to(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        with pytest.raises(ValueError, match="Recipient"):
            exp.export(SAMPLE_RESULTS, {"from": "me@example.com", "to": ""})

    def test_export_raises_on_missing_from(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        with pytest.raises(ValueError, match="Sender"):
            exp.export(SAMPLE_RESULTS, {"from": "", "to": "you@example.com"})

    def test_export_calls_smtp_via_mx(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")

        mock_server = MagicMock()
        mock_smtp_cls = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("vtscore.exporters.email_smtp._resolve_mx", return_value="mx.example.com"),
            patch("vtscore.exporters.email_smtp.smtplib.SMTP", mock_smtp_cls),
        ):
            result = exp.export(
                SAMPLE_RESULTS,
                {"from": "me@my-domain.example", "to": "you@example.com"},
            )

        mock_smtp_cls.assert_called_once_with("mx.example.com", 25, timeout=30)
        mock_server.sendmail.assert_called_once()
        sender, recipients, _ = mock_server.sendmail.call_args.args
        assert sender == "me@my-domain.example"
        assert recipients == ["you@example.com"]
        assert "message" in result
        assert "you@example.com" in result["message"]

    def test_plain_text_builder(self):
        from vtscore.exporters.email_smtp import _build_plain_text

        text = _build_plain_text(SAMPLE_RESULTS)
        assert "Auto-Detect Results" in text
        assert "dog_bark" in text
        assert "cat_meow" in text
        assert "bark1.wav" in text

    def test_html_builder(self):
        from vtscore.exporters.email_smtp import _build_html

        html = _build_html(SAMPLE_RESULTS)
        assert "<html>" in html
        assert "dog_bark" in html
        assert "cat_meow" in html
        assert "bark1.wav" in html


# ---------------------------------------------------------------------------
# API – GET /api/exporters
# ---------------------------------------------------------------------------


class TestGetExportersEndpoint:
    def test_returns_200(self, client):
        res = client.get("/api/exporters")
        assert res.status_code == 200

    def test_returns_list(self, client):
        res = client.get("/api/exporters")
        data = res.get_json()
        assert isinstance(data, list)

    def test_contains_builtin_exporters(self, client):
        res = client.get("/api/exporters")
        names = {e["name"] for e in res.get_json()}
        assert "gui" in names
        assert "server_json_file" in names
        assert "server_csv_file" in names
        assert "email_smtp" in names

    def test_each_entry_has_required_keys(self, client):
        res = client.get("/api/exporters")
        for entry in res.get_json():
            assert "name" in entry
            assert "display_name" in entry
            assert "description" in entry
            assert "icon" in entry
            assert "fields" in entry


# ---------------------------------------------------------------------------
# API – POST /api/exporters/export
# ---------------------------------------------------------------------------


class TestExportEndpoint:
    def test_missing_exporter_name_returns_422(self, client):
        # Schema-level validation (required ``exporter_name``) → 422.
        res = client.post(
            "/api/exporters/export",
            json={"results": SAMPLE_RESULTS},
        )
        assert res.status_code == 422
        assert "exporter_name" in str(res.get_json()["errors"])

    def test_unknown_exporter_returns_404(self, client):
        res = client.post(
            "/api/exporters/export",
            json={"exporter_name": "unicorn", "results": SAMPLE_RESULTS},
        )
        assert res.status_code == 404
        # The app-level ``NotFound`` errorhandler reformats 404s to
        # ``{"error": "Not Found", ...}`` regardless of the
        # ``message=`` passed to ``abort()``.
        assert "error" in res.get_json()

    def test_gui_exporter_returns_success(self, client):
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "gui",
                "field_values": {},
                "results": SAMPLE_RESULTS,
            },
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "message" in data
        assert "display_results" in data

    def test_server_json_exporter_creates_file(self, client):
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = Path(tmpdir) / "export.json"
            res = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": str(fpath)},
                    "results": SAMPLE_RESULTS,
                },
            )
            assert res.status_code == 200
            data = res.get_json()
            assert data["success"] is True
            assert fpath.exists()
            written = json.loads(fpath.read_text())
            assert written["detectors_run"] == 2

    def test_server_json_exporter_missing_filepath_returns_422(self, client):
        # Phase B: empty required fields are rejected by the per-plugin
        # marshmallow schema (422 with the standard ``errors`` envelope)
        # before ``.export()`` is called.
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": ""},
                "results": SAMPLE_RESULTS,
            },
        )
        assert res.status_code == 422

    def test_server_json_exporter_missing_field_uses_default(self, client):
        """Phase B: the route falls back to the field's declared default.

        ``server_json_file`` declares a ``{YYYYMMDD-HHMMSS}``-stamped
        default for ``filepath``, so an export with no ``filepath`` at
        all proceeds with that default - same behaviour as if the
        frontend had submitted the pre-filled default verbatim.
        """
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {},  # 'filepath' omitted; load_default kicks in
                "results": SAMPLE_RESULTS,
            },
        )
        assert res.status_code == 200

    def test_email_exporter_sends_via_mx(self, client):
        mock_server = MagicMock()
        mock_smtp_cls = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with (
            patch("vtscore.exporters.email_smtp._resolve_mx", return_value="mx.example.com"),
            patch("vtscore.exporters.email_smtp.smtplib.SMTP", mock_smtp_cls),
        ):
            res = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "email_smtp",
                    "field_values": {"from": "me@my-domain.example", "to": "you@example.com"},
                    "results": SAMPLE_RESULTS,
                },
            )
        assert res.status_code == 200
        data = res.get_json()
        assert data["success"] is True
        assert "you@example.com" in data["message"]
        mock_server.sendmail.assert_called_once()

    def test_export_with_empty_results_dict(self, client):
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "gui",
                "field_values": {},
                "results": {},
            },
        )
        assert res.status_code == 200
        assert res.get_json()["success"] is True

    def test_export_with_no_results_key(self, client):
        """results defaults to {} when omitted."""
        res = client.post(
            "/api/exporters/export",
            json={"exporter_name": "gui"},
        )
        assert res.status_code == 200

    def test_non_json_body_treated_as_empty(self, client):
        res = client.post(
            "/api/exporters/export",
            data="not json",
            content_type="text/plain",
        )
        # flask-smorest's schema-level rejection of unparseable / empty
        # bodies surfaces as 422 (``exporter_name`` required).
        assert res.status_code == 422

    def test_path_traversal_absolute_rejected(self, client):
        """Absolute paths outside the allowed directory must be rejected."""
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": "/etc/passwd"},
                "results": SAMPLE_RESULTS,
            },
        )
        assert res.status_code == 400
        msg = res.get_json()["message"].lower()
        assert "outside" in msg or "must be within" in msg

    def test_path_traversal_relative_rejected(self, client):
        """Relative paths that escape the base directory must be rejected."""
        res = client.post(
            "/api/exporters/export",
            json={
                "exporter_name": "server_json_file",
                "field_values": {"filepath": "../../../etc/shadow"},
                "results": SAMPLE_RESULTS,
            },
        )
        assert res.status_code == 400

    def test_export_oserror_returns_500_and_logs_traceback(self, client, caplog):
        """An OSError from an exporter should return 500 and log the full traceback."""
        import logging

        with caplog.at_level(logging.ERROR, logger="vtsearch.routes.labels.exporters"):
            with patch(
                "vtscore.exporters.server_json_file.ServerJsonLabelsetExporter.export",
                side_effect=OSError("No space left on device"),
            ):
                res = client.post(
                    "/api/exporters/export",
                    json={
                        "exporter_name": "server_json_file",
                        "field_values": {"filepath": "data/test_output.json"},
                        "results": SAMPLE_RESULTS,
                    },
                )
                assert res.status_code == 500
                assert "No space left on device" in res.get_json()["message"]

        # The traceback should be logged server-side
        assert any("No space left on device" in r.message for r in caplog.records)
        assert any(r.exc_info for r in caplog.records if "No space left" in r.message)

    def test_export_permission_error_returns_500_and_logs_traceback(self, client, caplog):
        """A PermissionError from an exporter should return 500 and log the full traceback."""
        import logging

        with caplog.at_level(logging.ERROR, logger="vtsearch.routes.labels.exporters"):
            with patch(
                "vtscore.exporters.server_json_file.ServerJsonLabelsetExporter.export",
                side_effect=PermissionError("Permission denied"),
            ):
                res = client.post(
                    "/api/exporters/export",
                    json={
                        "exporter_name": "server_json_file",
                        "field_values": {"filepath": "data/test_output.json"},
                        "results": SAMPLE_RESULTS,
                    },
                )
                assert res.status_code == 500
                assert "Permission denied" in res.get_json()["message"]

        assert any("Permission denied" in r.message for r in caplog.records)
        assert any(r.exc_info for r in caplog.records if "Permission denied" in r.message)
