"""Tests for the Labelset Exporter abstraction.

Covers:
- PluginField and LabelsetExporter base classes
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


def _multi_user_provider(user_dir: Path):
    """A LoginProvider that confines every request to *user_dir*.

    Path confinement only applies in multi-user mode; single-user / no-auth
    mode is unrestricted. These tests force multi-user mode to exercise the
    boundary.
    """
    from vtsearch.auth import LoginProvider

    class _Provider(LoginProvider):
        name = "test_multi_exporter"

        def get_user(self, request):
            return "testuser"

        def is_authenticated(self, request):
            return True

        def get_user_data_dir(self, username, base_data_dir):
            return user_dir

    return _Provider()


# ---------------------------------------------------------------------------
# PluginField
# ---------------------------------------------------------------------------


class TestExporterField:
    def test_to_dict_contains_required_keys(self):
        from vtscore.exporters.base import PluginField

        f = PluginField(key="fp", label="File Path", field_type="text")
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
        from vtscore.exporters.base import PluginField

        f = PluginField(key="x", label="X", field_type="text")
        assert f.required is True
        assert f.default == ""
        assert f.placeholder == ""
        assert f.options == []
        assert f.description == ""

    def test_custom_values(self):
        from vtscore.exporters.base import PluginField

        f = PluginField(
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
        from vtscore.exporters.base import PluginField, LabelsetExporter

        class Dummy(LabelsetExporter):
            name = "dummy"
            display_name = "Dummy"
            description = "A test exporter."
            icon = "🧪"
            fields = [PluginField(key="k", label="K", field_type="text")]

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


class TestGuiOriginFormatting:
    """The gui exporter's ``_format_origin`` and its use in the CLI/streaming
    output paths (origin display string preceding the name)."""

    def test_format_origin_none_is_empty(self):
        from vtscore.exporters.gui import _format_origin

        assert _format_origin({"filename": "x.wav"}) == ""  # no "origin" key
        assert _format_origin({"origin": None}) == ""

    def test_format_origin_renders_display_string(self):
        from vtscore.exporters.gui import _format_origin

        hit = {"origin": {"importer": "http_archive", "params": {"url": "https://ex.com/d.zip"}}}
        assert _format_origin(hit) == "http_archive(https://ex.com/d.zip)"

    def test_format_origin_importerless_falls_back_to_str(self):
        from vtscore.exporters.gui import _format_origin

        # A malformed origin dict (no "importer") makes Origin.from_dict raise;
        # the helper degrades to ``str(origin)`` rather than crashing the export.
        hit = {"origin": {"params": {"path": "/data"}}}
        out = _format_origin(hit)
        assert out == str({"params": {"path": "/data"}})

    def test_export_cli_name_falls_back_to_filename(self, capsys):
        from vtscore.exporters import get_exporter

        results = {
            "detectors_run": 1,
            "results": {
                "det1": {
                    "total_hits": 1,
                    "hits": [{"id": 1, "filename": "only_filename.wav", "score": 0.9}],
                },
            },
        }
        get_exporter("gui").export_cli(results, {})
        captured = capsys.readouterr()
        # No origin_name and no origin → prints just the bare filename.
        assert "only_filename.wav" in captured.out

    def test_export_cli_streaming_prints_origin_before_name(self, capsys):
        from vtscore.exporters import get_exporter

        header = {"detectors": ["det1"]}

        def records():
            yield (
                "det1",
                {
                    "id": 1,
                    "filename": "clip.mp4",
                    "origin_name": "clip.mp4",
                    "origin": {"importer": "server_folder", "params": {"path": "/vids"}},
                    "label": "good",
                },
            )
            # A bad hit is skipped in the streaming (predicted-Good) output.
            yield "det1", {"id": 2, "filename": "skip.mp4", "label": "bad"}

        res = get_exporter("gui").export_cli_streaming(header, records(), {})
        captured = capsys.readouterr()
        assert "folder(/vids)" in captured.out
        assert "clip.mp4" in captured.out
        assert "skip.mp4" not in captured.out
        assert "1 hit(s)" in res["message"]
        assert "1 detector(s)" in res["message"]

    def test_export_cli_streaming_no_good_hits(self, capsys):
        from vtscore.exporters import get_exporter

        def records():
            yield "det1", {"id": 1, "filename": "b.mp4", "label": "bad"}

        res = get_exporter("gui").export_cli_streaming({"detectors": ["det1"]}, records(), {})
        captured = capsys.readouterr()
        assert "No items predicted as Good" in captured.out
        assert "0 hit(s)" in res["message"]


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

    def test_fields_are_from_to_subject_and_batch_size(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        keys = {f.key for f in exp.fields}
        assert keys == {"from", "to", "subject", "batch_size"}

    def test_subject_field_is_optional_and_templated(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        subject = next(f for f in exp.fields if f.key == "subject")
        assert subject.required is False
        assert "YYYYMMDD" in subject.template_vars

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

    @pytest.mark.parametrize(
        "bad_addr",
        ["@example.com", "you@", "you@localhost", "you example@x.com", "noatsign", "you@@x.com"],
    )
    def test_export_rejects_malformed_recipient(self, bad_addr):
        """Addresses the bare ``"@" in`` check let through must now be rejected
        before any MX lookup (M34)."""
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        with pytest.raises(ValueError, match="Recipient"):
            exp.export(SAMPLE_RESULTS, {"from": "me@example.com", "to": bad_addr})

    @pytest.mark.parametrize("bad_addr", ["@example.com", "me@", "me@localhost", "no domain@"])
    def test_export_rejects_malformed_sender(self, bad_addr):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        with pytest.raises(ValueError, match="Sender"):
            exp.export(SAMPLE_RESULTS, {"from": bad_addr, "to": "you@example.com"})

    def test_is_valid_email_accepts_normal_addresses(self):
        from vtscore.exporters.email_smtp import _is_valid_email

        assert _is_valid_email("you@example.com")
        assert _is_valid_email("first.last+tag@sub.example.co.uk")
        assert not _is_valid_email("@example.com")
        assert not _is_valid_email("foo@bar")
        assert not _is_valid_email("")

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

    def _mock_smtp(self):
        mock_server = MagicMock()
        mock_smtp_cls = MagicMock()
        mock_smtp_cls.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)
        return mock_server, mock_smtp_cls

    def _sent_subject(self, mock_server):
        _, _, raw_msg = mock_server.sendmail.call_args.args
        import email

        return email.message_from_string(raw_msg)["Subject"]

    def test_custom_subject_is_used(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        mock_server, mock_smtp_cls = self._mock_smtp()

        with (
            patch("vtscore.exporters.email_smtp._resolve_mx", return_value="mx.example.com"),
            patch("vtscore.exporters.email_smtp.smtplib.SMTP", mock_smtp_cls),
        ):
            exp.export(
                SAMPLE_RESULTS,
                {"from": "me@my-domain.example", "to": "you@example.com", "subject": "Nightly run 2026-07-14"},
            )

        assert self._sent_subject(mock_server) == "Nightly run 2026-07-14"

    def test_blank_subject_falls_back_to_generated(self):
        from vtscore.exporters import get_exporter

        exp = get_exporter("email_smtp")
        mock_server, mock_smtp_cls = self._mock_smtp()

        with (
            patch("vtscore.exporters.email_smtp._resolve_mx", return_value="mx.example.com"),
            patch("vtscore.exporters.email_smtp.smtplib.SMTP", mock_smtp_cls),
        ):
            exp.export(
                SAMPLE_RESULTS,
                {"from": "me@my-domain.example", "to": "you@example.com", "subject": ""},
            )

        subject = self._sent_subject(mock_server)
        assert "Auto-Detect" in subject
        assert "audio" in subject

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
        all proceeds with that default; same behaviour as if the
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

    def test_email_exporter_substitutes_subject_template(self, client):
        """A {template} var in the subject is resolved at route ingress."""
        import re

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
                    "field_values": {
                        "from": "me@my-domain.example",
                        "to": "you@example.com",
                        "subject": "Run {YYYYMMDD}",
                    },
                    "results": SAMPLE_RESULTS,
                },
            )
        assert res.status_code == 200
        import email as _email

        _, _, raw_msg = mock_server.sendmail.call_args.args
        subject = _email.message_from_string(raw_msg)["Subject"]
        assert "{YYYYMMDD}" not in subject
        assert re.fullmatch(r"Run \d{8}", subject)

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

    def test_path_traversal_absolute_rejected(self, client, tmp_path):
        """In multi-user mode, absolute paths outside the user dir are rejected."""
        from vtsearch.auth import get_login_provider, set_login_provider

        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = get_login_provider()
        try:
            set_login_provider(_multi_user_provider(user_dir))
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
        finally:
            set_login_provider(original)

    def test_path_traversal_relative_rejected(self, client, tmp_path):
        """In multi-user mode, relative paths that escape the user dir are rejected."""
        from vtsearch.auth import get_login_provider, set_login_provider

        user_dir = tmp_path / "testuser"
        user_dir.mkdir()
        original = get_login_provider()
        try:
            set_login_provider(_multi_user_provider(user_dir))
            res = client.post(
                "/api/exporters/export",
                json={
                    "exporter_name": "server_json_file",
                    "field_values": {"filepath": "../../../etc/shadow"},
                    "results": SAMPLE_RESULTS,
                },
            )
            assert res.status_code == 400
        finally:
            set_login_provider(original)

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


# ---------------------------------------------------------------------------
# Streaming CLI export (--stream-results)
# ---------------------------------------------------------------------------

_STREAM_HEADER = {
    "media_type": "audio",
    "detectors": [{"detector_name": "dog_bark", "threshold": 0.5}],
    "keep_negatives": False,
}


def _stream_records():
    """Two good hits and one bad hit, in chunk order (not score-sorted)."""
    yield "dog_bark", {"id": 2, "filename": "b2.wav", "category": "c", "score": 0.7, "label": "good"}
    yield "dog_bark", {"id": 9, "filename": "b9.wav", "category": "c", "score": 0.9, "label": "good"}
    yield "dog_bark", {"id": 4, "filename": "b4.wav", "category": "c", "score": 0.2, "label": "bad"}


class TestStreamingExportSupport:
    def test_streaming_exporters_advertise_support(self):
        from vtscore.exporters import get_exporter

        for name in ("gui", "server_json_file", "server_csv_file", "webhook", "email_smtp"):
            assert get_exporter(name).supports_streaming is True

    def test_non_streaming_exporters_do_not(self):
        from vtscore.exporters import get_exporter

        # holder is a hidden scaffold exporter with no incremental mode.
        assert get_exporter("holder").supports_streaming is False


class TestResolveStreamBatchSize:
    """The shared batch-size coercion used by the delivery streamers."""

    def test_none_and_blank_fall_back_to_default(self):
        from vtscore.exporters.base import resolve_stream_batch_size

        assert resolve_stream_batch_size(None) == 500
        assert resolve_stream_batch_size("") == 500
        assert resolve_stream_batch_size(None, default=10) == 10

    def test_int_and_numeric_string(self):
        from vtscore.exporters.base import resolve_stream_batch_size

        assert resolve_stream_batch_size(3) == 3
        assert resolve_stream_batch_size("7") == 7

    def test_non_positive_and_garbage_fall_back(self):
        from vtscore.exporters.base import resolve_stream_batch_size

        assert resolve_stream_batch_size(0) == 500
        assert resolve_stream_batch_size(-4) == 500
        assert resolve_stream_batch_size("nope") == 500


class TestServerJsonStreaming:
    def test_writes_ndjson_with_meta_header(self):
        from vtscore.exporters import get_exporter

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hits.ndjson"
            res = get_exporter("server_json_file").export_cli_streaming(
                _STREAM_HEADER, _stream_records(), {"filepath": str(out)}
            )
            lines = [json.loads(ln) for ln in out.read_text().splitlines() if ln.strip()]
            assert lines[0]["_meta"]["format"] == "vtsearch-hits-ndjson/v1"
            hits = lines[1:]
            assert [h["id"] for h in hits] == [2, 9, 4]  # chunk order preserved
            assert hits[0]["detector"] == "dog_bark"
            assert "3 hit(s)" in res["message"]
            assert not out.with_name(out.name + ".tmp").exists()


class TestServerCsvStreaming:
    def test_writes_csv_rows(self):
        import csv

        from vtscore.exporters import get_exporter

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "hits.csv"
            get_exporter("server_csv_file").export_cli_streaming(
                _STREAM_HEADER, _stream_records(), {"filepath": str(out)}
            )
            rows = list(csv.reader(out.read_text().splitlines()))
            assert rows[0][0] == "detector"
            assert "label" in rows[0]
            assert len(rows) == 4  # header + 3 hits
            assert rows[1][0] == "dog_bark"


class TestGuiStreaming:
    def test_prints_only_good_hits(self, capsys):
        from vtscore.exporters import get_exporter

        res = get_exporter("gui").export_cli_streaming(_STREAM_HEADER, _stream_records(), {})
        captured = capsys.readouterr()
        # Only the two good hits print; the bad one is filtered.
        assert "b2.wav" in captured.out
        assert "b9.wav" in captured.out
        assert "b4.wav" not in captured.out
        assert "2 hit(s)" in res["message"]


class TestStreamingExportTempCollision:
    """Regression (audit #14): two concurrent streaming exports to the same
    destination path must not clobber each other's temp file.

    Both exporters used to build a single fixed ``<name>.tmp`` sibling, so the
    first export to finish renamed/deleted that shared temp out from under the
    other — surfacing as ``FileNotFoundError`` from ``os.replace`` (or a
    corrupted merged file). Each export now writes to a per-writer-unique
    ``<name>.<pid>.<uuid>.tmp``.
    """

    def _run_collision(self, exporter_name: str, out: Path) -> None:
        import threading

        from vtscore.exporters import get_exporter

        a_opened = threading.Event()
        b_done = threading.Event()
        a_result: dict[str, object] = {}
        a_error: list[BaseException] = []

        def a_records():
            # By the time this iterator is first pulled, export A has already
            # opened its temp file and written any header. Park here until
            # export B has fully completed against the same path so we exercise
            # the collision window.
            a_opened.set()
            b_done.wait(timeout=5)
            yield "dog_bark", {"id": 1, "filename": "a1.wav", "category": "c", "score": 0.5, "label": "good"}

        def run_a():
            try:
                a_result["res"] = get_exporter(exporter_name).export_cli_streaming(
                    _STREAM_HEADER, a_records(), {"filepath": str(out)}
                )
            except BaseException as exc:  # noqa: BLE001 - re-raised via assertion
                a_error.append(exc)

        ta = threading.Thread(target=run_a)
        ta.start()
        assert a_opened.wait(timeout=5), "export A never started"

        # Export B runs to completion while A is parked mid-stream.
        b_res = get_exporter(exporter_name).export_cli_streaming(
            _STREAM_HEADER, _stream_records(), {"filepath": str(out)}
        )
        b_done.set()
        ta.join(timeout=5)

        assert not a_error, f"export A failed on temp collision: {a_error[0]!r}"
        assert "3 hit(s)" in b_res["message"]
        assert "1 hit(s)" in a_result["res"]["message"]  # type: ignore[index]
        # Destination is intact (last atomic replace wins) and no temp leaked.
        assert out.exists()
        assert list(out.parent.glob("*.tmp")) == []

    def test_json_concurrent_exports_to_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_collision("server_json_file", Path(tmp) / "hits.ndjson")

    def test_csv_concurrent_exports_to_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._run_collision("server_csv_file", Path(tmp) / "hits.csv")


class TestWebhookStreaming:
    """webhook.export_cli_streaming POSTs hits in fixed-size batches."""

    def _mock_post(self):
        resp = MagicMock()
        resp.status_code = 200
        resp.raise_for_status.return_value = None
        return resp

    def test_batches_hits_into_separate_posts(self):
        from vtscore.exporters import get_exporter

        with patch("vtscore.exporters.webhook.requests.post", return_value=self._mock_post()) as mock_post:
            res = get_exporter("webhook").export_cli_streaming(
                _STREAM_HEADER, _stream_records(), {"url": "https://ex.com/hook", "batch_size": 2}
            )

        # 3 hits, batch_size 2 → 2 POSTs (2 hits, then 1 hit).
        assert mock_post.call_count == 2
        first_payload = mock_post.call_args_list[0].kwargs["json"]
        second_payload = mock_post.call_args_list[1].kwargs["json"]
        assert first_payload["format"] == "vtsearch-hits-batch/v1"
        assert first_payload["batch_index"] == 0
        assert [h["id"] for h in first_payload["hits"]] == [2, 9]
        assert first_payload["hits"][0]["detector"] == "dog_bark"
        assert second_payload["batch_index"] == 1
        assert [h["id"] for h in second_payload["hits"]] == [4]
        assert "3 hit(s)" in res["message"]
        assert "2 batch(es)" in res["message"]

    def test_auth_header_forwarded(self):
        from vtscore.exporters import get_exporter

        with patch("vtscore.exporters.webhook.requests.post", return_value=self._mock_post()) as mock_post:
            get_exporter("webhook").export_cli_streaming(
                _STREAM_HEADER,
                _stream_records(),
                {"url": "https://ex.com/hook", "auth_header": "Bearer tok", "batch_size": 100},
            )

        assert mock_post.call_args_list[0].kwargs["headers"]["Authorization"] == "Bearer tok"

    def test_empty_run_still_posts_once(self):
        from vtscore.exporters import get_exporter

        def _empty():
            return
            yield  # pragma: no cover - makes this a generator

        with patch("vtscore.exporters.webhook.requests.post", return_value=self._mock_post()) as mock_post:
            res = get_exporter("webhook").export_cli_streaming(_STREAM_HEADER, _empty(), {"url": "https://ex.com/hook"})

        assert mock_post.call_count == 1
        assert mock_post.call_args_list[0].kwargs["json"]["hits"] == []
        assert "0 hit(s)" in res["message"]
        assert "1 batch(es)" in res["message"]

    def test_default_batch_size_sends_single_post(self):
        from vtscore.exporters import get_exporter

        with patch("vtscore.exporters.webhook.requests.post", return_value=self._mock_post()) as mock_post:
            get_exporter("webhook").export_cli_streaming(
                _STREAM_HEADER, _stream_records(), {"url": "https://ex.com/hook"}
            )

        # Default batch_size is 500, so all 3 hits ride in one POST.
        assert mock_post.call_count == 1
        assert len(mock_post.call_args_list[0].kwargs["json"]["hits"]) == 3


class TestEmailStreaming:
    """email_smtp.export_cli_streaming sends one email per fixed-size batch."""

    def _patches(self):
        server = MagicMock()
        smtp_cm = MagicMock()
        smtp_cm.__enter__.return_value = server
        return (
            server,
            patch("vtscore.exporters.email_smtp.smtplib.SMTP", return_value=smtp_cm),
            patch("vtscore.exporters.email_smtp._resolve_mx", return_value="mx.example.com"),
        )

    def test_one_email_per_batch(self):
        from vtscore.exporters import get_exporter

        server, smtp_p, mx_p = self._patches()
        with smtp_p as smtp, mx_p:
            res = get_exporter("email_smtp").export_cli_streaming(
                _STREAM_HEADER,
                _stream_records(),
                {"from": "vt@example.com", "to": "user@example.com", "batch_size": 2},
            )

        # 3 hits, batch_size 2 → 2 emails.
        assert smtp.call_count == 2
        assert server.sendmail.call_count == 2
        assert "3 hit(s)" in res["message"]
        assert "2 email(s)" in res["message"]
        assert res["to"] == "user@example.com"

    def test_empty_run_still_sends_one_email(self):
        from vtscore.exporters import get_exporter

        def _empty():
            return
            yield  # pragma: no cover - makes this a generator

        server, smtp_p, mx_p = self._patches()
        with smtp_p as smtp, mx_p:
            res = get_exporter("email_smtp").export_cli_streaming(
                _STREAM_HEADER, _empty(), {"from": "vt@example.com", "to": "user@example.com"}
            )

        assert smtp.call_count == 1
        assert server.sendmail.call_count == 1
        assert "0 hit(s)" in res["message"]
        assert "1 email(s)" in res["message"]

    def test_invalid_recipient_rejected(self):
        from vtscore.exporters import get_exporter

        with pytest.raises(ValueError, match="Recipient email address is invalid"):
            get_exporter("email_smtp").export_cli_streaming(
                _STREAM_HEADER, _stream_records(), {"from": "vt@example.com", "to": "not-an-email"}
            )
