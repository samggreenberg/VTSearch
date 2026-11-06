"""Tests for ``vtscore.detectors.import_record`` (#5334, #5620).

The record a detector keeps of the label importer that created it follows a
dataset origin's field policy, so an upload or a secret is never written to
the registry; read back for the Stats window, it uses the importer's own field
labels and words a checkbox as ``Yes`` / ``No``.
"""

from __future__ import annotations

from vtscore.detectors import import_record
from vtscore.detectors.import_record import build_import_record, import_rows, label_importer_display_name
from vtscore.labels.importers import get_label_importer
from vtscore.labels.importers.base import LabelImporter, PluginField


class _Importer(LabelImporter):
    name = "fake_labels"
    display_name = "Fake labels"
    description = ""
    fields = [
        PluginField("host", "Host", "text"),
        PluginField("token", "Token", "password"),
        PluginField("upload", "Upload", "file"),
        PluginField("recursive", "Recursive", "checkbox", default="false"),
        PluginField("note", "Note", "text", required=False),
        PluginField("tags", "Tags", "text", required=False, origin_serializer=lambda v: ",".join(v)),
        PluginField("cursor", "Cursor", "text", required=False, include_in_origin=False),
    ]


def test_records_the_importer_its_fields_and_the_time():
    record = build_import_record(_Importer(), {"host": "db.example.test", "recursive": True}, imported_at=123.0)
    assert record == {
        "importer": "fake_labels",
        "params": {"host": "db.example.test", "recursive": "true"},
        "imported_at": 123.0,
    }


def test_leaves_out_secrets_uploads_blanks_and_opted_out_fields():
    record = build_import_record(
        _Importer(),
        {"host": "h", "token": "s3cret", "upload": object(), "note": "", "cursor": "abc", "tags": ["a", "b"]},
    )
    assert record["params"] == {"host": "h", "recursive": "false", "tags": "a,b"}
    assert "s3cret" not in repr(record)


def test_defaults_the_time_to_now():
    import time

    before = time.time()
    record = build_import_record(_Importer(), {"host": "h"})
    assert before <= record["imported_at"] <= time.time()


# ---------------------------------------------------------------------------
# The record read back as the Stats window's Import rows (#5620)
# ---------------------------------------------------------------------------


class _NamingImporter(_Importer):
    """The docs' shape: the form holds an id, only the plugin knows its name."""

    name = "naming_labels"
    display_name = "Naming labels"

    def describe_origin_param(self, key, value):
        if key == "host":
            return "" if value == "hidden" else f"Host {value}"
        return super().describe_origin_param(key, value)


def _registered(*importers: LabelImporter):
    """A ``get_label_importer`` that knows *importers* and nothing else."""
    by_name = {imp.name: imp for imp in importers}
    return lambda name: by_name.get(name)


class TestDescribeOriginParamHook:
    def test_the_default_reads_a_checkbox_as_yes_or_no_and_anything_else_as_recorded(self):
        importer = _Importer()
        assert importer.describe_origin_param("recursive", "true") == "Yes"
        assert importer.describe_origin_param("recursive", "false") == "No"
        assert importer.describe_origin_param("host", "db.example.test") == "db.example.test"
        assert importer.describe_origin_param("not_a_field", "x") == "x"

    def test_the_stock_importers_word_their_path_as_recorded(self):
        for name in ("server_json_file", "server_csv_file"):
            importer = get_label_importer(name)
            assert importer is not None
            assert importer.describe_origin_param("filepath", "/data/labels.json") == "/data/labels.json"


class TestImportRows:
    def test_a_server_json_import_reads_in_its_own_field_labels(self):
        """The issue's own example (#5620)."""
        importer = get_label_importer("server_json_file")
        assert importer is not None
        record = build_import_record(importer, {"filepath": "/data/labels/cats.json"})

        rows = import_rows(record)
        assert [(row.key, row.label, row.value) for row in rows] == [
            ("filepath", "Server path", "/data/labels/cats.json")
        ]
        assert rows[0].description == importer.fields[0].description

    def test_a_checkbox_reads_yes_or_no(self, monkeypatch):
        monkeypatch.setattr(import_record, "get_label_importer", _registered(_Importer()))
        record = build_import_record(_Importer(), {"host": "h", "recursive": True})
        rows = {row.key: row for row in import_rows(record)}
        assert (rows["recursive"].label, rows["recursive"].value) == ("Recursive", "Yes")
        assert (rows["host"].label, rows["host"].value) == ("Host", "h")

    def test_an_importers_override_is_what_the_rows_show(self, monkeypatch):
        monkeypatch.setattr(import_record, "get_label_importer", _registered(_NamingImporter()))
        rows = {row.key: row for row in import_rows({"importer": "naming_labels", "params": {"host": "db"}})}
        assert rows["host"].value == "Host db"
        # "" leaves the row out.
        assert import_rows({"importer": "naming_labels", "params": {"host": "hidden"}}) == []

    def test_an_unregistered_importer_still_lists_its_params_as_recorded(self):
        """A detector outlives the plugin that imported it."""
        rows = import_rows({"importer": "gone", "params": {"some_flag": "true", "note": ""}})
        assert [(row.key, row.label, row.value, row.description) for row in rows] == [
            ("some_flag", "Some flag", "true", "")
        ]

    def test_a_detector_no_importer_created_has_no_rows(self):
        assert import_rows(None) == []
        assert import_rows({"importer": "server_json_file", "params": "nope"}) == []


class TestLabelImporterDisplayName:
    def test_a_registered_importer_reads_as_its_display_name(self):
        importer = get_label_importer("server_json_file")
        assert importer is not None
        assert label_importer_display_name("server_json_file") == importer.display_name == "Server JSON File"

    def test_an_unregistered_or_missing_importer_reads_as_given(self):
        assert label_importer_display_name("gone") == "gone"
        assert label_importer_display_name("") == ""
