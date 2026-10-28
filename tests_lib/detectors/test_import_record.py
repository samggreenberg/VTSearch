"""Tests for ``vtscore.detectors.import_record`` (#5334).

The record a detector keeps of the label importer that created it follows a
dataset origin's field policy, so an upload or a secret is never written to
the registry.
"""

from __future__ import annotations

from vtscore.detectors.import_record import build_import_record
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
