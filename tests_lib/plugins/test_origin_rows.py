"""Tests for :mod:`vtscore.plugins.origin_rows` (#5620), the row builder both Stats windows share.

A dataset's origin and a detector's import record read the same way: the
plugin's own field labels, ``Yes`` / ``No`` for a checkbox, and each value
worded by a callable the family hands in.
"""

from __future__ import annotations

from vtscore.plugins import PluginField
from vtscore.plugins.origin_rows import OriginRow, describe_field_value, format_param_key, param_rows

_FIELDS = [
    PluginField("path", "Server path", "server_path", description="A file on the server."),
    PluginField("recursive", "Include subfolders", "checkbox", default="false"),
]


def _describe(key: str, value: str) -> str:
    """The default wording a family's hook starts from."""
    return describe_field_value(next((f for f in _FIELDS if f.key == key), None), value)


class TestDescribeFieldValue:
    def test_a_checkbox_reads_yes_or_no(self):
        box = _FIELDS[1]
        assert describe_field_value(box, "true") == "Yes"
        assert describe_field_value(box, "false") == "No"
        assert describe_field_value(box, True) == "Yes"

    def test_anything_else_reads_as_given(self):
        assert describe_field_value(_FIELDS[0], "/p") == "/p"
        assert describe_field_value(None, 32) == "32"


def test_format_param_key_spells_a_key_out():
    assert format_param_key("dig_archives") == "Dig archives"
    assert format_param_key("") == ""


class TestParamRows:
    def test_rows_carry_the_fields_own_labels_and_descriptions(self):
        assert param_rows(_FIELDS, {"path": "/p", "recursive": "true"}, describe=_describe) == [
            OriginRow(key="path", label="Server path", value="/p", description="A file on the server."),
            OriginRow(key="recursive", label="Include subfolders", value="Yes"),
        ]

    def test_a_key_naming_no_field_is_spelled_out_unless_the_caller_labels_it(self):
        assert param_rows(_FIELDS, {"extra_key": "x"}, describe=_describe) == [
            OriginRow(key="extra_key", label="Extra key", value="x")
        ]
        rows = param_rows(_FIELDS, {"extra_key": "x"}, describe=_describe, label_for=str.upper)
        assert rows[0].label == "EXTRA_KEY"

    def test_empty_values_and_values_worded_as_empty_are_left_out(self):
        rows = param_rows(
            _FIELDS,
            {"path": "", "recursive": None, "note": "n", "kept": "k"},
            describe=lambda key, value: "" if key == "note" else value,
        )
        assert [row.key for row in rows] == ["kept"]

    def test_the_wording_sees_the_recorded_string(self):
        seen: list[tuple[str, str]] = []

        def describe(key: str, value: str) -> str:
            seen.append((key, value))
            return value

        param_rows(_FIELDS, {"recursive": True, "n": 3}, describe=describe)
        assert seen == [("recursive", "True"), ("n", "3")]

    def test_params_keep_their_recorded_order(self):
        rows = param_rows(_FIELDS, {"recursive": "true", "path": "/p"}, describe=_describe)
        assert [row.key for row in rows] == ["recursive", "path"]

    def test_params_that_are_not_a_mapping_have_no_rows(self):
        assert param_rows(_FIELDS, None, describe=_describe) == []
        assert param_rows(_FIELDS, "nope", describe=_describe) == []  # type: ignore[arg-type]

    def test_a_row_serialises_to_the_wire_shape(self):
        row = OriginRow(key="k", label="K", value="v")
        assert row.to_dict() == {"key": "k", "label": "K", "value": "v", "description": ""}
