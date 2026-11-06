"""A dataset origin worded for the Stats window (#5617).

``origin_rows`` turns the re-import recipe a dataset's origin records into the
rows a person reads: the importer's own field labels, ``Yes`` / ``No`` for a
checkbox, and the converters an import ran rather than ``source_specs`` JSON.
"""

from __future__ import annotations

import json

from vtscore.datasets.importers import get_importer
from vtscore.datasets.importers.origin_rows import (
    INCLUDE_MEDIA_LABEL,
    OriginRow,
    describe_source_specs,
    importer_display_name,
    origin_rows,
)

_FACES_FROM_IMAGES = [{"source_type": "image", "converter": "image2face", "params": {}}]


def _rows(origin, media_type=""):
    return {row.key: row for row in origin_rows(origin, media_type=media_type)}


class TestDescribeSourceSpecs:
    def test_a_converted_row_reads_as_its_converter(self):
        """The issue's own example: a Face dataset cut from images."""
        assert describe_source_specs(json.dumps(_FACES_FROM_IMAGES)) == "Images → Faces"

    def test_rows_that_are_all_direct_say_nothing(self):
        """Every import of a dataset's own type records this; it is not news."""
        assert describe_source_specs('[{"source_type": "image", "converter": null, "params": {}}]') == ""

    def test_a_direct_row_beside_a_converted_one_is_marked(self):
        specs = [
            {"source_type": "image", "converter": None, "params": {}},
            {"source_type": "document", "converter": "document2image", "params": {}},
        ]
        assert describe_source_specs(specs) == "Image (direct), Document → Images"

    def test_only_params_moved_off_their_defaults_are_named(self):
        """A spec carries every param; naming the untouched ones is noise."""
        specs = [
            {
                "source_type": "image",
                "converter": "image2face",
                "params": {"threshold": 0.5, "padding": "0.25", "min_size": 64, "align": "roll"},
            }
        ]
        assert describe_source_specs(specs) == "Images → Faces (Minimum face size (pixels): 64)"

    def test_an_unregistered_converter_shows_its_name(self):
        specs = [{"source_type": "image", "converter": "gone2nowhere", "params": {"k": 1}}]
        assert describe_source_specs(specs) == "gone2nowhere"

    def test_a_value_that_is_not_rows_is_returned_as_given(self):
        assert describe_source_specs("not json") == "not json"
        assert describe_source_specs('{"source_type": "image"}') == '{"source_type": "image"}'


class TestOriginRows:
    def test_a_folder_import_reads_in_its_own_field_labels(self):
        origin = get_importer("server_folder").build_origin(
            {"path": "/data/faces", "media_type": "face", "recursive": "true", "source_specs": _FACES_FROM_IMAGES}
        )
        rows = _rows(origin, media_type="face")

        assert rows["path"] == OriginRow(
            key="path", label="Folder or archive", value="/data/faces", description=rows["path"].description
        )
        assert rows["path"].description  # the field's own description, for a tooltip
        assert (rows["recursive"].label, rows["recursive"].value) == ("Include subfolders", "Yes")
        assert (rows["dig_archives"].label, rows["dig_archives"].value) == ("Include archives' contents", "No")
        assert (rows["source_specs"].label, rows["source_specs"].value) == (INCLUDE_MEDIA_LABEL, "Images → Faces")

    def test_the_datasets_own_media_type_is_left_out(self):
        """The window's Type row says it already."""
        origin = {"importer": "server_folder", "params": {"media_type": "face", "path": "/p"}}
        assert "media_type" not in _rows(origin, media_type="face")

    def test_a_media_type_other_than_the_datasets_is_named(self):
        origin = {"importer": "server_folder", "params": {"media_type": "video"}}
        assert _rows(origin, media_type="image")["media_type"].value == "Video"

    def test_direct_only_specs_and_empty_values_leave_no_row(self):
        origin = {
            "importer": "server_folder",
            "params": {"path": "", "source_specs": '[{"source_type": "image", "converter": null}]'},
        }
        assert origin_rows(origin, media_type="image") == []

    def test_params_keep_their_recorded_order(self):
        origin = {"importer": "server_folder", "params": {"recursive": "true", "path": "/p"}}
        assert [row.key for row in origin_rows(origin)] == ["recursive", "path"]

    def test_the_demo_importer_names_its_demo_and_converter(self):
        from vtscore.datasets.config import DEMO_DATASETS

        demo_id, entry = next((k, v) for k, v in DEMO_DATASETS.items() if v.get("label"))
        rows = _rows({"importer": "demo", "params": {"name": demo_id, "converter": "image2face"}})
        assert (rows["name"].label, rows["name"].value) == ("Dataset", entry["label"])
        assert (rows["converter"].label, rows["converter"].value) == ("Converter", "Images → Faces")

    def test_an_unregistered_importer_still_lists_its_params(self):
        """A dataset outlives the plugin that imported it."""
        origin = {
            "importer": "gone",
            "params": {"dig_archives": "false", "source_specs": json.dumps(_FACES_FROM_IMAGES)},
        }
        rows = _rows(origin)
        assert (rows["dig_archives"].label, rows["dig_archives"].value) == ("Dig archives", "false")
        assert rows["source_specs"].value == "Images → Faces"

    def test_a_malformed_origin_has_no_rows(self):
        assert origin_rows(None) == []
        assert origin_rows({"importer": "server_folder", "params": "nope"}) == []

    def test_a_row_serialises_to_the_wire_shape(self):
        row = OriginRow(key="k", label="K", value="v")
        assert row.to_dict() == {"key": "k", "label": "K", "value": "v", "description": ""}


class TestImporterDisplayName:
    def test_a_registered_importer_reads_as_its_display_name(self):
        assert importer_display_name("server_files") == get_importer("server_files").display_name

    def test_an_unregistered_or_missing_importer_reads_as_given(self):
        assert importer_display_name("gone") == "gone"
        assert importer_display_name("") == ""


class TestDescribeOriginParamHook:
    """An importer words its own params by overriding ``describe_origin_param``."""

    def test_an_override_is_what_the_rows_show(self, monkeypatch):
        importer = get_importer("server_folder")
        monkeypatch.setattr(
            type(importer),
            "describe_origin_param",
            lambda self, key, value: "" if key == "path" else f"<{value}>",
        )
        rows = _rows({"importer": "server_folder", "params": {"path": "/p", "recursive": "true"}})
        assert "path" not in rows  # "" leaves the row out
        assert rows["recursive"].value == "<true>"
