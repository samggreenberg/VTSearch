"""Tests for :meth:`LabelImporter.default_detector_name` (issue #5342).

The hook names the detector New Detector › Trained creates from a label
importer.  Its base implementation must give exactly the names the form
gave before the hook existed (the uploaded file's stem, else
``"<display_name> detector"``), so the stock importers are unaffected and
only a plugin that overrides it names detectors differently.
"""

from __future__ import annotations

from vtscore.labels.importers import get_label_importer
from vtscore.labels.importers.base import LabelImporter, PluginField
from vtscore.plugins.uploads import BytesIOUploadedFile


class _UploadImporter(LabelImporter):
    name = "_stub_upload_labels"
    display_name = "Upload Labels"
    description = "Test importer with a file field."
    fields = [
        PluginField("note", "Note", "text"),
        PluginField("file", "Labels file", "file"),
    ]


class _QueryImporter(LabelImporter):
    """The issue's shape: the form holds an id, only the plugin knows its name."""

    name = "_stub_query_labels"
    display_name = "Saved Query"
    description = "Test importer that names detectors after a saved query."
    fields = [PluginField("query_id", "Query", "select", options=["q-8f31"])]

    def default_detector_name(self, field_values):
        if field_values.get("query_id") == "q-8f31":
            return "Q1 Field Survey"
        return super().default_detector_name(field_values)


class TestBaseDerivation:
    def test_names_the_detector_after_the_chosen_files_stem(self):
        assert _UploadImporter().default_detector_name({"file": "barking dogs.json"}) == "barking dogs"

    def test_reads_an_upload_object_too(self):
        upload = BytesIOUploadedFile(b"{}", "dog barks.csv")
        assert _UploadImporter().default_detector_name({"file": upload}) == "dog barks"

    def test_drops_a_directory_and_collapses_whitespace(self):
        name = _UploadImporter().default_detector_name({"file": "C:\\exports\\my   cat\tlabels.json"})
        assert name == "my cat labels"

    def test_keeps_a_dotfile_name_whole(self):
        assert _UploadImporter().default_detector_name({"file": ".labels"}) == ".labels"

    def test_falls_back_to_the_importer_name_without_a_file(self):
        assert _UploadImporter().default_detector_name({"note": "anything"}) == "Upload Labels detector"
        assert _UploadImporter().default_detector_name({}) == "Upload Labels detector"

    def test_stock_server_importers_keep_their_names(self):
        """A server path is not an upload, so the name is the importer's, as before."""
        importer = get_label_importer("server_json_file")
        assert importer is not None
        name = importer.default_detector_name({"filepath": "/data/labels/dogs.json"})
        assert name == f"{importer.display_name} detector"


class TestOverride:
    def test_a_plugin_can_resolve_an_opaque_selection(self):
        assert _QueryImporter().default_detector_name({"query_id": "q-8f31"}) == "Q1 Field Survey"

    def test_super_gives_the_base_fallback(self):
        assert _QueryImporter().default_detector_name({"query_id": "other"}) == "Saved Query detector"
