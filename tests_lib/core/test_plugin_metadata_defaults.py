"""Tests for ``PluginBase`` auto-derivation of ``name`` /
``display_name`` / ``description`` defaults."""

from __future__ import annotations

from vtscore.plugins import PluginBase, PluginField


class _FakeDatasetImporter(PluginBase):
    """An abstract intermediate that mimics the in-tree
    DatasetImporter shape; concrete subclasses below verify the
    default-derivation walks past it cleanly."""

    _is_plugin_family_base = True
    fields: list[PluginField] = []


class TestDefaultName:
    def test_strips_family_suffix(self):
        class MyShinyDatasetImporter(_FakeDatasetImporter):
            """One-liner doc."""

        assert MyShinyDatasetImporter.name == "my_shiny"

    def test_explicit_name_wins(self):
        class ServerThingDatasetImporter(_FakeDatasetImporter):
            """Doc."""

            name = "server_thing_file"

        assert ServerThingDatasetImporter.name == "server_thing_file"

    def test_camel_case_to_snake(self):
        class HTTPArchiveDatasetImporter(_FakeDatasetImporter):
            """Doc."""

        # ``HTTP`` stays together (a run of capitals before a lower-case
        # start) and is split before ``Archive``.
        assert HTTPArchiveDatasetImporter.name == "http_archive"

    def test_no_suffix_match_falls_back_to_snake_of_full_name(self):
        class StandaloneThing(PluginBase):
            """Doc."""

            _is_plugin_family_base = True
            fields: list[PluginField] = []

        class Concrete(StandaloneThing):
            """Doc."""

        # No suffix matched on ``Concrete`` itself (too short), so the
        # snake-cased class name comes through verbatim.
        assert Concrete.name == "concrete"


class TestDefaultDisplayName:
    def test_title_cases_words_of_name(self):
        class MyShinyDatasetImporter(_FakeDatasetImporter):
            """Doc."""

        assert MyShinyDatasetImporter.display_name == "My Shiny"

    def test_explicit_display_name_wins(self):
        class FooDatasetImporter(_FakeDatasetImporter):
            """Doc."""

            display_name = "The Foo Importer"

        assert FooDatasetImporter.display_name == "The Foo Importer"


class TestDefaultDescription:
    def test_uses_first_docstring_line(self):
        class FooDatasetImporter(_FakeDatasetImporter):
            """The first line.

            And then a body paragraph.
            """

        assert FooDatasetImporter.description == "The first line."

    def test_explicit_description_wins(self):
        class FooDatasetImporter(_FakeDatasetImporter):
            """Some other docstring."""

            description = "Explicit value."

        assert FooDatasetImporter.description == "Explicit value."

    def test_empty_doc_yields_empty_string(self):
        class FooDatasetImporter(_FakeDatasetImporter):
            pass

        assert FooDatasetImporter.description == ""


class TestAbstractIntermediateNotPolluted:
    def test_family_base_name_skipped(self):
        # The literal abstract base name ``DatasetImporter`` would
        # otherwise snake-case to ``"dataset_importer"`` and pollute
        # every concrete subclass that didn't declare ``name``.
        from vtscore.datasets.importers.base import DatasetImporter

        # The abstract base must NOT have a derived ``name`` stamped
        # onto its own ``__dict__`` (it may declare nothing, in which
        # case ``getattr`` raises; check ``__dict__`` directly).
        assert "name" not in DatasetImporter.__dict__

    def test_concrete_in_tree_plugin_keeps_explicit_name(self):
        # Auto-derivation must never overwrite an explicit class attr.
        from vtscore.datasets.importers.server_folder import ServerFolderDatasetImporter

        assert ServerFolderDatasetImporter.name == "server_folder"

    def test_opt_out_marker_skips_auto_derive(self):
        class ThirdPartyAbstractBase(_FakeDatasetImporter):
            """A third-party intermediate that opts out."""

            _is_plugin_family_base = True

        # The intermediate itself has no ``name`` in its own __dict__.
        assert "name" not in ThirdPartyAbstractBase.__dict__

        class Concrete(ThirdPartyAbstractBase):
            """Doc."""

        # The concrete subclass below the opt-out intermediate gets the
        # normal auto-derived ``name``, not the intermediate's class
        # name.
        assert Concrete.name == "concrete"


class TestMediaConverterPropertyNotStomped:
    def test_concrete_converter_inherits_property(self):
        # ``MediaConverter.name`` is a property derived from
        # ``source_type`` + ``target_type``.  Auto-derivation must not
        # replace it on concrete subclasses.
        from vtscore.converters.video2image import Video2ImageMediaConverter  # noqa: PLC0415

        # ``name`` resolves to the property's computed value on an
        # instance (e.g. ``"video2image"``), proving the property
        # survived.
        instance = Video2ImageMediaConverter()
        assert instance.name == "video2image"
