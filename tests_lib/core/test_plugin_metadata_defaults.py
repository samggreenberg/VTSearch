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


class TestDefaultIcon:
    def test_derives_letter_from_display_name(self):
        class GoogleDatasetImporter(_FakeDatasetImporter):
            """Doc."""

        assert GoogleDatasetImporter.icon == "G"

    def test_explicit_icon_wins(self):
        class GoogleDatasetImporter(_FakeDatasetImporter):
            """Doc."""

            icon = "☁️"  # cloud

        assert GoogleDatasetImporter.icon == "☁️"

    def test_overrides_family_stock_icon(self):
        # ``ImporterBase.icon`` is the generic "\U0001f50c" plug emoji
        # shared by every dataset importer that doesn't customise it;
        # a concrete subclass should get its own letter instead.
        from vtscore.datasets.importers.base import DatasetImporter

        class FreshDatasetImporter(DatasetImporter):
            """Doc."""

            fields: list[PluginField] = []

            def run(self, field_values):  # noqa: D102
                raise NotImplementedError

        assert FreshDatasetImporter.icon == "F"

    def test_no_alpha_display_name_keeps_family_stock_icon(self):
        class _123DatasetImporter(_FakeDatasetImporter):
            """Doc."""

            name = "123"
            display_name = "123"

        # No alphabetic character to derive a letter from, so the
        # inherited family-stock icon (empty on the fake base) is left
        # untouched rather than being blanked out.
        assert _123DatasetImporter.icon == ""


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


class TestFamilyBaseContributesItsSuffix:
    """A family base's ``__name__`` is strippable by its own subclasses.

    This is what lets a new plugin family be declared by marking one class,
    instead of also appending its name to a central suffix table that a
    later family will forget to update.
    """

    def test_subclass_strips_the_base_name(self):
        class AcmeWidgetBase(PluginBase):
            """A third-party family base."""

            _is_plugin_family_base = True
            fields: list[PluginField] = []

        class ShinyAcmeWidgetBase(AcmeWidgetBase):
            """Doc."""

        assert ShinyAcmeWidgetBase.name == "shiny"

    def test_base_name_beats_a_generic_suffix(self):
        """Longest match wins, so the family's own name is preferred over
        the generic tail even though both are suffixes of the class name."""

        class AcmeWidgetExporter(PluginBase):
            """A third-party family base whose name ends in ``Exporter``."""

            _is_plugin_family_base = True
            fields: list[PluginField] = []

        class ShinyAcmeWidgetExporter(AcmeWidgetExporter):
            """Doc."""

        # Not ``shiny_acme_widget``, which is what stripping the generic
        # ``Exporter`` alone would have produced.
        assert ShinyAcmeWidgetExporter.name == "shiny"

    def test_non_strippable_base_withholds_its_name(self):
        class AcmeSyncSource(PluginBase):
            """A base whose name must not be stripped."""

            _is_plugin_family_base = True
            _strippable_family_base = False
            fields: list[PluginField] = []

        class ShinyAcmeSyncSource(AcmeSyncSource):
            """Doc."""

        # Falls through to the generic ``Source``, exactly as it did before
        # family bases contributed suffixes at all.
        assert ShinyAcmeSyncSource.name == "shiny_acme_sync"

    def test_contribution_is_scoped_to_the_subclass_mro(self):
        """One family's base name never leaks into another family.

        The suffix set is read off the deriving class's own MRO rather than
        a process-global registry, so the derived name can't depend on which
        unrelated modules happened to be imported first.
        """

        class AcmeWidgetBase(PluginBase):
            """Family A."""

            _is_plugin_family_base = True
            fields: list[PluginField] = []

        class OtherFamilyBase(PluginBase):
            """Family B, which knows nothing about family A."""

            _is_plugin_family_base = True
            fields: list[PluginField] = []

        class ShinyAcmeWidgetBase(OtherFamilyBase):
            """A family-B plugin whose name happens to end in family A's."""

        assert ShinyAcmeWidgetBase.name == "shiny_acme_widget_base"


class TestStockIconDetectedByDefiningClass:
    """An inherited icon counts as "no icon chosen" when a *family base*
    defines it, rather than when it matches one of the emoji this repo
    happens to ship."""

    def test_third_party_family_stock_icon_is_replaced(self):
        class AcmeWidgetBase(PluginBase):
            """A third-party family base with its own generic glyph."""

            _is_plugin_family_base = True
            icon = "🚀"
            fields: list[PluginField] = []

        class ShinyAcmeWidgetBase(AcmeWidgetBase):
            """Doc."""

        assert ShinyAcmeWidgetBase.icon == "S"

    def test_icon_inherited_from_a_non_base_ancestor_is_kept(self):
        class AcmeWidgetBase(PluginBase):
            """Doc."""

            _is_plugin_family_base = True
            fields: list[PluginField] = []

        class DeliberateAcmeWidgetBase(AcmeWidgetBase):
            """A concrete plugin that chose an icon."""

            icon = "🚀"

        class VariantAcmeWidgetBase(DeliberateAcmeWidgetBase):
            """A subclass of it, which inherits that deliberate choice."""

        assert VariantAcmeWidgetBase.icon == "🚀"

    def test_explicit_icon_matching_another_familys_stock_glyph_survives(self):
        """The old rule compared codepoints against a table of our own
        emoji, so a plugin that deliberately picked one of them was at the
        mercy of where it sat in the MRO."""

        class AcmeWidgetBase(PluginBase):
            """Doc."""

            _is_plugin_family_base = True
            icon = "🔌"
            fields: list[PluginField] = []

        class ShinyAcmeWidgetBase(AcmeWidgetBase):
            """Doc."""

            icon = "🔌"  # the dataset-importer family's stock plug, on purpose

        assert ShinyAcmeWidgetBase.icon == "🔌"
