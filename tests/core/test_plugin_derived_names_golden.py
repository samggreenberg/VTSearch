"""Golden list pinning every in-tree plugin's derived metadata.

A plugin's ``name`` is a **registry key**: it is what ``get_exporter("…")``
looks up, what a third party writes into an entry-point config, what
``origin.params`` records, and what persisted settings store.  Most of it is
produced by :func:`vtscore.plugins._default_plugin_name`, which strips a
family suffix off the class name and snake-cases the remainder.  A change to
that derivation renames plugins silently — no error, no warning, just a
lookup that stops resolving on somebody's install.

This file is the tripwire.  Every :class:`~vtscore.plugins.PluginBase`
subclass shipped in ``vtscore`` / ``vtsearch`` gets one row recording:

1. ``_default_plugin_name(cls)`` — the *pure derivation*, evaluated even for
   classes that declare an explicit ``name``.  This is the column that
   actually pins the suffix-stripping rules, because it samples the real
   class names we ship against the real function.
2. the effective ``name`` / ``display_name`` / ``icon`` after
   ``__init_subclass__`` has run (``None`` where the attribute is absent or
   is a non-string descriptor, e.g. ``MediaConverter.name`` is a property).

Adding a plugin is expected to add a row here; **changing** a row means you
renamed a registry key, and that is a breaking change for anyone whose
config names the old value.

See also ``tests_lib/core/test_plugin_metadata_defaults.py`` (unit tests for
the derivation rules) and
``tests_lib/core/test_plugin_name_suffix_contract.py`` (the out-of-tree
half: the derivation's behaviour on class names this repo does not ship).
"""

from __future__ import annotations

import functools

import pytest

from vtscore.plugins import PluginBase, _default_plugin_name

# (derived_name, name, display_name, icon); ``None`` = attribute absent or
# not a plain string.
GOLDEN: dict[str, tuple[str, str | None, str | None, str | None]] = {
    "vtscore.converters.audio2image.Audio2ImageMediaConverter": (
        "audio2_image",
        None,
        "Audio → Image (spectrogram)",
        "A",
    ),
    "vtscore.converters.audio2text.Audio2TextMediaConverter": ("audio2_text", None, "Audio → Text (Whisper ASR)", "A"),
    "vtscore.converters.base.MediaConverter": ("media", None, "", ""),
    "vtscore.converters.document2image.Document2ImageMediaConverter": (
        "document2_image",
        None,
        "Document → Images",
        "D",
    ),
    "vtscore.converters.document2text.Document2TextMediaConverter": ("document2_text", None, "Document → Text", "D"),
    "vtscore.converters.image2face.Image2FaceMediaConverter": ("image2_face", None, "Images → Faces", "I"),
    "vtscore.converters.image2text.Image2TextMediaConverter": ("image2_text", None, "Image → Text (OCR)", "I"),
    "vtscore.converters.video2audio.Video2AudioMediaConverter": ("video2_audio", None, "Video → Audio", "V"),
    "vtscore.converters.video2image.Video2ImageMediaConverter": ("video2_image", None, "Video → Images", "V"),
    "vtscore.datasets.importers.base.core.ImporterBase": ("importer_base", None, None, "🔌"),
    "vtscore.datasets.importers.base.dataset_importer.DatasetImporter": ("dataset", None, None, "🔌"),
    "vtscore.datasets.importers.combine_datasets.CombineDatasetsImporter": (
        "combine_datasets",
        "combine_datasets",
        "Combined Datasets",
        "🔀",
    ),
    "vtscore.datasets.importers.demo.DemoDatasetImporter": ("demo", "demo", "Downloaded Media", "🗄"),
    "vtscore.datasets.importers.http_archive.HttpArchiveDatasetImporter": (
        "http_archive",
        "http_archive",
        "Import from URL",
        "🌐",
    ),
    "vtscore.datasets.importers.local_archive_member.LocalArchiveMemberImporter": (
        "local_archive_member",
        "local_archive_member",
        "Archive members (no extract)",
        "📦",
    ),
    "vtscore.datasets.importers.local_files.LocalFilesDatasetImporter": ("local_files", "local_files", "Files", "📄"),
    "vtscore.datasets.importers.local_folder.LocalFolderDatasetImporter": (
        "local_folder",
        "local_folder",
        "Folder",
        "📁",
    ),
    "vtscore.datasets.importers.pickle.PickleDatasetImporter": ("pickle", "pickle", "Upload Saved Dataset", "📤"),
    "vtscore.datasets.importers.server_files.ServerFilesDatasetImporter": (
        "server_files",
        "server_files",
        "Manifest",
        "🗂",
    ),
    "vtscore.datasets.importers.server_folder.ServerFolderDatasetImporter": (
        "server_folder",
        "server_folder",
        "Folder",
        "📁",
    ),
    "vtscore.datasets.importers.synthetic.SyntheticDatasetImporter": (
        "synthetic",
        "synthetic",
        "Synthetic Media",
        "🏭",
    ),
    "vtscore.datasource_importers.base.DataSourceImporter": ("data_source", None, None, "📥"),
    "vtscore.datasource_importers.server_file.ServerFileDataSourceImporter": (
        "server_file",
        "server_file",
        "Server File",
        "📄",
    ),
    "vtscore.datasource_importers.url_download.UrlDownloadDataSourceImporter": (
        "url_download",
        "url_download",
        "URL",
        "🌐",
    ),
    "vtscore.exporters.base.ResultsExporter": ("results", None, None, "📤"),
    "vtscore.exporters.email_smtp.EmailResultsExporter": ("email", "email_smtp", "Send by Email", "📧"),
    "vtscore.exporters.gui.DisplayResultsExporter": ("display", "gui", "Display Results", "🖥️"),
    "vtscore.exporters.open_url.OpenUrlResultsExporter": ("open_url", "open_url", "Open in Website", "🔗"),
    "vtscore.exporters.portable_detector.PortableDetectorResultsExporter": (
        "portable_detector",
        "portable_detector",
        "Portable Detector Bundle",
        "📦",
    ),
    "vtscore.exporters.server_csv_file.ServerCsvResultsExporter": (
        "server_csv",
        "server_csv_file",
        "Server CSV File",
        "🖥",
    ),
    "vtscore.exporters.server_json_file.ServerJsonResultsExporter": (
        "server_json",
        "server_json_file",
        "Server JSON File",
        "🖥",
    ),
    "vtscore.exporters.webhook.WebhookResultsExporter": ("webhook", "webhook", "Webhook (HTTP POST)", "🌐"),
    "vtscore.labels.importers.base.LabelImporter": ("label", None, None, "🏷️"),
    "vtscore.labels.importers.server_csv_file.ServerCsvLabelImporter": (
        "server_csv",
        "server_csv_file",
        "Server CSV File",
        "🖥",
    ),
    "vtscore.labels.importers.server_json_file.ServerJsonLabelImporter": (
        "server_json",
        "server_json_file",
        "Server JSON File",
        "🖥",
    ),
    "vtscore.labels.sources.base.LabelsetSource": ("labelset", None, None, "🔄"),
    "vtscore.labels.sources.server_json_file.ServerFileLabelsetSource": (
        "server_file",
        "server_json_file",
        "Server JSON File",
        "🖥",
    ),
    "vtscore.seed_importers.base.SeedImporter": ("seed", None, None, "🌱"),
    "vtscore.sync.SyncSource": ("sync", None, None, "🔄"),
    "vtsearch.settings_io.exporters.base.SettingsExporter": ("settings", None, None, "📤"),
    "vtsearch.settings_io.exporters.local_json_file.LocalFileSettingsExporter": (
        "local_file",
        "local_json_file",
        "Local JSON File",
        "📁",
    ),
    "vtsearch.settings_io.exporters.server_json_file.ServerFileSettingsExporter": (
        "server_file",
        "server_json_file",
        "Server JSON File",
        "🖥",
    ),
    "vtsearch.settings_io.importers.base.SettingsImporter": ("settings", None, None, "⚙️"),
    "vtsearch.settings_io.importers.local_json_file.LocalFileSettingsImporter": (
        "local_file",
        "local_json_file",
        "Local JSON File",
        "📁",
    ),
    "vtsearch.settings_io.importers.server_json_file.ServerFileSettingsImporter": (
        "server_file",
        "server_json_file",
        "Server JSON File",
        "🖥",
    ),
    "vtsearch.settings_io.sources.base.SettingsSource": ("settings", None, None, "🔄"),
    "vtsearch.settings_io.sources.server_json_file.ServerFileSettingsSource": (
        "server_file",
        "server_json_file",
        "Server JSON File",
        "🖥",
    ),
}


def _import_all_plugin_families() -> None:
    """Import + discover every in-tree plugin family.

    Each ``list_*`` call runs the registry's package scan, which imports the
    concrete plugin modules and therefore triggers
    ``PluginBase.__init_subclass__`` for every shipped plugin.
    """
    from vtscore import converters, datasource_importers, exporters, seed_importers, sync  # noqa: F401
    from vtscore.datasets import importers as dataset_importers
    from vtscore.labels import importers as label_importers
    from vtscore.labels import sources as labelset_sources
    from vtsearch.settings_io import exporters as settings_exporters
    from vtsearch.settings_io import importers as settings_importers
    from vtsearch.settings_io import sources as settings_sources

    for discover in (
        converters.list_converters,
        datasource_importers.list_datasource_importers,
        seed_importers.list_seed_importers,
        label_importers.list_label_importers,
        labelset_sources.list_labelset_sources,
        exporters.list_exporters,
        dataset_importers.list_importers,
        settings_exporters.list_settings_exporters,
        settings_importers.list_settings_importers,
        settings_sources.list_settings_sources,
    ):
        discover()


@functools.cache
def _in_tree_plugin_classes() -> dict[str, type]:
    """Return every shipped ``PluginBase`` subclass, keyed by import path.

    Filtered to the ``vtscore`` / ``vtsearch`` packages so that plugin
    classes defined by *other test modules* in the same worker (there are
    several) can never leak into the comparison.
    """
    _import_all_plugin_families()

    found: dict[str, type] = {}

    def walk(cls: type) -> None:
        for sub in cls.__subclasses__():
            path = f"{sub.__module__}.{sub.__qualname__}"
            if path in found:
                continue
            if sub.__module__.split(".")[0] in ("vtscore", "vtsearch"):
                found[path] = sub
            walk(sub)

    walk(PluginBase)
    return found


def _observed(cls: type) -> tuple[str, str | None, str | None, str | None]:
    def as_str(attr: str) -> str | None:
        value = getattr(cls, attr, None)
        return value if isinstance(value, str) else None

    return (_default_plugin_name(cls), as_str("name"), as_str("display_name"), as_str("icon"))


class TestDerivedNameGoldenList:
    def test_roster_matches(self):
        """Every shipped plugin has a golden row, and vice versa.

        A new plugin fails here until its row is added — which is the point:
        the row is where you notice that the class name you picked derives a
        different registry key than you expected.
        """
        actual = set(_in_tree_plugin_classes())
        expected = set(GOLDEN)
        assert actual - expected == set(), "plugin(s) missing a golden row"
        assert expected - actual == set(), "golden row(s) for plugin(s) that no longer exist"

    @pytest.mark.parametrize("path", sorted(GOLDEN))
    def test_metadata_unchanged(self, path):
        cls = _in_tree_plugin_classes()[path]
        assert _observed(cls) == GOLDEN[path]
