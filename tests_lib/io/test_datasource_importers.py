"""Library-tier tests for the DataSource Importer plugin family."""

import pytest

from vtscore.datasource_importers import (
    DataSourceImporter,
    FetchedMediaItem,
    get_datasource_importer,
    list_datasource_importers,
)
from vtscore.plugins import PluginField


class TestDatasourceImporterRegistry:
    def test_builtins_discovered(self):
        names = [imp.name for imp in list_datasource_importers()]
        assert "server_file" in names
        assert "url_download" in names

    def test_get_by_name(self):
        imp = get_datasource_importer("server_file")
        assert imp is not None
        assert imp.display_name == "Server File"

    def test_get_unknown_returns_none(self):
        assert get_datasource_importer("nope_not_real") is None


class TestDatasourceImporterMetadata:
    def test_to_dict_includes_category_and_fields(self):
        imp = get_datasource_importer("url_download")
        d = imp.to_dict()
        assert d["category"] == "services"
        assert d["name"] == "url_download"
        assert [f["key"] for f in d["fields"]] == ["url"]
        assert d["fields"][0]["field_type"] == "url"

    def test_family_suffix_stripped_from_auto_name(self):
        class MyShinyDataSourceImporter(DataSourceImporter):
            """Fetches shiny things."""

            fields = []

            def fetch(self, field_values):  # pragma: no cover - never called
                raise NotImplementedError

        imp = MyShinyDataSourceImporter()
        assert imp.name == "my_shiny"
        assert imp.display_name == "My Shiny"
        assert imp.description == "Fetches shiny things."

    def test_family_base_keeps_stock_icon_and_no_derived_name(self):
        # The abstract base must not have auto-derived metadata stamped on
        # it (that would leak down the MRO to every concrete subclass).
        assert "name" not in DataSourceImporter.__dict__
        assert DataSourceImporter.icon == "\U0001f4e5"

    def test_default_category_is_services(self):
        class ThingDataSourceImporter(DataSourceImporter):
            fields = []

            def fetch(self, field_values):  # pragma: no cover - never called
                raise NotImplementedError

        assert ThingDataSourceImporter().category == "services"

    def test_get_field_options_default_raises(self):
        imp = get_datasource_importer("server_file")
        with pytest.raises(NotImplementedError):
            imp.get_field_options("path", {})


class TestServerFileFetch:
    def test_fetch_reads_bytes_and_filename(self, tmp_path):
        f = tmp_path / "clip.wav"
        f.write_bytes(b"RIFFxxxxWAVE")
        item = get_datasource_importer("server_file").fetch({"path": str(f)})
        assert isinstance(item, FetchedMediaItem)
        assert item.data == b"RIFFxxxxWAVE"
        assert item.filename == "clip.wav"

    def test_fetch_missing_file_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="File not found"):
            get_datasource_importer("server_file").fetch({"path": str(tmp_path / "nope.wav")})

    def test_fetch_empty_path_raises_value_error(self):
        with pytest.raises(ValueError, match="required"):
            get_datasource_importer("server_file").fetch({"path": "   "})


class TestUrlDownloadFetch:
    def test_private_url_rejected_without_network(self):
        with pytest.raises(ValueError, match="private/internal"):
            get_datasource_importer("url_download").fetch({"url": "http://127.0.0.1/x.wav"})

    def test_empty_url_raises_value_error(self):
        with pytest.raises(ValueError, match="required"):
            get_datasource_importer("url_download").fetch({"url": ""})

    def test_filename_derived_from_url_path(self):
        from vtscore.datasource_importers.url_download import _filename_from_url

        assert _filename_from_url("https://x.test/a/b/dog%20bark.wav?tok=1") == "dog bark.wav"
        assert _filename_from_url("https://x.test/") == "download.bin"


class TestDatasourceImporterFields:
    def test_fields_are_plugin_fields(self):
        for imp in list_datasource_importers():
            for f in imp.fields:
                assert isinstance(f, PluginField)

    def test_inventory_includes_family(self):
        from vtscore.plugins.inventory import gather_plugins

        inventory = gather_plugins()
        assert "datasource_importers" in inventory
        names = [e.name for e in inventory["datasource_importers"]]
        assert "server_file" in names and "url_download" in names
