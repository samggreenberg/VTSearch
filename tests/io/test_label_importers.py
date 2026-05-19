"""Tests for the Label Importer abstraction.

Covers:
- LabelImporterField and LabelImporter base classes
- Auto-discovery registry (list_label_importers, get_label_importer)
- GET /api/label-importers endpoint
- Built-in importers: server_json_file, server_csv_file
"""

from __future__ import annotations

import json

import pytest

import app as app_module


# ---------------------------------------------------------------------------
# LabelImporterField
# ---------------------------------------------------------------------------


class TestLabelImporterField:
    def test_to_dict_contains_required_keys(self):
        from vtscore.labels.importers.base import LabelImporterField

        f = LabelImporterField(key="file", label="My File", field_type="file")
        d = f.to_dict()
        assert d["key"] == "file"
        assert d["label"] == "My File"
        assert d["field_type"] == "file"
        assert "description" in d
        assert "accept" in d
        assert "options" in d
        assert "default" in d
        assert "required" in d
        assert "placeholder" in d

    def test_defaults(self):
        from vtscore.labels.importers.base import LabelImporterField

        f = LabelImporterField(key="x", label="X", field_type="text")
        assert f.required is True
        assert f.default == ""
        assert f.placeholder == ""
        assert f.options == []
        assert f.description == ""
        assert f.accept == ""

    def test_custom_values(self):
        from vtscore.labels.importers.base import LabelImporterField

        f = LabelImporterField(
            key="mode",
            label="Mode",
            field_type="select",
            options=["a", "b"],
            default="a",
            required=False,
            description="Pick one",
            placeholder="Choose…",
        )
        d = f.to_dict()
        assert d["options"] == ["a", "b"]
        assert d["default"] == "a"
        assert d["required"] is False


# ---------------------------------------------------------------------------
# LabelImporter base class
# ---------------------------------------------------------------------------


class TestLabelImporterBase:
    def _make_minimal(self):
        from vtscore.labels.importers.base import LabelImporter

        class Minimal(LabelImporter):
            name = "minimal"
            display_name = "Minimal"
            description = "A minimal label importer."
            fields = []

            def run(self, field_values):
                return []

        return Minimal()

    def test_run_raises_not_implemented_when_not_overridden(self):
        from vtscore.labels.importers.base import LabelImporter

        imp = LabelImporter()
        with pytest.raises(NotImplementedError):
            imp.run({})

    def test_to_dict_contains_standard_keys(self):
        imp = self._make_minimal()
        d = imp.to_dict()
        assert d["name"] == "minimal"
        assert d["display_name"] == "Minimal"
        assert d["description"] == "A minimal label importer."
        assert "icon" in d
        assert "fields" in d

    def test_default_icon(self):
        from vtscore.labels.importers.base import LabelImporter

        assert LabelImporter.icon == "🏷️"

    def test_custom_icon_in_to_dict(self):
        from vtscore.labels.importers.base import LabelImporter

        class Custom(LabelImporter):
            name = "c"
            display_name = "C"
            description = "C"
            icon = "🔖"
            fields = []

            def run(self, field_values):
                return []

        assert Custom().to_dict()["icon"] == "🔖"

    def test_validate_cli_field_values_raises_on_missing_required(self):
        from vtscore.labels.importers.base import LabelImporter, LabelImporterField

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [LabelImporterField("filepath", "File", "text", required=True)]

            def run(self, field_values):
                return []

        imp = Imp()
        with pytest.raises(ValueError, match="--filepath"):
            imp.validate_cli_field_values({})

    def test_validate_cli_field_values_passes_when_provided(self):
        from vtscore.labels.importers.base import LabelImporter, LabelImporterField

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [LabelImporterField("filepath", "File", "text", required=True)]

            def run(self, field_values):
                return []

        imp = Imp()
        imp.validate_cli_field_values({"filepath": "/some/path"})  # no raise

    def test_run_cli_delegates_to_run(self):
        from vtscore.labels.importers.base import LabelImporter

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values):
                return [{"md5": "abc", "label": "good"}]

        imp = Imp()
        result = imp.run_cli({})
        assert result == [{"md5": "abc", "label": "good"}]

    def test_add_cli_arguments_adds_text_field(self):
        import argparse

        from vtscore.labels.importers.base import LabelImporter, LabelImporterField

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [LabelImporterField("server", "Server", "text", description="DB host")]

            def run(self, field_values):
                return []

        parser = argparse.ArgumentParser()
        Imp().add_cli_arguments(parser)
        args = parser.parse_args(["--server", "localhost"])
        assert args.server == "localhost"

    def test_add_cli_arguments_select_adds_choices(self):
        import argparse

        from vtscore.labels.importers.base import LabelImporter, LabelImporterField

        class Imp(LabelImporter):
            name = "t"
            display_name = "T"
            description = "T"
            fields = [LabelImporterField("mode", "Mode", "select", options=["a", "b"], default="a")]

            def run(self, field_values):
                return []

        parser = argparse.ArgumentParser()
        Imp().add_cli_arguments(parser)
        args = parser.parse_args([])  # uses default
        assert args.mode == "a"
        with pytest.raises(SystemExit):
            parser.parse_args(["--mode", "invalid"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestLabelImporterRegistry:
    def test_list_label_importers_returns_builtins(self):
        from vtscore.labels.importers import list_label_importers

        names = {imp.name for imp in list_label_importers()}
        assert "server_json_file" in names
        assert "server_csv_file" in names

    def test_get_label_importer_known(self):
        from vtscore.labels.importers import get_label_importer

        for name in ("server_json_file", "server_csv_file"):
            imp = get_label_importer(name)
            assert imp is not None, f"Label importer '{name}' not found"
            assert imp.name == name

    def test_get_label_importer_unknown_returns_none(self):
        from vtscore.labels.importers import get_label_importer

        assert get_label_importer("no_such_importer") is None

    def test_each_importer_has_display_name_and_icon(self):
        from vtscore.labels.importers import list_label_importers

        for imp in list_label_importers():
            assert imp.display_name, f"{imp.name} missing display_name"
            assert imp.icon, f"{imp.name} missing icon"
            assert imp.description, f"{imp.name} missing description"

    def test_each_importer_fields_are_valid(self):
        from vtscore.labels.importers import list_label_importers

        valid_types = ("file", "text", "password", "number", "select", "server_path", "url", "email")
        for imp in list_label_importers():
            for f in imp.fields:
                assert f.key, f"{imp.name} has a field without a key"
                assert f.label, f"{imp.name} field '{f.key}' has no label"
                assert f.field_type in valid_types, f"{imp.name} field '{f.key}' has unknown type '{f.field_type}'"


# ---------------------------------------------------------------------------
# API – GET /api/label-importers
# ---------------------------------------------------------------------------


class TestGetLabelImportersEndpoint:
    def test_returns_200(self, client):
        res = client.get("/api/label-importers")
        assert res.status_code == 200

    def test_returns_list(self, client):
        res = client.get("/api/label-importers")
        data = res.get_json()
        assert isinstance(data, list)

    def test_contains_builtin_importers(self, client):
        res = client.get("/api/label-importers")
        names = {entry["name"] for entry in res.get_json()}
        assert "server_json_file" in names
        assert "server_csv_file" in names

    def test_each_entry_has_required_keys(self, client):
        res = client.get("/api/label-importers")
        for entry in res.get_json():
            assert "name" in entry
            assert "display_name" in entry
            assert "description" in entry
            assert "icon" in entry
            assert "fields" in entry


# ---------------------------------------------------------------------------
# Server JSON file importer
# ---------------------------------------------------------------------------


class TestServerJsonLabelImporter:
    def _get_importer(self):
        from vtscore.labels.importers.server_json_file import LABEL_IMPORTER

        return LABEL_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "server_json_file"

    def test_display_name(self):
        assert "server" in self._get_importer().display_name.lower()

    def test_icon(self):
        assert self._get_importer().icon

    def test_has_filepath_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "filepath" in fields
        assert fields["filepath"].field_type == "server_path"

    def test_run_reads_server_file(self, tmp_path):
        payload = {"labels": [{"md5": "abc", "label": "good"}, {"md5": "def", "label": "bad"}]}
        p = tmp_path / "labels.json"
        p.write_text(json.dumps(payload))
        result = self._get_importer().run({"filepath": str(p)})
        assert len(result) == 2
        assert result[0]["md5"] == "abc"

    def test_run_raises_on_missing_file(self):
        with pytest.raises(ValueError, match="not found"):
            self._get_importer().run({"filepath": "/nonexistent/path.json"})

    def test_run_raises_on_empty_filepath(self):
        with pytest.raises(ValueError, match="file path"):
            self._get_importer().run({"filepath": ""})

    def test_run_raises_on_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json at all")
        with pytest.raises(ValueError, match="JSON"):
            self._get_importer().run({"filepath": str(p)})

    def test_run_raises_on_missing_labels_key(self, tmp_path):
        p = tmp_path / "no_labels.json"
        p.write_text(json.dumps({"data": []}))
        with pytest.raises(ValueError, match="labels"):
            self._get_importer().run({"filepath": str(p)})

    def test_run_cli_delegates_to_run(self, tmp_path):
        payload = {"labels": [{"md5": "xyz", "label": "bad"}]}
        p = tmp_path / "labels.json"
        p.write_text(json.dumps(payload))
        result = self._get_importer().run_cli({"filepath": str(p)})
        assert len(result) == 1
        assert result[0]["md5"] == "xyz"

    def test_api_route(self, client, tmp_path):
        md5 = app_module.medias[1]["md5"]
        payload = {"labels": [{"md5": md5, "label": "good"}]}
        p = tmp_path / "server_labels.json"
        p.write_text(json.dumps(payload))
        res = client.post(
            "/api/label-importers/import/server_json_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 1
        assert 1 in app_module.good_votes


# ---------------------------------------------------------------------------
# Server CSV file importer
# ---------------------------------------------------------------------------


class TestServerCsvLabelImporter:
    def _get_importer(self):
        from vtscore.labels.importers.server_csv_file import LABEL_IMPORTER

        return LABEL_IMPORTER

    def test_name(self):
        assert self._get_importer().name == "server_csv_file"

    def test_display_name(self):
        assert "server" in self._get_importer().display_name.lower()

    def test_has_filepath_field(self):
        fields = {f.key: f for f in self._get_importer().fields}
        assert "filepath" in fields
        assert fields["filepath"].field_type == "server_path"

    def test_run_reads_server_file(self, tmp_path):
        p = tmp_path / "labels.csv"
        p.write_text("md5,label\nabc123,good\ndef456,bad\n")
        result = self._get_importer().run({"filepath": str(p)})
        assert len(result) == 2
        labels = {r["md5"]: r["label"] for r in result}
        assert labels["abc123"] == "good"
        assert labels["def456"] == "bad"

    def test_run_raises_on_missing_file(self):
        with pytest.raises(ValueError, match="not found"):
            self._get_importer().run({"filepath": "/nonexistent/path.csv"})

    def test_run_raises_on_empty_filepath(self):
        with pytest.raises(ValueError, match="file path"):
            self._get_importer().run({"filepath": ""})

    def test_run_raises_on_missing_columns(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("hash,category\nabc,good\n")
        with pytest.raises(ValueError, match="md5"):
            self._get_importer().run({"filepath": str(p)})

    def test_run_cli_delegates_to_run(self, tmp_path):
        p = tmp_path / "labels.csv"
        p.write_text("md5,label\nxyz789,bad\n")
        result = self._get_importer().run_cli({"filepath": str(p)})
        assert len(result) == 1
        assert result[0]["md5"] == "xyz789"

    def test_api_route(self, client, tmp_path):
        md5_1 = app_module.medias[1]["md5"]
        md5_2 = app_module.medias[2]["md5"]
        p = tmp_path / "server_labels.csv"
        p.write_text(f"md5,label\n{md5_1},good\n{md5_2},bad\n")
        res = client.post(
            "/api/label-importers/import/server_csv_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 2
        assert 1 in app_module.good_votes
        assert 2 in app_module.bad_votes

    def test_csv_importer_preserves_origin_name(self, tmp_path):
        """CSV importer reads origin_name/filename/category columns."""
        from vtscore.labels.importers.server_csv_file import _parse_csv_bytes

        csv_text = "label,md5,origin_name,filename,category\ngood,abc123,clip.wav,clip.wav,music\n"
        result = _parse_csv_bytes(csv_text.encode())
        assert len(result) == 1
        assert result[0]["origin_name"] == "clip.wav"
        assert result[0]["filename"] == "clip.wav"
        assert result[0]["category"] == "music"

    def test_csv_importer_parses_origin_json(self, tmp_path):
        """CSV importer parses a JSON-serialised origin dict column."""
        import json

        from vtscore.labels.importers.server_csv_file import _parse_csv_bytes

        origin = {"importer": "demo", "params": {"name": "flowers102"}}
        origin_json = json.dumps(origin, sort_keys=True)
        csv_text = (
            f'label,md5,origin_name,origin\ngood,abc123,rose.jpg,"{origin_json.replace(chr(34), chr(34) + chr(34))}"\n'
        )
        result = _parse_csv_bytes(csv_text.encode())
        assert len(result) == 1
        assert result[0]["origin"] == origin

    def test_csv_importer_ignores_invalid_origin_json(self, tmp_path):
        """Non-JSON origin column values are silently ignored."""
        from vtscore.labels.importers.server_csv_file import _parse_csv_bytes

        csv_text = "label,md5,origin_name,origin\ngood,abc123,rose.jpg,not-json\n"
        result = _parse_csv_bytes(csv_text.encode())
        assert len(result) == 1
        assert "origin" not in result[0]

    def test_csv_with_unmatched_md5_does_not_silently_match_by_basename(self, client, tmp_path):
        """A CSV row whose md5 doesn't match anything in the current dataset
        must NOT silently fall back to matching by origin_name (basename).

        Otherwise a CSV produced from one dataset would silently mislabel any
        same-named file in another dataset, even when the underlying content
        differs.  Such rows should be reported as missing so the user can
        ingest the original media (via the resolver / ingest-missing flow)
        and have it matched by content hash.
        """
        origin_name_1 = app_module.medias[1].get("origin_name", "")
        origin_name_2 = app_module.medias[2].get("origin_name", "")
        assert origin_name_1, "Test media 1 must have an origin_name"
        csv_text = f"label,md5,origin_name\ngood,wrong_md5_1,{origin_name_1}\nbad,wrong_md5_2,{origin_name_2}\n"
        p = tmp_path / "cross_dataset.csv"
        p.write_text(csv_text)
        res = client.post(
            "/api/label-importers/import/server_csv_file",
            json={"filepath": str(p)},
        )
        assert res.status_code == 200
        result = res.get_json()
        assert result["applied"] == 0, "no labels should be applied via basename collision"
        assert 1 not in app_module.good_votes
        assert 2 not in app_module.bad_votes
        # The rows are reported as missing (subject to ingest-missing recovery).
        assert result["missing_count"] >= 0
