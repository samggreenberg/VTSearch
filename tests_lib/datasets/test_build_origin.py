"""Tests for the declarative ``DatasetImporter.build_origin`` framework.

These cover the four patterns the refactor consolidates:

- ``origin_suppressed`` short-circuits to empty params.
- ``PluginField.include_in_origin`` opts a field in/out.
- ``PluginField.origin_serializer`` customises the string form.
- ``extra_origin_keys`` copies non-PluginField keys into params,
  including the automatic ``source_specs`` injection for multi-media
  importers.

Plus the safety defaults: ``file`` and ``password`` field types are
excluded from origin by default, and explicit subclass overrides still
win (the backwards-compat shim).
"""

from __future__ import annotations

from typing import Any

from vtscore.datasets.importers.base import DatasetImporter, ImporterField


def _make_importer(
    *,
    name: str = "fake",
    fields: list[ImporterField] | None = None,
    multi_media: bool = False,
    extra_origin_keys: tuple[str, ...] = (),
    origin_suppressed: bool = False,
) -> DatasetImporter:
    """Construct a throwaway DatasetImporter for build_origin testing."""

    class _FakeImporter(DatasetImporter):
        pass

    _FakeImporter.name = name
    _FakeImporter.display_name = name
    _FakeImporter.description = ""
    _FakeImporter.fields = fields or []
    _FakeImporter.multi_media = multi_media
    _FakeImporter.extra_origin_keys = extra_origin_keys
    _FakeImporter.origin_suppressed = origin_suppressed
    return _FakeImporter()


class TestOriginSuppressed:
    def test_returns_empty_params_when_suppressed(self):
        imp = _make_importer(
            fields=[ImporterField("k", "K", "text")],
            origin_suppressed=True,
        )
        assert imp.build_origin({"k": "v"}) == {"importer": "fake", "params": {}}

    def test_default_is_not_suppressed(self):
        imp = _make_importer(fields=[ImporterField("k", "K", "text")])
        assert imp.build_origin({"k": "v"})["params"] == {"k": "v"}


class TestFieldTypeDefaults:
    def test_password_field_is_excluded_by_default(self):
        imp = _make_importer(
            fields=[
                ImporterField("user", "User", "text"),
                ImporterField("password", "Password", "password"),
            ],
        )
        params = imp.build_origin({"user": "alice", "password": "hunter2"})["params"]
        assert params == {"user": "alice"}

    def test_file_field_is_excluded_by_default(self):
        imp = _make_importer(
            fields=[
                ImporterField("name", "Name", "text"),
                ImporterField("upload", "Upload", "file"),
            ],
        )
        params = imp.build_origin({"name": "x", "upload": "/should/not/leak"})["params"]
        assert params == {"name": "x"}

    def test_include_in_origin_overrides_default(self):
        imp = _make_importer(
            fields=[
                ImporterField("debug_token", "Debug", "password", include_in_origin=True),
            ],
        )
        params = imp.build_origin({"debug_token": "ok-to-persist"})["params"]
        assert params == {"debug_token": "ok-to-persist"}

    def test_include_in_origin_can_omit_a_text_field(self):
        imp = _make_importer(
            fields=[
                ImporterField("keep", "Keep", "text"),
                ImporterField("drop", "Drop", "text", include_in_origin=False),
            ],
        )
        params = imp.build_origin({"keep": "a", "drop": "b"})["params"]
        assert params == {"keep": "a"}


class TestOriginSerializer:
    def test_list_field_is_comma_joined_via_serializer(self):
        imp = _make_importer(
            fields=[
                ImporterField(
                    "datasets",
                    "Datasets",
                    "text",
                    origin_serializer=lambda v: ",".join(v) if isinstance(v, list) else str(v),
                ),
            ],
        )
        params = imp.build_origin({"datasets": ["a.pkl", "b.pkl"]})["params"]
        assert params == {"datasets": "a.pkl,b.pkl"}

    def test_serializer_ignored_when_value_empty(self):
        imp = _make_importer(
            fields=[
                ImporterField("k", "K", "text", origin_serializer=lambda v: f"!{v}!"),
            ],
        )
        # Empty value is skipped before the serializer runs.
        assert imp.build_origin({"k": ""})["params"] == {}

    def test_default_lists_are_json_encoded(self):
        # Without an origin_serializer, list/dict values are JSON-encoded so
        # they round-trip through the string-only origin contract.
        imp = _make_importer(
            fields=[ImporterField("payload", "Payload", "text")],
        )
        params = imp.build_origin({"payload": [{"a": 1}]})["params"]
        assert params == {"payload": '[{"a": 1}]'}


class TestExtraOriginKeys:
    def test_extra_origin_keys_copy_from_field_values(self):
        imp = _make_importer(
            fields=[ImporterField("a", "A", "text")],
            extra_origin_keys=("transient",),
        )
        params = imp.build_origin({"a": "1", "transient": "abc"})["params"]
        assert params == {"a": "1", "transient": "abc"}

    def test_extra_origin_keys_json_encode_lists(self):
        imp = _make_importer(extra_origin_keys=("specs",))
        params = imp.build_origin({"specs": [{"x": 1}]})["params"]
        assert params == {"specs": '[{"x": 1}]'}

    def test_multi_media_auto_adds_source_specs(self):
        imp = _make_importer(multi_media=True)
        params = imp.build_origin({"source_specs": [{"source_type": "audio"}]})["params"]
        assert params == {"source_specs": '[{"source_type": "audio"}]'}

    def test_multi_media_keeps_existing_extra_keys(self):
        imp = _make_importer(
            multi_media=True,
            extra_origin_keys=("custom",),
        )
        params = imp.build_origin({"custom": "c", "source_specs": [{"source_type": "image"}]})["params"]
        assert params == {"custom": "c", "source_specs": '[{"source_type": "image"}]'}

    def test_multi_media_no_op_when_source_specs_empty(self):
        imp = _make_importer(multi_media=True)
        assert imp.build_origin({})["params"] == {}


class TestCheckboxStillSerialised:
    def test_checkbox_emits_true_false_strings(self):
        imp = _make_importer(
            fields=[
                ImporterField("on", "On", "checkbox", default="false"),
                ImporterField("off", "Off", "checkbox", default="false"),
            ],
        )
        params = imp.build_origin({"on": True, "off": False})["params"]
        assert params == {"on": "true", "off": "false"}

    def test_checkbox_default_when_missing(self):
        imp = _make_importer(
            fields=[ImporterField("on", "On", "checkbox", default="true")],
        )
        # Field absent from field_values; falls back to declared default.
        assert imp.build_origin({})["params"] == {"on": "true"}


class TestSubclassOverrideWins:
    """The backwards-compat shim: a third-party importer that still overrides
    ``build_origin`` keeps working unchanged."""

    def test_override_replaces_default_behavior(self):
        class _OverridingImporter(DatasetImporter):
            name = "overriding"
            display_name = "Overriding"
            description = ""
            fields = [
                ImporterField("password", "Password", "password"),
                ImporterField("user", "User", "text"),
            ]

            def build_origin(self, field_values: dict[str, Any]) -> dict[str, Any]:
                # Deliberately include the password; the override wins even
                # over the new "exclude password by default" rule.
                return {
                    "importer": self.name,
                    "params": {
                        "user": str(field_values.get("user", "")),
                        "password": str(field_values.get("password", "")),
                    },
                }

        imp = _OverridingImporter()
        params = imp.build_origin({"user": "alice", "password": "p"})["params"]
        assert params == {"user": "alice", "password": "p"}


class TestInTreeImportersAfterMigration:
    """Smoke tests confirming each in-tree importer's origin matches the
    pre-refactor expectations after deleting its override."""

    def test_server_folder_includes_source_specs_in_origin(self):
        from vtscore.datasets.importers.server_folder import IMPORTER

        origin = IMPORTER.build_origin(
            {
                "media_type": "audio",
                "path": "/data/x",
                "recursive": True,
                "source_specs": [{"source_type": "audio"}],
            }
        )
        assert origin["importer"] == "server_folder"
        assert origin["params"]["path"] == "/data/x"
        assert origin["params"]["media_type"] == "audio"
        assert origin["params"]["recursive"] == "true"
        assert origin["params"]["source_specs"] == '[{"source_type": "audio"}]'

    def test_http_archive_includes_source_specs_in_origin(self):
        from vtscore.datasets.importers.http_archive import IMPORTER

        origin = IMPORTER.build_origin(
            {
                "url": "https://example.com/a.zip",
                "media_type": "image",
                "source_specs": [{"source_type": "image"}],
            }
        )
        assert origin["importer"] == "http_archive"
        assert origin["params"]["url"] == "https://example.com/a.zip"
        assert origin["params"]["media_type"] == "image"
        assert origin["params"]["source_specs"] == '[{"source_type": "image"}]'

    def test_demo_omits_embedder_from_origin(self):
        from vtscore.datasets.importers.demo import IMPORTER

        origin = IMPORTER.build_origin({"name": "GTZAN", "embedder": "clap", "converter": "audio2image"})
        assert origin["importer"] == "demo"
        assert origin["params"] == {"name": "GTZAN", "converter": "audio2image"}

    def test_combine_datasets_comma_joins_list(self):
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        origin = IMPORTER.build_origin({"datasets": ["/a.pkl", "/b.pkl"], "name": "Combined"})
        assert origin["importer"] == "combine_datasets"
        # ``name`` is display-only; excluded from origin.
        assert origin["params"] == {"datasets": "/a.pkl,/b.pkl"}

    def test_combine_datasets_passes_through_string(self):
        from vtscore.datasets.importers.combine_datasets import IMPORTER

        origin = IMPORTER.build_origin({"datasets": "/a.pkl,/b.pkl"})
        assert origin["params"]["datasets"] == "/a.pkl,/b.pkl"

    def test_recaller_origin_is_empty(self):
        from vtscore.datasets.importers.recaller import IMPORTER

        origin = IMPORTER.build_origin({"query_id": "Q1", "media_type": "audio"})
        assert origin == {"importer": "recaller", "params": {}}

    def test_server_files_origin_includes_paths_file_and_media_type(self):
        from vtscore.datasets.importers.server_files import IMPORTER

        origin = IMPORTER.build_origin({"paths_file": "/tmp/list.txt", "media_type": "audio"})
        assert origin["importer"] == "server_files"
        assert origin["params"]["paths_file"] == "/tmp/list.txt"
        assert origin["params"]["media_type"] == "audio"
