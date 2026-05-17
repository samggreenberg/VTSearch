"""Tests for dynamic-options dataset importer fields.

Covers:

- :class:`~vtsearch.plugins.PluginField` ``dynamic_options`` /
  ``depends_on`` attributes serialise via ``to_dict``.
- :meth:`DatasetImporter.get_field_options` default raises
  ``NotImplementedError``.
- The ``POST /api/dataset/import/<name>/options`` route returns the
  importer's options, surfaces ``NotImplementedError`` as 501, and
  surfaces other plugin exceptions as 502 with the message body.
- The ReCaller scaffold declares its ``query_id`` field as dynamic and
  routes ``get_field_options`` through ``_rc_list_queries``.
"""

from __future__ import annotations

import pytest

from vtsearch.datasets.importers.base import DatasetImporter, ImporterField
from vtsearch.plugins import PluginField


class _StubImporter(DatasetImporter):
    name = "_stub_dynamic"
    display_name = "Stub Dynamic"
    description = "Test importer with dynamic-options field."
    fields = [
        ImporterField("media_type", "Media Type", "select", options=["audio", "image"], default="audio"),
        ImporterField(
            "query_id",
            "Query",
            "select",
            dynamic_options=True,
            depends_on=["media_type"],
        ),
    ]

    def __init__(self, fail_with: Exception | None = None) -> None:
        super().__init__()
        self._fail_with = fail_with

    def get_field_options(self, field_key, current_values):
        if self._fail_with is not None:
            raise self._fail_with
        if field_key == "query_id":
            mt = current_values.get("media_type", "audio")
            return [f"{mt}-q1", f"{mt}-q2"]
        return super().get_field_options(field_key, current_values)

    def run(self, field_values, medias, thin=False):  # pragma: no cover — unused here
        return None


@pytest.fixture
def registered_stub():
    """Register a stub importer in the live registry for the duration of a test."""
    from vtsearch.datasets.importers import get_importer

    registry = get_importer.__self__
    registry._ensure_discovered()

    stub = _StubImporter()
    registry._items[stub.name] = stub
    try:
        yield stub
    finally:
        registry._items.pop(stub.name, None)


# ---------------------------------------------------------------------------
# PluginField — dataclass attributes & serialisation
# ---------------------------------------------------------------------------


class TestPluginFieldDynamicAttrs:
    def test_defaults(self):
        f = PluginField(key="x", label="X", field_type="text")
        assert f.dynamic_options is False
        assert f.depends_on == []

    def test_to_dict_round_trip(self):
        f = PluginField(
            key="q",
            label="Q",
            field_type="select",
            dynamic_options=True,
            depends_on=["a", "b"],
        )
        d = f.to_dict()
        assert d["dynamic_options"] is True
        assert d["depends_on"] == ["a", "b"]

    def test_depends_on_is_independent_per_instance(self):
        # default_factory=list should not be shared between instances.
        a = PluginField(key="a", label="A", field_type="text")
        b = PluginField(key="b", label="B", field_type="text")
        a.depends_on.append("oops")
        assert b.depends_on == []


# ---------------------------------------------------------------------------
# DatasetImporter.get_field_options default behaviour
# ---------------------------------------------------------------------------


class TestGetFieldOptionsDefault:
    def test_default_raises_not_implemented(self):
        class Imp(DatasetImporter):
            name = "_t_default"
            display_name = "T"
            description = "T"
            fields = []

            def run(self, field_values, medias, thin=False):
                return None

        with pytest.raises(NotImplementedError):
            Imp().get_field_options("anything", {})


# ---------------------------------------------------------------------------
# /api/dataset/import/<name>/options route
# ---------------------------------------------------------------------------


class TestImporterFieldOptionsRoute:
    URL_TEMPLATE = "/api/dataset/import/{name}/options"

    def test_returns_options_from_importer(self, client, registered_stub):
        resp = client.post(
            self.URL_TEMPLATE.format(name=registered_stub.name),
            json={"field_key": "query_id", "values": {"media_type": "image"}},
        )
        assert resp.status_code == 200
        body = resp.get_json()
        assert body == {"options": ["image-q1", "image-q2"]}

    def test_unknown_importer_returns_404(self, client):
        resp = client.post(
            self.URL_TEMPLATE.format(name="does_not_exist"),
            json={"field_key": "query_id", "values": {}},
        )
        assert resp.status_code == 404

    def test_unknown_field_returns_400(self, client, registered_stub):
        resp = client.post(
            self.URL_TEMPLATE.format(name=registered_stub.name),
            json={"field_key": "no_such_field", "values": {}},
        )
        assert resp.status_code == 400
        # flask-smorest error envelope: handler-level rejects live under ``message``.
        assert "Unknown field" in resp.get_json()["message"]

    def test_static_field_returns_400(self, client, registered_stub):
        # ``media_type`` is static, not dynamic — calling options on it is rejected.
        resp = client.post(
            self.URL_TEMPLATE.format(name=registered_stub.name),
            json={"field_key": "media_type", "values": {}},
        )
        assert resp.status_code == 400
        assert "not dynamic" in resp.get_json()["message"]

    def test_missing_field_key_returns_400(self, client, registered_stub):
        # Schema-level validation (required ``field_key``) → 422.
        resp = client.post(
            self.URL_TEMPLATE.format(name=registered_stub.name),
            json={"values": {}},
        )
        assert resp.status_code == 422

    def test_non_object_values_returns_400(self, client, registered_stub):
        # Schema-level validation (``values`` must be a dict) → 422.
        resp = client.post(
            self.URL_TEMPLATE.format(name=registered_stub.name),
            json={"field_key": "query_id", "values": "not-an-object"},
        )
        assert resp.status_code == 422

    def test_not_implemented_returns_501(self, client):
        from vtsearch.datasets.importers import get_importer

        registry = get_importer.__self__
        registry._ensure_discovered()
        stub = _StubImporter(fail_with=NotImplementedError("nope"))
        registry._items[stub.name] = stub
        try:
            resp = client.post(
                self.URL_TEMPLATE.format(name=stub.name),
                json={"field_key": "query_id", "values": {}},
            )
            assert resp.status_code == 501
            assert resp.get_json()["message"] == "nope"
        finally:
            registry._items.pop(stub.name, None)

    def test_plugin_exception_returns_502(self, client):
        from vtsearch.datasets.importers import get_importer

        registry = get_importer.__self__
        registry._ensure_discovered()
        stub = _StubImporter(fail_with=RuntimeError("remote service down"))
        registry._items[stub.name] = stub
        try:
            resp = client.post(
                self.URL_TEMPLATE.format(name=stub.name),
                json={"field_key": "query_id", "values": {}},
            )
            assert resp.status_code == 502
            assert resp.get_json()["message"] == "remote service down"
        finally:
            registry._items.pop(stub.name, None)


# ---------------------------------------------------------------------------
# Importer metadata round-trips through to_dict() so the frontend sees it
# ---------------------------------------------------------------------------


class TestImporterMetadataExposes:
    def test_to_dict_includes_dynamic_field_props(self):
        stub = _StubImporter()
        d = stub.to_dict()
        query_field = next(f for f in d["fields"] if f["key"] == "query_id")
        assert query_field["dynamic_options"] is True
        assert query_field["depends_on"] == ["media_type"]

    def test_static_field_has_default_dynamic_props(self):
        stub = _StubImporter()
        d = stub.to_dict()
        media_type_field = next(f for f in d["fields"] if f["key"] == "media_type")
        assert media_type_field["dynamic_options"] is False
        assert media_type_field["depends_on"] == []


# ---------------------------------------------------------------------------
# ReCaller scaffold — exercises the importer-level wiring
# ---------------------------------------------------------------------------


class TestReCallerDynamicQueryId:
    def test_query_id_field_is_dynamic_with_media_type_dep(self):
        from vtsearch.datasets.importers import get_importer

        rc = get_importer("recaller")
        assert rc is not None
        query_field = next(f for f in rc.fields if f.key == "query_id")
        assert query_field.dynamic_options is True
        assert query_field.depends_on == ["media_type"]

    def test_get_field_options_routes_through_rc_list_queries(self, monkeypatch):
        from vtsearch.datasets.importers import get_importer
        from vtsearch.datasets.importers import recaller as rc_module

        rc = get_importer("recaller")
        captured: list[str] = []

        def fake_list(media_type: str) -> list[str]:
            captured.append(media_type)
            return ["q-aaa", "q-bbb"]

        monkeypatch.setattr(rc_module, "_rc_list_queries", fake_list)

        out = rc.get_field_options("query_id", {"media_type": "image"})
        assert out == ["q-aaa", "q-bbb"]
        assert captured == ["image"]

    def test_get_field_options_default_media_type(self, monkeypatch):
        from vtsearch.datasets.importers import get_importer
        from vtsearch.datasets.importers import recaller as rc_module

        rc = get_importer("recaller")
        seen: list[str] = []
        monkeypatch.setattr(rc_module, "_rc_list_queries", lambda mt: seen.append(mt) or [])

        # Empty current_values falls back to "audio".
        rc.get_field_options("query_id", {})
        assert seen == ["audio"]

    def test_get_field_options_unknown_key_raises(self):
        from vtsearch.datasets.importers import get_importer

        rc = get_importer("recaller")
        with pytest.raises(NotImplementedError):
            rc.get_field_options("not_a_field", {})
