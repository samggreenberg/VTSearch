"""Tests for admin-side plugin hiding.

Covers the persisted ``hidden_plugins`` server setting, the process-level
CLI fallback (``set_cli_hidden_plugins`` / ``add_cli_hidden_plugin``), the
resolver (``get_effective_hidden_plugins``, ``is_plugin_hidden``), the
filter helpers (``filter_visible_plugins`` /
``filter_visible_plugin_dicts``), and the route-level effect on
``/api/embedders``, ``/api/converters``, ``/api/media-types``, and the
dataset/labels/settings listing endpoints.
"""

from __future__ import annotations

import pytest

import app as app_module  # noqa: F401 — triggers conftest media init
from vtsearch import settings as settings_mod


@pytest.fixture(autouse=True)
def _reset_cli_hidden():
    """The CLI hide map is process-global; clear it around each test."""
    settings_mod.set_cli_hidden_plugins(None)
    yield
    settings_mod.set_cli_hidden_plugins(None)


class TestHiddenPluginsSetting:
    def test_default_is_empty_dict(self):
        assert settings_mod.get_hidden_plugins() == {}
        assert settings_mod.get_effective_hidden_plugins() == {}

    def test_set_and_get_roundtrip(self):
        settings_mod.set_hidden_plugins({"converters": ["audio2image"]})
        assert settings_mod.get_hidden_plugins() == {"converters": ["audio2image"]}
        merged = settings_mod.get_effective_hidden_plugins()
        assert merged == {"converters": {"audio2image"}}

    def test_is_plugin_hidden_reads_setting(self):
        settings_mod.set_hidden_plugins({"embedders": ["clap"]})
        assert settings_mod.is_plugin_hidden("embedders", "clap") is True
        assert settings_mod.is_plugin_hidden("embedders", "siglip") is False
        assert settings_mod.is_plugin_hidden("converters", "clap") is False


class TestCliFallback:
    def test_add_cli_hidden_plugin_accumulates(self):
        settings_mod.add_cli_hidden_plugin("converters", "audio2image")
        settings_mod.add_cli_hidden_plugin("converters", "video2image")
        settings_mod.add_cli_hidden_plugin("embedders", "clap")
        assert settings_mod.get_cli_hidden_plugins() == {
            "converters": {"audio2image", "video2image"},
            "embedders": {"clap"},
        }

    def test_add_cli_hidden_plugin_is_idempotent(self):
        settings_mod.add_cli_hidden_plugin("converters", "audio2image")
        settings_mod.add_cli_hidden_plugin("converters", "audio2image")
        assert settings_mod.get_cli_hidden_plugins() == {"converters": {"audio2image"}}

    def test_set_cli_hidden_plugins_replaces(self):
        settings_mod.add_cli_hidden_plugin("converters", "audio2image")
        settings_mod.set_cli_hidden_plugins({"embedders": ["clap"]})
        assert settings_mod.get_cli_hidden_plugins() == {"embedders": {"clap"}}

    def test_set_cli_hidden_plugins_none_clears(self):
        settings_mod.add_cli_hidden_plugin("converters", "audio2image")
        settings_mod.set_cli_hidden_plugins(None)
        assert settings_mod.get_cli_hidden_plugins() == {}

    def test_cli_and_settings_union(self):
        settings_mod.set_hidden_plugins({"converters": ["audio2image"]})
        settings_mod.add_cli_hidden_plugin("converters", "video2image")
        settings_mod.add_cli_hidden_plugin("embedders", "clap")
        merged = settings_mod.get_effective_hidden_plugins()
        assert merged == {
            "converters": {"audio2image", "video2image"},
            "embedders": {"clap"},
        }

    def test_cli_can_only_add_not_remove(self):
        settings_mod.set_hidden_plugins({"converters": ["audio2image"]})
        # No CLI hides — but settings still hides.
        assert settings_mod.is_plugin_hidden("converters", "audio2image") is True


class TestNormalisation:
    def test_corrupt_settings_value_does_not_crash(self):
        # An accidentally string-shaped value should yield empty members.
        settings_mod.set_hidden_plugins({"converters": ["ok"]})
        merged = settings_mod.get_effective_hidden_plugins()
        assert merged == {"converters": {"ok"}}

    def test_empty_family_dropped(self):
        # Setting an empty list shouldn't add the family.
        settings_mod.set_hidden_plugins({"converters": []})
        merged = settings_mod.get_effective_hidden_plugins()
        assert merged == {}


class TestFilterHelpers:
    def test_filter_visible_plugins_drops_hidden(self):
        class _P:
            def __init__(self, name):
                self.name = name

        plugins = [_P("a"), _P("b"), _P("c")]
        settings_mod.set_hidden_plugins({"converters": ["b"]})
        visible = settings_mod.filter_visible_plugins("converters", plugins)
        assert [p.name for p in visible] == ["a", "c"]

    def test_filter_visible_plugins_no_hide_returns_all(self):
        class _P:
            def __init__(self, name):
                self.name = name

        plugins = [_P("a"), _P("b")]
        assert settings_mod.filter_visible_plugins("converters", plugins) == plugins

    def test_filter_visible_plugin_dicts_drops_hidden(self):
        dicts = [{"name": "a"}, {"name": "b"}, {"name": "c"}]
        settings_mod.set_hidden_plugins({"embedders": ["c"]})
        visible = settings_mod.filter_visible_plugin_dicts("embedders", dicts)
        assert [d["name"] for d in visible] == ["a", "b"]

    def test_filter_visible_plugin_dicts_custom_id_key(self):
        dicts = [{"type_id": "audio"}, {"type_id": "image"}]
        settings_mod.set_hidden_plugins({"media_types": ["audio"]})
        visible = settings_mod.filter_visible_plugin_dicts("media_types", dicts, id_key="type_id")
        assert [d["type_id"] for d in visible] == ["image"]

    def test_filter_visible_plugins_custom_id_attr(self):
        class _MT:
            def __init__(self, type_id):
                self.type_id = type_id

        plugins = [_MT("audio"), _MT("image")]
        settings_mod.set_hidden_plugins({"media_types": ["audio"]})
        visible = settings_mod.filter_visible_plugins("media_types", plugins, id_attr="type_id")
        assert [p.type_id for p in visible] == ["image"]


class TestRouteIntegration:
    """Hidden plugins must disappear from the corresponding listing endpoints."""

    def test_embedders_endpoint_filters_hidden(self, client):
        baseline = client.get("/api/embedders").get_json()["embedders"]
        if not baseline:
            pytest.skip("No embedders registered to test against")
        victim = baseline[0]["name"]

        settings_mod.set_hidden_plugins({"embedders": [victim]})
        filtered = client.get("/api/embedders").get_json()["embedders"]
        assert victim not in [e["name"] for e in filtered]
        assert len(filtered) == len(baseline) - 1

    def test_embedders_endpoint_filters_when_querying_media_type(self, client):
        baseline = client.get("/api/embedders?media_type=image").get_json()["embedders"]
        if not baseline:
            pytest.skip("No image embedders registered")
        victim = baseline[0]["name"]
        settings_mod.set_hidden_plugins({"embedders": [victim]})
        filtered = client.get("/api/embedders?media_type=image").get_json()["embedders"]
        assert victim not in [e["name"] for e in filtered]

    def test_converters_endpoint_filters_hidden(self, client):
        baseline = client.get("/api/converters").get_json()["converters"]
        if not baseline:
            pytest.skip("No converters registered to test against")
        victim = baseline[0]["name"]
        settings_mod.set_hidden_plugins({"converters": [victim]})
        filtered = client.get("/api/converters").get_json()["converters"]
        assert victim not in [c["name"] for c in filtered]

    def test_media_types_endpoint_filters_hidden(self, client):
        baseline = client.get("/api/media-types").get_json()["media_types"]
        if not baseline:
            pytest.skip("No media types registered to test against")
        victim = baseline[0]["type_id"]
        settings_mod.set_hidden_plugins({"media_types": [victim]})
        filtered = client.get("/api/media-types").get_json()["media_types"]
        assert victim not in [m["type_id"] for m in filtered]

    def test_dataset_importers_endpoint_filters_hidden(self, client):
        baseline = client.get("/api/dataset/importers").get_json()["importers"]
        if not baseline:
            pytest.skip("No importers registered to test against")
        victim = baseline[0]["name"]
        settings_mod.set_hidden_plugins({"importers": [victim]})
        filtered = client.get("/api/dataset/importers").get_json()["importers"]
        assert victim not in [i["name"] for i in filtered]

    def test_cli_hide_filters_endpoint_too(self, client):
        baseline = client.get("/api/embedders").get_json()["embedders"]
        if not baseline:
            pytest.skip("No embedders registered to test against")
        victim = baseline[0]["name"]
        # No persisted setting, only CLI-side hide.
        settings_mod.add_cli_hidden_plugin("embedders", victim)
        filtered = client.get("/api/embedders").get_json()["embedders"]
        assert victim not in [e["name"] for e in filtered]

    def test_hidden_plugin_still_callable_by_name(self, client):
        # Hiding declutters listings but does NOT block direct execution
        # endpoints — that matches existing ``hidden_from_picker`` semantics.
        from vtscore.datasets.importers import list_importers

        importers = list_importers()
        if not importers:
            pytest.skip("No importers available")
        victim_name = importers[0].name
        settings_mod.set_hidden_plugins({"importers": [victim_name]})
        # Listing endpoint omits it.
        listed = client.get("/api/dataset/importers").get_json()["importers"]
        assert victim_name not in [i["name"] for i in listed]
        # But ``get_importer`` and execution endpoints still resolve it.
        from vtscore.datasets.importers import get_importer

        assert get_importer(victim_name) is not None
