"""Tests for ``importlib.metadata`` entry-point discovery in :class:`PluginRegistry`.

Third-party packages can register a plugin by declaring an entry point in
the ``vtsearch.<family>`` group; this test stubs ``importlib.metadata``
and verifies the registry picks up entry-point plugins, surfaces failures
via warnings, and never lets an entry point silently shadow a built-in.
"""

from __future__ import annotations

import importlib.metadata
import warnings

import pytest

from vtscore.plugins import PluginRegistry


class _FakeEntryPoint:
    """Minimal stand-in for :class:`importlib.metadata.EntryPoint`."""

    def __init__(self, name: str, value: str, target) -> None:
        self.name = name
        self.value = value
        self.group = "vtsearch.test_family"
        self._target = target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


class _DummyPlugin:
    def __init__(self, name: str) -> None:
        self.name = name
        self.display_name = name.title()
        self.description = "test"


def _patch_entry_points(monkeypatch, entries: list[_FakeEntryPoint]) -> None:
    def fake_entry_points(*, group: str):
        return [e for e in entries if e.group == group]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


class TestEntryPointDiscovery:
    def test_loads_entry_point_plugin(self, monkeypatch):
        plugin = _DummyPlugin("from_entry_point")
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("my_plugin", "pkg.mod:OBJ", plugin)])

        registry: PluginRegistry = PluginRegistry(
            # Use an existing package whose own scan turns up nothing matching
            # the test sentinel, so the only registration comes from the entry
            # point.
            package="vtscore.plugins",
            sentinel="TEST_SENTINEL_NEVER_PRESENT",
            label="test plugin",
            entry_point_group="vtsearch.test_family",
        )
        assert registry.get("from_entry_point") is plugin
        assert plugin in registry.list()

    def test_built_in_wins_over_entry_point_name_clash(self, monkeypatch):
        from vtscore.datasets.importers import list_importers

        # server_folder is an established built-in importer.
        built_in_names = {imp.name for imp in list_importers()}
        assert "server_folder" in built_in_names

        rogue = _DummyPlugin("server_folder")
        _patch_entry_points(monkeypatch, [_FakeEntryPoint("rogue", "rogue:OBJ", rogue)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry: PluginRegistry = PluginRegistry(
                package="vtscore.datasets.importers",
                sentinel="IMPORTER",
                label="dataset importer",
                entry_point_group="vtsearch.test_family",
            )
            entry = registry.get("server_folder")
        # The built-in is returned, not the rogue stub.
        assert entry is not rogue
        # And a warning was emitted explaining the skip.
        assert any("clashes with built-in" in str(w.message) for w in caught)

    def test_failed_load_warns_and_continues(self, monkeypatch):
        good = _DummyPlugin("good_plugin")
        boom = _FakeEntryPoint("broken", "pkg:OBJ", RuntimeError("nope"))
        ok = _FakeEntryPoint("good", "pkg:OBJ", good)
        _patch_entry_points(monkeypatch, [boom, ok])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry: PluginRegistry = PluginRegistry(
                package="vtscore.plugins",
                sentinel="TEST_SENTINEL_NEVER_PRESENT",
                label="test plugin",
                entry_point_group="vtsearch.test_family",
            )
            assert registry.get("good_plugin") is good
        assert registry.get("broken") is None
        # A warning was raised for the failure.
        assert any("Failed to load" in str(w.message) for w in caught)

    def test_entry_point_without_name_is_skipped(self, monkeypatch):
        class _NoName:
            display_name = "Bad"
            description = ""

        _patch_entry_points(monkeypatch, [_FakeEntryPoint("noname", "pkg:OBJ", _NoName())])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            registry: PluginRegistry = PluginRegistry(
                package="vtscore.plugins",
                sentinel="TEST_SENTINEL_NEVER_PRESENT",
                label="test plugin",
                entry_point_group="vtsearch.test_family",
            )
            assert registry.list() == []
        assert any("no 'name' attribute" in str(w.message) for w in caught)

    def test_no_group_means_no_entry_point_scan(self, monkeypatch):
        # If a registry has no entry_point_group, importlib.metadata is never
        # consulted; even if our patched version would have happily loaded
        # something.
        called = {"count": 0}

        def fake_entry_points(*, group: str):
            called["count"] += 1
            return []

        monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)

        registry: PluginRegistry = PluginRegistry(
            package="vtscore.plugins",
            sentinel="TEST_SENTINEL_NEVER_PRESENT",
            label="test plugin",
            entry_point_group=None,
        )
        registry.list()
        assert called["count"] == 0


@pytest.mark.parametrize(
    "module_path,registry_var,expected_group",
    [
        ("vtscore.datasets.importers", "list_importers", "vtscore.importers"),
        ("vtscore.exporters", "list_exporters", "vtscore.exporters"),
        ("vtscore.labels.importers", "list_label_importers", "vtscore.label_importers"),
        ("vtscore.labels.sources", "list_labelset_sources", "vtscore.labelset_sources"),
        ("vtsearch.settings_io.importers", "list_settings_importers", "vtsearch.settings_importers"),
        ("vtsearch.settings_io.exporters", "list_settings_exporters", "vtsearch.settings_exporters"),
        ("vtsearch.settings_io.sources", "list_settings_sources", "vtsearch.settings_sources"),
        ("vtscore.converters", "list_converters", "vtscore.converters"),
        ("vtscore.datasets.sources", "list_media_sources", "vtscore.media_sources"),
    ],
)
def test_built_in_registries_declare_entry_point_groups(module_path, registry_var, expected_group):
    """Each plugin family must expose a stable entry-point group name."""
    import importlib

    mod = importlib.import_module(module_path)
    list_fn = getattr(mod, registry_var)
    # ``list_*`` is either a bound method on the registry (from
    # ``make_plugin_registry``) or a free function that delegates to a
    # private module-level ``_registry``.
    registry = getattr(list_fn, "__self__", None)
    if registry is None:
        registry = mod._registry
    assert registry._entry_point_group == expected_group
