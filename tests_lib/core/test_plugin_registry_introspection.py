"""Tests for :class:`PluginRegistry`'s read-only ``package`` / ``sentinel`` /
``entry_point_group`` properties.

``scripts/gen-docs-inventories.py`` reads them off each family's registry to
generate the plugin-families table, reaching the registry through the bound
``list`` accessor :func:`make_plugin_registry` returns.  Both halves are
pinned here: the properties echo the constructor without discovering, and the
accessors stay bound to their registry.
"""

from __future__ import annotations

from vtscore.plugins import PluginRegistry, make_plugin_registry


def test_properties_echo_constructor_without_discovering():
    registry: PluginRegistry = PluginRegistry(
        package="vtscore.exporters",
        sentinel="EXPORTER",
        label="results exporter",
        entry_point_group="vtscore.exporters",
        eager=False,
    )
    assert registry.package == "vtscore.exporters"
    assert registry.sentinel == "EXPORTER"
    assert registry.entry_point_group == "vtscore.exporters"
    assert registry._discovered is False


def test_entry_point_group_defaults_to_none():
    registry: PluginRegistry = PluginRegistry(
        package="vtscore.exporters", sentinel="EXPORTER", label="results exporter", eager=False
    )
    assert registry.entry_point_group is None


def test_factory_accessors_are_bound_to_their_registry():
    get, list_ = make_plugin_registry(
        package="vtscore.exporters",
        sentinel="EXPORTER",
        label="results exporter",
        entry_point_group="vtscore.exporters",
        eager=False,
    )
    registry = list_.__self__
    assert isinstance(registry, PluginRegistry)
    assert get.__self__ is registry
    assert (registry.package, registry.sentinel) == ("vtscore.exporters", "EXPORTER")
