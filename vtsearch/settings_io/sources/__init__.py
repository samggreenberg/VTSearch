"""Settings source registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``SETTINGS_SOURCE`` attribute that is a
:class:`~vtsearch.settings_io.sources.base.SettingsSource` instance.

Usage::

    from vtsearch.settings_io.sources import get_settings_source, list_settings_sources

    source = get_settings_source("server_json_file")
    for src in list_settings_sources():
        print(src.name, src.display_name)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtsearch.utils.registry import PluginRegistry

if TYPE_CHECKING:
    from vtsearch.settings_io.sources.base import SettingsSource

_registry: PluginRegistry[SettingsSource] = PluginRegistry(
    package="vtsearch.settings_io.sources",
    sentinel="SETTINGS_SOURCE",
    label="settings source",
)

get_settings_source = _registry.get
list_settings_sources = _registry.list

__all__ = [
    "get_settings_source",
    "list_settings_sources",
]
