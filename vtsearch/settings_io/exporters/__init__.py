"""Settings exporter registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``SETTINGS_EXPORTER`` attribute that is a
:class:`~vtsearch.settings_io.exporters.base.SettingsExporter` instance.

Usage::

    from vtsearch.settings_io.exporters import get_settings_exporter, list_settings_exporters

    exporter = get_settings_exporter("server_json_file")
    for exp in list_settings_exporters():
        print(exp.name, exp.display_name)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtsearch.utils.registry import PluginRegistry

if TYPE_CHECKING:
    from vtsearch.settings_io.exporters.base import SettingsExporter

_registry: PluginRegistry[SettingsExporter] = PluginRegistry(
    package="vtsearch.settings_io.exporters",
    sentinel="SETTINGS_EXPORTER",
    label="settings exporter",
)

get_settings_exporter = _registry.get
list_settings_exporters = _registry.list

__all__ = [
    "get_settings_exporter",
    "list_settings_exporters",
]
