"""Settings importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``SETTINGS_IMPORTER`` attribute that is a
:class:`~vtsearch.settings_io.importers.base.SettingsImporter` instance.

Usage::

    from vtsearch.settings_io.importers import get_settings_importer, list_settings_importers

    importer = get_settings_importer("server_json_file")
    for imp in list_settings_importers():
        print(imp.name, imp.display_name)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtsearch.utils.registry import PluginRegistry

if TYPE_CHECKING:
    from vtsearch.settings_io.importers.base import SettingsImporter

_registry: PluginRegistry[SettingsImporter] = PluginRegistry(
    package="vtsearch.settings_io.importers",
    sentinel="SETTINGS_IMPORTER",
    label="settings importer",
)

get_settings_importer = _registry.get
list_settings_importers = _registry.list

__all__ = [
    "get_settings_importer",
    "list_settings_importers",
]
