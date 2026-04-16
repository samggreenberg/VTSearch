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

from vtsearch.utils.registry import make_plugin_registry

get_settings_importer, list_settings_importers = make_plugin_registry(
    package=__name__,
    sentinel="SETTINGS_IMPORTER",
    label="settings importer",
)

__all__ = ["get_settings_importer", "list_settings_importers"]
