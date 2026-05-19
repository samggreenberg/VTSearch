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

from vtscore.plugins import make_plugin_registry

get_settings_exporter, list_settings_exporters = make_plugin_registry(
    package=__name__,
    sentinel="SETTINGS_EXPORTER",
    label="settings exporter",
    entry_point_group="vtsearch.settings_exporters",
)

__all__ = ["get_settings_exporter", "list_settings_exporters"]
