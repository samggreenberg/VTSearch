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

from vtsearch.utils.registry import make_plugin_registry

get_settings_source, list_settings_sources = make_plugin_registry(
    package=__name__,
    sentinel="SETTINGS_SOURCE",
    label="settings source",
)

__all__ = ["get_settings_source", "list_settings_sources"]
