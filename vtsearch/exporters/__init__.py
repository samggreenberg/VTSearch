"""Labelset-exporter registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``EXPORTER`` attribute that is a
:class:`~vtsearch.exporters.base.LabelsetExporter` instance.

Usage::

    from vtsearch.exporters import get_exporter, list_exporters

    exporter = get_exporter("server_json_file")
    for exp in list_exporters():
        print(exp.name, exp.display_name)
"""

from vtsearch.plugins import make_plugin_registry

get_exporter, list_exporters = make_plugin_registry(
    package=__name__,
    sentinel="EXPORTER",
    label="labelset exporter",
    entry_point_group="vtsearch.exporters",
)

__all__ = ["get_exporter", "list_exporters"]
