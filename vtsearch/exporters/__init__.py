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

from __future__ import annotations

from typing import TYPE_CHECKING

from vtsearch.utils.registry import PluginRegistry

if TYPE_CHECKING:
    from vtsearch.exporters.base import LabelsetExporter

_registry: PluginRegistry[LabelsetExporter] = PluginRegistry(
    package="vtsearch.exporters",
    sentinel="EXPORTER",
    label="labelset exporter",
)

get_exporter = _registry.get
list_exporters = _registry.list

__all__ = [
    "get_exporter",
    "list_exporters",
]
