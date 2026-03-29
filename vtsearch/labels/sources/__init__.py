"""Labelset source registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``LABELSET_SOURCE`` attribute that is a
:class:`~vtsearch.labels.sources.base.LabelsetSource` instance.

Usage::

    from vtsearch.labels.sources import get_labelset_source, list_labelset_sources

    source = get_labelset_source("server_json_file")
    for src in list_labelset_sources():
        print(src.name, src.display_name)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtsearch.utils.registry import PluginRegistry

if TYPE_CHECKING:
    from vtsearch.labels.sources.base import LabelsetSource

_registry: PluginRegistry[LabelsetSource] = PluginRegistry(
    package="vtsearch.labels.sources",
    sentinel="LABELSET_SOURCE",
    label="labelset source",
)

get_labelset_source = _registry.get
list_labelset_sources = _registry.list

__all__ = [
    "get_labelset_source",
    "list_labelset_sources",
]
