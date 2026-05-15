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

from vtsearch.plugins import make_plugin_registry

get_labelset_source, list_labelset_sources = make_plugin_registry(
    package=__name__,
    sentinel="LABELSET_SOURCE",
    label="labelset source",
    entry_point_group="vtsearch.labelset_sources",
)

__all__ = ["get_labelset_source", "list_labelset_sources"]
