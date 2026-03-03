"""Label importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``LABEL_IMPORTER`` attribute that is a
:class:`~vtsearch.labels.importers.base.LabelImporter` instance.

Usage::

    from vtsearch.labels.importers import get_label_importer, list_label_importers

    importer = get_label_importer("csv")
    for imp in list_label_importers():
        print(imp.name, imp.display_name)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtsearch.utils.registry import PluginRegistry

if TYPE_CHECKING:
    from vtsearch.labels.importers.base import LabelImporter

_registry: PluginRegistry[LabelImporter] = PluginRegistry(
    package="vtsearch.labels.importers",
    sentinel="LABEL_IMPORTER",
    label="label importer",
)

get_label_importer = _registry.get
list_label_importers = _registry.list

__all__ = [
    "get_label_importer",
    "list_label_importers",
]
