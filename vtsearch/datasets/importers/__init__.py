"""Dataset importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``IMPORTER`` attribute that is a
:class:`~vtsearch.datasets.importers.base.DatasetImporter` instance.

Usage::

    from vtsearch.datasets.importers import get_importer, list_importers

    importer = get_importer("folder")
    for imp in list_importers():
        print(imp.name, imp.display_name)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vtsearch.utils.registry import PluginRegistry

if TYPE_CHECKING:
    from vtsearch.datasets.importers.base import DatasetImporter

_registry: PluginRegistry[DatasetImporter] = PluginRegistry(
    package="vtsearch.datasets.importers",
    sentinel="IMPORTER",
    label="dataset importer",
)

get_importer = _registry.get
list_importers = _registry.list

__all__ = [
    "get_importer",
    "list_importers",
]
