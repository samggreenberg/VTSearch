"""Dataset importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``IMPORTER`` attribute that is a
:class:`~vtscore.datasets.importers.base.DatasetImporter` instance.

Usage::

    from vtscore.datasets.importers import get_importer, list_importers

    importer = get_importer("server_folder")
    for imp in list_importers():
        print(imp.name, imp.display_name)
"""

from vtscore.plugins import make_plugin_registry

get_importer, list_importers = make_plugin_registry(
    package=__name__,
    sentinel="IMPORTER",
    label="dataset importer",
    entry_point_group="vtscore.importers",
)

__all__ = ["get_importer", "list_importers"]
