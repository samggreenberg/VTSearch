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

from vtsearch.plugins import make_plugin_registry

get_label_importer, list_label_importers = make_plugin_registry(
    package=__name__,
    sentinel="LABEL_IMPORTER",
    label="label importer",
    entry_point_group="vtsearch.label_importers",
)

__all__ = ["get_label_importer", "list_label_importers"]
