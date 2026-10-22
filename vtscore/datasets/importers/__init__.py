"""Dataset importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``IMPORTER`` attribute: an instance of
:class:`~vtscore.datasets.importers.base.ImporterBase`.  Importers that drive
their own ingestion subclass ``ImporterBase`` directly; spec-driven and
service-style ones subclass its
:class:`~vtscore.datasets.importers.base.DatasetImporter`.  Third-party
packages register under the ``vtscore.importers`` entry-point group.

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
