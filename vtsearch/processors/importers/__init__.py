"""Processor-importer registry with auto-discovery.

Any package placed directly under this directory is automatically registered
if it exposes a module-level ``PROCESSOR_IMPORTER`` attribute that is a
:class:`~vtsearch.processors.importers.base.ProcessorImporter` instance.

Usage::

    from vtsearch.processors.importers import get_processor_importer, list_processor_importers

    importer = get_processor_importer("server_detector_file")
    for imp in list_processor_importers():
        print(imp.name, imp.display_name)
"""

from vtsearch.utils.registry import make_plugin_registry

get_processor_importer, list_processor_importers = make_plugin_registry(
    package=__name__,
    sentinel="PROCESSOR_IMPORTER",
    label="processor importer",
)

__all__ = ["get_processor_importer", "list_processor_importers"]
