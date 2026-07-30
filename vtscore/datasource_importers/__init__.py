"""DataSource Importer plugin registry.

DataSource importers fetch a *single media item* from some source (a URL,
a server path, a third-party service) so users can supply exemplar media
from the same kinds of places a dataset can come from.  See
:mod:`vtscore.datasource_importers.base` for the plugin contract.

Built-in importers live in sub-packages of this package; third-party
importers register through the ``vtscore.datasource_importers`` entry-point
group.  Each exposes a module-level ``DATASOURCE_IMPORTER`` instance.
"""

from vtscore.datasource_importers.base import DataSourceImporter, FetchedMediaItem
from vtscore.plugins import make_plugin_registry

get_datasource_importer, list_datasource_importers = make_plugin_registry(
    package=__name__,
    sentinel="DATASOURCE_IMPORTER",
    label="datasource importer",
    entry_point_group="vtscore.datasource_importers",
)

__all__ = [
    "DataSourceImporter",
    "FetchedMediaItem",
    "get_datasource_importer",
    "list_datasource_importers",
]
