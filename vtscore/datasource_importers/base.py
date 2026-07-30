"""Base class for DataSource Importers.

A **DataSource Importer** fetches a *single media item* from some source —
a URL, a file on the server, a third-party service — so the user can supply
exemplar media (e.g. to seed a detector search) from the same kinds of
places a whole dataset can come from.  It is the single-item sibling of
:class:`~vtscore.datasets.importers.base.DatasetImporter`: where a dataset
importer's ``run`` yields a whole collection, a datasource importer's
:meth:`~DataSourceImporter.fetch` returns exactly one item's bytes.

To add a new datasource importer, subclass :class:`DataSourceImporter`,
declare its class attributes and :meth:`~DataSourceImporter.fetch`, then
expose a module-level ``DATASOURCE_IMPORTER`` instance in a sub-package of
``vtscore/datasource_importers/`` (or via the
``vtscore.datasource_importers`` entry-point group)::

    from vtscore.datasource_importers.base import DataSourceImporter, FetchedMediaItem
    from vtscore.plugins import PluginField

    class PastebinDataSourceImporter(DataSourceImporter):
        \"\"\"Fetch a text snippet from a pastebin service.\"\"\"

        category = "services"
        fields = [
            PluginField(key="paste_id", label="Paste id", field_type="text", required=True),
        ]

        def fetch(self, field_values):
            data = ...  # download the snippet's bytes
            return FetchedMediaItem(data=data, filename=f"{field_values['paste_id']}.txt")

    DATASOURCE_IMPORTER = PastebinDataSourceImporter()

The web app renders each importer's :attr:`~vtscore.plugins.PluginBase.fields`
as a dynamic form (the same way the Add Dataset modal renders dataset
importers) and calls ``POST /api/datasource-import/<name>``, which saves the
fetched bytes into the server-side example-media directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vtscore.plugins import FieldOption, PluginBase


@dataclass
class FetchedMediaItem:
    """One media item returned by :meth:`DataSourceImporter.fetch`.

    Attributes
    ----------
    data:
        The item's raw bytes.
    filename:
        A human-meaningful filename for the item.  Its suffix matters: it
        drives media-type inference and how downstream code decodes the
        saved file, so keep the source's real extension.
    origin:
        Optional durable origin dict (``{"importer": <name>, "params":
        {...}}``) describing where the item came from, so it can be
        re-fetched on demand later (cross-dataset label resolution,
        re-deriving a deleted ``example_media/`` cache file).  Importers
        whose items have a stable external identity (a URL, a server path)
        should set this; leave it ``None`` when the bytes have no
        re-derivable source and the saved byte snapshot is the only record.
        Params named ``path`` are resolvable with no extra code (the
        resolver's generic path fallback); other param shapes need a
        matching :class:`~vtscore.datasets.sources.base.MediaSource`
        factory registered under the importer's name.
    """

    data: bytes
    filename: str
    origin: dict[str, Any] | None = None


class DataSourceImporter(PluginBase):
    """Abstract base class for single-media-item importers."""

    #: Stock emoji for the family (inbox tray).  Concrete subclasses that
    #: don't pick their own icon get a letter glyph instead (see
    #: ``_autoderive_plugin_metadata``).
    icon: str = "\U0001f4e5"

    #: Picker tab this importer belongs to in the example-media picker.
    #: Uses the same category ids as dataset importers ("services",
    #: "server", "local", "demo") so both families share one tab bar.
    category: str = "services"

    def fetch(self, field_values: dict[str, Any]) -> FetchedMediaItem:
        """Fetch one media item described by *field_values*.

        *field_values* is the validated + normalized form-field dict (text
        stripped, ``url`` / ``server_path`` fields security-checked; see
        :mod:`vtscore.plugins.normalize`).  ``file``-typed fields arrive as
        :class:`~vtscore.plugins.uploads.UploadedFile` objects.

        Raises :class:`ValueError` for bad user input (missing file,
        malformed reference); any other exception is reported as an
        upstream/source failure.
        """
        raise NotImplementedError

    def get_field_options(self, field_key: str, current_values: dict[str, Any]) -> list[FieldOption]:
        """Return the current options for a ``dynamic_options`` select field.

        Mirrors :meth:`vtscore.datasets.importers.base.ImporterBase.get_field_options`.
        Only called for fields declared with ``dynamic_options=True``.
        """
        raise NotImplementedError(f"{self.name} does not provide dynamic options for field '{field_key}'")

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["category"] = self.category
        return d
