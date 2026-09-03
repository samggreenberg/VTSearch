"""Base class for Seed Importers.

A **Seed Importer** produces a *batch of unlabeled seed media* for a brand
new (blank) detector.  Where a
:class:`~vtscore.datasource_importers.base.DataSourceImporter` fetches one
exemplar the user picked by hand, and a
:class:`~vtscore.labels.importers.base.LabelImporter` imports media that
already carry a ``good`` / ``bad`` verdict, a seed importer sits between
the two: it hands the user a pile of media that are *close but not quite*
what they are hunting for, with **no verdict attached**.

Seeds are queries, not labels
-----------------------------
Every item a seed importer returns lands in the new detector's ``examples``
list as ``{"type": "media", "value": ..., "labeled": False}``.  That flag is
what separates a seed from an exemplar the user picked themselves:

* it steers the first sort (the label view ranks the haystack against the
  centroid of every media example's embedding, seeds included), but
* it is **not** turned into a ``good``
  :class:`~vtscore.datasets.labelset.LabeledElement` and never casts a good
  vote — see :func:`~vtscore.detectors.media_seeding.is_labeled_example`.

So a seed says "start me near here", not "this one is a hit".  The user
labels from there.

Writing one
-----------
Subclass :class:`SeedImporter`, declare its class attributes and
:meth:`~SeedImporter.run`, then expose a module-level ``SEED_IMPORTER``
instance in a sub-package of ``vtscore/seed_importers/`` (in-tree) or via
the ``vtscore.seed_importers`` entry-point group (from your own package)::

    # my_pkg/seeds.py
    from vtscore.plugins import PluginField
    from vtscore.seed_importers.base import SeedImporter, SeedMediaItem

    class NeighborhoodSeedImporter(SeedImporter):
        \"\"\"Seed a detector from a saved cluster of near-miss media.\"\"\"

        display_name = "Neighborhood"
        fields = [
            PluginField(key="cluster_id", label="Cluster id", field_type="text"),
        ]

        def run(self, field_values):
            return [
                SeedMediaItem(data=blob, filename=name)
                for name, blob in fetch_cluster(field_values["cluster_id"])
            ]

    SEED_IMPORTER = NeighborhoodSeedImporter()

and in your ``pyproject.toml``::

    [project.entry-points."vtscore.seed_importers"]
    neighborhood = "my_pkg.seeds:SEED_IMPORTER"

The web app renders each importer's :attr:`~vtscore.plugins.PluginBase.fields`
as a dynamic form behind its own tab in the New Detector modal's **Blank**
flow, beside the stock Text and media tabs, and calls
``POST /api/seed-import/<name>``, which saves every returned item into the
server-side example-media directory.  An install with no seed importers
registered shows no extra tabs, so the family costs nothing when unused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vtscore.plugins import FieldOption, PluginBase


@dataclass
class SeedMediaItem:
    """One unlabeled seed returned by :meth:`SeedImporter.run`.

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
        re-fetched after the ``example_media/`` cache file is gone.  Same
        contract as
        :attr:`~vtscore.datasource_importers.base.FetchedMediaItem.origin`:
        set it when the item has a stable external identity (a URL, a
        server path), leave it ``None`` when the saved byte snapshot is the
        only record.  Params named ``path`` resolve with no extra code (the
        resolver's generic path fallback); other param shapes need a
        matching :class:`~vtscore.datasets.sources.base.MediaSource` factory
        registered under the importer's name.
    """

    data: bytes
    filename: str
    origin: dict[str, Any] | None = None


class SeedImporter(PluginBase):
    """Abstract base class for batch importers of unlabeled seed media."""

    #: Abstract family base: no auto-derived metadata, and concrete
    #: subclasses strip ``SeedImporter`` from their class names.
    _is_plugin_family_base = True

    #: Stock emoji for the family (seedling).  Concrete subclasses that
    #: don't pick their own icon get a letter glyph instead (see
    #: ``_autoderive_plugin_metadata``).
    icon: str = "\U0001f331"

    #: Upper bound on how many items one run may contribute.  The route
    #: keeps the first ``max_items`` and reports the truncation to the
    #: caller rather than filling ``example_media/`` with an unbounded
    #: batch; raise it on a subclass that legitimately seeds more.
    max_items: int = 100

    def run(self, field_values: dict[str, Any]) -> list[SeedMediaItem]:
        """Return the seed items described by *field_values*.

        *field_values* is the validated + normalized form-field dict (text
        stripped, ``url`` / ``server_path`` fields security-checked; see
        :mod:`vtscore.plugins.normalize`).  ``file``-typed fields arrive as
        :class:`~vtscore.plugins.uploads.UploadedFile` objects.

        Returning an empty list is a valid "nothing matched" answer and is
        reported to the user as such.  Raise :class:`ValueError` for bad
        user input (malformed reference, unknown id); any other exception
        is reported as an upstream/source failure.
        """
        raise NotImplementedError(f"{type(self).__name__}.run() is not implemented")

    def get_field_options(self, field_key: str, current_values: dict[str, Any]) -> list[FieldOption]:
        """Return the current options for a ``dynamic_options`` select field.

        Mirrors :meth:`vtscore.datasets.importers.base.ImporterBase.get_field_options`.
        Only called for fields declared with ``dynamic_options=True``.
        """
        raise NotImplementedError(f"{self.name} does not provide dynamic options for field '{field_key}'")

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["max_items"] = self.max_items
        return d
