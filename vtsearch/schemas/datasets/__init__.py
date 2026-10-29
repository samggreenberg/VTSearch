"""Schemas for dataset-management routes, one module per route group.

Each module mirrors a blueprint in ``vtsearch/routes/datasets/``:

* :mod:`~vtsearch.schemas.datasets.listings`: the read-only plugin listings
  (media types, embedders, clippers, cleaners, converters, importers) and
  the plugin descriptor schemas they nest.
* :mod:`~vtsearch.schemas.datasets.status`: dataset status and cancel.
* :mod:`~vtsearch.schemas.datasets.ui`: the demo dataset list, the
  browse-media-files picker, media-type detection, and the dashboard's
  disk and RAM probes.
* :mod:`~vtsearch.schemas.datasets.load`: ``/api/dataset/load-*`` and
  ``/api/dataset/clear``.
* :mod:`~vtsearch.schemas.datasets.staging`: the staging routes.
* :mod:`~vtsearch.schemas.datasets.registry`: the dataset registry routes.

:mod:`~vtsearch.schemas.datasets.importer_fields` holds the importer
field-options and suggested-name schemas, which the staging routes share
with the label, media and settings routes.

Import each schema from its own module; this package re-exports nothing.
"""
