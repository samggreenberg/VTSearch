"""Schemas for the staging routes (``vtsearch/routes/datasets/staging.py``).

The importer field-options and suggested-name schemas those routes also
use are shared with other route groups, so they live in
:mod:`vtsearch.schemas.datasets.importer_fields`.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


class _AvailableDatasetFileSchema(Schema):
    """One ``.pkl`` file listed by ``GET /api/dataset/available-files``."""

    name = fields.String(required=True)
    path = fields.String(required=True)
    size_mb = fields.Float(required=True)


class DatasetAvailableFilesResponseSchema(Schema):
    """Response for ``GET /api/dataset/available-files``."""

    files = fields.List(fields.Nested(_AvailableDatasetFileSchema), required=True)


class DatasetCombineResolutionSchema(Schema):
    """One per-embedder-type conflict resolution in a combine request.

    ``action`` is ``"reembed"`` (re-embed every source dataset to *embedder* so
    the whole combined dataset shares that concrete embedder) or ``"drop"``
    (leave that embedder type out of the combined dataset entirely).  ``embedder``
    is required for ``reembed`` and ignored for ``drop``.
    """

    action = fields.String(required=True, validate=validate.OneOf(["reembed", "drop"]))
    embedder = fields.String(load_default="")


class DatasetCombineRequestSchema(Schema):
    """Body for ``POST /api/dataset/combine``."""

    datasets = fields.List(
        fields.String(),
        required=True,
        validate=validate.Length(min=2),
        metadata={"description": "At least two server-side pickle file paths to merge."},
    )
    name = fields.String(load_default="")
    #: Per-embedder-type conflict resolutions, keyed by embedder type
    #: (``semantic`` / ``patch_semantic`` / ``structural``).  Present only when
    #: the sources bind conflicting embedders of the same type; the combine route
    #: refuses (400) a conflict left unresolved here.
    resolutions = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(DatasetCombineResolutionSchema),
        load_default=dict,
        metadata={"description": "Embedder-type -> {action, embedder} conflict resolutions."},
    )


class DatasetPromoteRequestSchema(Schema):
    """Body for ``POST /api/dataset/promote``.

    Promotes a set of media items from the active dataset into a brand-new
    saved dataset (e.g. the Find "Goods" pile). The items keep their
    original origins and embeddings; the new dataset gets a fresh
    ``created_at`` but inherits the source dataset's ``expires_at``.
    """

    name = fields.String(
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "Display name for the new dataset."},
    )
    media_ids = fields.List(
        fields.Integer(),
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "IDs of the media items (in the active dataset) to promote."},
    )


class DatasetStageFileResponseSchema(Schema):
    """Response for ``POST /api/dataset/stage-file`` (multipart upload).

    ``count`` and ``media_type`` are derived from a cheap pickle peek and
    fall back to ``0`` / ``"unknown"`` when the file can't be inspected.
    ``error`` carries the reason peek failed (empty string on success), so
    the UI can distinguish "valid pickle with 0 medias" from "couldn't
    parse this file".
    """

    path = fields.String(required=True)
    name = fields.String(required=True)
    count = fields.Integer(required=True)
    media_type = fields.String(required=True)
    error = fields.String(load_default="", dump_default="")


class DatasetStagingStartedResponseSchema(Schema):
    """Response for ``POST /api/dataset/stage-demo/<name>`` and the
    plugin-field staging routes that haven't migrated yet.

    ``task_id`` is the background staging-task tracker id (string) used by
    the ``loading-tasks`` SSE channel to poll progress and pick up the final
    ``staging_result``; it may be empty when no task was started."""

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    task_id = fields.String(required=True)


class DatasetStageDemoRequestSchema(Schema):
    """Body for ``POST /api/dataset/stage-demo/<name>``.

    ``name`` is supplied via the URL path; the optional ``converter`` /
    ``dataset_name`` override the demo's defaults.
    """

    converter = fields.String(load_default="")
    dataset_name = fields.String(load_default="")


class ClearStagingResponseSchema(Schema):
    """Response for ``DELETE /api/dataset/staging``."""

    ok = fields.Boolean(required=True)


__all__ = [
    "ClearStagingResponseSchema",
    "DatasetAvailableFilesResponseSchema",
    "DatasetCombineRequestSchema",
    "DatasetCombineResolutionSchema",
    "DatasetStageDemoRequestSchema",
    "DatasetStageFileResponseSchema",
    "DatasetStagingStartedResponseSchema",
]
