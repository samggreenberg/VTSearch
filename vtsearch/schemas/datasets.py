"""Schemas for dataset-management routes.

These cover the read-only / display-oriented dataset blueprints — listings
(media types, embedders, clippers, converters, importers), status / cancel,
and UI helpers (demo dataset list, file browser, dashboard). The heavier
modules (``load``, ``staging``, ``registry``) involve multipart upload,
binary streaming, and plugin-field-shaped bodies and are migrated
separately. See ``docs/plans/openapi-schema.md``.

The ``to_dict()`` payloads for media types, embedders, clippers,
converters, and importers are intentionally declared as ``fields.Dict()``
rather than nested schemas: the inner shapes are plugin-dependent and
already round-trip cleanly via ``to_dict()``. Re-declaring every field
would buy nothing — drift between schema and plugin metadata would be
caught at the *plugin* layer, not the route.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


# ---------------------------------------------------------------------------
# /api/media-types, /api/embedders, /api/clippers, /api/converters
# (read-only listings)
# ---------------------------------------------------------------------------


class _MediaTypeFilterQuerySchema(Schema):
    """Query string shared by ``GET /api/embedders`` and ``/api/clippers``."""

    media_type = fields.String(
        load_default="",
        metadata={
            "description": (
                "Optional ``type_id`` (e.g. ``image``) or "
                "``folder_import_name`` (e.g. ``images``). When provided, "
                "only matching entries are returned."
            )
        },
    )


class EmbeddersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/embedders``."""


class ClippersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/clippers``."""


class ConvertersListQuerySchema(Schema):
    """Query for ``GET /api/converters``.

    ``target`` and ``source`` are mutually exclusive filters; if both are
    supplied the handler prefers ``target``.
    """

    target = fields.String(load_default="")
    source = fields.String(load_default="")


class MediaTypesListResponseSchema(Schema):
    """Response for ``GET /api/media-types``."""

    media_types = fields.List(fields.Dict(), required=True)


class EmbeddersListResponseSchema(Schema):
    """Response for ``GET /api/embedders``."""

    embedders = fields.List(fields.Dict(), required=True)


class ClippersListResponseSchema(Schema):
    """Response for ``GET /api/clippers``."""

    clippers = fields.List(fields.Dict(), required=True)


class ConvertersListResponseSchema(Schema):
    """Response for ``GET /api/converters``."""

    converters = fields.List(fields.Dict(), required=True)


class DatasetImportersListResponseSchema(Schema):
    """Response for ``GET /api/dataset/importers``."""

    importers = fields.List(fields.Dict(), required=True)


class DatasetAllImportersListResponseSchema(Schema):
    """Response for ``GET /api/dataset/all-importers``.

    ``tabs`` is the picker-tab layout (id, label, icon, order); the
    ``combine_datasets`` importer is annotated with an ``enabled`` flag
    by the handler.
    """

    importers = fields.List(fields.Dict(), required=True)
    tabs = fields.List(fields.Dict(), required=True)


# ---------------------------------------------------------------------------
# /api/dataset/status, /api/dataset/cancel
# ---------------------------------------------------------------------------


class DatasetStatusResponseSchema(Schema):
    """Response for ``GET /api/dataset/status``."""

    loaded = fields.Boolean(required=True)
    num_medias = fields.Integer(required=True)
    has_votes = fields.Boolean(required=True)
    media_type = fields.String(allow_none=True, required=True)
    display_name = fields.String(required=True)
    num_dupes = fields.Integer(required=True)


class CancelDatasetLoadResponseSchema(Schema):
    """Response for ``POST /api/dataset/cancel`` and ``/cancel/<task_id>``."""

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# /api/dataset/demo-list, /api/dataset/demo-categories/<name>
# ---------------------------------------------------------------------------


class _DemoDatasetEntrySchema(Schema):
    """One entry in the ``GET /api/dataset/demo-list`` ``datasets`` array."""

    name = fields.String(required=True)
    label = fields.String(required=True)
    status = fields.String(required=True, validate=validate.OneOf(["ready", "needs_embedding", "needs_download"]))
    ready = fields.Boolean(required=True)
    num_files = fields.Integer(required=True)
    download_size_mb = fields.Float(required=True)
    description = fields.String(required=True)
    media_type = fields.String(required=True)
    num_categories = fields.Integer(required=True)
    available_converters = fields.List(fields.Dict(), required=True)
    pkl_embedder = fields.String(required=True)
    pkl_clipper = fields.String(required=True)


class DemoDatasetListQuerySchema(Schema):
    """Query string for ``GET /api/dataset/demo-list``.

    Both fields are optional cache-key filters: when supplied, a cached
    pkl is only considered ``"ready"`` if it was produced with the same
    embedder / clipper.
    """

    embedder = fields.String(load_default="")
    clipper = fields.String(load_default="")


class DemoDatasetListResponseSchema(Schema):
    """Response for ``GET /api/dataset/demo-list``."""

    datasets = fields.List(fields.Nested(_DemoDatasetEntrySchema), required=True)


class DemoCategoriesResponseSchema(Schema):
    """Response for ``GET /api/dataset/demo-categories/<name>``."""

    categories = fields.List(fields.String(), required=True)


# ---------------------------------------------------------------------------
# /api/browse-media-files, /api/browse-media-files/select
# ---------------------------------------------------------------------------


class BrowseMediaFilesQuerySchema(Schema):
    """Query for ``GET /api/browse-media-files``."""

    source = fields.String(
        load_default="",
        metadata={"description": "One of ``demo:<name>`` or ``folder``."},
    )
    path = fields.String(
        load_default="",
        metadata={"description": "Relative sub-path within the source root."},
    )


class _BrowseDirectoryEntrySchema(Schema):
    name = fields.String(required=True)
    path = fields.String(required=True)
    modified_at = fields.String(required=True)


class _BrowseFileEntrySchema(Schema):
    name = fields.String(required=True)
    path = fields.String(required=True)
    size_bytes = fields.Integer(required=True)
    modified_at = fields.String(required=True)


class BrowseMediaFilesResponseSchema(Schema):
    """Response for ``GET /api/browse-media-files``."""

    directories = fields.List(fields.Nested(_BrowseDirectoryEntrySchema), required=True)
    files = fields.List(fields.Nested(_BrowseFileEntrySchema), required=True)
    root_path = fields.String(required=True)


class BrowseMediaFilesSelectRequestSchema(Schema):
    """Body for ``POST /api/browse-media-files/select``."""

    source = fields.String(required=True, validate=validate.Length(min=1))
    path = fields.String(required=True, validate=validate.Length(min=1))


class BrowseMediaFilesSelectResponseSchema(Schema):
    """Response for ``POST /api/browse-media-files/select``."""

    filename = fields.String(required=True)
    original_name = fields.String(required=True)


# ---------------------------------------------------------------------------
# /api/dashboard/dataset-info, /api/dashboard/dataset-rename, /api/dashboard/disk-usage
# ---------------------------------------------------------------------------


class DashboardDatasetInfoResponseSchema(Schema):
    """Response for ``GET /api/dashboard/dataset-info``.

    ``source`` is the raw origin dict from the first media that has one;
    it can be ``null`` when no medias carry origin info.
    """

    name = fields.String(required=True)
    num_medias = fields.Integer(required=True)
    num_dupes = fields.Integer(required=True)
    media_type = fields.String(required=True)
    origin = fields.String(required=True)
    source = fields.Dict(allow_none=True, required=True)


class DashboardDatasetRenameRequestSchema(Schema):
    """Body for ``PUT /api/dashboard/dataset-rename``."""

    name = fields.String(required=True, validate=validate.Length(min=1))


class DashboardDatasetRenameResponseSchema(Schema):
    """Response for ``PUT /api/dashboard/dataset-rename``."""

    success = fields.Boolean(required=True)
    name = fields.String(required=True)


class DashboardDiskUsageResponseSchema(Schema):
    """Response for ``GET /api/dashboard/disk-usage``."""

    total = fields.Integer(required=True)
    used = fields.Integer(required=True)
    free = fields.Integer(required=True)
    path = fields.String(required=True)


__all__ = [
    "BrowseMediaFilesQuerySchema",
    "BrowseMediaFilesResponseSchema",
    "BrowseMediaFilesSelectRequestSchema",
    "BrowseMediaFilesSelectResponseSchema",
    "CancelDatasetLoadResponseSchema",
    "ClippersListQuerySchema",
    "ClippersListResponseSchema",
    "ConvertersListQuerySchema",
    "ConvertersListResponseSchema",
    "DashboardDatasetInfoResponseSchema",
    "DashboardDatasetRenameRequestSchema",
    "DashboardDatasetRenameResponseSchema",
    "DashboardDiskUsageResponseSchema",
    "DatasetAllImportersListResponseSchema",
    "DatasetImportersListResponseSchema",
    "DatasetStatusResponseSchema",
    "DemoCategoriesResponseSchema",
    "DemoDatasetListQuerySchema",
    "DemoDatasetListResponseSchema",
    "EmbeddersListQuerySchema",
    "EmbeddersListResponseSchema",
    "MediaTypesListResponseSchema",
]
