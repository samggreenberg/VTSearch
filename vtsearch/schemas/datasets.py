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

from marshmallow import Schema, ValidationError, fields, validate

from vtsearch.schemas.file_browser import BrowseDirectoryEntrySchema, BrowseFileEntrySchema


def _list_of_strings(value):
    """Validator: value must be a ``list`` whose every entry is a ``str``.

    Used by the readers ACL endpoint so that numeric or other non-string
    items are rejected at the schema layer (422) rather than silently
    coerced to strings by ``fields.String``'s deserializer.
    """
    if not isinstance(value, list) or not all(isinstance(r, str) for r in value):
        raise ValidationError("Must be a list of strings.")


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
        metadata={"description": "One of ``demo:<name>``, ``folder``, or ``server_fs``."},
    )
    path = fields.String(
        load_default="",
        metadata={"description": "Relative sub-path within the source root."},
    )


class BrowseMediaFilesResponseSchema(Schema):
    """Response for ``GET /api/browse-media-files``.

    The directory- and file-entry shapes are identical to the ones
    used by ``GET /api/browse`` (see ``vtsearch.schemas.file_browser``),
    so reuse those nested schemas — registering distinct
    ``_BrowseDirectoryEntry`` / ``_BrowseFileEntry`` schemas alongside
    the public names made the generated TS client emit duplicate
    identifiers (ng-openapi-gen strips the leading underscore).
    """

    directories = fields.List(fields.Nested(BrowseDirectoryEntrySchema), required=True)
    files = fields.List(fields.Nested(BrowseFileEntrySchema), required=True)
    root_path = fields.String(required=True)
    default_path = fields.String(
        load_default="",
        dump_default="",
        metadata={
            "description": (
                "Suggested initial relative sub-path for this source — for example "
                "the server user's home directory when ``source=server_fs``. Empty "
                "for sources where the root is already the right starting point."
            )
        },
    )


class BrowseMediaFilesSelectRequestSchema(Schema):
    """Body for ``POST /api/browse-media-files/select``."""

    source = fields.String(required=True, validate=validate.Length(min=1))
    path = fields.String(required=True, validate=validate.Length(min=1))


class BrowseMediaFilesSelectResponseSchema(Schema):
    """Response for ``POST /api/browse-media-files/select``."""

    filename = fields.String(required=True)
    original_name = fields.String(required=True)


class DetectMediaTypeQuerySchema(Schema):
    """Query for ``GET /api/dataset/detect-media-type``.

    The ``limit`` field is capped to ``[1, 500]`` by the handler; the
    schema only narrows the type. Invalid integer strings fall back to
    the default rather than rejecting the request, preserving the
    pre-migration permissiveness for this hint endpoint.
    """

    source = fields.String(
        load_default="folder",
        metadata={"description": "One of ``demo:<name>`` or ``folder`` (matches ``/api/browse-media-files``)."},
    )
    path = fields.String(load_default="")
    recursive = fields.Boolean(load_default=True)
    limit = fields.Integer(load_default=50)

    class Meta:
        unknown = "exclude"


class DetectMediaTypeResponseSchema(Schema):
    """Response for ``GET /api/dataset/detect-media-type``.

    Mirrors :func:`vtscore.datasets.media_type_detection.detect_media_types_in_folder`'s
    return value.
    """

    sample_size = fields.Integer(required=True)
    counts_by_type = fields.Dict(keys=fields.String(), values=fields.Integer(), required=True)
    extensions = fields.Dict(keys=fields.String(), values=fields.Integer(), required=True)
    dominant = fields.String(allow_none=True, required=True)
    truncated = fields.Boolean(required=True)


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


class DashboardRamUsageResponseSchema(Schema):
    """Response for ``GET /api/dashboard/ram-usage``."""

    total = fields.Integer(required=True)
    used = fields.Integer(required=True)
    free = fields.Integer(required=True)


# ---------------------------------------------------------------------------
# /api/dataset/load-* and /api/dataset/clear (vtsearch/routes/datasets/load.py)
# ---------------------------------------------------------------------------


class DatasetLoadDemoRequestSchema(Schema):
    """Body for ``POST /api/dataset/load-demo``.

    ``name`` must be a key of ``DEMO_DATASETS``; the handler returns 400
    if it isn't (the set isn't known at schema-build time).
    """

    name = fields.String(required=True, validate=validate.Length(min=1))
    embedder = fields.String(load_default="")
    clipper = fields.String(load_default="")
    converter = fields.String(load_default="")
    dataset_name = fields.String(load_default="")


class DatasetLoadFolderRequestSchema(Schema):
    """Body for ``POST /api/dataset/load-folder``."""

    path = fields.String(required=True, validate=validate.Length(min=1))
    media_type = fields.String(load_default="audio")


class DatasetLoadSourceRequestSchema(Schema):
    """Body for ``POST /api/dataset/load-source``.

    ``source`` is a raw origin dict (``{"importer": ..., "params": {...}}``)
    whose inner shape varies per importer and is validated by the
    handler (via ``can_reload_from_origin`` / ``reload_from_origin``).
    """

    source = fields.Dict(
        required=True,
        metadata={"description": "Origin dict as stored on medias."},
    )


class DatasetLoadStartedResponseSchema(Schema):
    """Response for ``POST /api/dataset/load-demo`` / ``load-file`` /
    ``load-folder`` / ``load-source`` / ``import-local-folder`` and for
    ``POST /api/dataset/combine`` in ``staging.py``.

    ``task_id`` is the background-task tracker id (string) used by the
    SSE progress stream; it may be empty when the load completes
    synchronously (rare).
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    task_id = fields.String(required=True)


class DatasetClearResponseSchema(Schema):
    """Response for ``POST /api/dataset/clear``."""

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# Staging routes (vtsearch/routes/datasets/staging.py)
# ---------------------------------------------------------------------------


class _AvailableDatasetFileSchema(Schema):
    """One ``.pkl`` file listed by ``GET /api/dataset/available-files``."""

    name = fields.String(required=True)
    path = fields.String(required=True)
    size_mb = fields.Float(required=True)


class DatasetAvailableFilesResponseSchema(Schema):
    """Response for ``GET /api/dataset/available-files``."""

    files = fields.List(fields.Nested(_AvailableDatasetFileSchema), required=True)


class DatasetCombineRequestSchema(Schema):
    """Body for ``POST /api/dataset/combine``."""

    datasets = fields.List(
        fields.String(),
        required=True,
        validate=validate.Length(min=2),
        metadata={"description": "At least two server-side pickle file paths to merge."},
    )
    name = fields.String(load_default="")


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
    plugin-field staging routes that haven't migrated yet."""

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)


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


class ImporterFieldOptionsRequestSchema(Schema):
    """Body for ``POST /api/dataset/import/<importer_name>/options``."""

    field_key = fields.String(required=True, validate=validate.Length(min=1))
    values = fields.Dict(load_default=dict)


class ImporterFieldOptionsResponseSchema(Schema):
    """Response for ``POST /api/dataset/import/<importer_name>/options``."""

    options = fields.List(fields.String(), required=True)


# ---------------------------------------------------------------------------
# Registry routes (vtsearch/routes/datasets/registry.py)
# ---------------------------------------------------------------------------


class DatasetsRegistryListResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry``.

    Each entry's inner shape is the registry record (plus a derived
    ``loaded`` flag and resolved ``clipper`` display name). Declared as
    ``fields.Dict`` to avoid duplicating the registry record schema —
    drift between schema and registry would be caught at the registry
    layer, not the route.
    """

    datasets = fields.List(fields.Dict(), required=True)


class DatasetRegistryLoadResponseSchema(Schema):
    """Response for ``POST /api/datasets/registry/<id>/load``.

    Successful kickoff returns ``task_id``; the "already loaded" path
    returns the same envelope with an empty ``task_id``.
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    task_id = fields.String(load_default="")


class DatasetRegistryOkResponseSchema(Schema):
    """Bare ``{"ok": true}`` response (unload, delete)."""

    ok = fields.Boolean(required=True)


class DatasetRegistryPreloadEmbedderResponseSchema(Schema):
    """Response for ``POST /api/datasets/registry/<id>/preload-embedder``.

    ``embedder`` is the name of the embedder being warmed in the
    background, or ``""`` when no embedder could be resolved (e.g. the
    dataset's media type has no registered embedder).
    """

    ok = fields.Boolean(required=True)
    embedder = fields.String(required=True)


class DatasetRegistryRenameRequestSchema(Schema):
    """Body for ``PUT /api/datasets/registry/<id>/rename``."""

    name = fields.String(required=True, validate=validate.Length(min=1))


class DatasetRegistryRenameResponseSchema(Schema):
    """Response for ``PUT /api/datasets/registry/<id>/rename``."""

    ok = fields.Boolean(required=True)
    name = fields.String(required=True)


class DatasetRegistryReadersRequestSchema(Schema):
    """Body for ``PUT /api/datasets/registry/<id>/readers``.

    Declared as ``fields.Raw`` with a custom validator (rather than
    ``fields.List(fields.String())``) so that numeric or other
    non-string items are rejected as 422 instead of being silently
    coerced to strings by ``fields.String``'s deserializer.
    """

    readers = fields.Raw(
        required=True,
        validate=_list_of_strings,
        metadata={
            "description": 'List of usernames; ``["*"]`` makes the dataset public.',
            "type": "array",
            "items": {"type": "string"},
        },
    )


class DatasetRegistryReadersResponseSchema(Schema):
    """Response for ``PUT /api/datasets/registry/<id>/readers``."""

    ok = fields.Boolean(required=True)
    readers = fields.List(fields.String(), required=True)


class DatasetRegistryStatsResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry/<id>/stats``."""

    num_items = fields.Integer(required=True)
    num_dupes = fields.Integer(required=True)
    file_type_counts = fields.Dict(keys=fields.String(), values=fields.Integer(), required=True)
    ingest_started_at = fields.Raw(allow_none=True)
    ingest_finished_at = fields.Raw(allow_none=True)
    origin = fields.String(required=True)
    source = fields.Dict(required=True)
    clipper = fields.String(required=True)
    embedder = fields.String(required=True)


__all__ = [
    "BrowseMediaFilesQuerySchema",
    "BrowseMediaFilesResponseSchema",
    "BrowseMediaFilesSelectRequestSchema",
    "BrowseMediaFilesSelectResponseSchema",
    "CancelDatasetLoadResponseSchema",
    "ClearStagingResponseSchema",
    "ClippersListQuerySchema",
    "ClippersListResponseSchema",
    "ConvertersListQuerySchema",
    "ConvertersListResponseSchema",
    "DashboardDatasetInfoResponseSchema",
    "DashboardDatasetRenameRequestSchema",
    "DashboardDatasetRenameResponseSchema",
    "DashboardDiskUsageResponseSchema",
    "DashboardRamUsageResponseSchema",
    "DatasetAllImportersListResponseSchema",
    "DatasetAvailableFilesResponseSchema",
    "DatasetClearResponseSchema",
    "DatasetCombineRequestSchema",
    "DatasetImportersListResponseSchema",
    "DatasetLoadDemoRequestSchema",
    "DatasetLoadFolderRequestSchema",
    "DatasetLoadSourceRequestSchema",
    "DatasetLoadStartedResponseSchema",
    "DatasetRegistryLoadResponseSchema",
    "DatasetRegistryReadersRequestSchema",
    "DatasetRegistryReadersResponseSchema",
    "DatasetRegistryRenameRequestSchema",
    "DatasetRegistryRenameResponseSchema",
    "DatasetRegistryStatsResponseSchema",
    "DatasetStageDemoRequestSchema",
    "DatasetStageFileResponseSchema",
    "DatasetStagingStartedResponseSchema",
    "DatasetStatusResponseSchema",
    "DatasetsRegistryListResponseSchema",
    "DemoCategoriesResponseSchema",
    "DemoDatasetListQuerySchema",
    "DemoDatasetListResponseSchema",
    "EmbeddersListQuerySchema",
    "EmbeddersListResponseSchema",
    "DatasetRegistryOkResponseSchema",
    "DatasetRegistryPreloadEmbedderResponseSchema",
    "ImporterFieldOptionsRequestSchema",
    "ImporterFieldOptionsResponseSchema",
    "MediaTypesListResponseSchema",
]
