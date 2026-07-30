"""Schemas for dataset-management routes.

These cover the read-only / display-oriented dataset blueprints: listings
(media types, embedders, clippers, converters, importers), status / cancel,
and UI helpers (demo dataset list, file browser, dashboard). The heavier
modules (``load``, ``staging``, ``registry``) involve multipart upload,
binary streaming, and plugin-field-shaped bodies and are migrated
separately. See ``docs/plans/openapi-schema.md``.

The ``to_dict()`` payloads split by whether the shape is fixed or
plugin-variable:

* **Media types, embedders, converters** have a *fixed* key set across every
  plugin (no subclass overrides ``to_dict``), so they are enumerated as
  nested schemas (``MediaTypeInfoSchema``, ``EmbedderInfoSchema``,
  ``ConverterInfoSchema``).  This gives the generated OpenAPI client a real
  typed model, letting the frontend drop its hand-written mirror.
* **Clippers and importers** stay opaque ``fields.Dict()``: concrete clippers
  add their own keys (``duration``, ``top_db``, …) and the importer payload
  nests plugin-field lists, so a nested schema would either lose keys or need
  an escape hatch anyway.  Drift there is caught at the *plugin* layer.
"""

from __future__ import annotations

from marshmallow import Schema, ValidationError, fields, validate

from vtsearch.schemas.file_browser import BrowseDirectoryEntrySchema, BrowseFileEntrySchema

#: Upper bound on user-supplied dataset names, mirroring
#: ``vtsearch.schemas.detectors.MAX_NAME_LENGTH``.  A name past this is already
#: unusable for display, and capping it here stops an absurdly long name from
#: ever reaching a filesystem path (and the uncaught ``OSError`` /
#: absolute-path leak that would follow).
MAX_NAME_LENGTH = 128


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


class CleanersListQuerySchema(_MediaTypeFilterQuerySchema):
    """Query for ``GET /api/cleaners``."""


class ConvertersListQuerySchema(Schema):
    """Query for ``GET /api/converters``.

    ``target`` and ``source`` are mutually exclusive filters; if both are
    supplied the handler prefers ``target``.
    """

    target = fields.String(load_default="")
    source = fields.String(load_default="")


class MediaTypeInfoSchema(Schema):
    """One ``MediaType.to_dict()`` payload (see ``vtscore/media/base.py``).

    Fixed shape: every media type emits the same keys, so this is a real
    nested schema rather than an opaque ``fields.Dict()``.  Only ``type_id``
    and ``name`` are guaranteed present enough to mark ``required``; the rest
    mirror the frontend's optional fields.
    """

    type_id = fields.String(required=True)
    name = fields.String(required=True)
    icon = fields.String()
    folder_import_name = fields.String()
    loops = fields.Boolean(
        metadata={"description": "Whether this media type's rendered form loops (e.g. short video/audio)."}
    )
    file_extensions = fields.List(
        fields.String(),
        metadata={"description": 'Glob patterns for files this media type claims, e.g. ``["*.jpg", "*.png"]``.'},
    )
    has_thumbnail = fields.Boolean(
        metadata={
            "description": (
                "Whether items of this type have a browsable thumbnail (image/video/document, and audio via "
                "its waveform PNG). Drives the VTSBrowse square-vs-hex bin shape and thumbnail painting."
            )
        }
    )
    importable = fields.Boolean(
        metadata={
            "description": (
                "Whether this type is a first-class ingestion category the user picks when importing (folder "
                "scan, file upload). ``false`` for a convert-in half type like ``face``."
            )
        }
    )
    embeddable = fields.Boolean(
        metadata={
            "description": (
                "Whether this type can be embedded (and therefore sorted / browsed / text-queried) on its own. "
                "``false`` for a convert-out half type like ``document`` that must be converted first."
            )
        }
    )
    converts_to = fields.List(
        fields.String(),
        metadata={
            "description": (
                "Embeddable target type_ids a non-embeddable type can convert into (first = default). "
                '``["image", "text"]`` for ``document``; empty for a directly-embeddable type.'
            )
        },
    )


class EmbedderInfoSchema(Schema):
    """One ``MediaEmbedder.to_dict()`` payload (see ``vtscore/media/embedder.py``).

    Fixed shape across all embedders (no subclass overrides ``to_dict``), so
    the payload is fully enumerated here instead of ``fields.Dict()``.
    """

    name = fields.String(required=True)
    display_name = fields.String(
        metadata={
            "description": (
                'Human-readable label shown in pickers, e.g. ``"SigLIP (general images)"``. Falls back to '
                "``name`` for legacy embedders that don't supply a friendlier label."
            )
        }
    )
    model_id = fields.String(
        allow_none=True,
        metadata={
            "description": (
                "Concrete pretrained-model identifier the embedder loads - usually a HuggingFace repo id "
                "(or a direct weights URL). ``null`` for embedders with no single downloadable model id "
                "(e.g. the classical SIFT/VLAD structural embedder)."
            )
        },
    )
    media_type_id = fields.String(required=True)
    is_default = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder is the recommended default for its media type (exactly one per media "
                'type). The dropdown surfaces this entry under a "Recommended" optgroup.'
            )
        }
    )
    supports_text = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder can embed text queries into the same vector space as its media. "
                "``false`` for vision-only encoders (DINOv3, Perception Encoder)."
            )
        }
    )
    supports_patch_regions = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder produces patch-level vectors and a hierarchical region tree per image. "
                "``true`` for patch-based encoders (DINOv2, DINOv3, EUPE)."
            )
        }
    )
    supports_geometric_verification = fields.Boolean(
        metadata={
            "description": (
                "Whether this embedder produces local features (keypoints + descriptors) for instance "
                "matching. ``true`` for structural embedders (SIFT/VLAD); ``false`` for every semantic embedder."
            )
        }
    )
    license_notice = fields.String(
        allow_none=True,
        metadata={
            "description": (
                "User-facing licence warning to show before the user picks this embedder. ``null`` for "
                "embedders with no special licensing constraints. Advisory only."
            )
        },
    )


class ConverterInfoSchema(Schema):
    """One ``MediaConverter.to_dict()`` payload (see ``vtscore/converters/base.py``).

    Fixed shape across all converters.  The plugin ``fields`` list (importer
    fields) is kept opaque here: the ``ImporterField`` shape is out of this
    slice's scope, and the frontend re-types it via ``ImporterField[]``.
    """

    name = fields.String(required=True)
    source_type = fields.String(required=True)
    target_type = fields.String(required=True)
    display_name = fields.String()
    description = fields.String()
    summary_template = fields.String(
        metadata={
            "description": (
                "One-line preview with ``{key}`` placeholders for each field. The native row of the importer "
                "source-specs picker substitutes the current field values. Falls back to ``description`` when empty."
            )
        }
    )
    #: Python attribute renamed to avoid shadowing ``marshmallow.fields``;
    #: mapped back to the ``fields`` wire key.
    field_list = fields.List(fields.Dict(), attribute="fields", data_key="fields")


class MediaTypesListResponseSchema(Schema):
    """Response for ``GET /api/media-types``."""

    media_types = fields.List(fields.Nested(MediaTypeInfoSchema), required=True)


class EmbeddersListResponseSchema(Schema):
    """Response for ``GET /api/embedders``."""

    embedders = fields.List(fields.Nested(EmbedderInfoSchema), required=True)


class ClippersListResponseSchema(Schema):
    """Response for ``GET /api/clippers``.

    The clipper ``to_dict()`` payload is genuinely plugin-variable (concrete
    clippers add their own keys, e.g. ``duration``/``top_db``), so this stays
    an opaque ``fields.Dict()`` rather than a nested schema.  See the module
    docstring.
    """

    clippers = fields.List(fields.Dict(), required=True)


class CleanersListResponseSchema(Schema):
    """Response for ``GET /api/cleaners``.

    Same opaque ``fields.Dict()`` treatment as ``ClippersListResponseSchema``:
    a cleaner's ``to_dict()`` carries whatever parameters that cleaner declares,
    plus the cleaner-only ``default_enabled`` flag.
    """

    cleaners = fields.List(fields.Dict(), required=True)


class ConvertersListResponseSchema(Schema):
    """Response for ``GET /api/converters``."""

    converters = fields.List(fields.Nested(ConverterInfoSchema), required=True)


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

    All three fields are optional cache-key filters: when supplied, a cached
    pkl is only considered ``"ready"`` if it was produced with the same
    embedder / clipper / converter.  ``converter`` names a convert-on-load
    step (e.g. ``document2image`` for the Document demo tab) and only affects
    demos whose media type matches the converter's source type — those are
    cached under the ``{name}__{converter}`` pickle key.
    """

    embedder = fields.String(load_default="")
    clipper = fields.String(load_default="")
    converter = fields.String(load_default="")


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
    so reuse those nested schemas; registering distinct
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
                "Suggested initial relative sub-path for this source, for example "
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

    name = fields.String(required=True, validate=validate.Length(min=1, max=MAX_NAME_LENGTH))


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
    embedders = fields.List(
        fields.String(),
        load_default=None,
        metadata={
            "description": (
                "v3 trio of create-time embedder picks (text / patch / structural). "
                "When set, every name is embedded so a multi-embedder dataset is produced; "
                "omitted falls back to the single `embedder`."
            )
        },
    )
    clipper = fields.String(load_default="")
    clipper_params = fields.Dict(
        load_default=None,
        metadata={
            "description": (
                'Optional parameter overrides for `clipper` (e.g. `{"duration": 5.0}`). '
                "Only applied when `clipper` names a real, non-default clipper."
            )
        },
    )
    cleaners = fields.List(
        fields.Dict(),
        load_default=None,
        metadata={
            "description": (
                "Cleanup gates to run on each finished unit before embedding, as "
                '`[{"name": "image_exif_orient", "params": {}}]`. Order is ignored; '
                "cleaners always run last, after the clipper / converter chain."
            )
        },
    )
    converter = fields.String(load_default="")
    dataset_name = fields.String(load_default="")
    build_projection = fields.String(
        load_default="false",
        metadata={"description": "When 'true', compute + persist the 2-D Browse projection at ingest."},
    )
    merge_near_duplicates = fields.String(
        load_default="false",
        metadata={"description": "When 'true', collapse near-duplicate media into dupe sets at ingest."},
    )


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
    ``POST /api/dataset/combine`` / ``promote`` in ``staging.py``.

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


class ImporterFieldOptionsRequestSchema(Schema):
    """Body for ``POST /api/dataset/import/<importer_name>/options``."""

    field_key = fields.String(required=True, validate=validate.Length(min=1))
    values = fields.Dict(load_default=dict)


class FieldOptionsSchema(Schema):
    """A single dropdown option for a dynamic-options field.

    ``value`` is what the form submits; ``label`` is the friendly text
    shown in the dropdown.  For plain-string options the two coincide; for
    ``(value, label)`` tuple options they differ so a dropdown can submit
    an opaque id while displaying a human-readable name.
    """

    value = fields.String(required=True)
    label = fields.String(required=True)


class ImporterFieldOptionsResponseSchema(Schema):
    """Response for ``POST /api/dataset/import/<importer_name>/options``."""

    options = fields.List(fields.Nested(FieldOptionsSchema), required=True)


# ---------------------------------------------------------------------------
# Registry routes (vtsearch/routes/datasets/registry.py)
# ---------------------------------------------------------------------------


class DatasetsRegistryListResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry``.

    Each entry's inner shape is the registry record (plus a derived
    ``loaded`` flag and resolved ``clipper`` display name). Declared as
    ``fields.Dict`` to avoid duplicating the registry record schema;
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

    name = fields.String(required=True, validate=validate.Length(min=1, max=MAX_NAME_LENGTH))


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


class DatasetDomainShiftResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry/<id>/domain-shift``.

    Reports how typical the *active* dataset's items look under the named
    reference dataset's coverage atlas.  ``frac_atypical`` is the observed
    fraction of items with typicality p-value below ``alpha`` (roughly the
    shifted proportion); under no shift it stays near ``expected_atypical``.
    ``shifted`` is True when the excess is both statistically clear and
    practically large — a detector trained on the reference dataset should
    not be trusted on the active one without hands-on verification.
    """

    reference_dataset_id = fields.String(required=True)
    n_items = fields.Integer(required=True)
    alpha = fields.Float(required=True)
    frac_atypical = fields.Float(required=True)
    expected_atypical = fields.Float(required=True)
    z_score = fields.Float(required=True)
    median_pvalue = fields.Float(required=True)
    shifted = fields.Boolean(required=True)


class DatasetRegistryStatsResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry/<id>/stats``.

    A superset of the Dashboard grid row: ``name``, ``media_type``,
    ``num_items``, ``created_at``, ``expires_at``, ``created_by`` and
    ``readers`` are the grid's own columns, so the Stats window can show
    everything the grid does while it covers the grid up.
    """

    name = fields.String(required=True)
    media_type = fields.String(required=True)
    num_items = fields.Integer(required=True)
    num_dupes = fields.Integer(required=True)
    file_type_counts = fields.Dict(
        keys=fields.String(),
        values=fields.Integer(),
        required=True,
        metadata={
            "description": (
                "File type → item count. The type is the item's filename extension, or the format "
                "sniffed from its bytes when it has none (a service importer may name items after an "
                "opaque content id). Items no signal could type land in a parenthesised "
                '"(unknown)" bucket.'
            )
        },
    )
    created_at = fields.Raw(allow_none=True)
    expires_at = fields.Raw(allow_none=True)
    created_by = fields.String(required=True)
    readers = fields.List(fields.String(), required=True)
    ingest_started_at = fields.Raw(allow_none=True)
    ingest_finished_at = fields.Raw(allow_none=True)
    origin = fields.String(required=True)
    source = fields.Dict(required=True)
    clipper = fields.String(required=True)
    embedder = fields.String(required=True)


class DuplicateSetMemberSchema(Schema):
    """One member of a collapsed duplicate set (its pre-collapse provenance)."""

    md5 = fields.String(required=True)
    filename = fields.String(required=True)
    category = fields.String(required=True)
    origin_name = fields.String(required=True)
    importer = fields.String(required=True)


class DuplicateSetSchema(Schema):
    """One collapsed duplicate set: its display name and every member."""

    name = fields.String(required=True)
    members = fields.List(fields.Nested(DuplicateSetMemberSchema), required=True)


class DatasetRegistryDuplicatesResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry/<id>/duplicates``."""

    duplicate_sets = fields.List(fields.Nested(DuplicateSetSchema), required=True)


__all__ = [
    "BrowseMediaFilesQuerySchema",
    "BrowseMediaFilesResponseSchema",
    "BrowseMediaFilesSelectRequestSchema",
    "BrowseMediaFilesSelectResponseSchema",
    "CancelDatasetLoadResponseSchema",
    "CleanersListQuerySchema",
    "CleanersListResponseSchema",
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
    "DatasetCombineResolutionSchema",
    "DatasetDomainShiftResponseSchema",
    "DatasetImportersListResponseSchema",
    "DatasetLoadDemoRequestSchema",
    "DatasetLoadFolderRequestSchema",
    "DatasetLoadSourceRequestSchema",
    "DatasetLoadStartedResponseSchema",
    "DatasetRegistryDuplicatesResponseSchema",
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
    "FieldOptionsSchema",
    "ImporterFieldOptionsRequestSchema",
    "ImporterFieldOptionsResponseSchema",
    "MediaTypesListResponseSchema",
]
