"""Schemas for the dataset registry routes (``vtsearch/routes/datasets/registry.py``)."""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtsearch.schemas.common import list_of_strings

#: Upper bound on user-supplied dataset names, mirroring
#: ``vtsearch.schemas.detectors.MAX_NAME_LENGTH``.  A name past this is already
#: unusable for display, and capping it here stops an absurdly long name from
#: ever reaching a filesystem path (and the uncaught ``OSError`` /
#: absolute-path leak that would follow).
MAX_NAME_LENGTH = 128


class DatasetRegistryEntrySchema(Schema):
    """One entry of ``GET /api/datasets/registry``.

    The persisted registry record (written in exactly one place,
    :func:`vtscore.datasets.registry.register_dataset`) plus the fields the
    route derives per request: ``loaded``, ``embedders_by_type``, the
    ``clipper`` display-name resolution, and the legacy fallbacks for
    ``bound_embedders`` / ``embedder_types``.  The writer's key set is closed,
    so this is a strict schema.

    Optionality mirrors what the frontend treats as optional (same convention
    as :class:`MediaTypeInfoSchema`), not what the current writer always
    emits: entries persisted by older versions can legitimately lack the
    newer fields.
    """

    id = fields.String(required=True)
    name = fields.String(required=True)
    media_type = fields.String(required=True)
    loaded = fields.Boolean(metadata={"description": "Whether this dataset is currently resident in memory."})
    num_items = fields.Integer(metadata={"description": "Item count after near-duplicate collapsing."})
    num_dupes = fields.Integer(metadata={"description": "How many items were collapsed as near-duplicates."})
    pkl_path = fields.String(metadata={"description": "Server-side path of the dataset pickle."})
    origin = fields.String(metadata={"description": "Name of the importer that produced this dataset."})
    source = fields.Dict(
        allow_none=True,
        metadata={
            "description": (
                "The importer's origin dict for this dataset (importer name plus its field values). Inner keys "
                "are importer-specific, so this stays an opaque map."
            )
        },
    )
    clipper = fields.String(
        metadata={
            "description": (
                'Resolved clipper display name, or ``"-"`` when the dataset used its media type\'s default clipper.'
            )
        }
    )
    embedder = fields.String(
        metadata={"description": "Name of the embedder this dataset's media were vectorised with."}
    )
    embedder_types = fields.List(
        fields.String(),
        metadata={
            "description": (
                "The embedder *types* this dataset supplies (``semantic`` / ``patch_semantic`` / ``structural``); "
                "a v3 trio dataset can supply several. Drives the detector/dataset compatibility gate."
            )
        },
    )
    bound_embedders = fields.List(
        fields.String(), metadata={"description": "Every concrete embedder this dataset binds (primary first)."}
    )
    embedders_by_type = fields.Dict(
        keys=fields.String(),
        values=fields.String(),
        metadata={
            "description": (
                'One concrete embedder per type it binds, e.g. ``{"semantic": "siglip", "patch_semantic": '
                '"dinov3_patch"}``. Drives the Combine-Datasets conflict detector.'
            )
        },
    )
    readers = fields.List(
        fields.String(),
        metadata={
            "description": (
                'Usernames granted read access. Empty means creator-only; ``["*"]`` means visible to everyone.'
            )
        },
    )
    created_by = fields.String()
    created_at = fields.Float(metadata={"description": "Unix timestamp (seconds) at which the dataset was registered."})
    file_type_counts = fields.Dict(
        keys=fields.String(),
        values=fields.Integer(),
        metadata={"description": "Per-extension file counts observed during ingest."},
    )
    ingest_started_at = fields.Float(allow_none=True)
    ingest_finished_at = fields.Float(allow_none=True)
    expires_at = fields.Float(
        allow_none=True,
        metadata={
            "description": (
                "Unix timestamp (seconds) at which this dataset ages off and is automatically removed; "
                "``null``/absent means it never expires."
            )
        },
    )
    import_group = fields.String(
        allow_none=True,
        metadata={
            "description": (
                "Id shared by every dataset one multi-dataset import produced, so siblings can be found from any "
                "one of them; ``null`` for a single-dataset import."
            )
        },
    )
    output_category = fields.String(
        metadata={
            "description": (
                "The ingestion category (a media type id) this dataset stands for within its import group, e.g. "
                '``"face"`` or ``"document"``; it can differ from ``media_type`` (a Document dataset holds '
                "rendered page images). Empty for a single-dataset import."
            )
        },
    )


class DatasetsRegistryListResponseSchema(Schema):
    """Response for ``GET /api/datasets/registry``."""

    datasets = fields.List(fields.Nested(DatasetRegistryEntrySchema), required=True)


class DatasetRegistryLoadResponseSchema(Schema):
    """Response for ``POST /api/datasets/registry/<id>/load``.

    Successful kickoff returns ``task_id``; the "already loaded" path
    returns the same envelope with an empty ``task_id``.
    """

    ok = fields.Boolean(required=True)
    message = fields.String(required=True)
    task_id = fields.String(load_default="")


class DatasetAutofindRequestSchema(Schema):
    """Body for ``POST /api/datasets/registry/<id>/autofind``.

    The body is optional.  Without ``detector_ids`` the run scores with the
    caller's AutoFind detectors (the dataset ⋯ **Run AutoFind**); with it, with
    exactly the detectors those registry ids name, drafts included (the
    Dashboard's big **Find** button).  Declared as ``fields.Raw`` with
    :func:`list_of_strings` so a non-string id is a 422 rather than coerced.
    """

    detector_ids = fields.Raw(
        load_default=None,
        validate=list_of_strings,
        metadata={
            "description": (
                "Registry ids of the detectors to run, in place of the caller's AutoFind list. "
                "Omit to run the AutoFind list."
            ),
            "type": "array",
            "items": {"type": "string"},
        },
    )


class DatasetRegistryOkResponseSchema(Schema):
    """Bare ``{"ok": true}`` response (unload, delete, dismissing a finished task)."""

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
        validate=list_of_strings,
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

    Reports how typical the *active* dataset's items look against the named
    reference dataset's own embeddings (a split-conformal kNN, #3348).
    ``n_items`` is the number of active items scored (at most 4 000, sampled).
    ``frac_atypical`` is the observed
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


class DatasetCoverageAtlasStatusSchema(Schema):
    """A loaded dataset's coverage-atlas state, on its stats response."""

    built = fields.Boolean(required=True, metadata={"description": "Whether the dataset has a coverage atlas."})
    building = fields.Boolean(
        required=True,
        metadata={"description": "Whether an on-demand build (POST .../coverage-atlas) is running."},
    )
    auto_build_threshold = fields.Integer(
        required=True,
        metadata={"description": "Item count above which loading leaves the atlas unbuilt."},
    )


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
    coverage_atlas = fields.Nested(
        DatasetCoverageAtlasStatusSchema,
        required=True,
        allow_none=True,
        metadata={"description": "The dataset's coverage-atlas state; null when the dataset is not loaded."},
    )


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
    "DatasetAutofindRequestSchema",
    "DatasetDomainShiftResponseSchema",
    "DatasetRegistryDuplicatesResponseSchema",
    "DatasetRegistryLoadResponseSchema",
    "DatasetRegistryOkResponseSchema",
    "DatasetRegistryPreloadEmbedderResponseSchema",
    "DatasetRegistryReadersRequestSchema",
    "DatasetRegistryReadersResponseSchema",
    "DatasetRegistryRenameRequestSchema",
    "DatasetRegistryRenameResponseSchema",
    "DatasetRegistryStatsResponseSchema",
    "DatasetsRegistryListResponseSchema",
]
