"""Schemas for the dataset UI helpers (``vtsearch/routes/datasets/ui.py``).

The demo dataset list, the browse-media-files picker, media-type detection,
and the dashboard's disk and RAM probes.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtsearch.schemas.datasets.listings import ConverterInfoSchema
from vtsearch.schemas.file_browser import BrowseDirectoryEntrySchema, BrowseFileEntrySchema

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
    available_converters = fields.List(fields.Nested(ConverterInfoSchema), required=True)
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
                "Suggested initial relative sub-path for this source. Empty for "
                "every source, since each one's root is already the right "
                "starting point."
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
    schema only narrows the type, so a ``limit`` that is not an integer is
    rejected with a 422.
    """

    source = fields.String(
        load_default="folder",
        metadata={
            "description": "One of ``demo:<name>``, ``folder``, or ``server_fs`` (matches ``/api/browse-media-files``)."
        },
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
# /api/dashboard/disk-usage
# ---------------------------------------------------------------------------


class _DashboardHeadroomSchema(Schema):
    """Whether a usage probe's free bytes are short, measured in datasets."""

    dataset_bytes = fields.Integer(
        required=True,
        metadata={
            "description": "Footprint of one dataset, the unit headroom is measured in: the largest "
            "registered dataset's on-disk size (pkl plus sidecars), or a stand-in when none is registered."
        },
    )
    dataset_bytes_source = fields.String(
        required=True,
        validate=validate.OneOf(["largest", "default"]),
        metadata={"description": "``largest`` when measured from the registry, ``default`` for the stand-in."},
    )
    low = fields.Boolean(
        required=True,
        metadata={
            "description": "True when ``free`` holds fewer than a few more datasets of ``dataset_bytes``; "
            "the Dashboard shows the bar on its Default setting only then."
        },
    )


class DashboardDiskUsageResponseSchema(_DashboardHeadroomSchema):
    """Response for ``GET /api/dashboard/disk-usage``."""

    total = fields.Integer(required=True)
    used = fields.Integer(required=True)
    free = fields.Integer(required=True)
    path = fields.String(required=True)


class DashboardRamUsageResponseSchema(_DashboardHeadroomSchema):
    """Response for ``GET /api/dashboard/ram-usage``."""

    total = fields.Integer(required=True)
    used = fields.Integer(required=True)
    free = fields.Integer(required=True)


__all__ = [
    "BrowseMediaFilesQuerySchema",
    "BrowseMediaFilesResponseSchema",
    "BrowseMediaFilesSelectRequestSchema",
    "BrowseMediaFilesSelectResponseSchema",
    "DashboardDiskUsageResponseSchema",
    "DashboardRamUsageResponseSchema",
    "DemoCategoriesResponseSchema",
    "DemoDatasetListQuerySchema",
    "DemoDatasetListResponseSchema",
]
