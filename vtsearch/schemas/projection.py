"""Schemas for the VTSBrowse projection endpoints."""

from __future__ import annotations

from marshmallow import Schema, fields, validate


class LevelMetaSchema(Schema):
    level = fields.Integer(required=True)
    radius = fields.Float(required=True)
    n_cells = fields.Integer(required=True)


class ProjectionMetaSchema(Schema):
    """Response for ``GET /api/projection/meta``."""

    status = fields.String(required=True, metadata={"description": "idle | building | ready | error"})
    projection_id = fields.String(load_default=None)
    bin_shape = fields.String(load_default=None, metadata={"description": "hex | square"})
    bounds = fields.List(fields.Float(), load_default=None)
    base_radius = fields.Float(load_default=None)
    tile_span = fields.Float(load_default=None)
    point_count = fields.Integer(load_default=None)
    levels = fields.List(fields.Nested(LevelMetaSchema), load_default=None)
    media_type = fields.String(load_default=None)
    method = fields.String(load_default=None)
    content_version = fields.Integer(
        load_default=0,
        metadata={
            "description": (
                "Monotonic version of the projection's membership. Stays at 0 for "
                "full-dataset projections; bumped when items are removed from a "
                "subset browse in place. Combined with ``projection_id`` to bust "
                "the immutable tile cache without re-framing the canvas."
            ),
        },
    )
    # Build progress (when status == "building")
    job_id = fields.String(load_default=None)
    current = fields.Integer(load_default=None)
    total = fields.Integer(load_default=None)
    message = fields.String(load_default=None)
    error = fields.String(load_default=None)


class ProjectionBuildResponseSchema(Schema):
    """Response for ``POST /api/projection/build``."""

    status = fields.String(required=True)
    job_id = fields.String(load_default=None)
    projection_id = fields.String(load_default=None)


class SubsetRemoveRequestSchema(Schema):
    """Body for ``POST /api/projection/subset/remove``."""

    ids = fields.List(
        fields.Integer(),
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "Media ids to drop from the current subset browse."},
    )
    shape = fields.String(
        load_default="hex",
        validate=validate.OneOf(["hex", "square"]),
        metadata={"description": "Bin shape whose updated meta to return (hex | square)."},
    )


class HexCellSchema(Schema):
    q = fields.Integer(required=True)
    r = fields.Integer(required=True)
    cx = fields.Float(required=True)
    cy = fields.Float(required=True)
    count = fields.Integer(required=True)
    rep_id = fields.Integer(required=True)
    member_ids = fields.List(
        fields.Integer(),
        load_default=None,
        metadata={"description": "All media ids aggregated in this cell (for selection)."},
    )


class TileResponseSchema(Schema):
    """Response for ``GET /api/projection/tiles/<level>/<tx>/<ty>``."""

    level = fields.Integer(required=True)
    tx = fields.Integer(required=True)
    ty = fields.Integer(required=True)
    cells = fields.List(fields.Nested(HexCellSchema), required=True)


__all__ = [
    "HexCellSchema",
    "LevelMetaSchema",
    "ProjectionBuildResponseSchema",
    "ProjectionMetaSchema",
    "SubsetRemoveRequestSchema",
    "TileResponseSchema",
]
