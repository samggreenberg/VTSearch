"""Schemas for the VTSBrowse projection endpoints."""

from __future__ import annotations

from marshmallow import Schema, fields


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


class HexCellSchema(Schema):
    q = fields.Integer(required=True)
    r = fields.Integer(required=True)
    cx = fields.Float(required=True)
    cy = fields.Float(required=True)
    count = fields.Integer(required=True)
    rep_id = fields.Integer(required=True)


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
    "TileResponseSchema",
]
