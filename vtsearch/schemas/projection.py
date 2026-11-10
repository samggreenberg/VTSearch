"""Schemas for the VTSBrowse projection endpoints."""

from __future__ import annotations

from typing import Any

from marshmallow import Schema, ValidationError, fields, validate

#: Most tiles one ``GET /api/projection/tiles`` may ask for. The canvas asks in
#: batches of 64; the cap only bounds what a hand-made URL can make the server do.
MAX_TILES_PER_BATCH = 128


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
    has_labels = fields.Boolean(
        load_default=None,
        metadata={
            "description": (
                "Whether region signpost labels exist for this projection "
                "(see GET /api/projection/labels). False until a labeler has "
                "run for the current layout."
            ),
        },
    )
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
    step = fields.Integer(
        load_default=None,
        metadata={"description": "Which coarse build phase is running (1-based)."},
    )
    total_steps = fields.Integer(
        load_default=None,
        metadata={"description": "How many coarse phases the whole build has."},
    )
    overall = fields.Float(
        load_default=None,
        metadata={
            "description": (
                "Whole-job completion fraction (0..1) stitched from the current "
                "phase and its within-phase counts, so one bar fills across the "
                "entire build instead of restarting at each phase."
            ),
        },
    )
    overall_step_end = fields.Float(
        load_default=None,
        metadata={
            "description": (
                "Whole-job fraction (0..1) at which the running phase's slice "
                "ends. With ``overall`` parked at the slice floor during a "
                "count-less phase (the UMAP fit), the pair brackets the build's "
                "true position so the client can shade the span as a bounded "
                "indeterminate zone."
            ),
        },
    )
    eta_seconds = fields.Float(
        load_default=None,
        metadata={
            "description": (
                "Estimated seconds remaining for the whole build, smoothed and "
                "snapped to a coarse ladder so the displayed figure does not "
                "twitch. ``null`` until the build has run long enough to "
                "extrapolate from."
            ),
        },
    )
    error = fields.String(load_default=None)


class ProjectionBuildResponseSchema(Schema):
    """Response for ``POST /api/projection/build``."""

    status = fields.String(required=True)
    job_id = fields.String(load_default=None)
    projection_id = fields.String(load_default=None)


class ProjectionBuildRequestSchema(Schema):
    """Body for ``POST /api/projection/build``; every field is optional."""

    ids = fields.List(
        fields.Integer(),
        load_default=None,
        allow_none=True,
        metadata={"description": "Project only these media, into the ephemeral subset layout."},
    )
    force = fields.Boolean(
        load_default=False,
        metadata={"description": "Discard the existing layout and re-fit UMAP (the Browser's Re-project)."},
    )


class ProjectionSubsetQuerySchema(Schema):
    """Query for the projection reads: which of the two layouts to read."""

    subset = fields.Boolean(
        load_default=False,
        metadata={"description": "Read the ephemeral subset layout rather than the full-dataset one."},
    )


class TileQuerySchema(ProjectionSubsetQuerySchema):
    """Query for ``GET /api/projection/tiles/...``."""

    v = fields.String(
        load_default=None,
        metadata={
            "description": (
                "Cache token (``<projection_id>:<content_version>``) the client appends so each "
                "layout's immutable tiles get their own URL. The server ignores it."
            )
        },
    )


class TileCoordList(fields.Field):
    """``tx:ty,tx:ty,…`` in a query string, as a list of ``(tx, ty)`` pairs.

    Tile coordinates may be negative (``-1:3``), so the pair separator is a
    colon rather than the minus-prone comma alone.
    """

    def _deserialize(self, value: Any, attr: str | None, data: Any, **kwargs: Any) -> list[tuple[int, int]]:
        if not isinstance(value, str) or not value.strip():
            raise ValidationError("Expected tiles as tx:ty pairs separated by commas.")
        coords: list[tuple[int, int]] = []
        for part in value.split(","):
            tx, sep, ty = part.strip().partition(":")
            try:
                if not sep:
                    raise ValueError
                coords.append((int(tx), int(ty)))
            except ValueError:
                raise ValidationError(f"Bad tile {part!r}: expected tx:ty, e.g. 3:-1.") from None
        if len(coords) > MAX_TILES_PER_BATCH:
            raise ValidationError(f"At most {MAX_TILES_PER_BATCH} tiles per request.")
        return coords


class TileBatchQuerySchema(TileQuerySchema):
    """Query for ``GET /api/projection/tiles``."""

    level = fields.Integer(
        required=True,
        validate=validate.Range(min=0),
        metadata={"description": "Pyramid level every requested tile is at."},
    )
    tiles = TileCoordList(
        required=True,
        metadata={
            "description": (
                f"The tiles to return, as ``tx:ty`` pairs separated by commas (``0:0,1:0,-1:2``), "
                f"at most {MAX_TILES_PER_BATCH}. Answered in this order, repeats dropped."
            ),
            "type": "string",
        },
    )


class SubsetRemoveRequestSchema(Schema):
    """Body for ``POST /api/projection/subset/remove``."""

    ids = fields.List(
        fields.Integer(),
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "Media ids to drop from the current subset browse."},
    )


class RegionLabelSchema(Schema):
    """One region signpost — a named region of the projection map."""

    level = fields.Float(
        required=True,
        metadata={
            "description": (
                "Pyramid zoom level the sign belongs to (0 = coarsest). May be "
                "fractional; the canvas interpolates visibility on a continuous axis."
            ),
        },
    )
    x = fields.Float(required=True, metadata={"description": "Anchor (projection space)."})
    y = fields.Float(required=True, metadata={"description": "Anchor (projection space)."})
    text = fields.String(required=True)
    score = fields.Float(
        load_default=1.0,
        metadata={"description": "Naming confidence; the de-clutter tiebreak."},
    )
    source = fields.String(
        load_default="",
        metadata={"description": 'Which namer produced the sign (e.g. "keyphrase", "llm").'},
    )
    has_coarser = fields.Boolean(
        load_default=True,
        metadata={
            "description": (
                "Whether a coarser sign names this region one zoom band out (a "
                "parent). False for a root region, which the canvas keeps "
                "visible when zoomed out instead of fading in under a parent."
            ),
        },
    )
    has_finer = fields.Boolean(
        load_default=True,
        metadata={
            "description": (
                "Whether a finer sign names this region one zoom band in (a "
                "child). False for a leaf region, which the canvas keeps visible "
                "when zoomed in instead of expiring with nothing to hand off to."
            ),
        },
    )


class ProjectionLabelsResponseSchema(Schema):
    """Response for ``GET /api/projection/labels``."""

    status = fields.String(
        required=True,
        metadata={"description": '"ready" when a projection exists (labels may be empty), else "idle".'},
    )
    projection_id = fields.String(load_default=None)
    labels = fields.List(fields.Nested(RegionLabelSchema), required=True)


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


class TileBatchResponseSchema(Schema):
    """Response for ``GET /api/projection/tiles``."""

    tiles = fields.List(
        fields.Nested(TileResponseSchema),
        required=True,
        metadata={"description": "One entry per requested tile, in request order."},
    )


__all__ = [
    "HexCellSchema",
    "LevelMetaSchema",
    "ProjectionBuildResponseSchema",
    "ProjectionLabelsResponseSchema",
    "ProjectionMetaSchema",
    "RegionLabelSchema",
    "SubsetRemoveRequestSchema",
    "TileBatchQuerySchema",
    "TileBatchResponseSchema",
    "TileResponseSchema",
]
