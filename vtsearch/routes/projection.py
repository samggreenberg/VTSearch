"""VTSBrowse projection routes: build, meta, and tile endpoints.

Kick off a background UMAP + tile pyramid build for the active dataset,
query its status/metadata, and stream tiles to the browse canvas.

HTTP only.  The lifecycle these endpoints drive — first fit, restore from the
dataset container, re-bin to the other shape, forced re-projection, ephemeral
subset fits, and in-place subset culls — lives in
:mod:`vtscore.projection.service`, which is Flask-free and takes an explicit
context.  What is left here is request parsing, error-code mapping, and cache
headers.

The pyramid tiles the projection as hexagons or squares, but the shape is a
fixed property of the dataset's **media type** — squares for browsable-thumbnail
media (image/video/document), hexes for the rest (audio/text) — not a per-request
choice.  Every endpoint derives the shape from the active dataset
(:func:`vtscore.projection.service.shape_for`); the client never sends it.  The
meta response reports the resolved ``bin_shape`` so the canvas knows which
lattice to draw.

Every ``vtscore.projection`` import here is function-local on purpose: the
package pulls numba/UMAP, and this module is imported at app startup by
``vtsearch.routes``.  That import cost belongs on the first Browse request,
not on every boot.
"""

from __future__ import annotations

import logging

from flask import after_this_request, request
from flask_smorest import Blueprint, abort

from vtsearch.schemas.projection import (
    ProjectionBuildResponseSchema,
    ProjectionLabelsResponseSchema,
    ProjectionMetaSchema,
    SubsetRemoveRequestSchema,
    TileResponseSchema,
)

logger = logging.getLogger(__name__)

projection_bp = Blueprint(
    "projection",
    __name__,
    description="VTSBrowse projection: UMAP layout + hex/square tile pyramid.",
)


def _is_subset(value: str | None) -> bool:
    """Whether a request targets the ephemeral subset projection."""
    return str(value).lower() in ("1", "true", "yes")


def _requested_ids(raw) -> list[int]:
    """Validate a request body's ``ids`` into a list of media ids.

    A malformed body is a 400 — the client sent the wrong shape.  An *empty*
    selection is not: the request is well-formed and there is simply nothing
    to project, which the service reports as a 409 like every other
    nothing-to-project case.
    """
    if not isinstance(raw, list):
        abort(400, message="`ids` must be a list of media ids.")
    try:
        requested = [int(c) for c in raw]
    except (TypeError, ValueError):
        abort(400, message="`ids` must be a list of integer media ids.")
    return requested


@projection_bp.route("/api/projection/build", methods=["POST"])
@projection_bp.response(200, ProjectionBuildResponseSchema)
@projection_bp.alt_response(409, description="Dataset is empty or has no embeddings.")
def build_projection():
    """Kick off (or short-circuit) a build of the dataset's projection.

    Body: ``{"ids": [...]?, "force": bool?}``.  The bin shape is not a request
    parameter — it is derived from the dataset's media type.

    Returns immediately.  If the pyramid is already cached or persisted, returns
    ``"ready"``.  Only the first build of a dataset, where no layout exists yet,
    runs UMAP in the background and returns a ``job_id`` for polling via
    ``GET /api/projection/meta``.

    ``ids`` projects only the supplied media (e.g. the positive results of a
    Find run), fitting UMAP on just their high-d vectors into an ephemeral
    subset layout held alongside the full-dataset one.

    ``force`` overrides every short-circuit: it discards the existing layout
    (cached + persisted for a full build, or the in-memory subset layout for an
    ``ids`` build) and runs a fresh UMAP fit over the current items, returning a
    new ``projection_id``.  This is the Browser's "Re-project" action.
    """
    from vtscore.projection import service
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    body = request.get_json(silent=True) or {}
    raw_ids = body.get("ids")
    ids = None if raw_ids is None else _requested_ids(raw_ids)

    try:
        return service.build_layout(ctx, ids=ids, force=bool(body.get("force")))
    except ValueError as exc:
        # The service's own "nothing to project" signals and the
        # embedding-matrix builder's ValueErrors mean the same thing to a
        # client: the dataset can't be laid out as it currently stands.
        abort(409, message=str(exc))


@projection_bp.route("/api/projection/subset/remove", methods=["POST"])
@projection_bp.arguments(SubsetRemoveRequestSchema)
@projection_bp.response(200, ProjectionMetaSchema)
@projection_bp.alt_response(409, description="No subset projection is currently built.")
def remove_from_subset(body: dict):
    """Drop ids from the current subset browse **without re-fitting UMAP**.

    The remaining points keep their exact 2-D positions and bins; only counts
    and representatives change.  The ``projection_id`` (layout identity) is
    preserved and a bumped ``content_version`` busts the otherwise-immutable
    tile cache.  Returns the updated subset meta for the dataset's shape.

    Powers the Browser's "Remove from Good" cull, which marks the items Bad via
    ``/api/medias/vote-bulk`` and then calls this to make them disappear.
    """
    from vtscore.projection import service
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    try:
        service.remove_subset_ids(ctx, body["ids"])
    except ValueError as exc:
        abort(409, message=str(exc))

    return service.layout_meta(ctx, service.shape_for(ctx), subset=True)


@projection_bp.route("/api/projection/meta", methods=["GET"])
@projection_bp.response(200, ProjectionMetaSchema)
def projection_meta():
    """Return projection/pyramid metadata and build status for the dataset.

    The bin shape is derived from the dataset's media type, not a query
    parameter.  When the pyramid is ready, includes bounds, zoom levels, cell
    sizing, point count, the dataset's media type, and the ``bin_shape`` the
    canvas should render.
    """
    from vtscore.projection import service
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    return service.layout_meta(
        ctx,
        service.shape_for(ctx),
        subset=_is_subset(request.args.get("subset")),
    )


@projection_bp.route("/api/projection/labels", methods=["GET"])
@projection_bp.response(200, ProjectionLabelsResponseSchema)
def projection_labels():
    """Return the region signpost labels for the active projection.

    The "street signs" the browse canvas letters over the map (see
    ``docs/plans/vtsbrowse-toponymy.md``): one entry per named region, each a
    text + a 2-D anchor in the frozen layout + the pyramid zoom level it
    belongs to.  The payload is tiny — the client fetches it once per
    ``projection_id``.

    Labels are optional decoration: a projection whose labeling pipeline
    hasn't run (or that predates it) answers an empty list, not an error, and
    ``status`` is ``"idle"`` only when no projection is built at all.  A label
    set computed for a *different* layout than the active pyramid's is treated
    as absent — anchors are meaningless off their own layout.
    """
    from vtscore.projection import service
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    return service.labels_payload(
        ctx,
        service.shape_for(ctx),
        subset=_is_subset(request.args.get("subset")),
    )


@projection_bp.route(
    "/api/projection/tiles/<int:level>/<int(signed=True):tx>/<int(signed=True):ty>",
    methods=["GET"],
)
@projection_bp.response(200, TileResponseSchema)
@projection_bp.alt_response(404, description="Tile not found or projection not ready.")
def get_tile(level: int, tx: int, ty: int):
    """Return the cells for one tile of the dataset's pyramid at ``(level, tx, ty)``.

    The bin shape is derived from the dataset's media type.  Because the
    projection is frozen at ingest, tiles are immutable for the life of the
    dataset and can be cached aggressively by the client.
    """
    from vtscore.projection import service
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    payload = service.tile_payload(
        ctx,
        service.shape_for(ctx),
        level,
        tx,
        ty,
        subset=_is_subset(request.args.get("subset")),
    )
    if payload is None:
        abort(404, message="Projection not built yet — call POST /api/projection/build first.")

    # Tiles are frozen at ingest, so let the browser serve repeat visits from its
    # HTTP cache without a round-trip — this is what keeps the hex grid from
    # blanking when you pan/zoom back over ground you've already seen. The tile
    # URL omits the dataset id (it rides on the X-Dataset-Id header), so we must
    # Vary on it or the cache could hand one dataset's tiles to another. Scoped
    # to this response only, and registered after the not-built check so a
    # not-yet-built projection is never cached. ``?subset`` is already part of
    # the URL.
    @after_this_request
    def _cache_tile(response):  # type: ignore[unused-ignore]
        response.headers["Cache-Control"] = "private, max-age=3600, immutable"
        response.vary.add("X-Dataset-Id")
        return response

    return payload


__all__ = ["projection_bp"]
