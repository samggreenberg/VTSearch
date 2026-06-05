"""VTSBrowse projection routes: build, meta, and tile endpoints.

Kick off a background UMAP + tile pyramid build for the active dataset,
query its status/metadata, and stream tiles to the browse canvas.

The pyramid can tile the projection as hexagons (default) or squares; the
``bin_shape`` selector threads through every endpoint.  The UMAP projection
itself is shape-independent and shared: switching shapes only re-bins the
frozen 2-D coordinates (fast), never re-fits UMAP.
"""

from __future__ import annotations

import logging

from flask import after_this_request, request
from flask_smorest import Blueprint, abort

from vtsearch.schemas.projection import (
    ProjectionBuildResponseSchema,
    ProjectionMetaSchema,
    TileResponseSchema,
)

logger = logging.getLogger(__name__)

projection_bp = Blueprint(
    "projection",
    __name__,
    description="VTSBrowse projection: UMAP layout + hex/square tile pyramid.",
)

#: Bin shapes the browse canvas can request.  Mirrors
#: ``vtscore.projection.BIN_SHAPES``; kept as a local literal so the route
#: validation does not import the (numba-pulling) projection package on the
#: request path.
_VALID_SHAPES = ("hex", "square")
_DEFAULT_SHAPE = "hex"


def _resolve_shape(value: str | None) -> str:
    """Validate a requested bin shape, defaulting to hex."""
    shape = value or _DEFAULT_SHAPE
    if shape not in _VALID_SHAPES:
        abort(400, message=f"Unknown bin shape {shape!r}; expected one of {_VALID_SHAPES}.")
    return shape


def _is_subset(value: str | None) -> bool:
    """Whether a request targets the ephemeral subset projection."""
    return str(value).lower() in ("1", "true", "yes")


def _subset_meta(ctx, bin_shape: str) -> dict:
    """Return projection/pyramid metadata + build status for the subset layout."""
    from vtscore.concurrency.async_jobs import projection_jobs

    pyr = ctx._subset_pyramids.get(bin_shape)
    if pyr is not None:
        meta = pyr.meta()
        meta["status"] = "ready"
        meta["media_type"] = _media_type_for(ctx)
        meta["method"] = ctx._subset_projection.method if ctx._subset_projection else None
        return meta

    if ctx._subset_job_id:
        job = projection_jobs.get(ctx._subset_job_id)
        if job is not None:
            if job.status in ("running", "pending"):
                return {
                    "status": "building",
                    "job_id": job.job_id,
                    "current": job.current,
                    "total": job.total,
                    "message": job.message,
                }
            if job.status == "error":
                return {"status": "error", "error": job.error or "projection build failed"}

    return {"status": "idle"}


def _media_type_for(ctx) -> str:
    """Return the media type of the first item in the dataset, or ``""``."""
    if not ctx.medias:
        return ""
    first = next(iter(ctx.medias.values()))
    return first.get("media_type", "audio")


def _pkl_path_for(dataset_id: str) -> str | None:
    """Return the pkl_path from the dataset registry, or ``None``."""
    from vtscore.datasets.registry import get_dataset

    entry = get_dataset(dataset_id)
    if entry is None:
        return None
    return entry.get("pkl_path") or None


def _try_load_persisted(ctx, sorted_ids: list[int], bin_shape: str) -> dict | None:
    """Try to restore the *bin_shape* pyramid from the dataset container.

    Returns a ready-response dict on success, or ``None`` if no valid
    persisted pyramid for that shape is available.
    """
    pkl_path = _pkl_path_for(ctx.dataset_id)
    if pkl_path is None:
        return None
    from vtscore.datasets.container import read_projection

    loaded = read_projection(pkl_path, bin_shape)
    if loaded is None:
        return None
    proj, pyr = loaded
    if set(proj.ids) != set(sorted_ids):
        logger.info("Persisted %s projection ids mismatch; will recompute.", bin_shape)
        return None
    ctx._projection = proj
    ctx._pyramids[bin_shape] = pyr
    return {"status": "ready", "projection_id": pyr.projection_id}


def _load_projection_coords(ctx, sorted_ids: list[int]):
    """Populate ``ctx._projection`` from any persisted bin shape, or return ``None``.

    The 2-D coordinates are shared across bin shapes, so any stored pyramid
    yields them — letting a shape that was never persisted be re-binned from
    the frozen layout instead of re-fitting UMAP.  The pyramid that supplied
    the coordinates is cached too, since it was deserialized anyway.
    """
    pkl_path = _pkl_path_for(ctx.dataset_id)
    if pkl_path is None:
        return None
    from vtscore.datasets.container import read_projection

    for shape in _VALID_SHAPES:
        loaded = read_projection(pkl_path, shape)
        if loaded is None:
            continue
        proj, pyr = loaded
        if set(proj.ids) != set(sorted_ids):
            continue
        ctx._projection = proj
        ctx._pyramids.setdefault(pyr.bin_shape, pyr)
        return proj
    return None


def _rebin_from_existing_layout(ctx, sorted_ids: list[int], bin_shape: str) -> dict | None:
    """Serve *bin_shape* without a UMAP fit, if at all possible.

    Tries, in order: this shape's persisted pyramid, then re-binning the shared
    frozen layout (already in memory or persisted under another shape).  Returns
    a ready-response dict, or ``None`` when no layout exists yet and UMAP must
    run.
    """
    from vtscore.projection import build_pyramid

    persisted = _try_load_persisted(ctx, sorted_ids, bin_shape)
    if persisted is not None:
        return persisted

    proj = ctx._projection
    if proj is None or set(proj.ids) != set(sorted_ids):
        proj = _load_projection_coords(ctx, sorted_ids)
    if proj is None:
        return None

    pyr = build_pyramid(proj, bin_shape=bin_shape)
    ctx._pyramids[bin_shape] = pyr
    _persist_projection(ctx.dataset_id, proj, pyr)
    return {"status": "ready", "projection_id": pyr.projection_id}


def _start_umap_build(ctx, sorted_ids: list[int], matrix, bin_shape: str) -> dict:
    """Start (or reuse) the background UMAP fit + pyramid build for *bin_shape*."""
    from vtscore.concurrency.async_jobs import projection_jobs
    from vtscore.projection import build_pyramid, fit_projection
    from vtscore.state.core import thread_dataset_context

    sig = (ctx.dataset_id, bin_shape, tuple(sorted_ids))
    job_cached = projection_jobs.cached_for(sig)
    if job_cached is not None and job_cached.result is not None:
        proj, pyr = job_cached.result
        ctx._projection = proj
        ctx._pyramids[bin_shape] = pyr
        return {"status": "ready", "projection_id": pyr.projection_id}

    mat_copy = matrix.copy()
    ids_copy = list(sorted_ids)
    dataset_id = ctx.dataset_id

    def _run(job):
        with thread_dataset_context(ctx):

            def _on_progress(status, message, current, total):
                job.update_progress(current, total, message)

            proj = fit_projection(mat_copy, ids_copy, on_progress=_on_progress)
            job.update_progress(0, 1, "building pyramid")
            pyr = build_pyramid(proj, bin_shape=bin_shape)
            job.update_progress(1, 1, "done")

            ctx._projection = proj
            ctx._pyramids[bin_shape] = pyr
            job.result = (proj, pyr)

            _persist_projection(dataset_id, proj, pyr)

    job = projection_jobs.start(sig, _run, dataset_id=ctx.dataset_id)
    return {"status": "building", "job_id": job.job_id}


def _dispatch_subset_build(ctx, ids_raw, bin_shape: str) -> dict:
    """Validate the request body's ``ids`` and route to the subset build."""
    if not isinstance(ids_raw, list):
        abort(400, message="`ids` must be a list of media ids.")
    try:
        requested = [int(c) for c in ids_raw]
    except (TypeError, ValueError):
        abort(400, message="`ids` must be a list of integer media ids.")
    if not requested:
        abort(409, message="No items selected — nothing to project.")
    if not ctx.medias:
        abort(409, message="Dataset is empty — nothing to project.")
    try:
        return _build_subset(ctx, requested, bin_shape)
    except ValueError as exc:
        abort(409, message=str(exc))


def _build_subset(ctx, requested_ids: list[int], bin_shape: str) -> dict:
    """Build (or reuse) an ephemeral UMAP projection over just *requested_ids*.

    Unlike the full-dataset path, the subset projection is computed from only
    the high-dimensional vectors of the requested ids (e.g. the positives of a
    Find run), held in dedicated ``_subset_*`` slots on the context, and never
    persisted.  The shared 2-D layout is reused across bin shapes, so a shape
    toggle re-bins in milliseconds instead of re-fitting UMAP.
    """
    from vtscore.embedding.matrix import get_embedding_submatrix
    from vtscore.projection import build_pyramid

    sorted_ids, matrix = get_embedding_submatrix(ctx, requested_ids)
    if matrix.size == 0:
        abort(409, message="None of the selected items have embeddings — nothing to project.")

    if ctx._subset_ids == sorted_ids:
        # Same subset: serve the cached pyramid, or re-bin the shared layout.
        pyr = ctx._subset_pyramids.get(bin_shape)
        if pyr is not None:
            return {"status": "ready", "projection_id": pyr.projection_id}
        proj = ctx._subset_projection
        if proj is not None:
            pyr = build_pyramid(proj, bin_shape=bin_shape)
            ctx._subset_pyramids[bin_shape] = pyr
            return {"status": "ready", "projection_id": pyr.projection_id}
    else:
        # A different subset was requested: drop the stale layout before fitting.
        ctx._subset_projection = None
        ctx._subset_pyramids = {}
        ctx._subset_ids = sorted_ids
        ctx._subset_job_id = None

    return _start_subset_umap_build(ctx, sorted_ids, matrix, bin_shape)


def _start_subset_umap_build(ctx, sorted_ids: list[int], matrix, bin_shape: str) -> dict:
    """Start (or reuse) the background subset UMAP fit + pyramid build."""
    from vtscore.concurrency.async_jobs import projection_jobs
    from vtscore.projection import build_pyramid, fit_projection
    from vtscore.state.core import thread_dataset_context

    sig = (ctx.dataset_id, "subset", bin_shape, tuple(sorted_ids))

    job_cached = projection_jobs.cached_for(sig)
    if job_cached is not None and job_cached.result is not None:
        proj, pyr = job_cached.result
        ctx._subset_projection = proj
        ctx._subset_pyramids[bin_shape] = pyr
        return {"status": "ready", "projection_id": pyr.projection_id}

    # Already building this exact subset + shape?  Reuse the in-flight job
    # instead of queueing a duplicate fit.
    if ctx._subset_job_id:
        existing = projection_jobs.get(ctx._subset_job_id)
        if existing is not None and existing.signature == sig and existing.status in ("running", "pending"):
            return {"status": "building", "job_id": existing.job_id}

    mat_copy = matrix.copy()
    ids_copy = list(sorted_ids)

    def _run(job):
        with thread_dataset_context(ctx):

            def _on_progress(status, message, current, total):
                job.update_progress(current, total, message)

            proj = fit_projection(mat_copy, ids_copy, on_progress=_on_progress)
            job.update_progress(0, 1, "building pyramid")
            pyr = build_pyramid(proj, bin_shape=bin_shape)
            job.update_progress(1, 1, "done")

            ctx._subset_projection = proj
            ctx._subset_pyramids[bin_shape] = pyr
            job.result = (proj, pyr)
            # Subset projections are ephemeral — never persisted.

    job = projection_jobs.start(sig, _run, dataset_id=ctx.dataset_id)
    ctx._subset_job_id = job.job_id
    return {"status": "building", "job_id": job.job_id}


@projection_bp.route("/api/projection/build", methods=["POST"])
@projection_bp.response(200, ProjectionBuildResponseSchema)
@projection_bp.alt_response(409, description="Dataset is empty or has no embeddings.")
def build_projection():
    """Kick off (or short-circuit) a build of the requested bin shape.

    Body: ``{"shape": "hex" | "square"}`` (defaults to hex).

    Returns immediately.  If the pyramid for this shape is already cached or
    persisted, returns ``"ready"``.  If the shared UMAP layout already exists
    (e.g. the other shape was built first), the new shape is re-binned inline
    and returned ready — no UMAP re-fit.  Only the first build of a dataset,
    where no layout exists yet, runs UMAP in the background and returns a
    ``job_id`` for polling via ``GET /api/projection/meta``.
    """
    from vtscore.embedding.matrix import get_embedding_matrix
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    body = request.get_json(silent=True) or {}
    shape = _resolve_shape(body.get("shape"))

    # Subset build: project only the supplied media ids (e.g. the positive
    # results of a Find run), fitting UMAP on just their high-d vectors.
    if body.get("ids") is not None:
        return _dispatch_subset_build(ctx, body["ids"], shape)

    cached = ctx._pyramids.get(shape)
    if cached is not None:
        return {"status": "ready", "projection_id": cached.projection_id}

    if not ctx.medias:
        abort(409, message="Dataset is empty — nothing to project.")

    try:
        sorted_ids, matrix = get_embedding_matrix(ctx)
    except ValueError as exc:
        abort(409, message=str(exc))

    if matrix.size == 0:
        abort(409, message="Dataset has no embeddings — nothing to project.")

    ready = _rebin_from_existing_layout(ctx, sorted_ids, shape)
    if ready is not None:
        return ready

    return _start_umap_build(ctx, sorted_ids, matrix, shape)


@projection_bp.route("/api/projection/meta", methods=["GET"])
@projection_bp.response(200, ProjectionMetaSchema)
def projection_meta():
    """Return projection/pyramid metadata and build status for a bin shape.

    Query: ``?shape=hex|square`` (defaults to hex).

    When the requested shape's pyramid is ready, includes bounds, zoom levels,
    cell sizing, point count, the dataset's media type, and the ``bin_shape``
    the canvas should render.
    """
    from vtscore.concurrency.async_jobs import projection_jobs
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    shape = _resolve_shape(request.args.get("shape"))

    if _is_subset(request.args.get("subset")):
        return _subset_meta(ctx, shape)

    pyr = ctx._pyramids.get(shape)
    if pyr is not None:
        meta = pyr.meta()
        meta["status"] = "ready"
        meta["media_type"] = _media_type_for(ctx)
        meta["method"] = ctx._projection.method if ctx._projection else None
        return meta

    job = projection_jobs.current()
    # A subset build shares the single projection runner; don't report its
    # progress as the full-dataset build's status.
    if job is not None and job.job_id == ctx._subset_job_id:
        job = None
    if job is not None and job.dataset_id == ctx.dataset_id:
        if job.status in ("running", "pending"):
            return {
                "status": "building",
                "job_id": job.job_id,
                "current": job.current,
                "total": job.total,
                "message": job.message,
            }
        if job.status == "error":
            return {"status": "error", "error": job.error or "projection build failed"}

    return {"status": "idle"}


@projection_bp.route(
    "/api/projection/tiles/<shape>/<int:level>/<int(signed=True):tx>/<int(signed=True):ty>",
    methods=["GET"],
)
@projection_bp.response(200, TileResponseSchema)
@projection_bp.alt_response(404, description="Tile not found or projection not ready.")
def get_tile(shape: str, level: int, tx: int, ty: int):
    """Return the cells for one tile of the *shape* pyramid at ``(level, tx, ty)``.

    Because the projection is frozen at ingest, tiles are immutable for the
    life of the dataset and can be cached aggressively by the client.
    """
    from vtscore.state.core import get_active_context

    shape = _resolve_shape(shape)
    ctx = get_active_context()
    if _is_subset(request.args.get("subset")):
        pyr = ctx._subset_pyramids.get(shape)
    else:
        pyr = ctx._pyramids.get(shape)
    if pyr is None:
        abort(404, message="Projection not built yet — call POST /api/projection/build first.")

    # Tiles are frozen at ingest, so let the browser serve repeat visits from its
    # HTTP cache without a round-trip — this is what keeps the hex grid from
    # blanking when you pan/zoom back over ground you've already seen. The tile
    # URL omits the dataset id (it rides on the X-Dataset-Id header), so we must
    # Vary on it or the cache could hand one dataset's tiles to another. Scoped
    # to this response only, and registered after the 404 check so a not-yet-built
    # projection is never cached. ``?subset`` is already part of the URL.
    @after_this_request
    def _cache_tile(response):  # type: ignore[unused-ignore]
        response.headers["Cache-Control"] = "private, max-age=3600, immutable"
        response.vary.add("X-Dataset-Id")
        return response

    tile = pyr.get_tile(level, tx, ty)
    if tile is None:
        return {"level": level, "tx": tx, "ty": ty, "cells": []}

    return tile.to_payload()


def _persist_projection(dataset_id: str, proj, pyr) -> None:
    """Best-effort save of the projection (for ``pyr``'s bin shape) into the container."""
    pkl_path = _pkl_path_for(dataset_id)
    if pkl_path is None:
        return
    try:
        from vtscore.datasets.container import append_projection

        append_projection(pkl_path, proj, pyr)
    except Exception:
        logger.warning("Failed to persist projection for %s", dataset_id, exc_info=True)


__all__ = ["projection_bp"]
