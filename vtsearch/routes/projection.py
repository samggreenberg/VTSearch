"""VTSBrowse projection routes: build, meta, and tile endpoints.

Kick off a background UMAP + hex-tile pyramid build for the active dataset,
query its status/metadata, and stream tiles to the browse canvas.
"""

from __future__ import annotations

import logging

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
    description="VTSBrowse projection: UMAP layout + hex-tile pyramid.",
)


def _media_type_for(ctx) -> str:
    """Return the media type of the first item in the dataset, or ``""``."""
    if not ctx.medias:
        return ""
    first = next(iter(ctx.medias.values()))
    return first.get("media_type", "audio")


@projection_bp.route("/api/projection/build", methods=["POST"])
@projection_bp.response(200, ProjectionBuildResponseSchema)
@projection_bp.alt_response(409, description="Dataset is empty or has no embeddings.")
def build_projection():
    """Kick off a background UMAP + pyramid build for the active dataset.

    Returns immediately with a ``job_id`` for polling via
    ``GET /api/projection/meta``.  If a projection is already cached on
    the context, returns it without rebuilding.
    """
    from vtscore.concurrency.async_jobs import projection_jobs
    from vtscore.embedding.matrix import get_embedding_matrix
    from vtscore.projection import build_pyramid, fit_projection, max_useful_levels
    from vtscore.state.core import get_active_context, thread_dataset_context

    ctx = get_active_context()

    if ctx._pyramid is not None:
        return {
            "status": "ready",
            "projection_id": ctx._pyramid.projection_id,
        }

    if not ctx.medias:
        abort(409, message="Dataset is empty — nothing to project.")

    try:
        sorted_ids, matrix = get_embedding_matrix(ctx)
    except ValueError as exc:
        abort(409, message=str(exc))

    if matrix.size == 0:
        abort(409, message="Dataset has no embeddings — nothing to project.")

    sig = (ctx.dataset_id, tuple(sorted_ids))
    cached = projection_jobs.cached_for(sig)
    if cached is not None and cached.result is not None:
        proj, pyr = cached.result
        ctx._projection = proj
        ctx._pyramid = pyr
        return {"status": "ready", "projection_id": pyr.projection_id}

    mat_copy = matrix.copy()
    ids_copy = list(sorted_ids)

    def _run(job):
        with thread_dataset_context(ctx):
            n_levels = max_useful_levels(len(ids_copy))

            def _on_progress(status, message, current, total):
                job.update_progress(current, total, message)

            proj = fit_projection(
                mat_copy,
                ids_copy,
                on_progress=_on_progress,
            )
            job.update_progress(0, 1, "building pyramid")
            pyr = build_pyramid(proj, n_levels=n_levels)
            job.update_progress(1, 1, "done")

            ctx._projection = proj
            ctx._pyramid = pyr
            job.result = (proj, pyr)

    job = projection_jobs.start(
        sig,
        _run,
        dataset_id=ctx.dataset_id,
    )
    return {"status": "building", "job_id": job.job_id}


@projection_bp.route("/api/projection/meta", methods=["GET"])
@projection_bp.response(200, ProjectionMetaSchema)
def projection_meta():
    """Return projection/pyramid metadata and build status.

    When the projection is ready, includes bounds, zoom levels, hex sizing,
    point count, and the dataset's media type (so the client knows which
    hover-preview behavior to use).
    """
    from vtscore.concurrency.async_jobs import projection_jobs
    from vtscore.state.core import get_active_context

    ctx = get_active_context()

    if ctx._pyramid is not None:
        meta = ctx._pyramid.meta()
        meta["status"] = "ready"
        meta["media_type"] = _media_type_for(ctx)
        meta["method"] = ctx._projection.method if ctx._projection else None
        return meta

    job = projection_jobs.current()
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


@projection_bp.route("/api/projection/tiles/<int:level>/<int:tx>/<int:ty>", methods=["GET"])
@projection_bp.response(200, TileResponseSchema)
@projection_bp.alt_response(404, description="Tile not found or projection not ready.")
def get_tile(level: int, tx: int, ty: int):
    """Return the hex cells for one tile at ``(level, tx, ty)``.

    Because the projection is frozen at ingest, tiles are immutable for the
    life of the dataset and can be cached aggressively by the client.
    """
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    pyr = ctx._pyramid
    if pyr is None:
        abort(404, message="Projection not built yet — call POST /api/projection/build first.")

    tile = pyr.get_tile(level, tx, ty)
    if tile is None:
        return {"level": level, "tx": tx, "ty": ty, "cells": []}

    return tile.to_payload()


__all__ = ["projection_bp"]
