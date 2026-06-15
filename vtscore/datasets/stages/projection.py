"""Projection stage: compute + persist the 2-D Browse projection at ingest.

Opt-in stage that fits UMAP on the cached embedding matrix, builds the
hex-tile pyramid, caches both on the context, and persists them into the
dataset container so the Browse canvas opens instantly instead of paying for
the fit lazily on first visit. Best-effort by contract: the dataset is
already registered and usable before this runs.
"""

from __future__ import annotations

import traceback

from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS


def _persist_projection_to_container(dataset_id: str, proj, pyr) -> None:
    """Best-effort save of the freshly-built projection into the dataset container.

    Mirrors ``vtsearch.routes.projection._persist_projection`` but resolves
    the container path through the dataset registry from inside the load
    pipeline.  Failures are swallowed (logged): the in-memory projection is
    already cached on the context, and a missing on-disk copy just means the
    next Browse open recomputes it.
    """
    from vtscore.datasets.registry import get_dataset  # noqa: PLC0415

    entry = get_dataset(dataset_id)
    if entry is None:
        return
    pkl_path = entry.get("pkl_path")
    if not pkl_path:
        return
    try:
        from vtscore.datasets.container import append_projection  # noqa: PLC0415

        append_projection(pkl_path, proj, pyr)
    except Exception:
        traceback.print_exc()


def _build_projection_stage(ctx, tracker, dataset_id: str) -> None:
    """Compute + persist the 2-D UMAP projection at ingest (opt-in).

    Runs inline as a load stage (after the dataset is registered) so the
    Browse canvas opens instantly instead of paying for the UMAP fit lazily
    on first visit.  This mirrors the on-demand
    ``POST /api/projection/build`` path: fit UMAP on the cached embedding
    matrix, build the hex-tile pyramid, cache both on the context, and
    persist them into the dataset container.

    Best-effort by contract: the dataset is already registered and usable
    before this runs, so any failure here (including a cancellation during
    the fit) must leave the dataset intact and merely fall back to the lazy
    Browse-time build.  The caller wraps this in a try/except for that
    reason.
    """
    from vtscore.embedding.matrix import get_embedding_matrix  # noqa: PLC0415
    from vtscore.projection import build_pyramid, fit_projection  # noqa: PLC0415

    def _progress(current: int, total: int, message: str) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            message,
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    try:
        sorted_ids, matrix = get_embedding_matrix(ctx)
    except ValueError:
        return
    if matrix.size == 0:
        return

    _progress(0, 0, "Building 2-D projection…")

    def _on_fit_progress(status: str, message: str, current: int, total: int) -> None:
        _progress(current, total, message or "Building 2-D projection…")

    proj = fit_projection(matrix, list(sorted_ids), on_progress=_on_fit_progress)
    _progress(0, 1, "Building tile pyramid…")
    # Build the default (hex) binning at ingest; the square binning is derived
    # lazily on first toggle in the browse view, then cached/persisted there.
    pyr = build_pyramid(proj)
    _progress(1, 1, "Projection ready")

    ctx._projection = proj
    ctx._pyramids[pyr.bin_shape] = pyr

    _persist_projection_to_container(dataset_id, proj, pyr)
