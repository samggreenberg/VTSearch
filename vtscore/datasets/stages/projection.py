"""Projection stage: compute + persist the 2-D Browse projection at ingest.

Opt-in stage that fits UMAP on the cached embedding matrix, builds the
hex-tile pyramid, caches both on the context, and persists them into the
dataset container so the Browse canvas opens instantly instead of paying for
the fit lazily on first visit. Best-effort by contract: the dataset is
already registered and usable before this runs.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS

if TYPE_CHECKING:
    from vtscore.state import DatasetContext


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


def _signpost_texts_stage(ctx: DatasetContext, tracker) -> None:
    """Compute + cache the per-media signpost texts at ingest (opt-in).

    Runs **before** the registry save so the texts land in the dataset
    pickle: Toponymy's contrastive keyphrase mining needs a text for *every*
    media (not just sampled exemplars), and the text is the only full-corpus
    model cost in the sign pipeline — computed once here, every later browse
    and Find→Browse subset re-fit reuses it.  No-op when signposting isn't
    possible for this dataset; best-effort by the same contract as the
    projection stage (the caller wraps it).
    """
    from vtscore.projection.signpost_prep import ensure_texts_for_dataset  # noqa: PLC0415

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

    _progress(0, 0, "Preparing signpost texts…")
    ensure_texts_for_dataset(ctx, _progress)


def _maybe_signpost_texts_stage(ctx: DatasetContext, fin, enabled: bool) -> None:
    """Run the signpost-texts stage when the projection opt-in is set.

    Owns the opt-in gate and the best-effort wrapper so the load pipeline's
    task body stays branch-free; a failure here (short of a cancellation
    raised through the tracker) costs only the cached texts, never the load.
    """
    if not enabled:
        return
    fin.begin("signpost_texts")
    try:
        _signpost_texts_stage(ctx, fin)
    except Exception:
        traceback.print_exc()


def _build_projection_stage(ctx: DatasetContext, tracker, dataset_id: str) -> None:
    """Compute + persist the 2-D UMAP projection and its signposts at ingest (opt-in).

    Runs inline as a load stage (after the dataset is registered) so the
    Browse canvas opens instantly instead of paying for the UMAP fit lazily
    on first visit.  This mirrors the on-demand
    ``POST /api/projection/build`` path: fit UMAP on the cached embedding
    matrix, build the hex-tile pyramid, run the signpost labeling over the
    frozen layout (see ``vtscore.projection.signpost_prep``), cache
    everything on the context, and persist it into the dataset container.

    "Mirrors" includes the knobs: the fit goes through the same
    :func:`~vtscore.projection.params.resolve_projection_params` the route
    uses, never ``fit_projection``'s signature defaults.  A layout fit under
    different params than the route would resolve is worse than no pre-build
    at all — the first Browse open either throws it away (params mismatch, so
    UMAP re-runs and the opt-in bought nothing) or keeps a layout nobody
    configured.

    Best-effort by contract: the dataset is already registered and usable
    before this runs, so any failure here (including a cancellation during
    the fit) must leave the dataset intact and merely fall back to the lazy
    Browse-time build.  The caller wraps this in a try/except for that
    reason.
    """
    from vtscore.embedding.matrix import get_embedding_matrix  # noqa: PLC0415
    from vtscore.projection import (  # noqa: PLC0415
        bin_shape_for_media_type,
        build_pyramid,
        fit_projection,
        resolve_projection_params,
    )

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
        # Cluster in the score embedder's space (patch-else-text; v3 routing).
        sorted_ids, matrix = get_embedding_matrix(ctx, ctx.routed_embedder("score"))
    except ValueError:
        return
    if matrix.size == 0:
        return

    _progress(0, 0, "Building 2-D projection…")

    def _on_fit_progress(status: str, message: str, current: int, total: int) -> None:
        _progress(current, total, message or "Building 2-D projection…")

    params = resolve_projection_params(ctx)
    proj = fit_projection(
        matrix,
        list(sorted_ids),
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        compact=params.compact,
        on_progress=_on_fit_progress,
    )
    _progress(0, 1, "Building tile pyramid…")
    # The bin shape is a fixed property of the dataset's media type — squares
    # for browsable-thumbnail media (image/video/document), hexes otherwise —
    # so build exactly the one shape this dataset will ever use.
    media_type = next(iter(ctx.medias.values())).get("media_type") if ctx.medias else None
    pyr = build_pyramid(proj, bin_shape=bin_shape_for_media_type(media_type))

    # Letter the frozen layout before it's cached/persisted, so the first
    # Browse open finds the signs together with the map (the canvas fetches
    # labels once per projection_id).  Reuses the texts the pre-registry
    # stage cached into the pickle; itself best-effort — a labeling failure
    # must not cost the user the projection they just paid for.
    try:
        from vtscore.projection.signpost_prep import prep_signposts  # noqa: PLC0415

        prep_signposts(ctx, proj, subset=False, on_progress=_progress)
    except Exception:
        traceback.print_exc()
    _progress(1, 1, "Projection ready")

    ctx._projection = proj
    ctx._pyramids[pyr.bin_shape] = pyr

    _persist_projection_to_container(dataset_id, proj, pyr)
