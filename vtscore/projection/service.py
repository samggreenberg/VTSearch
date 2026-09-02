"""The VTSBrowse layout lifecycle: build it, restore it, report on it, serve it.

The state machine behind ``/api/projection/*``, with no HTTP in it.  Every
function takes an explicit :class:`~vtscore.state.DatasetContext` and returns
plain data, so the whole lifecycle — first fit, restore-from-container,
re-bin to the other shape, forced re-projection, ephemeral subset fits, and
in-place subset culls — is exercisable without a Flask client.

**Where a layout comes from.**  A dataset's 2-D arrangement is fit once and
then frozen.  A build request is answered by the cheapest source that can
honestly serve it:

1. the pyramid already cached on the context for this bin shape,
2. the layout persisted in the dataset container (:mod:`.store`), if its ids
   and fit parameters still match,
3. a re-bin of the frozen coordinates that some *other* shape persisted — the
   coordinates are shared across shapes, so a hex/square toggle costs a
   re-bin, never a re-fit, and
4. only then a background UMAP fit.

``force`` (the Browser's "Re-project") skips all four and starts over.

**Full vs subset.**  A dataset carries two independent layouts: the
full-dataset one, which is persisted, and an ephemeral *subset* fit over just
some ids (the positives of a Find run), which lives in memory and is never
written to disk.  Every entry point here takes the same shape for both, and
the ``subset`` flag picks which set of context slots it reads and writes.

**The bin shape is not a choice.**  It is a fixed property of the dataset's
media type — squares for browsable-thumbnail media (image/video/document),
hexes for the rest — resolved by :func:`shape_for`.  Callers never pass one in
from a request.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from typing import TYPE_CHECKING

from vtscore.projection import store
from vtscore.projection.signpost_serve import label_set_for, prep_signposts_best_effort

if TYPE_CHECKING:
    from vtscore.projection.pyramid import Pyramid
    from vtscore.projection.umap_projection import Projection
    from vtscore.state import DatasetContext

logger = logging.getLogger(__name__)

#: The coarse phases every projection build walks through, in order:
#: 1. arranging the items (the UMAP fit), 2. tiling the layout into the pyramid,
#: 3. naming the regions (signposts).  Reported as ``step``/``total_steps`` on
#: the build meta so the client draws **one** whole-job bar that fills across the
#: build instead of three bars that each restart at zero.  The UMAP fit itself is
#: a single opaque call with no fraction to report (see
#: :mod:`vtscore.projection.umap_projection`), so phase position is the only
#: honest "how far along" signal for the longest part of the build.
BUILD_STEPS = 3

#: ``(step, message)`` sink for the coarse build phases above.
PhaseFn = Callable[[int, str], None]

#: ``(current, total, message)`` sink for within-phase progress.
ProgressFn = Callable[[int, int, str], None]


class NothingToProject(ValueError):
    """There is nothing to lay out: no items, no selection, or no embeddings.

    A ``ValueError`` subclass because the embedding-matrix builders already
    signal the same class of "this dataset can't be projected" with a plain
    ``ValueError``, and every caller wants to treat the two identically (the
    HTTP layer answers both with a 409 carrying the message).
    """


# ----------------------------------------------------------------------
# Context probes
# ----------------------------------------------------------------------


def media_type_for(ctx: DatasetContext) -> str:
    """Return the media type of the first item in the dataset, or ``""``."""
    if not ctx.medias:
        return ""
    first = next(iter(ctx.medias.values()))
    return first.get("media_type", "audio")


def shape_for(ctx: DatasetContext) -> str:
    """The bin shape this dataset is tiled with, fixed by its media type.

    Squares for browsable-thumbnail media (image/video/document), hexes for the
    rest (audio/text).  Not a user choice — see
    :func:`vtscore.projection.bin_shape_for_media_type`.
    """
    from vtscore.projection import bin_shape_for_media_type  # noqa: PLC0415

    return bin_shape_for_media_type(media_type_for(ctx))


# ----------------------------------------------------------------------
# Status reporting
# ----------------------------------------------------------------------


def build_progress(job) -> dict:
    """The ``status: building`` meta payload for an in-flight build *job*.

    Carries the within-phase counts **and** the whole-job structure: which of
    the :data:`BUILD_STEPS` phases is running, plus an ``overall`` completion
    fraction stitched from the two so the client's bar advances monotonically
    across the whole build.  A phase with no countable total (the UMAP fit)
    contributes its slice only when it ends — the bar parks inside it rather
    than pretending to a fraction the fit cannot report.  ``overall_step_end``
    (the fraction at which the running phase's slice ends) is published
    alongside so the client can shade the parked-to-slice-end span as a bounded
    indeterminate zone: "somewhere in here" is all the fit can honestly say.

    Every field is read straight off the job's
    :class:`~vtscore.concurrency.progress.ProgressTracker` snapshot rather than
    recomputed here.  That is what earns the build an ``eta_seconds``: the
    tracker derives one from the rate the job is actually sustaining, rebasing
    at each phase boundary and publishing it on a coarse sticky ladder, none of
    which a payload builder that only sees the current counts could do.
    """
    snap = job.progress.get()
    return {
        "status": "building",
        "job_id": job.job_id,
        "current": snap.get("current") or 0,
        "total": snap.get("total") or 0,
        "message": snap.get("message") or "",
        "step": snap.get("step") or None,
        "total_steps": snap.get("total_steps") or None,
        "overall": snap.get("overall"),
        "overall_step_end": snap.get("overall_step_end"),
        "eta_seconds": snap.get("eta_seconds"),
    }


def layout_meta(ctx: DatasetContext, bin_shape: str, *, subset: bool) -> dict:
    """Projection/pyramid metadata + build status for one of the two layouts.

    ``ready`` carries the pyramid's own meta (bounds, zoom levels, cell
    sizing, point count) plus what the canvas needs to draw over it.  With no
    pyramid, the layout's in-flight job — if any — decides between
    ``building``, ``error``, and ``idle``.

    ``content_version`` busts the otherwise-immutable tile cache.  Full-dataset
    layouts are never edited in place, so theirs is a constant ``0``; the
    subset's advances on every cull.
    """
    from vtscore.concurrency.async_jobs import projection_jobs  # noqa: PLC0415

    pyr = (ctx._subset_pyramids if subset else ctx._pyramids).get(bin_shape)
    if pyr is not None:
        proj = ctx._subset_projection if subset else ctx._projection
        meta = pyr.meta()
        meta["status"] = "ready"
        meta["media_type"] = media_type_for(ctx)
        meta["method"] = proj.method if proj else None
        meta["content_version"] = ctx._subset_content_version if subset else 0
        label_set = label_set_for(ctx, pyr, subset=subset)
        meta["has_labels"] = bool(label_set and label_set.labels)
        return meta

    job_id = ctx._subset_job_id if subset else ctx._full_job_id
    if job_id:
        job = projection_jobs.get(job_id)
        if job is not None:
            if job.status in ("running", "pending"):
                return build_progress(job)
            if job.status == "error":
                return {"status": "error", "error": job.error or "projection build failed"}

    return {"status": "idle"}


def labels_payload(ctx: DatasetContext, bin_shape: str, *, subset: bool) -> dict:
    """The region signpost labels to letter over one of the two layouts.

    Labels are optional decoration: a layout whose labeling pipeline hasn't
    run (or that predates it) answers an empty list, not an error, and
    ``status`` is ``"idle"`` only when no layout is built at all.
    """
    pyr = (ctx._subset_pyramids if subset else ctx._pyramids).get(bin_shape)
    if pyr is None:
        return {"status": "idle", "labels": []}

    label_set = label_set_for(ctx, pyr, subset=subset)
    return {
        "status": "ready",
        "projection_id": pyr.projection_id,
        "labels": label_set.payload() if label_set is not None else [],
    }


def tile_payload(
    ctx: DatasetContext,
    bin_shape: str,
    level: int,
    tx: int,
    ty: int,
    *,
    subset: bool,
) -> dict | None:
    """One tile of a layout's pyramid, or ``None`` when it isn't built yet.

    A tile outside the pyramid is not an error — it answers an empty cell
    list, which is how the canvas learns that corner of the map is empty.
    Only a missing *pyramid* returns ``None`` (the caller's 404).

    Each cell carries its full member id list so the canvas can render
    per-cell selection state and toggle a whole bin.  The pyramid keeps only
    counts + representatives, so the members are re-derived on demand from the
    frozen layout.
    """
    if subset:
        pyr = ctx._subset_pyramids.get(bin_shape)
        proj = ctx._subset_projection
    else:
        pyr = ctx._pyramids.get(bin_shape)
        proj = ctx._projection
    if pyr is None:
        return None

    tile = pyr.get_tile(level, tx, ty)
    if tile is None:
        return {"level": level, "tx": tx, "ty": ty, "cells": []}

    payload = tile.to_payload()
    if proj is not None and payload["cells"]:
        from vtscore.projection import tile_member_ids  # noqa: PLC0415

        members = tile_member_ids(pyr, proj, level, tx, ty)
        for cell in payload["cells"]:
            cell["member_ids"] = members.get((cell["q"], cell["r"]), [])

    return payload


# ----------------------------------------------------------------------
# Fitting a layout
# ----------------------------------------------------------------------


def fit_and_install_layout(
    ctx: DatasetContext,
    sorted_ids: list[int],
    matrix,
    bin_shape: str,
    *,
    subset: bool = False,
    on_phase: PhaseFn | None = None,
    on_progress: ProgressFn | None = None,
) -> tuple[Projection, Pyramid]:
    """Fit UMAP over *matrix*, tile it, letter it, and cache it on *ctx*.

    The one implementation of "make a layout", shared by the lazy Browse
    build, the opt-in ingest pre-build stage, and the subset fit.  Keeping it
    single-sourced is what stops the ingest stage from drifting away from the
    on-demand path: a layout fit under different knobs than the serve path
    resolves is worse than no pre-build at all, because the first Browse open
    throws it away and the pre-build bought nothing.

    Walks the three :data:`BUILD_STEPS` phases, reporting each through
    *on_phase* and within-phase counts through *on_progress*.  Signposting is
    best-effort and runs before the layout is cached, so the canvas finds the
    signs together with the map.  Full-dataset layouts are persisted into the
    dataset container; subset layouts are ephemeral and never written.
    """
    from vtscore.projection import build_pyramid, fit_projection  # noqa: PLC0415
    from vtscore.projection.params import resolve_projection_params  # noqa: PLC0415

    def _phase(step: int, message: str) -> None:
        if on_phase is not None:
            on_phase(step, message)

    def _on_fit_progress(status: str, message: str, current: int, total: int) -> None:
        if on_progress is not None:
            on_progress(current, total, message)

    params = resolve_projection_params(ctx)
    _phase(1, "arranging items")
    proj = fit_projection(
        matrix,
        list(sorted_ids),
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        compact=params.compact,
        on_progress=_on_fit_progress,
    )
    _phase(2, "building pyramid")
    pyr = build_pyramid(proj, bin_shape=bin_shape)
    # Fresh signs over exactly these items: for a subset the contrastive
    # keyphrases recompute against the subset's own siblings (better than any
    # filtered dataset-level signs), and the per-media texts were cached at
    # ingest, so this is the cheap half of the pipeline.
    _phase(3, "naming regions")
    prep_signposts_best_effort(ctx, proj, subset=subset, on_progress=on_progress)

    install_layout(ctx, proj, pyr, subset=subset)
    if not subset:
        store.persist_projection(ctx.dataset_id, proj, pyr)
    return proj, pyr


def install_layout(ctx: DatasetContext, proj: Projection, pyr: Pyramid, *, subset: bool = False) -> None:
    """Cache *proj* + *pyr* on *ctx* as the layout for ``pyr``'s bin shape."""
    if subset:
        ctx._subset_projection = proj
        ctx._subset_pyramids[pyr.bin_shape] = pyr
    else:
        ctx._projection = proj
        ctx._pyramids[pyr.bin_shape] = pyr


def reset_full_projection(ctx: DatasetContext) -> None:
    """Discard the cached + persisted full-dataset layout for a forced rebuild.

    A re-projection must not be short-circuited by the frozen layout, so clear
    the in-memory projection and every shape's cached pyramid, and drop the
    persisted entries from the container — otherwise a later load, or the
    not-yet-rebuilt other bin shape, would resurrect the stale coordinates
    (which are shared across shapes).  The subset layout is deliberately left
    standing: it is an independent fit the user may still be browsing.
    """
    ctx.reset_derived_caches(matrices=False, lookups=False, projection=True, subset=False)
    store.remove_persisted_projections(ctx.dataset_id)


# ----------------------------------------------------------------------
# The build entry point
# ----------------------------------------------------------------------


def build_layout(
    ctx: DatasetContext,
    *,
    ids: list[int] | None = None,
    force: bool = False,
) -> dict:
    """Ensure a layout exists for *ctx*, and report where that left things.

    Answers ``{"status": "ready", "projection_id": ...}`` when a layout was
    already available (or could be re-binned from frozen coordinates without a
    fit), or ``{"status": "building", "job_id": ...}`` when a background UMAP
    fit had to be started.  Returns immediately either way.

    With *ids*, projects only those media — an ephemeral subset layout held
    separately from the full-dataset one.  With *force*, discards whichever
    layout the request targets and re-fits from scratch.

    Raises :class:`NothingToProject` (or a plain ``ValueError`` from the
    embedding-matrix builder) when there is nothing to lay out.
    """
    bin_shape = shape_for(ctx)

    if ids is not None:
        return build_subset_layout(ctx, ids, bin_shape, force=force)

    if not force:
        cached = ctx._pyramids.get(bin_shape)
        if cached is not None:
            return {"status": "ready", "projection_id": cached.projection_id}

    if not ctx.medias:
        raise NothingToProject("Dataset is empty — nothing to project.")

    from vtscore.embedding.matrix import get_embedding_matrix  # noqa: PLC0415

    # The coverage atlas / projection clusters in the score embedder's space
    # (patch-else-text; the v3 routing table).
    sorted_ids, matrix = get_embedding_matrix(ctx, ctx.routed_embedder("score"))
    if matrix.size == 0:
        raise NothingToProject("Dataset has no embeddings — nothing to project.")

    if force:
        reset_full_projection(ctx)
    else:
        ready = rebin_from_existing_layout(ctx, sorted_ids, bin_shape)
        if ready is not None:
            return ready

    return start_full_build(ctx, sorted_ids, matrix, bin_shape, force=force)


def rebin_from_existing_layout(ctx: DatasetContext, sorted_ids: list[int], bin_shape: str) -> dict | None:
    """Serve *bin_shape* without a UMAP fit, if at all possible.

    Tries, in order: this shape's persisted pyramid, then re-binning the shared
    frozen layout (already in memory or persisted under another shape).  Returns
    a ready-response dict, or ``None`` when no layout exists yet and UMAP must
    run.
    """
    from vtscore.projection import build_pyramid  # noqa: PLC0415

    proj = ctx._projection
    if proj is None or set(proj.ids) != set(sorted_ids):
        # Nothing usable in memory: probe the container, preferring this
        # shape's own pyramid so a stored one is served rather than re-binned.
        loaded = store.load_any_persisted_layout(ctx, sorted_ids, prefer=bin_shape)
        if loaded is None:
            return None
        proj, pyr = loaded
        install_layout(ctx, proj, pyr)
        if pyr.bin_shape == bin_shape:
            return {"status": "ready", "projection_id": pyr.projection_id}
    else:
        # The coordinates are in memory but this shape has no pyramid; a
        # stored one for it beats re-binning, since it was built already.
        loaded = store.load_persisted_layout(ctx, sorted_ids, bin_shape)
        if loaded is not None:
            stored_proj, stored_pyr = loaded
            install_layout(ctx, stored_proj, stored_pyr)
            return {"status": "ready", "projection_id": stored_pyr.projection_id}

    pyr = build_pyramid(proj, bin_shape=bin_shape)
    ctx._pyramids[bin_shape] = pyr
    store.persist_projection(ctx.dataset_id, proj, pyr)
    return {"status": "ready", "projection_id": pyr.projection_id}


def start_full_build(
    ctx: DatasetContext,
    sorted_ids: list[int],
    matrix,
    bin_shape: str,
    *,
    force: bool = False,
) -> dict:
    """Start (or reuse) the background UMAP fit + pyramid build for *bin_shape*.

    With *force*, run a brand-new fit even when the ids are unchanged: the job
    signature is namespaced with a nonce so the result cache can't hand back the
    old frozen layout, yielding a fresh (differently-seeded) arrangement.
    """
    from vtscore.concurrency.async_jobs import projection_jobs  # noqa: PLC0415

    sig: tuple = (ctx.dataset_id, bin_shape, tuple(sorted_ids))
    if force:
        sig = (*sig, uuid.uuid4().hex)
    else:
        reused = _reuse_existing_build(ctx, sig, subset=False)
        if reused is not None:
            return reused

    run = _build_runner(ctx, sorted_ids, matrix, bin_shape, subset=False)
    job = projection_jobs.start(sig, run, dataset_id=ctx.dataset_id)
    ctx._full_job_id = job.job_id
    return {"status": "building", "job_id": job.job_id}


def build_subset_layout(
    ctx: DatasetContext,
    requested_ids: list[int],
    bin_shape: str,
    *,
    force: bool = False,
) -> dict:
    """Build (or reuse) an ephemeral UMAP projection over just *requested_ids*.

    Unlike the full-dataset path, the subset projection is computed from only
    the high-dimensional vectors of the requested ids (e.g. the positives of a
    Find run), held in dedicated ``_subset_*`` slots on the context, and never
    persisted.  The shared 2-D layout is reused across bin shapes, so a shape
    toggle re-bins in milliseconds instead of re-fitting UMAP.
    """
    from vtscore.embedding.matrix import get_embedding_submatrix  # noqa: PLC0415
    from vtscore.projection import build_pyramid  # noqa: PLC0415

    if not requested_ids:
        raise NothingToProject("No items selected — nothing to project.")
    if not ctx.medias:
        raise NothingToProject("Dataset is empty — nothing to project.")

    sorted_ids, matrix = get_embedding_submatrix(ctx, requested_ids, ctx.routed_embedder("score"))
    if matrix.size == 0:
        raise NothingToProject("None of the selected items have embeddings — nothing to project.")

    if not force and ctx._subset_ids == sorted_ids:
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
        # A different subset — or a forced re-projection — drops the stale layout
        # before fitting.  ``force`` re-fits even when the ids are unchanged, so
        # the survivors of a cull re-spread instead of keeping their old slots.
        # Drops the layout, its pyramids, the in-flight job id, the tile-cache
        # token, and the signposts anchored in the dropped layout.
        ctx.reset_derived_caches(matrices=False, lookups=False, projection=False, subset=True)
        ctx._subset_ids = sorted_ids

    return start_subset_build(ctx, sorted_ids, matrix, bin_shape, force=force)


def start_subset_build(
    ctx: DatasetContext,
    sorted_ids: list[int],
    matrix,
    bin_shape: str,
    *,
    force: bool = False,
) -> dict:
    """Start (or reuse) the background subset UMAP fit + pyramid build.

    With *force*, run a brand-new fit even for an unchanged id set (see
    :func:`start_full_build`): the signature carries a nonce so neither the
    result cache nor an in-flight job is reused.
    """
    from vtscore.concurrency.async_jobs import projection_jobs  # noqa: PLC0415

    sig: tuple = (ctx.dataset_id, "subset", bin_shape, tuple(sorted_ids))
    if force:
        sig = (*sig, uuid.uuid4().hex)
    else:
        reused = _reuse_existing_build(ctx, sig, subset=True)
        if reused is not None:
            return reused

    run = _build_runner(ctx, sorted_ids, matrix, bin_shape, subset=True)
    job = projection_jobs.start(sig, run, dataset_id=ctx.dataset_id)
    ctx._subset_job_id = job.job_id
    return {"status": "building", "job_id": job.job_id}


def remove_subset_ids(ctx: DatasetContext, ids: list[int]) -> None:
    """Drop *ids* from the current subset layout **without re-fitting UMAP**.

    Re-bins the frozen subset layout onto its existing grid minus the removed
    points: the remaining points keep their exact 2-D positions and bins, only
    counts/representatives change.  The ``projection_id`` (layout identity) is
    preserved and a bumped ``content_version`` busts the otherwise-immutable
    tile cache.  The served ``bounds`` shrink to the survivors' extent so the
    client re-frames to what's left (zoom-to-fit, minimap) instead of keeping
    dead space where the culled points were — safe because bin assignment is
    origin-independent; bounds only drive client framing.

    Raises :class:`NothingToProject` when no subset layout is built.
    """
    from dataclasses import replace  # noqa: PLC0415

    from vtscore.projection import rebin_like  # noqa: PLC0415
    from vtscore.projection import remove_ids as _remove_ids  # noqa: PLC0415

    proj = ctx._subset_projection
    if proj is None:
        raise NothingToProject("No subset projection to update — build it first.")

    new_proj = _remove_ids(proj, ids)
    ctx._subset_projection = new_proj
    ctx._subset_ids = list(new_proj.ids)
    ctx._subset_job_id = None
    ctx._subset_content_version += 1
    # Re-bin every shape that was already built, each on its own preserved grid,
    # then stamp the survivors' extent over rebin_like's kept template bounds.
    ctx._subset_pyramids = {
        s: replace(rebin_like(new_proj, pyr), bounds=new_proj.bounds) for s, pyr in ctx._subset_pyramids.items()
    }


# ----------------------------------------------------------------------
# Background-job plumbing
# ----------------------------------------------------------------------


def _reuse_existing_build(ctx: DatasetContext, sig: tuple, *, subset: bool) -> dict | None:
    """Answer *sig* from a finished or in-flight job, or ``None`` to start one.

    Two short-circuits, in order: the runner's result cache still holds the
    layout this exact signature produced (install it and report ready), or a
    job for this exact dataset + shape + id set is already running (report its
    id instead of queueing a duplicate fit behind it).
    """
    from vtscore.concurrency.async_jobs import projection_jobs  # noqa: PLC0415

    job_cached = projection_jobs.cached_for(sig)
    if job_cached is not None and job_cached.result is not None:
        proj, pyr = job_cached.result
        install_layout(ctx, proj, pyr, subset=subset)
        return {"status": "ready", "projection_id": pyr.projection_id}

    job_id = ctx._subset_job_id if subset else ctx._full_job_id
    if job_id:
        existing = projection_jobs.get(job_id)
        if existing is not None and existing.signature == sig and existing.status in ("running", "pending"):
            return {"status": "building", "job_id": existing.job_id}
    return None


def _build_runner(ctx: DatasetContext, sorted_ids: list[int], matrix, bin_shape: str, *, subset: bool):
    """The worker-thread body for a background fit of *sorted_ids*.

    Snapshots the matrix and id list so the job is independent of any later
    mutation of the dataset, and re-binds the worker thread to *ctx* (the
    runner is a single app-wide slot shared with every other dataset's build).
    """
    mat_copy = matrix.copy()
    ids_copy = list(sorted_ids)

    def _run(job):
        from vtscore.state.core import thread_dataset_context  # noqa: PLC0415

        with thread_dataset_context(ctx):
            proj, pyr = fit_and_install_layout(
                ctx,
                ids_copy,
                mat_copy,
                bin_shape,
                subset=subset,
                on_phase=lambda step, message: job.set_phase(step, BUILD_STEPS, message),
                on_progress=job.update_progress,
            )
            job.update_progress(1, 1, "done")
            job.result = (proj, pyr)

    return _run


__all__ = [
    "BUILD_STEPS",
    "NothingToProject",
    "build_layout",
    "build_progress",
    "build_subset_layout",
    "fit_and_install_layout",
    "install_layout",
    "labels_payload",
    "layout_meta",
    "media_type_for",
    "rebin_from_existing_layout",
    "remove_subset_ids",
    "reset_full_projection",
    "shape_for",
    "start_full_build",
    "start_subset_build",
    "tile_payload",
]
