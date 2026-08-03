"""VTSBrowse projection routes: build, meta, and tile endpoints.

Kick off a background UMAP + tile pyramid build for the active dataset,
query its status/metadata, and stream tiles to the browse canvas.

The pyramid tiles the projection as hexagons or squares, but the shape is a
fixed property of the dataset's **media type** — squares for browsable-thumbnail
media (image/video/document), hexes for the rest (audio/text) — not a per-request
choice.  Every endpoint derives the shape from the active dataset
(:func:`_shape_for`); the client never sends it.  The meta response reports the
resolved ``bin_shape`` so the canvas knows which lattice to draw.
"""

from __future__ import annotations

import logging
import uuid

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

#: Bin shapes a dataset can be tiled with.  Mirrors
#: ``vtscore.projection.BIN_SHAPES``; kept as a local literal so probing the
#: container for persisted coordinates does not import the (numba-pulling)
#: projection package on the request path.
_VALID_SHAPES = ("hex", "square")

#: The coarse phases every projection build walks through, in order:
#: 1. arranging the items (the UMAP fit), 2. tiling the layout into the pyramid,
#: 3. naming the regions (signposts).  Reported as ``step``/``total_steps`` on
#: the build meta so the client draws **one** whole-job bar that fills across the
#: build instead of three bars that each restart at zero.  The UMAP fit itself is
#: a single opaque call with no fraction to report (see
#: :mod:`vtscore.projection.umap_projection`), so phase position is the only
#: honest "how far along" signal for the longest part of the build.
_BUILD_STEPS = 3


def _build_progress(job) -> dict:
    """The ``status: building`` meta payload for an in-flight build *job*.

    Carries the within-phase counts **and** the whole-job structure: which of
    the :data:`_BUILD_STEPS` phases is running, plus an ``overall`` completion
    fraction stitched from the two so the client's bar advances monotonically
    across the whole build.  A phase with no countable total (the UMAP fit)
    contributes its slice only when it ends — the bar parks and pulses inside it
    rather than pretending to a fraction the fit cannot report.
    """
    total_steps = job.total_steps or 0
    step = job.step or 0
    within = job.current / job.total if job.total > 0 else 0.0
    overall = None
    if total_steps > 0 and step > 0:
        overall = min(1.0, max(0.0, (step - 1 + within) / total_steps))
    return {
        "status": "building",
        "job_id": job.job_id,
        "current": job.current,
        "total": job.total,
        "message": job.message,
        "step": step or None,
        "total_steps": total_steps or None,
        "overall": overall,
    }


def _shape_for(ctx) -> str:
    """The bin shape this dataset is tiled with, fixed by its media type.

    Squares for browsable-thumbnail media (image/video/document), hexes for the
    rest (audio/text).  Not a user choice — see
    :func:`vtscore.projection.bin_shape_for_media_type`.
    """
    from vtscore.projection import bin_shape_for_media_type

    return bin_shape_for_media_type(_media_type_for(ctx))


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
        meta["content_version"] = getattr(ctx, "_subset_content_version", 0)
        label_set = _label_set_for(ctx, pyr, subset=True)
        meta["has_labels"] = bool(label_set and label_set.labels)
        return meta

    if ctx._subset_job_id:
        job = projection_jobs.get(ctx._subset_job_id)
        if job is not None:
            if job.status in ("running", "pending"):
                return _build_progress(job)
            if job.status == "error":
                return {"status": "error", "error": job.error or "projection build failed"}

    return {"status": "idle"}


def _label_set_for(ctx, pyr, *, subset: bool):
    """The context's :class:`RegionLabelSet` for ``pyr``'s layout, or ``None``.

    Labels are pinned to the frozen layout they were computed from: a set whose
    ``projection_id`` doesn't match the active pyramid's is stale (the layout
    was re-fit underneath it) and is treated as absent rather than served over
    the wrong coordinates.

    When no set is cached for this layout, resolution order is: the set the
    labeling pipeline **persisted** into the dataset container (full layouts
    only — see :func:`_maybe_load_persisted_labels`), then lazily-derived
    **ground-truth signposts** from the dataset's category annotations (see
    :func:`_maybe_build_demo_signposts`).  The result — including an empty
    set, so a dataset with no signs isn't re-probed on every poll — is cached
    on the context.  A set left behind by the live labeling pipeline wins: it
    already matches the layout, so this never overwrites it.
    """
    label_set = ctx._subset_region_labels if subset else ctx._region_labels
    if label_set is not None and label_set.projection_id == pyr.projection_id:
        return label_set

    built = None if subset else _maybe_load_persisted_labels(ctx, pyr)
    if built is None:
        proj = ctx._subset_projection if subset else ctx._projection
        built = _maybe_build_demo_signposts(proj, pyr, ctx.medias)
    if subset:
        ctx._subset_region_labels = built
    else:
        # A background relabel (issue #2404) or build job may have installed a
        # matching set on ``_region_labels`` while we were resolving the
        # fallback; don't clobber it with a derived/empty stand-in — the
        # freshly-built signs already match this layout and are strictly better.
        current = ctx._region_labels
        if current is not None and current.projection_id == pyr.projection_id:
            return current
        ctx._region_labels = built
    if built is None or built.projection_id != pyr.projection_id:
        return None
    return built


def _maybe_load_persisted_labels(ctx, pyr):
    """Restore the signpost labels persisted next to the full-dataset layout.

    Served only over the exact layout they were computed from
    (``projection_id`` match), and only while their labeler signature still
    matches what the active pipeline would produce — a changed provider,
    embedder, or toponymy version makes them stale.  On a stale set we kick a
    background rebuild (:func:`_kick_relabel_if_idle`) so the signs self-heal
    to the active pipeline without a forced Re-project (issue #2404), and treat
    the stale set as absent meanwhile — the map goes briefly unlettered rather
    than serving signs the active labeler would no longer produce.

    When the active pipeline can't label this dataset at all (e.g. toponymy
    isn't installed here), a persisted set is still served: it's derived text
    pinned to the right layout, strictly better than nothing.
    """
    pkl_path = _pkl_path_for(ctx.dataset_id)
    if pkl_path is None:
        return None
    from vtscore.datasets.container import read_region_labels

    loaded = read_region_labels(pkl_path)
    if loaded is None:
        return None
    label_set, stored_signature = loaded
    if label_set.projection_id != pyr.projection_id:
        return None
    from vtscore.projection.signpost_prep import labeler_signature

    active = labeler_signature(ctx)
    if active is not None and active != stored_signature:
        _kick_relabel_if_idle(ctx, pyr)
        return None
    return label_set


def _kick_relabel_if_idle(ctx, pyr) -> None:
    """Rebuild a stale persisted label set in the background (issue #2404).

    A persisted set whose ``labeler_signature`` no longer matches the active
    pipeline is served as absent, but nothing re-runs the labeler while the
    layout stays cached/persisted — only a forced Re-project re-letters the
    map.  This kicks a best-effort background job that re-fits the signs over
    the frozen full-dataset layout and re-persists them under the current
    signature, so the stale signs self-heal on an ordinary Browse.

    Coalescing: repeated meta/labels polls all land here until the rebuild
    lands, so a job already in flight for this context short-circuits — one
    rebuild per stale layout, not one per poll.  Best-effort throughout: a
    missing projection or a labeling failure just leaves the map unlettered.
    """
    proj = ctx._projection
    if proj is None or proj.projection_id != pyr.projection_id:
        return

    from vtscore.concurrency.async_jobs import signpost_relabel_jobs

    job_id = ctx._relabel_job_id
    if job_id:
        existing = signpost_relabel_jobs.get(job_id)
        if existing is not None and existing.status in ("running", "pending"):
            return

    def _run(job):
        # Runs on the relabel runner's worker thread, which the JobManager has
        # already bound to this dataset's context.  ``prep_signposts`` (inside
        # the best-effort wrapper) re-fits the signs, caches them on
        # ``ctx._region_labels``, and re-persists them under the active
        # signature — overwriting the empty interim the serve path cached.
        _prep_signposts_best_effort(ctx, proj, job, subset=False)

    job = signpost_relabel_jobs.start(
        (ctx.dataset_id, "relabel", proj.projection_id),
        _run,
        dataset_id=ctx.dataset_id,
    )
    ctx._relabel_job_id = job.job_id


def _maybe_build_demo_signposts(proj, pyr, medias):
    """Derive ground-truth signposts for ``pyr``'s layout, or ``None``.

    Cheap-probes the medias for a hierarchical (``/``-separated) ``category``
    first, so non-demo datasets never pull the projection package on the
    request path or pay for a build.  Returns an id-pinned
    :class:`RegionLabelSet` (possibly empty) when the layout is usable, else
    ``None``.
    """
    if proj is None or proj.projection_id != pyr.projection_id:
        return None
    from vtscore.projection.demo_signposts import has_hierarchical_categories

    if not has_hierarchical_categories(medias):
        # Cache an empty, id-pinned set so we don't re-probe every poll.
        from vtscore.projection.labels import make_label_set

        return make_label_set(pyr.projection_id, [])

    from vtscore.projection.demo_signposts import build_category_signposts

    return build_category_signposts(proj, medias)


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


def _primary_embedder_for(ctx) -> str | None:
    """The embedder that produced this dataset's projected matrix (its primary)."""
    if ctx is None or not getattr(ctx, "medias", None):
        return None
    try:
        from vtscore.embedding.media_vectors import primary_embedder_name

        return primary_embedder_name(ctx.medias)
    except Exception:  # pragma: no cover - defensive; fall back to globals
        return None


def _umap_params(ctx=None) -> tuple[int, float]:
    """The active UMAP knobs (``n_neighbors``, ``min_dist``).

    Resolution: an explicit ``ServerSettings`` override wins; otherwise the
    per-embedder default (:data:`~vtscore.config.PROJECTION_DEFAULTS_BY_EMBEDDER`)
    keyed off the dataset's primary embedder; otherwise the global config default.
    An "explicit override" is a setting that differs from the global default, so an
    operator who deliberately picks the global value simply gets it.
    """
    from vtsearch import settings
    from vtscore.config import (
        PROJECTION_DEFAULTS_BY_EMBEDDER,
        PROJECTION_MIN_DIST,
        PROJECTION_N_NEIGHBORS,
    )

    s_n = settings.get_projection_n_neighbors()
    s_d = settings.get_projection_min_dist()
    emb = _primary_embedder_for(ctx)
    default = (PROJECTION_N_NEIGHBORS, PROJECTION_MIN_DIST)
    d_n, d_d = PROJECTION_DEFAULTS_BY_EMBEDDER.get(emb, default) if emb else default
    n = s_n if s_n != PROJECTION_N_NEIGHBORS else d_n
    d = s_d if s_d != PROJECTION_MIN_DIST else d_d
    return n, d


def _projection_params_match(proj, ctx=None) -> bool:
    """Whether a persisted projection was fit under the active UMAP params.

    Non-UMAP layouts (the PCA / trivial fallbacks for tiny datasets) ignore
    these knobs, so they always match.  A legacy projection with no stamped
    params is assumed to have used the config defaults, so it only mismatches
    once an operator has changed a setting away from the default — exactly
    when its layout must be recomputed.
    """
    import math

    if getattr(proj, "method", None) != "umap":
        return True
    from vtscore.config import PROJECTION_MIN_DIST, PROJECTION_N_NEIGHBORS

    stored_n = proj.n_neighbors if proj.n_neighbors is not None else PROJECTION_N_NEIGHBORS
    stored_d = proj.min_dist if proj.min_dist is not None else PROJECTION_MIN_DIST
    want_n, want_d = _umap_params(ctx)
    return stored_n == want_n and math.isclose(stored_d, want_d, abs_tol=1e-9)


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
    if not _projection_params_match(proj, ctx):
        logger.info("Persisted %s projection UMAP params changed; will recompute.", bin_shape)
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
        if not _projection_params_match(proj, ctx):
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


def _reset_full_projection(ctx) -> None:
    """Discard the cached + persisted full-dataset layout for a forced rebuild.

    A re-projection must not be short-circuited by the frozen layout, so clear
    the in-memory projection and every shape's cached pyramid, and drop the
    persisted entries from the container — otherwise a later load, or the
    not-yet-rebuilt other bin shape, would resurrect the stale coordinates
    (which are shared across shapes).
    """
    ctx._projection = None
    ctx._pyramids = {}
    ctx._region_labels = None
    pkl_path = _pkl_path_for(ctx.dataset_id)
    if pkl_path is None:
        return
    try:
        from vtscore.datasets.container import remove_projections

        remove_projections(pkl_path)
    except Exception:
        logger.warning("Failed to clear persisted projection for %s", ctx.dataset_id, exc_info=True)


def _prep_signposts_best_effort(ctx, proj, job, *, subset: bool) -> None:
    """Run the signpost labeling pipeline inside a build job, best-effort.

    Called between the pyramid build and the moment the layout is cached on
    the context: the canvas fetches labels once per ``projection_id`` when the
    meta first reports ready, so the signs must exist by then.  Any failure
    (or an environment without toponymy) leaves the map unlettered, never
    broken — signs are optional decoration.
    """
    try:
        from vtscore.projection.signpost_prep import prep_signposts

        def _on_progress(current: int, total: int, message: str) -> None:
            job.update_progress(current, total, message)

        prep_signposts(ctx, proj, subset=subset, on_progress=_on_progress)
    except Exception:
        logger.warning("Signpost labeling failed for %s", ctx.dataset_id, exc_info=True)


def _start_umap_build(ctx, sorted_ids: list[int], matrix, bin_shape: str, *, force: bool = False) -> dict:
    """Start (or reuse) the background UMAP fit + pyramid build for *bin_shape*.

    With *force*, run a brand-new fit even when the ids are unchanged: the job
    signature is namespaced with a nonce so the result cache can't hand back the
    old frozen layout, yielding a fresh (differently-seeded) arrangement.
    """
    from vtscore.concurrency.async_jobs import projection_jobs
    from vtscore.projection import build_pyramid, fit_projection
    from vtscore.state.core import thread_dataset_context

    sig = (ctx.dataset_id, bin_shape, tuple(sorted_ids))
    if force:
        sig = (*sig, uuid.uuid4().hex)
    else:
        job_cached = projection_jobs.cached_for(sig)
        if job_cached is not None and job_cached.result is not None:
            proj, pyr = job_cached.result
            ctx._projection = proj
            ctx._pyramids[bin_shape] = pyr
            return {"status": "ready", "projection_id": pyr.projection_id}

        # Already building this exact dataset + shape?  Reuse the in-flight job
        # instead of queueing a duplicate fit behind it.
        if ctx._full_job_id:
            existing = projection_jobs.get(ctx._full_job_id)
            if existing is not None and existing.signature == sig and existing.status in ("running", "pending"):
                return {"status": "building", "job_id": existing.job_id}

    mat_copy = matrix.copy()
    ids_copy = list(sorted_ids)
    dataset_id = ctx.dataset_id
    from vtscore.config import PROJECTION_COMPACT_DEFAULT

    compact = PROJECTION_COMPACT_DEFAULT
    n_neighbors, min_dist = _umap_params(ctx)

    def _run(job):
        with thread_dataset_context(ctx):

            def _on_progress(status, message, current, total):
                job.update_progress(current, total, message)

            job.set_phase(1, _BUILD_STEPS, "arranging items")
            proj = fit_projection(
                mat_copy,
                ids_copy,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                compact=compact,
                on_progress=_on_progress,
            )
            job.set_phase(2, _BUILD_STEPS, "building pyramid")
            pyr = build_pyramid(proj, bin_shape=bin_shape)
            job.set_phase(3, _BUILD_STEPS, "naming regions")
            _prep_signposts_best_effort(ctx, proj, job, subset=False)
            job.update_progress(1, 1, "done")

            ctx._projection = proj
            ctx._pyramids[bin_shape] = pyr
            job.result = (proj, pyr)

            _persist_projection(dataset_id, proj, pyr)

    job = projection_jobs.start(sig, _run, dataset_id=ctx.dataset_id)
    ctx._full_job_id = job.job_id
    return {"status": "building", "job_id": job.job_id}


def _dispatch_subset_build(ctx, ids_raw, bin_shape: str, *, force: bool = False) -> dict:
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
        return _build_subset(ctx, requested, bin_shape, force=force)
    except ValueError as exc:
        abort(409, message=str(exc))


def _build_subset(ctx, requested_ids: list[int], bin_shape: str, *, force: bool = False) -> dict:
    """Build (or reuse) an ephemeral UMAP projection over just *requested_ids*.

    Unlike the full-dataset path, the subset projection is computed from only
    the high-dimensional vectors of the requested ids (e.g. the positives of a
    Find run), held in dedicated ``_subset_*`` slots on the context, and never
    persisted.  The shared 2-D layout is reused across bin shapes, so a shape
    toggle re-bins in milliseconds instead of re-fitting UMAP.
    """
    from vtscore.embedding.matrix import get_embedding_submatrix
    from vtscore.projection import build_pyramid

    sorted_ids, matrix = get_embedding_submatrix(ctx, requested_ids, ctx.routed_embedder("score"))
    if matrix.size == 0:
        abort(409, message="None of the selected items have embeddings — nothing to project.")

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
        ctx._subset_projection = None
        ctx._subset_pyramids = {}
        ctx._subset_ids = sorted_ids
        ctx._subset_job_id = None
        ctx._subset_content_version = 0  # fresh layout — reset the tile cache token
        ctx._subset_region_labels = None  # signposts were anchored in the dropped layout

    return _start_subset_umap_build(ctx, sorted_ids, matrix, bin_shape, force=force)


def _start_subset_umap_build(ctx, sorted_ids: list[int], matrix, bin_shape: str, *, force: bool = False) -> dict:
    """Start (or reuse) the background subset UMAP fit + pyramid build.

    With *force*, run a brand-new fit even for an unchanged id set (see
    :func:`_start_umap_build`): the signature carries a nonce so neither the
    result cache nor an in-flight job is reused.
    """
    from vtscore.concurrency.async_jobs import projection_jobs
    from vtscore.projection import build_pyramid, fit_projection
    from vtscore.state.core import thread_dataset_context

    sig = (ctx.dataset_id, "subset", bin_shape, tuple(sorted_ids))
    if force:
        sig = (*sig, uuid.uuid4().hex)
    else:
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
    from vtscore.config import PROJECTION_COMPACT_DEFAULT

    compact = PROJECTION_COMPACT_DEFAULT
    n_neighbors, min_dist = _umap_params(ctx)

    def _run(job):
        with thread_dataset_context(ctx):

            def _on_progress(status, message, current, total):
                job.update_progress(current, total, message)

            job.set_phase(1, _BUILD_STEPS, "arranging items")
            proj = fit_projection(
                mat_copy,
                ids_copy,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                compact=compact,
                on_progress=_on_progress,
            )
            job.set_phase(2, _BUILD_STEPS, "building pyramid")
            pyr = build_pyramid(proj, bin_shape=bin_shape)
            # Fresh signs over just this subset: the contrastive keyphrases
            # recompute against the subset's own siblings (better than any
            # filtered dataset-level signs), and the per-media texts were
            # cached at ingest, so this is the cheap half of the pipeline.
            job.set_phase(3, _BUILD_STEPS, "naming regions")
            _prep_signposts_best_effort(ctx, proj, job, subset=True)
            job.update_progress(1, 1, "done")

            ctx._subset_projection = proj
            ctx._subset_pyramids[bin_shape] = pyr
            job.result = (proj, pyr)
            # Subset projections are ephemeral — never persisted (labels included).

    job = projection_jobs.start(sig, _run, dataset_id=ctx.dataset_id)
    ctx._subset_job_id = job.job_id
    return {"status": "building", "job_id": job.job_id}


@projection_bp.route("/api/projection/build", methods=["POST"])
@projection_bp.response(200, ProjectionBuildResponseSchema)
@projection_bp.alt_response(409, description="Dataset is empty or has no embeddings.")
def build_projection():
    """Kick off (or short-circuit) a build of the dataset's projection.

    Body: ``{"ids": [...]?, "force": bool?}``.  The bin shape is not a request
    parameter — it is derived from the dataset's media type (see
    :func:`_shape_for`).

    Returns immediately.  If the pyramid is already cached or persisted, returns
    ``"ready"``.  Only the first build of a dataset, where no layout exists yet,
    runs UMAP in the background and returns a ``job_id`` for polling via
    ``GET /api/projection/meta``.

    ``force`` overrides every short-circuit: it discards the existing layout
    (cached + persisted for a full build, or the in-memory subset layout for an
    ``ids`` build) and runs a fresh UMAP fit over the current items, returning a
    new ``projection_id``.  This is the Browser's "Re-project" action.
    """
    from vtscore.embedding.matrix import get_embedding_matrix
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    body = request.get_json(silent=True) or {}
    shape = _shape_for(ctx)
    # ``force`` re-fits UMAP over the currently displayed items even when a
    # layout already exists, replacing it with a fresh arrangement.  Powers the
    # Browser's "Re-project" button (e.g. to re-spread the survivors after a
    # cull, or just to reshuffle).
    force = bool(body.get("force"))

    # Subset build: project only the supplied media ids (e.g. the positive
    # results of a Find run), fitting UMAP on just their high-d vectors.
    if body.get("ids") is not None:
        return _dispatch_subset_build(ctx, body["ids"], shape, force=force)

    if not force:
        cached = ctx._pyramids.get(shape)
        if cached is not None:
            return {"status": "ready", "projection_id": cached.projection_id}

    if not ctx.medias:
        abort(409, message="Dataset is empty — nothing to project.")

    try:
        # The diversity tree / projection clusters in the score embedder's
        # space (patch-else-text; the v3 routing table).
        sorted_ids, matrix = get_embedding_matrix(ctx, ctx.routed_embedder("score"))
    except ValueError as exc:
        abort(409, message=str(exc))

    if matrix.size == 0:
        abort(409, message="Dataset has no embeddings — nothing to project.")

    if force:
        _reset_full_projection(ctx)
    else:
        ready = _rebin_from_existing_layout(ctx, sorted_ids, shape)
        if ready is not None:
            return ready

    return _start_umap_build(ctx, sorted_ids, matrix, shape, force=force)


@projection_bp.route("/api/projection/subset/remove", methods=["POST"])
@projection_bp.arguments(SubsetRemoveRequestSchema)
@projection_bp.response(200, ProjectionMetaSchema)
@projection_bp.alt_response(409, description="No subset projection is currently built.")
def remove_from_subset(body: dict):
    """Drop ids from the current subset browse **without re-fitting UMAP**.

    Re-bins the frozen subset layout onto its existing grid minus the removed
    points: the remaining points keep their exact 2-D positions and bins, only
    counts/representatives change.  The ``projection_id`` (layout identity) is
    preserved and a bumped ``content_version`` busts the otherwise-immutable
    tile cache.  The served ``bounds`` shrink to the survivors' extent so the
    client re-frames to what's left (zoom-to-fit, minimap) instead of keeping
    dead space where the culled points were — safe because bin assignment is
    origin-independent; bounds only drive client framing.  Returns the updated
    subset meta for the dataset's shape.

    Powers the Browser's "Remove from Good" cull, which marks the items Bad via
    ``/api/medias/vote-bulk`` and then calls this to make them disappear.
    """
    from dataclasses import replace

    from vtscore.projection import rebin_like
    from vtscore.projection import remove_ids as _remove_ids
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    shape = _shape_for(ctx)
    ids = body["ids"]

    proj = ctx._subset_projection
    if proj is None:
        abort(409, message="No subset projection to update — build it first.")

    new_proj = _remove_ids(proj, ids)
    ctx._subset_projection = new_proj
    ctx._subset_ids = list(new_proj.ids)
    ctx._subset_job_id = None
    ctx._subset_content_version = getattr(ctx, "_subset_content_version", 0) + 1
    # Re-bin every shape that was already built, each on its own preserved grid,
    # then stamp the survivors' extent over rebin_like's kept template bounds.
    ctx._subset_pyramids = {
        s: replace(rebin_like(new_proj, pyr), bounds=new_proj.bounds) for s, pyr in ctx._subset_pyramids.items()
    }

    return _subset_meta(ctx, shape)


@projection_bp.route("/api/projection/meta", methods=["GET"])
@projection_bp.response(200, ProjectionMetaSchema)
def projection_meta():
    """Return projection/pyramid metadata and build status for the dataset.

    The bin shape is derived from the dataset's media type (see
    :func:`_shape_for`), not a query parameter.  When the pyramid is ready,
    includes bounds, zoom levels, cell sizing, point count, the dataset's media
    type, and the ``bin_shape`` the canvas should render.
    """
    from vtscore.concurrency.async_jobs import projection_jobs
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    shape = _shape_for(ctx)

    if _is_subset(request.args.get("subset")):
        return _subset_meta(ctx, shape)

    pyr = ctx._pyramids.get(shape)
    if pyr is not None:
        meta = pyr.meta()
        meta["status"] = "ready"
        meta["media_type"] = _media_type_for(ctx)
        meta["method"] = ctx._projection.method if ctx._projection else None
        meta["content_version"] = 0  # full-dataset layouts are never edited in place
        label_set = _label_set_for(ctx, pyr, subset=False)
        meta["has_labels"] = bool(label_set and label_set.labels)
        return meta

    if ctx._full_job_id:
        job = projection_jobs.get(ctx._full_job_id)
        if job is not None:
            if job.status in ("running", "pending"):
                return _build_progress(job)
            if job.status == "error":
                return {"status": "error", "error": job.error or "projection build failed"}

    return {"status": "idle"}


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
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    shape = _shape_for(ctx)
    subset = _is_subset(request.args.get("subset"))
    pyr = (ctx._subset_pyramids if subset else ctx._pyramids).get(shape)
    if pyr is None:
        return {"status": "idle", "labels": []}

    label_set = _label_set_for(ctx, pyr, subset=subset)
    return {
        "status": "ready",
        "projection_id": pyr.projection_id,
        "labels": label_set.payload() if label_set is not None else [],
    }


@projection_bp.route(
    "/api/projection/tiles/<int:level>/<int(signed=True):tx>/<int(signed=True):ty>",
    methods=["GET"],
)
@projection_bp.response(200, TileResponseSchema)
@projection_bp.alt_response(404, description="Tile not found or projection not ready.")
def get_tile(level: int, tx: int, ty: int):
    """Return the cells for one tile of the dataset's pyramid at ``(level, tx, ty)``.

    The bin shape is derived from the dataset's media type (see
    :func:`_shape_for`).  Because the projection is frozen at ingest, tiles are
    immutable for the life of the dataset and can be cached aggressively by the
    client.
    """
    from vtscore.state.core import get_active_context

    ctx = get_active_context()
    shape = _shape_for(ctx)
    subset = _is_subset(request.args.get("subset"))
    if subset:
        pyr = ctx._subset_pyramids.get(shape)
        proj = ctx._subset_projection
    else:
        pyr = ctx._pyramids.get(shape)
        proj = ctx._projection
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

    payload = tile.to_payload()
    # Attach each cell its full member id list so the canvas can render
    # per-cell selection state and toggle a whole bin. The pyramid keeps only
    # counts + representatives, so this is re-derived on demand from the frozen
    # layout (see ``tile_member_ids``); it rides along on the immutable,
    # HTTP-cached tile response.
    if proj is not None and payload["cells"]:
        from vtscore.projection import tile_member_ids

        members = tile_member_ids(pyr, proj, level, tx, ty)
        for cell in payload["cells"]:
            cell["member_ids"] = members.get((cell["q"], cell["r"]), [])

    return payload


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
