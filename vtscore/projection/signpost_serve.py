"""Resolving the region signposts to serve over a frozen Browse layout.

The read side of the "street sign" name layer (see
``docs/plans/vtsbrowse-toponymy.md``); :mod:`vtscore.projection.signpost_prep`
is the write side.  Given a layout the canvas is about to draw, this module
answers *which* label set belongs on it, in resolution order:

1. a set already cached on the context for exactly this layout,
2. the set the labeling pipeline **persisted** beside the dataset (full
   layouts only), provided its labeler signature still matches what the
   active pipeline would produce,
3. lazily-derived **ground-truth signposts** from the dataset's category
   annotations (the demo datasets), and
4. nothing -- an unlettered map.

Every step is best-effort.  Signs are decoration: a missing labeler, an
uninstallable dependency, or a failed fit leaves the map unlettered, never
broken.  Labels are pinned to the ``projection_id`` they were fit against and
are treated as absent over any other layout, so a stale set is inert rather
than wrong -- anchors are meaningless off their own layout.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vtscore.projection import store

if TYPE_CHECKING:
    from vtscore.projection.labels import RegionLabelSet
    from vtscore.projection.pyramid import Pyramid
    from vtscore.projection.umap_projection import Projection
    from vtscore.state import DatasetContext

logger = logging.getLogger(__name__)

#: ``(current, total, message)`` progress sink, matching
#: :data:`vtscore.projection.signpost_prep.ProgressFn`.
ProgressFn = Callable[[int, int, str], None]


def label_set_for(ctx: DatasetContext, pyr: Pyramid, *, subset: bool) -> RegionLabelSet | None:
    """The context's :class:`RegionLabelSet` for ``pyr``'s layout, or ``None``.

    Labels are pinned to the frozen layout they were computed from: a set
    whose ``projection_id`` doesn't match the active pyramid's is stale (the
    layout was re-fit underneath it) and is treated as absent rather than
    served over the wrong coordinates.

    When no set is cached for this layout, resolution falls through the order
    in the module docstring.  The result -- including an empty set, so a
    dataset with no signs isn't re-probed on every poll -- is cached on the
    context.  A set left behind by the live labeling pipeline wins: it
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


def prep_signposts_best_effort(
    ctx: DatasetContext,
    proj: Projection,
    *,
    subset: bool,
    on_progress: ProgressFn | None = None,
) -> None:
    """Run the signpost labeling pipeline over *proj*, best-effort.

    Called between the pyramid build and the moment the layout is cached on
    the context: the canvas fetches labels once per ``projection_id`` when the
    meta first reports ready, so the signs must exist by then.  Any failure
    (or an environment without toponymy) leaves the map unlettered, never
    broken — signs are optional decoration.

    *on_progress* takes ``(current, total, message)``; a background job passes
    its own ``update_progress`` straight in.
    """
    try:
        from vtscore.projection.signpost_prep import prep_signposts  # noqa: PLC0415

        prep_signposts(ctx, proj, subset=subset, on_progress=on_progress)
    except Exception:
        logger.warning("Signpost labeling failed for %s", ctx.dataset_id, exc_info=True)


def _maybe_load_persisted_labels(ctx: DatasetContext, pyr: Pyramid) -> RegionLabelSet | None:
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
    pkl_path = store.pkl_path_for(ctx.dataset_id)
    if pkl_path is None:
        return None
    from vtscore.datasets.container import read_region_labels  # noqa: PLC0415

    loaded = read_region_labels(pkl_path)
    if loaded is None:
        return None
    label_set, stored_signature = loaded
    if label_set.projection_id != pyr.projection_id:
        return None
    from vtscore.projection.signpost_prep import labeler_signature  # noqa: PLC0415

    active = labeler_signature(ctx)
    if active is not None and active != stored_signature:
        _kick_relabel_if_idle(ctx, pyr)
        return None
    return label_set


def _kick_relabel_if_idle(ctx: DatasetContext, pyr: Pyramid) -> None:
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

    from vtscore.concurrency.async_jobs import signpost_relabel_jobs  # noqa: PLC0415

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
        prep_signposts_best_effort(ctx, proj, subset=False, on_progress=job.update_progress)

    job = signpost_relabel_jobs.start(
        (ctx.dataset_id, "relabel", proj.projection_id),
        _run,
        dataset_id=ctx.dataset_id,
    )
    ctx._relabel_job_id = job.job_id


def _maybe_build_demo_signposts(proj: Any, pyr: Pyramid, medias) -> RegionLabelSet | None:
    """Derive ground-truth signposts for ``pyr``'s layout, or ``None``.

    Cheap-probes the medias for a hierarchical (``/``-separated) ``category``
    first, so non-demo datasets never pay for a build.  Returns an id-pinned
    :class:`RegionLabelSet` (possibly empty) when the layout is usable, else
    ``None``.
    """
    if proj is None or proj.projection_id != pyr.projection_id:
        return None
    from vtscore.projection.demo_signposts import has_hierarchical_categories  # noqa: PLC0415

    if not has_hierarchical_categories(medias):
        # Cache an empty, id-pinned set so we don't re-probe every poll.
        from vtscore.projection.labels import make_label_set  # noqa: PLC0415

        return make_label_set(pyr.projection_id, [])

    from vtscore.projection.demo_signposts import build_category_signposts  # noqa: PLC0415

    return build_category_signposts(proj, medias)


__all__ = ["label_set_for", "prep_signposts_best_effort"]
