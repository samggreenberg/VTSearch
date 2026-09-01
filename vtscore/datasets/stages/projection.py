"""Projection stage: compute + persist the 2-D Browse projection at ingest.

Opt-in stage that fits UMAP on the cached embedding matrix, builds the
tile pyramid, letters it with signposts, caches everything on the context,
and persists it into the dataset container so the Browse canvas opens
instantly instead of paying for the fit lazily on first visit. Best-effort by
contract: the dataset is already registered and usable before this runs.

The fit itself is not implemented here — it is
:func:`vtscore.projection.service.fit_and_install_layout`, the same call the
lazy Browse-time build makes. That sharing is load-bearing rather than tidy:
a layout pre-built under different knobs (or a different bin shape) than the
serve path resolves is *worse* than no pre-build at all, because the first
Browse open either throws it away and re-fits or serves an arrangement nobody
configured. What is left in this module is the load-pipeline wiring: progress
reporting through the loading tracker, and the best-effort contract.
"""

from __future__ import annotations

import traceback
from typing import TYPE_CHECKING

from vtscore.datasets.stages._common import _TOTAL_LOAD_STEPS

if TYPE_CHECKING:
    from vtscore.state import DatasetContext

#: What the loading bar says during each of the build's three phases (see
#: :data:`vtscore.projection.service.BUILD_STEPS`), in the load pipeline's own
#: voice rather than the job runner's.
_PHASE_MESSAGES = {
    1: "Building 2-D projection…",
    2: "Building tile pyramid…",
    3: "Naming regions…",
}


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


def _build_projection_stage(ctx: DatasetContext, tracker) -> None:
    """Compute + persist the 2-D UMAP projection and its signposts at ingest.

    Runs inline as a load stage (after the dataset is registered) so the
    Browse canvas opens instantly instead of paying for the UMAP fit lazily
    on first visit.  Delegates the whole build to
    :func:`~vtscore.projection.service.fit_and_install_layout`, which is what
    ``POST /api/projection/build`` runs in the background — same knobs, same
    bin shape, same signpost pass, same persistence — so the layout this
    leaves behind is exactly the one the serve path would have produced.

    Best-effort by contract: the dataset is already registered and usable
    before this runs, so any failure here (including a cancellation during
    the fit) must leave the dataset intact and merely fall back to the lazy
    Browse-time build.  The caller wraps this in a try/except for that
    reason.
    """
    from vtscore.embedding.matrix import get_embedding_matrix  # noqa: PLC0415
    from vtscore.projection import service  # noqa: PLC0415

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

    def _on_phase(step: int, message: str) -> None:
        # Also the cancellation checkpoint between phases: ``_progress``
        # raises through the tracker, so a cancel lands before the next one.
        _progress(0, 0 if step == 1 else 1, _PHASE_MESSAGES.get(step, message))

    try:
        # Cluster in the score embedder's space (patch-else-text; v3 routing).
        sorted_ids, matrix = get_embedding_matrix(ctx, ctx.routed_embedder("score"))
    except ValueError:
        return
    if matrix.size == 0:
        return

    service.fit_and_install_layout(
        ctx,
        list(sorted_ids),
        matrix,
        service.shape_for(ctx),
        on_phase=_on_phase,
        on_progress=_progress,
    )
    _progress(1, 1, "Projection ready")
