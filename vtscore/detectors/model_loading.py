"""Resolve a detector's scoring model, training it on demand if needed.

:func:`resolve_or_train_detector` is the cold-path counterpart to the
``DetectorContext.model`` fast path: given a detector id and (optionally) its
on-disk data plus a media snapshot, it returns the MLP + threshold to score
with, training from the detector's labelset when no live model exists.

This logic lived inline in ``vtsearch/routes/detectors/scoring.py`` until it
grew its own resolution/embedding/training branches; it has no Flask or
request-context dependency, so it belongs in the library tier where it can be
exercised directly.

The training itself is **not** re-implemented here.  It used to be - a
hand-rolled read of each label's image-level vector, an md5-only in-dataset
lookup, and one negative per Bad label - which quietly made the same labelset
mean a different detector depending on which entry point trained it
(issue #3544, the sibling of #3525 one path over).  Everything below the
resolution/progress plumbing now delegates to
:func:`~vtscore.detectors.labelset_training.train_from_labelset`, the same
entry point the detector-load and learned-sort paths use.
"""

from __future__ import annotations

from typing import Any

from vtscore.concurrency.progress import update_find_progress


def resolve_or_train_detector(
    detector_id: str,
    det_data: dict | None,
    media_type: str,
    snap: dict | None,
    *,
    progress_step: int = 2,
    progress_total_steps: int = 4,
) -> tuple[Any | None, float, dict | None]:
    """Return (mlp, threshold, diagnostic) for *detector_id*.

    Tries the loaded :class:`~vtscore.state.core.DetectorContext` first.  Falls
    back to training on demand from the detector's labelset via
    :func:`~vtscore.detectors.labelset_training.train_from_labelset`, which
    resolves each element (in-dataset by origin ▸ md5 ▸ name, else through its
    origin importer), pools a Good element's ``region_box`` down to the raw
    patch under it, floods a Bad element's patch rows as negatives, and
    calibrates per bag.  Returns ``(None, _, diag)`` when training is not
    possible; *diag* is
    :func:`~vtscore.detectors.labelset_training.labelset_resolution_report`.

    **The head this returns is a MaxPatch head on a patch dataset**, so its
    callers score it through the ordinary max-pooled
    :func:`~vtscore.detectors.training.scoring_rows_for_snap` geometry - the
    geometry they were already using against the whole-image head this replaced,
    which was the train/score mismatch #3544 was filed for.  On any dataset whose
    embedder produces no patch grid every bag holds one row and the whole path
    collapses to the historical single-vector behaviour.

    Inclusion is a pure cutoff knob now (find-verification-workflow.md): a slide
    does **not** retrain or drop the MLP, it re-derives the threshold over the
    cached fold orderings.  ``train_from_labelset`` passes the detector context
    down to :func:`~vtscore.detectors.training.train_and_threshold`, which caches
    those orderings on it — without that cache a later Inclusion slide can't move
    the cutoff (it would silently no-op).
    """
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.dataset_sync import invalidate_detector_model_on_embedder_mismatch
    from vtscore.detectors.labelset_training import labelset_resolution_report, train_from_labelset
    from vtscore.embedding.binding import keying_embedder_for_snap
    from vtscore.state.core import DetectorContext, get_detector_context

    det_ctx = get_detector_context(detector_id)
    if det_ctx is not None:
        # Defense against H5: scoring Auto-Find detectors iterates contexts
        # that aren't the active one, so the before_request hook can't
        # have invalidated their stale MLPs.  Drop them here so the next
        # branch trains fresh against the detector's primary.  The keying
        # marker returns the detector's primary when the active dataset can
        # supply it (so a valid cached model survives), else the dataset score
        # precedence (so a genuine mismatch invalidates).  See patch-embedder.md
        # → "Per-detector primary embedder".
        snap_embedder = keying_embedder_for_snap(det_ctx, snap)
        invalidate_detector_model_on_embedder_mismatch(det_ctx, snap_embedder)
    if det_ctx is not None and det_ctx.model is not None:
        return det_ctx.model, det_ctx.threshold, None

    if det_data is None:
        return None, 0.5, None

    labelset = LabelSet.from_dict(det_data.get("labelset") or {})
    if not labelset.elements:
        return None, 0.5, None

    update_find_progress(
        "running",
        "Training detector from labels…",
        current=0,
        total=0,
        step=progress_step,
        total_steps=progress_total_steps,
    )

    # A never-loaded detector (the Auto-Find and portable-export cases) has no
    # context to train against, so it gets a throwaway one.  ``detector_id`` is
    # deliberately left empty on it: ``populate_label_embeddings`` ends by
    # calling ``record_detector_embedder`` to persist the space it embedded in,
    # and a scoring pass over a detector nobody loaded should not be what writes
    # that.  ``record_detector_embedder`` no-ops on an empty id.  A *loaded*
    # detector is handed its live context, exactly as the load path does: the
    # snapshot here is the active dataset, so the caches this populates are the
    # ones that context is supposed to hold, and the trained head lands on
    # ``det_ctx.model`` where the fast path above will find it next time.
    train_ctx = det_ctx
    if train_ctx is None:
        train_ctx = DetectorContext(
            "",
            name=det_data.get("name", "") or "",
            media_type=media_type,
            embedder_type=det_data.get("embedder_type", "") or "",
        )

    def _on_label(_name: str, current: int, total: int) -> None:
        # Resolving a label that isn't in *snap* costs an importer fetch (plus a
        # ``patch_forward`` on a patch detector), so it is the phase of a cold
        # train that can run long.  The final element hands over to the fold
        # fitting, which is the other one.
        done = current >= total
        update_find_progress(
            "running",
            "Cross-calibrating threshold…" if done else f"Resolving {total} label origins…",
            current=current,
            total=total,
            step=progress_step,
            total_steps=progress_total_steps,
        )

    if train_from_labelset(train_ctx, labelset, media_type=media_type, snap=snap, on_progress=_on_label):
        return train_ctx.model, train_ctx.threshold, None

    return None, 0.5, labelset_resolution_report(train_ctx, labelset, media_type=media_type, snap=snap)
