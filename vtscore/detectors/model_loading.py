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

from typing import TYPE_CHECKING, Any, Callable

from vtscore.concurrency.progress import update_find_progress

if TYPE_CHECKING:
    from vtscore.datasets.labelset import LabelSet
    from vtscore.state.core import DetectorContext


def labelset_signature(labelset: "LabelSet | None") -> tuple | None:
    """The identity of the training labels in *labelset*, or ``None`` for none.

    Sorted ``(label, stable_element_id, region_box)`` triples: the three things
    about an element that change the head trained from it.  Metadata and
    provenance are left out because training never reads them.  The region box
    is in because a Good label's box decides which patch it pools to, so
    redrawing one changes the detector as surely as flipping a label.

    Stable across a JSON round-trip, so the signature of a labelset in memory
    equals the signature of the same labelset read back from the detector file.
    ``None`` in, ``None`` out: callers with no labelset (the raw-vote learned
    sort) get a signature that no labelset matches.
    """
    if labelset is None:
        return None
    from vtscore.detectors.labelset_elements import stable_element_id

    return tuple(
        sorted(
            (
                el.label,
                stable_element_id(el),
                tuple(float(v) for v in el.region_box) if el.region_box is not None else (),
            )
            for el in labelset.elements
        )
    )


def cached_head_is_current(det_ctx: "DetectorContext", labelset: "LabelSet") -> bool:
    """Whether *det_ctx* holds a head trained from exactly *labelset*.

    ``det_ctx.model`` is a cache, and nothing that changes a detector's labels
    drops it: a vote, clearing the votes, a dataset switch and the dashboard's
    saved-label review all leave it in place.  Only a learned sort replaces it.
    So Find would keep scoring with the head from the last learned sort, or the
    last Find, until the app restarted (issue #4204).  Each writer of
    ``det_ctx.model`` records the signature of the labels it trained from, and a
    consumer reuses the head only while *labelset* (the one it just read from
    the detector file) still has that signature.  This also rejects the head a
    background learned sort stores after newer votes have already landed.
    """
    return det_ctx.model is not None and det_ctx.model_labels_sig == labelset_signature(labelset)


def resolve_or_train_detector(
    detector_id: str,
    det_data: dict | None,
    media_type: str,
    snap: dict | None,
    *,
    progress_step: int = 2,
    progress_total_steps: int = 4,
    on_progress: Callable[..., None] | None = None,
    use_loaded_context: bool = True,
    ctx_sink: list | None = None,
) -> tuple[Any | None, float, dict | None]:
    """Return (mlp, threshold, diagnostic) for *detector_id*.

    Tries the loaded :class:`~vtscore.state.core.DetectorContext` first, when
    its head was trained from the labelset in *det_data*
    (:func:`cached_head_is_current`).  Falls back to training on demand from
    the detector's labelset via
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

    *on_progress* receives the training progress (the
    :func:`~vtscore.concurrency.progress.update_find_progress` signature:
    ``status, message, current=, total=, step=, total_steps=``); ``None`` keeps
    the historical sink, the shared Find tracker.  A caller scoring off to the
    side of Find - the app's background AutoRun - passes its own task's sink so
    a cold train does not paint the Find bar, or leave it "running", behind a
    user who never asked for a Find.

    *use_loaded_context* ``False`` leaves a loaded detector's live
    :class:`~vtscore.state.core.DetectorContext` untouched: the head is trained
    on a throwaway context as if the detector were not loaded.  The live
    context's caches belong to the dataset the user is working in, so a scorer
    that runs unprompted against another dataset must neither invalidate them
    nor train into them.

    *ctx_sink*, when given, receives the detector context whose head and
    threshold are returned - the loaded one, or the throwaway a never-loaded
    detector trains on - so a caller can ask what the precision floor says
    about that threshold (:func:`vtscore.state.core.detector_floor_state`,
    #4247).  Nothing is appended when no head is returned.
    """
    report = on_progress if on_progress is not None else update_find_progress
    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.dataset_sync import invalidate_detector_model_on_embedder_mismatch
    from vtscore.detectors.labelset_training import labelset_resolution_report, train_from_labelset
    from vtscore.embedding.binding import keying_embedder_for_snap
    from vtscore.state.core import DetectorContext, get_detector_context

    det_ctx = get_detector_context(detector_id) if use_loaded_context else None
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
    labelset = LabelSet.from_dict((det_data or {}).get("labelset") or {})
    if det_ctx is not None and cached_head_is_current(det_ctx, labelset):
        if ctx_sink is not None:
            ctx_sink.append(det_ctx)
        return det_ctx.model, det_ctx.threshold, None

    if det_data is None:
        return None, 0.5, None

    if not labelset.elements:
        return None, 0.5, None

    report(
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
        report(
            "running",
            "Cross-calibrating threshold…" if done else f"Resolving {total} label origins…",
            current=current,
            total=total,
            step=progress_step,
            total_steps=progress_total_steps,
        )

    if train_from_labelset(train_ctx, labelset, media_type=media_type, snap=snap, on_progress=_on_label):
        if ctx_sink is not None:
            ctx_sink.append(train_ctx)
        return train_ctx.model, train_ctx.threshold, None

    return None, 0.5, labelset_resolution_report(train_ctx, labelset, media_type=media_type, snap=snap)
