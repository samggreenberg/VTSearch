"""Re-embed a loaded detector's labels when the active dataset's space changes.

:func:`maybe_start_label_reembed` is the cold-path fired when an
already-loaded detector is re-selected while the active dataset uses a
different embedder than the one its cached label embeddings were built
against (e.g. switching from a SigLIP-embedded image dataset to a
CLIP-embedded one).  It schedules a progress-tracked background re-embed so
the work is visible to the user instead of running lazily inside the next
vote / learned-sort request.

This logic lived inline in ``vtsearch/routes/detectors/registry.py`` until it
grew its own mismatch-detection + task-orchestration branches.  Everything it
touches except the thread-spawn helper already lives in the library tier, so
it belongs here where it can be exercised without a Flask client.  The one app-tier dependency —
``vtsearch.threading.spawn``, which replays the request's user thread-local
into the worker — is injected by the caller so this module stays Flask-free.
"""

from __future__ import annotations

from typing import Any, Callable

# A ``spawn``-style callable: ``spawn(target, *, name=...) -> Thread``.  The
# route injects ``vtsearch.threading.spawn`` (which carries the user /
# dataset / detector thread-locals into the worker); tests can inject a
# synchronous stand-in.
SpawnFn = Callable[..., Any]


def active_dataset_embedder_name(det_ctx=None) -> str:
    """Return the active dataset's model-keying marker for *det_ctx*, or ``""``.

    Under the per-detector primary model the marker is the detector's own
    primary whenever the active dataset can supply it (so the cache survives a
    dataset switch and no re-embed fires), else the dataset score precedence (so
    a genuine mismatch schedules a re-embed against the new dataset).  This
    agrees with the model-invalidation check in ``dataset_sync``.  See
    :func:`keying_embedder_for_snap` and patch-embedder.md → "Per-detector
    primary embedder".
    """
    from vtscore.embedding.binding import keying_embedder_for_snap
    from vtscore.state import snapshot_medias

    return keying_embedder_for_snap(det_ctx, snapshot_medias())


def embedder_display_name(embedder_name: str) -> str:
    """Return a human-friendly name for *embedder_name*, falling back to the id."""
    if not embedder_name:
        return ""
    from vtscore.media import get_embedder

    try:
        return get_embedder(embedder_name).display_name or embedder_name
    except KeyError:
        return embedder_name


def maybe_start_label_reembed(det_ctx, entry: dict, *, spawn: SpawnFn) -> str | None:
    """Fire a re-embed task when the active dataset uses a different embedder.

    Returns the task id when work was started, or ``None`` when the cache is
    already aligned (same embedder, empty dataset, no labelset cached, etc.)
    so the caller can return the synchronous fast-path.

    *spawn* starts the background worker (see :data:`SpawnFn`); it must replay
    the caller's execution context into the new thread.
    """
    new_embedder = active_dataset_embedder_name(det_ctx)
    if not new_embedder or not det_ctx.embedder or new_embedder == det_ctx.embedder:
        return None

    labelset = det_ctx.cached_labelset
    if labelset is None or not labelset.elements:
        # Nothing to re-embed; just update the stamp so we don't re-enter
        # this branch on every subsequent switch.
        det_ctx.embedder = new_embedder
        return None

    from vtscore.concurrency.progress import CancelledError, detector_loading_tasks
    from vtscore.state import get_active_context
    from vtscore.state.core import thread_dataset_context, thread_detector_context

    _thread_ds_ctx = get_active_context()
    media_type = det_ctx.cached_labelset_media_type or entry.get("media_type", "") or ""
    display = embedder_display_name(new_embedder)
    base_msg = f"Re-resolving labels for {display}…" if display else "Re-resolving labels…"

    task_id = f"_detreembed_{det_ctx.detector_id[:8]}"
    tracker = detector_loading_tasks.create_task(
        task_id,
        entry.get("name", det_ctx.detector_id),
        detector_id=det_ctx.detector_id,
        media_type=media_type,
        embedder=new_embedder,
    )
    tracker.update("loading", base_msg, 0, 0, step=1, total_steps=1)

    def reembed_task():
        from vtscore.detectors.labelset_training import train_from_labelset

        try:
            with thread_dataset_context(_thread_ds_ctx), thread_detector_context(det_ctx):

                def _embed_progress(name: str, done: int, total: int) -> None:
                    tracker.check_cancelled()
                    tracker.update("loading", base_msg, done, total, step=1, total_steps=1)

                # ``populate_label_embeddings`` clears the cache when it detects
                # the embedder change, so this rebuilds against the new embedder
                # from scratch. The stamp on ``det_ctx.embedder`` is updated
                # inside ``populate_label_embeddings``.
                from vtscore.state import snapshot_medias

                try:
                    train_from_labelset(
                        det_ctx,
                        labelset,
                        media_type=media_type,
                        snap=snapshot_medias(),
                        on_progress=_embed_progress,
                    )
                    tracker.update("idle", "", 0, 0, step=None, total_steps=None)
                except CancelledError:
                    tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
                except Exception as e:
                    import traceback as _tb

                    _tb.print_exc()
                    error_msg = str(e) or repr(e) or "Unknown error during label re-embedding"
                    tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
        finally:
            detector_loading_tasks.mark_finished(task_id)

    spawn(reembed_task, name=f"det-reembed-{det_ctx.detector_id[:8]}")
    return task_id
