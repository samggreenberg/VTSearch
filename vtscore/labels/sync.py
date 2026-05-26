"""Labelset source sync utilities.

Provides :func:`sync_to_labelset_source` which exports the current
detector's labels to its linked labelset source (if any), and
:func:`sync_from_labelset_source` which imports labels from the source.

``sync_to_labelset_source`` is **debounced and asynchronous**: each call
schedules a background push that fires after ``_DEBOUNCE_DELAY`` seconds
of quiet (currently 200ms).  Rapid voting bursts collapse into a single
sync run that uses the latest state, so a slow target (webhook, slow
disk) never stalls the voting request handler.  Use
:func:`flush_pending_label_syncs` to drain the queue synchronously when
deterministic behaviour is needed (tests, graceful shutdown).  An
``atexit`` hook calls ``flush_pending_label_syncs`` so the most recent
vote's push survives normal interpreter exit (Ctrl-C, gunicorn SIGQUIT,
``sys.exit``).  Hard kills (SIGKILL, ``os._exit``) bypass atexit and
still drop the last 200ms of work — accept that as the cost of debounce.

A module-level (NOT thread-local) flag, coordinated by ``_sync_lock``,
prevents re-exporting during an import pass — including from concurrent
``_push_to_labelset_source`` calls running on other threads.
"""

from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vtscore.state.core import DatasetContext, DetectorContext

logger = logging.getLogger(__name__)

# Serializes label sync operations and protects ``_syncing``.  A
# thread-local guard cannot stop a parallel ``sync_to`` on another
# thread from racing with a ``sync_from`` import.
_sync_lock = threading.RLock()
_syncing: bool = False


# Debounce window for ``sync_to_labelset_source``.  200ms coalesces a
# rapid voting burst into a single background push while staying short
# enough that the on-disk labels match the UI within a tick.
_DEBOUNCE_DELAY = 0.2


@dataclass
class _PendingSync:
    """One scheduled debounce + the contexts to run it under."""

    timer: threading.Timer
    user: str | None
    dataset_ctx: "DatasetContext"
    detector_ctx: "DetectorContext"


# Per-detector debounce slot.  Keyed by ``detector_id`` so concurrent
# voting on two detectors doesn't coalesce one detector's push into the
# other's window.
_pending_lock = threading.Lock()
_pending_syncs: dict[str, _PendingSync] = {}

# Held by ``_run_pending_sync`` while a worker is in flight and acquired
# by :func:`flush_pending_label_syncs` so flush() doesn't return while a
# debounced timer is still mid-write.  Separate from ``_sync_lock`` so a
# concurrent ``sync_from_labelset_source`` still serializes correctly.
_workers_lock = threading.Lock()


def sync_to_labelset_source() -> None:
    """Schedule a debounced background push to the active detector's labelset source.

    Returns immediately.  The actual push runs on a background timer
    thread ~200ms after the most recent call for this detector; further
    calls within the window restart the timer and overwrite the captured
    contexts (latest wins), so a burst of votes turns into one sync run.

    Silently skips if no labelset source is configured for the active
    detector, if no detector is active, or if a ``sync_from`` import is
    currently in progress (re-checked at execution time, not here).
    """
    from vtsearch.auth import get_current_user
    from vtscore.state.core import get_active_context, get_active_detector_context

    detector_ctx = get_active_detector_context()
    if detector_ctx is None or not detector_ctx.labelset_source:
        return

    detector_id = detector_ctx.detector_id
    if not detector_id:
        return

    user = get_current_user()
    dataset_ctx = get_active_context()

    with _pending_lock:
        existing = _pending_syncs.get(detector_id)
        if existing is not None:
            existing.timer.cancel()
        timer = threading.Timer(_DEBOUNCE_DELAY, _run_pending_sync, args=(detector_id,))
        timer.daemon = True
        _pending_syncs[detector_id] = _PendingSync(
            timer=timer,
            user=user,
            dataset_ctx=dataset_ctx,
            detector_ctx=detector_ctx,
        )
        timer.start()


def _run_pending_sync(detector_id: str) -> None:
    """Timer callback: pop the pending entry and run the actual push.

    Holds ``_workers_lock`` while running so :func:`flush_pending_label_syncs`
    can wait for an in-flight write to finish before returning.
    """
    with _workers_lock:
        with _pending_lock:
            entry = _pending_syncs.pop(detector_id, None)
        if entry is None:
            return
        _push_with_thread_context(entry)


def _push_with_thread_context(entry: _PendingSync) -> None:
    """Scope thread-local user / dataset / detector context, run the push, restore."""
    from vtsearch.auth import thread_user
    from vtscore.state.core import thread_dataset_context, thread_detector_context

    with (
        thread_user(entry.user),
        thread_dataset_context(entry.dataset_ctx),
        thread_detector_context(entry.detector_ctx),
    ):
        _push_to_labelset_source()


def flush_pending_label_syncs() -> None:
    """Run every pending debounced sync now and wait for in-flight workers.

    Used by tests (so they can assert the file was written) and by any
    graceful-shutdown path.  Cancels every pending timer, then runs the
    sync inline on the calling thread.  Acquires ``_workers_lock`` so any
    timer that has already fired and is mid-write is waited out before
    this returns.
    """
    with _workers_lock:
        with _pending_lock:
            entries = list(_pending_syncs.values())
            for entry in entries:
                entry.timer.cancel()
            _pending_syncs.clear()
        for entry in entries:
            _push_with_thread_context(entry)


def reset_label_sync_for_tests() -> None:
    """Cancel every pending sync and drop captured contexts (for conftest).

    Unlike :func:`flush_pending_label_syncs`, this does **not** run the
    pending pushes — it discards them, which is what the autouse
    ``reset_state`` fixture wants between tests so a sync scheduled by
    one test's contexts can't fire after those contexts are gone.
    """
    global _syncing

    with _workers_lock:
        with _pending_lock:
            for entry in _pending_syncs.values():
                entry.timer.cancel()
            _pending_syncs.clear()
        with _sync_lock:
            _syncing = False


# Drain any pending debounced push at interpreter exit so the most recent
# vote isn't dropped on Ctrl-C / SIGQUIT / sys.exit.  Fires once per
# process; no-op when the queue is empty.  SIGKILL / os._exit bypass
# atexit and still lose the last 200ms — unavoidable for any debounce.
atexit.register(flush_pending_label_syncs)


def _push_to_labelset_source() -> None:
    """Synchronously push current detector labels to the linked source.

    Reads the active detector and dataset contexts via the standard
    resolution chain — callers running on a background thread must arrange
    thread-local context propagation before invoking this.
    """
    from vtscore.state.core import get_active_detector_context

    ctx = get_active_detector_context()
    if ctx is None or not ctx.labelset_source:
        return

    cfg = ctx.labelset_source
    source_name = cfg.get("source_name", "")
    if not source_name:
        return

    from vtscore.labels.sources import get_labelset_source

    source = get_labelset_source(source_name)
    if source is None:
        logger.warning("Unknown labelset source: %s", source_name)
        return

    field_values = cfg.get("field_values", {})

    from vtscore.datasets.labelset import LabelSet
    from vtscore.detectors.dataset_sync import validated_vote_snapshot
    from vtscore.detectors.input_spec import build_detector_meta
    from vtscore.detectors.store import _detector_path, _read_detector

    # Read the detector JSON so the exported labelset can carry its
    # input_spec / media_type alongside the in-memory threshold.  Anything
    # missing is simply omitted from detector_meta.
    detector_data = _read_detector(_detector_path(ctx.name)) or {}
    threshold = ctx.threshold if ctx.model is not None else None
    detector_meta = build_detector_meta(detector_data, threshold=threshold)

    # Serialize against any in-progress sync_from on another thread.
    # Re-check the flag inside the lock so we never push partial state
    # back to the source during an import.
    with _sync_lock:
        if _syncing:
            return
        try:
            # Atomic snapshot so the votes we serialise are guaranteed to be
            # keyed in the same dataset's cid space as the medias they're
            # composed with — even under concurrent dataset-switch requests
            # on the same detector.  ``safe=False`` means we couldn't prove
            # consistency; pushing an empty labelset would clobber whatever
            # the external source has, so we skip this push and let the next
            # vote re-trigger the debounced timer.
            vote_snap = validated_vote_snapshot()
            if not vote_snap.safe:
                return
            labelset = LabelSet.from_clips_and_votes(
                vote_snap.medias,
                vote_snap.good_votes,
                vote_snap.bad_votes,
                vote_region_boxes=vote_snap.vote_region_boxes,
                detector_meta=detector_meta or None,
            )
            source.save(labelset, field_values)
        except Exception as exc:
            logger.exception("Failed to sync labels to source: %s", exc)


def sync_from_labelset_source(detector_id: str | None = None) -> list[dict[str, str]] | None:  # noqa: C901
    """Pull labels from the active detector's labelset source and apply them.

    Args:
        detector_id: If given, operate on this detector context. Otherwise
            use the currently active detector context.

    Returns:
        The imported label list, or ``None`` if no source is configured
        or the source file doesn't exist yet.
    """
    from vtscore.state.core import get_active_detector_context, get_detector_context

    if detector_id is not None:
        ctx = get_detector_context(detector_id)
    else:
        ctx = get_active_detector_context()

    if ctx is None or not ctx.labelset_source:
        return None

    cfg = ctx.labelset_source
    source_name = cfg.get("source_name", "")
    if not source_name:
        return None

    from vtscore.labels.sources import get_labelset_source

    source = get_labelset_source(source_name)
    if source is None:
        logger.warning("Unknown labelset source: %s", source_name)
        return None

    field_values = cfg.get("field_values", {})
    try:
        labelset = source.load_full(field_values)
    except Exception as exc:
        logger.exception("Failed to load from labelset source: %s", exc)
        return None

    if not labelset.elements and not labelset.detector_meta:
        return None

    # If the source carried a detector_meta block, fold its input_spec
    # (and media_type, when the receiver is missing one) into the
    # receiving detector's on-disk JSON.  threshold is intentionally
    # *not* persisted — the receiver will retrain its MLP from the
    # imported labels and recompute its own threshold.
    if labelset.detector_meta:
        from vtscore.detectors.input_spec import apply_detector_meta
        from vtscore.detectors.store import _detector_path, _read_detector, _write_detector

        det_path = _detector_path(ctx.name)
        det_data = _read_detector(det_path)
        if det_data is not None and apply_detector_meta(det_data, labelset.detector_meta):
            _write_detector(det_path, det_data)

    labels = [el.to_dict() for el in labelset.elements]
    if not labels:
        return labels

    # Hold _sync_lock for the entire apply pass so that any concurrent
    # sync_to_labelset_source() on another thread blocks (or skips, on
    # re-entry from this same thread) instead of pushing the pre-import
    # state back to the source.
    global _syncing
    applied_any = False
    with _sync_lock:
        _syncing = True
        try:
            from vtscore.state.core import get_active_context
            from vtscore.state.votes import apply_label

            ds_medias = get_active_context().medias
            for entry in labels:
                label = entry.get("label")
                md5 = entry.get("md5")
                if label not in ("good", "bad") or not md5:
                    continue

                # Find media by md5
                for mid, media in ds_medias.items():
                    if media.get("md5") == md5:
                        # System-driven auto-import on detector load; the
                        # ``record_detector_import`` call below covers the
                        # achievement credit instead of crediting each
                        # imported label as a user vote.
                        apply_label(mid, label, record_achievement=False)
                        applied_any = True
                        break
        finally:
            _syncing = False

    if applied_any:
        from vtsearch.achievements import record_detector_import

        record_detector_import(ctx.detector_id)

    return labels
