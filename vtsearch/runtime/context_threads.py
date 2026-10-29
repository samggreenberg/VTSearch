"""Background-thread helper that carries the current execution context.

VTSearch threads its execution context through three thread-locals: the
authenticated user (consulted by per-user settings reads / writes), the
active :class:`~vtscore.state.core.DatasetContext` (consulted by the
``medias`` / ``coverage_atlas`` proxies), and the active
:class:`~vtscore.state.core.DetectorContext` (consulted by the
``good_votes`` / ``bad_votes`` / ``model`` proxies).

A background thread that wants to write per-user settings, read the
active dataset's medias, or train against the active detector's labels
has to set all three thread-locals before its target body runs.
Forgetting one silently writes to the wrong user file or no-ops against
the empty fallback context.  This helper snapshots the calling thread's
context at spawn time and re-installs it inside the new thread, so the
plumbing is invisible to the caller.

Which idiom to reach for
------------------------

App-tier background work runs under one of exactly two idioms, and they
are chosen by *how the work is tracked*, not by how long it takes:

``JobManager`` (:mod:`vtscore.concurrency.async_jobs`)
    For work where only the **latest** request matters and a duplicate
    request should coalesce rather than run twice: training, sorting,
    eval.  One job runs at a time per manager, a second ``start()``
    parks in a single coalescing pending slot, and results are cached by
    signature so an unchanged re-request is free.  It owns its own
    context propagation (``JobManager._run`` replays the requester's
    user + dataset + detector the same way :func:`spawn` does), so code
    reaching for a manager never calls ``spawn``.

:func:`spawn` (this module)
    For everything else: work that is genuinely per-invocation and may
    run concurrently with siblings — loading a dataset, building a
    coverage atlas, promoting a selection, re-embedding a detector,
    ingesting a labelset.  ``spawn`` provides context replay and nothing
    else; whatever progress or cancellation surface the job needs is
    layered on by the caller.  In practice that is a
    :class:`~vtscore.concurrency.progress.ProgressTracker` obtained from
    the ``loading_tasks`` / ``detector_loading_tasks`` registries, whose
    per-task entries feed the ``loading-tasks`` SSE channel and the
    dashboard's Cancel button.  Callers that register a task that way
    must pass ``start=False`` and start the thread themselves, so the
    worker is registered before it runs — see the ``start`` parameter.

There is no third idiom in the app tier.  A bare ``threading.Thread`` in
``vtsearch/`` is a bug: it drops all three thread-locals, so the body
writes to the wrong user's settings file and reads the empty fallback
dataset.  (The library tier is different: ``vtscore`` cannot import this
module, so its own background helpers replay context with the
``vtscore.state.current_user`` / ``vtscore.state.core`` context managers
directly.)
"""

from __future__ import annotations

import threading
from typing import Any, Callable

__all__ = ["spawn"]


def spawn(
    target: Callable[..., Any],
    *args: Any,
    name: str | None = None,
    daemon: bool = True,
    start: bool = True,
    **kwargs: Any,
) -> threading.Thread:
    """Start *target* in a daemon thread with the current context replayed.

    Snapshots the calling thread's user, dataset context, and detector
    context at call time, then re-installs them inside the new thread
    before *target* runs.  This is the recommended way to spawn ad-hoc
    background work from a Flask request handler, plugin body, or other
    code that depends on the active context resolving the same way it
    does on the calling thread.

    Returns the :class:`threading.Thread` so callers can join it if they
    need to.  The thread is daemon by default so a forgotten join
    doesn't keep the interpreter alive at shutdown.

    Pass ``start=False`` to get the thread back *unstarted*; the
    snapshot is still taken here, on the calling thread, so the caller
    may start it later without losing the context.  This exists for the
    one thing a caller has to do between construction and start:
    register the worker with a
    :class:`~vtscore.concurrency.progress.LoadingTasksTracker` via
    ``set_worker``, which is documented to happen *before* the thread
    runs so a cancel arriving in the same instant can tell "not started
    yet" from "nothing here"::

        worker = spawn(task, name="ds-promote-abc123", start=False)
        loading_tasks.set_worker(task_id, worker)
        worker.start()

    See the module docstring for when to reach for ``spawn`` at all
    versus :class:`~vtscore.concurrency.async_jobs.JobManager`.
    """
    user_snapshot = _snapshot_user()
    ds_snapshot, det_snapshot = _snapshot_state_contexts()

    def _runner() -> None:
        _install_user(user_snapshot)
        _install_state_contexts(ds_snapshot, det_snapshot)
        try:
            target(*args, **kwargs)
        finally:
            # Clear so a reused thread doesn't carry stale context into
            # the next job.  The set/clear pattern mirrors what
            # JobManager._run does.
            _install_user(None)
            _install_state_contexts(None, None)

    thread = threading.Thread(target=_runner, name=name, daemon=daemon)
    if start:
        thread.start()
    return thread


def _snapshot_user() -> str | None:
    """Capture the user to propagate to the spawned thread.

    Resolution: an explicit thread-local user wins (the spawn lineage,
    set by a parent ``spawn`` call or by ``set_thread_user``); otherwise
    the current Flask request's ``g.user`` if there is one; otherwise
    ``None``.  We deliberately do *not* fall back to
    :func:`vtsearch.auth.get_current_user`'s ``"default"`` sentinel;
    that would clobber the spawned thread's thread-local with
    ``"default"`` even when no user was ever explicitly set, masking
    the "no user" state.
    """
    from vtsearch.auth import get_thread_user  # noqa: PLC0415

    tl = get_thread_user()
    if tl:
        return tl
    try:
        from flask import g  # noqa: PLC0415

        return getattr(g, "user", None)
    except RuntimeError:
        # No Flask request context (CLI / background).
        return None


def _install_user(user: str | None) -> None:
    from vtsearch.auth import set_thread_user  # noqa: PLC0415

    set_thread_user(user)


def _snapshot_state_contexts() -> tuple[Any, Any]:
    """Capture the active dataset + detector context as resolved on the
    calling thread.

    When the current request named an *unloaded* dataset / detector via
    ``X-Dataset-Id`` / ``X-Detector-Id`` (e.g. the active-context header
    still points at a since-evicted id), the proxy resolver raises
    ``DatasetNotLoadedError`` / ``DetectorNotLoadedError`` (the H16/H34
    contract for data-serving routes). Snapshotting context to hand off
    to a background thread is *not* a data-serving read, so that raise
    must not 409 the request that called :func:`spawn` — most notably the
    registry-load handler, whose whole job is to *recover* from an
    unloaded dataset. Fall back to ``None`` for the offending half; the
    spawned body either sets its own context (the load task creates a
    fresh one) or harmlessly resolves to the empty fallback.
    """
    # Library-tier imports; keep deferred so the helper stays importable
    # in environments where ``vtscore.state`` hasn't been initialised
    # (notably the lib-clean test runner).
    from vtscore.state.core import (  # noqa: PLC0415
        DatasetNotLoadedError,
        DetectorNotLoadedError,
    )
    from vtsearch.state import get_active_context, get_active_detector_context  # noqa: PLC0415

    try:
        ds_ctx = get_active_context()
    except DatasetNotLoadedError:
        ds_ctx = None
    try:
        det_ctx = get_active_detector_context()
    except DetectorNotLoadedError:
        det_ctx = None
    return ds_ctx, det_ctx


def _install_state_contexts(ds_ctx: Any, det_ctx: Any) -> None:
    from vtscore.state.core import (  # noqa: PLC0415
        set_thread_dataset_context,
        set_thread_detector_context,
    )

    set_thread_dataset_context(ds_ctx)
    set_thread_detector_context(det_ctx)
