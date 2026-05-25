"""Background-thread helper that carries the current execution context.

VTSearch threads its execution context through three thread-locals: the
authenticated user (consulted by per-user settings reads / writes), the
active :class:`~vtscore.state.core.DatasetContext` (consulted by the
``medias`` / ``diversity_tree`` proxies), and the active
:class:`~vtscore.state.core.DetectorContext` (consulted by the
``good_votes`` / ``bad_votes`` / ``model`` proxies).

A background thread that wants to write per-user settings, read the
active dataset's medias, or train against the active detector's labels
has to set all three thread-locals before its target body runs.
Forgetting one silently writes to the wrong user file or no-ops against
the empty fallback context.  This helper snapshots the calling thread's
context at spawn time and re-installs it inside the new thread, so the
plumbing is invisible to the caller.
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
    **kwargs: Any,
) -> threading.Thread:
    """Start *target* in a daemon thread with the current context replayed.

    Snapshots the calling thread's user, dataset context, and detector
    context at call time, then re-installs them inside the new thread
    before *target* runs.  This is the recommended way to spawn ad-hoc
    background work from a Flask request handler, plugin body, or other
    code that depends on the active context resolving the same way it
    does on the calling thread.

    Returns the started :class:`threading.Thread` so callers can join it
    if they need to.  The thread is daemon by default so a forgotten
    join doesn't keep the interpreter alive at shutdown.

    Long-running jobs that need cancellation / progress reporting should
    keep using :class:`~vtscore.concurrency.async_jobs.JobManager`, which
    already handles context propagation and exposes a richer
    job-tracking surface; ``spawn`` is for the simpler fire-and-forget
    cases.
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
    thread.start()
    return thread


def _snapshot_user() -> str | None:
    """Capture the current user, falling back to the thread-local when
    no Flask request context is active."""
    from vtsearch.auth import get_current_user, get_thread_user  # noqa: PLC0415

    try:
        # Inside a Flask request: ``g.user`` is set by the before_request
        # middleware.  Use ``get_current_user()`` so the resolution rules
        # stay in one place.
        return get_current_user()
    except Exception:
        # No request context (CLI / background): fall back to whatever
        # the calling thread had set, which may be None.
        return get_thread_user()


def _install_user(user: str | None) -> None:
    from vtsearch.auth import set_thread_user  # noqa: PLC0415

    set_thread_user(user)


def _snapshot_state_contexts() -> tuple[Any, Any]:
    """Capture the active dataset + detector context as resolved on the
    calling thread."""
    # Library-tier import — keep deferred so the helper stays importable
    # in environments where ``vtscore.state`` hasn't been initialised
    # (notably the lib-clean test runner).
    from vtsearch.state import get_active_context, get_active_detector_context  # noqa: PLC0415

    return get_active_context(), get_active_detector_context()


def _install_state_contexts(ds_ctx: Any, det_ctx: Any) -> None:
    from vtscore.state.core import (  # noqa: PLC0415
        set_thread_dataset_context,
        set_thread_detector_context,
    )

    set_thread_dataset_context(ds_ctx)
    set_thread_detector_context(det_ctx)
