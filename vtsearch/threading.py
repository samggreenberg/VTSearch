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
    """Capture the user to propagate to the spawned thread.

    Resolution: an explicit thread-local user wins (the spawn lineage,
    set by a parent ``spawn`` call or by ``set_thread_user``); otherwise
    the current Flask request's ``g.user`` if there is one; otherwise
    ``None``.  We deliberately do *not* fall back to
    :func:`vtsearch.auth.get_current_user`'s ``"default"`` sentinel —
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
