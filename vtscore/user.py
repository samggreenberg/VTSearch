"""Library-tier current-user identity.

Library code needs to know *who* triggered a piece of work: background
jobs replay the requesting user onto their worker thread so per-user
settings writes land in the right file, exporters interpolate
``{username}`` into output paths, plugin normalisation resolves
``{username}`` placeholders, and so on.

In the Flask app that identity comes from ``flask.g.user`` (set by the
``before_request`` middleware).  The library must not import Flask - or
:mod:`vtsearch` at all - so the request-scoped half is a **pluggable
resolver** mirroring :func:`vtscore.state.core.register_request_context_predicate`:
``vtsearch/shim/`` installs a Flask-aware resolver at app startup, and
the default resolver returns ``None`` so library-only callers (CLI,
embedders, tests) fall through to the thread-local.

The thread-local half lives here rather than in the app because it is
what background threads use to carry identity across a thread boundary,
and those threads are spawned by library code
(:class:`vtscore.concurrency.async_jobs.JobManager`,
:mod:`vtscore.datasets.load_pipeline`).  ``vtsearch.auth`` re-exports
every name in this module, so app-tier callers are unaffected and there
is exactly one thread-local backing store.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

#: Username reported when nothing identifies the caller (CLI, tests,
#: threads with no explicit user).
DEFAULT_USER = "default"

# Thread-local fallback used when there is no request-scoped user.
# Background threads spawned by request handlers should use the
# :func:`thread_user` context manager to propagate the user that
# triggered them (it snapshots and restores the prior value
# automatically); tests can use the bare :func:`set_thread_user` setter
# to pin or clear a user without a request context.
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Request-user resolver hook
# ---------------------------------------------------------------------------


def _default_request_user_resolver() -> str | None:
    return None


_request_user_resolver: Callable[[], str | None] = _default_request_user_resolver


def register_request_user_resolver(fn: Callable[[], str | None]) -> None:
    """Install the resolver consulted first by :func:`get_current_user`.

    *fn* returns the username for the in-flight request, or ``None`` when
    there is no request (background thread, CLI, library caller) so
    resolution falls through to the thread-local.  ``vtsearch/shim/``
    wires this to ``flask.g.user`` at app startup.
    """
    global _request_user_resolver
    _request_user_resolver = fn


def get_current_user() -> str:
    """Return the username for the current request / thread.

    Resolution order:

    1. The registered request-user resolver (``flask.g.user`` in the app).
    2. Thread-local fallback (scoped by the :func:`thread_user` context
       manager; background threads spawned from a request handler use
       this).
    3. :data:`DEFAULT_USER` (CLI, tests, threads with no explicit user).
    """
    request_user = _request_user_resolver()
    if request_user:
        return request_user
    tl_user = getattr(_thread_local, "user", None)
    if tl_user:
        return tl_user
    return DEFAULT_USER


# ---------------------------------------------------------------------------
# Thread-local user
# ---------------------------------------------------------------------------


def set_thread_user(username: str | None) -> None:
    """Set the thread-local user for the current thread.

    Prefer :func:`thread_user` (a context manager) for new code, as it saves
    and restores the prior value automatically, removing the need for a
    manual ``try/finally`` discipline that is easy to get wrong (and would
    leak across requests if these threads were ever pooled).

    Pass ``None`` to clear.
    """
    _thread_local.user = username


def get_thread_user() -> str | None:
    """Return the thread-local user, or ``None`` if unset."""
    return getattr(_thread_local, "user", None)


@contextmanager
def thread_user(username: str | None) -> Iterator[None]:
    """Scope the thread-local user to *username* for the ``with``-block.

    On entry, snapshots the prior thread-local user (if any) and sets it
    to *username*.  On exit, restores the snapshot, so nested scopes
    compose correctly and a pooled / reused thread cannot leak identity
    across jobs even if the inner body raises.

    Use this from background threads (or anywhere outside a request
    context) that need :func:`get_current_user` to resolve to a specific
    user::

        request_user = get_current_user()

        def task():
            with thread_user(request_user):
                ...  # per-user settings writes resolve correctly here
    """
    prev = getattr(_thread_local, "user", None)
    _thread_local.user = username
    try:
        yield
    finally:
        _thread_local.user = prev
