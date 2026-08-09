"""Who is this work being done for? - the Flask-free half of user identity.

Library code needs a username in a handful of places (per-user settings
writes, label-sync attribution, background jobs that must run "as" the
user who triggered them).  In the Flask app that identity comes off
``g.user``; on the CLI or in a plain library embedding there is no
request at all.

So identity resolves in three steps, none of which import Flask:

1. A pluggable **request-user resolver**.  Default returns ``None``;
   :mod:`vtsearch.auth` registers a Flask-aware one at import time, so an
   app process reads ``g.user`` and a library process doesn't.
2. The **thread-local** user, scoped by :func:`thread_user`.  Background
   threads spawned from a request use this to inherit the requester's
   identity.
3. ``"default"`` - the single-user / CLI / test answer.

``vtsearch.auth`` re-exports every name here, so there is exactly one
thread-local backing the whole app; importing either module gets you the
same storage.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Callable, Optional


#: Fallback username when nothing else identifies the caller.
DEFAULT_USER = "default"

_thread_local = threading.local()


def _default_request_user_resolver() -> str | None:
    """No request framework installed - nobody to ask."""
    return None


_request_user_resolver: Callable[[], Optional[str]] = _default_request_user_resolver


def register_request_user_resolver(fn: Callable[[], Optional[str]]) -> None:
    """Install the hook that reads the *request-scoped* user, if any.

    The resolver must return ``None`` (not raise) when there is no
    request in flight.  :mod:`vtsearch.auth` wires this to Flask's
    ``g.user`` at import time; the library default leaves it unset so
    :func:`get_current_user` falls through to the thread-local.
    """
    global _request_user_resolver
    _request_user_resolver = fn


def reset_request_user_resolver() -> None:
    """Restore the Flask-free default resolver (tests)."""
    global _request_user_resolver
    _request_user_resolver = _default_request_user_resolver


def get_current_user() -> str:
    """Return the username the current work belongs to.

    Resolution order: registered request-user resolver (``g.user`` under
    Flask) → thread-local (see :func:`thread_user`) → ``"default"``.
    """
    request_user = _request_user_resolver()
    if request_user:
        return request_user
    tl_user = getattr(_thread_local, "user", None)
    if tl_user:
        return tl_user
    return DEFAULT_USER


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
