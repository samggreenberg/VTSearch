"""The host-installed hooks that steer context resolution, and the shared thread-local.

``register_request_context_predicate``, ``register_dataset_context_resolver``
and ``register_detector_context_resolver`` *rebind* this module's globals,
and :func:`~vtscore.state.dataset_registry.get_active_context` /
:func:`~vtscore.state.detector_registry.get_active_detector_context` read
them at call time.  So read them by attribute (``resolution._dataset_context_resolver``),
never ``from vtscore.state.resolution import _dataset_context_resolver``: that
binds a copy a later ``register_*`` call does not update.  For the same
reason :mod:`vtscore.state.core` does not re-export them, and
:mod:`vtscore.host_seams` snapshots and restores them here.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

# Thread-local storage for the fallback dataset/detector context.
# Used by background threads and tests that operate outside a Flask
# request context.  Each thread sets its own value; no global "active"
# pointer exists.
_thread_local = threading.local()


# ---------------------------------------------------------------------------
# Flask-request predicate hook
# ---------------------------------------------------------------------------
# Returns True when execution is inside a Flask request that should be
# refused if no explicit dataset/detector was identified.  The vtscore
# library stays Flask-free; the Flask shim registers
# ``flask.has_request_context`` here at app startup.  Outside Flask
# (CLI, library callers, background threads), the default ``lambda: False``
# stays in place so :func:`get_active_context` keeps falling back to the
# empty context as before.
def _default_request_context_predicate() -> bool:
    return False


_request_context_predicate: Callable[[], bool] = _default_request_context_predicate


def register_request_context_predicate(fn: Callable[[], bool]) -> None:
    """Install the predicate used to decide whether to return the
    request-missing sentinel instead of the empty fallback context.

    The Flask shim wires this to :func:`flask.has_request_context` at
    startup so that a Flask request without an identified dataset/detector
    sees the frozen sentinel (which fails loudly on mutation) rather than
    silently landing on the empty global fallback.
    """
    global _request_context_predicate
    _request_context_predicate = fn


# ---------------------------------------------------------------------------
# Pluggable per-request context resolvers
# ---------------------------------------------------------------------------
# In the Flask app, the ``before_request`` hook resolves an
# ``X-Dataset-Id`` / ``X-Detector-Id`` header to the matching context and
# stashes it on ``g``.  The proxy objects then need to read it back.
#
# To keep this module Flask-free (so it can move into ``vtscore`` later -
# see ``../docs/architecture.md``), the read side is exposed as a
# **pluggable resolver**: a callable that returns the current request's
# DatasetContext / DetectorContext, or ``None`` if there is no request.
#
# The Flask integration lives in ``vtsearch/shim/`` and registers Flask-
# aware resolvers at app startup.  By default both resolvers return
# ``None``; callers fall back to the thread-local context (set by
# ``set_thread_*_context`` for background threads and tests).
# ---------------------------------------------------------------------------


def _default_context_resolver() -> Any:
    return None


_dataset_context_resolver: Callable[[], Any] = _default_context_resolver
_detector_context_resolver: Callable[[], Any] = _default_context_resolver


def register_dataset_context_resolver(fn: Callable[[], Any]) -> None:
    """Install the function used to resolve the current request's dataset context.

    The resolver should return a ``DatasetContext`` or ``None``.  The Flask
    shim installs a resolver that reads from ``flask.g`` at app startup;
    library-only callers can leave the default in place.
    """
    global _dataset_context_resolver
    _dataset_context_resolver = fn


def register_detector_context_resolver(fn: Callable[[], Any]) -> None:
    """Install the function used to resolve the current request's detector context.

    Counterpart to :func:`register_dataset_context_resolver`.
    """
    global _detector_context_resolver
    _detector_context_resolver = fn
