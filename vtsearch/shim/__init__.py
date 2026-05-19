"""Flask-aware glue between the (library-candidate) ``vtsearch.state`` core
and the Flask request lifecycle.

The library is being split into a Flask-free ``vtscore`` and a Flask-
wrapping ``vtsearch`` app — see ``docs/plans/extract-library.md``.  Any
piece of code that wants to read or override the per-request
``DatasetContext`` / ``DetectorContext`` via ``flask.g`` belongs here
so the core package can stay import-clean.
"""

from __future__ import annotations

from typing import Any


def _flask_dataset_context_resolver() -> Any:
    """Read the request-scoped dataset context off ``flask.g``.

    Returns ``None`` outside a request context (background threads, CLI,
    library callers) so :func:`vtsearch.state.core.get_active_context`
    can fall through to its thread-local / empty-context fallback.
    """
    from flask import g, has_request_context

    if has_request_context():
        return getattr(g, "_dataset_context", None)
    return None


def _flask_detector_context_resolver() -> Any:
    """Counterpart of :func:`_flask_dataset_context_resolver` for detectors."""
    from flask import g, has_request_context

    if has_request_context():
        return getattr(g, "_detector_context", None)
    return None


def register_flask_context_resolvers() -> None:
    """Install Flask-aware request-context resolvers on ``vtsearch.state.core``.

    Called once during Flask app startup.  After this, the ``medias`` /
    ``good_votes`` proxies and the ``get_active_*_context()`` helpers
    pick up whatever the ``before_request`` hook stashes on ``g`` for
    the duration of each request.
    """
    from vtsearch.state.core import (
        register_dataset_context_resolver,
        register_detector_context_resolver,
    )

    register_dataset_context_resolver(_flask_dataset_context_resolver)
    register_detector_context_resolver(_flask_detector_context_resolver)


def register_app_persistence_hooks() -> None:
    """Wire library-side persistence hooks to the app's ``vtsearch.settings``.

    The library exposes a few "let the app persist this" hook points so it
    doesn't import settings directly (see Phase 2 of
    ``docs/plans/extract-library.md``).  This function installs the Flask
    app's settings as the backing store for each hook.
    """
    from vtsearch.datasets.load_pipeline import register_last_embedder_persistence_hook
    from vtsearch.settings import set_last_embedder_for_media_type

    register_last_embedder_persistence_hook(set_last_embedder_for_media_type)


__all__ = [
    "register_app_persistence_hooks",
    "register_flask_context_resolvers",
]
