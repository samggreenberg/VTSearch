"""Route decorators that require an explicit dataset / detector context.

Both guards reject a request that names no context rather than letting it land
on whatever ``get_active_*_context()`` happens to resolve to; see the
docstrings for the audit finding (H34) they close.
"""

from __future__ import annotations

from functools import wraps
from typing import Callable

from flask import request
from flask_smorest import abort


def _has_context_id(header_name: str, query_param: str) -> bool:
    """Return True iff *header_name* or *query_param* identifies a context.

    Matches the precedence applied by ``before_request`` in ``app.py``:
    header first, then query-param fallback for browser-native requests
    (``<img src>`` / ``<audio src>`` / ``<video src>``) that bypass
    Angular's HttpClient interceptor.
    """
    return bool(request.headers.get(header_name) or request.args.get(query_param))


def require_detector_header(fn: Callable) -> Callable:
    """Route decorator: reject 400 if no ``X-Detector-Id`` is identified.

    Closes logical-bug-audit H34: vote-mutating endpoints used to silently
    write to whatever ``get_active_detector_context()`` resolved to when
    the client dropped the header. The frozen ``_request_missing_detector_context``
    sentinel catches the case where both the header *and* the thread-local
    are absent, but if any future code path sets the thread-local on a
    Flask request thread, a header-absent request would land on a stale
    detector. This guard rejects header-absent requests *before* the
    resolver chain runs, regardless of thread-local state (defence in
    depth).

    Apply to any endpoint that mutates ``DetectorContext`` state
    (``good_votes`` / ``bad_votes`` / ``label_history`` / ``vote_*`` /
    ``find_initial_labels`` / ``click_counter``). Pure reads, registry
    listings, and dashboard endpoints don't need it.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _has_context_id("X-Detector-Id", "detector_id"):
            abort(
                400,
                message=("X-Detector-Id header (or ?detector_id= query param) is required for this endpoint."),
            )
        return fn(*args, **kwargs)

    return wrapper


def require_dataset_header(fn: Callable) -> Callable:
    """Route decorator: reject 400 if no ``X-Dataset-Id`` is identified.

    Sister of :func:`require_detector_header`; closes the dataset-side
    analog of H34. Apply to any endpoint that mutates ``DatasetContext``
    state (``medias`` insertions, ``coverage_atlas`` rebuilds) or whose
    correctness depends on knowing which dataset's cid-keyed votes are
    being touched (label imports, fill-from-sort).
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _has_context_id("X-Dataset-Id", "dataset_id"):
            abort(
                400,
                message=("X-Dataset-Id header (or ?dataset_id= query param) is required for this endpoint."),
            )
        return fn(*args, **kwargs)

    return wrapper
