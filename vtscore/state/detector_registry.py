"""The loaded-detector store, the detector request-missing sentinel, and active-detector resolution.

The detector mirror of :mod:`vtscore.state.dataset_registry`, plus the
forced override (:func:`override_detector_context`) and the helpers that act
on every loaded detector at once.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import vtscore.state.resolution as _resolution
from vtscore.state.detector_context import DetectorContext
from vtscore.state.primitives import (
    _context_registry_lock,
    _freeze_into,
    _frozen_mutation_error,
    _state_lock,
)
from vtscore.state.resolution import _thread_local

# ---------------------------------------------------------------------------
# Detector context store and thread-local fallback
# ---------------------------------------------------------------------------

# Maps detector_id -> DetectorContext for every in-memory detector.
_detector_contexts: dict[str, DetectorContext] = {}

# Fallback context used when no detector is set.  Vote proxies delegate
# to this so that code accessing ``good_votes``, ``bad_votes``, etc. when
# no detector is loaded sees empty containers rather than crashing.
_empty_detector_context = DetectorContext("")


class _RequestMissingDetectorContext(DetectorContext):
    """Sentinel returned inside a Flask request when no detector was identified.

    Counterpart of :class:`_RequestMissingDatasetContext`: every container
    is frozen and the context refuses attribute assignment.  Without this
    sentinel, vote-mutation endpoints called without ``X-Detector-Id``
    would silently accumulate votes on the global
    ``_empty_detector_context`` (audit bug H13 / H14).
    """

    __slots__ = ()

    def __init__(self) -> None:
        # Copy every slot from a fresh, real ``DetectorContext`` (containers
        # frozen on the way in) rather than hand-listing them, so a slot added
        # to ``DetectorContext`` can never go missing here.
        _freeze_into(self, DetectorContext(""), "detector")
        # Use object.__setattr__ to bypass our own write guard.
        object.__setattr__(self, "detector_id", "__request_missing__")

    def __setattr__(self, name: str, value: Any) -> None:
        raise _frozen_mutation_error("detector")


_request_missing_detector_context = _RequestMissingDetectorContext()


def is_request_missing_detector_context(ctx: Any) -> bool:
    """Return True iff *ctx* is the request-missing detector sentinel."""
    return ctx is _request_missing_detector_context


def get_active_detector_context() -> DetectorContext:
    """Return the ``DetectorContext`` for the current execution context.

    Resolution order:
    1. Forced override (``override_detector_context`` context manager)
    2. Request-scoped context (set by ``before_request`` from ``X-Detector-Id`` header)
    3. Thread-local context (set by ``set_thread_detector_context``)
    4. Request-missing sentinel, inside a Flask request with no header
       and no thread-local; mutations raise
       :class:`RequestMissingContextError`.
    5. Empty fallback context for CLI / library callers outside Flask.
    """
    # 1. Forced override (set by override_detector_context context manager)
    forced = getattr(_thread_local, "forced_detector_context", None)
    if forced is not None:
        return forced
    # 2. Per-request override (Flask shim or whatever the host app registered)
    req_ctx = _resolution._detector_context_resolver()
    if req_ctx is not None:
        return req_ctx
    # 3. Thread-local fallback
    ctx = getattr(_thread_local, "detector_context", None)
    if ctx is not None:
        return ctx
    # 4. Inside a Flask request with no header and no thread-local → fail
    #    loudly on mutation instead of polluting _empty_detector_context.
    if _resolution._request_context_predicate():
        return _request_missing_detector_context
    return _empty_detector_context


@contextmanager
def override_detector_context(ctx: DetectorContext) -> Iterator[None]:
    """Force :func:`get_active_detector_context` to return *ctx* for the
    duration of the ``with`` block.

    Takes priority over the registered request resolver and the thread-local
    fallback.  Use this from call sites that need to swap the active detector
    inside their own body (typically when applying labels to a freshly-loaded
    detector that isn't the request's currently-active one) without having
    to know whether they're running inside a Flask request or a background
    thread.
    """
    prev = getattr(_thread_local, "forced_detector_context", None)
    _thread_local.forced_detector_context = ctx
    try:
        yield
    finally:
        _thread_local.forced_detector_context = prev


def set_thread_detector_context(ctx: DetectorContext | None) -> None:
    """Set the thread-local detector context for the current thread.

    Prefer :func:`thread_detector_context` (a context manager) for new
    code (it saves and restores the prior value automatically).
    """
    _thread_local.detector_context = ctx


def get_thread_detector_context() -> DetectorContext | None:
    """Return the thread-local detector context, or ``None``."""
    return getattr(_thread_local, "detector_context", None)


@contextmanager
def thread_detector_context(ctx: DetectorContext | None) -> Iterator[None]:
    """Scope the thread-local detector context to *ctx* for the ``with``-block.

    Snapshots the prior thread-local value on entry and restores it on
    exit, so nested scopes compose correctly and a reused / pooled
    thread cannot leak the wrong detector context across jobs.

    Unlike :class:`with_detector_context` (which looks the context up by
    ID from the registry and requires it to be registered), this helper
    takes a context object directly.
    """
    prev = getattr(_thread_local, "detector_context", None)
    _thread_local.detector_context = ctx
    try:
        yield
    finally:
        _thread_local.detector_context = prev


def register_detector_context(ctx: DetectorContext) -> None:
    """Add *ctx* to the detector context store, keyed by its ``detector_id``.

    Also clears the module-level progress cache so that stale training
    indicators from a previously-active detector are not reused.
    """
    from vtscore.detectors.labeling_progress import clear_progress_cache

    with _state_lock, _context_registry_lock:
        _detector_contexts[ctx.detector_id] = ctx
    # ``_progress_lock`` is acquired strictly outside ``_state_lock`` so the
    # two locks never establish a cross-module ordering (audit M1).
    clear_progress_cache()


def unregister_detector_context(detector_id: str) -> DetectorContext | None:
    """Remove and return the detector context for *detector_id*, or ``None``.

    Also clears the progress cache so stale cached steps from the removed
    detector are not used by a subsequent detector.
    """
    from vtscore.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        with _context_registry_lock:
            ctx = _detector_contexts.pop(detector_id, None)
        tl_ctx = getattr(_thread_local, "detector_context", None)
        if tl_ctx is not None and tl_ctx.detector_id == detector_id:
            _thread_local.detector_context = None
    # ``_progress_lock`` is acquired strictly outside ``_state_lock`` so the
    # two locks never establish a cross-module ordering (audit M1).
    clear_progress_cache()
    return ctx


def get_detector_context(detector_id: str) -> DetectorContext | None:
    """Return the detector context for *detector_id*, or ``None`` if not loaded.

    Lock-free with respect to ``_state_lock``, for the same reason as
    :func:`get_context`: the per-request ``X-Detector-Id`` resolution runs
    through here on every request, exempt routes included.
    """
    with _context_registry_lock:
        return _detector_contexts.get(detector_id)


def list_loaded_detector_ids() -> list[str]:
    """Return all detector IDs that have an in-memory context."""
    with _context_registry_lock:
        return list(_detector_contexts.keys())


def loaded_detector_contexts() -> list[DetectorContext]:
    """Return a snapshot of every loaded detector context.

    A snapshot rather than a live view so callers can mutate each context
    under ``_state_lock`` without iterating ``_detector_contexts`` while
    another thread registers one.
    """
    with _context_registry_lock:
        return list(_detector_contexts.values())


def clear_all_detector_contexts() -> None:
    """Remove all detector contexts and clear the thread-local.  For tests."""
    with _state_lock:
        with _context_registry_lock:
            _detector_contexts.clear()
        _thread_local.detector_context = None
        _empty_detector_context.__init__("")  # type: ignore[misc]


def invalidate_loaded_detector_models() -> None:
    """Drop the cached head and threshold on every loaded detector context.

    Called by the setters of training-relevant settings (``calibrate_count``,
    ``calibration_fraction``) so
    the next consumer that would otherwise short-circuit on the cached
    ``det_ctx.model`` / ``det_ctx.threshold`` (``/api/find-label``,
    ``/api/find``, ``/api/auto-detect``) retrains under the new setting.

    Only a learned sort retrains; a vote does not.  Label changes are caught
    separately, by the labelset signature those consumers check before reusing
    ``det_ctx.model`` (issue #4204), so this is purely about making them honour
    live setting changes.
    """
    with _state_lock:
        for ctx in loaded_detector_contexts():
            ctx.model = None
            ctx.threshold = 0.5
