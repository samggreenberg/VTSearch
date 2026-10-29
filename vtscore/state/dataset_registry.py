"""The loaded-dataset store, the dataset request-missing sentinel, and active-dataset resolution.

Every write to ``_contexts`` takes ``_context_registry_lock`` (see its comment
in :mod:`vtscore.state.primitives`), so :func:`get_context` answers without
waiting on ``_state_lock``.  The host hooks :func:`get_active_context` consults
are read through :mod:`vtscore.state.resolution` at call time, since
``register_*`` rebinds them there.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import vtscore.state.resolution as _resolution
from vtscore.state.dataset_context import DatasetContext
from vtscore.state.primitives import (
    _context_registry_lock,
    _freeze_into,
    _frozen_mutation_error,
    _state_lock,
)
from vtscore.state.resolution import _thread_local

# ---------------------------------------------------------------------------
# Dataset context store and thread-local fallback
# ---------------------------------------------------------------------------

# Maps dataset_id -> DatasetContext for every in-memory dataset.
_contexts: dict[str, DatasetContext] = {}

# Fallback context used when no dataset is set.  Proxies delegate to
# this so that code accessing ``medias`` when nothing is loaded sees empty
# containers rather than crashing.
_empty_dataset_context = DatasetContext("")


class _RequestMissingDatasetContext(DatasetContext):
    """Sentinel returned inside a Flask request when no dataset was identified.

    Behaves as an empty :class:`DatasetContext` for reads, but every
    container is a :class:`_FrozenDict` / :class:`_FrozenList` that raises
    :class:`RequestMissingContextError` on any mutation, and the context
    itself refuses attribute assignment.  This converts the "header was
    dropped and we silently fell back to the empty global context"
    failure mode (audit bugs H13 / H16) into a loud error at the actual
    write site.
    """

    __slots__ = ()

    def __init__(self) -> None:
        # Copy every slot from a fresh, real ``DatasetContext`` (containers
        # frozen on the way in) rather than hand-listing them, so a slot added
        # to ``DatasetContext`` can never go missing here.  ``medias`` is a
        # property on the base class; ``_freeze_into`` writes its backing slot
        # directly with a frozen dict, so every mutation raises.  The revision
        # counter is inert here (a frozen dict never mutates), but the slot
        # must exist for the property / bump machinery not to AttributeError.
        _freeze_into(self, DatasetContext(""), "dataset")
        # Use object.__setattr__ to bypass our own write guard.
        object.__setattr__(self, "dataset_id", "__request_missing__")

    def __setattr__(self, name: str, value: Any) -> None:
        raise _frozen_mutation_error("dataset")


_request_missing_dataset_context = _RequestMissingDatasetContext()


def is_request_missing_dataset_context(ctx: Any) -> bool:
    """Return True iff *ctx* is the request-missing dataset sentinel."""
    return ctx is _request_missing_dataset_context


def get_active_context() -> DatasetContext:
    """Return the ``DatasetContext`` for the current execution context.

    Resolution order:
    1. Request-scoped context (set by ``before_request`` from ``X-Dataset-Id`` header)
    2. Thread-local context (set by ``set_thread_dataset_context``, for
       background threads and tests)
    3. Request-missing sentinel, when inside a Flask request that didn't
       identify a dataset (registered via
       :func:`register_request_context_predicate`).  Reads see an empty
       context; writes raise :class:`RequestMissingContextError`.
    4. Empty fallback context for CLI / library callers outside any
       Flask request.
    """
    # 1. Per-request override (Flask shim or whatever the host app registered)
    req_ctx = _resolution._dataset_context_resolver()
    if req_ctx is not None:
        return req_ctx
    # 2. Thread-local fallback
    ctx = getattr(_thread_local, "dataset_context", None)
    if ctx is not None:
        return ctx
    # 3. Inside a Flask request with no header and no thread-local → fail
    #    loudly on mutation instead of silently writing into the global
    #    empty context.
    if _resolution._request_context_predicate():
        return _request_missing_dataset_context
    return _empty_dataset_context


def set_thread_dataset_context(ctx: DatasetContext | None) -> None:
    """Set the thread-local dataset context for the current thread.

    Prefer :func:`thread_dataset_context` (a context manager) for new code
    (it saves and restores the prior value automatically).

    Called by test fixtures and background threads to direct proxy
    resolution without global state.
    """
    _thread_local.dataset_context = ctx


def get_thread_dataset_context() -> DatasetContext | None:
    """Return the thread-local dataset context, or ``None``."""
    return getattr(_thread_local, "dataset_context", None)


@contextmanager
def thread_dataset_context(ctx: DatasetContext | None) -> Iterator[None]:
    """Scope the thread-local dataset context to *ctx* for the ``with``-block.

    Snapshots the prior thread-local value on entry and restores it on
    exit, so nested scopes compose correctly and a reused / pooled
    thread cannot leak the wrong dataset context across jobs.

    Unlike :class:`with_dataset_context` (which looks the context up by
    ID from the registry and requires it to be registered), this helper
    takes a context object directly; useful for newly-created contexts
    that have not yet been registered.
    """
    prev = getattr(_thread_local, "dataset_context", None)
    _thread_local.dataset_context = ctx
    try:
        yield
    finally:
        _thread_local.dataset_context = prev


def register_context(ctx: DatasetContext) -> None:
    """Add *ctx* to the context store, keyed by its ``dataset_id``."""
    with _state_lock, _context_registry_lock:
        _contexts[ctx.dataset_id] = ctx


def unregister_context(dataset_id: str) -> DatasetContext | None:
    """Remove and return the context for *dataset_id*, or ``None``."""
    with _state_lock:
        with _context_registry_lock:
            ctx = _contexts.pop(dataset_id, None)
        # Clear thread-local if it was pointing to the removed context.
        tl_ctx = getattr(_thread_local, "dataset_context", None)
        if tl_ctx is not None and tl_ctx.dataset_id == dataset_id:
            _thread_local.dataset_context = None
        return ctx


def rekey_dataset_context(old_id: str, new_id: str) -> bool:
    """Re-key a loaded dataset context from *old_id* to *new_id*.

    Returns whether a context was found and re-keyed. Lives here rather
    than in the caller that needs it (the dataset-registration flow, which
    swaps a provisional task id for the real dataset id) so that every
    write to ``_contexts`` goes through ``_context_registry_lock``: a
    caller mutating the dict directly would be invisible to the lock-free
    readers above, which is the whole point of that lock.
    """
    with _state_lock, _context_registry_lock:
        ctx = _contexts.get(old_id)
        if ctx is None:
            return False
        ctx.dataset_id = new_id
        # Insert under the new key before removing the old, so the context
        # is never briefly invisible to concurrent lookups.
        _contexts[new_id] = ctx
        if old_id != new_id:
            _contexts.pop(old_id, None)
        return True


def get_context(dataset_id: str) -> DatasetContext | None:
    """Return the context for *dataset_id*, or ``None`` if not loaded.

    Takes ``_context_registry_lock`` only, never ``_state_lock`` -- see the
    comment on that lock. Every request resolves its ``X-Dataset-Id`` header
    through here, so this lookup must not queue behind a long-running vote
    or load.
    """
    with _context_registry_lock:
        return _contexts.get(dataset_id)


def list_loaded_dataset_ids() -> list[str]:
    """Return all dataset IDs that have an in-memory context."""
    with _context_registry_lock:
        return list(_contexts.keys())


def clear_all_contexts() -> None:
    """Remove all dataset contexts and clear the thread-local.  For tests."""
    with _state_lock:
        with _context_registry_lock:
            _contexts.clear()
        _thread_local.dataset_context = None
        # Also reset the empty context's state
        _empty_dataset_context.__init__("")  # type: ignore[misc]
