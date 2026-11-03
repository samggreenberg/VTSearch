"""The bottom of the state context system: the two locks, the errors, the frozen containers.

Imports none of its siblings, so every other module in the context system
(see :mod:`vtscore.state.core`) can import it without a cycle.  The locks
are module singletons: import them, never re-create them.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from typing import Any


# Reentrant lock protecting all mutable state.
# RLock is used because some public functions call other public functions
# (e.g. apply_label_with_click_time -> assign_click_time).
_state_lock = threading.RLock()

# Lock protecting *only* the two context-store dicts (``_contexts`` in
# :mod:`~vtscore.state.dataset_registry`, ``_detector_contexts`` in
# :mod:`~vtscore.state.detector_registry`) -- which ids are loaded, not what the contexts
# hold.
#
# It exists so :func:`get_context` / :func:`get_detector_context` can answer
# "is this id loaded?" without touching ``_state_lock``, which a vote, a
# training pass or a dataset load holds for seconds at a time. Every request
# goes through that lookup -- the app's ``_set_request_context`` hook resolves
# the SPA's ``X-Dataset-Id`` / ``X-Detector-Id`` headers on the way in -- so
# taking ``_state_lock`` there blocked even the routes marked
# ``@state_sync_exempt`` (the jobs/active spinner poll, the SSE reconnect)
# for exactly as long as the blocker ran, which is the opposite of what the
# exemption promises (#3869).
#
# **Ordering: always innermost.** Code may take this lock while holding
# ``_state_lock``; never the reverse. Nothing calls out to arbitrary code
# while holding it -- every critical section guarded by it is a dict
# get/set/pop/copy and nothing more -- so a reader waits at most the length
# of one dict operation, whatever else is running.
_context_registry_lock = threading.Lock()


class DatasetNotLoadedError(LookupError):
    """The request explicitly named a dataset that is not loaded in memory.

    Raised by the request-scoped dataset resolver (and propagated through
    :func:`get_active_context`) when an ``X-Dataset-Id`` header (or
    ``?dataset_id=`` query param) was sent but no matching
    :class:`DatasetContext` is registered. Silent fallback to an empty
    context produced stale results that the client could not detect;
    see logical-bug-audit H16.
    """

    def __init__(self, dataset_id: str) -> None:
        super().__init__(f"dataset {dataset_id!r} is not loaded")
        self.dataset_id = dataset_id


class DetectorNotLoadedError(LookupError):
    """The request explicitly named a detector that is not loaded in memory.

    Detector counterpart of :class:`DatasetNotLoadedError`. See
    logical-bug-audit H16 / H34.
    """

    def __init__(self, detector_id: str) -> None:
        super().__init__(f"detector {detector_id!r} is not loaded")
        self.detector_id = detector_id


# ---------------------------------------------------------------------------
# Request-missing sentinel: frozen empty context returned when a Flask
# request didn't identify a dataset/detector (missing header or unloaded id).
# Reads see an empty context (so non-mutating endpoints continue working);
# any mutation raises ``RequestMissingContextError`` immediately so the
# silent-mistarget failure modes flagged by H13/H16 fail loudly instead.
# ---------------------------------------------------------------------------


class RequestMissingContextError(RuntimeError):
    """Raised when code tries to mutate the request-missing context sentinel.

    The sentinel is what :func:`get_active_context` /
    :func:`get_active_detector_context` return inside a Flask request when
    the client didn't identify a dataset/detector; either the
    ``X-Dataset-Id`` / ``X-Detector-Id`` header was missing, or it named an
    unloaded id.  Reads against the sentinel see an empty context (so
    listing/dashboard endpoints keep working); writes hit this exception so
    votes / labels / pile additions cannot silently land on the wrong
    target.
    """


def _frozen_mutation_error(kind: str) -> RequestMissingContextError:
    return RequestMissingContextError(
        f"Refusing to mutate the request-missing {kind} context. "
        f"This Flask request did not identify a {kind} (missing "
        f"X-{kind.capitalize()}-Id header / query param, or it named an "
        f"unloaded id). Mutation endpoints must identify the {kind} "
        f"explicitly."
    )


class _FrozenDict(dict):  # type: ignore[type-arg]
    """A ``dict`` that allows reads but raises on every mutation."""

    __slots__ = ("_kind",)

    def __init__(self, kind: str) -> None:
        super().__init__()
        # Bypass our own __setattr__ (dict subclasses don't get one by default,
        # but be explicit so adding one later doesn't break this).
        object.__setattr__(self, "_kind", kind)

    def __setitem__(self, key: Any, value: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def __delitem__(self, key: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def update(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def setdefault(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def pop(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def popitem(self) -> Any:
        raise _frozen_mutation_error(self._kind)

    def clear(self) -> None:
        raise _frozen_mutation_error(self._kind)


class _FrozenList(list):  # type: ignore[type-arg]
    """A ``list`` that allows reads but raises on every mutation."""

    __slots__ = ("_kind",)

    def __init__(self, kind: str) -> None:
        super().__init__()
        object.__setattr__(self, "_kind", kind)

    def append(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def extend(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def insert(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def remove(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def pop(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def clear(self) -> None:
        raise _frozen_mutation_error(self._kind)

    def __setitem__(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def __delitem__(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def __iadd__(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def __imul__(self, *a: Any, **k: Any) -> Any:
        raise _frozen_mutation_error(self._kind)

    def sort(self, *a: Any, **k: Any) -> None:
        raise _frozen_mutation_error(self._kind)

    def reverse(self) -> None:
        raise _frozen_mutation_error(self._kind)


def _iter_slots(cls: type) -> Iterator[str]:
    """Yield every ``__slots__`` name declared anywhere in *cls*'s MRO, once."""
    seen: set[str] = set()
    for klass in cls.__mro__:
        for name in getattr(klass, "__slots__", ()):
            if name not in seen:
                seen.add(name)
                yield name


def _freeze_into(sentinel: Any, template: Any, kind: str) -> None:
    """Copy every slot of *template* onto *sentinel*, freezing its containers.

    The request-missing sentinels are documented to read as *empty*
    contexts, so they must carry the full slot set of the class they
    subclass - a subclass with ``__slots__ = ()`` and no ``__getattr__``
    turns any missed slot into an ``AttributeError`` (a 500 on the exact
    dropped-header path the sentinel exists to serve).  Hand-maintaining
    that list drifted twice; instead we build a fresh real context and copy
    it across, so a newly added slot can never be missed (issue #2933).

    Every ``dict`` / ``list`` value is replaced by a :class:`_FrozenDict` /
    :class:`_FrozenList` so container mutations raise
    :class:`RequestMissingContextError` instead of silently landing on the
    sentinel.  Scalar / ``None`` slots are copied verbatim; the whole
    sentinel refuses attribute assignment anyway (``__setattr__``), so a
    write to one of those raises too.
    """
    for name in _iter_slots(type(template)):
        try:
            value = getattr(template, name)
        except AttributeError:
            # A slot the class declares but never initialises: leave it
            # unset here too, so the sentinel reads exactly like a real
            # freshly-constructed context.
            continue
        if isinstance(value, dict):
            value = _FrozenDict(kind)
        elif isinstance(value, list):
            value = _FrozenList(kind)
        object.__setattr__(sentinel, name, value)
