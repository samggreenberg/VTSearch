"""Core state variables and lock shared by all state submodules.

This module defines the mutable global state and the reentrant lock that
protects it.  All other ``state_*.py`` submodules import variables from
here rather than defining their own.

Multi-dataset support
---------------------
Per-dataset state is bundled in ``DatasetContext`` objects.  A context store
(``_contexts``) maps dataset IDs to their contexts, and ``_active_dataset_id``
tracks which one the UI is currently interacting with.

The module-level names ``medias``, ``good_votes``, ``bad_votes``, etc. are
**proxy objects** that transparently delegate to the active context's
underlying data structure.  This means all existing code that imports
these names continues to work without modification — reads and writes go
through to the active dataset's state.

When no context is active the proxies behave as empty containers (reads
return nothing, writes are silently discarded or raise where appropriate).
"""

from __future__ import annotations

import threading
from typing import Any


# Reentrant lock protecting all mutable state.
# RLock is used because some public functions call other public functions
# (e.g. clear_all -> clear_medias + clear_votes).
_state_lock = threading.RLock()


# ---------------------------------------------------------------------------
# DatasetContext — bundles all per-dataset mutable state
# ---------------------------------------------------------------------------


class DatasetContext:
    """All mutable state that belongs to a single loaded dataset."""

    __slots__ = (
        "dataset_id",
        "medias",
        "good_votes",
        "bad_votes",
        "label_history",
        "vote_click_times",
        "click_counter",
        "last_learned_scores",
        "textsort_suggestions",
        "diversity_tree",
        "find_initial_labels",
        "inclusion",
        "dataset_display_name",
    )

    def __init__(self, dataset_id: str = "") -> None:
        self.dataset_id: str = dataset_id
        self.medias: dict[int, dict[str, Any]] = {}
        self.good_votes: dict[int, None] = {}
        self.bad_votes: dict[int, None] = {}
        self.label_history: list[tuple[int, str, float]] = []
        self.vote_click_times: dict[int, int] = {}
        self.click_counter: int = 0
        self.last_learned_scores: dict[int, float] = {}
        self.textsort_suggestions: list[str] = []
        self.diversity_tree: Any = None  # DiversityTree | None
        self.find_initial_labels: dict[int, str] = {}
        self.inclusion: int | None = None
        self.dataset_display_name: str | None = None


# ---------------------------------------------------------------------------
# Context store and active-dataset pointer
# ---------------------------------------------------------------------------

# Maps dataset_id -> DatasetContext for every in-memory dataset.
_contexts: dict[str, DatasetContext] = {}

# The dataset_id of the context whose state the UI is currently using,
# or ``None`` when nothing is active.
_active_dataset_id: str | None = None

# Fallback context used when no dataset is active.  Proxies delegate to
# this so that code accessing ``medias``, ``good_votes``, etc. when nothing
# is loaded sees empty containers rather than crashing.
_empty_context = DatasetContext("")


def get_active_context() -> DatasetContext:
    """Return the active ``DatasetContext``, or the empty fallback."""
    if _active_dataset_id is not None:
        ctx = _contexts.get(_active_dataset_id)
        if ctx is not None:
            return ctx
    return _empty_context


def get_active_dataset_id() -> str | None:
    """Return the dataset_id of the active context, or ``None``."""
    return _active_dataset_id


def set_active_dataset_id(dataset_id: str | None) -> None:
    """Switch the active context to *dataset_id* (must already be in _contexts, or None)."""
    global _active_dataset_id
    _active_dataset_id = dataset_id


def register_context(ctx: DatasetContext) -> None:
    """Add *ctx* to the context store, keyed by its ``dataset_id``."""
    _contexts[ctx.dataset_id] = ctx


def unregister_context(dataset_id: str) -> DatasetContext | None:
    """Remove and return the context for *dataset_id*, or ``None``."""
    global _active_dataset_id
    ctx = _contexts.pop(dataset_id, None)
    if _active_dataset_id == dataset_id:
        _active_dataset_id = None
    return ctx


def get_context(dataset_id: str) -> DatasetContext | None:
    """Return the context for *dataset_id*, or ``None`` if not loaded."""
    return _contexts.get(dataset_id)


def list_loaded_dataset_ids() -> list[str]:
    """Return all dataset IDs that have an in-memory context."""
    return list(_contexts.keys())


def clear_all_contexts() -> None:
    """Remove all contexts and reset the active pointer.  For tests."""
    global _active_dataset_id
    _contexts.clear()
    _active_dataset_id = None
    # Also reset the empty context's state
    _empty_context.__init__("")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Proxy classes — delegate to the active context's underlying container
# ---------------------------------------------------------------------------


class _ProxyDict(dict):
    """Dict-like proxy that forwards all operations to a target dict.

    The *target_attr* names the attribute on ``DatasetContext`` that holds
    the real dict.  Every method fetches the target lazily via
    ``get_active_context()`` so that switching the active context is instant.

    Inherits from ``dict`` so that ``isinstance(medias, dict)`` is ``True``
    and existing code that type-checks works.
    """

    def __init__(self, target_attr: str) -> None:
        # Do NOT call super().__init__() with data — we are a proxy, not a
        # real container.  The empty super().__init__() satisfies the dict
        # constructor without storing anything in our own hash table.
        super().__init__()
        object.__setattr__(self, "_target_attr", target_attr)

    def _target(self) -> dict:
        return getattr(get_active_context(), object.__getattribute__(self, "_target_attr"))

    # -- Core dict methods, all forwarded -----------------------------------

    def __getitem__(self, key):
        return self._target().__getitem__(key)

    def __setitem__(self, key, value):
        self._target().__setitem__(key, value)

    def __delitem__(self, key):
        self._target().__delitem__(key)

    def __contains__(self, key):
        return self._target().__contains__(key)

    def __iter__(self):
        return self._target().__iter__()

    def __len__(self):
        return self._target().__len__()

    def __repr__(self):
        return f"_ProxyDict({object.__getattribute__(self, '_target_attr')!r}, {self._target()!r})"

    def __eq__(self, other):
        return self._target().__eq__(other)

    def __ne__(self, other):
        return self._target().__ne__(other)

    def __bool__(self):
        return bool(self._target())

    def get(self, key, default=None):
        return self._target().get(key, default)

    def keys(self):
        return self._target().keys()

    def values(self):
        return self._target().values()

    def items(self):
        return self._target().items()

    def pop(self, *args):
        return self._target().pop(*args)

    def setdefault(self, key, default=None):
        return self._target().setdefault(key, default)

    def update(self, *args, **kwargs):
        return self._target().update(*args, **kwargs)

    def clear(self):
        return self._target().clear()

    def copy(self):
        return self._target().copy()

    def __or__(self, other):
        return self._target().__or__(other)

    def __ior__(self, other):
        target = self._target()
        target.__ior__(other)
        return self

    def __reversed__(self):
        return self._target().__reversed__()


class _ProxyList(list):
    """List-like proxy forwarding to the active context's list attribute."""

    def __init__(self, target_attr: str) -> None:
        super().__init__()
        object.__setattr__(self, "_target_attr", target_attr)

    def _target(self) -> list:
        return getattr(get_active_context(), object.__getattribute__(self, "_target_attr"))

    def __getitem__(self, index):
        return self._target().__getitem__(index)

    def __setitem__(self, index, value):
        self._target().__setitem__(index, value)

    def __delitem__(self, index):
        self._target().__delitem__(index)

    def __contains__(self, item):
        return self._target().__contains__(item)

    def __iter__(self):
        return self._target().__iter__()

    def __len__(self):
        return self._target().__len__()

    def __repr__(self):
        return f"_ProxyList({object.__getattribute__(self, '_target_attr')!r}, {self._target()!r})"

    def __eq__(self, other):
        return self._target().__eq__(other)

    def __bool__(self):
        return bool(self._target())

    def __add__(self, other):
        return self._target().__add__(other)

    def __iadd__(self, other):
        target = self._target()
        target.__iadd__(other)
        return self

    def append(self, item):
        return self._target().append(item)

    def extend(self, items):
        return self._target().extend(items)

    def insert(self, index, item):
        return self._target().insert(index, item)

    def remove(self, item):
        return self._target().remove(item)

    def pop(self, *args):
        return self._target().pop(*args)

    def clear(self):
        return self._target().clear()

    def copy(self):
        return self._target().copy()

    def index(self, value, *args):
        return self._target().index(value, *args)

    def count(self, value):
        return self._target().count(value)

    def sort(self, **kwargs):
        return self._target().sort(**kwargs)

    def reverse(self):
        return self._target().reverse()


# ---------------------------------------------------------------------------
# Module-level proxy instances — these ARE the public names that all other
# modules import.  They look and behave like plain dicts/lists but delegate
# to whichever DatasetContext is currently active.
# ---------------------------------------------------------------------------

# Clips storage: id -> {id, type, duration, file_size, embedding, media_bytes, media_string, ...}
medias: dict[int, dict[str, Any]] = _ProxyDict("medias")  # type: ignore[assignment]

# Voting storage (OrderedDict behavior via dict in Python 3.7+)
good_votes: dict[int, None] = _ProxyDict("good_votes")  # type: ignore[assignment]
bad_votes: dict[int, None] = _ProxyDict("bad_votes")  # type: ignore[assignment]

# Combined label history: [(media_id, label, timestamp), ...]
label_history: list[tuple[int, str, float]] = _ProxyList("label_history")  # type: ignore[assignment]

# Click-time tracking: media_id -> click order (1-indexed).
vote_click_times: dict[int, int] = _ProxyDict("vote_click_times")  # type: ignore[assignment]

# Last learned-sort scores: media_id -> score (float in [0, 1]).
last_learned_scores: dict[int, float] = _ProxyDict("last_learned_scores")  # type: ignore[assignment]

# Text-sort suggestions: text queries that received a Good vote, most recent last.
textsort_suggestions: list[str] = _ProxyList("textsort_suggestions")  # type: ignore[assignment]

# Find-mode initial labels: media_id -> label assigned by the detector in find-label.
_find_initial_labels: dict[int, str] = _ProxyDict("find_initial_labels")  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Scalar per-dataset state — accessed via properties on the active context.
# These use module-level *properties* via a descriptor protocol so that
# ``import state_core; state_core._click_counter`` still works.
# For simplicity, we keep them as module-level getters/setters and update
# the active context directly.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scalar per-dataset state accessors
# ---------------------------------------------------------------------------
# _click_counter, inclusion, _dataset_display_name, _diversity_tree are
# scalar fields on DatasetContext.  Code in state submodules accesses them
# via ``_core._click_counter`` etc.  We keep backward-compatible module-
# level names that resolve to the active context via property-like access.
#
# For *writes*, submodules do ``_core._click_counter += 1``.  Python's
# module attribute protocol means ``_core._click_counter`` reads the
# module-level name.  To redirect these to the active context we provide
# explicit getter/setter helpers and update the call-sites.
# ---------------------------------------------------------------------------


def _get_click_counter() -> int:
    return get_active_context().click_counter


def _set_click_counter(value: int) -> None:
    get_active_context().click_counter = value


def _get_diversity_tree() -> Any:
    return get_active_context().diversity_tree


def _set_diversity_tree(value: Any) -> None:
    get_active_context().diversity_tree = value


def _get_dataset_display_name() -> str | None:
    return get_active_context().dataset_display_name


def _set_dataset_display_name(value: str | None) -> None:
    get_active_context().dataset_display_name = value


def _get_inclusion() -> int | None:
    return get_active_context().inclusion


def _set_inclusion(value: int | None) -> None:
    get_active_context().inclusion = value


# Legacy module-level names — these are still used by conftest.py's
# reset_state fixture which does ``_core._click_counter = 0``.
# We keep them for backward compat but they are NOT used by the
# proxy-aware submodules anymore (those call the getters/setters above).
_click_counter: int = 0
_dataset_display_name: str | None = None
_diversity_tree: Any = None
inclusion: int | None = None

# Autorun detectors/extractors/localizers are GLOBAL (not per-dataset).
autorun_detectors: dict[str, dict[str, Any]] = {}
autorun_extractors: dict[str, dict[str, Any]] = {}
autorun_localizers: dict[str, dict[str, Any]] = {}
