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
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


# Reentrant lock protecting all mutable state.
# RLock is used because some public functions call other public functions
# (e.g. clear_all -> clear_medias + clear_votes).
_state_lock = threading.RLock()


# ---------------------------------------------------------------------------
# Pluggable per-request context resolvers
# ---------------------------------------------------------------------------
# In the Flask app, the ``before_request`` hook resolves an
# ``X-Dataset-Id`` / ``X-Detector-Id`` header to the matching context and
# stashes it on ``g``.  The proxy objects then need to read it back.
#
# To keep this module Flask-free (so it can move into ``vtscore`` later —
# see ``docs/plans/extract-library.md``), the read side is exposed as a
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


# ---------------------------------------------------------------------------
# DatasetContext — bundles all per-dataset mutable state
# ---------------------------------------------------------------------------


class DatasetContext:
    """All mutable state that belongs to a single loaded dataset.

    Vote-related state (``good_votes``, ``bad_votes``, ``label_history``, etc.)
    lives in :class:`DetectorContext`, not here.  ``DatasetContext`` holds only
    dataset-intrinsic state: the media items, diversity tree, and display name.
    """

    __slots__ = (
        "dataset_id",
        "medias",
        "diversity_tree",
        "dataset_display_name",
        # Cached contiguous (N, D) float32 embedding matrix and the sorted
        # media-id list it corresponds to.  Built lazily on first access by
        # ``vtsearch.embedding.matrix.get_embedding_matrix`` and reused
        # across cosine sort, MLP scoring, and diversity-tree construction so
        # we don't rebuild a 10k-row matrix per call.
        "_emb_matrix_ids",
        "_emb_matrix",
    )

    def __init__(self, dataset_id: str = "") -> None:
        self.dataset_id: str = dataset_id
        self.medias: dict[int, dict[str, Any]] = {}
        self.diversity_tree: Any = None  # DiversityTree | None
        self.dataset_display_name: str | None = None
        self._emb_matrix_ids: list[int] | None = None
        self._emb_matrix: Any = None  # np.ndarray | None


class DetectorContext:
    """All mutable state that belongs to a single loaded detector.

    Bundles per-detector vote state, training artifacts, and cached in-memory
    data (MLP, threshold, training media with embeddings).  Multiple detectors
    can be loaded simultaneously; one is "active" (feeding the labeling UI).
    """

    __slots__ = (
        "detector_id",
        "name",
        "media_type",
        "embedder",
        # Vote state
        "good_votes",
        "bad_votes",
        "label_history",
        "vote_click_times",
        "vote_region_boxes",
        "click_counter",
        # Training artifacts
        "last_learned_scores",
        "textsort_suggestions",
        "find_initial_labels",
        "inclusion",
        # Cached in-memory data (never exported)
        "training_medias",  # voted media items with embeddings
        "label_embeddings",  # str → np.ndarray, keyed by stable_element_id
        "model",  # nn.Sequential | None — current trained MLP
        "threshold",  # decision threshold
        # Cross-dataset training-corpus counts (from on-disk labelset).  These
        # are independent of ``good_votes``/``bad_votes``, which only count
        # labels for media in the *currently loaded* dataset.  They drive the
        # frontend's "Sort by Learned" gating so a detector trained on dataset
        # A stays trainable when the user switches to dataset B.
        "labelset_good_count",
        "labelset_bad_count",
        # Dataset ID for which the cid-keyed vote state above is valid.
        # Media IDs are only meaningful within a single dataset, so when the
        # active dataset changes for a loaded detector we must clear the cid
        # dicts and re-derive them from the on-disk labelset against the new
        # dataset's medias.  See ``ensure_votes_match_active_dataset``.
        "votes_dataset_id",
        # Cached parsed labelset + mtime of the on-disk detector JSON the cache
        # was derived from.  Lets ``ensure_votes_match_active_dataset`` skip the
        # rehydrate (read+parse) when neither the active dataset nor the file
        # has changed, and lets ``learned_sort`` reuse the parsed labelset
        # instead of re-reading the JSON from disk on every click.
        "cached_labelset",  # LabelSet | None
        "cached_labelset_mtime",  # float
        "cached_labelset_media_type",  # str
        # Sync source
        "labelset_source",  # dict | None — {"source_name": "...", "field_values": {...}}
        # Calibration threshold cache.  Holds ``(key, threshold)`` where *key*
        # is a deterministic fingerprint of the inputs to
        # :func:`calculate_cross_calibration_threshold` (training vectors,
        # labels, inclusion, calibrate_count, calibration_fraction,
        # hidden_dim).  Reusing the cached threshold is safe iff *key*
        # matches — calibration is a deterministic function of these inputs
        # (seeded RNG), so a hit is a pure memoization, not a stale carry-
        # over from a previously trained model.
        "calibration_cache",  # tuple[Any, float] | None
    )

    def __init__(self, detector_id: str = "", *, name: str = "", media_type: str = "", embedder: str = "") -> None:
        self.detector_id: str = detector_id
        self.name: str = name
        self.media_type: str = media_type
        self.embedder: str = embedder
        # Vote state
        self.good_votes: dict[int, None] = {}
        self.bad_votes: dict[int, None] = {}
        self.label_history: list[tuple[int, str, float]] = []
        self.vote_click_times: dict[int, int] = {}
        # Per-good-vote region boxes (normalised x0, y0, x1, y1).  Only set when
        # the user drew a region as part of a yes-vote; absent for image-level
        # yes-votes and for every no-vote.  Patch-embedder v2.
        self.vote_region_boxes: dict[int, tuple[float, float, float, float]] = {}
        self.click_counter: int = 0
        # Training artifacts
        self.last_learned_scores: dict[int, float] = {}
        self.textsort_suggestions: list[str] = []
        self.find_initial_labels: dict[int, str] = {}
        self.inclusion: int | None = None
        # Cached in-memory data (never exported)
        self.training_medias: dict[int, dict[str, Any]] = {}
        # Embeddings for every saved labelset element, keyed by
        # stable_element_id.  Populated at detector load (resolve_file +
        # embed_file) and topped up when new votes come in.  Lets MLP
        # training and learned-sort use *all* saved labels — including
        # those whose underlying media isn't part of the active dataset.
        self.label_embeddings: dict[str, Any] = {}
        self.model: Any = None  # nn.Sequential | None
        self.threshold: float = 0.5
        self.labelset_good_count: int = 0
        self.labelset_bad_count: int = 0
        self.votes_dataset_id: str = ""
        self.cached_labelset: Any = None  # LabelSet | None
        self.cached_labelset_mtime: float = 0.0
        self.cached_labelset_media_type: str = ""
        # Sync source
        self.labelset_source: dict[str, Any] | None = None
        self.calibration_cache: tuple[Any, float] | None = None


# ---------------------------------------------------------------------------
# Dataset context store and thread-local fallback
# ---------------------------------------------------------------------------

# Maps dataset_id -> DatasetContext for every in-memory dataset.
_contexts: dict[str, DatasetContext] = {}

# Thread-local storage for the fallback dataset/detector context.
# Used by background threads and tests that operate outside a Flask
# request context.  Each thread sets its own value; no global "active"
# pointer exists.
_thread_local = threading.local()

# Fallback context used when no dataset is set.  Proxies delegate to
# this so that code accessing ``medias`` when nothing is loaded sees empty
# containers rather than crashing.
_empty_dataset_context = DatasetContext("")


def get_active_context() -> DatasetContext:
    """Return the ``DatasetContext`` for the current execution context.

    Resolution order:
    1. Request-scoped context (set by ``before_request`` from ``X-Dataset-Id`` header)
    2. Thread-local context (set by ``set_thread_dataset_context`` — for
       background threads and tests)
    3. Empty fallback context
    """
    # 1. Per-request override (Flask shim or whatever the host app registered)
    req_ctx = _dataset_context_resolver()
    if req_ctx is not None:
        return req_ctx
    # 2. Thread-local fallback
    ctx = getattr(_thread_local, "dataset_context", None)
    if ctx is not None:
        return ctx
    return _empty_dataset_context


def set_thread_dataset_context(ctx: DatasetContext | None) -> None:
    """Set the thread-local dataset context for the current thread.

    Called by test fixtures and background threads to direct proxy
    resolution without global state.
    """
    _thread_local.dataset_context = ctx


def get_thread_dataset_context() -> DatasetContext | None:
    """Return the thread-local dataset context, or ``None``."""
    return getattr(_thread_local, "dataset_context", None)


def register_context(ctx: DatasetContext) -> None:
    """Add *ctx* to the context store, keyed by its ``dataset_id``."""
    with _state_lock:
        _contexts[ctx.dataset_id] = ctx


def unregister_context(dataset_id: str) -> DatasetContext | None:
    """Remove and return the context for *dataset_id*, or ``None``."""
    with _state_lock:
        ctx = _contexts.pop(dataset_id, None)
        # Clear thread-local if it was pointing to the removed context.
        tl_ctx = getattr(_thread_local, "dataset_context", None)
        if tl_ctx is not None and tl_ctx.dataset_id == dataset_id:
            _thread_local.dataset_context = None
        return ctx


def get_context(dataset_id: str) -> DatasetContext | None:
    """Return the context for *dataset_id*, or ``None`` if not loaded."""
    with _state_lock:
        return _contexts.get(dataset_id)


def list_loaded_dataset_ids() -> list[str]:
    """Return all dataset IDs that have an in-memory context."""
    with _state_lock:
        return list(_contexts.keys())


def clear_all_contexts() -> None:
    """Remove all dataset contexts and clear the thread-local.  For tests."""
    with _state_lock:
        _contexts.clear()
        _thread_local.dataset_context = None
        # Also reset the empty context's state
        _empty_dataset_context.__init__("")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Detector context store and thread-local fallback
# ---------------------------------------------------------------------------

# Maps detector_id -> DetectorContext for every in-memory detector.
_detector_contexts: dict[str, DetectorContext] = {}

# Fallback context used when no detector is set.  Vote proxies delegate
# to this so that code accessing ``good_votes``, ``bad_votes``, etc. when
# no detector is loaded sees empty containers rather than crashing.
_empty_detector_context = DetectorContext("")


def get_active_detector_context() -> DetectorContext:
    """Return the ``DetectorContext`` for the current execution context.

    Resolution order:
    1. Forced override (``override_detector_context`` context manager)
    2. Request-scoped context (set by ``before_request`` from ``X-Detector-Id`` header)
    3. Thread-local context (set by ``set_thread_detector_context``)
    4. Empty fallback context
    """
    # 1. Forced override (set by override_detector_context context manager)
    forced = getattr(_thread_local, "forced_detector_context", None)
    if forced is not None:
        return forced
    # 2. Per-request override (Flask shim or whatever the host app registered)
    req_ctx = _detector_context_resolver()
    if req_ctx is not None:
        return req_ctx
    # 3. Thread-local fallback
    ctx = getattr(_thread_local, "detector_context", None)
    if ctx is not None:
        return ctx
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
    """Set the thread-local detector context for the current thread."""
    _thread_local.detector_context = ctx


def get_thread_detector_context() -> DetectorContext | None:
    """Return the thread-local detector context, or ``None``."""
    return getattr(_thread_local, "detector_context", None)


def register_detector_context(ctx: DetectorContext) -> None:
    """Add *ctx* to the detector context store, keyed by its ``detector_id``.

    Also clears the module-level progress cache so that stale training
    indicators from a previously-active detector are not reused.
    """
    from vtsearch.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        _detector_contexts[ctx.detector_id] = ctx
    # clear_progress_cache acquires its own lock; call outside _state_lock
    # to avoid lock-ordering concerns.
    clear_progress_cache()


def unregister_detector_context(detector_id: str) -> DetectorContext | None:
    """Remove and return the detector context for *detector_id*, or ``None``.

    Also clears the progress cache so stale cached steps from the removed
    detector are not used by a subsequent detector.
    """
    from vtsearch.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        ctx = _detector_contexts.pop(detector_id, None)
        tl_ctx = getattr(_thread_local, "detector_context", None)
        if tl_ctx is not None and tl_ctx.detector_id == detector_id:
            _thread_local.detector_context = None
    clear_progress_cache()
    return ctx


def get_detector_context(detector_id: str) -> DetectorContext | None:
    """Return the detector context for *detector_id*, or ``None`` if not loaded."""
    with _state_lock:
        return _detector_contexts.get(detector_id)


def list_loaded_detector_ids() -> list[str]:
    """Return all detector IDs that have an in-memory context."""
    with _state_lock:
        return list(_detector_contexts.keys())


def clear_all_detector_contexts() -> None:
    """Remove all detector contexts and clear the thread-local.  For tests."""
    with _state_lock:
        _detector_contexts.clear()
        _thread_local.detector_context = None
        _empty_detector_context.__init__("")  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Proxy classes — delegate to the active context's underlying container
# ---------------------------------------------------------------------------


class _ProxyDict(dict):
    """Dict-like proxy that forwards all operations to a target dict.

    The *target_attr* names the attribute on the active context that holds
    the real dict.  The *context_fn* is called to find that context.
    Every method fetches the target lazily so switching contexts is instant.

    Inherits from ``dict`` so that ``isinstance(medias, dict)`` is ``True``
    and existing code that type-checks works.
    """

    def __init__(self, target_attr: str, context_fn=None) -> None:
        # Do NOT call super().__init__() with data — we are a proxy, not a
        # real container.  The empty super().__init__() satisfies the dict
        # constructor without storing anything in our own hash table.
        super().__init__()
        object.__setattr__(self, "_target_attr", target_attr)
        object.__setattr__(self, "_context_fn", context_fn or get_active_context)

    def _target(self) -> dict:
        ctx_fn = object.__getattribute__(self, "_context_fn")
        return getattr(ctx_fn(), object.__getattribute__(self, "_target_attr"))

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
        # Snapshot keys so `for k in proxy` is safe against concurrent
        # mutation of the underlying dict by another thread.
        return iter(list(self._target()))

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
        # Snapshot to avoid RuntimeError under concurrent mutation.
        return reversed(list(self._target()))


class _ProxyList(list):
    """List-like proxy forwarding to the active context's list attribute."""

    def __init__(self, target_attr: str, context_fn=None) -> None:
        super().__init__()
        object.__setattr__(self, "_target_attr", target_attr)
        object.__setattr__(self, "_context_fn", context_fn or get_active_context)

    def _target(self) -> list:
        ctx_fn = object.__getattribute__(self, "_context_fn")
        return getattr(ctx_fn(), object.__getattribute__(self, "_target_attr"))

    def __getitem__(self, index):
        return self._target().__getitem__(index)

    def __setitem__(self, index, value):
        self._target().__setitem__(index, value)

    def __delitem__(self, index):
        self._target().__delitem__(index)

    def __contains__(self, item):
        return self._target().__contains__(item)

    def __iter__(self):
        # Snapshot elements so iteration is safe against concurrent mutation.
        return iter(list(self._target()))

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
# to the appropriate active context.
#
# ``medias`` delegates to the active **DatasetContext** (dataset-intrinsic).
# All vote/label proxies delegate to the active **DetectorContext**.
# ---------------------------------------------------------------------------

# Clips storage: id -> {id, type, duration, file_size, embedding, media_bytes, media_string, ...}
medias: dict[int, dict[str, Any]] = _ProxyDict("medias")  # type: ignore[assignment]

# Voting storage (OrderedDict behavior via dict in Python 3.7+)
good_votes: dict[int, None] = _ProxyDict("good_votes", get_active_detector_context)  # type: ignore[assignment]
bad_votes: dict[int, None] = _ProxyDict("bad_votes", get_active_detector_context)  # type: ignore[assignment]

# Combined label history: [(media_id, label, timestamp), ...]
label_history: list[tuple[int, str, float]] = _ProxyList("label_history", get_active_detector_context)  # type: ignore[assignment]

# Click-time tracking: media_id -> click order (1-indexed).
vote_click_times: dict[int, int] = _ProxyDict("vote_click_times", get_active_detector_context)  # type: ignore[assignment]

# Per-good-vote region boxes: media_id -> (x0, y0, x1, y1) in normalised image
# coords.  Only populated when the user drew a region as part of a yes-vote;
# absent for image-level yes-votes and for every no-vote.  Patch-embedder v2.
vote_region_boxes: dict[int, tuple[float, float, float, float]] = _ProxyDict(  # type: ignore[assignment]
    "vote_region_boxes",
    get_active_detector_context,
)

# Last learned-sort scores: media_id -> score (float in [0, 1]).
last_learned_scores: dict[int, float] = _ProxyDict("last_learned_scores", get_active_detector_context)  # type: ignore[assignment]

# Text-sort suggestions: text queries that received a Good vote, most recent last.
textsort_suggestions: list[str] = _ProxyList("textsort_suggestions", get_active_detector_context)  # type: ignore[assignment]

# Find-mode initial labels: media_id -> label assigned by the detector in find-label.
_find_initial_labels: dict[int, str] = _ProxyDict("find_initial_labels", get_active_detector_context)  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Scalar state accessors
# ---------------------------------------------------------------------------
# Dataset-intrinsic scalars (diversity_tree, dataset_display_name) delegate
# to the active DatasetContext.  Vote-related scalars (click_counter,
# inclusion) delegate to the active DetectorContext.
# ---------------------------------------------------------------------------


def _get_click_counter() -> int:
    return get_active_detector_context().click_counter


def _set_click_counter(value: int) -> None:
    get_active_detector_context().click_counter = value


def _get_diversity_tree() -> Any:
    return get_active_context().diversity_tree


def _set_diversity_tree(value: Any) -> None:
    get_active_context().diversity_tree = value


def _get_dataset_display_name() -> str | None:
    return get_active_context().dataset_display_name


def _set_dataset_display_name(value: str | None) -> None:
    get_active_context().dataset_display_name = value


def _get_inclusion() -> int | None:
    return get_active_detector_context().inclusion


def _set_inclusion(value: int | None) -> None:
    get_active_detector_context().inclusion = value


# ---------------------------------------------------------------------------
# Context managers for explicit, scoped context switching
# ---------------------------------------------------------------------------


class with_dataset_context:
    """Context manager for temporarily switching the active dataset.

    Saves the current active dataset ID on entry, switches to the
    requested *dataset_id*, and restores the original on exit —
    even if an exception occurs.

    Usage::

        with with_dataset_context("my_dataset"):
            # code here sees my_dataset's medias, diversity tree, etc.
            print(len(medias))
        # original dataset is restored here

    .. warning::
        This is NOT thread-safe.  Only use from a single thread or
        protect with ``_state_lock`` externally.
    """

    __slots__ = ("_target_id", "_previous_ctx")

    def __init__(self, dataset_id: str) -> None:
        self._target_id = dataset_id
        self._previous_ctx: DatasetContext | None = None

    def __enter__(self) -> DatasetContext:
        self._previous_ctx = get_thread_dataset_context()
        ctx = get_context(self._target_id)
        if ctx is None:
            raise ValueError(f"No dataset context registered for {self._target_id!r}")
        set_thread_dataset_context(ctx)
        return ctx

    def __exit__(self, *exc_info: object) -> None:
        set_thread_dataset_context(self._previous_ctx)


class with_detector_context:
    """Context manager for temporarily switching the active detector.

    Saves the current active detector ID on entry, switches to the
    requested *detector_id*, and restores the original on exit.

    Usage::

        with with_detector_context("my_detector"):
            # code here sees my_detector's votes, scores, etc.
            print(len(good_votes))
        # original detector is restored here

    .. warning::
        This is NOT thread-safe.  Only use from a single thread or
        protect with ``_state_lock`` externally.
    """

    __slots__ = ("_target_id", "_previous_ctx")

    def __init__(self, detector_id: str) -> None:
        self._target_id = detector_id
        self._previous_ctx: DetectorContext | None = None

    def __enter__(self) -> DetectorContext:
        self._previous_ctx = get_thread_detector_context()
        ctx = get_detector_context(self._target_id)
        if ctx is None:
            raise ValueError(f"No detector context registered for {self._target_id!r}")
        set_thread_detector_context(ctx)
        return ctx

    def __exit__(self, *exc_info: object) -> None:
        set_thread_detector_context(self._previous_ctx)
