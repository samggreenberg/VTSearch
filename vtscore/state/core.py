"""Core state types and lock shared by all state submodules.

Defines the per-dataset and per-detector context classes
(:class:`DatasetContext`, :class:`DetectorContext`), the resolution
functions that find the "active" context for the current request /
thread, and the reentrant lock that protects all mutable state.

Multi-dataset support
---------------------
Per-dataset state is bundled in :class:`DatasetContext` objects.  Per-
detector state lives in :class:`DetectorContext`.  Context stores map
each ID to its context, and per-request / thread-local resolvers determine
which one library helpers operate on.

The app-side facade
-------------------
Module-level convenience names (``medias``, ``good_votes``, …) used to
live here as proxy objects, but they belong to the app layer — the
library never imports them.  They now live in
:mod:`vtsearch.shim.state_proxies` and are re-exported from
:mod:`vtsearch.state` so existing app-tier imports continue to work.
See Phase 3 of ``docs/plans/extract-library.md``.
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
        # ``vtscore.embedding.matrix.get_embedding_matrix`` and reused
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
    from vtscore.detectors.labeling_progress import clear_progress_cache

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
    from vtscore.detectors.labeling_progress import clear_progress_cache

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
# Scalar state accessors
# ---------------------------------------------------------------------------
# These thin helpers wrap "give me the X of whatever context is currently
# active" so callers that operate on the active context — but don't need
# a full DatasetContext / DetectorContext reference — can stay one-liners.
# Dataset-intrinsic scalars (diversity_tree, dataset_display_name) delegate
# to the active DatasetContext.  Detector-related scalars (click_counter,
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
