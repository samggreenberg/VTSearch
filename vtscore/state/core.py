"""Core state types and lock shared by all state submodules.

Defines the per-dataset and per-detector context classes
(:class:`DatasetContext`, :class:`DetectorContext`), the resolution
functions that find the "active" context for the current request /
thread, and the reentrant lock that protects all mutable state.

Where it lives
--------------
The context system is split into sibling modules, each importing only from
the ones listed before it:

* :mod:`~vtscore.state.primitives` - ``_state_lock``, the registry lock and
  its ordering rule, the not-loaded / request-missing errors, and the frozen
  containers the request-missing sentinels are built from.
* :mod:`~vtscore.state.resolution` - the host-installed request predicate and
  context resolvers (``register_*``), and the shared thread-local.
* :mod:`~vtscore.state.dataset_context` / :mod:`~vtscore.state.detector_context`
  - :class:`DatasetContext` (with :class:`MediasDict`) and :class:`DetectorContext`.
* :mod:`~vtscore.state.dataset_registry` / :mod:`~vtscore.state.detector_registry`
  - the loaded-context stores, the request-missing sentinels, and the
  active-context resolution chain.
* :mod:`~vtscore.state.detector_lines` - a detector's line re-derived from its
  cached estimators (re-cuts, balance state, the acquisition cut).

This module re-exports every public name from them, so ``vtscore.state.core``
stays the import path for all of it, and holds what sits on top of both
registries: :func:`is_request_missing_context`, the scalar accessors and the
``with_*`` scoped switchers.  The three resolution hooks are the one thing it
does not re-export: ``register_*`` rebinds them in
:mod:`~vtscore.state.resolution`, so a copy here would go stale.  Read or
patch them there.  Likewise, patch a function in the module whose caller
looks it up (``detector_lines.recut_detector_threshold`` for
:func:`recompute_detector_thresholds`), not here.

Multi-dataset support
---------------------
Per-dataset state is bundled in :class:`DatasetContext` objects.  Per-
detector state lives in :class:`DetectorContext`.  Context stores map
each ID to its context, and per-request / thread-local resolvers determine
which one library helpers operate on.

The app-side facade
-------------------
The module-level convenience names (``medias``, ``good_votes``, …) are
proxy objects that belong to the app layer; the library never imports
them.  They live in :mod:`vtsearch.state_proxies` and are re-exported
from :mod:`vtsearch.state`.  See seam 3 (*Global state*) of
``../docs/architecture.md``.
"""

from __future__ import annotations

from typing import Any

from vtscore.state.dataset_context import DatasetContext, MediasDict  # noqa: F401
from vtscore.state.dataset_registry import (  # noqa: F401
    _request_missing_dataset_context,
    clear_all_contexts,
    get_active_context,
    get_context,
    get_thread_dataset_context,
    is_request_missing_dataset_context,
    list_loaded_dataset_ids,
    register_context,
    rekey_dataset_context,
    set_thread_dataset_context,
    thread_dataset_context,
    unregister_context,
)
from vtscore.state.detector_context import DetectorContext
from vtscore.state.detector_lines import (  # noqa: F401
    detector_acquisition_threshold,
    detector_balance_positives,
    detector_balance_proposal,
    detector_balance_state,
    detector_check_prompt,
    detector_line_inclusion,
    detector_line_labels,
    detector_precision_floor,
    detector_walk_positives,
    human_voted_ids,
    recompute_detector_thresholds,
    recompute_detector_thresholds_for_inclusion,
    recut_detector_threshold,
    user_beta,
)
from vtscore.state.detector_registry import (  # noqa: F401
    _empty_detector_context,
    _request_missing_detector_context,
    clear_all_detector_contexts,
    get_active_detector_context,
    get_detector_context,
    get_thread_detector_context,
    invalidate_loaded_detector_models,
    is_request_missing_detector_context,
    list_loaded_detector_ids,
    loaded_detector_contexts,
    override_detector_context,
    register_detector_context,
    set_thread_detector_context,
    thread_detector_context,
    unregister_detector_context,
)
from vtscore.state.primitives import (  # noqa: F401
    DatasetNotLoadedError,
    DetectorNotLoadedError,
    RequestMissingContextError,
    _state_lock,
)
from vtscore.state.resolution import (  # noqa: F401
    register_dataset_context_resolver,
    register_detector_context_resolver,
    register_request_context_predicate,
)


def is_request_missing_context(ctx: Any) -> bool:
    """Return True iff *ctx* is either request-missing sentinel."""
    return ctx is _request_missing_dataset_context or ctx is _request_missing_detector_context


# ---------------------------------------------------------------------------
# Scalar state accessors
# ---------------------------------------------------------------------------
# These thin helpers wrap "give me the X of whatever context is currently
# active" so callers that operate on the active context but don't need
# a full DatasetContext / DetectorContext reference can stay one-liners.
# Dataset-intrinsic scalars (coverage_atlas, dataset_display_name) delegate
# to the active DatasetContext.  Detector-related scalars (click_counter,
# beta) delegate to the active DetectorContext.
# ---------------------------------------------------------------------------


def _get_click_counter() -> int:
    return get_active_detector_context().click_counter


def _set_click_counter(value: int) -> None:
    get_active_detector_context().click_counter = value


def _get_dataset_display_name() -> str | None:
    return get_active_context().dataset_display_name


def _set_dataset_display_name(value: str | None) -> None:
    get_active_context().dataset_display_name = value


def _get_beta(ctx: DetectorContext | None = None) -> tuple[bool, float | None]:
    """``(seeded, value)`` for *ctx*'s balance (#4413), the active detector's by default."""
    if ctx is None:
        ctx = get_active_detector_context()
    return ctx.beta_seeded, ctx.beta


def _set_beta(value: float | None, ctx: DetectorContext | None = None) -> None:
    if ctx is None:
        ctx = get_active_detector_context()
    # The balance is cached per-detector for fast reads; it persists on the
    # detector's JSON (``vtscore.detectors.balance.keep_beta``, #4665) and as
    # the user's last pick (the caller's ``_persist_setting`` hook), both
    # written by ``set_beta``, not here.  When a Flask request identifies no detector
    # (e.g. the VTSBrowser, which has a dataset but no loaded detector), the
    # active context is the frozen request-missing sentinel; there is no
    # detector to cache the value on, so skip the cache write rather than
    # raising ``RequestMissingContextError``.  The user-settings persist still
    # runs, so a balance echoed back by a bulk settings save is a harmless
    # no-op instead of a 400.
    if is_request_missing_detector_context(ctx):
        return
    ctx.beta = None if value is None else float(value)
    ctx.beta_seeded = True


# ---------------------------------------------------------------------------
# Context managers for explicit, scoped context switching
# ---------------------------------------------------------------------------


class with_dataset_context:
    """Context manager for temporarily switching the active dataset.

    Saves the current active dataset ID on entry, switches to the
    requested *dataset_id*, and restores the original on exit
    even if an exception occurs.

    Usage::

        with with_dataset_context("my_dataset"):
            # code here sees my_dataset's medias, coverage atlas, etc.
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
