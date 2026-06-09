"""Library-tier state package: contexts, votes, clicks, diversity, lookup.

The library-only API for the dataset / detector context system and its
operations.  No Flask, no ``vtsearch.settings``, no proxy view - those
app-tier concerns live in :mod:`vtsearch.state` (a thin shim that
re-exports this package and layers the proxy view on top).
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

# Re-export the lock.  The app-tier proxies live in ``vtsearch.shim``.
from vtscore.state.core import _state_lock  # noqa: F401

# Re-export context management functions ---------------------------------
from vtscore.state.core import (  # noqa: F401
    DatasetContext,
    clear_all_contexts,
    get_active_context,
    get_context,
    get_thread_dataset_context,
    list_loaded_dataset_ids,
    register_context,
    set_thread_dataset_context,
    thread_dataset_context,
    unregister_context,
)

# Detector context management -----------------------------------------------
from vtscore.state.core import (  # noqa: F401
    DetectorContext,
    clear_all_detector_contexts,
    get_active_detector_context,
    get_detector_context,
    get_thread_detector_context,
    list_loaded_detector_ids,
    register_detector_context,
    set_thread_detector_context,
    thread_detector_context,
    unregister_detector_context,
)

# Scoped context managers ---------------------------------------------------
from vtscore.state.core import (  # noqa: F401
    with_dataset_context,
    with_detector_context,
)
import vtscore.state.core as _core  # noqa: F401 - for conftest direct access

# Re-export click tracking ------------------------------------------------
from vtscore.state.clicks import (  # noqa: F401
    assign_click_time,
    get_vote_click_times,
    remove_click_time,
)

# Re-export vote/label operations -----------------------------------------
from vtscore.state.votes import (  # noqa: F401
    add_label_to_history,
    add_textsort_suggestion,
    apply_label,
    apply_label_with_click_time,
    apply_labels_bulk_with_click_time,
    clear_votes,
    get_find_initial_labels,
    get_learned_scores,
    get_textsort_suggestions,
    set_find_initial_labels,
    set_vote,
    toggle_vote,
    update_learned_scores,
)

# Re-export diversity tree --------------------------------------------------
from vtscore.state.diversity import (  # noqa: F401
    build_diversity_tree,
    build_diversity_tree_for_context,
    diversity_tree_label,
    diversity_tree_next_sample,
    diversity_tree_unlabel,
    get_diversity_tree,
    resync_diversity_tree_to_detector,
)

# Re-export media lookup ----------------------------------------------------
from vtscore.state.media_lookup import (  # noqa: F401
    _origin_key,
    build_media_lookup,
    collapse_duplicates,
    find_missing_entries,
    get_dupe_count,
    next_media_id,
    resolve_media_ids,
)


# ---------------------------------------------------------------------------
# Cross-cutting helpers
# ---------------------------------------------------------------------------


def snapshot_medias() -> dict[int, dict[str, Any]]:
    """Return a shallow copy of the active dataset's medias dict.

    Use this instead of accessing the proxy ``medias`` directly when you
    need to iterate or access multiple entries.  The snapshot is taken
    under ``_state_lock`` so a concurrent ``clear_medias()`` cannot cause
    a TOCTOU race.
    """
    with _state_lock:
        return dict(_core.get_active_context().medias)


def get_media(media_id: int) -> dict[str, Any] | None:
    """Return a single media entry by ID from the active dataset, or ``None``."""
    with _state_lock:
        return _core.get_active_context().medias.get(media_id)


def clear_medias() -> None:
    """Clear all loaded medias from the active dataset's context.

    Drops the cached embedding matrix, the 2-D projection + tile pyramids
    (one per bin shape), diversity tree, dataset display name override, and
    the per-step progress model cache so RAM is released immediately rather
    than waiting for the next access.

    Clearing ``_projection``/``_pyramids`` is also a correctness guard: the
    build route serves a cached pyramid without re-checking the media-id
    signature, so a stale pyramid left over a reload-with-changed-contents
    would otherwise be returned for the new data.
    """
    from vtscore.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        ctx = _core.get_active_context()
        ctx.medias.clear()
        ctx._emb_matrix_ids = None
        ctx._emb_matrix = None
        ctx._projection = None
        ctx._pyramids = {}
        ctx.diversity_tree = None
        ctx.dataset_display_name = None
    # ``_progress_lock`` is acquired strictly outside ``_state_lock`` so the
    # two locks never establish a cross-module ordering (audit M1).
    clear_progress_cache()
    gc.collect()


def clear_all() -> None:
    """Clear all medias, votes, and label history.

    Each clear acquires ``_state_lock`` independently - the two operations
    are *not* atomic with respect to each other so the progress cache can
    be cleared (under ``_progress_lock``) outside ``_state_lock`` (audit
    M1).  The sole caller is dataset-load, which immediately rebuilds
    state, so the transient mid-clear view is acceptable.
    """
    clear_medias()
    clear_votes()


# ---------------------------------------------------------------------------
# Settings-persistence hooks (app-side wiring; library default = no-op)
# ---------------------------------------------------------------------------
# Each ``set_X`` wrapper below does its in-memory work (e.g. cache
# invalidation) and then delegates persistence to whatever the app
# installed.  ``vtsearch/shim/register_app_persistence_hooks()`` wires
# each key here to the matching ``vtsearch.settings.set_*`` function at
# app startup.  Library-only callers (no app) see in-memory mutation
# only.  See Phase 2 of ``../docs/architecture.md``.

_setting_persisters: dict[str, Callable[[Any], None]] = {}


def register_setting_persister(key: str, fn: Callable[[Any], None]) -> None:
    """Install the persistence callback for setting *key*.

    Recognised keys: ``inclusion``, ``calibrate_count``, ``calibration_fraction``,
    ``safe_thresholds``.  Called by ``vtsearch/shim`` at app startup.
    """
    _setting_persisters[key] = fn


def _persist_setting(key: str, value: Any) -> None:
    fn = _setting_persisters.get(key)
    if fn is not None:
        fn(value)


def get_inclusion() -> int:
    """Return the current inclusion setting (loaded from CoreConfig on first call)."""
    from vtscore.config import CoreConfig

    with _state_lock:
        val = _core._get_inclusion()
        if val is None:
            val = CoreConfig.from_settings().inclusion
            _core._set_inclusion(val)
        return val


def set_inclusion(value: int) -> None:
    """Set the global inclusion value and persist it via the registered hook."""
    with _state_lock:
        changed = value != _core._get_inclusion()
        _core._set_inclusion(value)
        _persist_setting("inclusion", value)
    # ``_progress_lock`` is acquired strictly outside ``_state_lock`` so the
    # two locks never establish a cross-module ordering (audit M1).
    # ``_ensure_cache`` self-heals if a concurrent reader observes the new
    # inclusion before this clear runs - it re-clears whenever
    # ``_cache_inclusion`` differs from the current value.
    if changed:
        from vtscore.detectors.labeling_progress import clear_progress_cache

        clear_progress_cache()
        # Inclusion is a pure cutoff knob: re-threshold from the cached fold
        # orderings instead of dropping the (inclusion-independent) MLP, so the
        # scores stay frozen across a slide.  See docs/plans/find-verification-workflow.md.
        _core.recompute_detector_thresholds_for_inclusion(value)


def get_dataset_display_name() -> str | None:
    """Return the current dataset display name override, or ``None``."""
    with _state_lock:
        return _core._get_dataset_display_name()


def set_dataset_display_name(name: str | None) -> None:
    """Set (or clear) the dataset display name override."""
    with _state_lock:
        _core._set_dataset_display_name(name)


def get_calibrate_count() -> int:
    """Return the number of calibration splits."""
    from vtscore.config import CoreConfig

    return CoreConfig.from_settings().calibrate_count


def set_calibrate_count(value: int) -> None:
    """Set the calibrate count.  Persistence delegated to the registered hook."""
    changed = value != get_calibrate_count()
    _persist_setting("calibrate_count", value)
    if changed:
        _core.invalidate_loaded_detector_models()


def get_calibration_fraction() -> float:
    """Return the calibration fraction."""
    from vtscore.config import CoreConfig

    return CoreConfig.from_settings().calibration_fraction


def set_calibration_fraction(value: float) -> None:
    """Set the calibration fraction.  Persistence delegated to the registered hook."""
    changed = value != get_calibration_fraction()
    _persist_setting("calibration_fraction", value)
    if changed:
        _core.invalidate_loaded_detector_models()


def get_safe_thresholds() -> bool:
    """Return whether safe-thresholds blending is enabled."""
    from vtscore.config import CoreConfig

    return CoreConfig.from_settings().safe_thresholds


def set_safe_thresholds(value: bool) -> None:
    """Set the safe-thresholds flag.  Persistence delegated to the registered hook."""
    changed = value != get_safe_thresholds()
    _persist_setting("safe_thresholds", value)
    if changed:
        _core.invalidate_loaded_detector_models()
