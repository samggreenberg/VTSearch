"""Global state management for medias and votes.

This module is a re-export facade.  The actual context classes and
resolvers live in :mod:`vtsearch.state.core`, helper functions are split
across the other ``state.*`` submodules, and the app-tier proxy
instances (``medias``, ``good_votes``, …) live in
:mod:`vtsearch.shim.state_proxies` — they are re-exported here so
``from vtsearch.state import medias`` continues to work for the app
layer.  See Phase 3 of ``docs/plans/extract-library.md``.

All public names are importable from ``vtsearch.state`` exactly as
before, so no call-sites need to change.
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

# Re-export the lock + per-context proxy instances.  The lock is library
# code; the proxies are app-tier glue that lives in ``vtsearch.shim``.
from vtsearch.state.core import _state_lock  # noqa: F401
from vtsearch.shim.state_proxies import (  # noqa: F401
    bad_votes,
    good_votes,
    label_history,
    last_learned_scores,
    medias,
    textsort_suggestions,
    vote_click_times,
    vote_region_boxes,
)

# Re-export context management functions ---------------------------------
from vtsearch.state.core import (  # noqa: F401
    DatasetContext,
    clear_all_contexts,
    get_active_context,
    get_context,
    get_thread_dataset_context,
    list_loaded_dataset_ids,
    register_context,
    set_thread_dataset_context,
    unregister_context,
)

# Detector context management -----------------------------------------------
from vtsearch.state.core import (  # noqa: F401
    DetectorContext,
    clear_all_detector_contexts,
    get_active_detector_context,
    get_detector_context,
    get_thread_detector_context,
    list_loaded_detector_ids,
    register_detector_context,
    set_thread_detector_context,
    unregister_detector_context,
)

# Scoped context managers ---------------------------------------------------
from vtsearch.state.core import (  # noqa: F401
    with_dataset_context,
    with_detector_context,
)
import vtsearch.state.core as _core  # noqa: F401 — for conftest direct access

# Re-export click tracking ------------------------------------------------
from vtsearch.state.clicks import (  # noqa: F401
    assign_click_time,
    get_vote_click_times,
    remove_click_time,
)

# Re-export vote/label operations -----------------------------------------
from vtsearch.state.votes import (  # noqa: F401
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
    toggle_vote,
    update_learned_scores,
)

# Re-export diversity tree --------------------------------------------------
from vtsearch.state.diversity import (  # noqa: F401
    build_diversity_tree,
    build_diversity_tree_for_context,
    diversity_tree_label,
    diversity_tree_next_sample,
    diversity_tree_unlabel,
    get_diversity_tree,
)

# Re-export media lookup ----------------------------------------------------
from vtsearch.state.media_lookup import (  # noqa: F401
    _origin_key,
    build_media_lookup,
    collapse_duplicates,
    find_missing_entries,
    get_dupe_count,
    next_media_id,
    resolve_media_ids,
)


# ---------------------------------------------------------------------------
# Functions that remain here (cross-cutting or settings wrappers)
# ---------------------------------------------------------------------------


def snapshot_medias() -> dict[int, dict[str, Any]]:
    """Return a shallow copy of the active dataset's medias dict.

    Use this instead of accessing ``medias`` directly when you need to
    iterate over medias or access multiple entries.  The snapshot is
    taken under ``_state_lock`` so a concurrent ``clear_medias()`` cannot
    cause a TOCTOU race (e.g. ``KeyError`` between ``list(medias.keys())``
    and ``medias[cid]``).
    """
    with _state_lock:
        return dict(_core.get_active_context().medias)


def get_media(media_id: int) -> dict[str, Any] | None:
    """Return a single media entry by ID from the active dataset, or ``None``.

    Thread-safe: holds ``_state_lock`` for the duration of the lookup.
    """
    with _state_lock:
        return _core.get_active_context().medias.get(media_id)


def clear_medias() -> None:
    """Clear all loaded medias from the active dataset's context.

    Removes all entries from the active dataset's medias dict in place.
    Does not affect votes or label history. Also clears the progress model
    cache since cached models reference media embeddings.  Also clears the
    diversity tree and the dataset display name override.
    """
    from vtsearch.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        ctx = _core.get_active_context()
        ctx.medias.clear()
        # Drop the cached embedding matrix so its RAM is released along with
        # the medias dict.  Lazy rebuild on next access would also handle it,
        # but releasing now is the friendly thing to do.
        ctx._emb_matrix_ids = None
        ctx._emb_matrix = None
        ctx.diversity_tree = None
        ctx.dataset_display_name = None
        clear_progress_cache()
    gc.collect()


def clear_all() -> None:
    """Clear all medias, votes, and label history.

    Convenience wrapper that calls :func:`clear_medias` followed by
    :func:`clear_votes`.
    """
    with _state_lock:
        clear_medias()
        clear_votes()


# ---------------------------------------------------------------------------
# Settings-persistence hooks (app-side wiring; library default = no-op)
# ---------------------------------------------------------------------------
# Each ``set_X`` wrapper below does its in-memory work (e.g. cache
# invalidation) and then delegates persistence to whatever the app
# installed.  ``vtsearch/shim/register_app_persistence_hooks()`` wires
# each key here to the matching ``vtsearch.settings.set_*`` function at
# app startup, so user prefs continue to round-trip through
# ``data/<user>/user_settings.json`` exactly as before.  Library-only
# callers (no app) see in-memory mutation only, which is the right
# default — they can install their own persister via this hook if they
# want JSON-on-disk persistence.  See Phase 2 of
# ``docs/plans/extract-library.md``.

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
    """Return the current inclusion setting.

    On first call the value is loaded from the persisted settings file via
    :class:`vtsearch.config.CoreConfig` so it survives app restarts.
    """
    from vtsearch.config import CoreConfig

    with _state_lock:
        val = _core._get_inclusion()
        if val is None:
            val = CoreConfig.from_settings().inclusion
            _core._set_inclusion(val)
        return val


def set_inclusion(value: int) -> None:
    """Set the global inclusion value and persist it.

    Also clears the progress model cache since cached models were trained
    with the old inclusion value.  The persistence hop is delegated to
    whatever the app registered via :func:`register_setting_persister` —
    library-only callers without that hook just get in-memory mutation.
    """
    with _state_lock:
        if value != _core._get_inclusion():
            from vtsearch.detectors.labeling_progress import clear_progress_cache

            clear_progress_cache()
        _core._set_inclusion(value)
        _persist_setting("inclusion", value)


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
    from vtsearch.config import CoreConfig

    return CoreConfig.from_settings().calibrate_count


def set_calibrate_count(value: int) -> None:
    """Set the calibrate count.  Persistence delegated to the registered hook."""
    _persist_setting("calibrate_count", value)


def get_calibration_fraction() -> float:
    """Return the calibration fraction."""
    from vtsearch.config import CoreConfig

    return CoreConfig.from_settings().calibration_fraction


def set_calibration_fraction(value: float) -> None:
    """Set the calibration fraction.  Persistence delegated to the registered hook."""
    _persist_setting("calibration_fraction", value)


def get_safe_thresholds() -> bool:
    """Return whether safe-thresholds blending is enabled."""
    from vtsearch.config import CoreConfig

    return CoreConfig.from_settings().safe_thresholds


def set_safe_thresholds(value: bool) -> None:
    """Set the safe-thresholds flag.  Persistence delegated to the registered hook."""
    _persist_setting("safe_thresholds", value)
