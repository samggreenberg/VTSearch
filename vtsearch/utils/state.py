"""Global state management for medias and votes.

This module is a re-export facade.  The actual state variables live in
``state_core``, and the functions are split across ``state_votes``,
``state_clicks``, ``state_processors``, ``state_diversity``, and
``state_media_lookup``.  Remaining functions (clear_medias, clear_all,
settings wrappers) live here.

All public names are importable from ``vtsearch.utils.state`` exactly
as before, so no call-sites need to change.
"""

from __future__ import annotations

import gc

# Re-export all state variables from state_core --------------------------
from vtsearch.utils.state_core import (  # noqa: F401
    _click_counter,
    _dataset_display_name,
    _diversity_tree,
    _state_lock,
    autorun_detectors,
    autorun_extractors,
    autorun_localizers,
    bad_votes,
    good_votes,
    inclusion,
    label_history,
    last_learned_scores,
    medias,
    textsort_suggestions,
    vote_click_times,
)
import vtsearch.utils.state_core as _core  # noqa: F401 — for conftest direct access

# Re-export click tracking ------------------------------------------------
from vtsearch.utils.state_clicks import (  # noqa: F401
    assign_click_time,
    get_vote_click_times,
    remove_click_time,
)

# Re-export vote/label operations -----------------------------------------
from vtsearch.utils.state_votes import (  # noqa: F401
    add_label_to_history,
    add_textsort_suggestion,
    apply_label,
    apply_label_with_click_time,
    clear_votes,
    get_learned_scores,
    get_textsort_suggestions,
    toggle_vote,
    update_learned_scores,
)

# Re-export processor CRUD -------------------------------------------------
from vtsearch.utils.state_processors import (  # noqa: F401
    add_autorun_detector,
    add_autorun_extractor,
    add_autorun_localizer,
    get_autodetect_detectors_by_media,
    get_autorun_detector_examples,
    get_autorun_detectors,
    get_autorun_detectors_by_media,
    get_autorun_extractors,
    get_autorun_extractors_by_media,
    get_autorun_localizers,
    get_autorun_localizers_by_media,
    remove_autorun_detector,
    remove_autorun_extractor,
    remove_autorun_localizer,
    rename_autorun_detector,
    rename_autorun_extractor,
    rename_autorun_localizer,
    set_autorun_detector_autodetect,
    set_autorun_detector_examples,
)

# Re-export diversity tree --------------------------------------------------
from vtsearch.utils.state_diversity import (  # noqa: F401
    build_diversity_tree,
    diversity_tree_label,
    diversity_tree_next_sample,
    diversity_tree_unlabel,
    get_diversity_tree,
)

# Re-export media lookup ----------------------------------------------------
from vtsearch.utils.state_media_lookup import (  # noqa: F401
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


def clear_medias() -> None:
    """Clear all loaded medias from memory.

    Removes all entries from the ``medias`` dict in place. Does not affect
    votes or label history. Also clears the progress model cache since
    cached models reference media embeddings.  Also clears the diversity tree
    and the dataset display name override.
    """
    from vtsearch.models.progress import clear_progress_cache

    with _state_lock:
        medias.clear()
        _core._diversity_tree = None
        _core._dataset_display_name = None
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


def get_inclusion() -> int:
    """Return the current inclusion setting.

    On first call the value is loaded from the persisted settings file so
    that it survives app restarts.
    """
    with _state_lock:
        if _core.inclusion is None:
            from vtsearch import settings

            _core.inclusion = settings.get_inclusion()
        return _core.inclusion


def set_inclusion(value: int) -> None:
    """Set the global inclusion value and persist it to the settings file.

    Also clears the progress model cache since cached models were trained
    with the old inclusion value.
    """
    with _state_lock:
        if value != _core.inclusion:
            from vtsearch.models.progress import clear_progress_cache

            clear_progress_cache()
        _core.inclusion = value

    from vtsearch import settings

    settings.set_inclusion(value)


def get_dataset_display_name() -> str | None:
    """Return the current dataset display name override, or ``None``."""
    with _state_lock:
        return _core._dataset_display_name


def set_dataset_display_name(name: str | None) -> None:
    """Set (or clear) the dataset display name override."""
    with _state_lock:
        _core._dataset_display_name = name


def get_calibrate_count() -> int:
    """Return the number of calibration splits from settings."""
    from vtsearch import settings

    return settings.get_calibrate_count()


def set_calibrate_count(value: int) -> None:
    """Set the calibrate count and persist it to the settings file."""
    from vtsearch import settings

    settings.set_calibrate_count(value)


def get_calibration_fraction() -> float:
    """Return the calibration fraction from settings."""
    from vtsearch import settings

    return settings.get_calibration_fraction()


def set_calibration_fraction(value: float) -> None:
    """Set the calibration fraction and persist it to the settings file."""
    from vtsearch import settings

    settings.set_calibration_fraction(value)


def get_safe_thresholds() -> bool:
    """Return whether safe thresholds blending is enabled.

    Reads the value from the persisted settings file.
    """
    from vtsearch import settings

    return settings.get_safe_thresholds()


def set_safe_thresholds(value: bool) -> None:
    """Set the safe thresholds flag and persist it to the settings file."""
    from vtsearch import settings

    settings.set_safe_thresholds(value)
