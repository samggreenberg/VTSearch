"""Library-tier state package: contexts, votes, clicks, coverage, lookup.

The library-only API for the dataset / detector context system and its
operations.  No Flask, no ``vtsearch.settings``, no proxy view - those
app-tier concerns live in :mod:`vtsearch.state` (a thin shim that
re-exports this package and layers the proxy view on top).
"""

from __future__ import annotations

import gc
import warnings
from collections.abc import Callable
from typing import Any

# Re-export the lock.  The app-tier proxies live in ``vtsearch.state_proxies``.
from vtscore.state.core import _state_lock  # noqa: F401

# Re-export context management functions ---------------------------------
from vtscore.state.core import (  # noqa: F401
    DatasetContext,
    MediasDict,
    clear_all_contexts,
    get_active_context,
    get_context,
    get_thread_dataset_context,
    list_loaded_dataset_ids,
    register_context,
    register_dataset_context_resolver,
    rekey_dataset_context,
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
    loaded_detector_contexts,
    override_detector_context,
    register_detector_context,
    register_detector_context_resolver,
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
    find_boundary_next,
    find_queue_ids,
    get_find_initial_labels,
    get_find_scores,
    get_learned_scores,
    get_textsort_suggestions,
    rethreshold_unverified_find_items,
    set_find_initial_labels,
    set_find_scores,
    set_vote,
    toggle_vote,
    update_learned_scores,
)

# Re-export coverage atlas ---------------------------------------------------
from vtscore.state.coverage import (  # noqa: F401
    COVERAGE_ATLAS_AUTO_THRESHOLD,
    build_coverage_atlas,
    build_coverage_atlas_for_context,
    build_coverage_atlas_serializable,
    coverage_atlas_label,
    coverage_atlas_next_sample,
    coverage_atlas_unlabel,
    get_coverage_atlas,
    restore_coverage_atlas_from_cache,
    resync_coverage_atlas_to_detector,
    should_auto_build_coverage_atlas,
)

# Re-export media lookup ----------------------------------------------------
from vtscore.state.media_lookup import (  # noqa: F401
    _origin_key,
    build_md5_lookup,
    build_media_lookup,
    cached_md5_lookup,
    cached_media_lookups,
    collapse_duplicates,
    find_missing_entries,
    get_dupe_count,
    next_media_id,
    resolve_media_ids,
)

# Re-export near-duplicate collapsing ---------------------------------------
from vtscore.media.near_dupes import (  # noqa: F401
    collapse_near_duplicates,
    phash_image,
    simhash_text,
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

    Drops every derived cache on the context (see
    :meth:`~vtscore.state.core.DatasetContext.reset_derived_caches`) plus the
    coverage atlas, dataset display name override, and the per-step progress
    model cache, so RAM is released immediately rather than waiting for the
    next access.

    Dropping the cached layouts is also a correctness guard: the build routes
    serve a cached full / subset pyramid without re-checking the media-id
    signature, so a stale pyramid left over a reload-with-changed-contents
    would otherwise be returned for the new data.
    """
    from vtscore.detectors.labeling_progress import clear_progress_cache

    with _state_lock:
        ctx = _core.get_active_context()
        ctx.medias.clear()  # bumps media_revision via MediasDict
        ctx.reset_derived_caches()
        ctx.coverage_atlas = None
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

#: Settings the library knows how to hand back to a host for persistence.
#: Only :func:`_persist_setting` calls below can ever fire a persister, so a key
#: outside this set is a typo in the host's wiring that would silently never
#: fire - the same failure mode :data:`vtscore.achievements_hooks.KNOWN_EVENTS`
#: exists to catch, so this seam rejects it the same way rather than accepting
#: a registration nothing will ever call.
KNOWN_SETTING_KEYS: frozenset[str] = frozenset(
    {"min_precision", "beta", "line_preference", "calibrate_count", "calibration_fraction"}
)

#: Setting keys the library used to persist and no longer does.  Registering a
#: persister for one is accepted with a ``DeprecationWarning`` rather than
#: refused, so a host written against the old key set still starts; the
#: persister is never called.  ``inclusion`` retired with Inclusion as a user
#: preference (#4269): it is fixed at 0, so there is nothing to persist.
_RETIRED_SETTING_KEYS: frozenset[str] = frozenset({"inclusion"})

_setting_persisters: dict[str, Callable[[Any], None]] = {}


def register_setting_persister(key: str, fn: Callable[[Any], None]) -> None:
    """Install the persistence callback for setting *key*.

    Called by ``vtsearch/shim`` at app startup.  *key* must be one of
    :data:`KNOWN_SETTING_KEYS`; a retired key (``inclusion``) is
    accepted with a ``DeprecationWarning`` and never fires.
    """
    if key in _RETIRED_SETTING_KEYS:
        warnings.warn(
            f"register_setting_persister({key!r}) is deprecated: {key!r} is no longer a setting the library "
            "persists, so this persister will never be called.",
            DeprecationWarning,
            stacklevel=2,
        )
        return
    if key not in KNOWN_SETTING_KEYS:
        raise ValueError(f"Unknown setting key {key!r}; known keys: {sorted(KNOWN_SETTING_KEYS)}")
    _setting_persisters[key] = fn


def _persist_setting(key: str, value: Any) -> None:
    fn = _setting_persisters.get(key)
    if fn is not None:
        fn(value)


def get_inclusion() -> int:
    """Deprecated: always ``0``.

    Inclusion is no longer a user preference (#4269).  The operating point is
    the precision floor (:func:`get_min_precision`), and a line with no promise
    is the Inclusion 0 cut.  Inclusion survives only as the internal unit the
    threshold machinery measures cuts in.
    """
    warnings.warn(
        "vtscore.state.get_inclusion() is deprecated: Inclusion is retired as a user preference and is "
        "always 0. Read the precision floor instead (vtscore.state.get_min_precision).",
        DeprecationWarning,
        stacklevel=2,
    )
    return 0


def set_inclusion(value: int) -> None:
    """Deprecated: accepts only ``0``, which changes nothing.

    Inclusion is no longer a user preference (#4269), so there is no stored
    value to move.  ``0`` is accepted with a ``DeprecationWarning``; any other
    value raises ``ValueError`` rather than being silently ignored.  Set a
    precision floor with :func:`set_min_precision` instead.
    """
    from vtscore.config.core_config import _retired_inclusion

    _retired_inclusion("vtscore.state.set_inclusion()", value)


def get_min_precision() -> float | None:
    """The active detector's precision floor, or ``None`` when no floor is set.

    Seeded from the user's setting (``CoreConfig.min_precision``) the first
    time it is read for a detector.  A float in ``(0, 1]`` is the fraction of
    what the cut returns that should be right (#4245): the line keeps the
    floor's starting candidate until a spot check measures it, then the set the
    check ended on (#4272).  ``None`` means no floor: the line is the Inclusion 0 cut, with no promise
    attempted.  The app always sets a floor; ``None`` survives for library
    callers (#4269).
    """
    from vtscore.config import CoreConfig

    with _state_lock:
        seeded, val = _core._get_min_precision()
        if not seeded:
            val = CoreConfig.from_settings().min_precision
            _core._set_min_precision(val)
        return val


def set_min_precision(value: float | None) -> None:
    """Set the active detector's precision floor (``None`` clears it) and persist it via the registered hook.

    A pure cutoff knob: the active detector re-cuts its cached estimators at
    the new floor (no retrain), and in Find mode the unverified items re-split
    over the frozen scores.  Other loaded detectors keep their
    own floor; one that has not read its floor yet takes this one.
    """
    if value is not None and not 0.0 < value <= 1.0:
        raise ValueError(f"precision floor must be in (0, 1], got {value!r}")
    with _state_lock:
        seeded, old = _core._get_min_precision()
        changed = not seeded or value != old
        _core._set_min_precision(value)
        _persist_setting("min_precision", value)
    if changed and get_line_preference() == "floor":
        _core.recompute_detector_thresholds(value)
        rethreshold_unverified_find_items()


def line_knobs() -> dict[str, float | None]:
    """The preference the active detector's line is drawn at, as the trainer's keywords (#4413).

    ``{"min_precision": P, "beta": None}`` under the floor,
    ``{"min_precision": None, "beta": b}`` under the balance: what every retrain
    and re-cut passes on, so one switch moves every line.
    """
    if get_line_preference() == "balance":
        return {"min_precision": None, "beta": get_beta()}
    return {"min_precision": get_min_precision(), "beta": None}


def get_beta() -> float:
    """The active detector's balance (#4413): F-beta's beta, seeded from the user's setting on first read.

    Draws the line when :func:`get_line_preference` is ``"balance"``: the
    band walk stopped at the F-beta peak, or the mixture's F-beta argmax
    under the balance's cap before any check.
    """
    from vtscore.config import CoreConfig

    with _state_lock:
        seeded, val = _core._get_beta()
        if not seeded:
            val = float(CoreConfig.from_settings().beta)
            _core._set_beta(val)
        return val


def set_beta(value: float) -> None:
    """Set the active detector's balance and persist it; under the balance preference the line re-cuts (no retrain)."""
    from vtscore.training.thresholds import BETA_MAX, BETA_MIN

    value = float(value)
    if not BETA_MIN <= value <= BETA_MAX:
        raise ValueError(f"beta must be in [{BETA_MIN}, {BETA_MAX}], got {value!r}")
    with _state_lock:
        seeded, old = _core._get_beta()
        changed = not seeded or value != old
        _core._set_beta(value)
        _persist_setting("beta", value)
    if changed and get_line_preference() == "balance":
        _core.recompute_detector_thresholds(None, beta=value)
        rethreshold_unverified_find_items()


def get_line_preference() -> str:
    """Which preference draws the line (#4413): ``"floor"`` (the precision floor) or ``"balance"`` (F-beta)."""
    from vtscore.config import CoreConfig

    try:
        return str(CoreConfig.from_settings().line_preference)
    except RuntimeError:  # a library-only process with no settings builder
        return "floor"


def set_line_preference(value: str) -> None:
    """Switch the preference that draws the line, persist it, and re-cut every loaded detector at it."""
    from vtscore.config.runtime import LINE_PREFERENCES

    if value not in LINE_PREFERENCES:
        raise ValueError(f"line_preference must be one of {LINE_PREFERENCES}, got {value!r}")
    with _state_lock:
        _persist_setting("line_preference", value)
    if value == "balance":
        _core.recompute_detector_thresholds(None, beta=get_beta())
    else:
        _core.recompute_detector_thresholds(get_min_precision())
    rethreshold_unverified_find_items()


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


def get_calibration_fraction() -> float | None:
    """Return the user's explicit calibration fraction, or ``None`` when unset.

    ``None`` means "no explicit setting": training resolves it to the
    per-space production split for the detector's embedder - 0.3
    single-vector, 0.5 patch (issue #3287; see
    :func:`vtscore.detectors.training.resolve_calibration_fraction`).
    """
    from vtscore.config import CoreConfig

    return CoreConfig.from_settings().calibration_fraction


def set_calibration_fraction(value: float | None) -> None:
    """Set the calibration fraction.  Persistence delegated to the registered hook.

    ``None`` clears the explicit setting back to the per-embedder automatic
    default.
    """
    changed = value != get_calibration_fraction()
    _persist_setting("calibration_fraction", value)
    if changed:
        _core.invalidate_loaded_detector_models()
