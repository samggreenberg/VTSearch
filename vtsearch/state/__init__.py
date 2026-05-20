"""App-tier ``vtsearch.state`` shim — re-exports library state + adds proxies.

The library state lives in :mod:`vtscore.state` (no Flask, no settings).
This package keeps ``from vtsearch.state import medias`` working for app
code by re-exporting every library name and adding the app-side proxy
view (``medias``, ``good_votes``, …) from :mod:`vtsearch.shim.state_proxies`.

See ``../../vtscore/docs/architecture.md`` Phase 8 — this file is the
canonical example of an "app-tier shim that re-exports the library names
alongside the proxies".
"""

from __future__ import annotations

# Re-export every public name from the library tier.
from vtscore.state import (  # noqa: F401
    DatasetContext,
    DetectorContext,
    _core,
    _persist_setting,
    _state_lock,
    add_label_to_history,
    add_textsort_suggestion,
    apply_label,
    apply_label_with_click_time,
    apply_labels_bulk_with_click_time,
    assign_click_time,
    build_diversity_tree,
    build_diversity_tree_for_context,
    build_media_lookup,
    clear_all,
    clear_all_contexts,
    clear_all_detector_contexts,
    clear_medias,
    clear_votes,
    collapse_duplicates,
    diversity_tree_label,
    diversity_tree_next_sample,
    diversity_tree_unlabel,
    find_missing_entries,
    get_active_context,
    get_active_detector_context,
    get_calibrate_count,
    get_calibration_fraction,
    get_context,
    get_dataset_display_name,
    get_detector_context,
    get_diversity_tree,
    get_dupe_count,
    get_find_initial_labels,
    get_inclusion,
    get_learned_scores,
    get_media,
    get_safe_thresholds,
    get_textsort_suggestions,
    get_thread_dataset_context,
    get_thread_detector_context,
    get_vote_click_times,
    list_loaded_dataset_ids,
    list_loaded_detector_ids,
    next_media_id,
    register_context,
    register_detector_context,
    register_setting_persister,
    remove_click_time,
    resolve_media_ids,
    set_calibrate_count,
    set_calibration_fraction,
    set_dataset_display_name,
    set_find_initial_labels,
    set_inclusion,
    set_safe_thresholds,
    set_thread_dataset_context,
    set_thread_detector_context,
    snapshot_medias,
    toggle_vote,
    unregister_context,
    unregister_detector_context,
    update_learned_scores,
    with_dataset_context,
    with_detector_context,
)

# App-tier proxy view — the convenience facade that makes
# ``from vtsearch.state import medias`` feel like a module-level dict.
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
