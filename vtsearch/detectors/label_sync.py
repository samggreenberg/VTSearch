"""Sync current votes into the loaded detector's labelset on disk.

Provides :func:`sync_labels_to_loaded_detector` which persists the active
detector's votes into the corresponding detector JSON file so the dashboard
stays up-to-date without an explicit save.

The sync is *non-destructive across datasets*: entries in the on-disk
labelset whose origin does not match anything in the currently-loaded
dataset are left untouched.  Entries whose origin matches a current-dataset
media item are reconciled against the active votes — replaced with the new
label, or removed when the user has untoggled the vote.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vtsearch.datasets.labelset import LabelSet
from vtsearch.state.core import DetectorContext


def _get_loaded_detector_state() -> tuple[dict[str, Any], Path, dict[str, Any], DetectorContext] | None:
    """Resolve the loaded detector's registry entry, on-disk path, payload, and context.

    Returns ``None`` (and the outer sync becomes a no-op) when any of the
    guards trip: we are in find mode, no detector is active, the registry
    entry is missing/nameless, or the detector JSON file is missing.
    """
    from vtsearch.detectors.registry import get_detector, is_find_mode
    from vtsearch.detectors.store import _detector_path, _read_detector
    from vtsearch.state import get_active_detector_context

    if is_find_mode():
        return None

    det_ctx = get_active_detector_context()
    loaded_id = det_ctx.detector_id if det_ctx.detector_id else None
    if not loaded_id:
        return None

    entry = get_detector(loaded_id)
    if not entry or not entry.get("name"):
        return None

    path = _detector_path(entry["name"])
    data = _read_detector(path)
    if data is None:
        return None

    return entry, path, data, det_ctx


def _merge_labelsets_across_datasets(
    existing_ls: LabelSet,
    current_ls: LabelSet,
    current_dataset_keys: set,
) -> LabelSet:
    """Merge a fresh per-dataset labelset into a cross-dataset existing one.

    Existing entries owned by the active dataset (key in ``current_dataset_keys``)
    are dropped — they get replaced by ``current_ls``. Other entries are kept
    verbatim. Duplicate keys across the resulting list are collapsed
    (first occurrence wins).
    """
    from vtsearch.datasets.labelset import element_key

    new_elements = []
    seen_keys: set = set()
    for el in existing_ls.elements:
        key = element_key(el)
        if key in current_dataset_keys:
            continue
        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        new_elements.append(el)
    for el in current_ls.elements:
        key = element_key(el)
        if key is not None:
            if key in seen_keys:
                continue
            seen_keys.add(key)
        new_elements.append(el)
    return LabelSet(new_elements)


def _refresh_detector_caches(
    det_ctx: DetectorContext,
    merged: LabelSet,
    path: Path,
    media_type: str,
) -> None:
    """Refresh the cached labelset, mtime, and vote counters after a write.

    Keeps ``ensure_votes_match_active_dataset`` from re-hydrating the file
    we just rewrote.
    """
    det_ctx.labelset_good_count = sum(1 for el in merged.elements if el.label == "good")
    det_ctx.labelset_bad_count = sum(1 for el in merged.elements if el.label == "bad")
    det_ctx.cached_labelset = merged
    det_ctx.cached_labelset_media_type = media_type or det_ctx.cached_labelset_media_type
    try:
        det_ctx.cached_labelset_mtime = path.stat().st_mtime
    except OSError:
        det_ctx.cached_labelset_mtime = 0.0


def sync_labels_to_loaded_detector() -> None:
    """Persist the current votes into the loaded detector's labelset (if any).

    Called automatically after each vote so the dashboard's "# Training" and
    "Last Trained" columns stay up to date without an explicit save.

    Skipped when the detector is in "find mode" (after ``/api/find-label``),
    because the global votes reflect scoring results on a different dataset,
    not the detector's original training labels.
    """
    state = _get_loaded_detector_state()
    if state is None:
        return
    entry, path, data, det_ctx = state

    from vtsearch.datasets.labelset import media_element_key
    from vtsearch.detectors.registry import update_detector
    from vtsearch.detectors.store import _write_detector
    from vtsearch.state import bad_votes, good_votes, snapshot_medias, vote_region_boxes

    snap = snapshot_medias()
    current_ls = LabelSet.from_clips_and_votes(
        snap,
        good_votes,
        bad_votes,
        expand_dupes=False,
        vote_region_boxes=dict(vote_region_boxes),
    )
    existing_ls = LabelSet.from_dict(data.get("labelset") or {})

    # Origin keys for *every* media in the active dataset (voted or not).
    # Existing labelset entries that match one of these keys are considered
    # "owned" by the active dataset and will be reconciled against the
    # current votes; entries that don't match are cross-dataset entries
    # and are preserved verbatim.
    current_dataset_keys = set()
    for media in snap.values():
        key = media_element_key(media)
        if key is not None:
            current_dataset_keys.add(key)

    merged = _merge_labelsets_across_datasets(existing_ls, current_ls, current_dataset_keys)
    data["labelset"] = merged.to_dict()
    _write_detector(path, data)

    _refresh_detector_caches(det_ctx, merged, path, data.get("media_type", "") or "")

    import time as _time

    update_detector(entry["id"], num_training=len(merged), last_trained_at=_time.time())
