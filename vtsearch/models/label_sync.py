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


def sync_labels_to_loaded_detector() -> None:
    """Persist the current votes into the loaded detector's labelset (if any).

    Called automatically after each vote so the dashboard's "# Training" and
    "Last Trained" columns stay up to date without an explicit save.

    Skipped when the detector is in "find mode" (after ``/api/find-label``),
    because the global votes reflect scoring results on a different dataset,
    not the detector's original training labels.
    """
    from vtsearch.models.detector_registry import get_detector, is_find_mode, update_detector
    from vtsearch.models.detector_store import _detector_path, _read_detector, _write_detector
    from vtsearch.utils import get_active_detector_context

    if is_find_mode():
        return

    det_ctx = get_active_detector_context()
    loaded_id = det_ctx.detector_id if det_ctx.detector_id else None
    if not loaded_id:
        return

    entry = get_detector(loaded_id)
    if not entry or not entry.get("name"):
        return

    det_name = entry["name"]
    path = _detector_path(det_name)
    data = _read_detector(path)
    if data is None:
        return

    from vtsearch.datasets.labelset import LabelSet, element_key, media_element_key
    from vtsearch.utils import bad_votes, good_votes, snapshot_medias, vote_region_boxes

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

    new_elements = []
    seen_keys = set()
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

    merged = LabelSet(new_elements)
    data["labelset"] = merged.to_dict()
    _write_detector(path, data)

    det_ctx.labelset_good_count = sum(1 for el in merged.elements if el.label == "good")
    det_ctx.labelset_bad_count = sum(1 for el in merged.elements if el.label == "bad")

    import time as _time

    update_detector(entry["id"], num_training=len(merged), last_trained_at=_time.time())
