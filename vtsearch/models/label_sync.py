"""Sync current votes into the loaded trainable model's labelset on disk.

Provides :func:`sync_labels_to_loaded_model` which persists the active
detector's votes into the corresponding trainable-model JSON file so the
dashboard stays up-to-date without an explicit save.

The sync is *non-destructive across datasets*: entries in the on-disk
labelset whose origin does not match anything in the currently-loaded
dataset are left untouched.  Entries whose origin matches a current-dataset
media item are reconciled against the active votes — replaced with the new
label, or removed when the user has untoggled the vote.
"""

from __future__ import annotations


def sync_labels_to_loaded_model() -> None:
    """Persist the current votes into the loaded model's labelset (if any).

    Called automatically after each vote so the dashboard's "# Training"
    and "Last Trained" columns stay up to date without an explicit save.

    Skipped when the model is in "find mode" (after ``/api/find-label``),
    because the global votes reflect scoring results on a different dataset,
    not the model's original training labels.
    """
    from vtsearch.models.registry import get_model, is_find_mode, update_model
    from vtsearch.models.trainable_model_store import _model_path, _read_model, _write_model
    from vtsearch.utils import get_active_detector_context

    if is_find_mode():
        return

    det_ctx = get_active_detector_context()
    loaded_id = det_ctx.detector_id if det_ctx.detector_id else None
    if not loaded_id:
        return

    entry = get_model(loaded_id)
    if not entry or not entry.get("trainable") or not entry.get("trainable_model_name"):
        return

    tm_name = entry["trainable_model_name"]
    path = _model_path(tm_name)
    data = _read_model(path)
    if data is None:
        return

    from vtsearch.datasets.labelset import LabelSet, element_key, media_element_key
    from vtsearch.utils import bad_votes, good_votes, snapshot_medias

    snap = snapshot_medias()
    current_ls = LabelSet.from_clips_and_votes(snap, good_votes, bad_votes, expand_dupes=False)
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
    _write_model(path, data)

    import time as _time

    update_model(entry["id"], num_training=len(merged), last_trained_at=_time.time())
