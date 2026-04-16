"""Sync current votes into the loaded trainable model's labelset on disk.

Provides :func:`sync_labels_to_loaded_model` which persists the active
detector's votes into the corresponding trainable-model JSON file so the
dashboard stays up-to-date without an explicit save.
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

    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.utils import bad_votes, good_votes, snapshot_medias

    labelset = LabelSet.from_clips_and_votes(snapshot_medias(), good_votes, bad_votes, expand_dupes=False)
    data["labelset"] = labelset.to_dict()
    _write_model(path, data)

    import time as _time

    update_model(entry["id"], num_training=len(labelset), last_trained_at=_time.time())
