"""Keep a loaded detector's cid-keyed vote state aligned with the active dataset.

A ``DetectorContext``'s ``good_votes`` / ``bad_votes`` (and related per-vote
state) are dicts keyed by integer media id — but media ids are only meaningful
within a single dataset.  When the same detector is used across two datasets
(e.g. trained on dataset A, then reused with dataset B), the in-memory cids
from A leak into B's space, so unrelated B-medias whose ids happen to coincide
with A's voted ids appear as voted in the labeling UI.  Worse, the next vote
in B feeds those stale cids into ``sync_labels_to_loaded_detector``, which
silently writes corrupt entries back to the on-disk labelset.

The on-disk labelset is the canonical source of truth (dataset-agnostic — its
elements are keyed by origin / md5).  This module re-derives the cid dicts
from that labelset whenever the active dataset has changed.
"""

from __future__ import annotations


def ensure_votes_match_active_dataset() -> None:
    """Rehydrate the active detector's cid-keyed vote state against the active dataset.

    No-op unless all of these hold:

    * a detector is loaded,
    * a dataset is loaded (and its id is non-empty),
    * the detector has a registry entry that points at a real on-disk file
      (so we have a labelset to rehydrate from), and
    * the detector's recorded ``votes_dataset_id`` differs from the active
      dataset id.

    When triggered, clears the per-dataset detector state (good_votes,
    bad_votes, label_history, vote_click_times, click_counter,
    last_learned_scores, find_initial_labels, training_medias) and replays
    ``restore_labels_from_detector`` against the active dataset's medias,
    then stamps ``votes_dataset_id`` so subsequent requests within the same
    dataset are no-ops.
    """
    from vtsearch.utils.state_core import (
        _state_lock,
        get_active_context,
        get_active_detector_context,
    )

    det_ctx = get_active_detector_context()
    if not det_ctx.detector_id:
        return
    ds_ctx = get_active_context()
    if not ds_ctx.dataset_id:
        # No active dataset; preserve whatever the detector last saw so a
        # request that happens to omit the dataset header doesn't wipe state.
        return
    if det_ctx.votes_dataset_id == ds_ctx.dataset_id:
        return

    from vtsearch.models.detector_registry import get_detector
    from vtsearch.models.detector_store import _detector_path, _read_detector
    from vtsearch.models.label_restoration import restore_labels_from_detector

    entry = get_detector(det_ctx.detector_id)
    if not entry or not entry.get("name"):
        # No registry entry (typical in unit tests that build DetectorContext
        # directly); nothing to rehydrate from.
        return

    path = _detector_path(entry["name"])
    data = _read_detector(path)
    if data is None:
        # Detector file missing — still mark the dataset transition so we
        # don't keep stale cids around.
        with _state_lock:
            det_ctx.good_votes.clear()
            det_ctx.bad_votes.clear()
            det_ctx.label_history.clear()
            det_ctx.vote_click_times.clear()
            det_ctx.click_counter = 0
            det_ctx.last_learned_scores.clear()
            det_ctx.find_initial_labels.clear()
            det_ctx.training_medias.clear()
            det_ctx.votes_dataset_id = ds_ctx.dataset_id
        return

    with _state_lock:
        # Re-check inside the lock — another request may have rehydrated us.
        if det_ctx.votes_dataset_id == ds_ctx.dataset_id:
            return
        det_ctx.good_votes.clear()
        det_ctx.bad_votes.clear()
        det_ctx.label_history.clear()
        det_ctx.vote_click_times.clear()
        det_ctx.click_counter = 0
        det_ctx.last_learned_scores.clear()
        det_ctx.find_initial_labels.clear()
        det_ctx.training_medias.clear()
        restore_labels_from_detector(data)
        det_ctx.votes_dataset_id = ds_ctx.dataset_id
