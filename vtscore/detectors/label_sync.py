"""Sync current votes into the loaded detector's labelset on disk.

Provides :func:`sync_labels_to_loaded_detector` which persists the active
detector's votes into the corresponding detector JSON file so the dashboard
stays up-to-date without an explicit save.

The sync is *non-destructive across datasets*: entries in the on-disk
labelset whose origin does not match anything in the currently-loaded
dataset are left untouched.  Entries whose origin matches a current-dataset
media item are reconciled against the active votes - replaced with the new
label, or removed when the user has untoggled the vote.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from vtscore.datasets.labelset import LabelSet
from vtscore.state.core import DetectorContext

# Serialises every read → merge → write pass over a detector JSON file:
# :func:`sync_labels_to_loaded_detector` here, plus the route-layer writers
# (``save_detector_labels``, ``vote_detector_label``,
# ``import_labels_into_detector``, ``find_corrections_to_detector``).  Two
# concurrent RMWs on the same detector each merge against a stale base and
# the last writer drops the other's just-written entries (lost update).  The
# write itself is atomic (os.replace), but atomicity doesn't serialise the
# read-modify-write.  Every taker acquires this lock *before* ``_state_lock``
# (via ``validated_vote_snapshot`` or an explicit ``with _state_lock`` inside
# the body), so no ordering cycle is possible.
#
# Public because those out-of-module takers are part of the contract: an
# out-of-package writer that does its own detector-JSON RMW *must* hold this
# lock.  Reach it through :mod:`vtscore.detectors.labelset_ops`.
label_sync_write_lock = threading.Lock()


def _get_loaded_detector_state() -> tuple[dict[str, Any], Path, dict[str, Any], DetectorContext] | None:
    """Resolve the loaded detector's registry entry, on-disk path, payload, and context.

    Returns ``None`` (and the outer sync becomes a no-op) when any of the
    guards trip: we are in find mode, no detector is active, the registry
    entry is missing/nameless, or the detector JSON file is missing.
    """
    from vtscore.detectors.registry import get_detector, is_find_mode
    from vtscore.detectors.store import _detector_path, _read_detector
    from vtscore.state import get_active_detector_context

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


def merge_labelsets_across_datasets(
    existing_ls: LabelSet,
    current_ls: LabelSet,
    current_dataset_medias: dict[int, dict[str, Any]],
) -> LabelSet:
    """Merge a fresh per-dataset labelset into a cross-dataset existing one.

    Existing entries that resolve to a media in *current_dataset_medias* are
    dropped - they get replaced by ``current_ls``, which is the authoritative
    record of what the user has voted there.  Entries that resolve to nothing
    were accumulated under other datasets and are kept verbatim.

    Ownership is decided by the *same* resolution
    (:func:`~vtscore.state.media_lookup.resolve_media_ids`, origin **or** md5)
    that :func:`~vtscore.detectors.label_restoration.restore_labels_from_detector`
    uses to turn labelset elements back into votes.  That symmetry is the whole
    point: an element that becomes a vote on load is an element ``current_ls``
    re-emits, so keeping the original beside it stores the same media twice.
    Comparing :func:`~vtscore.datasets.labelset.element_key` instead - which
    prefers origin and so misses an entry that shares only a content hash - is
    what made a 300-image pass report ``num_training: 356`` (issue #3174).

    Duplicate identities across the resulting list are collapsed (first
    occurrence wins), so a labelset that already carried duplicates heals on
    the next write.

    *current_dataset_medias* must be the same medias snapshot *current_ls* was
    composed from, so the two halves of the merge can't straddle a concurrent
    dataset switch.
    """
    from vtscore.datasets.labelset import element_identity_keys
    from vtscore.state import build_media_lookup, resolve_media_ids

    origin_lookup, md5_lookup, name_lookup = build_media_lookup(current_dataset_medias)

    new_elements = []
    seen_keys: set = set()

    def _take(el) -> bool:
        """Record *el*'s identities, returning False if one was already seen."""
        keys = element_identity_keys(el)
        if any(k in seen_keys for k in keys):
            return False
        seen_keys.update(keys)
        return True

    for el in existing_ls.elements:
        if resolve_media_ids(el.to_dict(), origin_lookup, md5_lookup, name_lookup):
            continue
        if not _take(el):
            continue
        new_elements.append(el)
    for el in current_ls.elements:
        if not _take(el):
            continue
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
    with label_sync_write_lock:
        _sync_labels_to_loaded_detector_locked()


def _sync_labels_to_loaded_detector_locked() -> None:
    """Body of :func:`sync_labels_to_loaded_detector`; caller holds the lock."""
    state = _get_loaded_detector_state()
    if state is None:
        return
    entry, path, data, det_ctx = state

    from vtscore.detectors.dataset_sync import validated_vote_snapshot
    from vtscore.detectors.registry import update_detector
    from vtscore.detectors.store import _write_detector

    vote_snap = validated_vote_snapshot()
    if not vote_snap.safe:
        # Vote dicts can't be proved keyed in the active dataset's cid space
        # (concurrent dataset switch on the same detector, missing
        # ``X-Dataset-Id`` header, or no registry entry).  Writing now with
        # an empty composition would erase the active dataset's on-disk
        # labels - drop this sync; the next vote will trigger another that
        # runs against a consistent snapshot.
        return
    snap = vote_snap.medias
    current_ls = LabelSet.from_clips_and_votes(
        snap,
        vote_snap.good_votes,
        vote_snap.bad_votes,
        expand_dupes=False,
        vote_region_boxes=vote_snap.vote_region_boxes,
        vote_provenance=vote_snap.vote_provenance,
    )
    existing_ls = LabelSet.from_dict(data.get("labelset") or {})

    # Existing labelset entries that resolve into the active dataset are
    # "owned" by it and get reconciled against the current votes; entries that
    # resolve to nothing are cross-dataset and are preserved verbatim.
    merged = merge_labelsets_across_datasets(existing_ls, current_ls, snap)
    data["labelset"] = merged.to_dict()
    _write_detector(path, data)

    _refresh_detector_caches(det_ctx, merged, path, data.get("media_type", "") or "")

    import time as _time

    update_detector(entry["id"], num_training=len(merged), last_trained_at=_time.time())
