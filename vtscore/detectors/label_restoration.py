"""Restore saved labels from a detector's labelset into votes.

Provides :func:`restore_labels_from_detector` which matches labelset entries
to loaded medias by origin, MD5, and origin_name, with a second pass
resolving origin files for cross-dataset scenarios.
"""

from __future__ import annotations

from vtscore.utils.hashing import file_md5


def _resolve_unmatched(unresolved: list, md5_lookup: dict[str, list[int]]) -> int:
    """Resolve unmatched labelset entries from their origin files.

    When a detector was trained on Dataset A and we're now on Dataset B,
    the origin+name keys won't match.  But if the same underlying file
    exists in both datasets, resolving the origin file and computing its
    MD5 lets us match by content hash.  Matched entries are labeled
    silently.

    Returns the number of labels restored via this fallback path.
    """
    import logging

    from vtscore.datasets.vote_provenance import read_provenance
    from vtscore.detectors.resolver import resolve_file_context
    from vtscore.state import apply_label

    _log = logging.getLogger(__name__)

    restored = 0
    for elem in unresolved:
        entry = elem.to_dict()
        origin = entry.get("origin")
        origin_name = entry.get("origin_name", "")
        filename = entry.get("filename", "")
        with resolve_file_context(origin, origin_name, filename) as resolved_path:
            if resolved_path is None:
                continue

            # Compute MD5 of the resolved file and check against loaded medias
            try:
                resolved_md5 = file_md5(resolved_path)
            except OSError:
                _log.debug("restore-labels: could not read resolved file %s", resolved_path)
                continue

            cids = md5_lookup.get(resolved_md5, [])
            if cids:
                for cid in cids:
                    apply_label(
                        cid,
                        elem.label,
                        silent=True,
                        region_box=elem.region_box,
                        provenance=read_provenance(elem.metadata),
                    )
                restored += 1
            else:
                _log.debug(
                    "restore-labels: resolved %s but MD5 %s not in loaded dataset",
                    resolved_path,
                    resolved_md5,
                )

    return restored


def restore_labels_from_detector(det_data: dict) -> int:
    """Restore saved labels from a detector's labelset into votes.

    Matches labelset entries to loaded medias by origin+origin_name, MD5, and
    origin_name fallback.  For entries that still don't match (cross-dataset
    scenario), resolves the original file from its origin trail, computes its
    MD5, and checks for a match in the loaded dataset.

    Restored labels are applied silently: ``label_history`` is not appended
    and the coverage atlas is not pre-marked, so per-dataset Smart/Stable
    trends and span/diversity coverage start fresh from the user's session
    votes.  The good/bad counts still reflect restored labels, so autopilot's
    initial Find Goods / Find Bads gates can skip ahead.

    The votes the restore leaves are recorded as the detector's
    ``label_history_base``: the learned sort trains on them as well as on the
    history, so the Smart / Stable cache replays the history on top of them
    (#5358).  Both callers run this on an empty history (a rehydrate clears it
    first, a load runs on a new context), so the votes it leaves are exactly
    what was in force before the history's first event.

    A Good element's ``region_box`` rides along into ``vote_region_boxes``,
    and any recorded surfacing provenance rides along into ``vote_provenance``.
    Dropping either here erases it the moment the next vote resyncs the
    labelset back to disk - the restored vote would be image-level and
    context-free, so the element it re-emits would be too.  (That is exactly
    the bug ``region_box`` had; provenance shares the hazard because the sync
    rebuilds the whole labelset from live vote state on every vote.)

    Returns the number of labels successfully restored.
    """
    restored = _apply_saved_labels(det_data)
    _record_history_base()
    return restored


def _apply_saved_labels(det_data: dict) -> int:
    """Apply *det_data*'s labelset to the votes, silently; :func:`restore_labels_from_detector`'s body."""
    from vtscore.datasets.labelset import LabeledElement, LabelSet
    from vtscore.datasets.vote_provenance import read_provenance
    from vtscore.state import (
        apply_label,
        cached_media_lookups,
        resolve_media_ids,
        snapshot_medias,
    )

    labelset_dict = det_data.get("labelset")
    if not labelset_dict:
        return 0

    labelset = LabelSet.from_dict(labelset_dict)
    if not labelset.elements:
        return 0

    snap = snapshot_medias()
    if not snap:
        return 0

    origin_lookup, md5_lookup, name_lookup = cached_media_lookups()

    restored = 0
    unresolved: list[LabeledElement] = []  # elements needing origin resolution
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        cids = resolve_media_ids(elem.to_dict(), origin_lookup, md5_lookup, name_lookup)
        if cids:
            for cid in cids:
                apply_label(
                    cid,
                    elem.label,
                    silent=True,
                    region_box=elem.region_box,
                    provenance=read_provenance(elem.metadata),
                )
            restored += 1
        else:
            unresolved.append(elem)

    # Second pass: resolve unmatched labels from their origin files.
    if unresolved:
        restored += _resolve_unmatched(unresolved, md5_lookup)

    return restored


def _record_history_base() -> None:
    """Record the active detector's votes as the base its label history replays on (#5358).

    Run after every restore, whatever it restored: a restore that found nothing
    to apply leaves no base, and must not leave the previous dataset's.
    """
    from vtscore.state import _state_lock, get_active_detector_context
    from vtscore.state.core import is_request_missing_detector_context

    with _state_lock:
        ctx = get_active_detector_context()
        if is_request_missing_detector_context(ctx):
            return
        ctx.label_history_base = (frozenset(ctx.good_votes), frozenset(ctx.bad_votes))
