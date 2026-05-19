"""Restore saved labels from a detector's labelset into votes.

Provides :func:`restore_labels_from_detector` which matches labelset entries
to loaded medias by origin, MD5, and origin_name, with a second pass
resolving origin files for cross-dataset scenarios.
"""

from __future__ import annotations


def restore_labels_from_detector(det_data: dict) -> int:  # noqa: C901
    """Restore saved labels from a detector's labelset into votes.

    Matches labelset entries to loaded medias by origin+origin_name, MD5, and
    origin_name fallback.  For entries that still don't match (cross-dataset
    scenario), resolves the original file from its origin trail, computes its
    MD5, and checks for a match in the loaded dataset.

    Restored labels are applied silently: ``label_history`` is not appended
    and the diversity tree is not pre-marked, so per-dataset Smart/Stable
    trends and span/diversity coverage start fresh from the user's session
    votes.  The good/bad counts still reflect restored labels, so autopilot's
    initial Find Goods / Find Bads gates can skip ahead.

    Returns the number of labels successfully restored.
    """
    from vtscore.datasets.labelset import LabeledElement, LabelSet
    from vtsearch.state import (
        apply_label,
        build_media_lookup,
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

    origin_lookup, md5_lookup, name_lookup = build_media_lookup(snap)

    restored = 0
    unresolved: list[LabeledElement] = []  # elements needing origin resolution
    for elem in labelset.elements:
        if elem.label not in ("good", "bad"):
            continue
        cids = resolve_media_ids(elem.to_dict(), origin_lookup, md5_lookup, name_lookup)
        if cids:
            for cid in cids:
                apply_label(cid, elem.label, silent=True)
            restored += 1
        else:
            unresolved.append(elem)

    # Second pass: resolve unmatched labels from their origin files.
    # When a detector was trained on Dataset A and we're now on Dataset B,
    # the origin+name keys won't match.  But if the same underlying file
    # exists in both datasets, resolving the origin file and computing its
    # MD5 lets us match by content hash.
    if unresolved:
        import hashlib
        import logging

        from vtscore.detectors.resolver import resolve_file_context

        _log = logging.getLogger(__name__)

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
                    h = hashlib.md5()
                    with open(resolved_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
                    resolved_md5 = h.hexdigest()
                except OSError:
                    _log.debug("restore-labels: could not read resolved file %s", resolved_path)
                    continue

                cids = md5_lookup.get(resolved_md5, [])
                if cids:
                    for cid in cids:
                        apply_label(cid, elem.label, silent=True)
                    restored += 1
                else:
                    _log.debug(
                        "restore-labels: resolved %s but MD5 %s not in loaded dataset",
                        resolved_path,
                        resolved_md5,
                    )

    return restored
