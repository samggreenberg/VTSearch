"""Labelset source sync utilities.

Provides :func:`sync_to_labelset_source` which exports the current
detector's labels to its linked labelset source (if any), and
:func:`sync_from_labelset_source` which imports labels from the source.

A thread-local guard prevents re-exporting during an import pass.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Thread-local guard to prevent re-exporting to the source during an
# import-from-source pass.
_syncing = threading.local()


def sync_to_labelset_source() -> None:
    """Push current detector labels to the linked labelset source (if any).

    Call this after vote operations when the active detector may have
    a labelset source attached.  Skips silently if no source is configured
    or if we are already inside a sync-from-source import.
    """
    if getattr(_syncing, "active", False):
        return

    from vtsearch.utils.state_core import get_active_detector_context

    ctx = get_active_detector_context()
    if ctx is None or not ctx.labelset_source:
        return

    cfg = ctx.labelset_source
    source_name = cfg.get("source_name", "")
    if not source_name:
        return

    from vtsearch.labels.sources import get_labelset_source

    source = get_labelset_source(source_name)
    if source is None:
        logger.warning("Unknown labelset source: %s", source_name)
        return

    field_values = cfg.get("field_values", {})

    from vtsearch.datasets.labelset import LabelSet
    from vtsearch.utils.state_core import good_votes, bad_votes, medias

    try:
        labelset = LabelSet.from_clips_and_votes(dict(medias), dict(good_votes), dict(bad_votes))
        source.save(labelset, field_values)
    except Exception as exc:
        logger.exception("Failed to sync labels to source: %s", exc)


def sync_from_labelset_source(detector_id: str | None = None) -> list[dict[str, str]] | None:
    """Pull labels from the active detector's labelset source and apply them.

    Args:
        detector_id: If given, operate on this detector context. Otherwise
            use the currently active detector context.

    Returns:
        The imported label list, or ``None`` if no source is configured
        or the source file doesn't exist yet.
    """
    from vtsearch.utils.state_core import get_active_detector_context, get_detector_context

    if detector_id is not None:
        ctx = get_detector_context(detector_id)
    else:
        ctx = get_active_detector_context()

    if ctx is None or not ctx.labelset_source:
        return None

    cfg = ctx.labelset_source
    source_name = cfg.get("source_name", "")
    if not source_name:
        return None

    from vtsearch.labels.sources import get_labelset_source

    source = get_labelset_source(source_name)
    if source is None:
        logger.warning("Unknown labelset source: %s", source_name)
        return None

    field_values = cfg.get("field_values", {})
    try:
        labels = source.load(field_values)
    except Exception as exc:
        logger.exception("Failed to load from labelset source: %s", exc)
        return None

    if not labels:
        return None

    # Apply under sync guard to prevent re-exporting back to source.
    _syncing.active = True
    try:
        from vtsearch.utils.state_votes import apply_label

        for entry in labels:
            label = entry.get("label")
            md5 = entry.get("md5")
            if label not in ("good", "bad") or not md5:
                continue

            # Find media by md5
            from vtsearch.utils.state_core import medias

            for mid, media in medias.items():
                if media.get("md5") == md5:
                    apply_label(mid, label)
                    break
    finally:
        _syncing.active = False

    return labels
