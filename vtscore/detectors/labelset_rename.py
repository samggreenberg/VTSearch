"""Helpers for handling labelset source files when a detector is renamed.

When a detector with a labelset source whose ``filepath`` template uses
``{detector_name}`` is renamed, the old on-disk file becomes orphaned -
the next sync writes to the NEW resolved path, leaving the OLD file
behind as garbage.  These helpers detect the orphaned-file situation
and perform an atomic move; :func:`follow_rename_with_labelset_file` does
both, and the rename routes call it before the renamed detector's first
push can create the new file.

Only the ``server_json_file`` labelset source has an on-disk
representation that benefits from a rename move; other source types
(remote APIs, etc.) are skipped silently.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def detect_pending_labelset_move(
    labelset_source: dict[str, Any] | None,
    *,
    detector_id: str,
    old_name: str,
    new_name: str,
) -> dict[str, str] | None:
    """Return ``{"old_path", "new_path"}`` if the rename orphans a labelset file.

    Returns ``None`` when:
      * no labelset source is configured,
      * the source is not the on-disk ``server_json_file`` plugin,
      * the resolved old and new paths are identical (template doesn't
        reference ``{detector_name}``),
      * the old file does not exist on disk, or
      * the new file already exists (so a move would clobber it).
    """
    if not labelset_source:
        return None
    if labelset_source.get("source_name") != "server_json_file":
        return None

    field_values = labelset_source.get("field_values") or {}
    from vtscore.labels.sources.server_json_file import resolve_filepath_for

    try:
        old_resolved = resolve_filepath_for(field_values, detector_id=detector_id, detector_name=old_name)
        new_resolved = resolve_filepath_for(field_values, detector_id=detector_id, detector_name=new_name)
    except ValueError:
        # Empty / malformed template - nothing to move.
        return None

    if old_resolved == new_resolved:
        return None

    old_path = Path(old_resolved)
    new_path = Path(new_resolved)
    if not old_path.is_file():
        return None
    if new_path.exists():
        # Don't propose a move that would overwrite an existing file.
        return None

    return {"old_path": str(old_path), "new_path": str(new_path)}


def move_labelset_file(old_path: str, new_path: str) -> bool:
    """Atomically rename *old_path* to *new_path*.

    Both paths are validated against the file-access base directory to
    prevent path traversal.  Returns ``True`` if the move happened,
    ``False`` if *old_path* doesn't exist (idempotent - a repeated click
    after a successful move is a no-op, not an error).

    Raises ``ValueError`` for invalid paths and ``FileExistsError`` if
    *new_path* already exists.
    """
    from vtscore.security.path_validation import (
        get_file_access_base_dir,
        validate_server_filepath,
    )

    base = get_file_access_base_dir()
    old_resolved = validate_server_filepath(old_path, base)
    new_resolved = validate_server_filepath(new_path, base)

    if not old_resolved.is_file():
        return False
    if new_resolved.exists():
        raise FileExistsError(f"Destination already exists: {new_resolved}")

    new_resolved.parent.mkdir(parents=True, exist_ok=True)
    os.replace(old_resolved, new_resolved)
    return True


def follow_rename_with_labelset_file(
    detector_id: str,
    *,
    old_name: str,
    new_name: str,
) -> dict[str, str] | None:
    """Move a detector's on-disk labelset source file to follow its rename.

    Resolves the detector's labelset source (see
    :func:`vtscore.labels.binding.resolve_labelset_source`), and when the
    rename orphans its file (:func:`detect_pending_labelset_move`) moves it to
    the new name's path.  Call it **before** writing the renamed detector:
    that write schedules a push to the new path, and a push that lands first
    leaves the old file behind (the move never overwrites).

    Returns the ``{"old_path", "new_path"}`` pair that moved, or ``None`` when
    nothing needed to (or could) move; a failed move is logged, not raised.
    """
    from vtscore.detectors.registry import get_detector  # noqa: PLC0415
    from vtscore.labels.binding import resolve_labelset_source  # noqa: PLC0415

    resolved = resolve_labelset_source(get_detector(detector_id), detector_id=detector_id)
    if resolved is None:
        return None
    pending = detect_pending_labelset_move(
        {"source_name": resolved.source_name, "field_values": resolved.field_values},
        detector_id=detector_id,
        old_name=old_name,
        new_name=new_name,
    )
    if pending is None:
        return None
    try:
        if move_labelset_file(pending["old_path"], pending["new_path"]):
            logger.info("moved labelset file %s -> %s after a rename", pending["old_path"], pending["new_path"])
            return pending
    except (OSError, ValueError) as exc:
        logger.warning("Could not move labelset file %s after a rename: %s", pending["old_path"], exc)
    return None
