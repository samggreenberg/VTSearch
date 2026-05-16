"""Filepath template expansion shared by file-based exporters.

Supports three placeholders, sanitised so user-controlled values cannot
escape the directory implied by the admin-configured template:

* ``{YYYYMMDD-HHMMSS}`` – current UTC time (e.g. ``20260516-143022``);
  defaulted into the path so consecutive runs do not silently overwrite
  one another.
* ``{detector_name}`` – the active :class:`DetectorContext`'s ``name``.
* ``{username}`` – the current request user
  (see :func:`vtsearch.auth.get_current_user`).
"""

from __future__ import annotations

from datetime import datetime, timezone


def resolve_export_filepath(filepath: str) -> str:
    """Return *filepath* with supported ``{...}`` placeholders substituted."""
    if "{YYYYMMDD-HHMMSS}" in filepath:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        filepath = filepath.replace("{YYYYMMDD-HHMMSS}", stamp)

    if "{detector_name}" in filepath:
        from vtsearch.security.path_validation import sanitize_template_value
        from vtsearch.state.core import get_active_detector_context

        ctx = get_active_detector_context()
        filepath = filepath.replace("{detector_name}", sanitize_template_value(ctx.name))

    if "{username}" in filepath:
        from vtsearch.auth import get_current_user
        from vtsearch.security.path_validation import sanitize_template_value

        filepath = filepath.replace("{username}", sanitize_template_value(get_current_user() or "default"))

    return filepath
