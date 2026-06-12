"""Filepath template expansion shared by file-based exporters.

Supports these placeholders, sanitised so user-controlled values cannot
escape the directory implied by the admin-configured template:

* ``{YYYYMMDD-HHMMSS}`` – current UTC time (e.g. ``20260516-143022``);
  defaulted into the path so consecutive runs do not silently overwrite
  one another.
* ``{YYYYMMDD}`` / ``{YYYY}`` / ``{MM}`` / ``{DD}`` – current UTC date
  parts, so a scheduled (e.g. daily) Auto-Find can write to a path named
  after today's date (``results_{YYYY}.{MM}.{DD}.csv``).
* ``{detector_name}`` – the active :class:`DetectorContext`'s ``name``.
* ``{username}`` – the current request user
  (see :func:`vtsearch.auth.get_current_user`).
"""

from __future__ import annotations

from datetime import datetime, timezone

#: strftime format per date/time placeholder. The closing brace makes each
#: placeholder unambiguous (``{YYYY}`` never matches inside ``{YYYYMMDD}``),
#: so substitution order is irrelevant.
_DATETIME_PLACEHOLDERS = {
    "{YYYYMMDD-HHMMSS}": "%Y%m%d-%H%M%S",
    "{YYYYMMDD}": "%Y%m%d",
    "{YYYY}": "%Y",
    "{MM}": "%m",
    "{DD}": "%d",
}


def resolve_export_filepath(filepath: str) -> str:
    """Return *filepath* with supported ``{...}`` placeholders substituted."""
    now = datetime.now(timezone.utc)
    for placeholder, fmt in _DATETIME_PLACEHOLDERS.items():
        if placeholder in filepath:
            filepath = filepath.replace(placeholder, now.strftime(fmt))

    if "{detector_name}" in filepath:
        from vtscore.security.path_validation import sanitize_template_value
        from vtscore.state.core import get_active_detector_context

        ctx = get_active_detector_context()
        filepath = filepath.replace("{detector_name}", sanitize_template_value(ctx.name))

    if "{username}" in filepath:
        from vtsearch.auth import get_current_user
        from vtscore.security.path_validation import sanitize_template_value

        filepath = filepath.replace("{username}", sanitize_template_value(get_current_user() or "default"))

    return filepath
