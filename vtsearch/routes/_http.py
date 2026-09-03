"""Request-body parsing and response-field formatting.

The small, plugin-agnostic HTTP odds and ends every route family reaches for:
two JSON body parsers that differ only in how they fail, the exception-detail
string that 500 bodies carry, and the mtime string a listing response shows.
"""

from __future__ import annotations

from flask import request

from vtsearch.errors import error_response


def get_json_safe() -> dict:
    """Parse the request body as JSON, returning ``{}`` on missing or invalid input.

    Unlike :func:`get_json_or_400`, this never returns an error tuple;
    it silently falls back to an empty dict.  Use this when the JSON body is
    entirely optional (e.g. endpoints that accept both GET and POST).
    """
    return request.get_json(force=True, silent=True) or {}


def get_json_or_400():
    """Parse the request body as JSON, returning a 400 response on failure.

    Returns:
        The parsed JSON data (usually a dict) on success, or a
        ``(response, 400)`` tuple that can be returned directly from a
        Flask view when the body is missing or unparseable.

    Usage in a route::

        data = get_json_or_400()
        if not isinstance(data, dict):
            return data  # it's already a (jsonify(...), 400) tuple
    """
    try:
        data = request.get_json(force=True)
    except Exception:
        return error_response("Invalid request body", 400)
    if data is None or not isinstance(data, dict):
        return error_response("Invalid request body", 400)
    return data


def _scrub_server_paths(text: str) -> str:
    """Rewrite absolute paths under the server data dir to be relative to it.

    A raw OS error (e.g. ``[Errno 36] File name too long: '/srv/app/data/
    detectors/x.json.tmp'``) carries the absolute server path; surfacing it
    verbatim in a 500 body leaks the deployment's on-disk layout.  Strip the
    prefix up to the data dir's parent so the message keeps the useful tail
    (``data/detectors/x.json.tmp``) without the mount point.
    """
    import os

    from vtscore.config import DATA_DIR

    prefix = f"{DATA_DIR.parent}{os.sep}"
    return text.replace(prefix, "")


def format_exception_detail(exc: BaseException) -> str:
    """Return ``"ExcType: first line"`` (or ``"ExcType"`` when the message is empty).

    Used in 500 response bodies so the UI can distinguish failure kinds
    (e.g. ``MemoryError`` from ``RuntimeError: embedder X not loaded``)
    without exposing a multi-line traceback.  Absolute server paths are
    scrubbed to their data-dir-relative tail so a raw OS error can't leak the
    deployment's directory layout.
    """
    text = str(exc)
    first = text.splitlines()[0].strip() if text else ""
    detail = f"{type(exc).__name__}: {first}" if first else type(exc).__name__
    return _scrub_server_paths(detail)


def format_mtime(entry) -> str:
    """Return the file/directory modification time as an ISO-8601 string.

    *entry* should be a :class:`pathlib.Path`.  Returns ``""`` on error.
    """
    import datetime

    try:
        ts = entry.stat().st_mtime
        return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except OSError:
        return ""
