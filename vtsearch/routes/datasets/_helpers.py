"""Shared helpers for the dataset route blueprints.

These were inlined into ``crud.py`` before the split; they now live here so
``listings``, ``status``, ``staging``, and ``load`` can share them without
the modules importing from each other.
"""

import json
from pathlib import PurePosixPath
from typing import Any

from flask import jsonify, request

from vtsearch.routes._shared import get_json_safe


def _normalize_media_type_param(value: str) -> str:
    """Accept both ``type_id`` (``"image"``) and ``folder_import_name`` (``"images"``)."""
    value = value.strip()
    if not value:
        return ""
    from vtscore.media import get_by_folder_name, normalize_type_id

    try:
        return get_by_folder_name(value).type_id
    except KeyError:
        return normalize_type_id(value)


def _extract_clipper_params(has_file_fields: bool) -> tuple[dict | None, Any]:
    """Read optional ``clipper_params`` from form/JSON, returning (params, error_response).

    JSON requests carry the value as a dict directly.  Multipart form
    requests carry it as a JSON-encoded string (since form fields are
    flat).  Either form is accepted; anything else is a 400 error.
    Returns ``(None, None)`` when no value is present.
    """
    if has_file_fields:
        raw = request.form.get("clipper_params") or ""
        if not raw:
            return None, None
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError) as exc:
            return None, (jsonify({"error": f"Invalid clipper_params: {exc}"}), 400)
    else:
        parsed = get_json_safe().get("clipper_params")
        if parsed is None or parsed == "":
            return None, None

    if not isinstance(parsed, dict):
        return None, (jsonify({"error": "clipper_params must be a JSON object"}), 400)
    return parsed, None


def _safe_relative_upload_path(filename: str) -> PurePosixPath | None:
    """Return *filename* as a sanitised relative POSIX path, or ``None`` if unsafe.

    Browsers send each file's ``webkitRelativePath`` (or basename) as the
    multipart filename.  Reject anything absolute or that would escape the
    upload root via ``..`` segments.  Empty path components and "." are
    skipped.
    """
    if not filename:
        return None
    raw = filename.replace("\\", "/")
    if raw.startswith("/"):
        return None
    parts: list[str] = []
    for segment in raw.split("/"):
        if not segment or segment == ".":
            continue
        if segment == ".." or "\x00" in segment:
            return None
        parts.append(segment)
    if not parts:
        return None
    return PurePosixPath(*parts)
