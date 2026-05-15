"""Shared helpers for the dataset route blueprints.

These were inlined into ``crud.py`` before the split; they now live here so
``listings``, ``status``, ``staging``, and ``load`` can share them without
the modules importing from each other.
"""

import io
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
    from vtsearch.media import get_by_folder_name, normalize_type_id

    try:
        return get_by_folder_name(value).type_id
    except KeyError:
        return normalize_type_id(value)


def _extract_importer_fields(importer):
    """Build field_values for a dataset importer from the current request.

    Unlike :func:`extract_plugin_fields`, this reads file contents into
    :class:`io.BytesIO` so they remain valid after the Flask request context
    ends (required for background-thread execution).

    Returns ``(field_values, None)`` on success, or ``(None, error_tuple)``
    when a required field is missing.
    """
    file_keys = {f.key for f in importer.fields if f.field_type == "file"}

    field_values: dict = {}
    if file_keys:
        for key in file_keys:
            if key not in request.files:
                return None, (jsonify({"error": f"Missing file field: {key!r}"}), 400)
            file_bytes = io.BytesIO(request.files[key].read())
            file_bytes.name = request.files[key].filename
            field_values[key] = file_bytes
        for f in importer.fields:
            if f.field_type != "file":
                field_values[f.key] = request.form.get(f.key, f.default)
        dataset_name = (request.form.get("dataset_name") or "").strip()
    else:
        body = request.get_json(force=True) or {}
        for f in importer.fields:
            if f.key not in body and f.required:
                return None, (jsonify({"error": f"Missing required field: {f.key!r}"}), 400)
            field_values[f.key] = body.get(f.key, f.default)
        dataset_name = str(body.get("dataset_name") or "").strip()

    if dataset_name:
        field_values["dataset_name"] = dataset_name

    return field_values, None


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
