"""Shared helpers for Flask route handlers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from flask import jsonify, request

if TYPE_CHECKING:
    from vtsearch.utils.registry import PluginBase

import vtsearch.utils.paths as _paths


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
        return jsonify({"error": "Invalid request body"}), 400
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400
    return data


# ---------------------------------------------------------------------------
# Plugin field extraction helpers
# ---------------------------------------------------------------------------


def extract_plugin_fields(plugin: PluginBase) -> dict:
    """Build a ``field_values`` dict from the current Flask request.

    Handles both ``multipart/form-data`` (when the plugin has ``"file"``
    fields) and JSON request bodies.  File fields are populated from
    ``request.files``; non-file fields come from ``request.form`` or the
    JSON body, falling back to the field's default value.
    """
    has_file_fields = any(f.field_type == "file" for f in plugin.fields)
    field_values: dict = {}

    if has_file_fields:
        for f in plugin.fields:
            if f.field_type == "file":
                field_values[f.key] = request.files.get(f.key)
            else:
                field_values[f.key] = request.form.get(f.key, f.default if f.default is not None else "")
    else:
        body = request.get_json(force=True, silent=True) or {}
        for f in plugin.fields:
            field_values[f.key] = body.get(f.key, f.default if f.default is not None else "")

    return field_values


def validate_required_fields(plugin: PluginBase, field_values: dict) -> tuple | None:
    """Check that all required non-file fields have non-empty values.

    Returns a ``(response, 400)`` tuple on failure, or ``None`` if all
    required fields are present.
    """
    missing = [
        f.key
        for f in plugin.fields
        if f.required and f.field_type != "file" and not str(field_values.get(f.key, "")).strip()
    ]
    if missing:
        return (
            jsonify({"error": f"Missing required field(s): {missing}", "missing_fields": missing}),
            400,
        )
    return None


def validate_filepath_field(field_values: dict) -> tuple | None:
    """Validate the ``filepath`` field against path traversal attacks.

    Returns a ``(response, 400)`` tuple on failure, or ``None`` if the
    path is valid or absent.
    """
    if "filepath" in field_values and str(field_values["filepath"]).strip():
        try:
            _paths.validate_server_filepath(str(field_values["filepath"]), base_dir=_paths.get_file_access_base_dir())
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
    return None


def run_plugin_or_error(plugin: PluginBase, method: str, *args):
    """Call a plugin method, catching ValueError and unexpected exceptions.

    Returns ``(result, None)`` on success or ``(None, (response, status))``
    on failure.
    """
    try:
        result = getattr(plugin, method)(*args)
    except ValueError as exc:
        return None, (jsonify({"error": str(exc)}), 400)
    except Exception as exc:  # pragma: no cover
        verb = method.replace("_", " ").capitalize()
        return None, (jsonify({"error": f"{verb} failed: {exc}"}), 500)
    return result, None


def get_request_field(key: str, has_file_fields: bool) -> str:
    """Read a single pass-through parameter from form data or JSON body."""
    if has_file_fields:
        return request.form.get(key, "")
    return (request.get_json(force=True, silent=True) or {}).get(key, "")


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
