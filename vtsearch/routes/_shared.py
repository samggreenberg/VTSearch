"""Shared helpers for Flask route handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from flask import jsonify, request

if TYPE_CHECKING:
    from vtsearch.utils.registry import PluginBase

import vtsearch.utils.paths as _paths

logger = logging.getLogger(__name__)


def get_json_safe() -> dict:
    """Parse the request body as JSON, returning ``{}`` on missing or invalid input.

    Unlike :func:`get_json_or_400`, this never returns an error tuple — it
    silently falls back to an empty dict.  Use this when the JSON body is
    entirely optional (e.g. endpoints that accept both GET and POST).
    """
    return request.get_json(force=True, silent=True) or {}


def get_plugin_or_404(get_fn, list_fn, name: str, type_label: str):
    """Look up a plugin by *name*, returning a 404 response on failure.

    Args:
        get_fn: Callable that takes a name and returns the plugin or ``None``.
        list_fn: Callable that returns an iterable of all known plugins.
        name: The plugin name to look up.
        type_label: Human-readable label for the plugin type (e.g.
            ``"exporter"``, ``"label importer"``).

    Returns:
        ``(plugin, None)`` on success, or ``(None, (response, 404))`` on
        failure — the caller returns the error tuple directly.
    """
    plugin = get_fn(name)
    if plugin is not None:
        return plugin, None
    known = [p.name for p in list_fn()]
    return None, (
        jsonify({"error": f"Unknown {type_label} '{name}'. Available: {known}"}),
        404,
    )


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
    if data is None or not isinstance(data, dict):
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
        body = get_json_safe()
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
    except Exception as exc:
        logger.exception("%s.%s() failed: %s", type(plugin).__name__, method, exc)
        verb = method.replace("_", " ").capitalize()
        return None, (jsonify({"error": f"{verb} failed: {exc}"}), 500)
    return result, None


def get_embedder_for_medias(media_dict: dict):
    """Return the appropriate embedder for the given medias, or ``None``.

    Looks up the ``"embedder"`` name stored on the first media entry; falls
    back to the default embedder for the detected ``"type"``.
    """
    if not media_dict:
        return None
    first = next(iter(media_dict.values()))
    embedder_name = first.get("embedder", "")
    media_type = first.get("type", "audio")

    from vtsearch.media import embedders_for_type, get_embedder  # noqa: PLC0415

    if embedder_name:
        try:
            return get_embedder(embedder_name)
        except KeyError:
            pass

    avail = embedders_for_type(media_type)
    return avail[0] if avail else None


def get_request_field(key: str, has_file_fields: bool) -> str:
    """Read a single pass-through parameter from form data or JSON body."""
    if has_file_fields:
        return request.form.get(key, "")
    return get_json_safe().get(key, "")


def format_exception_detail(exc: BaseException) -> str:
    """Return ``"ExcType: first line"`` (or ``"ExcType"`` when the message is empty).

    Used in 500 response bodies so the UI can distinguish failure kinds
    (e.g. ``MemoryError`` from ``RuntimeError: embedder X not loaded``)
    without exposing a multi-line traceback.
    """
    text = str(exc)
    first = text.splitlines()[0].strip() if text else ""
    return f"{type(exc).__name__}: {first}" if first else type(exc).__name__


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
