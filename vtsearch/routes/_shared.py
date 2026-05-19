"""Shared helpers for Flask route handlers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from flask import g, jsonify, request
from flask_smorest import abort
from marshmallow import ValidationError

if TYPE_CHECKING:
    from vtscore.plugins import PluginBase

import vtscore.security.path_validation as _paths

logger = logging.getLogger(__name__)


def error_response(error: str, status: int, detail: Any | None = None, **extra: Any):
    """Build a standardized JSON error response.

    Shape::

        {"error": "<short message>", "detail": <optional>, "request_id": "<id>", ...extra}

    ``request_id`` is pulled from ``flask.g`` when available, so clients can
    quote it in bug reports and operators can correlate it with structured
    logs. Additional keyword args become top-level fields (e.g.
    ``missing_fields=[...]``).
    """
    body: dict[str, Any] = {"error": error}
    if detail is not None:
        body["detail"] = detail
    rid = getattr(g, "request_id", None)
    if rid:
        body["request_id"] = rid
    body.update(extra)
    return jsonify(body), status


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
    return None, error_response(
        f"Unknown {type_label} '{name}'. Available: {known}",
        404,
        available=known,
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
        return error_response("Invalid request body", 400)
    if data is None or not isinstance(data, dict):
        return error_response("Invalid request body", 400)
    return data


# ---------------------------------------------------------------------------
# Plugin field extraction helpers
# ---------------------------------------------------------------------------


def validate_plugin_args(
    plugin: PluginBase,
    *,
    file_mode: str = "filestorage",
    extra_keys: tuple[str, ...] = (),
) -> dict:
    """Validate the request body against the plugin's per-field schema.

    Builds a marshmallow schema from the plugin's :attr:`fields`
    declaration (cached on the plugin instance) and runs the incoming
    request body through it.  Returns the validated ``field_values``
    dict on success, including file uploads keyed by their declared
    field name.

    Schema-level rejects (missing required field, invalid select value,
    unparseable number) surface as ``422`` with the standard
    ``errors`` envelope — matching the validation behaviour of routes
    using ``@blp.arguments(...)``.  Keys the schema doesn't recognise
    are dropped (the schema uses ``Meta.unknown = "exclude"``) — pass
    *extra_keys* to allow specific extra body fields through to the
    returned dict (e.g. ``"converters"``, ``"clipper"``, ``"name"``).

    Parameters
    ----------
    plugin:
        Any plugin instance with a :attr:`fields` declaration.
    file_mode:
        How to surface file uploads.  ``"filestorage"`` (default) keeps
        :class:`werkzeug.datastructures.FileStorage` objects — used by
        label-importer routes where the file is consumed synchronously.
        ``"bytesio"`` reads each upload into an in-memory
        :class:`io.BytesIO` carrying the original filename on its
        ``.name`` attribute — used by dataset-importer routes that hand
        the file off to a background thread (the request context, and
        the underlying ``FileStorage``, are torn down before the thread
        reads).
    extra_keys:
        Pass-through keys whose values should ride along on the
        returned dict if present in the request body.  Each is copied
        verbatim — type / shape validation is the route handler's
        responsibility.
    """
    from vtscore.plugins.schema import get_plugin_arg_schema  # noqa: PLC0415

    has_file_fields = any(f.field_type == "file" for f in plugin.fields)

    if has_file_fields:
        # ``request.form`` is a MultiDict; marshmallow handles plain dicts.
        body: dict = {k: v for k, v in request.form.items()}
    else:
        body = request.get_json(force=True, silent=True) or {}
        if not isinstance(body, dict):
            abort(400, message="Invalid request body")

    schema = get_plugin_arg_schema(plugin)
    try:
        # Schema.load() returns a wide list|dict|None per the marshmallow stubs;
        # our schema always emits a flat dict, so narrow explicitly.
        loaded = schema.load(body)
    except ValidationError as exc:
        abort(422, message="Validation error", errors={"json": exc.messages})
    validated: dict = loaded if isinstance(loaded, dict) else {}

    _populate_file_fields(plugin, validated, has_file_fields=has_file_fields, file_mode=file_mode)

    for key in extra_keys:
        if key in body:
            validated[key] = body[key]

    return validated


def _populate_file_fields(
    plugin: PluginBase,
    validated: dict,
    *,
    has_file_fields: bool,
    file_mode: str,
) -> None:
    """Read file uploads off the request and merge them into *validated*.

    Required file fields surface as a 422 with the standard ``errors``
    envelope (matching schema-level rejects); optional missing fields
    land in *validated* as ``None``.
    """
    import io  # noqa: PLC0415 — defer to avoid import cost when unused

    missing_files: list[str] = []
    for f in plugin.fields:
        if f.field_type != "file":
            continue
        storage = request.files.get(f.key) if has_file_fields else None
        if storage is None or not getattr(storage, "filename", None):
            if f.required:
                missing_files.append(f.key)
            else:
                validated[f.key] = None
            continue
        if file_mode == "bytesio":
            buf = io.BytesIO(storage.read())
            buf.name = storage.filename  # type: ignore[attr-defined]
            validated[f.key] = buf
        else:
            validated[f.key] = storage

    if missing_files:
        abort(
            422,
            message="Validation error",
            errors={"json": {k: ["Missing data for required field."] for k in missing_files}},
        )


def validate_filepath_field(field_values: dict) -> tuple | None:
    """Validate the ``filepath`` field against path traversal attacks.

    Returns a ``(response, 400)`` tuple on failure, or ``None`` if the
    path is valid or absent.
    """
    if "filepath" in field_values and str(field_values["filepath"]).strip():
        try:
            _paths.validate_server_filepath(str(field_values["filepath"]), base_dir=_paths.get_file_access_base_dir())
        except ValueError as exc:
            return error_response(str(exc), 400)
    return None


def run_plugin_or_error(plugin: PluginBase, method: str, *args):
    """Call a plugin method, catching ValueError and unexpected exceptions.

    Returns ``(result, None)`` on success or ``(None, (response, status))``
    on failure.
    """
    try:
        result = getattr(plugin, method)(*args)
    except ValueError as exc:
        return None, error_response(str(exc), 400)
    except Exception as exc:
        logger.exception("%s.%s() failed: %s", type(plugin).__name__, method, exc)
        verb = method.replace("_", " ").capitalize()
        return None, error_response(f"{verb} failed: {exc}", 500, detail=format_exception_detail(exc))
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

    from vtscore.media import embedders_for_type, get_embedder  # noqa: PLC0415

    if embedder_name:
        try:
            return get_embedder(embedder_name)
        except KeyError:
            pass

    avail = embedders_for_type(media_type)
    return avail[0] if avail else None


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
