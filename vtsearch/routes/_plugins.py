"""Plugin-facing route helpers: lookup, field options, argument validation.

Every plugin family (dataset / label / seed / datasource importers, results
exporters, converters) is driven from the HTTP layer through this one set of
helpers: resolve the plugin by name, resolve one dynamic field's option list,
validate a request body against the plugin's declared ``fields``, and call a
plugin method with the error envelope the API contract expects.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from flask import request
from flask_smorest import abort
from marshmallow import ValidationError

from vtsearch.errors import error_response
from vtsearch.routes._http import format_exception_detail

if TYPE_CHECKING:
    from vtscore.plugins import PluginBase

logger = logging.getLogger(__name__)


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
        failure; the caller returns the error tuple directly.
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


def _normalise_option(option: Any) -> dict[str, str]:
    """Coerce a ``get_field_options`` entry to a ``{"value", "label"}`` dict.

    Accepts both shapes a plugin may return (see
    :data:`vtscore.plugins.FieldOption`):

    - a plain string ``"foo"`` → ``{"value": "foo", "label": "foo"}`` (the
      value is shown verbatim as its own label);
    - a ``(value, label)`` 2-tuple/list ``("id", "Name")`` →
      ``{"value": "id", "label": "Name"}`` (the option submits the opaque
      ``value`` while displaying the friendly ``label``).

    Any other iterable is coerced via ``str`` on each part; a stray
    single-element pair falls back to using the value as its own label.
    """
    if isinstance(option, (tuple, list)):
        value = str(option[0]) if len(option) > 0 else ""
        label = str(option[1]) if len(option) > 1 else value
        return {"value": value, "label": label}
    text = str(option)
    return {"value": text, "label": text}


def plugin_field_options(plugin: PluginBase, body: dict) -> dict:
    """Resolve one ``dynamic_options`` field's option list for *plugin*.

    The body of every plugin family's options route (dataset importers,
    label importers, seed importers, datasource importers, results
    exporters): validate that the named field exists and is dynamic, call
    the plugin's ``get_field_options(field_key, current_values)`` with the
    supplied snapshot of form values, and coerce the result into the
    ``{"options": [{"value", "label"}, ...]}`` response shape.

    Args:
        plugin: The already-resolved plugin instance.
        body: The parsed ``ImporterFieldOptionsRequestSchema`` body
            (``field_key`` plus a ``values`` snapshot).

    Aborts:
        400 for an unknown or non-dynamic ``field_key``; 501 when the
        plugin does not implement the hook; 502 for any other plugin
        error (network failure, auth error) so the frontend can show the
        message inline; 500 when the hook returns a non-list.
    """
    field_key = body["field_key"].strip()
    values = body.get("values") or {}

    field = next((f for f in plugin.fields if f.key == field_key), None)
    if field is None:
        abort(400, message=f"Unknown field: {field_key!r}")
    if not getattr(field, "dynamic_options", False):
        abort(400, message=f"Field {field_key!r} is not dynamic")

    try:
        options = plugin.get_field_options(field_key, values)
    except NotImplementedError as exc:
        abort(501, message=str(exc) or "Importer does not implement get_field_options")
    except Exception as exc:  # noqa: BLE001 (surface remote-service errors verbatim)
        abort(502, message=str(exc) or type(exc).__name__)

    if not isinstance(options, list):
        abort(500, message="get_field_options must return a list")
    return {"options": [_normalise_option(o) for o in options]}


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
    ``errors`` envelope, matching the validation behaviour of routes
    using ``@blp.arguments(...)``.  Keys the schema doesn't recognise
    are dropped (the schema uses ``Meta.unknown = "exclude"``); pass
    *extra_keys* to allow specific extra body fields through to the
    returned dict (e.g. ``"converters"``, ``"clipper"``, ``"name"``).

    Parameters
    ----------
    plugin:
        Any plugin instance with a :attr:`fields` declaration.
    file_mode:
        How to surface file uploads.  Both modes satisfy the
        :class:`~vtscore.plugins.uploads.UploadedFile` protocol.
        ``"filestorage"`` (default) keeps the Werkzeug
        :class:`~werkzeug.datastructures.FileStorage` object; used by
        label-importer routes where the file is consumed synchronously.
        ``"bytesio"`` wraps the upload bytes in a
        :class:`~vtscore.plugins.uploads.BytesIOUploadedFile` carrying
        the original filename; used by dataset-importer routes that
        hand the file off to a background thread (the request context,
        and the underlying ``FileStorage``, are torn down before the
        thread reads).
    extra_keys:
        Pass-through keys whose values should ride along on the
        returned dict if present in the request body.  Each is copied
        verbatim; type / shape validation is the route handler's
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

    # Field-type-driven normalization: strip text, substitute declared
    # template vars, run validate_url / validate_server_filepath on
    # url / server_path fields.  See vtscore/plugins/normalize.py.
    from vtscore.plugins.normalize import normalize_field_values  # noqa: PLC0415

    try:
        normalize_field_values(plugin, validated)
    except ValueError as exc:
        abort(400, message=str(exc))

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
    from vtscore.plugins.uploads import BytesIOUploadedFile  # noqa: PLC0415

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
            validated[f.key] = BytesIOUploadedFile(storage.read(), storage.filename or "")
        else:
            validated[f.key] = storage

    if missing_files:
        abort(
            422,
            message="Validation error",
            errors={"json": {k: ["Missing data for required field."] for k in missing_files}},
        )


def validate_exporter_field_values(plugin: PluginBase, field_values: dict) -> dict:
    """Validate a nested ``field_values`` dict against *plugin*'s schema.

    Used by exporter routes whose request body is shaped as
    ``{"exporter_name": ..., "field_values": {...}}`` rather than the
    flat plugin-arg shape that :func:`validate_plugin_args` handles.
    Runs the plugin's per-field marshmallow schema for presence /
    type checks, then the shared
    :func:`~vtscore.plugins.normalize.normalize_field_values` pass for
    strip, template substitution, and URL / server-path validation.

    Schema-level rejects (missing required field, invalid select value)
    abort 422 with the standard ``errors`` envelope; URL / path
    validation failures abort 400.  Returns the validated + normalized
    dict on success.
    """
    from vtscore.plugins.normalize import normalize_field_values  # noqa: PLC0415
    from vtscore.plugins.schema import get_plugin_arg_schema  # noqa: PLC0415

    schema = get_plugin_arg_schema(plugin)
    try:
        loaded = schema.load(field_values)
    except ValidationError as exc:
        abort(422, message="Validation error", errors={"json": exc.messages})
    validated: dict = loaded if isinstance(loaded, dict) else {}

    try:
        normalize_field_values(plugin, validated)
    except ValueError as exc:
        abort(400, message=str(exc))

    return validated


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
