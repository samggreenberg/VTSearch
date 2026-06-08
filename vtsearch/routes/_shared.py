"""Shared helpers for Flask route handlers."""

from __future__ import annotations

import hashlib
import io
import logging
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from flask import g, jsonify, make_response, request, send_file
from flask_smorest import abort
from marshmallow import ValidationError

if TYPE_CHECKING:
    from vtscore.plugins import PluginBase

logger = logging.getLogger(__name__)


def image_thumbnail_response(image_bytes: bytes, fallback_mimetype: str, download_name: str):
    """Build a cached, downscaled-image thumbnail response from ``image_bytes``.

    Shared by every route that serves a small image tile (the media grid/list,
    saved detector labels, server-media examples).  The bytes are run through
    :func:`vtscore.media.image.thumbnail.make_image_thumbnail` so a gallery of
    many high-resolution items never forces the browser to decode every
    full-size bitmap at once; undecodable sources fall back to the original
    bytes.  An ``ETag`` fingerprints the *source* bytes so the browser reuses
    one thumbnail per item across scrolls and zoom levels, and conditional
    requests short-circuit to a 304 without regenerating the thumbnail.
    """
    from vtscore.media.image.thumbnail import make_image_thumbnail  # noqa: PLC0415

    etag = hashlib.md5(image_bytes).hexdigest()
    if etag in request.if_none_match:
        resp = make_response("", 304)
        resp.set_etag(etag)
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp

    result = make_image_thumbnail(image_bytes)
    thumb, mimetype = result if result is not None else (image_bytes, fallback_mimetype)

    resp = make_response(send_file(io.BytesIO(thumb), mimetype=mimetype, download_name=download_name))
    resp.set_etag(etag)
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


def _has_context_id(header_name: str, query_param: str) -> bool:
    """Return True iff *header_name* or *query_param* identifies a context.

    Matches the precedence applied by ``before_request`` in ``app.py``:
    header first, then query-param fallback for browser-native requests
    (``<img src>`` / ``<audio src>`` / ``<video src>``) that bypass
    Angular's HttpClient interceptor.
    """
    return bool(request.headers.get(header_name) or request.args.get(query_param))


def require_detector_header(fn: Callable) -> Callable:
    """Route decorator: reject 400 if no ``X-Detector-Id`` is identified.

    Closes logical-bug-audit H34: vote-mutating endpoints used to silently
    write to whatever ``get_active_detector_context()`` resolved to when
    the client dropped the header. The frozen ``_request_missing_detector_context``
    sentinel catches the case where both the header *and* the thread-local
    are absent, but if any future code path sets the thread-local on a
    Flask request thread, a header-absent request would land on a stale
    detector. This guard rejects header-absent requests *before* the
    resolver chain runs, regardless of thread-local state (defence in
    depth).

    Apply to any endpoint that mutates ``DetectorContext`` state
    (``good_votes`` / ``bad_votes`` / ``label_history`` / ``vote_*`` /
    ``find_initial_labels`` / ``click_counter``). Pure reads, registry
    listings, and dashboard endpoints don't need it.
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _has_context_id("X-Detector-Id", "detector_id"):
            abort(
                400,
                message=("X-Detector-Id header (or ?detector_id= query param) is required for this endpoint."),
            )
        return fn(*args, **kwargs)

    return wrapper


def require_dataset_header(fn: Callable) -> Callable:
    """Route decorator: reject 400 if no ``X-Dataset-Id`` is identified.

    Sister of :func:`require_detector_header`; closes the dataset-side
    analog of H34. Apply to any endpoint that mutates ``DatasetContext``
    state (``medias`` insertions, ``diversity_tree`` rebuilds) or whose
    correctness depends on knowing which dataset's cid-keyed votes are
    being touched (label imports, fill-from-sort).
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _has_context_id("X-Dataset-Id", "dataset_id"):
            abort(
                400,
                message=("X-Dataset-Id header (or ?dataset_id= query param) is required for this endpoint."),
            )
        return fn(*args, **kwargs)

    return wrapper


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

    Unlike :func:`get_json_or_400`, this never returns an error tuple;
    it silently falls back to an empty dict.  Use this when the JSON body is
    entirely optional (e.g. endpoints that accept both GET and POST).
    """
    return request.get_json(force=True, silent=True) or {}


def register_plugin_typed_routes(
    blueprint,
    *,
    list_plugins,
    path_template: str,
    endpoint_prefix: str,
    delegate,
    plugin_kwarg: str = "importer_name",
    extra_keys: tuple[str, ...] = (),
    extra_decorators: tuple[Callable, ...] = (),
    skip_file_plugins: bool = True,
) -> None:
    """Register per-plugin typed routes for a plugin-field endpoint.

    Each plugin-field route (e.g. ``POST /api/dataset/import/<importer_name>``)
    has a single parameterized URL rule whose request body shape depends
    on the named plugin's :attr:`fields` declaration.  This helper
    additionally registers one *static* URL rule per known plugin (e.g.
    ``POST /api/dataset/import/server_folder``,
    ``POST /api/dataset/import/pickle``) decorated with
    ``@blp.arguments(SchemaClass)``, where ``SchemaClass`` is built from
    that plugin's declared fields.  Werkzeug matches static URL rules
    before dynamic ones, so requests for known plugins land on the typed
    route (and surface in the OpenAPI spec with real per-field types);
    requests for unknown plugin names fall through to the parameterized
    fallback (preserving the legacy 404 error message that names the
    unknown plugin).

    Plugins with ``file`` fields are skipped by default (the fallback
    handles them as multipart/form-data); ``skip_file_plugins=False``
    will register them too but the spec won't describe the file body.

    Parameters
    ----------
    blueprint:
        The :class:`flask_smorest.Blueprint` to register on.
    list_plugins:
        Callable returning the list of plugins for this family (e.g.
        :func:`vtscore.datasets.list_importers`).  Called once at
        registration time.
    path_template:
        URL template containing ``{plugin_name}`` (e.g.
        ``"/api/dataset/import/{plugin_name}"``).  Other ``<...>``-style
        Flask path variables can appear elsewhere in the template (e.g.
        ``"/api/detectors/<name>/import-labels/{plugin_name}"``).
    endpoint_prefix:
        Flask endpoint name prefix; the per-plugin endpoint is
        ``f"{endpoint_prefix}__{plugin.name}"``.  Also used as the
        ``route_id`` argument to :func:`make_plugin_route_schema` for
        namespacing the generated schema class.
    delegate:
        The parameterized fallback view function (e.g. ``import_dataset``).
        The typed wrapper calls it with the plugin's name passed via
        *plugin_kwarg*; the fallback re-validates the request via
        :func:`validate_plugin_args`, so the typed wrapper itself does
        nothing with the validated body (its only purpose is to attach
        ``@arguments`` for spec generation).
    plugin_kwarg:
        Name of the keyword argument *delegate* expects to receive the
        plugin name as (matches the original ``<importer_name>``-style
        path variable name).
    extra_keys:
        Pass-through keys the route accepts alongside the plugin-declared
        fields (e.g. ``("source_specs", "clipper", "embedder",
        "dataset_name")``).  Declared as :class:`fields.Raw` on the
        generated schema so they appear in the OpenAPI spec.
    extra_decorators:
        Additional view-function decorators to apply *outside* of
        ``@arguments`` (e.g. :func:`require_dataset_header`).  Applied in
        order; the outermost decorator is last.
    skip_file_plugins:
        When True (default), plugins with at least one ``file``-typed
        field are not registered as typed routes (they continue to use
        the parameterized fallback).
    """
    from vtscore.plugins.schema import make_plugin_route_schema  # noqa: PLC0415

    for plugin in list_plugins():
        if skip_file_plugins and any(f.field_type == "file" for f in plugin.fields):
            continue
        schema_cls = make_plugin_route_schema(
            plugin,
            extra_keys=extra_keys,
            route_id=endpoint_prefix,
        )
        path = path_template.format(plugin_name=plugin.name)
        endpoint = f"{endpoint_prefix}__{plugin.name}"
        plugin_name = plugin.name

        def _make_view(_plugin_name: str):
            def _typed_view(body, **kwargs):  # noqa: ARG001 (body is validated for spec; delegate re-reads request)
                kwargs[plugin_kwarg] = _plugin_name
                return delegate(**kwargs)

            _typed_view.__name__ = endpoint
            _typed_view.__doc__ = delegate.__doc__
            return _typed_view

        view = _make_view(plugin_name)
        view = blueprint.arguments(schema_cls)(view)
        for dec in extra_decorators:
            view = dec(view)
        blueprint.route(path, methods=["POST"], endpoint=endpoint)(view)


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


def get_embedder_for_medias(media_dict: dict):
    """Return the appropriate embedder for the given medias, or ``None``.

    Looks up the ``"embedder"`` name stored on the first media entry; falls
    back to the default embedder for the detected ``"type"``.
    """
    if not media_dict:
        return None
    first = next(iter(media_dict.values()))
    embedder_name = first.get("embedder", "")
    media_type = first.get("media_type", "audio")

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
