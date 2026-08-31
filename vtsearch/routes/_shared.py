"""Shared helpers for Flask route handlers."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

from flask import g, jsonify, make_response, request, send_file
from flask_smorest import abort
from marshmallow import ValidationError
from werkzeug.exceptions import HTTPException

from vtscore.concurrency.progress import update_find_progress
from vtscore.utils.hashing import content_md5, new_md5

if TYPE_CHECKING:
    from vtscore.plugins import PluginBase

logger = logging.getLogger(__name__)


def _sniff_image_mimetype(data: bytes) -> str:
    """Best-effort image mimetype from magic bytes (PNG vs JPEG).

    Precomputed thumbnails are always either PNG (alpha / waveform / video
    frame) or JPEG (opaque image), so a two-way sniff is enough; anything
    unrecognised defaults to JPEG.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


def cached_thumbnail_response(thumb_bytes: bytes, download_name: str):
    """Serve already-final thumbnail bytes with an ``ETag`` and cache headers.

    Unlike :func:`image_thumbnail_response`, this does **no** decode/resize:
    the bytes are a thumbnail that was precomputed at ingest (the media dict's
    ``thumbnail_bytes``), so the request path just streams them.  The ``ETag``
    fingerprints the thumbnail bytes so the browser reuses one tile per item
    across scrolls and zoom levels, short-circuiting to a 304.
    """
    etag = content_md5(thumb_bytes)
    if etag in request.if_none_match:
        resp = make_response("", 304)
        resp.set_etag(etag)
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp

    resp = make_response(
        send_file(io.BytesIO(thumb_bytes), mimetype=_sniff_image_mimetype(thumb_bytes), download_name=download_name)
    )
    resp.set_etag(etag)
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


def image_thumbnail_response(
    image_bytes: bytes,
    fallback_mimetype: str,
    download_name: str,
    crop: object = None,
):
    """Build a cached, downscaled-image thumbnail response from ``image_bytes``.

    Shared by every route that serves a small image tile (the media grid/list,
    saved detector labels, server-media examples).  The bytes are run through
    :func:`vtscore.media.image.thumbnail.make_image_thumbnail` so a gallery of
    many high-resolution items never forces the browser to decode every
    full-size bitmap at once; undecodable sources fall back to the original
    bytes.  An ``ETag`` fingerprints the *source* bytes so the browser reuses
    one thumbnail per item across scrolls and zoom levels, and conditional
    requests short-circuit to a 304 without regenerating the thumbnail.

    When ``crop`` is a valid normalised ``(x0, y0, x1, y1)`` region (see
    :func:`vtscore.media.image.thumbnail.normalize_region_crop`), the thumbnail
    shows only that sub-region -- used so a region-voted item displays its crop
    rather than the whole frame.  The crop is folded into the ``ETag`` so a
    re-vote with a different box invalidates the cached tile.
    """
    from vtscore.media.image.thumbnail import (  # noqa: PLC0415
        make_image_thumbnail,
        normalize_region_crop,
    )

    crop_box = normalize_region_crop(crop) if crop is not None else None

    hasher = new_md5()
    hasher.update(image_bytes)
    if crop_box is not None:
        hasher.update(repr(crop_box).encode("ascii"))
    etag = hasher.hexdigest()
    if etag in request.if_none_match:
        resp = make_response("", 304)
        resp.set_etag(etag)
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp

    result = make_image_thumbnail(image_bytes, crop=crop_box)
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
    state (``medias`` insertions, ``coverage_atlas`` rebuilds) or whose
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


def find_idle() -> None:
    """Park the shared ``find_progress`` tracker at idle, clearing the step frame.

    The find-side sibling of ``vtsearch.routes.sorting._sort_idle``. Every
    scoring route (``/api/find``, ``/api/find-label``, ``/api/auto-detect``)
    reports through the one process-wide ``find_progress`` singleton and pushes
    it to every SSE client, so whichever route ran last owns leaving it parked
    at ``"idle"``.
    """
    update_find_progress("idle", "", step=None, total_steps=None)


@contextmanager
def find_idle_on_crash(recorder: Any = None) -> Iterator[None]:
    """Park ``find_progress`` at idle when the wrapped body dies unexpectedly.

    Every *anticipated* exit from a scoring route resets the tracker itself:
    ``_abort_find``, ``_abort_if_find_cancelled``, and the success path all end
    with an idle update. An unhandled exception (say an embedding-dimension
    mismatch raising ``RuntimeError`` mid-scoring) takes none of those paths, so
    without this guard the request 500s through the global error handler while
    the shared singleton stays at ``"running"`` on whatever step it died on —
    broadcast to every SSE client, and only cleared by the next Find.

    *recorder* (a :func:`vtscore.timing.record_task` handle, when the route runs
    one) is closed first, as a failed run: the idle update would otherwise trip
    its ``auto_finish`` hook and bank a crashed run's partial phase timings as a
    good cost sample.  ``abort()`` is left alone — it already parked the tracker
    and closed the recorder, and flask-smorest renders its envelope unchanged.
    """
    try:
        yield
    except HTTPException:
        raise
    except Exception:
        if recorder is not None:
            recorder.finish(ok=False)
        find_idle()
        raise


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


def windowed_sort_extras(results: list[dict], threshold: float | None) -> dict[str, Any]:
    """Register a full sorted ``results`` list and return the windowing extras.

    Stores *results* in the process-global :data:`sort_results_cache` keyed to
    the active (dataset, detector) pair and returns the extra fields a sort
    response carries so a client can page deeper without holding the whole list:

    - ``sort_token`` — opaque handle for ``GET /api/sort/page``; also the
      sort-generation token (a re-sort mints a new one).
    - ``total`` — full ranking length.
    - ``above_threshold`` — rows scoring at or above *threshold*.

    Additive: the caller still returns the full ``results`` today (the frontend
    windowed model lands in a later slice, see ``docs/plans/scalability.md``
    S3/S17/S19), so wiring this in never changes existing behaviour.
    """
    from vtscore.state.core import get_active_context, get_active_detector_context  # noqa: PLC0415
    from vtscore.state.sort_results_cache import count_above_threshold, sort_results_cache  # noqa: PLC0415

    dataset_id = getattr(get_active_context(), "dataset_id", "") or ""
    detector_id = getattr(get_active_detector_context(), "detector_id", "") or ""
    token = sort_results_cache.store(results, threshold, dataset_id=dataset_id, detector_id=detector_id)
    return {
        "sort_token": token,
        "total": len(results),
        "above_threshold": count_above_threshold(results, threshold),
    }


def windowed_sort_response(
    results: list[dict],
    threshold: float | None,
    acq_threshold: float | None = None,
) -> dict[str, Any]:
    """Build a sort-response body, windowing the transmitted ``results``.

    *acq_threshold* is the **acquisition** cut for a detector sort - the rank
    position Autopilot's Hard / New picks sample around, which since #2876 sits
    above the reporting ``threshold`` (see
    :func:`vtscore.state.core.detector_acquisition_threshold`).  Only the learned
    sort carries one; the text / example / label-file sorts have no detector
    behind them, so they leave it ``None`` and the client falls back to
    ``threshold``.  It is deliberately *not* fed to ``windowed_sort_extras``:
    ``above_threshold`` counts what the user is told matched, which is the
    reporting cut's job.

    Stores the full ranking (so ``/api/sort/page`` can serve any window) and
    returns ``{results, threshold, acq_threshold, sort_token, total,
    above_threshold, has_more_below}``.  Below :data:`SORT_WINDOW_THRESHOLD` the full list is
    transmitted unchanged and ``has_more_below`` is ``False`` — small / medium
    sorts behave exactly as before.  At or above it, only the initial head window
    rides the response and the client pages the rest.

    The threshold is read off the cache module at call time so tests can lower it
    via ``monkeypatch`` without generating tens of thousands of rows.
    """
    from vtscore.state import sort_results_cache as _cache_mod  # noqa: PLC0415

    extras = windowed_sort_extras(results, threshold)
    total = extras["total"]
    if total < _cache_mod.SORT_WINDOW_THRESHOLD:
        window = results
        has_more = False
    else:
        end = _cache_mod.initial_window_end(total, extras["above_threshold"])
        window = results[:end]
        has_more = end < total
    return {
        "results": window,
        "threshold": threshold,
        "acq_threshold": acq_threshold,
        "has_more_below": has_more,
        **extras,
    }


def get_embedder_for_medias(media_dict: dict):
    """Return the appropriate embedder for the given medias, or ``None``.

    Thin alias for :func:`vtscore.media.embedder_for_medias`, kept so route
    code can keep importing it from here.  The implementation moved to the
    library tier because the dataset-load pipeline needs it too, and
    reaching back into ``vtsearch.routes`` for it made a library code path
    hard-require Flask (issue #2931).
    """
    from vtscore.media import embedder_for_medias  # noqa: PLC0415

    return embedder_for_medias(media_dict)


#: Message used by both Semantic-lock guards below, so the API surfaces one
#: consistent explanation whichever route the request hit.
SEMANTIC_ONLY_MESSAGE = (
    "This server is locked to Semantic embedders (semantic_only). "
    "Patch Semantic and Structural embedders are unavailable here."
)


def abort_if_semantic_only_type(embedder_type: str) -> None:
    """Reject a non-Semantic detector *embedder_type* on a Semantic-locked server.

    The type is the detector's declared intent, so this is the one gate that
    keeps a hand-rolled ``POST /api/detectors`` (or a portable bundle carrying a
    Structural detector) from creating a detector this deployment can never
    run. Empty / ``"semantic"`` pass through untouched, as does every request
    when the lock is off.
    """
    if not embedder_type or embedder_type == "semantic":
        return
    from vtsearch.settings import get_effective_semantic_only  # noqa: PLC0415 - avoid import cycle

    if get_effective_semantic_only():
        abort(400, message=SEMANTIC_ONLY_MESSAGE)


def abort_if_semantic_only_embedders(embedder_names) -> None:
    """Reject patch / structural *embedder_names* on a Semantic-locked server.

    Guards the dataset-load routes, whose ``embedders`` trio arrives straight
    from the client: the pickers never offer a prototype embedder under the
    lock (``GET /api/embedders`` filters them out), so a request that names one
    is either stale or hand-rolled and should fail loudly rather than quietly
    binding a type the rest of the UI hides. Unknown names are left alone --
    they fail their own validation downstream.
    """
    names = [n for n in (embedder_names or ()) if n]
    if not names:
        return
    from vtsearch.settings import get_effective_semantic_only  # noqa: PLC0415 - avoid import cycle

    if not get_effective_semantic_only():
        return

    from vtscore.embedding.binding import embedder_type as _classify  # noqa: PLC0415

    offenders = sorted({n for n in names if _classify(n) in ("patch_semantic", "structural")})
    if offenders:
        abort(400, message=f"{SEMANTIC_ONLY_MESSAGE} Rejected: {', '.join(offenders)}.")


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
