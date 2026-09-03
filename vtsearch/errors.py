"""Global JSON error handlers and the single API error envelope.

Every JSON error the API emits -- whether it comes from a
``flask_smorest.abort()`` inside a route, from one of the global
``@app.errorhandler`` functions below, or from a route helper returning an
error tuple -- carries **one** shape::

    {
      "code": 404,                     # int HTTP status (flask-smorest)
      "status": "Not Found",           # HTTP status name  (flask-smorest)
      "message": "Detector 'x' not found",
      "request_id": "ab12cd34...",     # ours; correlates the error with the logs
      "errors": {...},                 # marshmallow validation errors, when any
      "detail": "RuntimeError: ...",   # ours, on 500s
      "error_code": "dataset_not_loaded",   # ours, when a slug is useful
      ...                              # any extra kwargs passed to abort()
    }

That was not always true.  A hand-rolled ``{"error", "detail", "request_id"}``
envelope used to coexist with flask-smorest's ``{"code", "status", "message",
"errors"}``, and a third flavour (a bare ``jsonify({"error": ...})``) carried no
``request_id`` at all.  The Angular client had to read both spellings -- an
``apiErrorMessage`` helper imported by seventeen components existed for no other
reason -- while the ~350 ``abort()`` sites, the overwhelming majority of the
API's errors, were the ones *without* a ``request_id``.
:class:`VTSearchApi` folds our additions into flask-smorest's payload instead,
so there is one envelope, documented once in the OpenAPI spec, and every API
error is quotable in a bug report.

The module also holds the ``@app.errorhandler`` functions that used to live
inline in ``app.py``: JSON 404/405 for ``/api/`` paths, the dataset/detector
not-loaded -> 409 mappings, the request-missing-context -> 400 mapping, and
the catch-all uncaught-exception -> 500.

``app.py`` calls :func:`register_error_handlers` to wire them onto its
module-level ``app``. Registration is explicit (rather than via
``@app.errorhandler`` decorators at import time) so the order the handlers
are registered in is preserved and obvious.
"""

from __future__ import annotations

import logging
from typing import Any

import marshmallow as ma
from flask import g, jsonify
from flask_smorest import Api
from flask_smorest.error_handler import ErrorSchema as _SmorestErrorSchema
from werkzeug.exceptions import HTTPException, MethodNotAllowed, NotFound
from werkzeug.http import HTTP_STATUS_CODES

from vtscore.state.core import (
    DatasetNotLoadedError,
    DetectorNotLoadedError,
    RequestMissingContextError,
)

#: ``abort()`` kwargs the framework consumes itself: the first four are read by
#: flask-smorest's handler, and ``schema`` / ``exc`` are webargs internals it
#: attaches on a validation failure (the live marshmallow ``Schema`` and the
#: ``ValidationError``).  Everything else a route passes (``available=[...]``,
#: ``dataset_id=...``) is surfaced verbatim as a top-level field -- what the
#: hand-rolled envelope's ``**extra`` used to do.
_SMOREST_RESERVED = frozenset({"message", "errors", "messages", "headers", "schema", "exc"})


def _json_safe(value: Any) -> bool:
    """True when *value* is a JSON primitive or a container built only of them.

    A belt-and-braces guard on the ``abort()`` kwarg pass-through: a future
    webargs/smorest release attaching a new non-serializable internal would
    otherwise 500 the very handler meant to render the error.  Cheap because
    it only ever runs on an error path.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


class ErrorSchema(_SmorestErrorSchema):
    """flask-smorest's error schema plus the fields VTSearch adds.

    Documents the envelope in ``/api/openapi.json`` (as the ``Error``
    component, keeping the name flask-smorest's own schema had) so the
    generated client and the snapshot at ``frontend/openapi.json`` describe
    what the app actually sends.  Like its base class it is never used to
    *dump* a payload -- :meth:`VTSearchApi.handle_http_exception` and
    :func:`error_payload` build the dict -- it exists purely for the spec.

    ``unknown = INCLUDE`` renders as ``additionalProperties: true``, which is
    the honest description: a route's ``abort()`` may add fields of its own
    (``available``, ``missing_fields``, ``dataset_id``) that no fixed schema
    can enumerate.
    """

    class Meta(_SmorestErrorSchema.Meta):
        unknown = ma.INCLUDE

    request_id = ma.fields.String(metadata={"description": "Correlates the error with the server logs"})
    detail = ma.fields.String(metadata={"description": "Exception type and first line, on 500s"})
    error_code = ma.fields.String(metadata={"description": "Machine-readable slug, e.g. 'dataset_not_loaded'"})


class VTSearchApi(Api):
    """``flask_smorest.Api`` whose error payload carries ``request_id``.

    flask-smorest's own handler renders ``{code, status, message, errors}``
    and drops every other ``abort()`` kwarg on the floor.  This override adds
    the three things the retired hand-rolled envelope had and smorest's does
    not:

    * ``request_id``, pulled from :data:`flask.g` (set by ``before_request``),
      so a user can quote it in a bug report and an operator can grep for it.
    * a ``message`` fallback, so a bare ``abort(404)`` or a werkzeug-raised
      ``NotFound`` still names the status instead of omitting the key.  That
      is what lets :func:`_handle_404` delegate here and stop discarding every
      ``abort(404, message=...)`` body.
    * pass-through of unreserved ``abort()`` kwargs as top-level fields.
    """

    ERROR_SCHEMA = ErrorSchema

    def handle_http_exception(self, error):
        payload, code, headers = super().handle_http_exception(error)
        if not payload.get("message"):
            payload["message"] = error.name
        rid = getattr(g, "request_id", None)
        if rid:
            payload["request_id"] = rid
        for key, value in (getattr(error, "data", None) or {}).items():
            if key not in _SMOREST_RESERVED and key not in payload and _json_safe(value):
                payload[key] = value
        return payload, code, headers


def error_payload(message: str, status: int, detail: Any | None = None, **extra: Any) -> dict[str, Any]:
    """Build the envelope described in this module's docstring, as a dict.

    Used by the error handlers below -- which hold an exception rather than a
    live request they could ``abort()`` out of -- and by the handful of route
    helpers that return an error tuple.  Routes themselves should call
    ``flask_smorest.abort(status, message=..., **extra)``; it lands on
    :meth:`VTSearchApi.handle_http_exception` and produces the same shape.

    ``request_id`` is omitted outside a request context (background threads),
    where :data:`flask.g` has none.
    """
    body: dict[str, Any] = {
        "code": status,
        "status": HTTP_STATUS_CODES.get(status, "Unknown Error"),
        "message": message,
    }
    rid = getattr(g, "request_id", None)
    if rid:
        body["request_id"] = rid
    if detail is not None:
        body["detail"] = detail
    body.update(extra)
    return body


def error_response(message: str, status: int, detail: Any | None = None, **extra: Any):
    """:func:`error_payload` as the ``(response, status)`` tuple a view returns."""
    return jsonify(error_payload(message, status, detail, **extra)), status


def _smorest_api(app):
    """Return the app's :class:`VTSearchApi`, or ``None`` before it is wired.

    The 404/405 handlers delegate to it rather than re-deriving the envelope,
    so a ``NotFound`` raised by ``abort(404, message=...)`` renders exactly as
    it would have without our more-specific handler in the way.
    """
    apis = (app.extensions.get("flask-smorest") or {}).get("apis") or {}
    for entry in apis.values():
        return entry.get("ext_obj")
    return None


def _delegate_to_smorest(exc):
    """Render *exc* through the flask-smorest handler, falling back to ours."""
    from flask import current_app

    api = _smorest_api(current_app)
    if api is not None:
        return api.handle_http_exception(exc)
    data = getattr(exc, "data", None) or {}
    return error_response(data.get("message") or exc.name, exc.code or 500)


def _handle_404(exc):
    """JSON 404 for ``/api/`` paths, preserving the aborting route's message.

    Werkzeug's default 404 renders as HTML, which is awful for an SPA. The
    frontend ``ErrorService`` needs JSON to show a useful banner with the
    request_id. Non-API paths fall through to the SPA's catch-all route
    (which serves index.html for client-side routing).

    Flask resolves the most specific exception class first, so registering
    anything for ``NotFound`` takes every 404 away from flask-smorest's
    ``HTTPException`` handler -- including the ~85 ``abort(404, message=...)``
    calls whose message rides on ``exc.data``.  This handler used to render
    ``exc.name`` and nothing else, so every one of those messages ("Dataset
    file missing for 'x'", "File not found: <name>", "Job not found") was
    discarded before the client saw it.  It now decides only *whether* the
    path is ours and hands the rendering back to smorest.
    """
    from flask import request as _req

    if not _req.path.startswith("/api/"):
        return exc
    return _delegate_to_smorest(exc)


def _handle_405(exc):
    """JSON 405 for wrong-method requests on ``/api/`` paths. See _handle_404."""
    from flask import request as _req

    if not _req.path.startswith("/api/"):
        return exc
    return _delegate_to_smorest(exc)


def _handle_dataset_not_loaded(exc):
    """Return a JSON 409 when ``X-Dataset-Id`` names an unloaded dataset.

    Raised by the Flask resolver when a route handler touches the
    dataset proxies and the header doesn't resolve to a loaded context.
    Replaces the silent fallback to the empty context that returned 200
    with stale data (logical-bug-audit H16). The frontend can
    distinguish 409 + ``error_code="dataset_not_loaded"`` from a generic
    500 and offer a load action.
    """
    return error_response(
        "Dataset is not loaded",
        409,
        dataset_id=exc.dataset_id,
        error_code="dataset_not_loaded",
    )


def _handle_detector_not_loaded(exc):
    """Detector counterpart of :func:`_handle_dataset_not_loaded` (H16/H34)."""
    return error_response(
        "Detector is not loaded",
        409,
        detector_id=exc.detector_id,
        error_code="detector_not_loaded",
    )


def _handle_request_missing_context(exc):
    """Convert ``RequestMissingContextError`` into a clean 400.

    Raised by the frozen ``_RequestMissingDatasetContext`` /
    ``_RequestMissingDetectorContext`` sentinels when a mutation endpoint
    was hit without an ``X-Dataset-Id`` / ``X-Detector-Id`` header and no
    thread-local pinned context exists. The unloaded-id case is handled
    separately by :func:`_handle_dataset_not_loaded` /
    :func:`_handle_detector_not_loaded` (409 with a specific code). See
    logical-bug-audit H13.
    """
    from flask import request as _req

    if not _req.path.startswith("/api/"):
        raise exc
    return error_response(str(exc), 400)


def _handle_uncaught_exception(exc):
    """Return a standardized JSON 500 for any uncaught exception on /api/.

    Without this, an unhandled exception in a route renders Flask's HTML
    debug page (in dev) or a plain 500 (in prod), neither of which carries the
    request_id the user needs to file a bug. ``HTTPException`` is
    excluded so flask-smorest's own handler (and the 404/405 handlers
    above) keep their semantics.
    """
    if isinstance(exc, HTTPException):
        return exc
    from flask import request as _req

    from vtsearch.routes._http import format_exception_detail

    logging.getLogger(__name__).exception("Unhandled exception on %s %s", _req.method, _req.path)
    if not _req.path.startswith("/api/"):
        raise exc
    return error_response("Internal server error", 500, detail=format_exception_detail(exc))


def register_error_handlers(app) -> None:
    """Register all global JSON error handlers on ``app``.

    Registration order is preserved from the original inline
    ``@app.errorhandler`` sequence in ``app.py``.
    """
    app.errorhandler(NotFound)(_handle_404)
    app.errorhandler(MethodNotAllowed)(_handle_405)
    app.errorhandler(DatasetNotLoadedError)(_handle_dataset_not_loaded)
    app.errorhandler(DetectorNotLoadedError)(_handle_detector_not_loaded)
    app.errorhandler(RequestMissingContextError)(_handle_request_missing_context)
    app.errorhandler(Exception)(_handle_uncaught_exception)
