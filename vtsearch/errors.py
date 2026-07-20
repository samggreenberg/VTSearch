"""Global JSON error handlers for the VTSearch app.

Holds the ``@app.errorhandler`` functions that used to live inline in
``app.py``: JSON 404/405 for ``/api/`` paths, the dataset/detector
not-loaded → 409 mappings, the request-missing-context → 400 mapping, and
the catch-all uncaught-exception → 500.

``app.py`` calls :func:`register_error_handlers` to wire them onto its
module-level ``app``. Registration is explicit (rather than via
``@app.errorhandler`` decorators at import time) so the order the handlers
are registered in is preserved and obvious.
"""

import logging

from werkzeug.exceptions import MethodNotAllowed, NotFound

from vtscore.state.core import (
    DatasetNotLoadedError,
    DetectorNotLoadedError,
    RequestMissingContextError,
)


def _handle_404(exc):
    """JSON 404 for unknown ``/api/`` paths.

    Werkzeug's default 404 renders as HTML, which is awful for an SPA. The
    frontend ``ErrorService`` needs JSON to show a useful banner with the
    request_id. Non-API paths fall through to the SPA's catch-all route
    (which serves index.html for client-side routing).

    Scoped to ``NotFound`` specifically so flask-smorest's own
    ``HTTPException`` handler (which renders ``{message, errors}`` for
    marshmallow validation failures and custom ``abort()`` calls) keeps
    handling 400/422/etc.; Flask resolves the most specific exception
    class first.
    """
    from flask import request as _req

    from vtsearch.routes._shared import error_response

    if not _req.path.startswith("/api/"):
        return exc
    return error_response(exc.name, 404)


def _handle_405(exc):
    """JSON 405 for wrong-method requests on ``/api/`` paths. See _handle_404."""
    from flask import request as _req

    from vtsearch.routes._shared import error_response

    if not _req.path.startswith("/api/"):
        return exc
    return error_response(exc.name, 405)


def _handle_dataset_not_loaded(exc):
    """Return a JSON 409 when ``X-Dataset-Id`` names an unloaded dataset.

    Raised by the Flask resolver when a route handler touches the
    dataset proxies and the header doesn't resolve to a loaded context.
    Replaces the silent fallback to the empty context that returned 200
    with stale data (logical-bug-audit H16). The frontend can
    distinguish 409 + ``code="dataset_not_loaded"`` from a generic 500
    and offer a load action.
    """
    from vtsearch.routes._shared import error_response

    return error_response(
        "Dataset is not loaded",
        409,
        dataset_id=exc.dataset_id,
        code="dataset_not_loaded",
    )


def _handle_detector_not_loaded(exc):
    """Detector counterpart of :func:`_handle_dataset_not_loaded` (H16/H34)."""
    from vtsearch.routes._shared import error_response

    return error_response(
        "Detector is not loaded",
        409,
        detector_id=exc.detector_id,
        code="detector_not_loaded",
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

    from vtsearch.routes._shared import error_response

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
    from werkzeug.exceptions import HTTPException as _HTTPException

    if isinstance(exc, _HTTPException):
        return exc
    from flask import request as _req

    from vtsearch.routes._shared import error_response, format_exception_detail

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
