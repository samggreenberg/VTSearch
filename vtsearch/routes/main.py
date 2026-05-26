"""Blueprint for main application routes.

``/api/version`` is the only JSON API exposed here and is described via
``flask_smorest`` decorators so it appears in ``/api/openapi.json``.
The other routes serve static files / the Angular SPA shell and stay
plain Flask routes (no schema, no spec presence) - they share the same
``flask_smorest.Blueprint`` object since that class is a regular Flask
``Blueprint`` subclass and only decorated routes contribute to the
spec.
"""

from __future__ import annotations

from pathlib import Path

from flask import Response, current_app, send_from_directory
from flask_smorest import Blueprint

from vtsearch import __version__
from vtsearch.schemas.main import VersionSchema

main_bp = Blueprint(
    "main",
    __name__,
    description="App version and SPA / static-file serving.",
)


@main_bp.route("/api/version")
@main_bp.response(200, VersionSchema)
def version() -> dict:
    """Return the app version (UTC timestamp of the last dev->main merge)."""
    return {"version": __version__}


def _static_dir() -> Path:
    """Return the static directory path."""
    return Path(current_app.root_path) / "static"


def _serve_angular_index() -> Response:
    """Serve the Angular SPA index.html."""
    return send_from_directory(str(_static_dir()), "index.html")


@main_bp.route("/")
def index() -> Response:
    """Serve the Angular SPA entry point."""
    return _serve_angular_index()


@main_bp.route("/dashboard")
@main_bp.route("/label")
def angular_routes(**kwargs: object) -> Response:
    """Serve the Angular SPA for known client-side routes.

    Angular Router handles these paths on the client side; the server
    just needs to return ``index.html`` for all of them.
    """
    return _serve_angular_index()


@main_bp.route("/favicon.ico")
def favicon() -> tuple[str, int] | Response:
    """Serve the site favicon from the static directory.

    Returns:
        The ``favicon.ico`` file from ``static/`` if it exists,
        otherwise an empty ``(str, int)`` tuple with HTTP status 204.
    """
    static = _static_dir()
    if not (static / "favicon.ico").exists():
        return "", 204
    return send_from_directory(str(static), "favicon.ico", mimetype="image/x-icon")


@main_bp.route("/favicon-<variant>.ico")
def favicon_variant(variant: str) -> tuple[str, int] | Response:
    """Serve a favicon variant (smile, frown, surprised) from the static directory."""
    allowed = {"smile", "frown", "surprised"}
    if variant not in allowed:
        return "", 404
    static = _static_dir()
    filename = f"favicon-{variant}.ico"
    if not (static / filename).exists():
        return "", 204
    return send_from_directory(str(static), filename, mimetype="image/x-icon")


@main_bp.route("/logo.svg")
def logo() -> tuple[str, int] | Response:
    """Serve the site logo from the static directory.

    Returns:
        The ``logo.svg`` file from ``static/`` if it exists,
        otherwise an empty ``(str, int)`` tuple with HTTP status 204.
    """
    static = _static_dir()
    if not (static / "logo.svg").exists():
        return "", 204
    return send_from_directory(str(static), "logo.svg", mimetype="image/svg+xml")


@main_bp.route("/<path:path>")
def catch_all(path: str) -> Response:
    """Serve static files at root paths and fall back to Angular SPA.

    The Angular build output (main.js, polyfills.js, styles.css, etc.) is
    referenced from index.html with ``<base href="/">``, so the browser
    requests them at ``/main.js`` rather than ``/static/main.js``.  This
    route serves those files when they exist, and falls back to
    ``index.html`` for any other path so that Angular Router can handle
    client-side navigation.

    Paths under ``/api/`` are excluded - unmatched API routes should
    return 404, not the SPA page.
    """
    # Don't intercept API routes; let Flask return its default 404.
    if path.startswith("api/"):
        from flask import abort

        abort(404)
    static = _static_dir()
    # Resolve the candidate and ensure it stays within the static directory
    # to prevent path-traversal attacks (e.g. ../../etc/passwd).
    try:
        candidate = (static / path).resolve()
    except (OSError, ValueError):
        return _serve_angular_index()
    if candidate.is_file() and str(candidate).startswith(str(static.resolve())):
        return send_from_directory(str(static), path)
    return _serve_angular_index()
