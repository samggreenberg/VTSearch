"""Blueprint for main application routes."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, current_app, jsonify, send_from_directory

from vtsearch import __version__

main_bp = Blueprint("main", __name__)


@main_bp.route("/api/version")
def version() -> Response:
    """Return the app version (UTC timestamp of the last dev->main merge)."""
    return jsonify({"version": __version__})


@main_bp.route("/openapi.json")
def openapi_json() -> Response:
    """Return the auto-generated OpenAPI 3.0 spec for this Flask app."""
    from vtsearch.openapi import generate_openapi_spec

    spec = generate_openapi_spec(current_app, version=__version__)
    return jsonify(spec)


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

    Paths under ``/api/`` are excluded — unmatched API routes should
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
