"""Flask routes for the server file browser.

Provides a generic file-browser API so the frontend can let users navigate
the server filesystem and pick files instead of having to type paths
by hand.

Migrated to ``flask_smorest`` so the route is described in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``. Schema-level
failures (e.g. unparseable query params) surface as 422 with the
standard ``errors`` envelope; handler-level rejects (path traversal,
permission denied) keep their HTTP codes (400 / 403) with the standard
``message`` envelope. 404s are intercepted by the app-level
``NotFound`` errorhandler in ``app.py`` and keep the legacy
``{"error": "Not Found", "request_id": ...}`` shape.

Endpoints
---------
GET  /api/browse
    List directories and files at a given path within the allowed base.
"""

from __future__ import annotations

from pathlib import Path

from flask_smorest import Blueprint, abort

from vtsearch.routes._shared import format_mtime
from vtsearch.schemas.file_browser import BrowseQuerySchema, BrowseResponseSchema

import vtscore.security.path_validation as _paths

file_browser_bp = Blueprint(
    "file_browser",
    __name__,
    description="Server-side file browser for picking files from the user's allowed root.",
)


def _get_browse_root() -> Path:
    """Return the root directory users are allowed to browse.

    In single-user mode this is the first entry of
    :data:`vtscore.config.SERVER_ROOTS` (which defaults to
    ``Path.cwd()`` when the env var is unset).  In multi-user mode it is
    the current user's data directory.
    """
    base = _paths.get_file_access_base_dir()
    if base is None:
        from vtscore.config import SERVER_ROOTS  # noqa: PLC0415

        return SERVER_ROOTS[0]
    return base


@file_browser_bp.route("/api/browse")
@file_browser_bp.arguments(BrowseQuerySchema, location="query")
@file_browser_bp.response(200, BrowseResponseSchema)
@file_browser_bp.alt_response(400, description="Invalid path (traversal blocked).")
@file_browser_bp.alt_response(403, description="Permission denied reading the requested directory.")
@file_browser_bp.alt_response(404, description="Directory not found within the allowed root.")
def browse(query: dict):  # noqa: C901
    """List directories and files at a relative path.

    Returns a directories+files listing with names, relative paths,
    modification times, and (for files) sizes in bytes. The server's
    absolute root is intentionally omitted from the response.
    """
    subpath = query["path"].strip()
    extensions_param = query["extensions"].strip()

    root = _get_browse_root().resolve()

    allowed_exts: set[str] | None = None
    if extensions_param:
        allowed_exts = set()
        for ext in extensions_param.split(","):
            ext = ext.strip().lower()
            if ext and not ext.startswith("."):
                ext = "." + ext
            if ext:
                allowed_exts.add(ext)

    if subpath:
        target = (root / subpath).resolve()
    else:
        target = root

    try:
        target.relative_to(root)
    except ValueError:
        abort(400, message="Invalid path")

    if not target.is_dir():
        abort(404, message="Directory not found")

    directories: list[dict] = []
    files: list[dict] = []

    try:
        entries = sorted(target.iterdir())
    except PermissionError:
        abort(403, message="Permission denied")

    for entry in entries:
        if entry.name.startswith("."):
            continue
        # is_dir() / is_file() / stat() follow symlinks by default, so an
        # in-root symlink pointing outside root would otherwise be listed
        # like an ordinary entry and leak the target's name/size/mtime.
        # Resolve symlinks up front and skip any that escape root.
        if entry.is_symlink():
            try:
                entry.resolve(strict=True).relative_to(root)
            except (OSError, ValueError):
                continue
        rel = str(entry.relative_to(root))
        if entry.is_dir():
            directories.append({"name": entry.name, "path": rel, "modified_at": format_mtime(entry)})
        elif entry.is_file():
            if allowed_exts is not None and entry.suffix.lower() not in allowed_exts:
                continue
            try:
                size = entry.stat().st_size
            except OSError:
                size = 0
            files.append({"name": entry.name, "path": rel, "size_bytes": size, "modified_at": format_mtime(entry)})

    current_path = str(target.relative_to(root)) if target != root else ""

    return {
        "directories": directories,
        "files": files,
        "current_path": current_path,
    }
