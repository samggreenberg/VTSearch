"""Flask routes for the server file browser.

Provides a generic file-browser API so the frontend can let users navigate
the server filesystem and pick files instead of having to type paths
by hand.

Migrated to ``flask_smorest`` so the route is described in
``/api/openapi.json``. Schema-level
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

    In single-user / no-auth mode this is the filesystem root (``/``) so the
    lone trusted user can navigate to any folder on the server, matching the
    unrestricted server-path import validation.  In multi-user mode it is the
    current user's data directory, confining the browser to that subtree.
    """
    base = _paths.get_file_access_base_dir()
    if base is None:
        return Path("/")
    return base


def _display_path(entry: Path, root: Path) -> str:
    """Return the path to expose for *entry* under *root*.

    Confinement always uses ``relative_to(root)``; the *displayed* form
    keeps the leading slash when the browse root is the filesystem root
    (single-user / no-auth mode) so the path the user picks is a usable
    absolute path. Otherwise the bare relative form (e.g.
    ``exp/mlucio/walkingtour.txt``) is handed to the importer/exporter,
    whose path validation resolves relative paths against the process CWD;
    the leading ``/`` would be silently dropped and the wrong file opened.

    For a confined per-user root we keep the relative form so the server's
    absolute filesystem layout never leaks into responses.
    """
    rel = str(entry.relative_to(root))
    if root == Path(root.anchor):  # filesystem root, e.g. "/"
        return "/" + rel if rel != "." else ""
    return rel


def _parse_allowed_exts(extensions_param: str) -> set[str] | None:
    """Parse the comma-separated ``extensions`` query param.

    Returns a normalized set of lowercase, dot-prefixed extensions, or
    ``None`` when no extension filter was supplied (param empty).
    """
    if not extensions_param:
        return None
    allowed_exts: set[str] = set()
    for ext in extensions_param.split(","):
        ext = ext.strip().lower()
        if ext and not ext.startswith("."):
            ext = "." + ext
        if ext:
            allowed_exts.add(ext)
    return allowed_exts


def _entry_to_listing(entry: Path, root: Path, allowed_exts: set[str] | None) -> tuple[str, dict] | None:
    """Classify *entry* under *root* and build its listing dict.

    Returns ``("dir", dict)`` for a directory, ``("file", dict)`` for a
    file, or ``None`` when the entry should be skipped (hidden name,
    out-of-root symlink, or a file filtered out by *allowed_exts*).
    """
    if entry.name.startswith("."):
        return None
    # is_dir() / is_file() / stat() follow symlinks by default, so an
    # in-root symlink pointing outside root would otherwise be listed
    # like an ordinary entry and leak the target's name/size/mtime.
    # Resolve symlinks up front and skip any that escape root.
    if entry.is_symlink():
        try:
            entry.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            return None
    rel = _display_path(entry, root)
    if entry.is_dir():
        return "dir", {"name": entry.name, "path": rel, "modified_at": format_mtime(entry)}
    if entry.is_file():
        if allowed_exts is not None and entry.suffix.lower() not in allowed_exts:
            return None
        try:
            size = entry.stat().st_size
        except OSError:
            size = 0
        return "file", {
            "name": entry.name,
            "path": rel,
            "size_bytes": size,
            "modified_at": format_mtime(entry),
        }
    return None


@file_browser_bp.route("/api/browse")
@file_browser_bp.arguments(BrowseQuerySchema, location="query")
@file_browser_bp.response(200, BrowseResponseSchema)
@file_browser_bp.alt_response(400, description="Invalid path (traversal blocked).")
@file_browser_bp.alt_response(403, description="Permission denied reading the requested directory.")
@file_browser_bp.alt_response(404, description="Directory not found within the allowed root.")
def browse(query: dict):
    """List directories and files at a relative path.

    Returns a directories+files listing with names, relative paths,
    modification times, and (for files) sizes in bytes. The server's
    absolute root is intentionally omitted from the response.
    """
    subpath = query["path"].strip()
    extensions_param = query["extensions"].strip()

    root = _get_browse_root().resolve()

    allowed_exts = _parse_allowed_exts(extensions_param)

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
        listing = _entry_to_listing(entry, root, allowed_exts)
        if listing is None:
            continue
        kind, item = listing
        if kind == "dir":
            directories.append(item)
        else:
            files.append(item)

    current_path = _display_path(target, root) if target != root else ""

    return {
        "directories": directories,
        "files": files,
        "current_path": current_path,
    }
