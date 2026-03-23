"""Flask routes for the server file browser.

Provides a generic file-browser API so the frontend can let users navigate
the server filesystem and pick files — instead of having to type paths
by hand.

Endpoints
---------
GET  /api/browse
    List directories and files at a given path within the allowed base.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from vtsearch.routes.helpers import format_mtime

import vtsearch.utils.paths as _paths

file_browser_bp = Blueprint("file_browser", __name__)


def _get_browse_root() -> Path:
    """Return the root directory users are allowed to browse.

    In single-user mode this is ``Path.cwd()``; in multi-user mode it is
    the current user's data directory.
    """
    base = _paths.get_file_access_base_dir()
    if base is None:
        return Path.cwd()
    return base


@file_browser_bp.route("/api/browse")
def browse():
    """List directories and files at a relative path.

    Query parameters:

    * ``path`` — relative path within the allowed root (default ``""``).
    * ``extensions`` — comma-separated list of file extensions to show,
      e.g. ``".csv,.json"``.  When omitted all files are listed.

    Returns::

        {
            "directories": [{"name": "subdir", "path": "subdir"}, ...],
            "files": [
                {"name": "my_labels.csv", "path": "my_labels.csv", "size_bytes": 1234},
                ...
            ],
            "current_path": "data/labels",
            "root": "/home/user/project"
        }
    """
    subpath = request.args.get("path", "").strip()
    extensions_param = request.args.get("extensions", "").strip()

    root = _get_browse_root().resolve()

    # Parse optional extension filter
    allowed_exts: set[str] | None = None
    if extensions_param:
        allowed_exts = set()
        for ext in extensions_param.split(","):
            ext = ext.strip().lower()
            if ext and not ext.startswith("."):
                ext = "." + ext
            if ext:
                allowed_exts.add(ext)

    # Resolve target, preventing traversal
    if subpath:
        target = (root / subpath).resolve()
    else:
        target = root

    try:
        target.relative_to(root)
    except ValueError:
        return jsonify({"error": "Invalid path"}), 400

    if not target.is_dir():
        return jsonify({"error": "Directory not found"}), 404

    directories: list[dict] = []
    files: list[dict] = []

    try:
        entries = sorted(target.iterdir())
    except PermissionError:
        return jsonify({"error": "Permission denied"}), 403

    for entry in entries:
        if entry.name.startswith("."):
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

    return jsonify({
        "directories": directories,
        "files": files,
        "current_path": current_path,
    })
