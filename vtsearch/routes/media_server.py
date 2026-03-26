"""Blueprint for server media file management and example-sort routes."""

from pathlib import Path

from flask import Blueprint, jsonify, request

from vtsearch.config import DATA_DIR
from vtsearch.routes.helpers import get_json_or_400
from vtsearch.utils import snapshot_medias

media_server_bp = Blueprint("media_server", __name__)

#: Default directory for server-side example media files (single-user fallback).
SERVER_MEDIA_DIR = DATA_DIR / "example_media"


def _get_server_media_dir() -> Path:
    """Return the per-user server media directory.

    In multi-user mode each user gets their own ``example_media/``
    subdirectory inside their data dir, preventing cross-user file access.
    """
    from vtsearch.auth import DefaultLoginProvider, get_login_provider, get_user_data_dir

    provider = get_login_provider()
    if isinstance(provider, DefaultLoginProvider):
        return SERVER_MEDIA_DIR
    return get_user_data_dir() / "example_media"


@media_server_bp.route("/api/server-media-files/upload", methods=["POST"])
def upload_server_media_file():
    """Upload a media file to data/example_media/ and return its filename."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    import uuid

    suffix = Path(file.filename).suffix or ".bin"
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    media_dir = _get_server_media_dir()
    media_dir.mkdir(parents=True, exist_ok=True)
    dest = media_dir / safe_name
    file.save(dest)

    return jsonify({"filename": safe_name, "original_name": file.filename}), 201


@media_server_bp.route("/api/server-media-files", methods=["GET"])
def list_server_media_files():
    """List media files saved on the server in the user's example_media/ dir."""
    media_dir = _get_server_media_dir()
    if not media_dir.is_dir():
        return jsonify({"files": []})

    files = []
    for p in sorted(media_dir.iterdir()):
        if p.is_file() and not p.name.startswith("."):
            files.append(
                {
                    "name": p.stem,
                    "filename": p.name,
                    "size_bytes": p.stat().st_size,
                }
            )
    return jsonify({"files": files})


@media_server_bp.route("/api/example-sort-server", methods=["POST"])
def example_sort_server():
    """Sort medias by similarity to a server-side media file."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    filename = data.get("filename", "").strip()
    if not filename:
        return jsonify({"error": "filename is required"}), 400

    media_dir = _get_server_media_dir()
    file_path = media_dir / filename

    # Ensure path doesn't escape the server media directory
    try:
        file_path.resolve().relative_to(media_dir.resolve())
    except ValueError:
        return jsonify({"error": "Invalid filename"}), 400

    if not file_path.is_file():
        return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        from vtsearch.routes.sorting import _example_sort_from_path

        results, thresh = _example_sort_from_path(file_path)
        return jsonify({"results": results, "threshold": thresh})
    except Exception:
        import logging

        logging.getLogger(__name__).exception("example-sort-server failed")
        return jsonify({"error": "Example sort failed"}), 500


@media_server_bp.route("/api/example-sort-origin", methods=["POST"])
def example_sort_origin():
    """Sort medias by similarity to a media file resolved from an origin.

    Accepts a JSON body with ``origin`` (an origin dict as stored on medias)
    and ``key`` (the item key / relative path within the source).  The file
    is fetched via the :class:`~vtsearch.datasets.sources.base.MediaSource`
    abstraction, embedded, and used for cosine-similarity sorting.

    Example request::

        {
            "origin": {"importer": "folder", "params": {"path": "/data/sounds"}},
            "key": "subdir/audio123.wav"
        }
    """
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    origin = data.get("origin")
    if not isinstance(origin, dict):
        return jsonify({"error": "origin dict is required"}), 400

    key = data.get("key", "").strip()
    if not key:
        return jsonify({"error": "key is required"}), 400

    if not snapshot_medias():
        return jsonify({"error": "No medias loaded"}), 400

    from vtsearch.datasets.sources import get_source_for_origin

    source = get_source_for_origin(origin)
    if source is None:
        return jsonify({"error": f"No media source available for origin type: {origin.get('importer', '')}"}), 400

    try:
        file_path = source.fetch_item(key)
        if file_path is None:
            return jsonify({"error": f"File not found: {key}"}), 404

        from vtsearch.routes.sorting import _example_sort_from_path

        results, thresh = _example_sort_from_path(file_path)
        return jsonify({"results": results, "threshold": thresh})
    except Exception:
        import logging

        logging.getLogger(__name__).exception("example-sort-origin failed")
        return jsonify({"error": "Example sort failed"}), 500
    finally:
        source.cleanup()
