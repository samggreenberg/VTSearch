"""Blueprint for server media file management and example-sort routes."""

import io
from pathlib import Path

from flask import Blueprint, jsonify, request, send_file

from vtsearch.config import DATA_DIR
from vtsearch.routes.helpers import get_json_or_400
from vtsearch.utils import snapshot_medias

media_server_bp = Blueprint("media_server", __name__)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
_AUDIO_EXTS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".aac"}
_VIDEO_EXTS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}


def _media_type_from_ext(suffix: str) -> str:
    s = suffix.lower()
    if s in _IMAGE_EXTS:
        return "image"
    if s in _AUDIO_EXTS:
        return "audio"
    if s in _VIDEO_EXTS:
        return "video"
    return ""


_IMAGE_MIMETYPES = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}

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
    """Upload a media file to data/example_media/ and return its filename.

    Optional form fields:

    * ``media_type`` — required when ``crop_params`` is present (``"audio"``
      or ``"image"``); identifies which bounded clipper to apply.
    * ``crop_params`` — JSON object with the user-cropped bounds.  When set,
      the file is cropped server-side before being saved, so the persisted
      example is the cropped sub-region (and downstream code that reads
      the saved file by name does not need any changes).
    """
    import json
    import uuid

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    suffix = Path(file.filename).suffix or ".bin"
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    media_dir = _get_server_media_dir()
    media_dir.mkdir(parents=True, exist_ok=True)
    dest = media_dir / safe_name
    file.save(dest)

    crop_raw = request.form.get("crop_params")
    if crop_raw:
        try:
            crop_params = json.loads(crop_raw)
        except (TypeError, ValueError):
            crop_params = None
        if isinstance(crop_params, dict):
            media_type = request.form.get("media_type", "").strip()
            try:
                from vtsearch.media.cropping import crop_file_bytes

                cropped = crop_file_bytes(dest, media_type, crop_params)
                dest.write_bytes(cropped)
            except (ValueError, FileNotFoundError):
                dest.unlink(missing_ok=True)
                return jsonify({"error": "Invalid crop_params for this media type"}), 400

    return jsonify({"filename": safe_name, "original_name": file.filename}), 201


@media_server_bp.route("/api/server-media-files/<path:filename>/thumbnail", methods=["GET"])
def server_media_file_thumbnail(filename: str):
    """Return a small image preview of an example media file.

    Used by the new-model dialog to show the user which example was selected.
    For images this serves the file bytes; for audio it returns a waveform PNG;
    for video it returns the middle-frame PNG.  Other media types return 404.
    """
    media_dir = _get_server_media_dir()
    file_path = media_dir / filename

    try:
        file_path.resolve().relative_to(media_dir.resolve())
    except ValueError:
        return jsonify({"error": "Invalid filename"}), 400

    if not file_path.is_file():
        return jsonify({"error": f"File not found: {filename}"}), 404

    suffix = file_path.suffix.lower()
    media_type = _media_type_from_ext(suffix)

    if media_type == "image":
        mimetype = _IMAGE_MIMETYPES.get(suffix, "image/jpeg")
        return send_file(
            io.BytesIO(file_path.read_bytes()),
            mimetype=mimetype,
            download_name=file_path.name,
        )

    if media_type == "audio":
        from vtsearch.media.audio.media_type import generate_waveform_thumbnail_from_file

        thumb = generate_waveform_thumbnail_from_file(file_path)
        if thumb is None:
            return jsonify({"error": "Could not generate audio thumbnail"}), 500
        return send_file(
            io.BytesIO(thumb),
            mimetype="image/png",
            download_name=f"{file_path.stem}_waveform.png",
        )

    if media_type == "video":
        from vtsearch.media.video.media_type import generate_video_thumbnail_from_file

        thumb = generate_video_thumbnail_from_file(file_path)
        if thumb is None:
            return jsonify({"error": "Could not generate video thumbnail"}), 500
        return send_file(
            io.BytesIO(thumb),
            mimetype="image/png",
            download_name=f"{file_path.stem}_frame.png",
        )

    return jsonify({"error": f"No thumbnail available for {suffix}"}), 404


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
    """Sort medias by similarity to a server-side media file.

    Optional ``crop_params`` body field carries a JSON-compatible dict with
    bounds for a user-cropped sub-region (audio: ``{"start", "end"}``;
    image: ``{"box": [...]}``).  When set, the file is cropped server-side
    before embedding.
    """
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

    crop_params = data.get("crop_params") if isinstance(data.get("crop_params"), dict) else None

    try:
        from vtsearch.routes.sorting import _apply_crop_or_keep, _example_sort_from_path

        if crop_params:
            # Crop into a temp file so the saved server-side example is unchanged.
            import uuid

            from vtsearch.config import DATA_DIR

            DATA_DIR.mkdir(exist_ok=True)
            tmp = DATA_DIR / f"temp_example_{uuid.uuid4().hex}{file_path.suffix or '.bin'}"
            tmp.write_bytes(file_path.read_bytes())
            try:
                _apply_crop_or_keep(tmp, crop_params)
                results, thresh = _example_sort_from_path(tmp)
            finally:
                tmp.unlink(missing_ok=True)
        else:
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
            "origin": {"importer": "server_folder", "params": {"path": "/data/sounds"}},
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

    crop_params = data.get("crop_params") if isinstance(data.get("crop_params"), dict) else None

    try:
        file_path = source.fetch_item(key)
        if file_path is None:
            return jsonify({"error": f"File not found: {key}"}), 404

        from vtsearch.routes.sorting import _apply_crop_or_keep, _example_sort_from_path

        if crop_params:
            import uuid

            from vtsearch.config import DATA_DIR

            DATA_DIR.mkdir(exist_ok=True)
            tmp = DATA_DIR / f"temp_example_{uuid.uuid4().hex}{file_path.suffix or '.bin'}"
            tmp.write_bytes(file_path.read_bytes())
            try:
                _apply_crop_or_keep(tmp, crop_params)
                results, thresh = _example_sort_from_path(tmp)
            finally:
                tmp.unlink(missing_ok=True)
        else:
            results, thresh = _example_sort_from_path(file_path)
        return jsonify({"results": results, "threshold": thresh})
    except Exception:
        import logging

        logging.getLogger(__name__).exception("example-sort-origin failed")
        return jsonify({"error": "Example sort failed"}), 500
    finally:
        source.cleanup()
