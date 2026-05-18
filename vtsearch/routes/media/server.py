"""Blueprint for server media file management and example-sort routes.

Migrated to ``flask_smorest`` so the JSON-shaped routes appear in the
``/api/openapi.json`` spec. See ``docs/plans/openapi-schema.md``.

The thumbnail GET route serves binary bytes (or an error JSON) and only
declares its ``alt_response`` error codes; the success body is not
described. The upload POST route is multipart — its body is left
undescribed (no ``arguments`` decorator) but the success body is
declared via ``response``.
"""

import io
from pathlib import Path

from flask import request, send_file
from flask_smorest import Blueprint, abort
from werkzeug.exceptions import HTTPException

from vtsearch.config import DATA_DIR
from vtsearch.routes._shared import format_exception_detail
from vtsearch.schemas.media import (
    ExampleSortByIdRequestSchema,
    ExampleSortOriginRequestSchema,
    ExampleSortResponseSchema,
    ExampleSortServerRequestSchema,
    ServerMediaFromMediaIdRequestSchema,
    ServerMediaListResponseSchema,
    ServerMediaUploadResponseSchema,
)
from vtsearch.state import get_media, snapshot_medias

media_server_bp = Blueprint(
    "media_server",
    __name__,
    description="Server-side example media files and example-sort routes.",
)


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
@media_server_bp.response(201, ServerMediaUploadResponseSchema)
@media_server_bp.alt_response(
    400,
    description=(
        "Missing or malformed multipart body (no file / no filename / invalid crop_params for the given media type)."
    ),
)
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
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

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
                abort(400, message="Invalid crop_params for this media type")

    return {"filename": safe_name, "original_name": file.filename}


@media_server_bp.route("/api/server-media-files/<path:filename>/thumbnail", methods=["GET"])
@media_server_bp.alt_response(400, description="Filename escapes the example_media directory.")
@media_server_bp.alt_response(404, description="File not found, or media type has no thumbnail.")
@media_server_bp.alt_response(500, description="Thumbnail could not be generated.")
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
        abort(400, message="Invalid filename")

    if not file_path.is_file():
        abort(404, message=f"File not found: {filename}")

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
            abort(500, message="Could not generate audio thumbnail")
        return send_file(
            io.BytesIO(thumb),
            mimetype="image/png",
            download_name=f"{file_path.stem}_waveform.png",
        )

    if media_type == "video":
        from vtsearch.media.video.media_type import generate_video_thumbnail_from_file

        thumb = generate_video_thumbnail_from_file(file_path)
        if thumb is None:
            abort(500, message="Could not generate video thumbnail")
        return send_file(
            io.BytesIO(thumb),
            mimetype="image/png",
            download_name=f"{file_path.stem}_frame.png",
        )

    abort(404, message=f"No thumbnail available for {suffix}")


@media_server_bp.route("/api/server-media-files", methods=["GET"])
@media_server_bp.response(200, ServerMediaListResponseSchema)
def list_server_media_files():
    """List media files saved on the server in the user's example_media/ dir."""
    media_dir = _get_server_media_dir()
    if not media_dir.is_dir():
        return {"files": []}

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
    return {"files": files}


@media_server_bp.route("/api/example-sort-server", methods=["POST"])
@media_server_bp.arguments(ExampleSortServerRequestSchema)
@media_server_bp.response(200, ExampleSortResponseSchema)
@media_server_bp.alt_response(400, description="Filename escapes the example_media directory.")
@media_server_bp.alt_response(404, description="File not found.")
@media_server_bp.alt_response(500, description="Example sort failed.")
def example_sort_server(body: dict):
    """Sort medias by similarity to a server-side media file.

    Optional ``crop_params`` body field carries a JSON-compatible dict with
    bounds for a user-cropped sub-region (audio: ``{"start", "end"}``;
    image: ``{"box": [...]}``).  When set, the file is cropped server-side
    before embedding.
    """
    filename = body["filename"].strip()
    if not filename:
        abort(400, message="filename is required")

    media_dir = _get_server_media_dir()
    file_path = media_dir / filename

    # Ensure path doesn't escape the server media directory
    try:
        file_path.resolve().relative_to(media_dir.resolve())
    except ValueError:
        abort(400, message="Invalid filename")

    if not file_path.is_file():
        abort(404, message=f"File not found: {filename}")

    crop_params = body.get("crop_params") if isinstance(body.get("crop_params"), dict) else None

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
        return {"results": results, "threshold": thresh}
    except HTTPException:
        raise
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("example-sort-server failed")
        abort(500, message=f"Example sort failed: {format_exception_detail(exc)}")


@media_server_bp.route("/api/example-sort-origin", methods=["POST"])
@media_server_bp.arguments(ExampleSortOriginRequestSchema)
@media_server_bp.response(200, ExampleSortResponseSchema)
@media_server_bp.alt_response(
    400,
    description="No media source for the given origin, or no medias loaded.",
)
@media_server_bp.alt_response(404, description="File not found at the given key.")
@media_server_bp.alt_response(500, description="Example sort failed.")
def example_sort_origin(body: dict):
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
    origin = body["origin"]
    key = body["key"].strip()
    if not key:
        abort(400, message="key is required")

    if not snapshot_medias():
        abort(400, message="No medias loaded")

    from vtsearch.datasets.sources import get_source_for_origin

    source = get_source_for_origin(origin)
    if source is None:
        abort(400, message=f"No media source available for origin type: {origin.get('importer', '')}")

    crop_params = body.get("crop_params") if isinstance(body.get("crop_params"), dict) else None

    try:
        file_path = source.fetch_item(key)
        if file_path is None:
            abort(404, message=f"File not found: {key}")

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
        return {"results": results, "threshold": thresh}
    except HTTPException:
        raise
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("example-sort-origin failed")
        abort(500, message=f"Example sort failed: {format_exception_detail(exc)}")
    finally:
        source.cleanup()


def _media_extension(media: dict) -> str:
    """Return the file extension (with leading dot) for a loaded media.

    Falls back to ``.wav`` for audio (the on-the-fly serving format),
    ``.bin`` for unknown types.
    """
    filename = media.get("filename", "")
    suffix = Path(filename).suffix
    if suffix:
        return suffix
    media_type = media.get("type", "")
    if media_type == "audio":
        return ".wav"
    if media_type == "image":
        return ".jpg"
    if media_type == "video":
        return ".mp4"
    if media_type == "text":
        return ".txt"
    return ".bin"


def _resolve_media_bytes(media: dict) -> bytes | None:
    """Return the raw bytes for a loaded media item.

    Mirrors :func:`vtsearch.routes.media.list._resolve_bytes` but is also
    aware of text media (which stores its content as ``media_string``).
    """
    media_bytes = media.get("media_bytes")
    if media_bytes is not None:
        return media_bytes
    media_path = media.get("media_path")
    if media_path:
        p = Path(media_path)
        if p.exists():
            return p.read_bytes()
    media_string = media.get("media_string")
    if media_string is not None:
        return media_string.encode("utf-8")
    return None


@media_server_bp.route("/api/example-sort-by-id", methods=["POST"])
@media_server_bp.arguments(ExampleSortByIdRequestSchema)
@media_server_bp.response(200, ExampleSortResponseSchema)
@media_server_bp.alt_response(400, description="No medias loaded, or media_id not in the loaded snapshot.")
@media_server_bp.alt_response(404, description="Media not found, or its bytes are unavailable when cropping is requested.")
@media_server_bp.alt_response(500, description="Example sort failed.")
def example_sort_by_id(body: dict):
    """Sort medias by similarity to an already-loaded media item.

    When ``crop_params`` is absent the existing ``media["embedding"]``
    is reused — no fetch, no re-embed.  When set, the media's bytes are
    materialised, cropped, and re-embedded before sorting.
    """
    media_id = body["media_id"]

    if not snapshot_medias():
        abort(400, message="No medias loaded")

    media = get_media(media_id)
    if media is None:
        abort(400, message=f"Media id {media_id} is not loaded")

    crop_params = body.get("crop_params") if isinstance(body.get("crop_params"), dict) else None

    try:
        from vtsearch.routes.sorting import (
            _apply_crop_or_keep,
            _cosine_sort,
            _example_sort_from_path,
        )

        if crop_params:
            media_bytes = _resolve_media_bytes(media)
            if media_bytes is None:
                abort(404, message="Media bytes unavailable for cropping")

            import uuid

            DATA_DIR.mkdir(exist_ok=True)
            tmp = DATA_DIR / f"temp_example_{uuid.uuid4().hex}{_media_extension(media)}"
            tmp.write_bytes(media_bytes)
            try:
                _apply_crop_or_keep(tmp, crop_params)
                results, thresh = _example_sort_from_path(tmp)
            finally:
                tmp.unlink(missing_ok=True)
        else:
            embedding = media.get("embedding")
            if embedding is None:
                abort(400, message="Media has no embedding (cannot sort)")
            results, thresh = _cosine_sort(embedding)
        return {"results": results, "threshold": thresh}
    except HTTPException:
        raise
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("example-sort-by-id failed")
        abort(500, message=f"Example sort failed: {format_exception_detail(exc)}")


@media_server_bp.route("/api/server-media-files/from-media-id", methods=["POST"])
@media_server_bp.arguments(ServerMediaFromMediaIdRequestSchema)
@media_server_bp.response(201, ServerMediaUploadResponseSchema)
@media_server_bp.alt_response(400, description="media_id not loaded, or invalid crop_params.")
@media_server_bp.alt_response(404, description="Media bytes unavailable.")
def server_media_file_from_media_id(body: dict):
    """Save a loaded media's bytes to the example_media/ directory.

    Used by the right-click ``Use as detector seed`` flow: the loaded
    media is materialised to disk so the new-detector form can reference
    it as ``media_example`` (the same field the upload path returns).
    """
    import uuid

    media_id = body["media_id"]
    media = get_media(media_id)
    if media is None:
        abort(400, message=f"Media id {media_id} is not loaded")

    media_bytes = _resolve_media_bytes(media)
    if media_bytes is None:
        abort(404, message="Media bytes unavailable")

    crop_params = body.get("crop_params") if isinstance(body.get("crop_params"), dict) else None
    if crop_params:
        media_type = media.get("type", "")
        try:
            from vtsearch.media.cropping import crop_file_bytes

            DATA_DIR.mkdir(exist_ok=True)
            tmp = DATA_DIR / f"temp_seed_{uuid.uuid4().hex}{_media_extension(media)}"
            tmp.write_bytes(media_bytes)
            try:
                media_bytes = crop_file_bytes(tmp, media_type, crop_params)
            finally:
                tmp.unlink(missing_ok=True)
        except (ValueError, FileNotFoundError):
            abort(400, message="Invalid crop_params for this media type")

    suffix = _media_extension(media)
    safe_name = f"{uuid.uuid4().hex}{suffix}"
    media_dir = _get_server_media_dir()
    media_dir.mkdir(parents=True, exist_ok=True)
    dest = media_dir / safe_name
    dest.write_bytes(media_bytes)

    original = media.get("filename") or media.get("origin_name") or f"media_{media_id}{suffix}"
    return {"filename": safe_name, "original_name": original}
