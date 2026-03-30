"""Blueprint for media-related routes."""

from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, jsonify, make_response, request, send_file

from vtsearch.media.base import MediaResponse
from vtsearch.routes.helpers import get_json_or_400
from vtsearch.utils import (
    _state_lock,
    apply_label,
    build_media_lookup,
    get_media,
    medias,
    next_media_id,
    snapshot_medias,
    toggle_vote,
)

medias_bp = Blueprint("medias", __name__)

# Extensions the browser's <video> element can play natively.
_BROWSER_VIDEO_EXTS = {".mp4", ".m4v", ".webm", ".ogg", ".ogv"}


def _transcode_to_mp4(src_bytes: bytes, filename: str) -> bytes | None:
    """Transcode video bytes to browser-playable MP4.

    Tries ffmpeg first (preserves audio, H.264 output).  Falls back to OpenCV
    (video-only, MPEG-4 Part 2) so that videos are at least viewable when
    ffmpeg is not installed.

    Returns the MP4 bytes on success, or ``None`` if transcoding fails.
    """
    ext = Path(filename).suffix or ".avi"
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / f"input{ext}"
        dst_path = Path(tmpdir) / "output.mp4"
        src_path.write_bytes(src_bytes)

        # --- Attempt 1: ffmpeg (best quality, preserves audio) -----------
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-i", str(src_path),
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-movflags", "+faststart",
                    str(dst_path),
                ],
                capture_output=True,
                timeout=120,
                check=True,
            )
        except FileNotFoundError:
            pass  # ffmpeg not installed — fall through to cv2
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
        else:
            if dst_path.exists() and dst_path.stat().st_size > 0:
                return dst_path.read_bytes()

        # --- Attempt 2: OpenCV (video-only, no audio) --------------------
        try:
            import cv2  # noqa: PLC0415

            cap = cv2.VideoCapture(str(src_path))
            if not cap.isOpened():
                return None
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if w <= 0 or h <= 0:
                    return None

                fourcc = cv2.VideoWriter.fourcc(*"mp4v")
                cv2_dst = Path(tmpdir) / "cv2_output.mp4"
                writer = cv2.VideoWriter(str(cv2_dst), fourcc, fps, (w, h))
                if not writer.isOpened():
                    return None
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        writer.write(frame)
                finally:
                    writer.release()
            finally:
                cap.release()

            if cv2_dst.exists() and cv2_dst.stat().st_size > 0:
                return cv2_dst.read_bytes()
        except Exception:
            pass

    return None


def _resolve_bytes(media: dict) -> bytes | None:
    """Return media bytes, lazy-loading from media_path if needed."""
    media_bytes = media.get("media_bytes")
    if media_bytes is not None:
        return media_bytes
    media_path = media.get("media_path")
    if media_path:
        p = Path(media_path)
        if p.exists():
            with open(p, "rb") as f:
                return f.read()
    return None


def _send_video_bytes(data: bytes, mimetype: str, download_name: str) -> Response:
    """Serve video bytes with HTTP range-request support.

    Browsers require range requests to read video metadata (duration, codecs)
    and to support seeking.  Without this, ``<video>`` elements show "0:00".
    """
    total = len(data)
    range_header = request.headers.get("Range")

    if range_header:
        # Parse "bytes=START-END" (END is optional)
        try:
            byte_range = range_header.replace("bytes=", "").strip()
            parts = byte_range.split("-", 1)
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else total - 1
        except (ValueError, IndexError):
            start, end = 0, total - 1

        end = min(end, total - 1)
        length = end - start + 1

        resp = make_response(data[start : end + 1])
        resp.status_code = 206
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{total}"
        resp.headers["Content-Length"] = str(length)
    else:
        resp = make_response(data)
        resp.headers["Content-Length"] = str(total)

    resp.headers["Content-Type"] = mimetype
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Disposition"] = f'inline; filename="{download_name}"'
    return resp


def _resolve_string(media: dict) -> str | None:
    """Return media string, lazy-loading from media_path if needed."""
    media_string = media.get("media_string")
    if media_string is not None:
        return media_string
    media_path = media.get("media_path")
    if media_path:
        p = Path(media_path)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None


def _flask_response(mr: MediaResponse) -> Response:
    """Convert a framework-agnostic :class:`MediaResponse` to a Flask response."""
    if isinstance(mr.data, dict):
        return jsonify(mr.data)
    return send_file(io.BytesIO(mr.data), mimetype=mr.mimetype, download_name=mr.download_name)


@medias_bp.route("/api/medias")
def list_medias() -> Response:
    """Return metadata for all loaded medias as a JSON array.

    Excludes heavyweight fields (``embedding``, ``media_bytes``,
    ``media_string``) from the response.

    Each item contains the required fields ``id``, ``type``, ``filename``,
    ``md5``, and a ``custom_metadata`` dict of display-worthy key/value
    pairs contributed by the media type and/or importer.

    Returns:
        A JSON array of media metadata dicts.
    """
    from vtsearch.media import get as get_media_type  # noqa: PLC0415

    result: list[dict[str, Any]] = []
    for c in snapshot_medias().values():
        media_type_id = c.get("type", "audio")
        media_data: dict[str, Any] = {
            "id": c["id"],
            "type": media_type_id,
            "filename": c.get("filename", f"media_{c['id']}.wav"),
            "md5": c["md5"],
        }
        # Build custom_metadata from media type + any importer-supplied fields
        try:
            mt = get_media_type(media_type_id)
            custom: dict[str, Any] = mt.display_metadata(c)
        except KeyError:
            custom = {}
        importer_custom = c.get("custom_metadata")
        if importer_custom:
            custom.update(importer_custom)
        media_data["custom_metadata"] = custom
        if "origin_name" in c:
            media_data["origin_name"] = c["origin_name"]
        if "description" in c:
            media_data["description"] = c["description"]
        result.append(media_data)
    return jsonify(result)


@medias_bp.route("/api/medias/batch", methods=["POST"])
def batch_medias() -> tuple[Response, int] | Response:
    """Return metadata for a specific set of media IDs.

    This is the paginated counterpart of ``GET /api/medias`` — callers
    request only the IDs they need (e.g. the ones currently visible in a
    virtual-scrolling viewport), keeping payload size bounded.

    Request body (JSON):
        ``{"ids": [1, 2, 3, ...]}`` — list of integer media IDs.

    Returns:
        A JSON array of media metadata dicts (same shape as the full
        ``/api/medias`` response) for the requested IDs that exist.
        Unknown IDs are silently omitted.
    """
    from vtsearch.media import get as get_media_type  # noqa: PLC0415

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    ids = data.get("ids")
    if not isinstance(ids, list):
        return jsonify({"error": "ids must be a list"}), 400

    snap = snapshot_medias()
    result: list[dict[str, Any]] = []
    for cid in ids:
        if not isinstance(cid, int):
            continue
        c = snap.get(cid)
        if c is None:
            continue
        media_type_id = c.get("type", "audio")
        media_data: dict[str, Any] = {
            "id": c["id"],
            "type": media_type_id,
            "filename": c.get("filename", f"media_{c['id']}.wav"),
            "md5": c["md5"],
        }
        try:
            mt = get_media_type(media_type_id)
            custom: dict[str, Any] = mt.display_metadata(c)
        except KeyError:
            custom = {}
        importer_custom = c.get("custom_metadata")
        if importer_custom:
            custom.update(importer_custom)
        media_data["custom_metadata"] = custom
        if "origin_name" in c:
            media_data["origin_name"] = c["origin_name"]
        if "description" in c:
            media_data["description"] = c["description"]
        result.append(media_data)
    return jsonify(result)


@medias_bp.route("/api/medias/<int:media_id>/audio")
def media_audio(media_id: int) -> tuple[Response, int] | Response:
    """Stream the WAV audio bytes for a single media item.

    Args:
        media_id: Integer media ID from the URL path.

    Returns:
        A ``audio/wav`` file response on success (HTTP 200), or a JSON error
        response with HTTP 404 if the media does not exist.
    """
    c = get_media(media_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        return jsonify({"error": "media not available"}), 404
    return send_file(
        io.BytesIO(media_bytes),
        mimetype="audio/wav",
        download_name=f"media_{media_id}.wav",
    )


@medias_bp.route("/api/medias/<int:media_id>/video")
def media_video(media_id: int) -> tuple[Response, int] | Response:
    """Stream the video bytes for a single video media item.

    Determines the MIME type from the media's filename extension, defaulting to
    ``video/mp4`` for unrecognised extensions.

    Args:
        media_id: Integer media ID from the URL path.

    Returns:
        A video file response with the appropriate MIME type on success
        (HTTP 200), a JSON 404 error if the media does not exist, or a JSON 400
        error if the media exists but is not of type ``"video"``.
    """
    c = get_media(media_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "video":
        return jsonify({"error": "not a video"}), 400

    filename = c.get("filename", "")
    ext = Path(filename).suffix.lower() if filename else ".mp4"

    # For browser-incompatible formats (e.g. .avi), transcode to MP4 and
    # cache the result so subsequent requests are instant.
    if ext not in _BROWSER_VIDEO_EXTS:
        cached = c.get("_transcoded_mp4")
        if cached is not None:
            return _send_video_bytes(cached, "video/mp4", f"media_{media_id}.mp4")

        media_bytes = _resolve_bytes(c)
        if media_bytes is None:
            return jsonify({"error": "media not available"}), 404

        transcoded = _transcode_to_mp4(media_bytes, filename)
        if transcoded is not None:
            c["_transcoded_mp4"] = transcoded
            return _send_video_bytes(transcoded, "video/mp4", f"media_{media_id}.mp4")
        # ffmpeg unavailable — serve raw bytes as best-effort fallback
        return _send_video_bytes(media_bytes, "video/mp4", f"media_{media_id}{ext}")

    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        return jsonify({"error": "media not available"}), 404

    if ext == ".webm":
        mimetype = "video/webm"
    elif ext in (".ogg", ".ogv"):
        mimetype = "video/ogg"
    else:
        mimetype = "video/mp4"

    return _send_video_bytes(media_bytes, mimetype, f"media_{media_id}{ext}")


@medias_bp.route("/api/medias/<int:media_id>/image")
def media_image(media_id: int) -> tuple[Response, int] | Response:
    """Stream the image bytes for a single image media item.

    Determines the MIME type from the media's filename extension, defaulting to
    ``image/jpeg`` for unrecognised extensions.

    Args:
        media_id: Integer media ID from the URL path.

    Returns:
        An image file response with the appropriate MIME type on success
        (HTTP 200), a JSON 404 error if the media does not exist, or a JSON 400
        error if the media exists but is not of type ``"image"``.
    """
    c = get_media(media_id)
    if not c:
        return jsonify({"error": "not found"}), 404

    media_type = c.get("type")

    # For non-image types, delegate to the media type's image_response if available
    if media_type and media_type != "image":
        from vtsearch.media import get as get_media_type  # noqa: PLC0415

        try:
            mt = get_media_type(media_type)
        except KeyError:
            mt = None
        if mt and hasattr(mt, "image_response"):
            resp = mt.image_response(c)
            if resp is not None:
                return send_file(
                    io.BytesIO(resp.data),
                    mimetype=resp.mimetype,
                    download_name=resp.download_name,
                )
        return jsonify({"error": "no image available"}), 400

    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        return jsonify({"error": "media not available"}), 404

    # Determine mimetype based on filename extension
    filename = c.get("filename", "")
    if filename.endswith(".png"):
        mimetype = "image/png"
    elif filename.endswith(".gif"):
        mimetype = "image/gif"
    elif filename.endswith(".webp"):
        mimetype = "image/webp"
    elif filename.endswith(".bmp"):
        mimetype = "image/bmp"
    else:
        mimetype = "image/jpeg"

    return send_file(
        io.BytesIO(media_bytes),
        mimetype=mimetype,
        download_name=f"media_{media_id}{Path(filename).suffix if filename and Path(filename).suffix else '.jpg'}",
    )


@medias_bp.route("/api/medias/<int:media_id>/paragraph")
@medias_bp.route("/api/medias/<int:media_id>/text")
def media_paragraph(media_id: int) -> tuple[Response, int] | Response:
    """Return the text content and statistics for a single text media item.

    Args:
        media_id: Integer media ID from the URL path.

    Returns:
        A JSON object with keys ``"content"`` (str), ``"word_count"`` (int),
        and ``"character_count"`` (int) on success (HTTP 200), a JSON 404
        error if the media does not exist, or a JSON 400 error if the media
        exists but is not of type ``"text"``.
    """
    c = get_media(media_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") not in ("text", "paragraph"):
        return jsonify({"error": "not a text media"}), 400

    content = _resolve_string(c)
    if content is None:
        return jsonify({"error": "media not available"}), 404

    return jsonify(
        {
            "content": content,
            "word_count": c.get("word_count", 0) or len(content.split()),
            "character_count": c.get("character_count", 0) or len(content),
        }
    )


@medias_bp.route("/api/medias/<int:media_id>/media")
def media_generic(media_id: int) -> tuple[Response, int] | Response:
    """Serve the media content for any type via a single generic endpoint.

    Determines the media type from the media item's ``"type"`` field and delegates
    to the registered :class:`~vtsearch.media.base.MediaType`'s
    :meth:`~vtsearch.media.base.MediaType.media_response` method.  This
    endpoint works for all current and future media types without modification.

    Args:
        media_id: Integer media ID from the URL path.

    Returns:
        The media content with the appropriate MIME type on success (HTTP 200),
        or a JSON error response for HTTP 404 (media not found) or HTTP 400
        (unrecognised media type).
    """
    c = get_media(media_id)
    if not c:
        return jsonify({"error": "not found"}), 404

    from vtsearch.media import get as media_get

    try:
        mt = media_get(c.get("type", ""))
    except KeyError:
        return jsonify({"error": f"unsupported media type: {c.get('type')}"}), 400

    return _flask_response(mt.media_response(c))


@medias_bp.route("/api/medias/<int:media_id>/vote", methods=["POST"])
def vote_media(media_id: int) -> tuple[Response, int] | Response:
    """Record or toggle a good/bad vote for a single media item.

    Voting behaviour (toggle semantics):

    - If ``vote == "good"`` and the media is already in ``good_votes``, the vote
      is *removed* (toggled off).
    - If ``vote == "good"`` and the media is not yet in ``good_votes``, it is
      added to ``good_votes`` (removed from ``bad_votes`` if present) and the
      event is appended to ``label_history``.
    - The same toggle logic applies symmetrically for ``vote == "bad"``.

    Args:
        media_id: Integer media ID from the URL path.

    Request body (JSON):
        ``{"vote": "good"}`` or ``{"vote": "bad"}``.

    Returns:
        ``{"ok": True}`` (HTTP 200) on success, or a JSON error response for:

        - HTTP 404 – media not found.
        - HTTP 400 – request body is missing, malformed, or ``vote`` is not
          ``"good"`` or ``"bad"``.
    """
    if get_media(media_id) is None:
        return jsonify({"error": "not found"}), 404

    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    vote = data.get("vote")
    if vote not in ("good", "bad"):
        return jsonify({"error": "vote must be 'good' or 'bad'"}), 400

    toggle_vote(media_id, vote)

    from vtsearch.routes.trainable_models import sync_labels_to_loaded_model

    sync_labels_to_loaded_model()

    from vtsearch.labels.sync import sync_to_labelset_source

    sync_to_labelset_source()

    return jsonify({"ok": True})


@medias_bp.route("/api/medias/add-to-pile", methods=["POST"])
def add_media_to_pile() -> tuple[Response, int] | Response:
    """Upload a media file and add it directly to the Good or Bad pile.

    If a media with the same MD5 already exists in the dataset, the existing
    media is voted accordingly.  Otherwise the file is embedded using the
    dataset's embedder, inserted as a new media item, and then voted.

    Request (``multipart/form-data``):
        - ``file``: The media file to upload.
        - ``label``: ``"good"`` or ``"bad"``.

    Returns:
        ``{"ok": true, "media_id": <int>, "is_new": <bool>}`` on success
        (HTTP 200/201), or a JSON error response.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    label = request.form.get("label", "")
    if label not in ("good", "bad"):
        return jsonify({"error": "label must be 'good' or 'bad'"}), 400

    file_bytes = file.read()
    if not file_bytes:
        return jsonify({"error": "Empty file"}), 400

    file_md5 = hashlib.md5(file_bytes).hexdigest()

    # Check if a media with this MD5 already exists
    snap = snapshot_medias()
    _, md5_lookup, _ = build_media_lookup(snap)
    existing_cids = md5_lookup.get(file_md5, [])

    if existing_cids:
        # MD5 match — vote the existing media(s).
        for cid in existing_cids:
            apply_label(cid, label)
        return jsonify({"ok": True, "media_id": existing_cids[0], "is_new": False})

    # No match — embed and insert as new media.
    if not snap:
        return jsonify({"error": "No dataset loaded. Load a dataset first."}), 400

    first_media = next(iter(snap.values()))
    dataset_media_type = first_media.get("type", "audio")
    dataset_embedder_name = first_media.get("embedder", "")

    from vtsearch.media import embedders_for_type, get_embedder

    embedder = None
    if dataset_embedder_name:
        try:
            embedder = get_embedder(dataset_embedder_name)
        except KeyError:
            pass
    if embedder is None:
        avail = embedders_for_type(dataset_media_type)
        embedder = avail[0] if avail else None
    if embedder is None:
        return jsonify({"error": "No embedder available for the current dataset type."}), 400

    # Write to a temporary file for embedding
    import tempfile

    original_filename = file.filename or "upload.bin"
    suffix = Path(original_filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    try:
        embedding = embedder.embed_media(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if embedding is None:
        return jsonify({"error": "Failed to embed the uploaded file."}), 400

    # Generate thumbnail for audio media
    thumb = None
    if dataset_media_type == "audio":
        from vtsearch.media.audio.media_type import generate_waveform_thumbnail  # noqa: PLC0415

        thumb = generate_waveform_thumbnail(file_bytes)

    with _state_lock:
        new_id = next_media_id(medias)
        new_media: dict[str, Any] = {
            "id": new_id,
            "type": dataset_media_type,
            "embedder": dataset_embedder_name,
            "md5": file_md5,
            "embedding": embedding,
            "media_bytes": file_bytes,
            "filename": original_filename,
            "file_size": len(file_bytes),
            "category": "",
            "origin": {
                "importer": "add_to_pile",
                "params": {"filename": original_filename},
            },
            "origin_name": original_filename,
        }
        if thumb is not None:
            new_media["thumbnail_bytes"] = thumb
        medias[new_id] = new_media

    apply_label(new_id, label)

    return jsonify({"ok": True, "media_id": new_id, "is_new": True}), 201
