"""Blueprint for media-related routes.

Migrated to ``flask_smorest`` so the JSON-shaped routes appear in the
``/api/openapi.json`` spec. See ``docs/plans/openapi-schema.md``.

The binary-streaming routes (``audio``, ``video``, ``image``, ``media``)
declare only their JSON error responses via ``alt_response``; the
success body is a streamed file whose mimetype is chosen at runtime, so
the spec leaves it undescribed (mirroring ``detectors/labels.py``'s
preview / thumbnail routes). The multipart ``add-to-pile`` route
describes its success body via ``response`` but omits ``arguments``
because the request shape isn't a single marshmallow schema.
"""

from __future__ import annotations

import hashlib
import io
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from flask import Response, jsonify, make_response, request, send_file
from flask_smorest import Blueprint, abort

from vtsearch.media.audio.ffmpeg import get_ffmpeg_exe
from vtsearch.media.base import MediaResponse
from vtsearch.schemas.media import (
    MediaAddToPileResponseSchema,
    MediaBatchRequestSchema,
    MediaBatchResponseSchema,
    MediaIdsListResponseSchema,
    MediaParagraphResponseSchema,
    MediaVoteRequestSchema,
    MediaVoteResponseSchema,
)
from vtsearch.state import (
    _state_lock,
    apply_label,
    build_media_lookup,
    get_media,
    medias,
    next_media_id,
    snapshot_medias,
    toggle_vote,
)

medias_bp = Blueprint(
    "medias",
    __name__,
    description="List, fetch, stream, and vote on individual media items.",
)

# Extensions the browser's <video> element can play natively.
_BROWSER_VIDEO_EXTS = {".mp4", ".m4v", ".webm", ".ogg", ".ogv"}


def _parse_region_box(raw: Any) -> tuple[float, float, float, float] | None:
    """Validate a ``region_box`` field from a vote request body.

    Returns ``None`` when the field is absent or explicitly null.  Otherwise
    coerces a 4-element list/tuple of numbers in ``[0, 1]`` into a float
    4-tuple.  Raises :class:`ValueError` with a user-facing message when the
    value is malformed.
    """
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ValueError("region_box must be a 4-element list of numbers")
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("region_box entries must be numbers") from exc
    for v in (x0, y0, x1, y1):
        if not (0.0 <= v <= 1.0):
            raise ValueError("region_box entries must be in [0, 1]")
    return (x0, y0, x1, y1)


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
            ffmpeg = get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(src_path),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "23",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
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

        # Reject unsatisfiable ranges (start past EOF, negative, or
        # inverted). RFC 7233 requires 416 with Content-Range: bytes */N.
        if start < 0 or start >= total or start > end:
            resp = make_response(b"")
            resp.status_code = 416
            resp.headers["Content-Range"] = f"bytes */{total}"
            resp.headers["Content-Length"] = "0"
            resp.headers["Content-Type"] = mimetype
            resp.headers["Accept-Ranges"] = "bytes"
            return resp

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


@medias_bp.route("/api/medias/ids")
@medias_bp.response(200, MediaIdsListResponseSchema(many=True))
def list_media_ids():
    """Return a lightweight listing of all loaded medias.

    Each item contains only the fields the frontend needs to build virtual
    scrollers and to inspect the dataset's media type / embedder *before*
    any item becomes visible: ``id``, ``type``, and ``embedder`` (when set).
    Display-worthy metadata (``filename``, ``md5``, ``custom_metadata``,
    ``origin_name``, ``description``, ``clip_*``) is fetched on demand for
    the items currently in the viewport via ``POST /api/medias/batch``.

    This replaces the previous unpaginated ``GET /api/medias`` endpoint,
    which serialised the full metadata for every media in the dataset on
    every call — a 10-20× larger payload that the cache + batch pattern
    makes unnecessary.
    """
    result: list[dict[str, Any]] = []
    for c in snapshot_medias().values():
        item: dict[str, Any] = {"id": c["id"], "type": c.get("type", "audio")}
        embedder = c.get("embedder")
        if embedder:
            item["embedder"] = embedder
        result.append(item)
    return result


@medias_bp.route("/api/medias/batch", methods=["POST"])
@medias_bp.arguments(MediaBatchRequestSchema)
@medias_bp.response(200, MediaBatchResponseSchema(many=True))
def batch_medias(body: dict):
    """Return full metadata for a specific set of media IDs.

    Callers request only the IDs they need (e.g. the ones currently visible
    in a virtual-scrolling viewport), keeping payload size bounded.  Use
    :func:`list_media_ids` (``GET /api/medias/ids``) to discover which
    media IDs exist in the loaded dataset.

    Returns a JSON array of media metadata dicts; unknown IDs are silently
    omitted.
    """
    from vtsearch.media import get as get_media_type  # noqa: PLC0415

    ids = body["ids"]

    snap = snapshot_medias()
    result: list[dict[str, Any]] = []
    for cid in ids:
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
        if c.get("embedder"):
            media_data["embedder"] = c["embedder"]
        for clip_key in ("clip_start", "clip_end", "clip_index", "clip_box"):
            if clip_key in c:
                media_data[clip_key] = c[clip_key]
        result.append(media_data)
    return result


@medias_bp.route("/api/medias/<int:media_id>/audio")
@medias_bp.alt_response(404, description="Media not found, or media bytes unavailable.")
def media_audio(media_id: int):
    """Stream the WAV audio bytes for a single media item.

    Returns an ``audio/wav`` file response on success (HTTP 200), or a 404
    JSON envelope if the media does not exist or its bytes are unavailable.
    """
    c = get_media(media_id)
    if not c:
        abort(404, message="not found")
    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        abort(404, message="media not available")
    return send_file(
        io.BytesIO(media_bytes),
        mimetype="audio/wav",
        download_name=f"media_{media_id}.wav",
    )


@medias_bp.route("/api/medias/<int:media_id>/video")
@medias_bp.alt_response(400, description="Media is not a video.")
@medias_bp.alt_response(404, description="Media not found, or media bytes unavailable.")
@medias_bp.alt_response(415, description="Source format requires ffmpeg/opencv which are unavailable.")
def media_video(media_id: int):
    """Stream the video bytes for a single video media item.

    Determines the MIME type from the media's filename extension, defaulting
    to ``video/mp4`` for unrecognised extensions.
    """
    c = get_media(media_id)
    if not c:
        abort(404, message="not found")
    if c.get("type") != "video":
        abort(400, message="not a video")

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
            abort(404, message="media not available")

        transcoded = _transcode_to_mp4(media_bytes, filename)
        if transcoded is not None:
            c["_transcoded_mp4"] = transcoded
            return _send_video_bytes(transcoded, "video/mp4", f"media_{media_id}.mp4")
        # ffmpeg and OpenCV both unavailable — cannot transcode
        abort(
            415,
            message=f"Cannot play {ext} videos: install ffmpeg or opencv-python-headless to enable transcoding",
        )

    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        abort(404, message="media not available")

    if ext == ".webm":
        mimetype = "video/webm"
    elif ext in (".ogg", ".ogv"):
        mimetype = "video/ogg"
    else:
        mimetype = "video/mp4"

    return _send_video_bytes(media_bytes, mimetype, f"media_{media_id}{ext}")


@medias_bp.route("/api/medias/<int:media_id>/image")
@medias_bp.alt_response(400, description="Media is not an image and has no image_response delegate.")
@medias_bp.alt_response(404, description="Media not found, or media bytes unavailable.")
def media_image(media_id: int):
    """Stream the image bytes for a single image media item.

    Determines the MIME type from the media's filename extension, defaulting
    to ``image/jpeg`` for unrecognised extensions. For non-image media types
    that declare an ``image_response`` hook (audio waveforms, video frames),
    the route delegates to that hook.
    """
    c = get_media(media_id)
    if not c:
        abort(404, message="not found")

    media_type = c.get("type")

    # For non-image types, delegate to the media type's image_response if available
    if media_type and media_type != "image":
        from vtsearch.media import get as get_media_type  # noqa: PLC0415

        try:
            mt = get_media_type(media_type)
        except KeyError:
            mt = None
        image_response_fn = getattr(mt, "image_response", None) if mt else None
        if image_response_fn is not None:
            resp = image_response_fn(c)
            if resp is not None:
                return send_file(
                    io.BytesIO(resp.data),
                    mimetype=resp.mimetype,
                    download_name=resp.download_name,
                )
        abort(400, message="no image available")

    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        abort(404, message="media not available")

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
@medias_bp.response(200, MediaParagraphResponseSchema)
@medias_bp.alt_response(400, description="Media is not a text item.")
@medias_bp.alt_response(404, description="Media not found, or media content unavailable.")
def media_paragraph(media_id: int):
    """Return the text content and statistics for a single text media item."""
    c = get_media(media_id)
    if not c:
        abort(404, message="not found")
    if c.get("type") not in ("text", "paragraph"):
        abort(400, message="not a text media")

    content = _resolve_string(c)
    if content is None:
        abort(404, message="media not available")

    return {
        "content": content,
        "word_count": c.get("word_count", 0) or len(content.split()),
        "character_count": c.get("character_count", 0) or len(content),
    }


@medias_bp.route("/api/medias/<int:media_id>/media")
@medias_bp.alt_response(400, description="Media has an unsupported media type.")
@medias_bp.alt_response(404, description="Media not found.")
def media_generic(media_id: int):
    """Serve the media content for any type via a single generic endpoint.

    Determines the media type from the media item's ``"type"`` field and
    delegates to the registered :class:`~vtsearch.media.base.MediaType`'s
    :meth:`~vtsearch.media.base.MediaType.media_response` method. This
    endpoint works for all current and future media types without
    modification. Response body is either binary (image/audio/video bytes
    with the appropriate mimetype) or JSON (text content); the spec leaves
    the success body undescribed.
    """
    c = get_media(media_id)
    if not c:
        abort(404, message="not found")

    from vtsearch.media import get as media_get

    try:
        mt = media_get(c.get("type", ""))
    except KeyError:
        abort(400, message=f"unsupported media type: {c.get('type')}")

    return _flask_response(mt.media_response(c))


@medias_bp.route("/api/medias/<int:media_id>/vote", methods=["POST"])
@medias_bp.arguments(MediaVoteRequestSchema)
@medias_bp.response(200, MediaVoteResponseSchema)
@medias_bp.alt_response(400, description="region_box is malformed, or a bad-vote carries a region_box.")
@medias_bp.alt_response(404, description="Media not found.")
def vote_media(body: dict, media_id: int):
    """Record or toggle a good/bad vote for a single media item.

    Voting behaviour (toggle semantics):

    - If ``vote == "good"`` and the media is already in ``good_votes``, the
      vote is *removed* (toggled off).
    - If ``vote == "good"`` and the media is not yet in ``good_votes``, it is
      added to ``good_votes`` (removed from ``bad_votes`` if present) and the
      event is appended to ``label_history``.
    - The same toggle logic applies symmetrically for ``vote == "bad"``.

    Yes-votes may carry ``"region_box": [x0, y0, x1, y1]`` (normalised image
    coords in ``[0, 1]``) to designate the good region. No-votes that
    include ``region_box`` are rejected — by design no-votes are
    image-level always (patch-embedder v2).
    """
    if get_media(media_id) is None:
        abort(404, message="not found")

    vote = body["vote"]

    try:
        region_box = _parse_region_box(body.get("region_box"))
    except ValueError as exc:
        abort(400, message=str(exc))
    if vote == "bad" and region_box is not None:
        abort(400, message="no-votes cannot carry a region_box")

    toggle_vote(media_id, vote, region_box=region_box)

    from vtsearch.detectors.label_sync import sync_labels_to_loaded_detector

    sync_labels_to_loaded_detector()

    from vtsearch.labels.sync import sync_to_labelset_source

    sync_to_labelset_source()

    return {"ok": True}


@medias_bp.route("/api/medias/add-to-pile", methods=["POST"])
@medias_bp.response(200, MediaAddToPileResponseSchema)
@medias_bp.alt_response(
    201,
    schema=MediaAddToPileResponseSchema,
    description="A new media was embedded and inserted before being voted.",
)
@medias_bp.alt_response(
    400,
    description=(
        "Missing or malformed multipart body (no file / empty file / "
        "invalid label), no dataset loaded, or no embedder available."
    ),
)
def add_media_to_pile():
    """Upload a media file and add it directly to the Good or Bad pile.

    If a media with the same MD5 already exists in the dataset, the existing
    media is voted accordingly. Otherwise the file is embedded using the
    dataset's embedder, inserted as a new media item, and then voted. The
    request body is multipart/form-data with ``file`` (the upload) and
    ``label`` (``"good"`` or ``"bad"``).
    """
    if "file" not in request.files:
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

    label = request.form.get("label", "")
    if label not in ("good", "bad"):
        abort(400, message="label must be 'good' or 'bad'")

    file_bytes = file.read()
    if not file_bytes:
        abort(400, message="Empty file")

    file_md5 = hashlib.md5(file_bytes).hexdigest()

    # Check if a media with this MD5 already exists
    snap = snapshot_medias()
    _, md5_lookup, _ = build_media_lookup(snap)
    existing_cids = md5_lookup.get(file_md5, [])

    if existing_cids:
        # MD5 match — vote the existing media(s).
        for cid in existing_cids:
            apply_label(cid, label)
        return {"ok": True, "media_id": existing_cids[0], "is_new": False}

    # No match — embed and insert as new media.
    if not snap:
        abort(400, message="No dataset loaded. Load a dataset first.")

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
        abort(400, message="No embedder available for the current dataset type.")

    # Write to a temporary file for embedding
    import tempfile

    original_filename = file.filename or "upload.bin"
    suffix = Path(original_filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    from vtsearch.media.embedder import media_from_path  # noqa: PLC0415

    try:
        embedding = embedder.embed_media(media_from_path(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    if embedding is None:
        abort(400, message="Failed to embed the uploaded file.")

    # Generate thumbnail for non-image media
    thumb = None
    if dataset_media_type == "audio":
        from vtsearch.media.audio.media_type import generate_waveform_thumbnail  # noqa: PLC0415

        thumb = generate_waveform_thumbnail(file_bytes)
    elif dataset_media_type == "video":
        from vtsearch.media.video.media_type import generate_video_thumbnail  # noqa: PLC0415

        thumb = generate_video_thumbnail(file_bytes)

    new_media: dict[str, Any] = {
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

    with _state_lock:
        new_id = next_media_id(medias)
        new_media["id"] = new_id
        medias[new_id] = new_media

    apply_label(new_id, label)

    return {"ok": True, "media_id": new_id, "is_new": True}, 201
