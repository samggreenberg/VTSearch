"""Blueprint for media-related routes."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from flask import Blueprint, Response, jsonify, send_file

from vtsearch.media.base import MediaResponse
from vtsearch.routes.helpers import get_json_or_400
from vtsearch.utils import (
    get_media,
    snapshot_medias,
    toggle_vote,
)

medias_bp = Blueprint("medias", __name__)


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

    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        return jsonify({"error": "media not available"}), 404

    # Determine mimetype based on filename extension
    filename = c.get("filename", "")
    if filename.endswith(".webm"):
        mimetype = "video/webm"
    elif filename.endswith(".mov"):
        mimetype = "video/quicktime"
    elif filename.endswith(".avi"):
        mimetype = "video/x-msvideo"
    else:
        mimetype = "video/mp4"

    ext = Path(filename).suffix if filename else ".mp4"
    return send_file(
        io.BytesIO(media_bytes),
        mimetype=mimetype,
        download_name=f"media_{media_id}{ext}",
    )


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
    if c.get("type") != "image":
        return jsonify({"error": "not an image"}), 400

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
def media_paragraph(media_id: int) -> tuple[Response, int] | Response:
    """Return the text content and statistics for a single paragraph media item.

    Args:
        media_id: Integer media ID from the URL path.

    Returns:
        A JSON object with keys ``"content"`` (str), ``"word_count"`` (int),
        and ``"character_count"`` (int) on success (HTTP 200), a JSON 404
        error if the media does not exist, or a JSON 400 error if the media
        exists but is not of type ``"paragraph"``.
    """
    c = get_media(media_id)
    if not c:
        return jsonify({"error": "not found"}), 404
    if c.get("type") != "paragraph":
        return jsonify({"error": "not a paragraph"}), 400

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

    return jsonify({"ok": True})
