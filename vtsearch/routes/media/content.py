"""Per-media content routes: stream a media's bytes, thumbnail, or text.

The binary-streaming routes (``audio``, ``video``, ``image``, ``media``)
declare only their JSON error responses via ``alt_response``; the
success body is a streamed file whose mimetype is chosen at runtime, so
the spec leaves it undescribed (mirroring ``detectors/labels.py``'s
preview / thumbnail routes).

The routes register on :data:`vtsearch.routes.media.list.medias_bp`; the
helpers behind them live in :mod:`vtsearch.routes.media._serving`.
"""

from __future__ import annotations

import io
from pathlib import Path

from flask import Response, make_response, request, send_file
from flask_smorest import abort

from vtscore.media.audio.sniff import sniff_audio_format
from vtscore.utils.hashing import content_md5
from vtsearch.routes._media_response import cached_thumbnail_response, image_thumbnail_response
from vtsearch.routes.media._serving import (
    _BROWSER_VIDEO_EXTS,
    _MIMETYPE_BY_EXT,
    _archive_member_response,
    _audio_member_mimetype,
    _flask_response,
    _get_media_variant,
    _parse_region_query,
    _resolve_bytes,
    _resolve_string,
    _send_video_bytes,
    _transcode_to_mp4,
)
from vtsearch.routes.media.list import medias_bp
from vtsearch.schemas.media import MediaParagraphResponseSchema, MediaVariantQuerySchema


@medias_bp.route("/api/medias/<int:media_id>/audio")
@medias_bp.arguments(MediaVariantQuerySchema, location="query")
@medias_bp.alt_response(404, description="Media not found, or media bytes unavailable.")
def media_audio(query: dict, media_id: int):
    """Stream the audio bytes for a single media item.

    Returns the bytes on success (HTTP 200), typed by their container (an MP3
    import is ``audio/mpeg``, a clip slice ``audio/wav``), or a 404 JSON
    envelope if the media does not exist or its bytes are unavailable.

    Accepts ``?variant=original`` to stream the pre-clean payload of an item a
    cleaner rewrote at load time (see :func:`_variant_media`).
    """
    c = _get_media_variant(media_id)
    if not c:
        abort(404, message="not found")
    filename = c.get("filename", "")
    ext = Path(filename).suffix or ".wav"
    streamed = _archive_member_response(
        c, f"media_{media_id}{ext}", "audio/wav", mimetype=_audio_member_mimetype(filename or ext)
    )
    if streamed is not None:
        return streamed
    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        abort(404, message="media not available")
    # A stable ``ETag`` (the media's content hash) lets the browser reuse the
    # audio across re-selections.  The single-fetch object-URL path in the
    # audio player still issues a cold GET per selection, and other surfaces
    # (label view, hover preview) re-request the same clip on replay; the ETag
    # turns those into 304s.  Mirrors the thumbnail route's conditional logic.
    etag = c.get("md5") or content_md5(media_bytes)
    if etag in request.if_none_match:
        resp = make_response("", 304)
        resp.set_etag(etag)
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp
    mimetype, sniffed_ext = sniff_audio_format(media_bytes)
    resp = make_response(
        send_file(
            io.BytesIO(media_bytes),
            mimetype=mimetype,
            download_name=f"media_{media_id}{sniffed_ext}",
        )
    )
    resp.set_etag(etag)
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


@medias_bp.route("/api/medias/<int:media_id>/video")
@medias_bp.arguments(MediaVariantQuerySchema, location="query")
@medias_bp.alt_response(400, description="Media is not a video.")
@medias_bp.alt_response(404, description="Media not found, or media bytes unavailable.")
@medias_bp.alt_response(415, description="Source format requires ffmpeg/opencv which are unavailable.")
def media_video(query: dict, media_id: int):
    """Stream the video bytes for a single video media item.

    Determines the MIME type from the media's filename extension, defaulting
    to ``video/mp4`` for unrecognised extensions.  Accepts
    ``?variant=original`` (see :func:`_variant_media`).
    """
    c = _get_media_variant(media_id)
    if not c:
        abort(404, message="not found")
    if c.get("media_type") != "video":
        abort(400, message="not a video")

    filename = c.get("filename", "")
    ext = Path(filename).suffix.lower() if filename else ".mp4"

    # For browser-incompatible formats (e.g. .avi), transcode to MP4 and
    # cache the result so subsequent requests are instant.
    if ext not in _BROWSER_VIDEO_EXTS:
        return _serve_transcoded_video(c, media_id, ext, filename)

    # Archive-member video: stream the member with Range so the browser only
    # downloads the bytes it plays -- never extracting or fully buffering it.
    streamed = _archive_member_response(c, f"media_{media_id}{ext}", "video/mp4")
    if streamed is not None:
        return streamed

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


def _serve_transcoded_video(c: dict, media_id: int, ext: str, filename: str) -> Response:
    """Serve a browser-incompatible video by transcoding it to MP4 (cached).

    Resolves the source bytes (which may stream from an archive member),
    transcodes to H.264 MP4, memoises the result on the in-memory media, and
    ``abort``-s 415 when neither ffmpeg nor OpenCV is available.
    """
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
    abort(
        415,
        message=f"Cannot play {ext} videos: install ffmpeg or opencv-python-headless to enable transcoding",
    )


def _resolve_display_image(media_id: int) -> tuple[bytes, str, str]:
    """Resolve the displayable image for a media item.

    Returns ``(image_bytes, mimetype, download_name)`` for the bytes the
    ``/image`` route would serve, ``abort``-ing with the matching error when
    no image is available.  For non-image types it delegates to the media
    type's ``image_response`` hook (audio waveforms, video frames); for image
    types it streams the source bytes with a mimetype derived from the
    filename extension.  Shared by the full-image and thumbnail routes, so
    ``?variant=original`` reaches both.
    """
    c = _get_media_variant(media_id)
    if not c:
        abort(404, message="not found")

    media_type = c.get("media_type")

    # For non-image types, delegate to the media type's ``image_response``
    # hook; a type with no paintable form returns None and we 400.
    if media_type and media_type != "image":
        from vtscore.media import get as get_media_type  # noqa: PLC0415

        try:
            mt = get_media_type(media_type)
        except KeyError:
            mt = None
        resp = mt.image_response(c) if mt else None
        # ``MediaResponse.data`` is ``bytes | dict`` (text types serve JSON),
        # but an image is always bytes.  A hook handing back a dict is a
        # broken plugin, not an image: treat it as "no picture" rather than
        # letting it reach ``send_file``.
        if resp is not None and isinstance(resp.data, (bytes, bytearray)):
            return bytes(resp.data), resp.mimetype, resp.download_name
        abort(400, message="no image available")

    media_bytes = _resolve_bytes(c)
    if media_bytes is None:
        abort(404, message="media not available")

    # Determine mimetype based on filename extension
    filename = c.get("filename", "")
    suffix = Path(filename).suffix if filename and Path(filename).suffix else ""
    mimetype = _MIMETYPE_BY_EXT.get(suffix, "image/jpeg")

    suffix = suffix or ".jpg"
    return media_bytes, mimetype, f"media_{media_id}{suffix}"


@medias_bp.route("/api/medias/<int:media_id>/image")
@medias_bp.arguments(MediaVariantQuerySchema, location="query")
@medias_bp.alt_response(400, description="Media is not an image and its image_response hook yielded nothing.")
@medias_bp.alt_response(404, description="Media not found, or media bytes unavailable.")
def media_image(query: dict, media_id: int):
    """Stream the image bytes for a single image media item.

    Determines the MIME type from the media's filename extension, defaulting
    to ``image/jpeg`` for unrecognised extensions. For non-image media types
    the route delegates to the type's ``image_response`` hook (audio
    waveforms, video frames), which returns ``None`` for a type with no
    paintable form.
    """
    data, mimetype, download_name = _resolve_display_image(media_id)
    return send_file(io.BytesIO(data), mimetype=mimetype, download_name=download_name)


@medias_bp.route("/api/medias/<int:media_id>/thumbnail")
@medias_bp.arguments(MediaVariantQuerySchema, location="query")
@medias_bp.alt_response(400, description="Media is not an image and its image_response hook yielded nothing.")
@medias_bp.alt_response(404, description="Media not found, or media bytes unavailable.")
def media_thumbnail(query: dict, media_id: int):
    """Stream a downscaled thumbnail of a media item's image.

    Grid and list tiles use this instead of ``/image`` so a gallery of many
    high-resolution items doesn't force the browser to decode every full-size
    bitmap at once.  The thumbnail is bounded to a fixed longest-side length
    (see :data:`vtscore.media.image.thumbnail.DEFAULT_MAX_DIM`) and is the
    same regardless of zoom level, so an ``ETag`` lets the browser reuse it
    across scrolls and zoom changes.

    When the media carries a precomputed ``thumbnail_bytes`` (generated at
    ingest for image/audio/video), the bytes are streamed directly with no
    request-time decode/resize -- the path that keeps a fresh browse-canvas
    zoom responsive.  Media without one (old pickles, thin loads, undecodable
    SVGs) fall back to generating the thumbnail from the display image.

    An optional ``region=x0,y0,x1,y1`` query (normalised fractions in
    ``[0, 1]``) crops the thumbnail to a sub-region of the image -- used so the
    Good pile shows a region-voted item's crop rather than the whole frame.
    A region request always regenerates from the display image (the
    precomputed full-frame thumbnail can't be cropped after the fact).

    ``?variant=original`` regenerates from the pre-clean payload for the same
    reason: the stored ``thumbnail_bytes`` describes the canonical (cleaned)
    item.
    """
    c = _get_media_variant(media_id)
    if not c:
        abort(404, message="not found")
    region = _parse_region_query(request.args.get("region"))
    if region is not None:
        data, src_mimetype, _ = _resolve_display_image(media_id)
        return image_thumbnail_response(data, src_mimetype, f"thumb_{media_id}", crop=region)
    thumb = c.get("thumbnail_bytes")
    if thumb:
        return cached_thumbnail_response(thumb, f"thumb_{media_id}")
    data, src_mimetype, _ = _resolve_display_image(media_id)
    # Memoise the generated full-frame thumbnail onto the in-memory media so
    # subsequent cold fetches (each fresh browse-canvas zoom/pan) stream the
    # bytes instead of re-decoding the full-resolution original every time.
    # Datasets backed by an external media dir lose their ingest-time
    # ``thumbnail_bytes`` on save and never regenerate them on load, so without
    # this every tile pays a request-time decode+resize forever.  In-memory
    # only -- the bytes ride in the existing dataset context and are never
    # written to disk unless the user explicitly exports.
    from vtscore.media.image.thumbnail import make_image_thumbnail  # noqa: PLC0415

    result = make_image_thumbnail(data)
    if result is not None:
        c["thumbnail_bytes"] = result[0]
        return cached_thumbnail_response(result[0], f"thumb_{media_id}")
    return image_thumbnail_response(data, src_mimetype, f"thumb_{media_id}")


@medias_bp.route("/api/medias/<int:media_id>/paragraph")
@medias_bp.route("/api/medias/<int:media_id>/text")
@medias_bp.arguments(MediaVariantQuerySchema, location="query")
@medias_bp.response(200, MediaParagraphResponseSchema)
@medias_bp.alt_response(400, description="Media is not a text item.")
@medias_bp.alt_response(404, description="Media not found, or media content unavailable.")
def media_paragraph(query: dict, media_id: int):
    """Return the text content and statistics for a single text media item.

    Accepts ``?variant=original`` (see :func:`_variant_media`).  The word /
    character counts are recomputed from the served content in that case,
    since the stored counts describe the canonical (cleaned) text.
    """
    c = _get_media_variant(media_id)
    if not c:
        abort(404, message="not found")
    if c.get("media_type") not in ("text", "paragraph"):
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
@medias_bp.arguments(MediaVariantQuerySchema, location="query")
@medias_bp.alt_response(400, description="Media has an unsupported media type.")
@medias_bp.alt_response(404, description="Media not found.")
def media_generic(query: dict, media_id: int):
    """Serve the media content for any type via a single generic endpoint.

    Determines the media type from the media item's ``"type"`` field and
    delegates to the registered :class:`~vtscore.media.base.MediaType`'s
    :meth:`~vtscore.media.base.MediaType.media_response` method. This
    endpoint works for all current and future media types without
    modification. Response body is either binary (image/audio/video bytes
    with the appropriate mimetype) or JSON (text content); the spec leaves
    the success body undescribed.  Accepts ``?variant=original`` (see
    :func:`_variant_media`).
    """
    c = _get_media_variant(media_id)
    if not c:
        abort(404, message="not found")

    from vtscore.media import get as media_get

    try:
        mt = media_get(c.get("media_type", ""))
    except KeyError:
        abort(400, message=f"unsupported media type: {c.get('media_type')}")

    return _flask_response(mt.media_response(c))
