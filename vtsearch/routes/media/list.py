"""Blueprint for media-related routes.

Migrated to ``flask_smorest`` so the JSON-shaped routes appear in the
``/api/openapi.json`` spec.

The binary-streaming routes (``audio``, ``video``, ``image``, ``media``)
declare only their JSON error responses via ``alt_response``; the
success body is a streamed file whose mimetype is chosen at runtime, so
the spec leaves it undescribed (mirroring ``detectors/labels.py``'s
preview / thumbnail routes). The multipart ``add-to-pile`` route
describes its success body via ``response`` but omits ``arguments``
because the request shape isn't a single marshmallow schema.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from flask import Response, jsonify, make_response, request, send_file
from flask_smorest import Blueprint, abort

from vtscore.embedding.media_vectors import init_embeddings, media_embedder_names
from vtscore.media.audio.ffmpeg import get_ffmpeg_exe
from vtscore.media.base import MediaResponse
from vtscore.security.path_validation import resolve_media_file_path
from vtscore.utils.hashing import content_md5
from vtscore.datasets.vote_provenance import normalize_provenance
from vtscore.utils.hits import hit_custom_metadata
from vtsearch.schemas.media import (
    MediaAddToPileResponseSchema,
    MediaBatchRequestSchema,
    MediaBatchResponseSchema,
    MediaIdsListResponseSchema,
    MediaParagraphResponseSchema,
    MediaVariantQuerySchema,
    MediaVoteBulkRequestSchema,
    MediaVoteBulkResponseSchema,
    MediaVoteRequestSchema,
    MediaVoteResponseSchema,
)
from vtsearch.routes._shared import (
    cached_thumbnail_response,
    image_thumbnail_response,
    require_dataset_header,
    require_detector_header,
)
from vtsearch.state import (
    _state_lock,
    apply_label,
    cached_md5_lookup,
    get_media,
    medias,
    next_media_id,
    set_vote,
    snapshot_medias,
)

medias_bp = Blueprint(
    "medias",
    __name__,
    description="List, fetch, stream, and vote on individual media items.",
)

# Extensions the browser's <video> element can play natively.
_BROWSER_VIDEO_EXTS = {".mp4", ".m4v", ".webm", ".ogg", ".ogv"}

# Image filename extension -> served mimetype. Unlisted extensions fall back to
# ``image/jpeg`` (see :func:`_resolve_display_image`).
_MIMETYPE_BY_EXT = {
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


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


def _parse_region_query(raw: str | None) -> tuple[float, float, float, float] | None:
    """Parse a ``region=x0,y0,x1,y1`` thumbnail query argument.

    Returns ``None`` when the argument is absent or malformed (the route then
    serves the whole-image thumbnail).  Out-of-range / degenerate / near-full
    boxes are tolerated here and canonicalised downstream by
    :func:`vtscore.media.image.thumbnail.normalize_region_crop`.
    """
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(p) for p in parts)
    except ValueError:
        return None
    return (x0, y0, x1, y1)


def _transcode_to_mp4(src_bytes: bytes, filename: str) -> bytes | None:  # noqa: C901
    """Transcode video bytes to browser-playable MP4.

    Tries ffmpeg first (preserves audio, H.264 output).  Falls back to OpenCV
    (video-only, MPEG-4 Part 2) so that videos are at least viewable when
    ffmpeg is not installed - but *only* then: importing ``cv2`` pulls a
    vendored OpenSSL into the process, which aborts the interpreter on
    FIPS-enabled hosts (see :mod:`vtscore.media.video.decode`), so it is not
    worth reaching for when ffmpeg is present and simply failed on this file.

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
            pass  # ffmpeg not installed; fall through to cv2
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        else:
            return dst_path.read_bytes() if dst_path.exists() and dst_path.stat().st_size > 0 else None

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


def _variant_media(media: dict) -> dict:
    """Return the payload variant the request's ``variant`` query asks for.

    Media that a :class:`~vtscore.media.cleaner.MediaCleaner` rewrote at load
    time keep a snapshot of their pre-clean payload under the ``original_*``
    keys (see ``docs/plans/media-cleaners.md``).  The *canonical* payload is
    always the cleaned one, so every route serves that by default and *is*
    returned the very same dict - no copy, so request-time memoisation (the
    thumbnail route's ``thumbnail_bytes``, the video route's
    ``_transcoded_mp4``) still lands on the live media.

    ``?variant=original`` instead returns a throwaway view with the snapshot
    promoted into the canonical keys, so the ordinary resolution chain streams
    the original bytes without any route having to know cleaners exist.  Every
    non-inline backing (``media_path`` / ``media_url``) is stripped from the
    view: those point at the *source* file, which the resolver would otherwise
    fall through to and serve as if it were the snapshot.  ``md5`` is dropped
    too, so ETag-emitting routes hash the bytes they actually serve rather than
    labelling the original with the cleaned item's hash.

    An unknown variant, or ``original`` on a media with no snapshot, falls back
    to the canonical payload: a stale bookmark should show the item, not 404.
    """
    if request.args.get("variant") != "original":
        return media

    from vtscore.datasets.clipper_chain import has_original_payload  # noqa: PLC0415

    if not has_original_payload(media):
        return media

    view = dict(media)
    view["media_bytes"] = media.get("original_media_bytes")
    view["media_string"] = media.get("original_media_string")
    if media.get("original_duration") is not None:
        view["duration"] = media["original_duration"]
    # Derived metadata (counts, dimensions, hash, thumbnail) all describe the
    # canonical payload, so drop it and let each route's existing
    # "recompute when absent" fallback derive it from what is actually served.
    for key in (
        "media_path",
        "media_url",
        "thumbnail_bytes",
        "md5",
        "_transcoded_mp4",
        "word_count",
        "character_count",
    ):
        view.pop(key, None)
    return view


def _get_media_variant(media_id: int) -> dict | None:
    """Fetch a loaded media and apply the request's ``variant`` selection."""
    media = get_media(media_id)
    if not media:
        return None
    return _variant_media(media)


def _resolve_bytes(media: dict) -> bytes | None:
    """Return a media's full bytes via the media type's resolver.

    Delegates to :meth:`MediaType._resolve_media_bytes` so every byte route
    shares one resolution chain (inline bytes -> lazy clip -> archive member
    -> local path -> remote URL) instead of the old path-only duplicate.  Media
    whose type isn't registered fall back to the inline-bytes / local-path
    pair.
    """
    from vtscore.media import get as get_media_type  # noqa: PLC0415

    try:
        mt = get_media_type(media.get("media_type", ""))
    except KeyError:
        mt = None
    if mt is not None:
        return mt._resolve_media_bytes(media)

    media_bytes = media.get("media_bytes")
    if media_bytes is not None:
        return media_bytes
    media_path = media.get("media_path")
    if media_path:
        p = resolve_media_file_path(media_path)
        if p is not None and p.exists():
            with open(p, "rb") as f:
                return f.read()
    return None


def _member_mimetype(filename: str, default: str) -> str:
    """Guess a mimetype for an archive member from its name, else *default*."""
    import mimetypes  # noqa: PLC0415

    guessed, _ = mimetypes.guess_type(filename)
    return guessed or default


def _audio_member_mimetype(filename: str) -> str:
    """Pick the ``Content-Type`` for an archive-member served via ``/audio``.

    WebDataset-style audio events ship in a few container shapes:

    * **demuxed AAC** (microvent) - ``.aac`` -> ``audio/aac``.
    * **audio in an MP4 container** (multivent-raw rides the audio in the video
      MP4; ``.m4a`` audio chunks) - ``mimetypes`` reports ``.mp4`` as
      *``video/mp4``*, but when an ``<audio>`` element requests it we want the
      audio mimetype so the browser plays the audio track.  Map any ``video/*``
      container (mp4 / quicktime) to ``audio/mp4``.

    Anything else keeps its guessed ``audio/*`` type, normalising the legacy
    ``audio/x-wav`` to ``audio/wav``, and falls back to ``audio/wav`` when the
    extension is unknown (the on-the-fly serving format).
    """
    import mimetypes  # noqa: PLC0415

    guessed, _ = mimetypes.guess_type(filename)
    if not guessed:
        return "audio/wav"
    if guessed == "audio/x-wav":
        return "audio/wav"
    if guessed.startswith("video/"):
        return "audio/mp4"
    return guessed


def _parse_range_header(range_header: str, total: int) -> tuple[int, int] | None:
    """Resolve a ``Range`` header into inclusive ``(start, end)`` byte offsets.

    Returns ``None`` when the requested range is unsatisfiable, which RFC 7233
    says to answer with a 416.

    Three forms are recognised: ``bytes=START-END``, ``bytes=START-`` ("to the
    end"), and the **suffix** form ``bytes=-N`` ("the last *N* bytes"), which
    some players use to read an MP4 ``moov`` atom parked at the tail of the
    file.  A suffix range must not be mistaken for an unparseable one: falling
    back to the whole payload would return a 206 containing far more than was
    asked for, defeating the point of a ranged read straight out of an archive
    member.  A header that genuinely can't be parsed still degrades to the
    whole payload.
    """
    try:
        byte_range = range_header.replace("bytes=", "").strip()
        parts = byte_range.split("-", 1)
        if parts[0] == "":
            # Suffix range: the last N bytes.  N larger than the payload
            # means "the whole thing"; N == 0 requests nothing, which RFC
            # 7233 makes unsatisfiable.
            suffix_len = int(parts[1])
            if suffix_len <= 0:
                return None
            start = max(0, total - suffix_len)
            end = total - 1
        else:
            start = int(parts[0])
            end = int(parts[1]) if parts[1] else total - 1
    except (ValueError, IndexError):
        start, end = 0, total - 1

    end = min(end, total - 1)
    # Reject unsatisfiable ranges (start past EOF, negative, or inverted).
    if start < 0 or start >= total or start > end:
        return None
    return start, end


def _unsatisfiable_range_response(total: int, mimetype: str) -> Response:
    """Build the RFC 7233 416 response for an unsatisfiable ``Range``."""
    resp = make_response(b"")
    resp.status_code = 416
    resp.headers["Content-Range"] = f"bytes */{total}"
    resp.headers["Content-Length"] = "0"
    resp.headers["Content-Type"] = mimetype
    resp.headers["Accept-Ranges"] = "bytes"
    return resp


def _send_streamed_range(total, read_slice, mimetype: str, download_name: str) -> Response:
    """Serve *total* bytes with HTTP Range support, reading only what's asked.

    *read_slice(start, length)* returns the requested byte slice (``length``
    is ``None`` for "to end").  Unlike :func:`_send_video_bytes` this never
    buffers the whole payload for a partial request -- the backing reader
    (an archive member) is seeked to *start* and only the served slice is
    read, so playing a few seconds out of a large member transfers a few
    seconds of bytes.
    """
    range_header = request.headers.get("Range")
    if range_header:
        parsed = _parse_range_header(range_header, total)
        if parsed is None:
            return _unsatisfiable_range_response(total, mimetype)
        start, end = parsed
        data = read_slice(start, end - start + 1)
        resp = make_response(data)
        resp.status_code = 206
        resp.headers["Content-Range"] = f"bytes {start}-{end}/{total}"
        resp.headers["Content-Length"] = str(len(data))
    else:
        data = read_slice(0, None)
        resp = make_response(data)
        resp.headers["Content-Length"] = str(total)

    resp.headers["Content-Type"] = mimetype
    resp.headers["Accept-Ranges"] = "bytes"
    resp.headers["Content-Disposition"] = f'inline; filename="{download_name}"'
    return resp


def _archive_member_response(
    media: dict, download_name: str, default_mimetype: str, mimetype: str | None = None
) -> Response | None:
    """Serve an archive-member media by streaming a single member, or ``None``.

    Returns ``None`` for media that aren't archive-member-backed (the caller
    then falls through to its normal resolution), or when the member can't be
    located in its shard.  Otherwise returns a Range-capable response that
    reads only the requested bytes straight out of the tar/zip, never
    extracting or fully buffering the member.

    *mimetype*, when given, is used verbatim; otherwise it is guessed from the
    member name (falling back to *default_mimetype*).  The audio route passes an
    explicit value because audio events ride in containers ``mimetypes`` would
    label ``video/*`` (see :func:`_audio_member_mimetype`).
    """
    from vtscore.datasets.archive_stream import (  # noqa: PLC0415
        ArchiveMemberError,
        archive_member_ref,
        member_size,
        read_member_range,
    )

    ref = archive_member_ref(media)
    if ref is None:
        return None
    archive_path, member = ref
    if resolve_media_file_path(archive_path) is None:
        return None
    try:
        total = member_size(archive_path, member)
    except (ArchiveMemberError, OSError):
        return None

    def read_slice(start, length):
        return read_member_range(archive_path, member, start, length)

    resolved = mimetype or _member_mimetype(media.get("filename") or member, default_mimetype)
    return _send_streamed_range(total, read_slice, resolved, download_name)


def _send_video_bytes(data: bytes, mimetype: str, download_name: str) -> Response:
    """Serve video bytes with HTTP range-request support.

    Browsers require range requests to read video metadata (duration, codecs)
    and to support seeking.  Without this, ``<video>`` elements show "0:00".
    """
    total = len(data)
    range_header = request.headers.get("Range")

    if range_header:
        parsed = _parse_range_header(range_header, total)
        # RFC 7233 requires 416 with Content-Range: bytes */N.
        if parsed is None:
            return _unsatisfiable_range_response(total, mimetype)
        start, end = parsed
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
        p = resolve_media_file_path(media_path)
        if p is not None and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
    return None


def _flask_response(mr: MediaResponse) -> Response:
    """Convert a framework-agnostic :class:`MediaResponse` to a Flask response."""
    if isinstance(mr.data, dict):
        return jsonify(mr.data)
    return send_file(io.BytesIO(mr.data), mimetype=mr.mimetype, download_name=mr.download_name)


def _attach_optional_fields(out: dict[str, Any], media: dict[str, Any]) -> None:
    """Copy the sometimes-present per-media fields onto a batch payload *out*.

    ``has_original`` marks an item some
    :class:`~vtscore.media.cleaner.MediaCleaner` actually rewrote at load time
    (it kept a pre-clean snapshot), which is what gates the detail viewer's
    Clean/Original toggle and the byte routes' ``?variant=original``.
    """
    from vtscore.datasets.clipper_chain import has_original_payload  # noqa: PLC0415

    if has_original_payload(media):
        out["has_original"] = True
    if "origin_name" in media:
        out["origin_name"] = media["origin_name"]
    if "description" in media:
        out["description"] = media["description"]
    _attach_embedder_fields(out, media)


def _attach_embedder_fields(out: dict[str, Any], media: dict[str, Any]) -> None:
    """Copy a media's embedder identity onto a serialized payload *out*.

    Sets the singular ``embedder`` (the recorded primary) when present and the
    full ``embedders`` list (every bound embedder name, v3 trio) so the frontend
    can resolve dataset capabilities across the whole binding, not just the
    primary.  Both are omitted when the media carries no embedder.
    """
    if media.get("embedder"):
        out["embedder"] = media["embedder"]
    names = media_embedder_names(media)
    if names:
        out["embedders"] = names


@medias_bp.route("/api/medias/ids")
@medias_bp.response(200, MediaIdsListResponseSchema(many=True))
def list_media_ids():
    """Return a lightweight listing of all loaded medias.

    Each item contains only the fields the frontend needs to build virtual
    scrollers and to inspect the dataset's media type / embedder *before*
    any item becomes visible: ``id``, ``media_type``, and ``embedder`` (when set).
    Display-worthy metadata (``filename``, ``md5``, ``custom_metadata``,
    ``origin_name``, ``description``, ``clip_*``) is fetched on demand for
    the items currently in the viewport via ``POST /api/medias/batch``.

    This replaces the previous unpaginated ``GET /api/medias`` endpoint,
    which serialised the full metadata for every media in the dataset on
    every call; a 10-20x larger payload that the cache + batch pattern
    makes unnecessary.
    """
    result: list[dict[str, Any]] = []
    for c in snapshot_medias().values():
        item: dict[str, Any] = {"id": c["id"], "media_type": c.get("media_type", "audio")}
        _attach_embedder_fields(item, c)
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
    omitted.  Each item's ``custom_metadata`` is the media type's
    ``display_metadata`` - including the curated "Source" / "Derived Via" /
    "Imported Via" provenance lines distilled from ``origin.params`` - with
    any importer-supplied ``custom_metadata`` layered on top, minus the
    ``embedding`` plumbing key.
    """
    from vtscore.datasets.archive_stream import archive_member_ref  # noqa: PLC0415
    from vtscore.media import get as get_media_type  # noqa: PLC0415

    ids = body["ids"]

    snap = snapshot_medias()
    result: list[dict[str, Any]] = []
    for cid in ids:
        c = snap.get(cid)
        if c is None:
            continue
        media_type_id = c.get("media_type", "audio")
        media_data: dict[str, Any] = {
            "id": c["id"],
            "media_type": media_type_id,
            "filename": c.get("filename", f"media_{c['id']}.wav"),
            "md5": c["md5"],
        }
        try:
            mt = get_media_type(media_type_id)
            custom: dict[str, Any] = mt.display_metadata(c)
        except KeyError:
            custom = {}
        # Re-derived rather than read straight off the media: an importer may
        # ship a pre-computed vector nested inside ``custom_metadata`` via
        # ``custom_metadata_map``, and a numpy array here would fail JSON
        # encoding for the whole batch.
        importer_custom = hit_custom_metadata(c)
        if importer_custom:
            custom.update(importer_custom)
        media_data["custom_metadata"] = custom
        _attach_optional_fields(media_data, c)
        # ``clip_start`` / ``clip_end`` are a *playback window into the whole
        # served file*: every player (hover previews, bin popup, selection
        # panel, and the center-panel audio/video players) seeks to
        # ``clip_start`` and loops within ``[clip_start, clip_end]``.  That
        # contract holds for video (clips share the parent's bytes) and for
        # archive-member windowed audio (the whole AAC/MP4 member is served),
        # but *not* for the audio clipper / lazy-clip resolver, which serve the
        # already-sliced clip bytes (a short WAV in its own 0-based timeline)
        # while stamping the original absolute offsets.  Handing those offsets
        # to a player makes it seek past the end of the short file and play
        # silence (e.g. a 5-second TUT clip cut from a 4-minute soundscape,
        # ``clip_start`` ~120s).  So suppress the window for byte-sliced audio;
        # the offsets still reach the UI as provenance via ``custom_metadata``
        # ("Clip Start" / "Clip End") from ``display_metadata`` above.
        audio_serves_slice = media_type_id == "audio" and archive_member_ref(c) is None
        for clip_key in ("clip_start", "clip_end", "clip_index", "clip_box"):
            if clip_key not in c:
                continue
            if audio_serves_slice and clip_key in ("clip_start", "clip_end"):
                continue
            media_data[clip_key] = c[clip_key]
        result.append(media_data)
    return result


@medias_bp.route("/api/medias/<int:media_id>/audio")
@medias_bp.arguments(MediaVariantQuerySchema, location="query")
@medias_bp.alt_response(404, description="Media not found, or media bytes unavailable.")
def media_audio(query: dict, media_id: int):
    """Stream the WAV audio bytes for a single media item.

    Returns an ``audio/wav`` file response on success (HTTP 200), or a 404
    JSON envelope if the media does not exist or its bytes are unavailable.

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
    resp = make_response(
        send_file(
            io.BytesIO(media_bytes),
            mimetype="audio/wav",
            download_name=f"media_{media_id}.wav",
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


@medias_bp.route("/api/medias/<int:media_id>/vote", methods=["POST"])
@medias_bp.arguments(MediaVoteRequestSchema)
@medias_bp.response(200, MediaVoteResponseSchema)
@medias_bp.alt_response(
    400, description="region_box or provenance is malformed, or a non-good target carries a region_box."
)
@medias_bp.alt_response(404, description="Media not found.")
@require_dataset_header
@require_detector_header
def vote_media(body: dict, media_id: int):
    """Set a single media's vote to an **absolute target state**.

    ``target`` is one of:

    - ``"good"``: set to good (overrides ``bad`` if present).
    - ``"bad"``: set to bad (overrides ``good`` if present).
    - ``"none"``: un-vote (remove any existing vote).

    Behaviour is **idempotent**: sending the current state is a no-op
    that does not append to ``label_history``, does not credit
    achievements, and returns the existing click-time.  This is the
    fix for logical-bug-audit H1; two stale-view tabs that race the
    same media no longer alternate ADD/REMOVE on the server, so the
    achievement counter no longer inflates beyond the number of real
    labeling decisions.

    Yes-targets may carry ``"region_box": [x0, y0, x1, y1]`` (normalised
    image coords in ``[0, 1]``) to designate the good region.
    ``"bad"`` and ``"none"`` targets must not include ``region_box``; by
    design no-votes are image-level always (patch-embedder v2).

    An optional ``provenance`` block records how the item was surfaced
    (flow / autopilot phase / select mode / sort / rank / score).  It is
    stored only when the call actually changes the vote state, so an
    idempotent re-send from a stale tab cannot overwrite what the original
    click recorded.  See :mod:`vtscore.datasets.vote_provenance`.

    Returns ``{"ok": true, "state": <new state>, "click_time": <int|null>}``
    so the client can reconcile its optimistic view directly from the
    response.
    """
    if get_media(media_id) is None:
        abort(404, message="not found")

    target = body["target"]

    try:
        region_box = _parse_region_box(body.get("region_box"))
    except ValueError as exc:
        abort(400, message=str(exc))
    if target != "good" and region_box is not None:
        abort(400, message="region_box is only valid on 'good' targets")

    try:
        provenance = normalize_provenance(body.get("provenance"))
    except ValueError as exc:
        abort(400, message=str(exc))

    _old, new_state, click_time = set_vote(media_id, target, region_box=region_box, provenance=provenance)

    # Persist the resulting labelset.  A failure here (e.g. ``os.replace``
    # EBUSY/ENOSPC under ``_write_detector``) used to bubble as an
    # uncaught 500 with no rollback, leaving the in-memory vote committed
    # while the on-disk labelset stayed at its prior value (audit finding
    # H30).  Now we explicitly surface the failure as a 500 with a clear
    # message; the in-memory mutation is rare enough to be acceptable
    # since the next vote re-merges and retries.  ``sync_to_labelset_source``
    # stays best-effort: it's the debounced background-timer scheduling
    # call, so only a synchronous scheduling failure could fault here.
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

    try:
        sync_labels_to_loaded_detector()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("vote_media: detector label sync failed")
        abort(500, message=f"Failed to persist vote to detector store: {exc}")

    from vtscore.labels.sync import sync_to_labelset_source

    try:
        sync_to_labelset_source()
    except Exception:
        import logging

        logging.getLogger(__name__).exception("vote_media: labelset source scheduling failed")

    return {"ok": True, "state": new_state, "click_time": click_time}


@medias_bp.route("/api/medias/vote-bulk", methods=["POST"])
@medias_bp.arguments(MediaVoteBulkRequestSchema)
@medias_bp.response(200, MediaVoteBulkResponseSchema)
@medias_bp.alt_response(400, description="No ids supplied, or provenance is malformed.")
@require_dataset_header
@require_detector_header
def vote_media_bulk(body: dict):
    """Apply one absolute vote target to many medias in a single request.

    Mirrors ``/api/medias/<id>/vote`` for a batch: each id is set to
    ``target`` with the same idempotent semantics (including Find-mode
    verification — a good/bad target marks the item verified), then the
    detector labelset is persisted **once** rather than per id.  Bulk votes
    are image-level (no region boxes).  Powers the Browser's "Verified Good" /
    "Verified Bad" actions, which mark a hand-selected set good/bad, verify
    them, and drop them from the browse.

    Ids that aren't in the loaded dataset are skipped and reported back in
    ``missing``; ``changed`` counts only the ids whose state actually moved
    (idempotent re-applies don't count).
    """
    target = body["target"]
    ids = body["ids"]
    if not ids:
        abort(400, message="No ids supplied")

    # A batch action over a hand-selected set is its own surfacing flow; the
    # client may refine it (with the sort it was selecting from), but never
    # has to.
    try:
        provenance = normalize_provenance(body.get("provenance") or {"flow": "bulk"})
    except ValueError as exc:
        abort(400, message=str(exc))

    changed = 0
    missing: list[int] = []
    for media_id in ids:
        if get_media(media_id) is None:
            missing.append(media_id)
            continue
        # Bulk "Verified Good/Bad" over a hand-selected set is not an
        # individual hand-click, so it must not build a Marathoner streak;
        # count_streak=False still credits the other vote achievements.
        old, new, _click_time = set_vote(media_id, target, count_streak=False, provenance=provenance)
        if old != new:
            changed += 1

    # Persist the resulting labelset once for the whole batch.  Mirrors the
    # single-vote route's H30 handling: a write failure surfaces as a 500
    # rather than leaving the in-memory votes silently un-persisted.
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector

    try:
        sync_labels_to_loaded_detector()
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("vote_media_bulk: detector label sync failed")
        abort(500, message=f"Failed to persist votes to detector store: {exc}")

    from vtscore.labels.sync import sync_to_labelset_source

    try:
        sync_to_labelset_source()
    except Exception:
        import logging

        logging.getLogger(__name__).exception("vote_media_bulk: labelset source scheduling failed")

    return {"ok": True, "changed": changed, "missing": missing}


def _resolve_embedder(dataset_embedder_name: str, dataset_media_type: str):
    """Resolve the embedder to use for an add-to-pile upload.

    Prefers the dataset's recorded embedder by name; falls back to the first
    embedder registered for the dataset's media type.  Returns ``None`` when no
    embedder is available (the caller ``abort``-s 400 in that case).
    """
    from vtscore.media import embedders_for_type, get_embedder  # noqa: PLC0415

    if dataset_embedder_name:
        try:
            return get_embedder(dataset_embedder_name)
        except KeyError:
            pass
    avail = embedders_for_type(dataset_media_type)
    return avail[0] if avail else None


def _embed_upload(embedder, file_bytes: bytes, original_filename: str):
    """Embed an add-to-pile upload via a temporary file.

    Writes *file_bytes* to a temp file (preserving the upload's suffix so the
    embedder can sniff the format), embeds it, and cleans the temp file up.
    ``abort``-s 400 when the embedder returns ``None``.
    """
    import tempfile  # noqa: PLC0415

    suffix = Path(original_filename).suffix or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    from vtscore.media.embedder import media_from_path  # noqa: PLC0415

    try:
        embedding = embedder.embed_media(media_from_path(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    if embedding is None:
        abort(400, message="Failed to embed the uploaded file.")
    return embedding


def _make_pile_thumbnail(media_type: str, file_bytes: bytes) -> bytes | None:
    """Precompute the grid/list thumbnail for an add-to-pile upload.

    Dispatches per media type (audio waveform, video frame, image downscale) so
    the request path never decodes the full-resolution upload on a cold tile
    fetch.  Returns ``None`` for types without a thumbnail generator.
    """
    if media_type == "audio":
        from vtscore.media.audio.media_type import generate_waveform_thumbnail  # noqa: PLC0415

        return generate_waveform_thumbnail(file_bytes)
    if media_type == "video":
        from vtscore.media.video.media_type import generate_video_thumbnail  # noqa: PLC0415

        return generate_video_thumbnail(file_bytes)
    if media_type == "image":
        from vtscore.media.image.thumbnail import make_image_thumbnail  # noqa: PLC0415

        result = make_image_thumbnail(file_bytes)
        return result[0] if result is not None else None
    return None


def _read_pile_upload():
    """Parse and validate the add-to-pile multipart request.

    Returns ``(file, file_bytes, label)`` where *file* is the Werkzeug
    ``FileStorage``, *file_bytes* its non-empty contents, and *label* one of
    ``"good"``/``"bad"``.  ``abort``-s 400 on any missing/empty/invalid field.
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

    return file, file_bytes, label


def _insert_or_collide(new_media: dict[str, Any], file_md5: str) -> tuple[list[int], int, bool]:
    """Insert *new_media* under ``_state_lock``, deduping on a concurrent MD5.

    Re-checks the MD5 lookup under the lock immediately before inserting.
    Embedding ran without the lock (it can take seconds), so a concurrent
    request uploading the same bytes may have inserted a media with this MD5 in
    the meantime. Without this re-check, both requests would each insert a fresh
    media, producing duplicates with identical md5/embedding/bytes.
    (logical-bug-audit H32.)

    Returns ``(target_cids, target_id, is_new)``: on a collision the existing
    cids are returned with ``is_new=False``; otherwise the freshly-assigned id
    is returned with ``is_new=True``.
    """
    with _state_lock:
        md5_lookup_now = cached_md5_lookup()
        collided_cids = md5_lookup_now.get(file_md5, [])
        if collided_cids:
            return list(collided_cids), collided_cids[0], False
        target_id = next_media_id(medias)
        new_media["id"] = target_id
        medias[target_id] = new_media
        return [target_id], target_id, True


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
@require_dataset_header
@require_detector_header
def add_media_to_pile():
    """Upload a media file and add it directly to the Good or Bad pile.

    If a media with the same MD5 already exists in the dataset, the existing
    media is voted accordingly. Otherwise the file is embedded using the
    dataset's embedder, inserted as a new media item, and then voted. The
    request body is multipart/form-data with ``file`` (the upload) and
    ``label`` (``"good"`` or ``"bad"``).
    """
    file, file_bytes, label = _read_pile_upload()
    file_md5 = content_md5(file_bytes)

    # First-pass MD5 lookup (outside _state_lock). When this hits we can
    # skip the expensive embedding step and vote the existing media right
    # away. When it misses we still have to re-check under the lock just
    # before insertion (see below) because embedding holds no lock and a
    # concurrent upload of the same bytes could land between the two.
    snap = snapshot_medias()
    md5_lookup = cached_md5_lookup()
    existing_cids = md5_lookup.get(file_md5, [])

    if existing_cids:
        for cid in existing_cids:
            apply_label(cid, label, provenance={"flow": "seed_example"})
        _sync_pile_label_to_storage()
        return {"ok": True, "media_id": existing_cids[0], "is_new": False}

    # No match: embed and insert as new media.
    if not snap:
        abort(400, message="No dataset loaded. Load a dataset first.")

    first_media = next(iter(snap.values()))
    dataset_media_type = first_media.get("media_type", "audio")
    dataset_embedder_name = first_media.get("embedder", "")

    embedder = _resolve_embedder(dataset_embedder_name, dataset_media_type)
    if embedder is None:
        abort(400, message="No embedder available for the current dataset type.")

    original_filename = file.filename or "upload.bin"
    embedding = _embed_upload(embedder, file_bytes, original_filename)

    # Precompute the grid/list thumbnail so the request path never decodes the
    # full-resolution upload on a cold tile fetch (matches the ingest path).
    thumb = _make_pile_thumbnail(dataset_media_type, file_bytes)

    new_media: dict[str, Any] = {
        "media_type": dataset_media_type,
        "embedder": dataset_embedder_name,
        "md5": file_md5,
        "embeddings": init_embeddings(dataset_embedder_name, embedding),
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

    target_cids, target_id, is_new = _insert_or_collide(new_media, file_md5)

    for cid in target_cids:
        apply_label(cid, label, provenance={"flow": "seed_example"})
    _sync_pile_label_to_storage()

    if is_new:
        return {"ok": True, "media_id": target_id, "is_new": True}, 201
    return {"ok": True, "media_id": target_id, "is_new": False}


def _sync_pile_label_to_storage() -> None:
    """Persist a newly-applied add-to-pile label to the detector's storage.

    Mirrors the tail of :func:`vote_media` so the in-memory good/bad-votes
    update also reaches the detector's on-disk labelset and any configured
    :class:`LabelsetSource`. Without this, the label vanishes the next time
    ``ensure_votes_match_active_dataset`` rehydrates the detector from disk.
    (logical-bug-audit H33.)
    """
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector  # noqa: PLC0415

    sync_labels_to_loaded_detector()

    from vtscore.labels.sync import sync_to_labelset_source  # noqa: PLC0415

    sync_to_labelset_source()
