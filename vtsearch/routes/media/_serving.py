"""Serving helpers for the ``medias`` blueprint's route modules.

The non-route plumbing behind the per-media byte routes in
:mod:`vtsearch.routes.media.content`: payload-variant selection, byte and
string resolution, archive-member streaming, HTTP ``Range`` handling, and the
ffmpeg / OpenCV transcode for videos a browser can't play.  Also the
``region_box`` validation the vote route in :mod:`vtsearch.routes.media.vote`
applies.
"""

from __future__ import annotations

import io
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from flask import Response, jsonify, make_response, request, send_file

from vtscore.media.audio.ffmpeg import get_ffmpeg_exe
from vtscore.media.base import MediaResponse
from vtscore.security.path_validation import resolve_media_file_path
from vtsearch.state import get_media

# Extensions the browser's <video> element can play natively.
_BROWSER_VIDEO_EXTS = {".mp4", ".m4v", ".webm", ".ogg", ".ogv"}

# Image filename extension -> served mimetype. Unlisted extensions fall back to
# ``image/jpeg`` (see :func:`~vtsearch.routes.media.content._resolve_display_image`).
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
    the original bytes without any route having to know cleaners exist.  A
    cleaned unit of a reference import keeps no snapshot; its Original is
    replayed from the source instead
    (:func:`~vtscore.media.lazy_clip.lazy_original_bytes`), and is served as
    nothing when that replay no longer matches the import.  Every
    non-inline backing (``media_path`` / ``media_url``) is stripped from the
    view: those point at the *source* file, which the resolver would otherwise
    fall through to and serve as if it were the snapshot.  ``md5`` is dropped
    too, so ETag-emitting routes hash the bytes they actually serve rather than
    labelling the original with the cleaned item's hash.

    ``original`` on a media with no snapshot falls back to the canonical
    payload: a stale bookmark should show the item, not 404.  An unknown
    variant never gets here - every route that calls this validates its query
    with :class:`~vtsearch.schemas.media.MediaVariantQuerySchema`, which
    rejects it with a 422.
    """
    if request.args.get("variant") != "original":
        return media

    from vtscore.datasets.clipper_chain import has_original_payload  # noqa: PLC0415

    if not has_original_payload(media):
        return media

    view = dict(media)
    view["media_bytes"] = media.get("original_media_bytes")
    view["media_string"] = media.get("original_media_string")
    if view["media_bytes"] is None and view["media_string"] is None:
        from vtscore.media.lazy_clip import lazy_original_bytes  # noqa: PLC0415

        replayed = lazy_original_bytes(media)
        if media.get("media_type") == "text":
            view["media_string"] = replayed.decode("utf-8", errors="replace") if replayed is not None else None
        else:
            view["media_bytes"] = replayed
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
    """Return media string, lazy-loading from its recipe or media_path if needed.

    A lazy recipe (see :mod:`vtscore.media.lazy_clip`) comes before the
    whole-file read, since ``media_path`` names its *source*; a cleaned unit
    whose replay no longer matches the import yields ``None``.
    """
    media_string = media.get("media_string")
    if media_string is not None:
        return media_string
    from vtscore.media.lazy_clip import lazy_clip_bytes, recipe_is_exclusive  # noqa: PLC0415

    derived = lazy_clip_bytes(media)
    if derived is not None:
        return derived.decode("utf-8", errors="replace")
    if recipe_is_exclusive(media):
        return None
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
