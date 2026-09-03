"""Shaping a media item into an HTTP response.

Two halves of the same job: the image endpoints that stream a media's pixels
back as a cacheable thumbnail, and the JSON filter that decides which keys of
a media dict are safe to serialize into a response body at all.
"""

from __future__ import annotations

import io

from flask import make_response, request, send_file

from vtscore.embedding.media_vectors import EMBEDDINGS_KEY
from vtscore.utils.hashing import content_md5, new_md5
from vtscore.utils.hits import hit_custom_metadata


def _sniff_image_mimetype(data: bytes) -> str:
    """Best-effort image mimetype from magic bytes (PNG vs JPEG).

    Precomputed thumbnails are always either PNG (alpha / waveform / video
    frame) or JPEG (opaque image), so a two-way sniff is enough; anything
    unrecognised defaults to JPEG.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/jpeg"


def cached_thumbnail_response(thumb_bytes: bytes, download_name: str):
    """Serve already-final thumbnail bytes with an ``ETag`` and cache headers.

    Unlike :func:`image_thumbnail_response`, this does **no** decode/resize:
    the bytes are a thumbnail that was precomputed at ingest (the media dict's
    ``thumbnail_bytes``), so the request path just streams them.  The ``ETag``
    fingerprints the thumbnail bytes so the browser reuses one tile per item
    across scrolls and zoom levels, short-circuiting to a 304.
    """
    etag = content_md5(thumb_bytes)
    if etag in request.if_none_match:
        resp = make_response("", 304)
        resp.set_etag(etag)
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp

    resp = make_response(
        send_file(io.BytesIO(thumb_bytes), mimetype=_sniff_image_mimetype(thumb_bytes), download_name=download_name)
    )
    resp.set_etag(etag)
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


def image_thumbnail_response(
    image_bytes: bytes,
    fallback_mimetype: str,
    download_name: str,
    crop: object = None,
):
    """Build a cached, downscaled-image thumbnail response from ``image_bytes``.

    Shared by every route that serves a small image tile (the media grid/list,
    saved detector labels, server-media examples).  The bytes are run through
    :func:`vtscore.media.image.thumbnail.make_image_thumbnail` so a gallery of
    many high-resolution items never forces the browser to decode every
    full-size bitmap at once; undecodable sources fall back to the original
    bytes.  An ``ETag`` fingerprints the *source* bytes so the browser reuses
    one thumbnail per item across scrolls and zoom levels, and conditional
    requests short-circuit to a 304 without regenerating the thumbnail.

    When ``crop`` is a valid normalised ``(x0, y0, x1, y1)`` region (see
    :func:`vtscore.media.image.thumbnail.normalize_region_crop`), the thumbnail
    shows only that sub-region -- used so a region-voted item displays its crop
    rather than the whole frame.  The crop is folded into the ``ETag`` so a
    re-vote with a different box invalidates the cached tile.
    """
    from vtscore.media.image.thumbnail import (  # noqa: PLC0415
        make_image_thumbnail,
        normalize_region_crop,
    )

    crop_box = normalize_region_crop(crop) if crop is not None else None

    hasher = new_md5()
    hasher.update(image_bytes)
    if crop_box is not None:
        hasher.update(repr(crop_box).encode("ascii"))
    etag = hasher.hexdigest()
    if etag in request.if_none_match:
        resp = make_response("", 304)
        resp.set_etag(etag)
        resp.headers["Cache-Control"] = "private, max-age=86400"
        return resp

    result = make_image_thumbnail(image_bytes, crop=crop_box)
    thumb, mimetype = result if result is not None else (image_bytes, fallback_mimetype)

    resp = make_response(send_file(io.BytesIO(thumb), mimetype=mimetype, download_name=download_name))
    resp.set_etag(etag)
    resp.headers["Cache-Control"] = "private, max-age=86400"
    return resp


#: Media keys never serialized into an API response (large binary/vector data).
#: ``embeddings`` is the v3 dict-keyed vector store
#: (:mod:`vtscore.embedding.media_vectors`); ``embedding`` is its dropped
#: legacy singular form, kept here so a media dict rehydrated from an old
#: pickle can't leak one either.
_HEAVYWEIGHT_KEYS = (
    EMBEDDINGS_KEY,
    "embedding",
    "media_bytes",
    "media_string",
    "thumbnail_bytes",
)


def media_info_for_response(media: dict) -> dict:
    """Return a copy of *media* safe to serialize into an API response.

    Two filters, because one is not enough.  The top-level
    :data:`_HEAVYWEIGHT_KEYS` sweep drops the vectors and the raw bytes, and
    then ``custom_metadata`` is re-derived through
    :func:`vtscore.utils.hits.hit_custom_metadata`: an importer may ship a
    pre-computed vector *nested* inside it via ``custom_metadata_map``, which
    a top-level key filter cannot see and which the free-form
    ``fields.Dict`` in the response schemas waves straight through.  Left in,
    it either balloons the response or fails JSON encoding outright.

    The re-derived dict is fresh, so a route that mutates the returned
    ``custom_metadata`` cannot reach back into the loaded media.
    """
    info = {k: v for k, v in media.items() if k not in _HEAVYWEIGHT_KEYS}
    if isinstance(info.get("custom_metadata"), dict):
        info["custom_metadata"] = hit_custom_metadata(media)
    return info
