"""The ``medias`` blueprint and its listing routes.

Built on ``flask_smorest`` so the JSON-shaped routes appear in the
``/api/openapi.json`` spec.

:data:`medias_bp` carries every ``/api/medias/...`` route.  This module
registers the listing ones (the id listing, batch metadata, and a converter
output's source item); the rest register on the same blueprint from sibling
modules, which :mod:`vtsearch.routes.media` imports after this one:

* :mod:`vtsearch.routes.media.content` - stream a media's bytes, thumbnail
  or text;
* :mod:`vtsearch.routes.media.vote` - set one media's vote, or many at once;
* :mod:`vtsearch.routes.media.pile` - upload a file straight into the Good or
  Bad pile.

The helpers the content routes share live in
:mod:`vtsearch.routes.media._serving`.
"""

from __future__ import annotations

from typing import Any

from flask_smorest import Blueprint, abort

from vtscore.datasets import source_item
from vtscore.embedding.media_vectors import media_embedder_names
from vtscore.media.provenance import CLIP_WINDOW_FIELDS
from vtscore.utils.hits import hit_custom_metadata
from vtsearch.routes._source import abort_source_not_loaded, source_locator
from vtsearch.schemas.media import (
    MediaBatchRequestSchema,
    MediaBatchResponseSchema,
    MediaIdsListResponseSchema,
    MediaSourceBatchEntrySchema,
    MediaSourceBatchRequestSchema,
    MediaSourceResponseSchema,
)
from vtsearch.routes._context import MISSING_DATASET_HEADER, require_dataset_header
from vtsearch.state import get_media, snapshot_medias

medias_bp = Blueprint(
    "medias",
    __name__,
    description="List, fetch, stream, and vote on individual media items.",
)


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
        for clip_key in CLIP_WINDOW_FIELDS:
            if clip_key not in c:
                continue
            if audio_serves_slice and clip_key in ("clip_start", "clip_end"):
                continue
            media_data[clip_key] = c[clip_key]
        result.append(media_data)
    return result


#: Why a media's source cannot be named, by :class:`~vtscore.datasets.source_item.SourceLocation` status.
#: ``source_item.SIBLING_NOT_LOADED`` is not here: it is the 409, not a 404.
_SOURCE_NOT_FOUND_REPLIES = {
    source_item.NOT_DERIVED: ("not_derived", "This item was not made by a converter, so it has no source item."),
    source_item.NO_SIBLING: (
        "no_source_dataset",
        "No dataset you can read from the same import holds this item's source.",
    ),
    source_item.NOT_FOUND: ("source_not_found", "No item in the source dataset matches this item's source file."),
}


@medias_bp.route("/api/medias/<int:media_id>/source")
@medias_bp.response(200, MediaSourceResponseSchema)
@medias_bp.alt_response(
    404,
    description=(
        "Media not found; or it is no converter output (``error_code: not_derived``); or no readable "
        "dataset of its import holds its source type (``no_source_dataset``); or the source dataset "
        "has no matching item (``source_not_found``)."
    ),
)
@medias_bp.alt_response(
    409,
    description=(
        "The source dataset is registered but not loaded (``error_code: source_not_loaded``); "
        "``dataset_id`` names it for ``POST /api/datasets/registry/<id>/load``."
    ),
)
@medias_bp.alt_response(400, description=MISSING_DATASET_HEADER)
@require_dataset_header
def media_source(media_id: int):
    """Resolve a converter output to the item it was made from, in its sibling dataset.

    A face crop's photo, a video frame's video: found as a **media item** in
    the dataset the same multi-dataset import produced for the source type
    (the shared ``import_group``), since the source file itself is usually
    gone.  ``box`` is where the output sat in its source (``source_box``), for
    outlining it.  The sibling must be loaded; when it is not, the 409 names
    it so the client can offer to load it.  Only siblings the current user may
    read are considered.  See :mod:`vtscore.datasets.source_item`.
    """
    media = get_media(media_id)
    if media is None:
        abort(404, message="not found")
    found = source_locator().locate(media)
    if found.status == source_item.SIBLING_NOT_LOADED:
        abort_source_not_loaded(found.dataset_id)
    if found.status != source_item.RESOLVED:
        error_code, message = _SOURCE_NOT_FOUND_REPLIES[found.status]
        abort(404, message=message, error_code=error_code)
    return {"dataset_id": found.dataset_id, "media_id": found.media_id, "box": found.box}


@medias_bp.route("/api/medias/source-batch", methods=["POST"])
@medias_bp.arguments(MediaSourceBatchRequestSchema)
@medias_bp.response(200, MediaSourceBatchEntrySchema(many=True))
@medias_bp.alt_response(
    409,
    description=(
        "A requested id's source dataset is registered but not loaded (``error_code: source_not_loaded``); "
        "``dataset_id`` names it."
    ),
)
@medias_bp.alt_response(400, description=MISSING_DATASET_HEADER)
@require_dataset_header
def media_source_batch(body: dict):
    """Batch form of ``GET /api/medias/<id>/source``, for a whole results page.

    Returns one entry per id that resolved, in request order; ids that do not
    (unknown, not a converter output, no source dataset, no matching item) are
    omitted, as ``POST /api/medias/batch`` omits unknown ids.  If any id needs
    a source dataset that is not loaded the whole request is a 409 naming it:
    load it and ask again.
    """
    snap = snapshot_medias()
    locator = source_locator()
    result: list[dict[str, Any]] = []
    for media_id in body["ids"]:
        media = snap.get(media_id)
        if media is None:
            continue
        found = locator.locate(media)
        if found.status == source_item.SIBLING_NOT_LOADED:
            abort_source_not_loaded(found.dataset_id)
        if found.status == source_item.RESOLVED:
            result.append(
                {"id": media_id, "dataset_id": found.dataset_id, "media_id": found.media_id, "box": found.box}
            )
    return result
