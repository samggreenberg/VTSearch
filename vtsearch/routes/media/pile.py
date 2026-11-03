"""The add-to-pile route: upload a media file straight into the Good or Bad pile.

The multipart ``add-to-pile`` route describes its success body via
``response`` but omits ``arguments`` because the request shape isn't a
single marshmallow schema.

The route registers on :data:`vtsearch.routes.media.list.medias_bp`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import request
from flask_smorest import abort

from vtscore.embedding.media_vectors import init_embeddings
from vtscore.utils.hashing import content_md5
from vtsearch.routes._context import require_dataset_header, require_detector_header
from vtsearch.routes.media.list import medias_bp
from vtsearch.schemas.media import MediaAddToPileResponseSchema
from vtsearch.state import (
    _state_lock,
    apply_label,
    cached_md5_lookup,
    medias,
    next_media_id,
    snapshot_medias,
)


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

    Mirrors the tail of :func:`~vtsearch.routes.media.vote.vote_media` so the
    in-memory good/bad-votes update also reaches the detector's on-disk
    labelset, whose write schedules the push to any configured
    :class:`LabelsetSource`. Without this, the label vanishes the next time
    ``ensure_votes_match_active_dataset`` rehydrates the detector from disk.
    (logical-bug-audit H33.)
    """
    from vtscore.detectors.label_sync import sync_labels_to_loaded_detector  # noqa: PLC0415

    sync_labels_to_loaded_detector()
