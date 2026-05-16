"""Schemas for the media-handling routes (list / server / embed).

Listing / per-media routes (``vtsearch/routes/media/list.py``):

* ``GET  /api/medias/ids`` — :class:`MediaIdsListResponseSchema`
* ``POST /api/medias/batch`` — :class:`MediaBatchRequestSchema` →
                                :class:`MediaBatchResponseSchema`
* ``POST /api/medias/<id>/vote`` — :class:`MediaVoteRequestSchema` →
                                    :class:`MediaVoteResponseSchema`
* ``GET  /api/medias/<id>/paragraph`` and ``GET /api/medias/<id>/text`` —
        :class:`MediaParagraphResponseSchema`
* ``POST /api/medias/add-to-pile`` — :class:`MediaAddToPileResponseSchema`
        (multipart body; ``arguments`` decorator omitted because the
        request shape isn't a single marshmallow schema, but the JSON
        success body is described)

The four binary GET routes (``audio``, ``video``, ``image``,
``media``) declare only their JSON error responses via ``alt_response``;
the success body is a streamed file with mimetype chosen at runtime, so
the spec leaves it undescribed (mirroring
``detectors/labels.py``'s preview / thumbnail routes).

Server media files + example-sort routes (``vtsearch/routes/media/server.py``):

* ``GET  /api/server-media-files`` — :class:`ServerMediaListResponseSchema`
* ``POST /api/server-media-files/upload`` — :class:`ServerMediaUploadResponseSchema`
        (multipart body; ``arguments`` decorator omitted)
* ``GET  /api/server-media-files/<filename>/thumbnail`` — binary,
        :class:`alt_response` only.
* ``POST /api/example-sort-server`` — :class:`ExampleSortServerRequestSchema` →
                                       :class:`ExampleSortResponseSchema`
* ``POST /api/example-sort-origin`` — :class:`ExampleSortOriginRequestSchema` →
                                       :class:`ExampleSortResponseSchema`

The ``POST /api/embed`` route (``vtsearch/routes/media/embed.py``) is a
dual-mode dispatcher (multipart upload OR JSON body) and is left
undecorated on the same ``flask_smorest.Blueprint`` — see the module
docstring there.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


# ---------------------------------------------------------------------------
# /api/medias/ids and /api/medias/batch
# ---------------------------------------------------------------------------


class _MediaIdEntrySchema(Schema):
    """One entry in the lightweight ``GET /api/medias/ids`` listing."""

    id = fields.Integer(required=True)
    type = fields.String(required=True)
    embedder = fields.String()


class MediaIdsListResponseSchema(Schema):
    """Response wrapper for ``GET /api/medias/ids``.

    The handler returns a bare list, so we declare the schema as
    ``many=True`` via the ``Meta`` class. Consumers of the OpenAPI spec
    see a ``type: array`` shape.
    """

    id = fields.Integer(required=True)
    type = fields.String(required=True)
    embedder = fields.String()

    class Meta:
        # The list shape is signalled by ``many=True`` at the decorator
        # call site (`response(200, MediaIdsListResponseSchema(many=True))`).
        pass


class MediaBatchRequestSchema(Schema):
    """Body for ``POST /api/medias/batch``."""

    ids = fields.List(
        fields.Integer(),
        required=True,
        metadata={"description": "Media IDs to fetch metadata for."},
    )


class _MediaBatchEntrySchema(Schema):
    """One entry in the ``POST /api/medias/batch`` response array.

    Importers populate different keys, so unknown fields flow through
    on dump. ``custom_metadata`` is a free-form dict whose inner keys
    vary per importer / media type.
    """

    id = fields.Integer(required=True)
    type = fields.String(required=True)
    filename = fields.String(required=True)
    md5 = fields.String(required=True)
    custom_metadata = fields.Dict(required=True)
    origin_name = fields.String()
    description = fields.String()
    embedder = fields.String()
    clip_start = fields.Float()
    clip_end = fields.Float()
    clip_index = fields.Integer()
    clip_box = fields.List(fields.Float())

    class Meta:
        unknown = "include"


class MediaBatchResponseSchema(_MediaBatchEntrySchema):
    """Response wrapper for ``POST /api/medias/batch``.

    Used with ``many=True`` at the decorator call site so the OpenAPI
    description is ``type: array`` of per-media metadata dicts.
    """


# ---------------------------------------------------------------------------
# /api/medias/<id>/vote
# ---------------------------------------------------------------------------


class MediaVoteRequestSchema(Schema):
    """Body for ``POST /api/medias/<id>/vote``.

    ``region_box`` is a 4-tuple of normalised image coords (``[x0, y0,
    x1, y1]`` in ``[0, 1]``). Per the patch-embedder v2 design, only
    good votes may carry a region; the handler rejects bad votes with a
    region_box.
    """

    vote = fields.String(
        required=True,
        validate=validate.OneOf(["good", "bad"]),
        metadata={"description": "``good`` or ``bad``."},
    )
    region_box = fields.List(
        fields.Float(),
        allow_none=True,
        metadata={
            "description": (
                "Optional 4-element ``[x0, y0, x1, y1]`` in normalised image "
                "coordinates ``[0, 1]``. Only valid on good votes."
            ),
        },
    )


class MediaVoteResponseSchema(Schema):
    """Response for ``POST /api/medias/<id>/vote`` (success path)."""

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# /api/medias/<id>/paragraph and /api/medias/<id>/text
# ---------------------------------------------------------------------------


class MediaParagraphResponseSchema(Schema):
    """Response for ``GET /api/medias/<id>/paragraph`` / ``/text``."""

    content = fields.String(required=True)
    word_count = fields.Integer(required=True)
    character_count = fields.Integer(required=True)


# ---------------------------------------------------------------------------
# /api/medias/add-to-pile
# ---------------------------------------------------------------------------


class MediaAddToPileResponseSchema(Schema):
    """Response for ``POST /api/medias/add-to-pile`` (success path).

    The handler returns 200 when the file's MD5 matches an existing
    media (``is_new = false``) or 201 when a new media is embedded and
    inserted (``is_new = true``). The shape is identical in both cases.
    """

    ok = fields.Boolean(required=True)
    media_id = fields.Integer(required=True)
    is_new = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# /api/server-media-files (CRUD + thumbnail + example-sort)
# ---------------------------------------------------------------------------


class _ServerMediaFileEntrySchema(Schema):
    """One entry in the ``GET /api/server-media-files`` ``files`` array."""

    name = fields.String(required=True, metadata={"description": "File name stem (no extension)."})
    filename = fields.String(required=True, metadata={"description": "Full file name with extension."})
    size_bytes = fields.Integer(required=True)


class ServerMediaListResponseSchema(Schema):
    """Response for ``GET /api/server-media-files``."""

    files = fields.List(fields.Nested(_ServerMediaFileEntrySchema), required=True)


class ServerMediaUploadResponseSchema(Schema):
    """Response for ``POST /api/server-media-files/upload`` (success path).

    ``filename`` is the server-generated UUID name (the persistence
    key); ``original_name`` is the user's original file name, kept for
    display.
    """

    filename = fields.String(required=True)
    original_name = fields.String(required=True)


class ExampleSortServerRequestSchema(Schema):
    """Body for ``POST /api/example-sort-server``.

    ``crop_params`` is free-form (audio: ``{"start", "end"}``; image:
    ``{"box": [...]}``) and validated by the bounded clipper, not by
    this schema.
    """

    filename = fields.String(required=True, validate=validate.Length(min=1))
    crop_params = fields.Dict(allow_none=True)


class ExampleSortOriginRequestSchema(Schema):
    """Body for ``POST /api/example-sort-origin``.

    ``origin`` is a serialised origin dict (``{"importer": ..., "params":
    {...}}``); the inner shape varies per importer and is not validated
    at the schema layer.
    """

    origin = fields.Dict(required=True, metadata={"description": "Origin dict as stored on medias."})
    key = fields.String(required=True, validate=validate.Length(min=1))
    crop_params = fields.Dict(allow_none=True)


class _SortResultEntrySchema(Schema):
    """One scored media in an example-sort ``results`` list.

    Patch-aware embedders (DINOv2/v3, EUPE) also emit ``best_region``;
    single-vector embedders omit it.
    """

    id = fields.Integer(required=True)
    similarity = fields.Float(required=True)
    best_region = fields.List(fields.Float())

    class Meta:
        unknown = "include"


class ExampleSortResponseSchema(Schema):
    """Response for ``POST /api/example-sort-server`` and ``/api/example-sort-origin``."""

    results = fields.List(fields.Nested(_SortResultEntrySchema), required=True)
    threshold = fields.Float(required=True)


__all__ = [
    "ExampleSortOriginRequestSchema",
    "ExampleSortResponseSchema",
    "ExampleSortServerRequestSchema",
    "MediaAddToPileResponseSchema",
    "MediaBatchRequestSchema",
    "MediaBatchResponseSchema",
    "MediaIdsListResponseSchema",
    "MediaParagraphResponseSchema",
    "MediaVoteRequestSchema",
    "MediaVoteResponseSchema",
    "ServerMediaListResponseSchema",
    "ServerMediaUploadResponseSchema",
]
