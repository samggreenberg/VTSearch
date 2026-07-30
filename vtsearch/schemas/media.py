"""Schemas for the media-handling routes (list / server / embed).

Listing / per-media routes (``vtsearch/routes/media/list.py``):

* ``GET  /api/medias/ids`` -> :class:`MediaIdsListResponseSchema`
* ``POST /api/medias/batch`` -> :class:`MediaBatchRequestSchema` ->
                                :class:`MediaBatchResponseSchema`
* ``POST /api/medias/<id>/vote`` -> :class:`MediaVoteRequestSchema` ->
                                    :class:`MediaVoteResponseSchema`
* ``GET  /api/medias/<id>/paragraph`` and ``GET /api/medias/<id>/text`` ->
        :class:`MediaParagraphResponseSchema`
* ``POST /api/medias/add-to-pile`` -> :class:`MediaAddToPileResponseSchema`
        (multipart body; ``arguments`` decorator omitted because the
        request shape isn't a single marshmallow schema, but the JSON
        success body is described)

The four binary GET routes (``audio``, ``video``, ``image``,
``media``) declare only their JSON error responses via ``alt_response``;
the success body is a streamed file with mimetype chosen at runtime, so
the spec leaves it undescribed (mirroring
``detectors/labels.py``'s preview / thumbnail routes).

Server media files + example-sort routes (``vtsearch/routes/media/server.py``):

* ``GET  /api/server-media-files`` -> :class:`ServerMediaListResponseSchema`
* ``POST /api/server-media-files/upload`` -> :class:`ServerMediaUploadResponseSchema`
        (multipart body; ``arguments`` decorator omitted)
* ``GET  /api/server-media-files/<filename>/thumbnail``: binary,
        :class:`alt_response` only.
* ``POST /api/example-sort-server`` -> :class:`ExampleSortServerRequestSchema` ->
                                       :class:`ExampleSortResponseSchema`
* ``POST /api/example-sort-origin`` -> :class:`ExampleSortOriginRequestSchema` ->
                                       :class:`ExampleSortResponseSchema`
* ``POST /api/example-sort-by-id`` -> :class:`ExampleSortByIdRequestSchema` ->
                                      :class:`ExampleSortResponseSchema`
* ``POST /api/server-media-files/from-media-id`` ->
        :class:`ServerMediaFromMediaIdRequestSchema` ->
        :class:`ServerMediaUploadResponseSchema`

The ``POST /api/embed`` route (``vtsearch/routes/media/embed.py``) is a
dual-mode dispatcher (multipart upload OR JSON body) and is left
undecorated on the same ``flask_smorest.Blueprint``; see the module
docstring there.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


class OriginSchema(Schema):
    """A serialised :class:`~vtscore.datasets.origin.Origin`.

    ``Origin.to_dict`` is the single writer and always emits exactly these two
    keys, so the *envelope* is fixed even though the ``params`` map inside it
    is importer-specific.  Enumerating the envelope is what lets clients read
    ``origin.importer`` directly instead of indexing into an opaque dict.
    """

    importer = fields.String(required=True)
    params = fields.Dict(
        keys=fields.String(),
        metadata={"description": "Identifying import parameters; the key set is importer-specific."},
    )


# ---------------------------------------------------------------------------
# Shared payload-variant query
# ---------------------------------------------------------------------------


class MediaVariantQuerySchema(Schema):
    """Query shared by every per-media payload route.

    ``variant=original`` streams the pre-clean payload of an item a
    :class:`~vtscore.media.cleaner.MediaCleaner` rewrote at load time (see
    ``docs/plans/media-cleaners.md``); the default streams the canonical
    (cleaned) payload.  An item with no snapshot ignores the parameter, so a
    stale link keeps working.
    """

    variant = fields.String(
        load_default="",
        validate=validate.OneOf(["", "original"]),
        metadata={
            "description": (
                "``original`` to stream the pre-clean payload of a cleaned item; "
                "omitted or empty for the canonical (cleaned) payload."
            )
        },
    )

    class Meta:
        # Tolerate the request-context params (``dataset_id`` / ``detector_id``)
        # that browser-native requests smuggle in as query args, plus the
        # thumbnail route's ``region``, matching SortPageQuerySchema.
        unknown = "exclude"


# ---------------------------------------------------------------------------
# /api/medias/ids and /api/medias/batch
# ---------------------------------------------------------------------------


class _MediaIdEntrySchema(Schema):
    """One entry in the lightweight ``GET /api/medias/ids`` listing."""

    id = fields.Integer(required=True)
    media_type = fields.String(required=True)
    embedder = fields.String()
    embedders = fields.List(fields.String())


class MediaIdsListResponseSchema(Schema):
    """Response wrapper for ``GET /api/medias/ids``.

    The handler returns a bare list, so we declare the schema as
    ``many=True`` via the ``Meta`` class. Consumers of the OpenAPI spec
    see a ``type: array`` shape.
    """

    id = fields.Integer(required=True)
    media_type = fields.String(required=True)
    embedder = fields.String()
    embedders = fields.List(fields.String())

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


class MediaEntrySchema(Schema):
    """The per-media metadata block the API serves for one item.

    Used directly by ``POST /api/medias/batch`` (via
    :class:`MediaBatchResponseSchema`) and as the base of the auto-detect hit
    schema (``vtsearch.schemas.detectors._HitSchema``), which adds a score.

    The field list is an **allowlist**, and deliberately so: a declared
    marshmallow schema drops every undeclared key on dump, which is what keeps
    a media's embedding vectors and raw bytes out of the response even if a
    caller forgets to strip them first.  ``custom_metadata`` is the sanctioned
    escape hatch for importer-specific display fields.
    """

    id = fields.Integer(required=True)
    media_type = fields.String(required=True)
    filename = fields.String(required=True)
    md5 = fields.String(required=True)
    custom_metadata = fields.Dict(required=True)
    origin_name = fields.String()
    description = fields.String()
    embedder = fields.String()
    embedders = fields.List(fields.String())
    clip_start = fields.Float()
    clip_end = fields.Float()
    clip_index = fields.Integer()
    clip_box = fields.List(fields.Float())
    has_original = fields.Boolean(
        metadata={
            "description": (
                "Present and ``true`` when a MediaCleaner rewrote this item at "
                "load time and its pre-clean payload was kept. The byte routes "
                "then accept ``?variant=original`` and the detail viewer offers "
                "a Clean/Original toggle. Absent otherwise."
            )
        },
    )

    class Meta:
        # Response-only schema, so this governs nothing at runtime — dump is an
        # allowlist either way.  ``exclude`` rather than ``include`` so the
        # generated OpenAPI model says what the server actually sends: no
        # ``additionalProperties``, hence no index signature for a frontend
        # typo to hide behind.
        unknown = "exclude"


class MediaBatchResponseSchema(MediaEntrySchema):
    """Response wrapper for ``POST /api/medias/batch``.

    Used with ``many=True`` at the decorator call site so the OpenAPI
    description is ``type: array`` of per-media metadata dicts.
    """


# ---------------------------------------------------------------------------
# /api/medias/<id>/vote
# ---------------------------------------------------------------------------


class MediaVoteRequestSchema(Schema):
    """Body for ``POST /api/medias/<id>/vote``.

    The endpoint uses **absolute-target** semantics: ``target`` is the
    state the caller wants the media to be in after the call.  Repeats
    are idempotent; sending ``target="good"`` on a media that's already
    good does nothing, does not append to ``label_history``, and does
    not credit achievements.  This closes the H1 counter-inflation race
    where two stale-view tabs each thought they were toggling the same
    media and the server alternated ADD/REMOVE on every click.

    ``region_box`` is a 4-tuple of normalised image coords (``[x0, y0,
    x1, y1]`` in ``[0, 1]``). Per the patch-embedder v2 design, only
    good votes may carry a region; the handler rejects ``"bad"`` /
    ``"none"`` targets that carry a region_box.

    Unknown fields are silently dropped so the frontend can attach
    advisory keys (e.g. ``confidence``, ``note``) without breaking the
    schema check.
    """

    target = fields.String(
        required=True,
        validate=validate.OneOf(["good", "bad", "none"]),
        metadata={
            "description": ("Absolute target state: ``good``, ``bad``, or ``none`` (un-vote). Idempotent."),
        },
    )
    region_box = fields.List(
        fields.Float(),
        allow_none=True,
        metadata={
            "description": (
                "Optional 4-element ``[x0, y0, x1, y1]`` in normalised image "
                "coordinates ``[0, 1]``. Only valid when ``target`` is ``good``."
            ),
        },
    )

    class Meta:
        unknown = "exclude"


class MediaVoteResponseSchema(Schema):
    """Response for ``POST /api/medias/<id>/vote`` (success path).

    Returns the post-call state so the client can reconcile its optimistic
    view without needing a follow-up ``GET /api/votes``.  ``click_time`` is
    the ordinal assigned when the target is ``good``/``bad`` and the call
    actually changed the state; ``null`` on un-vote and on idempotent calls.
    """

    ok = fields.Boolean(required=True)
    state = fields.String(
        required=True,
        validate=validate.OneOf(["good", "bad", "none"]),
        metadata={"description": "The media's vote state after the call."},
    )
    click_time = fields.Integer(
        required=True,
        allow_none=True,
        metadata={
            "description": (
                "Click-time ordinal of the new label, or ``null`` when the target is ``none`` or the call was a no-op."
            ),
        },
    )


# ---------------------------------------------------------------------------
# /api/medias/vote-bulk
# ---------------------------------------------------------------------------


class MediaVoteBulkRequestSchema(Schema):
    """Body for ``POST /api/medias/vote-bulk``.

    Applies one **absolute target** vote to every id in ``ids`` in a single
    request, persisting the detector labelset once at the end.  Per-id
    semantics match ``/api/medias/<id>/vote`` (idempotent, achievement-gated);
    bulk votes are always image-level, so there is no ``region_box``.  Powers
    the Browser's "Remove from Good" cull, which marks a hand-selected batch
    of false-positives ``bad`` in one shot.

    Unknown fields are silently dropped, matching :class:`MediaVoteRequestSchema`.
    """

    ids = fields.List(
        fields.Integer(),
        required=True,
        metadata={"description": "Media ids to apply the target vote to."},
    )
    target = fields.String(
        required=True,
        validate=validate.OneOf(["good", "bad", "none"]),
        metadata={
            "description": "Absolute target state applied to every id: ``good``, ``bad``, or ``none``. Idempotent.",
        },
    )

    class Meta:
        unknown = "exclude"


class MediaVoteBulkResponseSchema(Schema):
    """Response for ``POST /api/medias/vote-bulk`` (success path)."""

    ok = fields.Boolean(required=True)
    changed = fields.Integer(
        required=True,
        metadata={
            "description": "How many ids actually changed state (idempotent re-applies and missing ids excluded).",
        },
    )
    missing = fields.List(
        fields.Integer(),
        required=True,
        metadata={"description": "Requested ids that were not present in the loaded dataset."},
    )


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

    ``filenames`` carries one or more server-side example files.  With a
    single entry the behaviour is the classic example sort; with several,
    the haystack is ranked against the centroid (mean of the L2-normalised
    embeddings) of all examples.

    ``crop_params`` is free-form (audio: ``{"start", "end"}``; image:
    ``{"box": [...]}``) and validated by the bounded clipper, not by
    this schema.  It applies to a single example, so the handler rejects
    it when more than one filename is given.
    """

    filenames = fields.List(
        fields.String(validate=validate.Length(min=1)),
        required=True,
        validate=validate.Length(min=1),
    )
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


class ExampleSortByIdRequestSchema(Schema):
    """Body for ``POST /api/example-sort-by-id``.

    Sorts the loaded snapshot by similarity to an already-loaded media,
    identified by its in-memory ``media_id``.  When ``crop_params`` is
    absent, the existing ``media["embeddings"]`` vector is reused (no fetch or
    re-embed).  When set, the media's bytes are materialised, cropped,
    and re-embedded before sorting.
    """

    media_id = fields.Integer(required=True)
    crop_params = fields.Dict(allow_none=True)


class ServerMediaFromMediaIdRequestSchema(Schema):
    """Body for ``POST /api/server-media-files/from-media-id``.

    Materialises the bytes of a loaded media to the per-user
    ``example_media/`` directory and returns the saved filename so the
    new-detector flow can reference it as ``media_example``.
    """

    media_id = fields.Integer(required=True)
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
    "ExampleSortByIdRequestSchema",
    "ExampleSortOriginRequestSchema",
    "ExampleSortResponseSchema",
    "ExampleSortServerRequestSchema",
    "MediaAddToPileResponseSchema",
    "MediaBatchRequestSchema",
    "MediaBatchResponseSchema",
    "MediaIdsListResponseSchema",
    "MediaParagraphResponseSchema",
    "MediaVariantQuerySchema",
    "MediaVoteRequestSchema",
    "MediaVoteResponseSchema",
    "ServerMediaFromMediaIdRequestSchema",
    "ServerMediaListResponseSchema",
    "ServerMediaUploadResponseSchema",
]
