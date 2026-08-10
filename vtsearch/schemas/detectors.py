"""Schemas for the detector CRUD, registry, scoring, find, and label APIs.

CRUD endpoints (``vtsearch/routes/detectors/crud.py``):

* ``GET    /api/detectors``                 -> :class:`DetectorsListResponseSchema`
* ``POST   /api/detectors``                 -> :class:`DetectorCreateRequestSchema` ->
                                              :class:`DetectorCreateResponseSchema`
* ``GET    /api/detectors/<name>``          -> :class:`DetectorDetailSchema`
* ``DELETE /api/detectors/<name>``          -> :class:`DetectorDeleteResponseSchema`
* ``PUT    /api/detectors/<name>/rename``   -> :class:`DetectorRenameRequestSchema` ->
                                              :class:`DetectorRenameResponseSchema`
* ``PUT    /api/detectors/<name>/examples`` -> :class:`DetectorExamplesRequestSchema` ->
                                              :class:`DetectorExamplesResponseSchema`
* ``POST   /api/detectors/combine``         -> :class:`DetectorCombineRequestSchema` ->
                                              :class:`DetectorCombineResponseSchema`

Registry endpoints (``vtsearch/routes/detectors/registry.py``):

* ``GET    /api/detectors/registry``                   -> :class:`DetectorRegistryListResponseSchema`
* ``POST   /api/detectors/registry``                   -> :class:`DetectorRegistryCreateRequestSchema` ->
                                                         :class:`DetectorRegistryCreateResponseSchema`
* ``POST   /api/detectors/registry/load``              -> :class:`DetectorRegistryLoadRequestSchema` ->
                                                         :class:`DetectorRegistryLoadResponseSchema`
* ``POST   /api/detectors/registry/<id>/unload``       -> :class:`DetectorRegistryUnloadResponseSchema`
* ``DELETE /api/detectors/registry/<id>``              -> :class:`DetectorRegistryDeleteResponseSchema`
* ``PUT    /api/detectors/registry/<id>/rename``       -> :class:`DetectorRegistryRenameRequestSchema` ->
                                                         :class:`DetectorRegistryRenameResponseSchema`
* ``PUT    /api/detectors/registry/<id>/autofind``      -> :class:`DetectorRegistryAutofindRequestSchema` ->
                                                         :class:`DetectorRegistryAutofindResponseSchema`
* ``POST   /api/detectors/cancel/<task_id>``           -> :class:`DetectorCancelResponseSchema`

Scoring endpoints (``vtsearch/routes/detectors/scoring.py``):

* ``POST /api/find-label`` -> :class:`FindLabelRequestSchema` ->
                              :class:`FindLabelResponseSchema`
* ``POST /api/auto-detect`` -> :class:`AutoDetectRequestSchema` ->
                              :class:`AutoDetectResponseSchema`

Find endpoints (``vtsearch/routes/detectors/find.py``):

* ``POST /api/find/check-labels`` -> :class:`FindCheckLabelsRequestSchema` ->
                                    :class:`FindCheckLabelsResponseSchema`
* ``POST /api/find`` -> :class:`FindRequestSchema` ->
                      :class:`FindResponseSchema`
* ``POST /api/find/cancel`` -> :class:`FindCancelResponseSchema`

Label endpoints (``vtsearch/routes/detectors/labels.py``):

* ``POST /api/detectors/<name>/labels`` -> :class:`DetectorSaveLabelsResponseSchema`
* ``GET  /api/detectors/<name>/labels-detail`` -> :class:`DetectorLabelsDetailResponseSchema`
* ``POST /api/detectors/<name>/labels/<element_id>/vote`` ->
        :class:`DetectorLabelVoteRequestSchema` -> :class:`DetectorLabelVoteResponseSchema`
* ``GET  /api/detectors/<name>/labels/<element_id>/preview`` and
  ``GET  /api/detectors/<name>/labels/<element_id>/thumbnail`` serve binary
  responses (or a text-content JSON for text media). These routes declare
  their non-default JSON responses via ``alt_response``; the success body is
  not described in the spec.

The ``POST /api/detectors/<name>/import-labels/<importer_name>`` and
``POST /api/detectors/registry/from-labelset/<importer>`` routes stay on the
legacy flask pattern: their request shape depends on the chosen label
importer's plugin fields, which don't fit a static marshmallow schema (see
"Routes absent from the spec" in ``docs/API.md``).

The labelset-element shape is shared with :mod:`vtsearch.schemas.labels`.
"""

from __future__ import annotations

from marshmallow import INCLUDE, Schema, ValidationError, fields, post_dump, validate

from vtsearch.schemas.labels import LabeledElementSchema
from vtsearch.schemas.media import MediaEntrySchema, OriginSchema

#: Upper bound on user-supplied detector names.  A name this long is already
#: past any reasonable display use, and capping it here keeps the derived
#: ``<slug>.json`` (plus its longer ``.tmp`` sibling) comfortably under the
#: filesystem ``NAME_MAX`` (255) so a rename/create can never raise an uncaught
#: ``OSError`` whose message would leak the absolute server path.  The
#: ``_slug`` truncation in ``vtscore.detectors.store`` is the filesystem-level
#: backstop for names that reach the store by other paths.
MAX_NAME_LENGTH = 128


def _list_of_strings(value):
    """Validator: *value* must be a ``list`` whose every entry is a ``str``.

    Mirrors the dataset readers validator so non-string items are rejected at
    the schema layer (422) rather than coerced.
    """
    if not isinstance(value, list) or not all(isinstance(r, str) for r in value):
        raise ValidationError("Must be a list of strings.")


class _ExampleSchema(Schema):
    """An entry in a detector's ``examples`` list.

    ``type`` is ``"text"`` (the value is a query string) or ``"media"``
    (the value is a filename in ``data/example_media/``). The shape is
    intentionally permissive; the route does not validate inner keys
    today, and unknown keys round-trip through.
    """

    type = fields.String(required=True, validate=validate.OneOf(["text", "media"]))
    value = fields.String(required=True)

    class Meta:
        unknown = "include"


class _DetectorSummarySchema(Schema):
    """Summary entry returned by ``GET /api/detectors``."""

    name = fields.String(required=True)
    text_query = fields.String(required=True)
    media_example = fields.String(required=True)
    media_type = fields.String(required=True)
    examples = fields.List(fields.Dict())
    num_labels = fields.Integer(required=True)
    created_at = fields.Float(required=True)
    # The detector's locked embedder type (the *kind* of vector space it
    # trains/scores in): "semantic" / "patch_semantic" / "structural".  Chosen at
    # create time, immutable.  Empty for a legacy detector that has neither a
    # type nor a (migratable) primary name.  See patch-embedder.md →
    # "Per-detector embedder type".
    embedder_type = fields.String()


class DetectorsListResponseSchema(Schema):
    """Response for ``GET /api/detectors``."""

    detectors = fields.List(fields.Nested(_DetectorSummarySchema), required=True)


class _LabelSetSchema(Schema):
    """The ``labelset`` field of an on-disk detector JSON file."""

    labels = fields.List(fields.Nested(LabeledElementSchema), required=True)

    class Meta:
        unknown = "include"


class DetectorDetailSchema(Schema):
    """Full detector returned by ``GET /api/detectors/<name>``.

    Mirrors the on-disk JSON written by ``_write_detector``.
    """

    name = fields.String(required=True)
    text_query = fields.String()
    media_example = fields.String()
    media_type = fields.String()
    examples = fields.List(fields.Dict())
    created_at = fields.Float()
    embedder_type = fields.String()
    labelset = fields.Nested(_LabelSetSchema)
    combined_from = fields.List(fields.String())

    class Meta:
        # Detector files may carry transitional / extension fields; let
        # them flow through on dump rather than silently dropping them.
        unknown = "include"


class DetectorCreateRequestSchema(Schema):
    """Body for ``POST /api/detectors``.

    The route requires ``name`` and ``media_type``; at least one of
    ``text_query``, ``media_example``, or ``examples`` must be present
    (enforced inside the handler; marshmallow can't express
    "at least one of these three" as a single declaration).
    """

    name = fields.String(required=True, validate=validate.Length(min=1, max=MAX_NAME_LENGTH))
    media_type = fields.String(required=True, validate=validate.Length(min=1))
    text_query = fields.String(load_default="")
    media_example = fields.String(load_default="")
    examples = fields.List(fields.Dict(), load_default=None, allow_none=True)
    # The detector's locked embedder type ("semantic" / "patch_semantic" /
    # "structural").  Empty lets the server pick: the sole type a
    # single-type dataset supplies, or a 400 on a multi-type dataset (the client
    # must choose).  A concrete embedder name is also accepted and classified.
    embedder_type = fields.String(load_default="")


class DetectorCreateResponseSchema(Schema):
    """Response for ``POST /api/detectors``."""

    success = fields.Boolean(required=True)
    name = fields.String(required=True)
    text_query = fields.String(required=True)
    media_example = fields.String(required=True)
    media_type = fields.String(required=True)
    examples = fields.List(fields.Dict(), required=True)
    num_labels = fields.Integer(required=True)


class DetectorDeleteResponseSchema(Schema):
    """Response for ``DELETE /api/detectors/<name>``."""

    success = fields.Boolean(required=True)
    name = fields.String(required=True)


class DetectorRenameRequestSchema(Schema):
    """Body for ``PUT /api/detectors/<name>/rename``."""

    new_name = fields.String(required=True, validate=validate.Length(min=1, max=MAX_NAME_LENGTH))


class PendingLabelsetMoveSchema(Schema):
    """Labelset file paths that were orphaned by a detector rename.

    Populated on a rename response when the detector has a configured
    labelset source whose filepath template (``{detector_name}`` /
    ``{detector_id}``) resolves to a different on-disk location after
    the rename, and the OLD file still exists.  The frontend uses
    these paths to prompt the user to move the file and to invoke
    :class:`DetectorLabelsetMoveRequestSchema` on confirmation.
    """

    old_path = fields.String(required=True)
    new_path = fields.String(required=True)


class DetectorRenameResponseSchema(Schema):
    """Response for ``PUT /api/detectors/<name>/rename``."""

    success = fields.Boolean(required=True)
    old_name = fields.String(required=True)
    new_name = fields.String(required=True)
    pending_labelset_move = fields.Nested(PendingLabelsetMoveSchema, allow_none=True, load_default=None)


class DetectorExamplesRequestSchema(Schema):
    """Body for ``PUT /api/detectors/<name>/examples``."""

    examples = fields.List(fields.Dict(), required=True)


class DetectorExamplesResponseSchema(Schema):
    """Response for ``PUT /api/detectors/<name>/examples``."""

    success = fields.Boolean(required=True)
    name = fields.String(required=True)
    examples = fields.List(fields.Dict(), required=True)


class DetectorCombineRequestSchema(Schema):
    """Body for ``POST /api/detectors/combine``."""

    names = fields.List(
        fields.String(),
        required=True,
        validate=validate.Length(min=2, error="names must be a list of at least 2 detector names"),
    )
    new_name = fields.String(required=True, validate=validate.Length(min=1, max=MAX_NAME_LENGTH))
    conflict_policy = fields.String(load_default="drop")


class DetectorCombineResponseSchema(Schema):
    """Response for ``POST /api/detectors/combine``."""

    success = fields.Boolean(required=True)
    name = fields.String(required=True)
    media_type = fields.String(required=True)
    num_labels = fields.Integer(required=True)
    combined_from = fields.List(fields.String(), required=True)
    source_label_counts = fields.List(fields.Integer(), required=True)
    examples = fields.List(fields.Dict(), required=True)


# ---------------------------------------------------------------------------
# Registry schemas (vtsearch/routes/detectors/registry.py)
# ---------------------------------------------------------------------------


class DetectorRegistryEntrySchema(Schema):
    """One entry in the in-memory detector registry.

    The ``loaded`` / ``detector_loaded`` / ``autofind`` / ``last_trained_at``
    fields are added by ``GET /api/detectors/registry`` (they are not
    persisted on the registry entry itself). They are absent on the
    response from ``POST /api/detectors/registry`` (registration). Both
    cases reuse this schema; consumers should treat all four as optional.
    """

    id = fields.String(required=True)
    name = fields.String(required=True)
    media_type = fields.String(required=True)
    num_training = fields.Integer()
    text_query = fields.String()
    media_example = fields.String()
    # Seed-example list (``[{"type": "text"|"media", "value": ...}, ...]``),
    # mirroring the detector JSON.  The dashboard hands it to the label
    # session so Autopilot's example sort can use every media example.
    examples = fields.List(fields.Dict())
    created_by = fields.String()
    created_at = fields.Float()
    # Access list (mirrors datasets): usernames allowed besides the creator,
    # ``["*"]`` = public. ``is_owner`` is computed per-request for the caller.
    readers = fields.List(fields.String())
    is_owner = fields.Boolean()
    loaded = fields.Boolean()
    detector_loaded = fields.Boolean()
    autofind = fields.Boolean()
    last_trained_at = fields.Float(allow_none=True)
    # Stamped the first time a detector trains against a dataset and
    # persisted on the registry entry, so the smart preload predictor and
    # the dashboard's cross-embedder check both see it without having to
    # load the detector.  Loaded contexts override the persisted value
    # when present.  Empty string for detectors that have never trained.
    embedder = fields.String()
    # The detector's locked embedder type ("semantic" / "patch_semantic" /
    # "structural"), chosen at create time.  Drives the dashboard's
    # type-based detector/dataset compatibility gate.  See patch-embedder.md →
    # "Per-detector embedder type".
    embedder_type = fields.String()

    class Meta:
        # Registry entries may carry extension keys (e.g. future per-row
        # status flags); let them flow through on dump.
        unknown = "include"


class DetectorRegistryListResponseSchema(Schema):
    """Response for ``GET /api/detectors/registry``."""

    detectors = fields.List(fields.Nested(DetectorRegistryEntrySchema), required=True)


class DetectorRegistryCreateRequestSchema(Schema):
    """Body for ``POST /api/detectors/registry``.

    The route requires a non-empty ``name`` and a specific ``media_type``
    (not the ``"any"`` placeholder). ``media_type='any'`` is rejected
    inside the handler to keep the error message specific.
    """

    name = fields.String(required=True, validate=validate.Length(min=1, max=MAX_NAME_LENGTH))
    media_type = fields.String(required=True, validate=validate.Length(min=1))
    text_query = fields.String(load_default="")
    media_example = fields.String(load_default="")
    trainable = fields.Boolean(load_default=False)
    examples = fields.List(fields.Dict(), load_default=None, allow_none=True)
    # The detector's locked embedder type ("semantic" / "patch_semantic" /
    # "structural").  Empty lets the server pick (sole type a single-type
    # dataset supplies, else 400 on a multi-type dataset).  A concrete embedder
    # name is also accepted and classified.  See patch-embedder.md →
    # "Per-detector embedder type".
    embedder_type = fields.String(load_default="")


class DetectorRegistryCreateResponseSchema(Schema):
    """Response for ``POST /api/detectors/registry``."""

    ok = fields.Boolean(required=True)
    detector = fields.Nested(DetectorRegistryEntrySchema, required=True)


class DetectorRegistryLoadRequestSchema(Schema):
    """Body for ``POST /api/detectors/registry/load``.

    ``detector_id`` is optional and may be ``null``; passing ``null`` (or
    omitting the field) unloads the currently-active detector instead of
    loading a new one.
    """

    detector_id = fields.String(load_default=None, allow_none=True)


class DetectorRegistryLoadResponseSchema(Schema):
    """Response for ``POST /api/detectors/registry/load``.

    The shape varies based on the requested operation:

    * Unload (``detector_id`` was null) → ``labels_restored`` / ``examples_seeded``.
    * Already-loaded detector → same as unload (zero counts).
    * Fresh load → ``task_id`` (the background loader's tracker) + ``message``.

    All four optional keys are declared here so the spec covers every
    branch; consumers should check whichever fields they need.
    """

    ok = fields.Boolean(required=True)
    labels_restored = fields.Integer()
    examples_seeded = fields.Integer()
    message = fields.String()
    task_id = fields.String()


class DetectorRegistryUnloadResponseSchema(Schema):
    """Response for ``POST /api/detectors/registry/<id>/unload``."""

    ok = fields.Boolean(required=True)
    message = fields.String()


class DetectorRegistryDeleteResponseSchema(Schema):
    """Response for ``DELETE /api/detectors/registry/<id>``."""

    ok = fields.Boolean(required=True)


class DetectorRegistryRenameRequestSchema(Schema):
    """Body for ``PUT /api/detectors/registry/<id>/rename``.

    Note the field name: the registry rename uses ``name`` (matching the
    frontend's call site, not ``new_name`` as in the CRUD rename route.
    Renaming both endpoints would be a frontend-visible breaking change,
    so the discrepancy is deliberate.
    """

    name = fields.String(required=True, validate=validate.Length(min=1, max=MAX_NAME_LENGTH))


class DetectorRegistryRenameResponseSchema(Schema):
    """Response for ``PUT /api/detectors/registry/<id>/rename``."""

    ok = fields.Boolean(required=True)
    name = fields.String(required=True)
    pending_labelset_move = fields.Nested(PendingLabelsetMoveSchema, allow_none=True, load_default=None)


class DetectorLabelsetMoveRequestSchema(Schema):
    """Body for ``POST /api/detectors/registry/<id>/labelset-source/move-file``."""

    old_path = fields.String(required=True, validate=validate.Length(min=1))
    new_path = fields.String(required=True, validate=validate.Length(min=1))


class DetectorLabelsetMoveResponseSchema(Schema):
    """Response for ``POST /api/detectors/registry/<id>/labelset-source/move-file``."""

    ok = fields.Boolean(required=True)
    moved = fields.Boolean(required=True)
    old_path = fields.String(required=True)
    new_path = fields.String(required=True)


class DetectorRegistryAutofindRequestSchema(Schema):
    """Body for ``PUT /api/detectors/registry/<id>/autofind``."""

    autofind = fields.Boolean(required=True)


class DetectorRegistryAutofindResponseSchema(Schema):
    """Response for ``PUT /api/detectors/registry/<id>/autofind``."""

    ok = fields.Boolean(required=True)
    autofind = fields.Boolean(required=True)


class DetectorRegistryReadersRequestSchema(Schema):
    """Body for ``PUT /api/detectors/registry/<id>/readers``.

    Declared as ``fields.Raw`` with a custom validator (rather than
    ``fields.List(fields.String())``) so non-string items are rejected as 422
    instead of being silently coerced. Mirrors the dataset readers schema.
    """

    readers = fields.Raw(
        required=True,
        validate=_list_of_strings,
        metadata={
            "description": 'List of usernames; ``["*"]`` makes the detector public.',
            "type": "array",
            "items": {"type": "string"},
        },
    )


class DetectorRegistryReadersResponseSchema(Schema):
    """Response for ``PUT /api/detectors/registry/<id>/readers``."""

    ok = fields.Boolean(required=True)
    readers = fields.List(fields.String(), required=True)


class DetectorRegistryStatsResponseSchema(Schema):
    """Response for ``GET /api/detectors/registry/<id>/stats``.

    Counts and metadata only — never embeddings or MLP weights (see the
    "No Persisted Vectors" rule). ``num_positive_resolved`` is how many of
    the detector's positive labels currently resolve into the active
    dataset (the set the Browse button would project), with
    ``active_dataset_name`` naming that dataset (``""`` when none is loaded).
    """

    name = fields.String(required=True)
    media_type = fields.String(required=True)
    num_positive = fields.Integer(required=True)
    num_negative = fields.Integer(required=True)
    num_total = fields.Integer(required=True)
    num_positive_resolved = fields.Integer(required=True)
    active_dataset_name = fields.String(required=True)
    embedder = fields.String(required=True)
    # The detector's locked embedder type ("semantic" / "patch_semantic" /
    # "structural"); ``""`` for a legacy detector with neither type nor a
    # migratable primary name.
    embedder_type = fields.String(required=True)
    text_query = fields.String(required=True)
    media_example = fields.String(required=True)
    clipper = fields.String(required=True)
    created_at = fields.Raw(allow_none=True)
    last_trained_at = fields.Raw(allow_none=True)
    created_by = fields.String(required=True)
    readers = fields.List(fields.String(), required=True)
    autofind = fields.Boolean(required=True)


class DetectorBrowsePositivesResponseSchema(Schema):
    """Response for ``POST /api/detectors/registry/<id>/browse-positives``.

    ``dataset_id`` is the synthetic, in-memory browse context the canvas
    navigates to; ``task_id`` is the detector-loading task whose progress the
    dashboard row renders while the positives are resolved + embedded.
    """

    ok = fields.Boolean(required=True)
    dataset_id = fields.String(required=True)
    task_id = fields.String(required=True)
    media_type = fields.String(required=True)


class DetectorBrowsePositivesReleaseResponseSchema(Schema):
    """Response for ``POST /api/detectors/registry/<id>/browse-positives/release``."""

    ok = fields.Boolean(required=True)
    released = fields.Boolean(required=True)


class DetectorCancelResponseSchema(Schema):
    """Response for ``POST /api/detectors/cancel/<task_id>``."""

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# Scoring schemas (vtsearch/routes/detectors/scoring.py)
# ---------------------------------------------------------------------------


class _FindLabelResultSchema(Schema):
    """One ``{id, score}`` entry in the ``results`` list returned by find-label.

    Patch-region-aware datasets (DINOv2/v3, EUPE) additionally carry
    ``best_region`` - the normalised ``[x0, y0, x1, y1]`` box of the region
    that drove this media's score - so the gallery thumbnails and focus-view
    Highlight overlay can outline it.  Omitted for single-vector datasets.
    """

    id = fields.Integer(required=True)
    score = fields.Float(required=True)
    best_region = fields.List(fields.Float(), required=False)


class FindLabelRequestSchema(Schema):
    """Body for ``POST /api/find-label``.

    The dataset to score against comes from the request-scoped context
    set by ``before_request`` from the ``X-Dataset-Id`` header (or the
    ``dataset_id`` query param for browser-native requests). The body
    intentionally does not carry a dataset selector; letting the body
    override the header allowed a confused client to score one dataset
    while ``replace_all=True`` wiped a different detector's votes.
    """

    detector_id = fields.String(required=True, validate=validate.Length(min=1))


class FindLabelResponseSchema(Schema):
    """Response for ``POST /api/find-label`` (success path)."""

    ok = fields.Boolean(required=True)
    results = fields.List(fields.Nested(_FindLabelResultSchema), required=True)
    threshold = fields.Float(required=True)
    good_count = fields.Integer(required=True)
    bad_count = fields.Integer(required=True)
    detector_name = fields.String(required=True)


class FindQueueIdsQuerySchema(Schema):
    """Query for ``GET /api/find/queue-ids``."""

    filter = fields.String(
        load_default="unverified_good",
        validate=validate.OneOf(["unverified_good", "good"]),
        metadata={
            "description": (
                "Which Find positive set to return: ``unverified_good`` (the "
                "left work queue — above-cutoff items not yet verified) or "
                "``good`` (verified-good + unverified positives)."
            )
        },
    )

    class Meta:
        # Tolerate request-context params (dataset_id / detector_id) smuggled in
        # as query args, matching the other GET query schemas.
        unknown = "exclude"


class FindQueueIdsResponseSchema(Schema):
    """Response for ``GET /api/find/queue-ids`` — the work-queue ids in rank order."""

    ids = fields.List(fields.Integer(), required=True)
    count = fields.Integer(required=True)


class FindBoundaryNextQuerySchema(Schema):
    """Query for ``GET /api/find/boundary-next``."""

    side = fields.String(
        load_default="above",
        validate=validate.OneOf(["above", "below"]),
        metadata={"description": "Preferred face of the cutoff to serve next; falls back to the other side."},
    )
    # Named ``exclude_id`` (not ``exclude``) because ``exclude`` is a reserved
    # attribute on ``marshmallow.Schema``; ``data_key`` keeps the query param
    # ``?exclude=``.
    exclude_id = fields.Integer(
        data_key="exclude",
        load_default=None,
        allow_none=True,
        metadata={
            "description": "Media id to skip (the item just voted, if its verified-state may not be observed yet)."
        },
    )

    class Meta:
        unknown = "exclude"


class FindBoundaryNextResponseSchema(Schema):
    """Response for ``GET /api/find/boundary-next``.

    ``id``/``side`` are both ``null`` when no unverified item remains on either
    side of the cutoff (the done state).
    """

    id = fields.Integer(required=True, allow_none=True)
    side = fields.String(required=True, allow_none=True, validate=validate.OneOf(["above", "below"]))


class _HitSchema(MediaEntrySchema):
    """One scored media entry inside an auto-detect ``hits`` list.

    A media dict augmented with a ``score``.  It reuses
    :class:`~vtsearch.schemas.media.MediaEntrySchema` so the media half
    is enumerated exactly once, in the place that already vetted which media
    fields are safe to serve.

    That enumeration is load-bearing, not cosmetic: a declared schema **drops**
    undeclared keys on dump (``unknown = "include"`` is load-only), and that is
    the last line of defence keeping a media's embedding vectors out of the
    response.  Do not add a ``post_dump`` passthrough here — it would serve the
    whole media dict, vectors included.
    """

    score = fields.Float(required=True)
    origin = fields.Nested(
        OriginSchema,
        allow_none=True,
        metadata={"description": "Where this item was ingested from; rendered as the results table's Origin column."},
    )


class _AutoDetectResultSchema(Schema):
    """One entry in ``AutoDetectResponseSchema.results``."""

    detector_name = fields.String(required=True)
    threshold = fields.Float(required=True)
    total_hits = fields.Integer(required=True)
    hits = fields.List(fields.Nested(_HitSchema), required=True)
    negative_hits = fields.List(fields.Nested(_HitSchema), required=True)


class _AutoFindExportStatusSchema(Schema):
    """Outcome of auto-exporting an Auto-Find run's results.

    Built by ``_run_autofind_export``: a fixed ``{exporter, success}`` base
    plus ``message`` on success / ``error`` on failure, and then whatever extra
    keys the chosen exporter's outcome dict carried (``filepath`` for
    file-based exporters, and so on).  Those extras are exporter-specific, so
    the base is enumerated and the rest passed through, same as the clipper
    descriptors in :mod:`vtsearch.schemas.datasets`.
    """

    exporter = fields.String(required=True)
    success = fields.Boolean(required=True)
    message = fields.String(metadata={"description": "Human-readable outcome; present on success."})
    error = fields.String(metadata={"description": "Failure reason; present instead of ``message`` on failure."})
    open_url = fields.String(
        metadata={
            "description": (
                "An ``http(s)`` URL the exporter formatted for the browser to open, "
                "offered as a link in the Auto-Detect Results modal. Declared rather "
                "than left to the pass-through below because the frontend acts on it."
            )
        }
    )

    class Meta:
        unknown = INCLUDE

    @post_dump(pass_original=True)
    def _include_exporter_extras(self, data: dict, original: dict, **_: object) -> dict:
        # ``unknown = INCLUDE`` only affects ``load``; a declared schema drops
        # undeclared keys on ``dump``.  Safe to re-merge here: the values come
        # from an exporter's own JSON-serialisable outcome dict, not a media
        # dict, so there are no vectors or raw bytes to leak.
        if isinstance(original, dict):
            for k, v in original.items():
                if k not in data:
                    data[k] = v
        return data


class AutoDetectRequestSchema(Schema):
    """Body for ``POST /api/auto-detect``.

    The body is optional; omitting ``detector_name`` runs every Auto-Find
    detector for the active dataset's media type. Passing a name restricts
    the run to that one detector (which must already be flagged for
    Auto-Find; otherwise the handler returns 404).
    """

    detector_name = fields.String(load_default="")


class AutoDetectResponseSchema(Schema):
    """Response for ``POST /api/auto-detect`` (success path).

    ``results`` is keyed by detector name. flask-smorest dumps ``Dict``
    fields as ``additionalProperties`` in the OpenAPI spec.
    """

    media_type = fields.String(required=True)
    detectors_run = fields.Integer(required=True)
    results = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(_AutoDetectResultSchema),
        required=True,
    )
    # Names from the caller's ``autofind_detectors`` list whose detector file
    # no longer exists on disk (deleted out from under the list, or a stale
    # reference from another user's deletion). Reported so a scheduled run
    # never silently drops a detector the user thinks is still active.
    missing_detectors = fields.List(fields.String(), required=True)
    # Present only when an Auto-Find results exporter is configured: the
    # outcome of auto-exporting these results.
    auto_export = fields.Nested(_AutoFindExportStatusSchema)


# ---------------------------------------------------------------------------
# Find schemas (vtsearch/routes/detectors/find.py)
# ---------------------------------------------------------------------------


class FindCheckLabelsRequestSchema(Schema):
    """Body for ``POST /api/find/check-labels``.

    Both arrays default to empty; the route returns an empty
    ``warnings`` list when either is missing.
    """

    dataset_ids = fields.List(fields.String(), load_default=list)
    detector_ids = fields.List(fields.String(), load_default=list)


class _FindCheckLabelsWarningSchema(Schema):
    """One per-detector resolution warning."""

    detector_name = fields.String(required=True)
    total_labels = fields.Integer(required=True)
    resolved_labels = fields.Integer(required=True)
    failed_labels = fields.Integer(required=True)


class FindCheckLabelsResponseSchema(Schema):
    """Response for ``POST /api/find/check-labels``."""

    warnings = fields.List(fields.Nested(_FindCheckLabelsWarningSchema), required=True)


class FindRequestSchema(Schema):
    """Body for ``POST /api/find``.

    Empty arrays are accepted at the schema layer; the handler rejects
    them with a 400 + ``message`` so it can reset ``find_progress`` to
    idle on the way out (schema-level validation skips the handler
    entirely, leaving the tracker stale).
    """

    dataset_ids = fields.List(fields.String(), load_default=list)
    detector_ids = fields.List(fields.String(), load_default=list)


class _FindVerdictSchema(Schema):
    """One entry in a result row's ``detector_verdicts`` map.

    Verdict is ``"Good"``, ``"Bad"``, ``"Error"``, or ``"N/A"``.
    """

    verdict = fields.String(required=True, validate=validate.OneOf(["Good", "Bad", "Error", "N/A"]))
    score = fields.Float(required=True)


class _FindResultRowSchema(Schema):
    """One row in ``FindResponseSchema.results`` / ``negative_results``."""

    id = fields.Integer(required=True)
    filename = fields.String(required=True)
    md5 = fields.String(required=True)
    origin_name = fields.String(required=True)
    origin = fields.Nested(OriginSchema, allow_none=True)
    dataset_name = fields.String(required=True)
    detector_verdicts = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(_FindVerdictSchema),
        required=True,
    )


class FindResponseSchema(Schema):
    """Response for ``POST /api/find`` (success path)."""

    results = fields.List(fields.Nested(_FindResultRowSchema), required=True)
    negative_results = fields.List(fields.Nested(_FindResultRowSchema), required=True)
    datasets = fields.List(fields.String(), required=True)
    detectors = fields.List(fields.String(), required=True)
    media_type = fields.String(required=True)
    multiple_datasets = fields.Boolean(required=True)
    multiple_detectors = fields.Boolean(required=True)
    total_hits = fields.Integer(required=True)


class FindCancelResponseSchema(Schema):
    """Response for ``POST /api/find/cancel``.

    Signals every in-flight scoring path that reads ``find_progress``
    (``/api/find``, ``/api/find-label``, and ``/api/auto-detect``) to stop
    cooperatively. The progress tracker holds a single cancel event that
    long-running loops poll between iterations.
    """

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# Label schemas (vtsearch/routes/detectors/labels.py)
# ---------------------------------------------------------------------------


class DetectorSaveLabelsResponseSchema(Schema):
    """Response for ``POST /api/detectors/<name>/labels``."""

    success = fields.Boolean(required=True)
    name = fields.String(required=True)
    num_labels = fields.Integer(required=True)


class _DetectorLabelViewSchema(Schema):
    """One element in a detector's labels-detail response.

    Mirrors :func:`vtscore.detectors.labelset_elements.build_element_view`.
    """

    id = fields.String(required=True)
    label = fields.String(required=True, validate=validate.OneOf(["good", "bad"]))
    media_type = fields.String(required=True)
    name = fields.String(required=True)
    filename = fields.String(required=True)
    origin_name = fields.String(required=True)
    md5 = fields.String(required=True)
    cid = fields.Integer(allow_none=True)
    time = fields.Float(required=True)
    score = fields.Float(required=True)
    # Normalised [x0, y0, x1, y1] region box for a region vote, else null. The
    # thumbnail route crops to this box; the frontend uses it only to bust the
    # cached tile when the box changes.
    region_box = fields.List(fields.Float(), allow_none=True)


class DetectorLabelsDetailResponseSchema(Schema):
    """Response for ``GET /api/detectors/<name>/labels-detail``."""

    good = fields.List(fields.Nested(_DetectorLabelViewSchema), required=True)
    bad = fields.List(fields.Nested(_DetectorLabelViewSchema), required=True)
    media_type = fields.String(required=True)


class DetectorLabelVoteRequestSchema(Schema):
    """Body for ``POST /api/detectors/<name>/labels/<element_id>/vote``.

    ``target`` is the absolute end state for the element, not a clicked
    direction: ``"good"`` / ``"bad"`` set the element's label, ``"remove"``
    drops it from the labelset. Sending an absolute target makes repeated
    requests from stale tabs idempotent (logical-bug-audit H1).
    """

    target = fields.String(required=True, validate=validate.OneOf(["good", "bad", "remove"]))


class DetectorLabelVoteResponseSchema(Schema):
    """Response for ``POST /api/detectors/<name>/labels/<element_id>/vote``.

    ``action`` is one of ``"removed"``, ``"flipped"``, ``"unchanged"``, or
    ``"not_found"``.
    """

    ok = fields.Boolean(required=True)
    action = fields.String(required=True)


class FindStatsSweepPointSchema(Schema):
    """One point on the Stats FP/FN-vs-inclusion sweep."""

    inclusion = fields.Integer(required=True)
    threshold = fields.Float(required=True)
    false_pos = fields.Integer(required=True)
    false_neg = fields.Integer(required=True)


class FindStatsResponseSchema(Schema):
    """Response for ``GET /api/find/stats`` (detector evaluation over the
    adopted Find label set)."""

    # Adopted totals over ALL items (unverified flood-filled at the current
    # cutoff, like Export/Browse/ToDataset); verified_count is how many of
    # those the human actually checked.
    total_good = fields.Integer(required=True)
    total_bad = fields.Integer(required=True)
    verified_count = fields.Integer(required=True)
    # 2x2 confusion of adopted label vs. the detector's call (find_initial_labels).
    confirmed_good = fields.Integer(required=True)
    confirmed_bad = fields.Integer(required=True)
    culled_false_pos = fields.Integer(required=True)
    rescued_false_neg = fields.Integer(required=True)
    # Derived rates.
    agreements = fields.Integer(required=True)
    corrections = fields.Integer(required=True)
    agreement_rate = fields.Float(required=True)
    precision = fields.Float(required=True)
    # Run context.
    inclusion = fields.Integer(required=True)
    threshold = fields.Float(required=True)
    # True when the detector's labelset changed (Find corrections folded in +
    # retrain) after this evaluation was scored, so these numbers reflect the
    # previous detector version.  Drives the "out of date" note in the UI.
    stale = fields.Boolean(required=True)
    # FP/FN at every inclusion from -10..10 over all adopted items.
    sweep = fields.List(fields.Nested(FindStatsSweepPointSchema), required=True)


class FindEvidenceCoverageResponseSchema(Schema):
    """Response for ``GET /api/find/evidence-coverage``.

    The cross-user complement to the atlas domain-shift report: how much of the
    active dataset the detector is calling *without labeled evidence behind the
    call*, measured from the detector's own labelset (re-embedded in memory at
    load), so it fires even when the training haystack was never handed over.
    See docs/plans/coverage-atlas.md §6.1 (phase v0).
    """

    # False until the report could be computed (a scored Find run plus a
    # resolvable labelset with cached embeddings); the UI hides the section
    # when False rather than showing zeroes.
    available = fields.Boolean(required=True)
    # Number of scored items the report covers, and the labelset sizes it was
    # calibrated against.
    n_items = fields.Integer(required=True)
    n_pos_labels = fields.Integer(required=True)
    n_neg_labels = fields.Integer(required=True)
    # kNN rank and significance level (mirrors the domain-shift report's scale).
    k = fields.Integer(required=True)
    alpha = fields.Float(required=True)
    # Share with support p-value D < alpha — an evidence vacuum for the predicted
    # class — its expectation under the well-supported null (= alpha), and the
    # binomial z of the excess.
    frac_unsupported = fields.Float(required=True)
    expected_unsupported = fields.Float(required=True)
    z_score = fields.Float(required=True)
    median_support = fields.Float(required=True)
    # Share with trust-score TS < 1 (closer to the other class's evidence) and
    # the median TS.
    frac_low_trust = fields.Float(required=True)
    median_trust = fields.Float(required=True)
    # Headline: the vacuum excess is both statistically clear and practically
    # large (z > 3 and frac_unsupported >= 2*alpha).
    unsupported = fields.Boolean(required=True)


class FindCorrectionsToDetectorResponseSchema(Schema):
    """Response for ``POST /api/find/corrections-to-detector``.

    Reports how many corrections were folded into the active detector's
    labelset and the resulting labelset size.  The current Find session stays
    frozen; the retrained detector applies on the next scoring pass.
    """

    ok = fields.Boolean(required=True)
    name = fields.String(required=True)
    corrections_added = fields.Integer(required=True)
    num_labels = fields.Integer(required=True)


__all__ = [
    "AutoDetectRequestSchema",
    "AutoDetectResponseSchema",
    "DetectorBrowsePositivesReleaseResponseSchema",
    "DetectorBrowsePositivesResponseSchema",
    "DetectorCancelResponseSchema",
    "DetectorCombineRequestSchema",
    "DetectorCombineResponseSchema",
    "DetectorCreateRequestSchema",
    "DetectorCreateResponseSchema",
    "DetectorDeleteResponseSchema",
    "DetectorDetailSchema",
    "DetectorExamplesRequestSchema",
    "DetectorExamplesResponseSchema",
    "DetectorLabelsDetailResponseSchema",
    "DetectorLabelsetMoveRequestSchema",
    "DetectorLabelsetMoveResponseSchema",
    "DetectorLabelVoteRequestSchema",
    "DetectorLabelVoteResponseSchema",
    "DetectorRegistryAutofindRequestSchema",
    "DetectorRegistryAutofindResponseSchema",
    "DetectorRegistryCreateRequestSchema",
    "DetectorRegistryCreateResponseSchema",
    "DetectorRegistryDeleteResponseSchema",
    "DetectorRegistryEntrySchema",
    "DetectorRegistryListResponseSchema",
    "DetectorRegistryLoadRequestSchema",
    "DetectorRegistryLoadResponseSchema",
    "DetectorRegistryRenameRequestSchema",
    "DetectorRegistryRenameResponseSchema",
    "DetectorRegistryStatsResponseSchema",
    "DetectorRegistryUnloadResponseSchema",
    "DetectorRenameRequestSchema",
    "DetectorRenameResponseSchema",
    "DetectorSaveLabelsResponseSchema",
    "DetectorsListResponseSchema",
    "FindBoundaryNextQuerySchema",
    "FindBoundaryNextResponseSchema",
    "FindCancelResponseSchema",
    "FindCheckLabelsRequestSchema",
    "FindCheckLabelsResponseSchema",
    "FindCorrectionsToDetectorResponseSchema",
    "FindEvidenceCoverageResponseSchema",
    "FindLabelRequestSchema",
    "FindLabelResponseSchema",
    "FindQueueIdsQuerySchema",
    "FindQueueIdsResponseSchema",
    "FindRequestSchema",
    "FindResponseSchema",
    "FindStatsResponseSchema",
    "FindStatsSweepPointSchema",
    "PendingLabelsetMoveSchema",
]
