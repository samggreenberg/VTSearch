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
* ``PUT    /api/detectors/registry/<id>/autorun``      -> :class:`DetectorRegistryAutorunRequestSchema` ->
                                                         :class:`DetectorRegistryAutorunResponseSchema`
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
importer's plugin fields, which don't fit a static marshmallow schema (see the
*Open questions / Plugin field endpoints* section of
``docs/plans/openapi-schema.md``).

The labelset-element shape is shared with :mod:`vtsearch.schemas.labels`.
"""

from __future__ import annotations

from marshmallow import Schema, ValidationError, fields, validate

from vtsearch.schemas.labels import LabeledElementSchema


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

    name = fields.String(required=True, validate=validate.Length(min=1))
    media_type = fields.String(required=True, validate=validate.Length(min=1))
    text_query = fields.String(load_default="")
    media_example = fields.String(load_default="")
    examples = fields.List(fields.Dict(), load_default=None, allow_none=True)


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

    new_name = fields.String(required=True, validate=validate.Length(min=1))


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
    new_name = fields.String(required=True, validate=validate.Length(min=1))
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

    The ``loaded`` / ``detector_loaded`` / ``autorun`` / ``last_trained_at``
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
    created_by = fields.String()
    created_at = fields.Float()
    # Access list (mirrors datasets): usernames allowed besides the creator,
    # ``["*"]`` = public. ``is_owner`` is computed per-request for the caller.
    readers = fields.List(fields.String())
    is_owner = fields.Boolean()
    loaded = fields.Boolean()
    detector_loaded = fields.Boolean()
    autorun = fields.Boolean()
    last_trained_at = fields.Float(allow_none=True)
    # Stamped the first time a detector trains against a dataset and
    # persisted on the registry entry, so the smart preload predictor and
    # the dashboard's cross-embedder check both see it without having to
    # load the detector.  Loaded contexts override the persisted value
    # when present.  Empty string for detectors that have never trained.
    embedder = fields.String()

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

    name = fields.String(required=True, validate=validate.Length(min=1))
    media_type = fields.String(required=True, validate=validate.Length(min=1))
    text_query = fields.String(load_default="")
    media_example = fields.String(load_default="")
    trainable = fields.Boolean(load_default=False)
    examples = fields.List(fields.Dict(), load_default=None, allow_none=True)


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

    name = fields.String(required=True, validate=validate.Length(min=1))


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


class DetectorRegistryAutorunRequestSchema(Schema):
    """Body for ``PUT /api/detectors/registry/<id>/autorun``."""

    autorun = fields.Boolean(required=True)


class DetectorRegistryAutorunResponseSchema(Schema):
    """Response for ``PUT /api/detectors/registry/<id>/autorun``."""

    ok = fields.Boolean(required=True)
    autorun = fields.Boolean(required=True)


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


class DetectorCancelResponseSchema(Schema):
    """Response for ``POST /api/detectors/cancel/<task_id>``."""

    ok = fields.Boolean(required=True)


# ---------------------------------------------------------------------------
# Scoring schemas (vtsearch/routes/detectors/scoring.py)
# ---------------------------------------------------------------------------


class _FindLabelResultSchema(Schema):
    """One ``{id, score}`` entry in the ``results`` list returned by find-label."""

    id = fields.Integer(required=True)
    score = fields.Float(required=True)


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


class _HitSchema(Schema):
    """One scored media entry inside an auto-detect / find ``hits`` list.

    The shape is a media dict (filename, type, md5, origin, ...) augmented
    with a ``score`` (and, for ``/api/find``, a ``verdict``). The set of
    media-dict fields is intentionally open; different importers
    populate different keys.
    """

    id = fields.Integer(required=True)
    score = fields.Float(required=True)

    class Meta:
        unknown = "include"


class _AutoDetectResultSchema(Schema):
    """One entry in ``AutoDetectResponseSchema.results``."""

    detector_name = fields.String(required=True)
    threshold = fields.Float(required=True)
    total_hits = fields.Integer(required=True)
    hits = fields.List(fields.Nested(_HitSchema), required=True)
    negative_hits = fields.List(fields.Nested(_HitSchema), required=True)


class AutoDetectRequestSchema(Schema):
    """Body for ``POST /api/auto-detect``.

    The body is optional; omitting ``detector_name`` runs every autorun
    detector for the active dataset's media type. Passing a name restricts
    the run to that one detector (which must already be flagged for
    autorun; otherwise the handler returns 404).
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
    # Present only when an Auto-Find results exporter is configured: the
    # outcome of auto-exporting these results. ``{exporter, success,
    # message?, error?, ...}`` (extra keys like ``filepath`` may appear for
    # file-based exporters). See ``docs/plans/auto-find-settings-tab.md``.
    auto_export = fields.Dict(keys=fields.String(), values=fields.Raw(), required=False)


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
    origin = fields.Dict(allow_none=True)
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


class DetectorLabelsDetailResponseSchema(Schema):
    """Response for ``GET /api/detectors/<name>/labels-detail``."""

    good = fields.List(fields.Nested(_DetectorLabelViewSchema), required=True)
    bad = fields.List(fields.Nested(_DetectorLabelViewSchema), required=True)
    media_type = fields.String(required=True)


class DetectorLabelVoteRequestSchema(Schema):
    """Body for ``POST /api/detectors/<name>/labels/<element_id>/vote``."""

    vote = fields.String(required=True, validate=validate.OneOf(["good", "bad"]))


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
    adopted Find label set). See docs/plans/find-verification-workflow.md."""

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
    # FP/FN at every inclusion from -10..10 over all adopted items.
    sweep = fields.List(fields.Nested(FindStatsSweepPointSchema), required=True)


__all__ = [
    "AutoDetectRequestSchema",
    "AutoDetectResponseSchema",
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
    "DetectorRegistryAutorunRequestSchema",
    "DetectorRegistryAutorunResponseSchema",
    "DetectorRegistryCreateRequestSchema",
    "DetectorRegistryCreateResponseSchema",
    "DetectorRegistryDeleteResponseSchema",
    "DetectorRegistryEntrySchema",
    "DetectorRegistryListResponseSchema",
    "DetectorRegistryLoadRequestSchema",
    "DetectorRegistryLoadResponseSchema",
    "DetectorRegistryRenameRequestSchema",
    "DetectorRegistryRenameResponseSchema",
    "DetectorRegistryUnloadResponseSchema",
    "DetectorRenameRequestSchema",
    "DetectorRenameResponseSchema",
    "DetectorSaveLabelsResponseSchema",
    "DetectorsListResponseSchema",
    "FindCancelResponseSchema",
    "FindCheckLabelsRequestSchema",
    "FindCheckLabelsResponseSchema",
    "FindLabelRequestSchema",
    "FindLabelResponseSchema",
    "FindRequestSchema",
    "FindResponseSchema",
    "FindStatsResponseSchema",
    "FindStatsSweepPointSchema",
    "PendingLabelsetMoveSchema",
]
