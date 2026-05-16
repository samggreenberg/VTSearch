"""Schemas for the detector CRUD and registry APIs (``/api/detectors/*``).

CRUD endpoints (``vtsearch/routes/detectors/crud.py``):

* ``GET    /api/detectors``                 — :class:`DetectorsListResponseSchema`
* ``POST   /api/detectors``                 — :class:`DetectorCreateRequestSchema` →
                                              :class:`DetectorCreateResponseSchema`
* ``GET    /api/detectors/<name>``          — :class:`DetectorDetailSchema`
* ``DELETE /api/detectors/<name>``          — :class:`DetectorDeleteResponseSchema`
* ``PUT    /api/detectors/<name>/rename``   — :class:`DetectorRenameRequestSchema` →
                                              :class:`DetectorRenameResponseSchema`
* ``PUT    /api/detectors/<name>/examples`` — :class:`DetectorExamplesRequestSchema` →
                                              :class:`DetectorExamplesResponseSchema`
* ``POST   /api/detectors/combine``         — :class:`DetectorCombineRequestSchema` →
                                              :class:`DetectorCombineResponseSchema`

Registry endpoints (``vtsearch/routes/detectors/registry.py``):

* ``GET    /api/detectors/registry``                   — :class:`DetectorRegistryListResponseSchema`
* ``POST   /api/detectors/registry``                   — :class:`DetectorRegistryCreateRequestSchema` →
                                                         :class:`DetectorRegistryCreateResponseSchema`
* ``POST   /api/detectors/registry/load``              — :class:`DetectorRegistryLoadRequestSchema` →
                                                         :class:`DetectorRegistryLoadResponseSchema`
* ``POST   /api/detectors/registry/<id>/unload``       — :class:`DetectorRegistryUnloadResponseSchema`
* ``DELETE /api/detectors/registry/<id>``              — :class:`DetectorRegistryDeleteResponseSchema`
* ``PUT    /api/detectors/registry/<id>/rename``       — :class:`DetectorRegistryRenameRequestSchema` →
                                                         :class:`DetectorRegistryRenameResponseSchema`
* ``PUT    /api/detectors/registry/<id>/autorun``      — :class:`DetectorRegistryAutorunRequestSchema` →
                                                         :class:`DetectorRegistryAutorunResponseSchema`
* ``POST   /api/detectors/cancel/<task_id>``           — :class:`DetectorCancelResponseSchema`

The ``POST /api/detectors/registry/from-labelset/<importer>`` route stays on
the legacy flask pattern: its request shape depends on the chosen label
importer's plugin fields, which don't fit a static marshmallow schema (see the
*Open questions / Plugin field endpoints* section of
``docs/plans/openapi-schema.md``).

The labelset-element shape is shared with :mod:`vtsearch.schemas.labels`.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate

from vtsearch.schemas.labels import LabeledElementSchema


class _ExampleSchema(Schema):
    """An entry in a detector's ``examples`` list.

    ``type`` is ``"text"`` (the value is a query string) or ``"media"``
    (the value is a filename in ``data/example_media/``). The shape is
    intentionally permissive — the route does not validate inner keys
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
    (enforced inside the handler — marshmallow can't express
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


class DetectorRenameResponseSchema(Schema):
    """Response for ``PUT /api/detectors/<name>/rename``."""

    success = fields.Boolean(required=True)
    old_name = fields.String(required=True)
    new_name = fields.String(required=True)


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
    loaded = fields.Boolean()
    detector_loaded = fields.Boolean()
    autorun = fields.Boolean()
    last_trained_at = fields.Float(allow_none=True)

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

    ``detector_id`` is optional and may be ``null`` — passing ``null`` (or
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
    frontend's call site) — not ``new_name`` as in the CRUD rename route.
    Renaming both endpoints would be a frontend-visible breaking change,
    so the discrepancy is deliberate.
    """

    name = fields.String(required=True, validate=validate.Length(min=1))


class DetectorRegistryRenameResponseSchema(Schema):
    """Response for ``PUT /api/detectors/registry/<id>/rename``."""

    ok = fields.Boolean(required=True)
    name = fields.String(required=True)


class DetectorRegistryAutorunRequestSchema(Schema):
    """Body for ``PUT /api/detectors/registry/<id>/autorun``."""

    autorun = fields.Boolean(required=True)


class DetectorRegistryAutorunResponseSchema(Schema):
    """Response for ``PUT /api/detectors/registry/<id>/autorun``."""

    ok = fields.Boolean(required=True)
    autorun = fields.Boolean(required=True)


class DetectorCancelResponseSchema(Schema):
    """Response for ``POST /api/detectors/cancel/<task_id>``."""

    ok = fields.Boolean(required=True)


__all__ = [
    "DetectorCancelResponseSchema",
    "DetectorCombineRequestSchema",
    "DetectorCombineResponseSchema",
    "DetectorCreateRequestSchema",
    "DetectorCreateResponseSchema",
    "DetectorDeleteResponseSchema",
    "DetectorDetailSchema",
    "DetectorExamplesRequestSchema",
    "DetectorExamplesResponseSchema",
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
    "DetectorsListResponseSchema",
]
