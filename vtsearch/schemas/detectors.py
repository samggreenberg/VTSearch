"""Schemas for the detector CRUD API (``/api/detectors/*``).

Covers the endpoints exposed by ``vtsearch/routes/detectors/crud.py``:

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


__all__ = [
    "DetectorCombineRequestSchema",
    "DetectorCombineResponseSchema",
    "DetectorCreateRequestSchema",
    "DetectorCreateResponseSchema",
    "DetectorDeleteResponseSchema",
    "DetectorDetailSchema",
    "DetectorExamplesRequestSchema",
    "DetectorExamplesResponseSchema",
    "DetectorRenameRequestSchema",
    "DetectorRenameResponseSchema",
    "DetectorsListResponseSchema",
]
