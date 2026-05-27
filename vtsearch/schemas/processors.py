"""Schemas for the processors API (extractor / localizer / pregen routes).

CRUD endpoints (``vtsearch/routes/processors/crud.py``):

* ``GET    /api/autorun-extractors``              -> :class:`AutorunExtractorsListResponseSchema`
* ``POST   /api/autorun-extractors``              -> :class:`AutorunExtractorCreateRequestSchema` ->
                                                    :class:`AutorunProcessorCreateResponseSchema`
* ``DELETE /api/autorun-extractors/<name>``       -> :class:`AutorunProcessorDeleteResponseSchema`
* ``PUT    /api/autorun-extractors/<name>/rename`` -> :class:`AutorunProcessorRenameRequestSchema` ->
                                                    :class:`AutorunProcessorRenameResponseSchema`
* ``GET    /api/autorun-localizers``              -> :class:`AutorunLocalizersListResponseSchema`
* ``POST   /api/autorun-localizers``              -> :class:`AutorunLocalizerCreateRequestSchema` ->
                                                    :class:`AutorunProcessorCreateResponseSchema`
* ``DELETE /api/autorun-localizers/<name>``       -> :class:`AutorunProcessorDeleteResponseSchema`
* ``PUT    /api/autorun-localizers/<name>/rename`` -> :class:`AutorunProcessorRenameRequestSchema` ->
                                                    :class:`AutorunProcessorRenameResponseSchema`
* ``GET    /api/pregen-processors``               -> :class:`PregenProcessorsListResponseSchema`
* ``POST   /api/pregen-processors/add``           -> :class:`PregenProcessorsAddResponseSchema`

Scoring endpoints (``vtsearch/routes/processors/scoring.py``):

* ``POST /api/extract``       -> :class:`ExtractRequestSchema` -> :class:`ExtractResponseSchema`
* ``POST /api/auto-extract``  -> :class:`AutoExtractResponseSchema`
* ``POST /api/localize``      -> :class:`LocalizeRequestSchema` -> :class:`LocalizeResponseSchema`
* ``POST /api/auto-localize`` -> :class:`AutoLocalizeResponseSchema`

The extract / localize routes accept a request body with the processor's
``config`` dict, whose inner keys vary per processor type. The ``config``
field is declared as a permissive :class:`marshmallow.fields.Dict` so any
shape passes the schema layer; the handler still rejects configs that
don't construct a valid processor.

Note the ``POST /api/auto-extract`` / ``/api/auto-localize`` routes take
no body; they read the autorun store and the active dataset. They're
modelled as plain ``POST`` with no ``arguments`` decorator.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


# ---------------------------------------------------------------------------
# CRUD schemas
# ---------------------------------------------------------------------------


class _AutorunExtractorEntrySchema(Schema):
    """One entry in the autorun-extractor store."""

    name = fields.String(required=True)
    extractor_type = fields.String(required=True)
    media_type = fields.String(required=True)
    config = fields.Dict(required=True)
    created_at = fields.Float(required=True)


class _AutorunLocalizerEntrySchema(Schema):
    """One entry in the autorun-localizer store."""

    name = fields.String(required=True)
    localizer_type = fields.String(required=True)
    media_type = fields.String(required=True)
    config = fields.Dict(required=True)
    created_at = fields.Float(required=True)


class AutorunExtractorsListResponseSchema(Schema):
    """Response for ``GET /api/autorun-extractors``."""

    extractors = fields.List(fields.Nested(_AutorunExtractorEntrySchema), required=True)


class AutorunLocalizersListResponseSchema(Schema):
    """Response for ``GET /api/autorun-localizers``."""

    localizers = fields.List(fields.Nested(_AutorunLocalizerEntrySchema), required=True)


class AutorunExtractorCreateRequestSchema(Schema):
    """Body for ``POST /api/autorun-extractors``."""

    name = fields.String(required=True, validate=validate.Length(min=1))
    media_type = fields.String(required=True, validate=validate.Length(min=1))
    extractor_type = fields.String(required=True, validate=validate.Length(min=1))
    config = fields.Dict(required=True)


class AutorunLocalizerCreateRequestSchema(Schema):
    """Body for ``POST /api/autorun-localizers``."""

    name = fields.String(required=True, validate=validate.Length(min=1))
    media_type = fields.String(required=True, validate=validate.Length(min=1))
    localizer_type = fields.String(required=True, validate=validate.Length(min=1))
    config = fields.Dict(required=True)


class AutorunProcessorCreateResponseSchema(Schema):
    """Response for autorun-extractor / autorun-localizer creation."""

    success = fields.Boolean(required=True)
    name = fields.String(required=True)


class AutorunProcessorDeleteResponseSchema(Schema):
    """Response for ``DELETE /api/autorun-extractors/<name>`` / localizers."""

    success = fields.Boolean(required=True)


class AutorunProcessorRenameRequestSchema(Schema):
    """Body for ``PUT /api/autorun-extractors/<name>/rename`` / localizers."""

    new_name = fields.String(required=True, validate=validate.Length(min=1))


class AutorunProcessorRenameResponseSchema(Schema):
    """Response for autorun-extractor / autorun-localizer rename."""

    success = fields.Boolean(required=True)
    new_name = fields.String(required=True)


class _PregenProcessorEntrySchema(Schema):
    """One entry in the pregen-processor list.

    ``kind`` is ``"extractor"`` or ``"localizer"``; it tells the
    /api/pregen-processors/add handler which autorun store to route the
    processor into.
    """

    name = fields.String(required=True)
    kind = fields.String(required=True, validate=validate.OneOf(["extractor", "localizer"]))
    processor_type = fields.String(required=True)
    media_type = fields.String(required=True)
    config = fields.Dict(required=True)


class PregenProcessorsListResponseSchema(Schema):
    """Response for ``GET /api/pregen-processors``."""

    processors = fields.List(fields.Nested(_PregenProcessorEntrySchema), required=True)


class PregenProcessorsAddResponseSchema(Schema):
    """Response for ``POST /api/pregen-processors/add``."""

    success = fields.Boolean(required=True)
    added = fields.List(fields.String(), required=True)


# ---------------------------------------------------------------------------
# Scoring schemas
# ---------------------------------------------------------------------------


class _ProcessorHitSchema(Schema):
    """One scored media entry in an extract / localize ``results`` list.

    The shape is a media dict augmented with an ``extractions`` or
    ``localizations`` array. The set of media-dict fields is intentionally
    open; different importers populate different keys, and the
    extraction/localization arrays themselves are processor-specific
    (bounding boxes, text strings, etc.).
    """

    id = fields.Integer(required=True)
    extractions = fields.List(fields.Dict())
    localizations = fields.List(fields.Dict())

    class Meta:
        unknown = "include"


class ExtractRequestSchema(Schema):
    """Body for ``POST /api/extract``.

    ``name`` is optional; when absent the handler labels the extractor as
    ``"adhoc"``. ``config`` is processor-specific and validated by the
    factory rather than at the schema layer.
    """

    name = fields.String(load_default="")
    extractor_type = fields.String(required=True, validate=validate.Length(min=1))
    config = fields.Dict(required=True)


class ExtractResponseSchema(Schema):
    """Response for ``POST /api/extract`` (success path)."""

    extractor_name = fields.String(required=True)
    media_type = fields.String(required=True)
    total_medias_with_hits = fields.Integer(required=True)
    results = fields.List(fields.Nested(_ProcessorHitSchema), required=True)


class LocalizeRequestSchema(Schema):
    """Body for ``POST /api/localize``."""

    name = fields.String(load_default="")
    localizer_type = fields.String(required=True, validate=validate.Length(min=1))
    config = fields.Dict(required=True)


class LocalizeResponseSchema(Schema):
    """Response for ``POST /api/localize`` (success path)."""

    localizer_name = fields.String(required=True)
    media_type = fields.String(required=True)
    total_medias_with_hits = fields.Integer(required=True)
    results = fields.List(fields.Nested(_ProcessorHitSchema), required=True)


class _AutoExtractPerProcessorSchema(Schema):
    """One entry in ``AutoExtractResponseSchema.results`` (keyed by name)."""

    extractor_name = fields.String(required=True)
    total_medias_with_hits = fields.Integer(required=True)
    results = fields.List(fields.Nested(_ProcessorHitSchema), required=True)


class AutoExtractResponseSchema(Schema):
    """Response for ``POST /api/auto-extract`` (success path).

    ``results`` is keyed by extractor name. flask-smorest dumps ``Dict``
    fields as ``additionalProperties`` in the OpenAPI spec.
    """

    media_type = fields.String(required=True)
    extractors_run = fields.Integer(required=True)
    results = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(_AutoExtractPerProcessorSchema),
        required=True,
    )


class _AutoLocalizePerProcessorSchema(Schema):
    """One entry in ``AutoLocalizeResponseSchema.results`` (keyed by name)."""

    localizer_name = fields.String(required=True)
    total_medias_with_hits = fields.Integer(required=True)
    results = fields.List(fields.Nested(_ProcessorHitSchema), required=True)


class AutoLocalizeResponseSchema(Schema):
    """Response for ``POST /api/auto-localize`` (success path)."""

    media_type = fields.String(required=True)
    localizers_run = fields.Integer(required=True)
    results = fields.Dict(
        keys=fields.String(),
        values=fields.Nested(_AutoLocalizePerProcessorSchema),
        required=True,
    )


__all__ = [
    "AutoExtractResponseSchema",
    "AutoLocalizeResponseSchema",
    "AutorunExtractorCreateRequestSchema",
    "AutorunExtractorsListResponseSchema",
    "AutorunLocalizerCreateRequestSchema",
    "AutorunLocalizersListResponseSchema",
    "AutorunProcessorCreateResponseSchema",
    "AutorunProcessorDeleteResponseSchema",
    "AutorunProcessorRenameRequestSchema",
    "AutorunProcessorRenameResponseSchema",
    "ExtractRequestSchema",
    "ExtractResponseSchema",
    "LocalizeRequestSchema",
    "LocalizeResponseSchema",
    "PregenProcessorsAddResponseSchema",
    "PregenProcessorsListResponseSchema",
]
