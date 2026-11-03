"""Schemas for importer field options and suggested names.

``POST /api/dataset/import/<importer_name>/options`` and
``/suggested-name`` are staging routes, but the same request and response
shapes back the label-importer, label-exporter, media and settings-io
routes, so they get a module of their own rather than one route group's.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


class ImporterFieldOptionsRequestSchema(Schema):
    """Body for ``POST /api/dataset/import/<importer_name>/options``."""

    field_key = fields.String(required=True, validate=validate.Length(min=1))
    values = fields.Dict(load_default=dict)


class FieldOptionsSchema(Schema):
    """A single dropdown option for a dynamic-options field.

    ``value`` is what the form submits; ``label`` is the friendly text
    shown in the dropdown.  For plain-string options the two coincide; for
    ``(value, label)`` tuple options they differ so a dropdown can submit
    an opaque id while displaying a human-readable name.
    """

    value = fields.String(required=True)
    label = fields.String(required=True)


class ImporterFieldOptionsResponseSchema(Schema):
    """Response for ``POST /api/dataset/import/<importer_name>/options``."""

    options = fields.List(fields.Nested(FieldOptionsSchema), required=True)


class ImporterSuggestedNameRequestSchema(Schema):
    """Body for ``POST /api/dataset/import/<importer_name>/suggested-name``.

    Shared by ``POST /api/label-importers/suggested-name/<importer_name>``,
    which names a detector rather than a dataset.
    """

    values = fields.Dict(load_default=dict)


class ImporterSuggestedNameResponseSchema(Schema):
    """Response for ``POST /api/dataset/import/<importer_name>/suggested-name``.

    ``dataset_name`` is what the importer would name a dataset built from
    the supplied form values, i.e. the value the Dataset Name box is
    prefilled with while the user is still typing.
    """

    dataset_name = fields.String(required=True)


__all__ = [
    "FieldOptionsSchema",
    "ImporterFieldOptionsRequestSchema",
    "ImporterFieldOptionsResponseSchema",
]
