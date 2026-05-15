"""Schemas for the labels API (``/api/labels/*``).

Covers the three JSON-only routes in ``vtsearch/routes/labels/vote.py``:

* ``GET  /api/labels/export``       — :class:`LabelsExportResponseSchema`
* ``POST /api/labels/import``       — :class:`LabelsImportRequestSchema` →
                                       :class:`LabelsImportResponseSchema`
* ``POST /api/labels/fill-from-sort`` — :class:`FillFromSortRequestSchema` →
                                         :class:`FillFromSortResponseSchema`

The plugin-field routes in ``importers.py`` / ``exporters.py`` are
deferred — see the *Open questions* section of
``docs/plans/openapi-schema.md`` (plugin field endpoints).
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


class LabeledElementSchema(Schema):
    """A single entry in an exported :class:`~vtsearch.datasets.labelset.LabelSet`.

    Mirrors :meth:`vtsearch.datasets.labelset.LabeledElement.to_dict`. Only
    ``md5`` and ``label`` are guaranteed; the other keys appear when the
    underlying element has them set. ``is_correction`` and
    ``custom_metadata`` are added by the export route (under
    ``find_initial_labels`` and ``enrich=true`` respectively).
    """

    md5 = fields.String(required=True)
    label = fields.String(required=True, metadata={"description": "``good`` or ``bad``."})
    origin = fields.Dict(allow_none=True)
    origin_name = fields.String()
    filename = fields.String()
    category = fields.String()
    metadata = fields.Dict(allow_none=True)
    region_box = fields.List(fields.Float())
    is_correction = fields.Boolean()
    custom_metadata = fields.Dict()

    class Meta:
        # Element-level extension keys (e.g. enrichment-added columns)
        # flow through on dump without being dropped.
        unknown = "include"


class LabelsExportQuerySchema(Schema):
    """Query string for ``GET /api/labels/export``."""

    goods_only = fields.Boolean(load_default=False, metadata={"description": "If true, export only good labels."})
    label_filter = fields.String(
        load_default="",
        validate=validate.OneOf(["", "good", "bad", "both", "corrections"]),
        metadata={
            "description": (
                "Filter mode: ``good``, ``bad``, ``both`` (default), or "
                "``corrections`` (entries where the user changed the "
                "detector's original label). Overrides ``goods_only``."
            ),
        },
    )
    enrich = fields.Boolean(
        load_default=False,
        metadata={
            "description": (
                "If true, include per-entry ``custom_metadata`` and a top-level ``available_columns`` list."
            ),
        },
    )

    class Meta:
        # Tolerate unrelated query params (e.g. ``dataset_id`` /
        # ``detector_id`` used by the request-context middleware).
        unknown = "exclude"


class LabelsExportResponseSchema(Schema):
    """Response for ``GET /api/labels/export``.

    The shape is :meth:`LabelSet.to_dict` plus optional ``available_columns``
    when ``enrich=true``.
    """

    labels = fields.List(fields.Nested(LabeledElementSchema), required=True)
    available_columns = fields.List(fields.String())

    class Meta:
        unknown = "include"


class LabelsImportRequestSchema(Schema):
    """Body for ``POST /api/labels/import``.

    Per-entry parsing happens inside the route (which gracefully skips
    entries with unknown / wrong-typed labels). The schema only enforces
    "the top-level ``labels`` value must be a list of objects"; the
    legacy permissive per-entry handling is preserved.
    """

    labels = fields.List(fields.Dict(), required=True)

    class Meta:
        # Accept legacy keys (``dataset_creation_info``, etc.) for
        # round-tripping with the export endpoint.
        unknown = "include"


class LabelsImportResponseSchema(Schema):
    """Response for ``POST /api/labels/import``."""

    applied = fields.Integer(required=True)
    skipped = fields.Integer(required=True)


class FillFromSortRequestSchema(Schema):
    """Body for ``POST /api/labels/fill-from-sort``."""

    sort_results = fields.List(
        fields.Dict(),
        required=True,
        metadata={"description": "List of ``{id, score}`` dicts from a sort run."},
    )
    threshold = fields.Float(required=True)
    sides = fields.String(
        load_default="good",
        validate=validate.OneOf(["good", "bad", "both"]),
    )
    confirm = fields.Boolean(
        load_default=False,
        metadata={
            "description": "If false (default), return counts only. If true, apply the labels.",
        },
    )


class FillFromSortResponseSchema(Schema):
    """Combined response for ``POST /api/labels/fill-from-sort``.

    Dry run (``confirm=false``) returns ``good_count`` / ``bad_count``.
    Confirmed (``confirm=true``) returns ``good_applied`` / ``bad_applied``
    plus a ``results`` dict suitable for any exporter. All fields are
    declared optional because the two shapes are disjoint.
    """

    good_count = fields.Integer()
    bad_count = fields.Integer()
    good_applied = fields.Integer()
    bad_applied = fields.Integer()
    results = fields.Dict()


__all__ = [
    "FillFromSortRequestSchema",
    "FillFromSortResponseSchema",
    "LabeledElementSchema",
    "LabelsExportQuerySchema",
    "LabelsExportResponseSchema",
    "LabelsImportRequestSchema",
    "LabelsImportResponseSchema",
]
