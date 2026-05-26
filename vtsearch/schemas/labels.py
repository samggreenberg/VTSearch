"""Schemas for the labels API (``/api/labels/*``).

Covers the three JSON-only routes in ``vtsearch/routes/labels/vote.py``:

* ``GET  /api/labels/export``       - :class:`LabelsExportResponseSchema`
* ``POST /api/labels/import``       - :class:`LabelsImportRequestSchema` →
                                       :class:`LabelsImportResponseSchema`
* ``POST /api/labels/fill-from-sort`` - :class:`FillFromSortRequestSchema` →
                                         :class:`FillFromSortResponseSchema`

Listing / non-plugin-field routes in ``importers.py`` / ``exporters.py``:

* ``GET  /api/exporters``                       - :class:`ExportersListResponseSchema`
* ``POST /api/exporters/export``                - :class:`RunExportRequestSchema` →
                                                   :class:`RunExportResponseSchema`
* ``GET  /api/label-importers``                 - :class:`LabelImportersListResponseSchema`
* ``POST /api/label-importers/ingest-missing``  - :class:`IngestMissingRequestSchema` →
                                                   :class:`IngestMissingResponseSchema`

The per-plugin shape of ``field_values`` on ``POST /api/exporters/export``
is intentionally declared as ``fields.Dict()``: the inner keys vary per
exporter and ``field_values`` is validated inside the handler against the
selected plugin's :attr:`fields` declaration. The multipart-or-JSON
``POST /api/label-importers/import/<importer_name>`` route stays on the
legacy plain-Flask path (no decorator) for the same reason - see the
*Resolved questions / Plugin field endpoints* section of
``docs/plans/openapi-schema.md``.
"""

from __future__ import annotations

from marshmallow import Schema, fields, validate


class LabeledElementSchema(Schema):
    """A single entry in an exported :class:`~vtscore.datasets.labelset.LabelSet`.

    Mirrors :meth:`vtscore.datasets.labelset.LabeledElement.to_dict`. Only
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


# ---------------------------------------------------------------------------
# Exporter and label-importer plugin routes
# (vtsearch/routes/labels/exporters.py, vtsearch/routes/labels/importers.py)
# ---------------------------------------------------------------------------


class _PluginEntrySchema(Schema):
    """Shared shape for plugin-listing endpoints.

    Mirrors :meth:`vtscore.plugins.PluginBase.to_dict`; the ``fields``
    array's inner shape mirrors :meth:`vtscore.plugins.PluginField.to_dict`
    but is declared as ``fields.Dict()`` to avoid duplicating the source
    of truth across schema and dataclass.
    """

    name = fields.String(required=True)
    display_name = fields.String(required=True)
    description = fields.String(required=True)
    icon = fields.String(required=True)
    ui_mode = fields.String(required=True)
    hidden_from_picker = fields.Boolean(required=True)
    # Renamed to avoid shadowing :attr:`marshmallow.Schema.fields` (a
    # ``dict[str, Field]`` registry on the base class). ``data_key`` /
    # ``attribute`` keep the wire name as ``"fields"`` on both load and
    # dump.
    plugin_fields = fields.List(
        fields.Dict(),
        required=True,
        data_key="fields",
        attribute="fields",
    )


class ExporterEntrySchema(_PluginEntrySchema):
    """One entry in ``GET /api/exporters``."""


class RunExportRequestSchema(Schema):
    """Body for ``POST /api/exporters/export``.

    ``field_values`` is permissive (``fields.Dict``) because its keys
    depend on the named exporter; the handler validates the inner shape
    against the selected plugin.
    """

    exporter_name = fields.String(required=True, validate=validate.Length(min=1))
    field_values = fields.Dict(load_default=dict)
    results = fields.Dict(load_default=dict)


class RunExportResponseSchema(Schema):
    """Response for ``POST /api/exporters/export``.

    Each declared key corresponds to an exporter-specific payload field;
    only ``success`` and ``message`` are always present, and the rest
    are documented optionals. ``display_results`` is the GUI exporter's
    pass-through of the auto-detect results dict (or LabelSet).
    """

    success = fields.Boolean(required=True)
    message = fields.String()
    display_results = fields.Raw()


class LabelImporterEntrySchema(_PluginEntrySchema):
    """One entry in ``GET /api/label-importers``."""


class IngestMissingRequestSchema(Schema):
    """Body for ``POST /api/label-importers/ingest-missing``."""

    entries = fields.List(
        fields.Dict(),
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "Label entries whose medias must be re-ingested."},
    )


class IngestMissingResponseSchema(Schema):
    """Response for ``POST /api/label-importers/ingest-missing``.

    ``failed`` lists per-entry failures from the label-application
    pass - see logical-bug-audit H31.  A single entry that raises during
    ``apply_label`` no longer aborts the rest of the batch.
    """

    ingested = fields.Integer(required=True)
    applied = fields.Integer(required=True)
    failed_count = fields.Integer(required=True)
    failed = fields.List(fields.Dict(), required=True)
    message = fields.String(required=True)


class RunLabelImporterResponseSchema(Schema):
    """Response for ``POST /api/label-importers/import/<importer_name>``.

    The route stays on the legacy plain-Flask path (request body is a
    plugin-field shape), but the success body is the same on every
    importer, so we declare it here for cross-reference. Currently
    *not* attached to the route via ``@response`` - kept for the
    eventual unified plugin-field migration.

    ``failed`` carries per-entry application failures (logical-bug-audit
    H31).  ``missing`` still tracks entries whose media couldn't be
    located or ingested; ``failed`` is the new field for entries whose
    ``apply_label`` call raised.
    """

    applied = fields.Integer(required=True)
    skipped = fields.Integer(required=True)
    missing_count = fields.Integer(required=True)
    missing = fields.List(fields.Dict(), required=True)
    ingested = fields.Integer(required=True)
    failed_count = fields.Integer(required=True)
    failed = fields.List(fields.Dict(), required=True)
    message = fields.String(required=True)


__all__ = [
    "ExporterEntrySchema",
    "FillFromSortRequestSchema",
    "FillFromSortResponseSchema",
    "IngestMissingRequestSchema",
    "IngestMissingResponseSchema",
    "LabelImporterEntrySchema",
    "LabeledElementSchema",
    "LabelsExportQuerySchema",
    "LabelsExportResponseSchema",
    "LabelsImportRequestSchema",
    "LabelsImportResponseSchema",
    "RunExportRequestSchema",
    "RunExportResponseSchema",
    "RunLabelImporterResponseSchema",
]
