"""Flask routes for the Labelset Exporter API.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.

Endpoints
---------
GET  /api/exporters
    List all registered exporters with their metadata and field definitions.

POST /api/exporters/export
    Run a specific exporter on auto-detect results supplied in the request
    body. ``field_values`` is permissive at the schema layer because its
    inner keys depend on the named exporter; the handler validates it
    against the selected plugin's :attr:`fields`. Schema-level failures
    (missing ``exporter_name``) surface as 422; handler-level rejects
    (unknown exporter, missing plugin field, invalid filepath, plugin
    error) keep their original HTTP codes (404 / 400 / 500) with the
    standard ``message`` envelope.
"""

from __future__ import annotations

import logging

from flask_smorest import Blueprint, abort

from vtscore.exporters import get_exporter, list_exporters
from vtsearch.routes._shared import validate_exporter_field_values
from vtsearch.schemas.labels import (
    ExporterEntrySchema,
    RunExportRequestSchema,
    RunExportResponseSchema,
)

logger = logging.getLogger(__name__)

exporters_bp = Blueprint(
    "exporters",
    __name__,
    description="List and run labelset exporters.",
)


@exporters_bp.route("/api/exporters", methods=["GET"])
@exporters_bp.response(200, ExporterEntrySchema(many=True))
def get_exporters():
    """Return a list of all registered labelset exporters."""
    from vtsearch.settings import filter_visible_plugins

    return [exp.to_dict() for exp in filter_visible_plugins("exporters", list_exporters())]


@exporters_bp.route("/api/exporters/export", methods=["POST"])
@exporters_bp.arguments(RunExportRequestSchema)
@exporters_bp.response(200, RunExportResponseSchema)
@exporters_bp.alt_response(400, description="Missing plugin field or invalid filepath.")
@exporters_bp.alt_response(404, description="Unknown exporter name.")
@exporters_bp.alt_response(500, description="Exporter raised an unexpected error.")
def run_export(body: dict):
    """Run the named exporter on the supplied auto-detect results."""
    exporter_name = body["exporter_name"].strip()
    if not exporter_name:
        abort(422, message="exporter_name is required")

    exporter = get_exporter(exporter_name)
    if exporter is None:
        known = [p.name for p in list_exporters()]
        abort(404, message=f"Unknown exporter '{exporter_name}'. Available: {known}")

    field_values = validate_exporter_field_values(exporter, dict(body.get("field_values") or {}))
    results: dict = dict(body.get("results") or {})

    try:
        outcome = exporter.export(results, field_values)
    except ValueError as exc:
        abort(400, message=str(exc))
    except Exception as exc:
        logger.exception("%s.export() failed: %s", type(exporter).__name__, exc)
        abort(500, message=str(exc))

    return {"success": True, **(outcome or {})}
