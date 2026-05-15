"""Flask routes for the Labelset Exporter API.

Endpoints
---------
GET  /api/exporters
    List all registered exporters with their metadata and field definitions.

POST /api/exporters/export
    Run a specific exporter on auto-detect results supplied in the request
    body.  Body (JSON)::

        {
            "exporter_name": "server_json_file",
            "field_values":  {"filepath": "/home/user/results.json"},
            "results":       { ...auto-detect results dict... }
        }

    Returns::

        {"success": true, "message": "...", ...exporter-specific keys...}
"""

from __future__ import annotations

from flask import Blueprint, jsonify

from vtsearch.exporters import get_exporter, list_exporters
from vtsearch.routes._shared import (
    get_json_safe,
    get_plugin_or_404,
    run_plugin_or_error,
    validate_filepath_field,
    validate_required_fields,
)

exporters_bp = Blueprint("exporters", __name__)


# ---------------------------------------------------------------------------
# GET /api/exporters
# ---------------------------------------------------------------------------


@exporters_bp.route("/api/exporters", methods=["GET"])
def get_exporters():
    """Return a list of all registered labelset exporters."""
    return jsonify([exp.to_dict() for exp in list_exporters()])


# ---------------------------------------------------------------------------
# POST /api/exporters/export
# ---------------------------------------------------------------------------


@exporters_bp.route("/api/exporters/export", methods=["POST"])
def run_export():
    """Run the named exporter on the supplied auto-detect results.

    Request body (JSON):

    .. code-block:: json

        {
            "exporter_name": "server_json_file",
            "field_values":  {"filepath": "/home/user/results.json"},
            "results":       {}
        }

    ``field_values`` and ``results`` are both optional – they default to
    empty dicts – but a valid ``exporter_name`` is required.
    """
    data = get_json_safe()

    exporter_name = data.get("exporter_name", "").strip()
    if not exporter_name:
        return jsonify({"error": "exporter_name is required"}), 400

    exporter, err = get_plugin_or_404(get_exporter, list_exporters, exporter_name, "exporter")
    if err:
        return err

    field_values: dict = data.get("field_values", {}) or {}
    results: dict = data.get("results", {}) or {}

    err = validate_required_fields(exporter, field_values)
    if err:
        return err

    err = validate_filepath_field(field_values)
    if err:
        return err

    outcome, err = run_plugin_or_error(exporter, "export", results, field_values)
    if err:
        return err

    return jsonify({"success": True, **outcome})
