"""Flask routes for the Settings Import/Export API.

Endpoints
---------
GET  /api/settings-importers
    List all registered settings importers with their metadata and fields.

POST /api/settings-importers/import/<importer_name>
    Run the named settings importer and apply the imported settings.

GET  /api/settings-exporters
    List all registered settings exporters with their metadata and fields.

POST /api/settings-exporters/export
    Run the named settings exporter on the current application settings.
"""

from __future__ import annotations

import logging

from flask import Blueprint, jsonify

from vtsearch import settings

logger = logging.getLogger(__name__)
from vtsearch.routes.helpers import (
    extract_plugin_fields,
    get_json_safe,
    get_plugin_or_404,
    run_plugin_or_error,
    validate_filepath_field,
    validate_required_fields,
)
from vtsearch.settings_io.exporters import get_settings_exporter, list_settings_exporters
from vtsearch.settings_io.importers import get_settings_importer, list_settings_importers

settings_io_bp = Blueprint("settings_io", __name__)


# ---------------------------------------------------------------------------
# GET /api/settings-importers
# ---------------------------------------------------------------------------


@settings_io_bp.route("/api/settings-importers", methods=["GET"])
def get_settings_importers():
    """Return a list of all registered settings importers."""
    return jsonify([imp.to_dict() for imp in list_settings_importers()])


# ---------------------------------------------------------------------------
# POST /api/settings-importers/import/<importer_name>
# ---------------------------------------------------------------------------


@settings_io_bp.route("/api/settings-importers/import/<importer_name>", methods=["POST"])
def run_settings_import(importer_name: str):
    """Run the named settings importer and apply imported settings."""
    importer, err = get_plugin_or_404(
        get_settings_importer, list_settings_importers, importer_name, "settings importer"
    )
    if err:
        return err

    field_values = extract_plugin_fields(importer)

    err = validate_required_fields(importer, field_values)
    if err:
        return err

    err = validate_filepath_field(field_values)
    if err:
        return err

    imported_settings, err = run_plugin_or_error(importer, "run", field_values)
    if err:
        return err

    if not isinstance(imported_settings, dict):
        return jsonify({"error": "Importer did not return a valid settings dict"}), 500

    # Apply imported settings through the settings API
    _apply_settings(imported_settings)

    return jsonify({
        "success": True,
        "message": f"Imported {len(imported_settings)} setting(s) via {importer.display_name}.",
        "keys": list(imported_settings.keys()),
    })


# ---------------------------------------------------------------------------
# GET /api/settings-exporters
# ---------------------------------------------------------------------------


@settings_io_bp.route("/api/settings-exporters", methods=["GET"])
def get_settings_exporters():
    """Return a list of all registered settings exporters."""
    return jsonify([exp.to_dict() for exp in list_settings_exporters()])


# ---------------------------------------------------------------------------
# POST /api/settings-exporters/export
# ---------------------------------------------------------------------------


@settings_io_bp.route("/api/settings-exporters/export", methods=["POST"])
def run_settings_export():
    """Run the named settings exporter on current application settings."""
    data = get_json_safe()

    exporter_name = data.get("exporter_name", "").strip()
    if not exporter_name:
        return jsonify({"error": "exporter_name is required"}), 400

    exporter, err = get_plugin_or_404(
        get_settings_exporter, list_settings_exporters, exporter_name, "settings exporter"
    )
    if err:
        return err

    field_values: dict = data.get("field_values", {}) or {}

    # Validate required fields
    missing = [
        f.key
        for f in exporter.fields
        if f.required and (field_values.get(f.key) is None or not str(field_values.get(f.key, "")).strip())
    ]
    if missing:
        return jsonify({"error": f"Missing required field(s): {missing}", "missing_fields": missing}), 400

    err = validate_filepath_field(field_values)
    if err:
        return err

    settings_data = settings.get_all()

    try:
        outcome = exporter.export(settings_data, field_values)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logger.exception("Settings export failed (%s): %s", exporter_name, exc)
        return jsonify({"error": f"Export failed: {exc}"}), 500

    return jsonify({"success": True, **outcome})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Re-export from settings module for backward compatibility and local use.
from vtsearch.settings import _apply_settings  # noqa: F401
