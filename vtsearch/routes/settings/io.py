"""Flask routes for the Settings Import/Export API.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``.  See ``docs/plans/openapi-schema.md``.

Endpoints
---------
GET  /api/settings-importers
    List all registered settings importers with their metadata and fields.

POST /api/settings-importers/import/<importer_name>
    Run the named settings importer and apply the imported settings.
    The request body shape depends on the importer plugin and isn't
    described in the OpenAPI spec; runtime validation goes through
    :func:`validate_plugin_args` (per-plugin schema built from the
    importer's :attr:`fields`), so missing required fields / invalid
    select values raise 422.  See *Resolved questions / Plugin field
    endpoints* in ``docs/plans/openapi-schema.md``.

GET  /api/settings-exporters
    List all registered settings exporters with their metadata and fields.

POST /api/settings-exporters/export
    Run the named settings exporter on the current per-user settings.
    Schema-level validation failures (missing ``exporter_name``) surface
    as 422; handler-level rejects (unknown exporter, missing plugin
    field, invalid ``filepath``, exporter raised) keep their HTTP codes
    (404 / 400 / 500) with the standard ``message`` envelope.
"""

from __future__ import annotations

import logging

from flask import jsonify
from flask_smorest import Blueprint, abort

from vtsearch import settings
from vtsearch.routes._shared import (
    get_plugin_or_404,
    run_plugin_or_error,
    validate_filepath_field,
    validate_plugin_args,
)
from vtsearch.schemas.settings_io import (
    RunSettingsExportRequestSchema,
    RunSettingsExportResponseSchema,
    SettingsExporterEntrySchema,
    SettingsImporterEntrySchema,
)
from vtsearch.settings_io.exporters import get_settings_exporter, list_settings_exporters
from vtsearch.settings_io.importers import get_settings_importer, list_settings_importers

logger = logging.getLogger(__name__)


settings_io_bp = Blueprint(
    "settings_io",
    __name__,
    description="List and run settings importers / exporters.",
)


# ---------------------------------------------------------------------------
# GET /api/settings-importers
# ---------------------------------------------------------------------------


@settings_io_bp.route("/api/settings-importers", methods=["GET"])
@settings_io_bp.response(200, SettingsImporterEntrySchema(many=True))
def get_settings_importers():
    """Return a list of all registered settings importers."""
    from vtsearch.settings import filter_visible_plugins

    return [imp.to_dict() for imp in filter_visible_plugins("settings_importers", list_settings_importers())]


# ---------------------------------------------------------------------------
# POST /api/settings-importers/import/<importer_name>
#
# Plugin-field route — body shape depends on the importer plugin and isn't
# described in the OpenAPI spec.  Runtime validation goes through
# :func:`validate_plugin_args` (per-plugin schema built from the importer's
# :attr:`fields`), so missing required fields / invalid select values
# raise 422.  See ``docs/plans/openapi-schema.md`` (Resolved questions /
# Plugin field endpoints).
# ---------------------------------------------------------------------------


@settings_io_bp.route("/api/settings-importers/import/<importer_name>", methods=["POST"])
def run_settings_import(importer_name: str):
    """Run the named settings importer and apply imported settings.

    Plugin-dependent body shape: not described in the OpenAPI spec.
    """
    importer, err = get_plugin_or_404(
        get_settings_importer, list_settings_importers, importer_name, "settings importer"
    )
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values = validate_plugin_args(importer)

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

    return jsonify(
        {
            "success": True,
            "message": f"Imported {len(imported_settings)} setting(s) via {importer.display_name}.",
            "keys": list(imported_settings.keys()),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/settings-exporters
# ---------------------------------------------------------------------------


@settings_io_bp.route("/api/settings-exporters", methods=["GET"])
@settings_io_bp.response(200, SettingsExporterEntrySchema(many=True))
def get_settings_exporters():
    """Return a list of all registered settings exporters."""
    from vtsearch.settings import filter_visible_plugins

    return [exp.to_dict() for exp in filter_visible_plugins("settings_exporters", list_settings_exporters())]


# ---------------------------------------------------------------------------
# POST /api/settings-exporters/export
# ---------------------------------------------------------------------------


@settings_io_bp.route("/api/settings-exporters/export", methods=["POST"])
@settings_io_bp.arguments(RunSettingsExportRequestSchema)
@settings_io_bp.response(200, RunSettingsExportResponseSchema)
@settings_io_bp.alt_response(400, description="Missing plugin field or invalid filepath.")
@settings_io_bp.alt_response(404, description="Unknown exporter name.")
@settings_io_bp.alt_response(500, description="Exporter raised an unexpected error.")
def run_settings_export(body: dict):
    """Run the named settings exporter on the current per-user settings."""
    exporter_name = body["exporter_name"].strip()
    if not exporter_name:
        abort(422, message="exporter_name is required")

    exporter = get_settings_exporter(exporter_name)
    if exporter is None:
        known = [p.name for p in list_settings_exporters()]
        abort(404, message=f"Unknown settings exporter '{exporter_name}'. Available: {known}")

    field_values: dict = dict(body.get("field_values") or {})

    missing = [
        f.key
        for f in exporter.fields
        if f.required and (field_values.get(f.key) is None or not str(field_values.get(f.key, "")).strip())
    ]
    if missing:
        abort(400, message=f"Missing required field(s): {missing}", missing_fields=missing)

    if "filepath" in field_values and str(field_values["filepath"]).strip():
        import vtscore.security.path_validation as _paths

        try:
            _paths.validate_server_filepath(str(field_values["filepath"]), base_dir=_paths.get_file_access_base_dir())
        except ValueError as exc:
            abort(400, message=str(exc))

    # Export only the current user's per-user settings, not the merged
    # view (which would also carry shared server-tier infra keys).
    settings_data = settings.get_user_settings()

    try:
        outcome = exporter.export(settings_data, field_values)
    except ValueError as exc:
        abort(400, message=str(exc))
    except Exception as exc:
        logger.exception("Settings export failed (%s): %s", exporter_name, exc)
        abort(500, message=f"Export failed: {exc}")

    return {"success": True, **(outcome or {})}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Re-export from settings module for backward compatibility and local use.
from vtsearch.settings import _apply_settings  # noqa: F401, E402
