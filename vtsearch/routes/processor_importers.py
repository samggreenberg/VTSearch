"""Flask routes for the Processor Importer API.

Endpoints
---------
GET  /api/processor-importers
    List all registered processor importers with their metadata and field definitions.

POST /api/processor-importers/import/<importer_name>
    Run the named processor importer.  Accepts ``multipart/form-data`` (when the
    importer has a ``"file"`` field) or JSON (for text-only importers).

    The importer returns processor data (weights, threshold, media_type).
    A ``name`` field is required (from form data or JSON body) to save the
    result as a autorun detector.

    Returns::

        {"success": true, "name": "<str>", "media_type": "<str>", ...}
"""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch.auth import get_current_user
from vtsearch.processors.importers import get_processor_importer, list_processor_importers
from vtsearch.routes.helpers import extract_plugin_fields, get_json_safe, get_plugin_or_404, run_plugin_or_error, validate_filepath_field, validate_required_fields
from vtsearch.utils import add_autorun_detector

processor_importers_bp = Blueprint("processor_importers", __name__)


# ---------------------------------------------------------------------------
# GET /api/processor-importers
# ---------------------------------------------------------------------------


@processor_importers_bp.route("/api/processor-importers", methods=["GET"])
def get_processor_importers():
    """Return a list of all registered processor importers."""
    return jsonify([imp.to_dict() for imp in list_processor_importers()])


# ---------------------------------------------------------------------------
# POST /api/processor-importers/import/<importer_name>
# ---------------------------------------------------------------------------


@processor_importers_bp.route("/api/processor-importers/import/<importer_name>", methods=["POST"])
def run_processor_import(importer_name: str):
    """Run the named processor importer and save the result as a autorun detector.

    Accepts ``multipart/form-data`` when the importer has a ``"file"`` field,
    or ``application/json`` for text-only importers.  In both cases the route
    builds a ``field_values`` dict and passes it to
    :meth:`~vtsearch.processors.importers.base.ProcessorImporter.run`.

    A ``name`` field is required to identify the saved detector.

    Returns JSON with ``success``, ``name``, ``media_type``, and any extra
    keys returned by the importer.
    """
    importer, err = get_plugin_or_404(get_processor_importer, list_processor_importers, importer_name, "processor importer")
    if err:
        return err

    field_values = extract_plugin_fields(importer)

    # name comes from form data or JSON body (not a plugin field)
    has_file_fields = any(f.field_type == "file" for f in importer.fields)
    if has_file_fields:
        name = request.form.get("name", "").strip()
    else:
        name = get_json_safe().get("name", "").strip()

    if not name:
        return jsonify({"error": "name is required"}), 400

    err = validate_required_fields(importer, field_values)
    if err:
        return err

    err = validate_filepath_field(field_values)
    if err:
        return err

    result, err = run_plugin_or_error(importer, "run", field_values)
    if err:
        return err

    if not isinstance(result, dict):
        return jsonify({"error": "Importer did not return a dict."}), 500

    media_type = result.get("media_type", "audio")
    weights = result.get("weights")
    threshold = result.get("threshold", 0.5)

    if not weights:
        return jsonify({"error": "Importer result missing 'weights'."}), 500

    # Use suggested name from the importer if the user didn't provide one
    # (already checked above that name is non-empty, but importer may suggest)
    add_autorun_detector(
        name,
        media_type,
        weights,
        threshold,
        created_by=get_current_user(),
        good_origins=result.get("good_origins"),
        bad_origins=result.get("bad_origins"),
        inclusion=result.get("inclusion", 0),
    )

    # Register in the persistent model registry for the dashboard grid.
    from vtsearch.models.registry import find_by_detector_name, register_model

    if not find_by_detector_name(name):
        register_model(
            name=name,
            media_type=media_type,
            trainable=False,
            detector_name=name,
            created_by=get_current_user(),
        )

    response: dict = {
        "success": True,
        "name": name,
        "media_type": media_type,
    }
    # Forward extra keys from the importer (loaded, skipped, etc.)
    for key in ("loaded", "skipped"):
        if key in result:
            response[key] = result[key]

    return jsonify(response)
