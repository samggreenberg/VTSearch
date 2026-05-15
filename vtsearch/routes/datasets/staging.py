"""Dataset staging, importer dispatch, and the combine-datasets endpoint."""

from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request

import vtsearch.security.path_validation as _paths
from vtsearch.config import EMBEDDINGS_DIR
from vtsearch.datasets import DEMO_DATASETS, get_importer, list_importers
from vtsearch.datasets.load_pipeline import (
    STAGING_DIR,
    _run_importer_in_background,
    _stage_importer_in_background,
)
from vtsearch.routes._shared import get_plugin_or_404, get_request_field
from vtsearch.routes.datasets._helpers import (
    _extract_clipper_params,
    _extract_importer_fields,
)
from vtsearch.security.pickle import peek_pickle_dataset_summary

datasets_staging_bp = Blueprint("datasets_staging", __name__)


@datasets_staging_bp.route("/api/dataset/available-files")
def available_dataset_files():
    """List ``.pkl`` files in the embeddings directory."""
    files = []
    if EMBEDDINGS_DIR.exists():
        for pkl in sorted(EMBEDDINGS_DIR.glob("*.pkl")):
            files.append(
                {
                    "name": pkl.stem,
                    "path": str(pkl),
                    "size_mb": round(pkl.stat().st_size / (1024 * 1024), 1),
                }
            )
    return jsonify({"files": files})


@datasets_staging_bp.route("/api/dataset/combine", methods=["POST"])
def combine_datasets_route():
    """Combine multiple pickle datasets in a background thread."""
    body = request.get_json(force=True) or {}
    dataset_paths = body.get("datasets", [])
    name = str(body.get("name", "") or "").strip()

    if not isinstance(dataset_paths, list) or len(dataset_paths) < 2:
        return jsonify({"error": "Provide at least two dataset file paths."}), 400

    _base = _paths.get_file_access_base_dir()
    for p in dataset_paths:
        try:
            _paths.validate_server_filepath(str(p), base_dir=_base)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not Path(p).exists():
            return jsonify({"error": f"File not found: {p}"}), 400

    importer = get_importer("combine_datasets")
    if importer is None:
        return jsonify({"error": "combine_datasets importer not available"}), 500

    task_id = _run_importer_in_background(importer, {"datasets": dataset_paths, "name": name})
    return jsonify({"ok": True, "message": "Combining datasets...", "task_id": str(task_id) if task_id else ""})


@datasets_staging_bp.route("/api/dataset/stage-file", methods=["POST"])
def stage_file():
    """Upload a ``.pkl`` file and save it to the staging directory."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staging_path = STAGING_DIR / f"stage_{uuid4().hex}.pkl"
    file.save(staging_path)

    # Peek the pkl's dict structure cheaply — embeddings and inline media
    # bytes are skipped, so this stays light even on multi-GB uploads.
    try:
        with open(staging_path, "rb") as f:
            peeked = peek_pickle_dataset_summary(f)
        if isinstance(peeked, dict) and "medias" in peeked:
            media_dict = peeked["medias"]
        elif isinstance(peeked, dict):
            media_dict = peeked
        else:
            media_dict = {}
        count = len(media_dict) if isinstance(media_dict, dict) else 0
        media_type = "unknown"
        if isinstance(media_dict, dict) and media_dict:
            first = next(iter(media_dict.values()))
            if isinstance(first, dict):
                media_type = first.get("type", "audio") or "unknown"
        del peeked, media_dict
    except Exception:
        count = 0
        media_type = "unknown"

    name = file.filename or "Uploaded dataset"
    return jsonify({"path": str(staging_path), "name": name, "count": count, "media_type": media_type})


@datasets_staging_bp.route("/api/dataset/stage-import/<importer_name>", methods=["POST"])
def stage_import(importer_name: str):
    """Run a registered importer in staging mode."""
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err

    field_values, field_err = _extract_importer_fields(importer)
    if field_err:
        return field_err

    # Pass through optional keys not declared as plugin fields.
    file_keys = {f.key for f in importer.fields if f.field_type == "file"}
    for key in ("converters", "source_specs"):
        val = get_request_field(key, bool(file_keys))
        if val:
            field_values[key] = val

    _stage_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Staging started"})


@datasets_staging_bp.route("/api/dataset/stage-demo/<name>", methods=["POST"])
def stage_demo(name: str):
    """Stage a demo dataset as a temporary ``.pkl`` file."""
    if name not in DEMO_DATASETS:
        return jsonify({"error": "Invalid dataset name"}), 400

    importer = get_importer("demo")
    if importer is None:
        return jsonify({"error": "demo importer not available"}), 500

    body = request.get_json(force=True, silent=True) or {}
    converter_name = body.get("converter", "")
    dataset_name = str(body.get("dataset_name") or "").strip()

    field_values: dict = {"name": name}
    if converter_name:
        field_values["converter"] = converter_name
    if dataset_name:
        field_values["dataset_name"] = dataset_name

    label = dataset_name or DEMO_DATASETS[name].get("label", name)
    _stage_importer_in_background(importer, field_values, label=label)
    return jsonify({"ok": True, "message": "Staging demo dataset..."})


@datasets_staging_bp.route("/api/dataset/staging", methods=["DELETE"])
def clear_staging():
    """Remove all files from the staging directory."""
    if STAGING_DIR.exists():
        for f in STAGING_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
    return jsonify({"ok": True})


@datasets_staging_bp.route("/api/dataset/import/<importer_name>/options", methods=["POST"])
def importer_field_options(importer_name: str):
    """Return dropdown options for a dynamic-options field.

    Body::

        {"field_key": "query_id", "values": {"media_type": "audio", ...}}

    The importer's ``get_field_options(field_key, current_values)`` is
    called with the supplied snapshot of current form values.  Returns
    ``{"options": [...]}`` on success.  Errors from the plugin (network
    failure, auth error, etc.) are surfaced as a 502 with the original
    message so the frontend can display them inline.
    """
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err

    body = request.get_json(force=True, silent=True) or {}
    field_key = str(body.get("field_key") or "").strip()
    values = body.get("values") or {}
    if not field_key:
        return jsonify({"error": "Missing required field: 'field_key'"}), 400
    if not isinstance(values, dict):
        return jsonify({"error": "'values' must be an object"}), 400

    field = next((f for f in importer.fields if f.key == field_key), None)
    if field is None:
        return jsonify({"error": f"Unknown field: {field_key!r}"}), 400
    if not getattr(field, "dynamic_options", False):
        return jsonify({"error": f"Field {field_key!r} is not dynamic"}), 400

    try:
        options = importer.get_field_options(field_key, values)
    except NotImplementedError as exc:
        return jsonify({"error": str(exc) or "Importer does not implement get_field_options"}), 501
    except Exception as exc:  # noqa: BLE001 — surface remote-service errors verbatim
        return jsonify({"error": str(exc) or type(exc).__name__}), 502

    if not isinstance(options, list):
        return jsonify({"error": "get_field_options must return a list"}), 500
    return jsonify({"options": [str(o) for o in options]})


@datasets_staging_bp.route("/api/dataset/import/<importer_name>", methods=["POST"])
def import_dataset(importer_name: str):
    """Run a registered importer by name in a background thread."""
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err

    field_values, field_err = _extract_importer_fields(importer)
    if field_err:
        return field_err

    # Pass through optional keys not declared as plugin fields.
    file_keys = {f.key for f in importer.fields if f.field_type == "file"}
    for key in ("converters", "source_specs", "clipper", "embedder"):
        val = get_request_field(key, bool(file_keys))
        if val:
            field_values[key] = val

    clipper_params, params_err = _extract_clipper_params(bool(file_keys))
    if params_err:
        return params_err
    if clipper_params is not None:
        field_values["clipper_params"] = clipper_params

    task_id = _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""})
