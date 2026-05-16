"""Dataset staging, importer dispatch, and the combine-datasets endpoint.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.

JSON-shaped routes (available-files, combine, stage-demo, clear-staging,
importer-field-options) use the standard ``@arguments`` + ``@response``
decorators; schema-level validation failures surface as 422.
Multipart-upload routes (``stage-file``) and plugin-field routes
(``stage-import/<importer>``, ``import/<importer>``) keep ``@arguments``
omitted — the latter pair stays on the legacy plain-Flask path because
the request body is a plugin-field shape that doesn't fit a static
marshmallow schema (see *Resolved questions / Plugin field endpoints*
in the plan doc).
"""

from pathlib import Path
from uuid import uuid4

from flask import jsonify, request
from flask_smorest import Blueprint, abort

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
from vtsearch.schemas.datasets import (
    ClearStagingResponseSchema,
    DatasetAvailableFilesResponseSchema,
    DatasetCombineRequestSchema,
    DatasetLoadStartedResponseSchema,
    DatasetStageDemoRequestSchema,
    DatasetStageFileResponseSchema,
    DatasetStagingStartedResponseSchema,
    ImporterFieldOptionsRequestSchema,
    ImporterFieldOptionsResponseSchema,
)
from vtsearch.security.pickle import peek_pickle_dataset_summary

datasets_staging_bp = Blueprint(
    "datasets_staging",
    __name__,
    description="Stage, combine, and import datasets.",
)


@datasets_staging_bp.route("/api/dataset/available-files")
@datasets_staging_bp.response(200, DatasetAvailableFilesResponseSchema)
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
    return {"files": files}


@datasets_staging_bp.route("/api/dataset/combine", methods=["POST"])
@datasets_staging_bp.arguments(DatasetCombineRequestSchema)
@datasets_staging_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_staging_bp.alt_response(400, description="Invalid or missing dataset path.")
@datasets_staging_bp.alt_response(500, description="The combine_datasets importer is unavailable.")
def combine_datasets_route(body: dict):
    """Combine multiple pickle datasets in a background thread."""
    dataset_paths = body["datasets"]
    name = str(body.get("name", "") or "").strip()

    _base = _paths.get_file_access_base_dir()
    for p in dataset_paths:
        try:
            _paths.validate_server_filepath(str(p), base_dir=_base)
        except ValueError as exc:
            abort(400, message=str(exc))
        if not Path(p).exists():
            abort(400, message=f"File not found: {p}")

    importer = get_importer("combine_datasets")
    if importer is None:
        abort(500, message="combine_datasets importer not available")

    task_id = _run_importer_in_background(importer, {"datasets": dataset_paths, "name": name})
    return {"ok": True, "message": "Combining datasets...", "task_id": str(task_id) if task_id else ""}


@datasets_staging_bp.route("/api/dataset/stage-file", methods=["POST"])
@datasets_staging_bp.response(200, DatasetStageFileResponseSchema)
@datasets_staging_bp.alt_response(400, description="Multipart body has no file or empty filename.")
def stage_file():
    """Upload a ``.pkl`` file and save it to the staging directory."""
    if "file" not in request.files:
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

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
    return {"path": str(staging_path), "name": name, "count": count, "media_type": media_type}


# ---------------------------------------------------------------------------
# POST /api/dataset/stage-import/<importer_name>
#
# Plugin-field route — multipart-or-JSON depending on the importer's
# declared ``fields``. Stays on the legacy plain-Flask path; not in spec.
# See *Resolved questions / Plugin field endpoints* in the plan doc.
# ---------------------------------------------------------------------------


@datasets_staging_bp.route("/api/dataset/stage-import/<importer_name>", methods=["POST"])
def stage_import(importer_name: str):
    """Run a registered importer in staging mode.

    Plugin-dependent body shape: not described in the OpenAPI spec.
    """
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values, field_err = _extract_importer_fields(importer)
    if field_err:
        return field_err
    assert field_values is not None  # narrowed by field_err check

    # Pass through optional keys not declared as plugin fields.
    file_keys = {f.key for f in importer.fields if f.field_type == "file"}
    for key in ("converters", "source_specs"):
        val = get_request_field(key, bool(file_keys))
        if val:
            field_values[key] = val

    _stage_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Staging started"})


@datasets_staging_bp.route("/api/dataset/stage-demo/<name>", methods=["POST"])
@datasets_staging_bp.arguments(DatasetStageDemoRequestSchema)
@datasets_staging_bp.response(200, DatasetStagingStartedResponseSchema)
@datasets_staging_bp.alt_response(400, description="Unknown demo dataset name.")
@datasets_staging_bp.alt_response(500, description="The demo importer is unavailable.")
def stage_demo(body: dict, name: str):
    """Stage a demo dataset as a temporary ``.pkl`` file."""
    if name not in DEMO_DATASETS:
        abort(400, message="Invalid dataset name")

    importer = get_importer("demo")
    if importer is None:
        abort(500, message="demo importer not available")

    converter_name = body.get("converter", "")
    dataset_name = str(body.get("dataset_name") or "").strip()

    field_values: dict = {"name": name}
    if converter_name:
        field_values["converter"] = converter_name
    if dataset_name:
        field_values["dataset_name"] = dataset_name

    label = dataset_name or DEMO_DATASETS[name].get("label", name)
    _stage_importer_in_background(importer, field_values, label=label)
    return {"ok": True, "message": "Staging demo dataset..."}


@datasets_staging_bp.route("/api/dataset/staging", methods=["DELETE"])
@datasets_staging_bp.response(200, ClearStagingResponseSchema)
def clear_staging():
    """Remove all files from the staging directory."""
    if STAGING_DIR.exists():
        for f in STAGING_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
    return {"ok": True}


@datasets_staging_bp.route("/api/dataset/import/<importer_name>/options", methods=["POST"])
@datasets_staging_bp.arguments(ImporterFieldOptionsRequestSchema)
@datasets_staging_bp.response(200, ImporterFieldOptionsResponseSchema)
@datasets_staging_bp.alt_response(400, description="Unknown or non-dynamic field key.")
@datasets_staging_bp.alt_response(404, description="Unknown importer name.")
@datasets_staging_bp.alt_response(500, description="get_field_options did not return a list.")
@datasets_staging_bp.alt_response(501, description="Importer does not implement get_field_options.")
@datasets_staging_bp.alt_response(502, description="Remote service backing dynamic options raised an error.")
def importer_field_options(body: dict, importer_name: str):
    """Return dropdown options for a dynamic-options field.

    The importer's ``get_field_options(field_key, current_values)`` is
    called with the supplied snapshot of current form values. Errors
    from the plugin (network failure, auth error, etc.) are surfaced as
    a 502 with the original message so the frontend can display them
    inline.
    """
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_key = body["field_key"].strip()
    values = body.get("values") or {}

    field = next((f for f in importer.fields if f.key == field_key), None)
    if field is None:
        abort(400, message=f"Unknown field: {field_key!r}")
    if not getattr(field, "dynamic_options", False):
        abort(400, message=f"Field {field_key!r} is not dynamic")

    try:
        options = importer.get_field_options(field_key, values)
    except NotImplementedError as exc:
        abort(501, message=str(exc) or "Importer does not implement get_field_options")
    except Exception as exc:  # noqa: BLE001 — surface remote-service errors verbatim
        abort(502, message=str(exc) or type(exc).__name__)

    if not isinstance(options, list):
        abort(500, message="get_field_options must return a list")
    return {"options": [str(o) for o in options]}


# ---------------------------------------------------------------------------
# POST /api/dataset/import/<importer_name>
#
# Plugin-field route — same legacy treatment as ``stage-import``.
# ---------------------------------------------------------------------------


@datasets_staging_bp.route("/api/dataset/import/<importer_name>", methods=["POST"])
def import_dataset(importer_name: str):
    """Run a registered importer by name in a background thread.

    Plugin-dependent body shape: not described in the OpenAPI spec.
    """
    importer, err = get_plugin_or_404(get_importer, list_importers, importer_name, "importer")
    if err:
        return err
    assert importer is not None  # narrowed by err check

    field_values, field_err = _extract_importer_fields(importer)
    if field_err:
        return field_err
    assert field_values is not None  # narrowed by field_err check

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
