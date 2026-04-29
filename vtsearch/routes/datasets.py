"""Blueprint for dataset management routes.

This module covers media-type/clipper/converter listings, dataset import and
staging, demo dataset loading, and origin reload.  Closely related blueprints:

* :mod:`vtsearch.routes.datasets_ui` — Dashboard, demo list, display name
* :mod:`vtsearch.routes.datasets_registry` — Registry CRUD (load/unload/rename/etc.)
* :mod:`vtsearch.datasets.load_pipeline` — Background loading, staging, origin management
"""

import io
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from uuid import uuid4

from flask import Blueprint, jsonify, request, send_file

from vtsearch.config import DATA_DIR, EMBEDDINGS_DIR
from vtsearch.routes.helpers import get_json_or_400, get_json_safe, get_plugin_or_404, get_request_field
from vtsearch.datasets import DEMO_DATASETS, export_dataset_to_file, get_importer, list_importers
from vtsearch.datasets.loader import safe_pickle_load
from vtsearch.datasets.registry import (
    is_loaded as _reg_is_loaded,
    list_datasets as _reg_list_all,
    remove_loaded_id as _reg_remove_loaded,
    add_loaded_id as _reg_add_loaded,
    unregister_dataset as _reg_unregister,
    update_dataset as _reg_update,
    get_dataset as _reg_get,
)
from vtsearch.utils import (
    bad_votes,
    cancel_dataset_progress,
    get_dataset_display_name,
    get_dupe_count,
    get_progress,
    good_votes,
    snapshot_medias,
    unregister_context,
)
from vtsearch.utils.progress import loading_tasks as _loading_tasks
import vtsearch.utils.paths as _paths

# Re-export loading helpers so existing importers keep working.
from vtsearch.datasets.load_pipeline import (  # noqa: F401
    STAGING_DIR,
    _apply_clipper,
    _auto_register_dataset,
    _load_embedder_for_clips,
    _load_embedder_with_progress as _load_embedder_for_clips_with_progress,
    _origin_to_str,
    _run_importer_in_background,
    _run_origin_load_in_background,
    _stage_importer_in_background,
    clear_dataset,
)
from vtsearch.routes.datasets_ui import datasets_ui_bp  # noqa: F401

datasets_bp = Blueprint("datasets", __name__)


def _normalize_media_type_param(value: str) -> str:
    """Accept both ``type_id`` (``"image"``) and ``folder_import_name`` (``"images"``)."""
    value = value.strip()
    if not value:
        return ""
    from vtsearch.media import get_by_folder_name, normalize_type_id

    try:
        return get_by_folder_name(value).type_id
    except KeyError:
        return normalize_type_id(value)


# Register the UI sub-blueprint.
datasets_bp.register_blueprint(datasets_ui_bp)


# ---------------------------------------------------------------------------
# Media types
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/media-types")
def media_types_list():
    """Return all registered media types with their metadata."""
    from vtsearch.media import all_types_dict

    return jsonify({"media_types": all_types_dict()})


@datasets_bp.route("/api/embedders")
def embedders_list():
    """Return all registered embedders, optionally filtered by media type.

    Query parameters:
        media_type: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only embedders whose
            ``media_type_id`` matches are returned.
    """
    from vtsearch.media import all_embedders_dict, embedders_for_type

    media_type = _normalize_media_type_param(request.args.get("media_type", ""))
    if media_type:
        embedders = [e.to_dict() for e in embedders_for_type(media_type)]
    else:
        embedders = all_embedders_dict()

    return jsonify({"embedders": embedders})


# ---------------------------------------------------------------------------
# Clipper chooser
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/clippers")
def clippers_list():
    """Return all clippers, optionally filtered by media type.

    Query parameters:
        media_type: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only clippers whose
            ``media_type`` matches are returned.
    """
    from vtsearch.media import all_clippers_dict, clippers_for_type

    media_type = _normalize_media_type_param(request.args.get("media_type", ""))
    if media_type:
        clippers = [c.to_dict() for c in clippers_for_type(media_type)]
    else:
        clippers = all_clippers_dict()

    return jsonify({"clippers": clippers})


# ---------------------------------------------------------------------------
# Converter chooser
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/converters")
def converters_list():
    """Return all converters, optionally filtered by source or target media type.

    Query parameters:
        target: A ``type_id`` (e.g. ``"image"``) or ``folder_import_name``
            (e.g. ``"images"``).  When provided, only converters whose
            ``target_type`` matches are returned.
        source: A ``type_id`` (e.g. ``"video"``) or ``folder_import_name``
            (e.g. ``"videos"``).  When provided, only converters whose
            ``source_type`` matches are returned.
    """
    from vtsearch.converters import list_converters, list_converters_for_source, list_converters_for_target

    target = _normalize_media_type_param(request.args.get("target", ""))
    source = _normalize_media_type_param(request.args.get("source", ""))

    if target:
        converters = list_converters_for_target(target)
    elif source:
        converters = list_converters_for_source(source)
    else:
        converters = list_converters()

    return jsonify({"converters": [c.to_dict() for c in converters]})


# ---------------------------------------------------------------------------
# Status / progress
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/status")
def dataset_status():
    """Return the current dataset status."""
    snap = snapshot_medias()
    media_type = None
    if snap:
        media_type = next(iter(snap.values())).get("type", "audio")
    return jsonify(
        {
            "loaded": len(snap) > 0,
            "num_medias": len(snap),
            "has_votes": len(good_votes) + len(bad_votes) > 0,
            "media_type": media_type,
            "display_name": get_dataset_display_name(),
            "num_dupes": get_dupe_count(),
        }
    )


@datasets_bp.route("/api/dataset/progress")
def dataset_progress():
    """Return the current progress of long-running operations.

    For backward compatibility this returns a single progress dict.
    Prefers the first active loading task if any, otherwise falls back
    to the legacy global tracker (used by staging operations).
    """
    tasks = _loading_tasks.list_tasks()
    active = [t for t in tasks if t.get("status") != "idle"]
    if active:
        return jsonify(active[0])
    # Check if any just-finished task has an error to report
    errored = [t for t in tasks if t.get("error")]
    if errored:
        return jsonify(errored[0])
    return jsonify(get_progress())


@datasets_bp.route("/api/dataset/loading-tasks")
def dataset_loading_tasks():
    """Return all active dataset loading tasks with their progress."""
    return jsonify({"tasks": _loading_tasks.list_tasks()})


@datasets_bp.route("/api/dataset/cancel", methods=["POST"])
def cancel_dataset_load():
    """Cancel dataset load/import operations.

    Cancels all active loading tasks and the legacy global tracker.
    """
    _loading_tasks.cancel_all()
    cancel_dataset_progress()
    return jsonify({"ok": True})


@datasets_bp.route("/api/dataset/cancel/<task_id>", methods=["POST"])
def cancel_dataset_load_task(task_id: str):
    """Cancel a specific dataset loading task."""
    ok = _loading_tasks.cancel_task(task_id)
    if not ok:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Importer discovery
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/importers")
def dataset_importers():
    """List all registered importers (excluding those with non-form UI)."""
    extended = [imp.to_dict() for imp in list_importers() if imp.ui_mode == "form"]
    return jsonify({"importers": extended})


@datasets_bp.route("/api/dataset/all-importers")
def dataset_all_importers():
    """List all registered importers (including built-in ones)."""
    all_importers = [imp.to_dict() for imp in list_importers()]

    # Annotate combine_datasets with an enabled flag: requires 2+ saved
    # datasets sharing the same media type.
    from collections import Counter

    type_counts = Counter(e.get("media_type") for e in _reg_list_all())
    can_combine = any(c >= 2 for c in type_counts.values())
    for imp_dict in all_importers:
        if imp_dict["name"] == "combine_datasets":
            imp_dict["enabled"] = can_combine
            break

    return jsonify({"importers": all_importers})


# ---------------------------------------------------------------------------
# Available dataset files (for the Combine Existing Datasets UI)
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/available-files")
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


@datasets_bp.route("/api/dataset/combine", methods=["POST"])
def combine_datasets_route():
    """Combine multiple pickle datasets in a background thread."""
    body = request.get_json(force=True) or {}
    dataset_paths = body.get("datasets", [])

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

    task_id = _run_importer_in_background(importer, {"datasets": dataset_paths})
    return jsonify({"ok": True, "message": "Combining datasets...", "task_id": str(task_id) if task_id else ""})


# ---------------------------------------------------------------------------
# Staging endpoints (for the combine-datasets UI)
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/stage-file", methods=["POST"])
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

    # Peek inside the pkl to get count and media type.
    try:
        with open(staging_path, "rb") as f:
            data = safe_pickle_load(f)
        if isinstance(data, dict) and "medias" in data:
            media_dict = data["medias"]
        elif isinstance(data, dict):
            media_dict = data
        else:
            media_dict = {}
        count = len(media_dict)
        if media_dict:
            first = next(iter(media_dict.values()))
            media_type = first.get("type", "audio")
        else:
            media_type = "unknown"
        del data, media_dict
    except Exception:
        count = 0
        media_type = "unknown"

    name = file.filename or "Uploaded dataset"
    return jsonify({"path": str(staging_path), "name": name, "count": count, "media_type": media_type})


def _extract_importer_fields(importer):
    """Build field_values for a dataset importer from the current request.

    Unlike :func:`extract_plugin_fields`, this reads file contents into
    :class:`io.BytesIO` so they remain valid after the Flask request context
    ends (required for background-thread execution).

    Returns ``(field_values, None)`` on success, or ``(None, error_tuple)``
    when a required field is missing.
    """
    file_keys = {f.key for f in importer.fields if f.field_type == "file"}

    field_values: dict = {}
    if file_keys:
        for key in file_keys:
            if key not in request.files:
                return None, (jsonify({"error": f"Missing file field: {key!r}"}), 400)
            file_bytes = io.BytesIO(request.files[key].read())
            file_bytes.name = request.files[key].filename
            field_values[key] = file_bytes
        for f in importer.fields:
            if f.field_type != "file":
                field_values[f.key] = request.form.get(f.key, f.default)
    else:
        body = request.get_json(force=True) or {}
        for f in importer.fields:
            if f.key not in body and f.required:
                return None, (jsonify({"error": f"Missing required field: {f.key!r}"}), 400)
            field_values[f.key] = body.get(f.key, f.default)

    return field_values, None


@datasets_bp.route("/api/dataset/stage-import/<importer_name>", methods=["POST"])
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
    for key in ("converters",):
        val = get_request_field(key, bool(file_keys))
        if val:
            field_values[key] = val

    _stage_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Staging started"})


@datasets_bp.route("/api/dataset/stage-demo/<name>", methods=["POST"])
def stage_demo(name: str):
    """Stage a demo dataset as a temporary ``.pkl`` file."""
    if name not in DEMO_DATASETS:
        return jsonify({"error": "Invalid dataset name"}), 400

    importer = get_importer("demo")
    if importer is None:
        return jsonify({"error": "demo importer not available"}), 500

    body = request.get_json(force=True, silent=True) or {}
    converter_name = body.get("converter", "")

    field_values: dict = {"name": name}
    if converter_name:
        field_values["converter"] = converter_name

    label = DEMO_DATASETS[name].get("label", name)
    _stage_importer_in_background(importer, field_values, label=label)
    return jsonify({"ok": True, "message": "Staging demo dataset..."})


@datasets_bp.route("/api/dataset/staging", methods=["DELETE"])
def clear_staging():
    """Remove all files from the staging directory."""
    if STAGING_DIR.exists():
        for f in STAGING_DIR.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Generic import endpoint
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/import/<importer_name>", methods=["POST"])
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
    for key in ("converters", "clipper", "embedder"):
        val = get_request_field(key, bool(file_keys))
        if val:
            field_values[key] = val

    task_id = _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""})


# ---------------------------------------------------------------------------
# Local-folder upload — files come from the user's browser machine
# ---------------------------------------------------------------------------

LOCAL_UPLOADS_DIR = DATA_DIR / "local_uploads"


def _safe_relative_upload_path(filename: str) -> PurePosixPath | None:
    """Return *filename* as a sanitised relative POSIX path, or ``None`` if unsafe.

    Browsers send each file's ``webkitRelativePath`` (or basename) as the
    multipart filename.  Reject anything absolute or that would escape the
    upload root via ``..`` segments.  Empty path components and "." are
    skipped.
    """
    if not filename:
        return None
    raw = filename.replace("\\", "/")
    if raw.startswith("/"):
        return None
    parts: list[str] = []
    for segment in raw.split("/"):
        if not segment or segment == ".":
            continue
        if segment == ".." or "\x00" in segment:
            return None
        parts.append(segment)
    if not parts:
        return None
    return PurePosixPath(*parts)


@datasets_bp.route("/api/dataset/import-local-folder", methods=["POST"])
def import_local_folder():
    """Import a folder uploaded from the user's *browser* machine.

    The browser uses ``<input type="file" webkitdirectory>`` to let the
    user pick a directory; each selected ``File`` is appended to the
    multipart body under the key ``"files"`` with its ``webkitRelativePath``
    as the multipart filename.  We stream each file to a temporary
    directory on the server (preserving sub-directory structure) and then
    delegate to the regular folder importer to do the actual scanning,
    embedding, and dataset registration.  The temp directory is removed
    once the importer finishes (success or failure).
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    importer = get_importer("server_folder")
    if importer is None:
        return jsonify({"error": "server_folder importer not available"}), 500

    media_type = (request.form.get("media_type") or "").strip()
    if not media_type:
        return jsonify({"error": "Missing required field: 'media_type'"}), 400

    embedder = (request.form.get("embedder") or "").strip()
    clipper = (request.form.get("clipper") or "").strip()
    converters = (request.form.get("converters") or "").strip()
    recursive_raw = (request.form.get("recursive") or "true").strip().lower()
    recursive = recursive_raw not in ("false", "0", "no", "off")
    clipper_params_raw = request.form.get("clipper_params") or ""
    clipper_params: dict | None = None
    if clipper_params_raw:
        try:
            clipper_params = json.loads(clipper_params_raw)
            if not isinstance(clipper_params, dict):
                raise ValueError("clipper_params must be a JSON object")
        except (ValueError, TypeError) as exc:
            return jsonify({"error": f"Invalid clipper_params: {exc}"}), 400

    LOCAL_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload_dir = Path(tempfile.mkdtemp(prefix="local_folder_", dir=LOCAL_UPLOADS_DIR))

    saved = 0
    try:
        for f in files:
            rel = _safe_relative_upload_path(f.filename or "")
            if rel is None:
                continue
            dest = upload_dir / Path(*rel.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            f.save(dest)
            saved += 1
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    if saved == 0:
        shutil.rmtree(upload_dir, ignore_errors=True)
        return jsonify({"error": "No valid files in upload"}), 400

    field_values: dict = {
        "path": str(upload_dir),
        "media_type": media_type,
        "recursive": recursive,
    }
    if embedder:
        field_values["embedder"] = embedder
    if clipper:
        field_values["clipper"] = clipper
        if clipper_params is not None:
            field_values["clipper_params"] = clipper_params
    if converters:
        field_values["converters"] = converters

    # Origin is intentionally synthetic — the on-disk path is a temp dir we
    # are about to delete, so storing it on each media would be misleading
    # and ``can_reload_from_origin`` would (correctly) refuse to reload.
    origin = {
        "importer": "server_folder",
        "params": {"path": "<browser_upload>", "media_type": media_type},
    }

    clipper_name = field_values.pop("clipper", "") or ""
    inner_clipper_params = field_values.pop("clipper_params", None)
    field_values["clipper"] = clipper_name
    if clipper_name and not clipper_name.endswith("_default"):
        field_values["skip_embedding"] = True

    def _load(target_medias):
        try:
            importer.run(field_values, target_medias)
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    from vtsearch.auth import get_current_user
    from vtsearch.datasets.load_pipeline import _normalize_media_type

    task_id = _run_origin_load_in_background(
        _load,
        origin,
        name="Local folder upload",
        clipper=clipper_name,
        clipper_params=inner_clipper_params,
        embedder=embedder,
        created_by=get_current_user(),
        media_type=_normalize_media_type(media_type),
    )
    return jsonify({"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""})


# ---------------------------------------------------------------------------
# Demo dataset load
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/load-demo", methods=["POST"])
def load_demo_dataset_route():
    """Load a demo dataset in a background thread.

    When a ``converter`` is specified, the demo data is loaded using its
    original media type, then converted to the converter's target type.
    The resulting dataset has the *target* type, not the demo's original
    type.
    """
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    dataset_name = data.get("name")
    embedder_name = data.get("embedder", "")
    clipper_name = data.get("clipper", "")
    converter_name = data.get("converter", "")

    if not dataset_name or dataset_name not in DEMO_DATASETS:
        return jsonify({"error": "Invalid dataset name"}), 400

    importer = get_importer("demo")
    if importer is None:
        return jsonify({"error": "demo importer not available"}), 500

    demo_info = DEMO_DATASETS[dataset_name]
    field_values: dict = {"name": dataset_name}
    # Inject media_type so the loading task exposes it to the frontend,
    # allowing the "guessed type" logic to consider in-progress loads.
    if converter_name:
        # When a converter is used, the resulting dataset has the converter's
        # target type, not the demo's original type.
        from vtsearch.converters import get_converter  # noqa: PLC0415

        conv = get_converter(converter_name)
        if conv is not None:
            field_values["media_type"] = conv.target_type
        else:
            field_values["media_type"] = demo_info.get("media_type", "")
        field_values["converter"] = converter_name
    else:
        field_values["media_type"] = demo_info.get("media_type", "")
    if clipper_name:
        field_values["clipper"] = clipper_name
    if embedder_name:
        field_values["embedder"] = embedder_name

    task_id = _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""})


# ---------------------------------------------------------------------------
# Legacy endpoints – kept for backward compatibility.
# These now delegate to the appropriate importer internally.
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/load-file", methods=["POST"])
def load_dataset_file():
    """Load a dataset from an uploaded pickle file."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400

    importer = get_importer("pickle")
    # Read file contents before passing to background thread, since the
    # Flask FileStorage stream is only valid during the request lifecycle.
    file_bytes = io.BytesIO(file.read())
    file_bytes.name = file.filename
    task_id = _run_importer_in_background(importer, {"file": file_bytes})
    return jsonify({"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""})


@datasets_bp.route("/api/dataset/load-folder", methods=["POST"])
def load_dataset_folder():
    """Generate dataset from a folder of media files."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    folder_path = data.get("path")
    media_type = data.get("media_type", "audio")  # Default to audio

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    try:
        _paths.validate_server_filepath(str(folder_path), base_dir=_paths.get_file_access_base_dir())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return jsonify({"error": "Invalid folder path"}), 400

    importer = get_importer("server_folder")
    task_id = _run_importer_in_background(importer, {"path": str(folder), "media_type": media_type})
    return jsonify({"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""})


# ---------------------------------------------------------------------------
# Export / clear
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/export")
def export_dataset():
    """Export the current dataset to a pickle file."""
    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No dataset loaded"}), 400

    try:
        dataset_bytes = export_dataset_to_file(snap)
        return send_file(
            io.BytesIO(dataset_bytes),
            mimetype="application/octet-stream",
            download_name="vtsearch_dataset.pkl",
            as_attachment=True,
        )
    except Exception:
        import logging

        logging.getLogger(__name__).exception("Dataset export failed")
        return jsonify({"error": "Dataset export failed"}), 500


@datasets_bp.route("/api/dataset/clear", methods=["POST"])
def clear_dataset_route():
    """Clear the request-scoped dataset from memory.

    Uses the ``X-Dataset-Id`` header (via ``get_active_context()``) to
    determine which dataset to clear.
    """
    from vtsearch.utils import get_active_context

    ctx = get_active_context()
    ds_id = ctx.dataset_id if ctx.dataset_id else None
    if ds_id:
        unregister_context(ds_id)
        _reg_remove_loaded(ds_id)
    else:
        clear_dataset()
    return jsonify({"ok": True})



@datasets_bp.route("/api/dataset/load-source", methods=["POST"])
def load_dataset_from_source():
    """Reload a dataset from a stored source origin dict."""
    data = get_json_safe()
    source = data.get("source")
    if not isinstance(source, dict):
        return jsonify({"error": "source must be an origin dict"}), 400
    return _load_from_origin(source)


def _load_from_origin(source: dict):
    """Start loading a dataset from a raw origin dict (internal helper).

    Special pseudo-origins (``"dupe_set"``) are handled inline.
    All real importers (including ``"demo"``) are dispatched generically via
    :meth:`~DatasetImporter.reload_from_origin`.
    """
    importer_name = source.get("importer", "")

    # --- pseudo-origins (not real importers) ---

    if importer_name == "dupe_set":
        members = source.get("members", [])
        if members:
            member_origin = members[0].get("origin")
            if isinstance(member_origin, dict):
                return _load_from_origin(member_origin)
        return jsonify({"error": "Cannot reload from dupe_set origin"}), 400

    # --- real importers: generic dispatch ---

    importer = get_importer(importer_name)
    if importer is None:
        return jsonify({"error": f"Unknown importer: {importer_name}"}), 400

    if not importer.can_reload_from_origin(source):
        return jsonify({"error": f"Cannot reload from {importer_name} origin (source not available)"}), 400

    field_values = importer.reload_from_origin(source)
    if field_values is None:
        return jsonify({"error": f"Cannot reload from {importer_name} origin"}), 400

    # Validate any server file paths in the field values
    _base = _paths.get_file_access_base_dir()
    for key, val in field_values.items():
        if isinstance(val, str) and ("/" in val or "\\" in val):
            try:
                _paths.validate_server_filepath(val, base_dir=_base)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

    task_id = _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""})
