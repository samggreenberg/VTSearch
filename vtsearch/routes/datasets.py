"""Blueprint for dataset management routes.

This module is a re-export facade.  The helpers and routes are split across:

* ``datasets_loading`` — Background loading, staging, origin management
* ``datasets_ui`` — Dashboard, demo list, display name

Route handlers that are tightly coupled to import/load/registry logic
remain here.
"""

import gc
import io
import threading
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request, send_file

from vtsearch.config import EMBEDDINGS_DIR
from vtsearch.routes.helpers import get_json_or_400
from vtsearch.datasets import DEMO_DATASETS, export_dataset_to_file, get_importer, list_importers
from vtsearch.datasets.loader import load_dataset_from_pickle, safe_pickle_load
from vtsearch.datasets.registry import (
    get_loaded_id as _reg_loaded_id,
    list_datasets as _reg_list_datasets,
    rename_dataset as _reg_rename,
    set_loaded_id as _reg_set_loaded,
    unregister_dataset as _reg_unregister,
    update_dataset as _reg_update,
    get_dataset as _reg_get,
)
from vtsearch.utils import (
    bad_votes,
    build_diversity_tree,
    collapse_duplicates,
    medias,
    get_dupe_count,
    get_progress,
    good_votes,
    set_dataset_display_name,
    snapshot_medias,
    update_progress,
)
import vtsearch.utils.paths as _paths

# Re-export loading helpers so existing importers keep working.
from vtsearch.routes.datasets_loading import (  # noqa: F401
    STAGING_DIR,
    _apply_clipper,
    _auto_register_dataset,
    _load_embedder_for_clips,
    _origin_to_str,
    _run_importer_in_background,
    _run_origin_load_in_background,
    _set_clip_origins,
    _stage_importer_in_background,
    clear_dataset,
)
from vtsearch.routes.datasets_ui import datasets_ui_bp  # noqa: F401

datasets_bp = Blueprint("datasets", __name__)

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
    from vtsearch.media import all_embedders_dict, embedders_for_type, get_by_folder_name

    media_type = request.args.get("media_type", "").strip()
    if media_type:
        # Accept both type_id ("image") and folder_import_name ("images").
        try:
            mt = get_by_folder_name(media_type)
            media_type = mt.type_id
        except KeyError:
            pass  # assume it is already a type_id
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
    from vtsearch.media import all_clippers_dict, clippers_for_type, get_by_folder_name

    media_type = request.args.get("media_type", "").strip()
    if media_type:
        # Accept both type_id ("image") and folder_import_name ("images").
        try:
            mt = get_by_folder_name(media_type)
            media_type = mt.type_id
        except KeyError:
            pass  # assume it is already a type_id
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
    from vtsearch.media import get_by_folder_name

    target = request.args.get("target", "").strip()
    source = request.args.get("source", "").strip()

    if target:
        try:
            mt = get_by_folder_name(target)
            target = mt.type_id
        except KeyError:
            pass
        converters = list_converters_for_target(target)
    elif source:
        try:
            mt = get_by_folder_name(source)
            source = mt.type_id
        except KeyError:
            pass
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
            "num_dupes": get_dupe_count(),
        }
    )


@datasets_bp.route("/api/dataset/progress")
def dataset_progress():
    """Return the current progress of long-running operations."""
    return jsonify(get_progress())


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

    for p in dataset_paths:
        try:
            _paths.validate_server_filepath(str(p))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        if not Path(p).exists():
            return jsonify({"error": f"File not found: {p}"}), 400

    importer = get_importer("combine_datasets")
    if importer is None:
        return jsonify({"error": "combine_datasets importer not available"}), 500

    _run_importer_in_background(importer, {"datasets": dataset_paths})
    return jsonify({"ok": True, "message": "Combining datasets..."})


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


@datasets_bp.route("/api/dataset/stage-import/<importer_name>", methods=["POST"])
def stage_import(importer_name: str):
    """Run a registered importer in staging mode."""
    importer = get_importer(importer_name)
    if importer is None:
        return jsonify({"error": f"Unknown importer: {importer_name!r}"}), 404

    file_keys = {f.key for f in importer.fields if f.field_type == "file"}

    field_values: dict = {}
    if file_keys:
        for key in file_keys:
            if key not in request.files:
                return jsonify({"error": f"Missing file field: {key!r}"}), 400
            # Read file contents before passing to background thread.
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
                return jsonify({"error": f"Missing required field: {f.key!r}"}), 400
            field_values[f.key] = body.get(f.key, f.default)

    # Pass through the optional "converters" key.
    if file_keys:
        conv = request.form.get("converters", "")
    else:
        conv = (request.get_json(force=True) or {}).get("converters", "")
    if conv:
        field_values["converters"] = conv

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
    importer = get_importer(importer_name)
    if importer is None:
        return jsonify({"error": f"Unknown importer: {importer_name!r}"}), 404

    file_keys = {f.key for f in importer.fields if f.field_type == "file"}

    # Build field_values from either multipart or JSON body.
    field_values: dict = {}
    if file_keys:
        for key in file_keys:
            if key not in request.files:
                return jsonify({"error": f"Missing file field: {key!r}"}), 400
            # Read file contents before passing to background thread, since the
            # Flask FileStorage stream is only valid during the request lifecycle.
            file_bytes = io.BytesIO(request.files[key].read())
            file_bytes.name = request.files[key].filename
            field_values[key] = file_bytes
        # Non-file fields come from form data when using multipart.
        for f in importer.fields:
            if f.field_type != "file":
                field_values[f.key] = request.form.get(f.key, f.default)
    else:
        body = request.get_json(force=True) or {}
        for f in importer.fields:
            if f.key not in body and f.required:
                return jsonify({"error": f"Missing required field: {f.key!r}"}), 400
            field_values[f.key] = body.get(f.key, f.default)

    # Pass through the optional "converters" key for importers that support
    # converter selection (e.g. folder, http_archive).
    if file_keys:
        converters_val = request.form.get("converters", "")
    else:
        converters_val = (request.get_json(force=True) or {}).get("converters", "")
    if converters_val:
        field_values["converters"] = converters_val

    # Pass through the optional "clipper" key for post-import clipping.
    if file_keys:
        clipper_val = request.form.get("clipper", "")
    else:
        clipper_val = (request.get_json(force=True) or {}).get("clipper", "")
    if clipper_val:
        field_values["clipper"] = clipper_val

    # Pass through the optional "embedder" key for embedder selection.
    if file_keys:
        embedder_val = request.form.get("embedder", "")
    else:
        embedder_val = (request.get_json(force=True) or {}).get("embedder", "")
    if embedder_val:
        field_values["embedder"] = embedder_val

    _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started"})


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

    field_values: dict = {"name": dataset_name}
    if converter_name:
        field_values["converter"] = converter_name
    if clipper_name:
        field_values["clipper"] = clipper_name
    if embedder_name:
        field_values["embedder"] = embedder_name

    _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started"})


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
    _run_importer_in_background(importer, {"file": file_bytes})
    return jsonify({"ok": True, "message": "Loading started"})


@datasets_bp.route("/api/dataset/load-folder", methods=["POST"])
def load_dataset_folder():
    """Generate dataset from a folder of media files."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    folder_path = data.get("path")
    media_type = data.get("media_type", "sounds")  # Default to sounds for backward compatibility

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

    try:
        _paths.validate_server_filepath(str(folder_path))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return jsonify({"error": "Invalid folder path"}), 400

    importer = get_importer("folder")
    _run_importer_in_background(importer, {"path": str(folder), "media_type": media_type})
    return jsonify({"ok": True, "message": "Loading started"})


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
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@datasets_bp.route("/api/dataset/clear", methods=["POST"])
def clear_dataset_route():
    """Clear the current dataset."""
    clear_dataset()
    _reg_set_loaded(None)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Dataset registry endpoints
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/datasets/registry")
def list_registered_datasets():
    """Return all registered datasets with their loaded state."""
    entries = _reg_list_datasets()
    loaded_id = _reg_loaded_id()
    for entry in entries:
        entry["loaded"] = entry["id"] == loaded_id
        if entry["loaded"]:
            entry["num_dupes"] = get_dupe_count()
        else:
            entry.setdefault("num_dupes", 0)
        entry.setdefault("embedder", "")
    return jsonify({"datasets": entries})


@datasets_bp.route("/api/datasets/registry/<dataset_id>/load", methods=["POST"])
def load_registered_dataset(dataset_id: str):
    """Load a registered dataset from its saved pkl file."""
    entry = _reg_get(dataset_id)
    if entry is None:
        return jsonify({"error": "Dataset not found in registry"}), 404

    pkl_path = entry.get("pkl_path", "")
    if not pkl_path or not Path(pkl_path).is_file():
        return jsonify({"error": f"Saved dataset file not found: {pkl_path}"}), 404

    _LOAD_STEPS = 4  # read pickle, process items, build diversity index, warm up embedder

    # Set progress to "loading" synchronously so the frontend never sees a
    # stale "idle" status from a previous operation before the thread starts.
    update_progress("loading", "Loading dataset from file...", step=1, total_steps=_LOAD_STEPS)

    def _pickle_progress(status, message, current, total):
        update_progress(status, message, current, total, step=2, total_steps=_LOAD_STEPS)

    def load_task():
        try:
            update_progress(
                "loading", "Clearing previous dataset…", 0, 0,
                step=1, total_steps=_LOAD_STEPS,
            )
            clear_dataset()
            _reg_set_loaded(None)
            gc.collect()
            load_dataset_from_pickle(Path(pkl_path), medias, on_progress=_pickle_progress)
            update_progress(
                "loading", "Removing duplicates…", 0, 0,
                step=3, total_steps=_LOAD_STEPS,
            )
            collapse_duplicates(medias)
            update_progress(
                "loading", "Building diversity index…", 0, 0,
                step=3, total_steps=_LOAD_STEPS,
            )
            build_diversity_tree()
            _reg_set_loaded(dataset_id)
            # Update item count and dupe count in case they changed
            _reg_update(dataset_id, num_items=len(medias), num_dupes=get_dupe_count())
            set_dataset_display_name(entry.get("name", ""))
            _load_embedder_for_clips()
        except MemoryError:
            medias.clear()
            gc.collect()
            update_progress("idle", "", 0, 0, "Out of memory — dataset too large.", step=None, total_steps=None)
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e), step=None, total_steps=None)

    # Signal "loading" before the thread starts so frontend polling never
    # sees a stale "idle" from a previous load and prematurely stops.
    update_progress("loading", "Preparing to load dataset…", 0, 0, step=1, total_steps=_LOAD_STEPS)

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Loading started"})


@datasets_bp.route("/api/datasets/registry/<dataset_id>/unload", methods=["POST"])
def unload_registered_dataset(dataset_id: str):
    """Unload the currently loaded dataset (clear from memory)."""
    loaded = _reg_loaded_id()
    if loaded != dataset_id:
        return jsonify({"error": "This dataset is not currently loaded"}), 400
    clear_dataset()
    _reg_set_loaded(None)
    return jsonify({"ok": True})


@datasets_bp.route("/api/datasets/registry/<dataset_id>", methods=["DELETE"])
def delete_registered_dataset(dataset_id: str):
    """Remove a dataset from the registry and delete its pkl file."""
    loaded = _reg_loaded_id()
    if loaded == dataset_id:
        clear_dataset()
        _reg_set_loaded(None)
    ok = _reg_unregister(dataset_id)
    if not ok:
        return jsonify({"error": "Dataset not found"}), 404
    return jsonify({"ok": True})


@datasets_bp.route("/api/datasets/registry/<dataset_id>/rename", methods=["PUT"])
def rename_registered_dataset(dataset_id: str):
    """Rename a registered dataset."""
    data = request.get_json(force=True, silent=True) or {}
    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "name is required"}), 400
    ok = _reg_rename(dataset_id, new_name)
    if not ok:
        return jsonify({"error": "Dataset not found"}), 404
    # Also update display name if this is the loaded dataset
    if _reg_loaded_id() == dataset_id:
        set_dataset_display_name(new_name)
    return jsonify({"ok": True, "name": new_name})


@datasets_bp.route("/api/dataset/load-source", methods=["POST"])
def load_dataset_from_source():
    """Reload a dataset from a stored source origin dict."""
    data = request.get_json(force=True, silent=True) or {}
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
    for key, val in field_values.items():
        if isinstance(val, str) and ("/" in val or "\\" in val):
            try:
                _paths.validate_server_filepath(val)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

    _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started"})
