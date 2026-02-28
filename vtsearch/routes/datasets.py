"""Blueprint for dataset management routes."""

import gc
import io
import pickle
import threading
from pathlib import Path
from uuid import uuid4

from flask import Blueprint, jsonify, request, send_file

from vtsearch.config import DATA_DIR, EMBEDDINGS_DIR
from vtsearch.datasets import DEMO_DATASETS, export_dataset_to_file, get_importer, list_importers, load_demo_dataset
from vtsearch.utils import (
    bad_votes,
    build_diversity_tree,
    clear_all,
    collapse_duplicates,
    get_dataset_display_name,
    medias,
    get_dupe_count,
    get_progress,
    good_votes,
    set_dataset_display_name,
    update_progress,
)

datasets_bp = Blueprint("datasets", __name__)

# Names of importers that have dedicated, hand-crafted UI sections in the
# frontend.  They are excluded from the generic /api/dataset/importers list
# so the frontend doesn't render a duplicate panel for them.
_BUILTIN_IMPORTER_NAMES = {"pickle", "combine_datasets"}


@datasets_bp.route("/api/media-types")
def media_types_list():
    """Return all registered media types with their metadata.

    The frontend uses this endpoint to render media type UI dynamically
    (icons, tab titles, clip rendering modes, etc.) instead of hardcoding
    the available media types.

    Returns a JSON object::

        {
          "media_types": [
            {
              "type_id": "audio",
              "name": "Audio",
              "icon": "🔊",
              "tab_title": "Sounds",
              "folder_import_name": "sounds",
              "loops": true,
              "file_extensions": ["*.wav", "*.mp3", ...]
            },
            ...
          ]
        }
    """
    from vtsearch.media import all_types_dict

    return jsonify({"media_types": all_types_dict()})


def _load_embedder_for_clips() -> None:
    """Eagerly load the embedder for the current dataset's media type.

    Called right after a dataset finishes loading so the first text sort
    doesn't have to wait for the model download.  ``load_models()`` is
    idempotent, so this is a no-op when the model is already warm (e.g.
    after a folder import that already called ``embed_media()``).

    After loading the model, a dummy text embedding is run to warm up the
    text encoder branch.  Models like CLAP, CLIP, and X-CLIP have separate
    media and text encoder sub-networks; data ingest only exercises the
    media branch, leaving the text branch cold.  Without this warmup the
    first user-initiated text sort would stall on PyTorch's lazy
    initialisation for that branch.
    """
    if not medias:
        return
    media_type = next(iter(medias.values())).get("type", "audio")
    from vtsearch.media import get as media_get

    try:
        mt = media_get(media_type)
    except KeyError:
        return
    mt.load_models()
    # Warm up the text encoder so the first text sort is instant.
    update_progress("loading", "Warming up text encoder…", 0, 0)
    try:
        mt.embed_text("warmup")
    except Exception:
        pass
    update_progress("idle", "Ready")


def clear_dataset():
    """Clear the current dataset, votes, and all related state."""
    clear_all()


def _set_clip_origins(clips_dict: dict, origin: dict) -> None:
    """Set origin and origin_name on medias that don't already have them.

    Called after an importer finishes populating the medias dict.  Clips
    that already carry their own origin (e.g. loaded from a pickle that
    recorded per-element provenance) are left untouched.
    """
    for media in clips_dict.values():
        if media.get("origin") is None:
            media["origin"] = origin
        if not media.get("origin_name"):
            media["origin_name"] = media.get("filename", "")


def _run_importer_in_background(importer, field_values: dict) -> None:
    """Start *importer*.run() in a daemon thread after clearing the dataset."""

    def load_task():
        try:
            clear_dataset()
            gc.collect()
            importer.run(field_values, medias)
            _set_clip_origins(medias, importer.build_origin(field_values))
            collapse_duplicates(medias)
            _load_embedder_for_clips()
            build_diversity_tree()
        except MemoryError:
            medias.clear()
            gc.collect()
            update_progress(
                "idle", "", 0, 0,
                "Out of memory — this dataset is too large. "
                "Try a smaller dataset or free up system RAM.",
            )
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Staging – import datasets to temporary pkl files for the combine flow
# ---------------------------------------------------------------------------

STAGING_DIR = DATA_DIR / "staging"


def _stage_importer_in_background(importer, field_values: dict, label: str = "") -> None:
    """Run *importer*.run() in a daemon thread, saving the result to a staging pkl.

    Unlike ``_run_importer_in_background``, this does **not** modify the global
    ``medias`` dict.  Instead it writes a temporary ``.pkl`` file to
    :data:`STAGING_DIR` and sets the ``staging_result`` field on the progress
    tracker when finished.
    """

    def stage_task():
        try:
            temp_medias: dict = {}
            importer.run(field_values, temp_medias)

            if not temp_medias:
                update_progress("idle", "", 0, 0, "Import produced no medias.")
                return

            first = next(iter(temp_medias.values()))
            media_type = first.get("type", "audio")
            count = len(temp_medias)
            name = label or importer.display_name

            data_bytes = export_dataset_to_file(temp_medias)
            del temp_medias
            gc.collect()

            STAGING_DIR.mkdir(parents=True, exist_ok=True)
            staging_path = STAGING_DIR / f"stage_{uuid4().hex}.pkl"
            staging_path.write_bytes(data_bytes)
            del data_bytes
            gc.collect()

            update_progress(
                "idle", f"Staged: {name} ({count} medias)", 100, 100,
                staging_result={"path": str(staging_path), "name": name, "count": count, "media_type": media_type},
            )
        except MemoryError:
            gc.collect()
            update_progress(
                "idle", "", 0, 0,
                "Out of memory — this dataset is too large. "
                "Try a smaller dataset or free up system RAM.",
            )
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=stage_task, daemon=True)
    thread.start()


# ---------------------------------------------------------------------------
# Status / progress
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/status")
def dataset_status():
    """Return the current dataset status."""
    media_type = None
    if medias:
        media_type = next(iter(medias.values())).get("type", "audio")
    return jsonify(
        {
            "loaded": len(medias) > 0,
            "num_medias": len(medias),
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
    """List all registered importers (excluding those with dedicated UI).

    The frontend uses this endpoint to auto-render any importer that isn't
    already handled by a hard-coded UI panel (i.e. anything beyond the
    built-in pickle/folder/demo importers).

    Returns a JSON object::

        {
          "importers": [
            {
              "name": "sftp",
              "display_name": "SFTP Server",
              "description": "...",
              "fields": [ { "key": ..., "label": ..., "field_type": ..., ... }, ... ]
            },
            ...
          ]
        }
    """
    extended = [imp.to_dict() for imp in list_importers() if imp.name not in _BUILTIN_IMPORTER_NAMES]
    return jsonify({"importers": extended})


@datasets_bp.route("/api/dataset/all-importers")
def dataset_all_importers():
    """List all registered importers (including built-in ones).

    Used by the dashboard's dataset importer picker modal, which needs
    to show every available way to add a dataset.

    Returns a JSON object::

        {
          "importers": [
            {
              "name": "pickle",
              "display_name": "Pickle File",
              "description": "...",
              "fields": [ ... ]
            },
            ...
          ]
        }
    """
    all_importers = [imp.to_dict() for imp in list_importers()]
    return jsonify({"importers": all_importers})


# ---------------------------------------------------------------------------
# Available dataset files (for the Combine Existing Datasets UI)
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/available-files")
def available_dataset_files():
    """List ``.pkl`` files in the embeddings directory.

    Returns a JSON object::

        {
          "files": [
            {"name": "esc50_animals.pkl", "path": "/abs/path/to/esc50_animals.pkl", "size_mb": 12.3},
            ...
          ]
        }
    """
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
    """Combine multiple pickle datasets in a background thread.

    Expects a JSON body with a ``"datasets"`` key containing a list of
    absolute paths to ``.pkl`` files::

        {"datasets": ["/path/to/a.pkl", "/path/to/b.pkl"]}

    Returns ``{"ok": true}`` immediately; poll ``/api/dataset/progress``
    to track progress.
    """
    body = request.get_json(force=True) or {}
    dataset_paths = body.get("datasets", [])

    if not isinstance(dataset_paths, list) or len(dataset_paths) < 2:
        return jsonify({"error": "Provide at least two dataset file paths."}), 400

    for p in dataset_paths:
        if not Path(p).exists():
            return jsonify({"error": f"File not found: {p}"}), 404

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
    """Upload a ``.pkl`` file and save it to the staging directory.

    Returns basic metadata so the frontend can display the staged dataset
    in the combine list without loading the full dataset into memory.
    """
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
            data = pickle.load(f)  # noqa: S301
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
    """Run a registered importer in staging mode.

    Same interface as ``/api/dataset/import/<name>`` but saves the result to
    a temporary ``.pkl`` file instead of loading it into the app.  Poll
    ``/api/dataset/progress`` for status; on completion the response will
    contain a ``staging_result`` object.
    """
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

    _stage_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Staging started"})


@datasets_bp.route("/api/dataset/stage-demo/<name>", methods=["POST"])
def stage_demo(name: str):
    """Stage a demo dataset as a temporary ``.pkl`` file.

    Returns immediately; poll ``/api/dataset/progress`` for status.
    On completion the response will contain a ``staging_result`` object.
    """
    if name not in DEMO_DATASETS:
        return jsonify({"error": "Invalid dataset name"}), 400

    def stage_task():
        try:
            temp_medias: dict = {}
            load_demo_dataset(name, temp_medias)

            if not temp_medias:
                update_progress("idle", "", 0, 0, "Demo produced no medias.")
                return

            first = next(iter(temp_medias.values()))
            media_type = first.get("type", "audio")
            count = len(temp_medias)
            label = DEMO_DATASETS[name].get("label", name)

            data_bytes = export_dataset_to_file(temp_medias)
            del temp_medias
            gc.collect()

            STAGING_DIR.mkdir(parents=True, exist_ok=True)
            staging_path = STAGING_DIR / f"stage_{uuid4().hex}.pkl"
            staging_path.write_bytes(data_bytes)
            del data_bytes
            gc.collect()

            update_progress(
                "idle", f"Staged: {label} ({count} medias)", 100, 100,
                staging_result={"path": str(staging_path), "name": label, "count": count, "media_type": media_type},
            )
        except MemoryError:
            gc.collect()
            update_progress(
                "idle", "", 0, 0,
                "Out of memory — this dataset is too large. "
                "Try a smaller dataset or free up system RAM.",
            )
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=stage_task, daemon=True)
    thread.start()
    return jsonify({"ok": True, "message": "Staging demo dataset..."})


@datasets_bp.route("/api/dataset/staging", methods=["DELETE"])
def clear_staging():
    """Remove all files from the staging directory."""
    if STAGING_DIR.exists():
        for f in STAGING_DIR.iterdir():
            f.unlink(missing_ok=True)
    return jsonify({"ok": True})


# ---------------------------------------------------------------------------
# Generic import endpoint
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/import/<importer_name>", methods=["POST"])
def import_dataset(importer_name: str):
    """Run a registered importer by name in a background thread.

    For importers that have a field with ``field_type="file"``, the request
    must be ``multipart/form-data`` with the file stored under the field's
    ``key``.  All other field values are read from the form data (multipart)
    or from the JSON body.

    Returns ``{"ok": true, "message": "Loading started"}`` immediately; poll
    ``/api/dataset/progress`` to track progress.
    """
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

    _run_importer_in_background(importer, field_values)
    return jsonify({"ok": True, "message": "Loading started"})


# ---------------------------------------------------------------------------
# Demo datasets  (special-cased: their own discovery + load endpoints)
# ---------------------------------------------------------------------------


def _folder_has_content(folder) -> bool:
    """Return True if *folder* exists and contains at least one entry."""
    return folder is not None and folder.exists() and any(folder.iterdir())


@datasets_bp.route("/api/dataset/demo-list")
def demo_dataset_list():
    """List available demo datasets.

    Each dataset has a ``status`` field with one of three values:

    * ``"ready"`` – embeddings are cached and source data is present.
    * ``"needs_embedding"`` – source data is downloaded but not yet embedded.
    * ``"needs_download"`` – source data must be downloaded (and then embedded).
    """
    # Only include demo datasets whose media type is currently registered.
    from vtsearch.media import get as media_get

    demos = []
    for name, dataset_info in DEMO_DATASETS.items():
        media_type = dataset_info.get("media_type", "audio")

        # Skip datasets whose media type is not loaded into VTSearch.
        try:
            media_get(media_type)
        except KeyError:
            continue

        pkl_file = EMBEDDINGS_DIR / f"{name}.pkl"
        has_pkl = pkl_file.exists()

        required_folder = dataset_info.get("required_folder")
        has_source = _folder_has_content(required_folder)

        # Determine three-state status
        if has_pkl:
            if required_folder is not None and not has_source:
                # Stale pkl – source data was removed since last embed
                status = "needs_download"
            else:
                status = "ready"
        else:
            if required_folder is not None and has_source:
                status = "needs_embedding"
            else:
                status = "needs_download"

        # Calculate number of files from slice parameters
        num_categories = len(dataset_info["categories"])
        slice_start = dataset_info.get("slice_start", 0)
        slice_end = dataset_info.get("slice_end")
        if slice_end is not None:
            per_cat = slice_end - slice_start
        else:
            per_cat = 40  # generic fallback
        num_files = num_categories * per_cat

        # Calculate download size from the DemoDataset metadata
        if status == "ready":
            download_size_mb = pkl_file.stat().st_size / (1024 * 1024)
        elif status == "needs_embedding":
            download_size_mb = 0
        else:
            # Use the download_size_mb from DemoDataset metadata
            download_size_mb = dataset_info.get("download_size_mb", 0)

        demos.append(
            {
                "name": name,
                "label": dataset_info.get("label", name),
                "status": status,
                "ready": status == "ready",
                "num_files": num_files,
                "download_size_mb": round(download_size_mb, 1),
                "description": dataset_info.get("description", ""),
                "media_type": media_type,
                "num_categories": num_categories,
            }
        )
    return jsonify({"datasets": demos})


@datasets_bp.route("/api/dataset/load-demo", methods=["POST"])
def load_demo_dataset_route():
    """Load a demo dataset in a background thread."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    dataset_name = data.get("name")

    if not dataset_name or dataset_name not in DEMO_DATASETS:
        return jsonify({"error": "Invalid dataset name"}), 400

    def load_task():
        try:
            clear_dataset()
            gc.collect()
            load_demo_dataset(dataset_name, medias)
            demo_origin = {"importer": "demo", "params": {"name": dataset_name}}
            _set_clip_origins(medias, demo_origin)
            collapse_duplicates(medias)
            _load_embedder_for_clips()
            build_diversity_tree()
        except MemoryError:
            medias.clear()
            gc.collect()
            update_progress(
                "idle", "", 0, 0,
                "Out of memory — this dataset is too large. "
                "Try a smaller dataset or free up system RAM.",
            )
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=load_task, daemon=True)
    thread.start()

    return jsonify({"ok": True, "message": "Loading started"})


# ---------------------------------------------------------------------------
# Legacy endpoints – kept for backward compatibility.
# These now delegate to the appropriate importer internally.
# ---------------------------------------------------------------------------


@datasets_bp.route("/api/dataset/load-file", methods=["POST"])
def load_dataset_file():
    """Load a dataset from an uploaded pickle file.

    Delegates to the ``pickle`` importer.  Kept for backward compatibility
    with existing frontends and scripts.
    """
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
    """Generate dataset from a folder of media files.

    Delegates to the ``folder`` importer.  Kept for backward compatibility
    with existing frontends and scripts.
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    folder_path = data.get("path")
    media_type = data.get("media_type", "sounds")  # Default to sounds for backward compatibility

    if not folder_path:
        return jsonify({"error": "No folder path provided"}), 400

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
    if not medias:
        return jsonify({"error": "No dataset loaded"}), 400

    try:
        dataset_bytes = export_dataset_to_file(medias)
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
    return jsonify({"ok": True})


@datasets_bp.route("/api/dashboard/dataset-info")
def dashboard_dataset_info():
    """Return metadata about the currently loaded dataset for the dashboard.

    Returns a JSON object with ``name``, ``num_medias``, ``media_type``, and
    ``origin`` extracted from the first media that has origin info.
    """
    if not medias:
        return jsonify({"error": "No dataset loaded"}), 404

    first = next(iter(medias.values()))
    media_type = first.get("type", "audio")
    num_medias = len(medias)

    # Determine origin from the first media that has one
    origin = None
    for m in medias.values():
        o = m.get("origin")
        if o:
            importer = o.get("importer", "")
            params = o.get("params", {})
            # Build a human-readable origin string
            if importer == "demo":
                origin = f"demo:{params.get('name', '')}"
            elif importer == "pickle":
                origin = f"file:{params.get('filename', '')}"
            elif importer == "folder":
                origin = f"folder:{params.get('path', '')}"
            elif importer:
                origin = importer
            break

    # Use display name override if set, otherwise derive from origin
    display_name = get_dataset_display_name()
    if display_name:
        name = display_name
    else:
        name = origin or "Untitled"
        if origin and ":" in origin:
            name = origin.split(":", 1)[1] or origin

    # Build a source dict that can be used to reload the dataset later
    source = None
    for m in medias.values():
        o = m.get("origin")
        if isinstance(o, dict):
            source = o
            break

    return jsonify({
        "name": name,
        "num_medias": num_medias,
        "num_dupes": get_dupe_count(),
        "media_type": media_type,
        "origin": origin or "unknown",
        "source": source,
    })


@datasets_bp.route("/api/dashboard/dataset-rename", methods=["PUT"])
def dashboard_dataset_rename():
    """Set a custom display name for the currently loaded dataset."""
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Invalid request body"}), 400

    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "name is required"}), 400

    set_dataset_display_name(new_name)
    return jsonify({"success": True, "name": new_name})


@datasets_bp.route("/api/dataset/load-source", methods=["POST"])
def load_dataset_from_source():
    """Reload a dataset from a stored source origin dict.

    Accepts JSON with a ``source`` key containing the raw origin dict
    (as returned by ``/api/dashboard/dataset-info``).  Supports demo,
    pickle, folder, and other importer origins.

    Returns ``{"ok": true}`` and begins loading in a background thread.
    Poll ``/api/progress`` for status.
    """
    data = request.get_json(force=True, silent=True) or {}
    source = data.get("source")
    if not isinstance(source, dict):
        return jsonify({"error": "source must be an origin dict"}), 400
    return _load_from_origin(source)


def _load_from_origin(source: dict):
    """Start loading a dataset from a raw origin dict (internal helper)."""
    importer_name = source.get("importer", "")
    params = source.get("params", {})

    if importer_name == "dupe_set":
        members = source.get("members", [])
        if members:
            member_origin = members[0].get("origin")
            if isinstance(member_origin, dict):
                return _load_from_origin(member_origin)
        return jsonify({"error": "Cannot reload from dupe_set origin"}), 400

    if importer_name == "demo":
        demo_name = params.get("name", "")
        if demo_name not in DEMO_DATASETS:
            return jsonify({"error": f"Unknown demo dataset: {demo_name}"}), 400

        def load_task():
            try:
                clear_dataset()
                gc.collect()
                load_demo_dataset(demo_name, medias)
                _set_clip_origins(medias, {"importer": "demo", "params": {"name": demo_name}})
                collapse_duplicates(medias)
                _load_embedder_for_clips()
                build_diversity_tree()
            except MemoryError:
                medias.clear()
                gc.collect()
                update_progress("idle", "", 0, 0, "Out of memory — dataset too large.")
            except Exception as e:
                update_progress("idle", "", 0, 0, str(e))

        threading.Thread(target=load_task, daemon=True).start()
        return jsonify({"ok": True, "message": "Loading started"})

    if importer_name == "pickle":
        pkl_path = params.get("path", "")
        if not pkl_path or not Path(pkl_path).is_file():
            return jsonify({"error": f"Pickle file not found: {pkl_path}"}), 400
        importer = get_importer("pickle")
        if importer is None:
            return jsonify({"error": "pickle importer not available"}), 500
        _run_importer_in_background(importer, {"pkl_path": pkl_path})
        return jsonify({"ok": True, "message": "Loading started"})

    if importer_name == "folder":
        folder_path = params.get("path", "")
        if not folder_path or not Path(folder_path).is_dir():
            return jsonify({"error": f"Folder not found: {folder_path}"}), 400
        importer = get_importer("folder")
        if importer is None:
            return jsonify({"error": "folder importer not available"}), 500
        field_values = {"path": folder_path}
        media_type = params.get("media_type", "")
        if media_type:
            field_values["media_type"] = media_type
        _run_importer_in_background(importer, field_values)
        return jsonify({"ok": True, "message": "Loading started"})

    # Generic fallback: try the named importer with the stored params
    importer = get_importer(importer_name)
    if importer is None:
        return jsonify({"error": f"Unknown importer: {importer_name}"}), 400
    _run_importer_in_background(importer, params)
    return jsonify({"ok": True, "message": "Loading started"})
