"""Dataset load endpoints: demo, file, folder, browser-folder upload, source
reload, plus export and clear.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``.

Routes with a single marshmallow-able body (``load-demo``, ``load-folder``,
``load-source``, ``clear``) use the standard ``@arguments`` + ``@response``
decorators. Schema-level validation failures (missing required ``path`` /
``name`` / ``source``) surface as 422 with the standard ``errors``
envelope; handler-level rejects (unknown demo, invalid path, importer
not available) keep their HTTP codes (400 / 500) with the standard
``message`` envelope (except 404s, which the app-level ``NotFound``
handler intercepts — see plan doc).

Routes whose request body is multipart (``import-local-folder``,
``load-file``) or whose success body is a binary stream (``export``)
omit ``@arguments`` and declare error responses via ``alt_response`` —
same pattern as ``add-to-pile`` / ``server-media-files/upload`` /
``server-media-files/<f>/thumbnail`` in ``media/list.py`` and
``media/server.py``.
"""

import io
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from flask import request, send_file
from flask_smorest import Blueprint, abort

import vtsearch.security.path_validation as _paths
from vtsearch.config import DATA_DIR
from vtsearch.datasets import DEMO_DATASETS, export_dataset_to_file, get_importer
from vtsearch.datasets.load_pipeline import (
    _run_importer_in_background,
    _run_origin_load_in_background,
    clear_dataset,
)
from vtsearch.datasets.registry import remove_loaded_id as _reg_remove_loaded
from vtsearch.routes._shared import format_exception_detail
from vtsearch.routes.datasets._helpers import _safe_relative_upload_path
from vtsearch.schemas.datasets import (
    DatasetClearResponseSchema,
    DatasetLoadDemoRequestSchema,
    DatasetLoadFolderRequestSchema,
    DatasetLoadSourceRequestSchema,
    DatasetLoadStartedResponseSchema,
)
from vtsearch.state import snapshot_medias, unregister_context

datasets_load_bp = Blueprint(
    "datasets_load",
    __name__,
    description="Load datasets from demos, files, folders, origins; export and clear.",
)

LOCAL_UPLOADS_DIR = DATA_DIR / "local_uploads"


@datasets_load_bp.route("/api/dataset/import-local-folder", methods=["POST"])
@datasets_load_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_load_bp.alt_response(
    400,
    description=(
        "Multipart body is missing the ``files`` field, has no valid files, is "
        "missing ``media_type``, or carries malformed ``clipper_params`` / "
        "``vectors_file``."
    ),
)
@datasets_load_bp.alt_response(500, description="The server_folder importer is unavailable.")
def import_local_folder():  # noqa: C901
    """Import a folder uploaded from the user's *browser* machine.

    The browser uses ``<input type="file" webkitdirectory>`` to let the
    user pick a directory; each selected ``File`` is appended to the
    multipart body under the key ``"files"`` with its ``webkitRelativePath``
    as the multipart filename.  We stream each file to a temporary
    directory on the server (preserving sub-directory structure) and then
    delegate to the regular folder importer to do the actual scanning,
    embedding, and dataset registration.  The temp directory is removed
    once the importer finishes (success or failure).

    Local-files uploads may additionally include a ``vectors_file`` form
    field carrying a ``.npz`` archive of pre-computed embedding vectors
    keyed by uploaded-file name (basename or relative path).  Files
    whose name matches an NPZ key reuse the supplied vector instead of
    running the embedding model.
    """
    files = request.files.getlist("files")
    if not files:
        abort(400, message="No files uploaded")

    importer = get_importer("server_folder")
    if importer is None:
        abort(500, message="server_folder importer not available")

    media_type = (request.form.get("media_type") or "").strip()
    if not media_type:
        abort(400, message="Missing required field: 'media_type'")

    embedder = (request.form.get("embedder") or "").strip()
    clipper = (request.form.get("clipper") or "").strip()
    converters = (request.form.get("converters") or "").strip()
    source_specs = (request.form.get("source_specs") or "").strip()
    recursive_raw = (request.form.get("recursive") or "true").strip().lower()
    recursive = recursive_raw not in ("false", "0", "no", "off")
    user_dataset_name = (request.form.get("dataset_name") or "").strip()
    clipper_params_raw = request.form.get("clipper_params") or ""
    clipper_params: dict | None = None
    if clipper_params_raw:
        try:
            clipper_params = json.loads(clipper_params_raw)
            if not isinstance(clipper_params, dict):
                raise ValueError("clipper_params must be a JSON object")
        except (ValueError, TypeError) as exc:
            abort(400, message=f"Invalid clipper_params: {exc}")

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
        abort(400, message="No valid files in upload")

    # Optional .npz of pre-computed embedding vectors.  Saved into the
    # upload directory and parsed before kicking off the importer; the
    # resulting mapping is handed to the server_folder importer via its
    # ``content_vectors`` attribute (cleared in a ``finally`` block to
    # avoid bleeding into unrelated runs of the singleton).
    content_vectors: dict[str, Any] = {}
    vectors_file_storage = request.files.get("vectors_file")
    if vectors_file_storage and vectors_file_storage.filename:
        from vtsearch.datasets.importers._npz_vectors import read_npz_filenames_and_vectors

        npz_path = upload_dir / "__vtsearch_vectors__.npz"
        try:
            vectors_file_storage.save(npz_path)
            content_vectors = dict(read_npz_filenames_and_vectors(npz_path))
        except Exception as exc:
            shutil.rmtree(upload_dir, ignore_errors=True)
            abort(400, message=f"Invalid vectors_file: {exc}")
        finally:
            # The npz is no longer needed once it's parsed; remove it so
            # the importer doesn't see it as a media file.
            try:
                npz_path.unlink()
            except FileNotFoundError:
                pass

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
    if source_specs:
        field_values["source_specs"] = source_specs

    # Origin is intentionally synthetic — the on-disk path is a temp dir we
    # are about to delete, so storing it on each media would be misleading
    # and ``can_reload_from_origin`` would (correctly) refuse to reload.
    origin = {
        "importer": "server_folder",
        "params": {"path": "<browser_upload>", "media_type": media_type},
    }

    clipper_name = field_values.pop("clipper", "") or ""
    inner_clipper_params = field_values.pop("clipper_params", None)
    from vtsearch.datasets.load_pipeline import (
        _normalize_media_type,
        _parse_chain_field,
        auto_chunk_size,
        consume_chunks_into,
    )

    inner_chain_steps = _parse_chain_field(field_values.pop("clipper_chain", None))
    field_values["clipper"] = clipper_name
    if (clipper_name and not clipper_name.endswith("_default")) or inner_chain_steps:
        field_values["skip_embedding"] = True

    from vtsearch.auth import get_current_user

    use_chunked = getattr(importer, "supports_chunked", False)
    chunk_size = auto_chunk_size(media_type) if use_chunked else 0

    def _load(target_medias):
        previous_vectors = importer.content_vectors
        if content_vectors:
            importer.content_vectors = content_vectors
        try:
            if use_chunked:
                consume_chunks_into(target_medias, importer.run_chunked(field_values, chunk_size))
            else:
                importer.run(field_values, target_medias)
        finally:
            if content_vectors:
                importer.content_vectors = previous_vectors
            shutil.rmtree(upload_dir, ignore_errors=True)

    task_id = _run_origin_load_in_background(
        _load,
        origin,
        name=user_dataset_name or "Local folder upload",
        clipper=clipper_name,
        clipper_params=inner_clipper_params,
        chain_steps=inner_chain_steps,
        embedder=embedder,
        created_by=get_current_user(),
        media_type=_normalize_media_type(media_type),
    )
    return {"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""}


@datasets_load_bp.route("/api/dataset/load-demo", methods=["POST"])
@datasets_load_bp.arguments(DatasetLoadDemoRequestSchema)
@datasets_load_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_load_bp.alt_response(400, description="Unknown demo dataset name.")
@datasets_load_bp.alt_response(500, description="The demo importer is unavailable.")
def load_demo_dataset_route(body: dict):
    """Load a demo dataset in a background thread.

    When a ``converter`` is specified, the demo data is loaded using its
    original media type, then converted to the converter's target type.
    The resulting dataset has the *target* type, not the demo's original
    type.
    """
    dataset_name = body.get("name")
    embedder_name = body.get("embedder", "")
    clipper_name = body.get("clipper", "")
    converter_name = body.get("converter", "")
    user_dataset_name = (body.get("dataset_name") or "").strip()

    if not dataset_name or dataset_name not in DEMO_DATASETS:
        abort(400, message="Invalid dataset name")

    importer = get_importer("demo")
    if importer is None:
        abort(500, message="demo importer not available")

    demo_info = DEMO_DATASETS[dataset_name]
    field_values: dict = {"name": dataset_name}
    if user_dataset_name:
        field_values["dataset_name"] = user_dataset_name
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
    return {"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""}


@datasets_load_bp.route("/api/dataset/load-file", methods=["POST"])
@datasets_load_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_load_bp.alt_response(400, description="Multipart body is missing a file or no file is selected.")
def load_dataset_file():
    """Load a dataset from an uploaded pickle file."""
    if "file" not in request.files:
        abort(400, message="No file provided")

    file = request.files["file"]
    if not file.filename:
        abort(400, message="No file selected")

    importer = get_importer("pickle")
    # Read file contents before passing to background thread, since the
    # Flask FileStorage stream is only valid during the request lifecycle.
    file_bytes = io.BytesIO(file.read())
    file_bytes.name = file.filename
    task_id = _run_importer_in_background(importer, {"file": file_bytes})
    return {"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""}


@datasets_load_bp.route("/api/dataset/load-folder", methods=["POST"])
@datasets_load_bp.arguments(DatasetLoadFolderRequestSchema)
@datasets_load_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_load_bp.alt_response(400, description="Invalid or missing folder path.")
def load_dataset_folder(body: dict):
    """Generate dataset from a folder of media files."""
    folder_path = body.get("path")
    media_type = body.get("media_type", "audio")

    if not folder_path:
        abort(400, message="No folder path provided")

    try:
        _paths.validate_server_filepath(str(folder_path), base_dir=_paths.get_file_access_base_dir())
    except ValueError as exc:
        abort(400, message=str(exc))

    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        abort(400, message="Invalid folder path")

    importer = get_importer("server_folder")
    task_id = _run_importer_in_background(importer, {"path": str(folder), "media_type": media_type})
    return {"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""}


@datasets_load_bp.route("/api/dataset/load-source", methods=["POST"])
@datasets_load_bp.arguments(DatasetLoadSourceRequestSchema)
@datasets_load_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_load_bp.alt_response(400, description="Unknown importer, unreloadable origin, or invalid path inside origin.")
def load_dataset_from_source(body: dict):
    """Reload a dataset from a stored source origin dict."""
    return _load_from_origin(body["source"])


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
        abort(400, message="Cannot reload from dupe_set origin")

    # --- real importers: generic dispatch ---

    importer = get_importer(importer_name)
    if importer is None:
        abort(400, message=f"Unknown importer: {importer_name}")

    if not importer.can_reload_from_origin(source):
        abort(400, message=f"Cannot reload from {importer_name} origin (source not available)")

    field_values = importer.reload_from_origin(source)
    if field_values is None:
        abort(400, message=f"Cannot reload from {importer_name} origin")

    # Validate any server file paths in the field values
    _base = _paths.get_file_access_base_dir()
    for key, val in field_values.items():
        if isinstance(val, str) and ("/" in val or "\\" in val):
            try:
                _paths.validate_server_filepath(val, base_dir=_base)
            except ValueError as exc:
                abort(400, message=str(exc))

    task_id = _run_importer_in_background(importer, field_values)
    return {"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""}


@datasets_load_bp.route("/api/dataset/export")
@datasets_load_bp.alt_response(400, description="No dataset is currently loaded.")
@datasets_load_bp.alt_response(500, description="Dataset export failed unexpectedly.")
def export_dataset():
    """Export the current dataset to a pickle file.

    Success returns a binary ``application/octet-stream`` download; that
    body is left undescribed in the spec (mirroring the audio / video /
    image streaming routes in ``media/list.py``).
    """
    snap = snapshot_medias()
    if not snap:
        abort(400, message="No dataset loaded")

    try:
        dataset_bytes = export_dataset_to_file(snap)
        return send_file(
            io.BytesIO(dataset_bytes),
            mimetype="application/octet-stream",
            download_name="vtsearch_dataset.pkl",
            as_attachment=True,
        )
    except Exception as exc:
        import logging

        logging.getLogger(__name__).exception("Dataset export failed")
        abort(500, message=f"Dataset export failed: {format_exception_detail(exc)}")


@datasets_load_bp.route("/api/dataset/clear", methods=["POST"])
@datasets_load_bp.response(200, DatasetClearResponseSchema)
def clear_dataset_route():
    """Clear the request-scoped dataset from memory.

    Uses the ``X-Dataset-Id`` header (via ``get_active_context()``) to
    determine which dataset to clear.
    """
    from vtsearch.state import get_active_context

    ctx = get_active_context()
    ds_id = ctx.dataset_id if ctx.dataset_id else None
    if ds_id:
        unregister_context(ds_id)
        _reg_remove_loaded(ds_id)
    else:
        clear_dataset()
    return {"ok": True}
