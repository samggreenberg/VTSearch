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
``import-local-files``, ``load-file``) or whose success body is a binary stream (``export``)
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

from flask import request, send_file
from flask_smorest import Blueprint, abort

import vtscore.security.path_validation as _paths
from vtscore.config import DATA_DIR
from vtscore.datasets import DEMO_DATASETS, export_dataset_to_file, get_importer
from vtscore.datasets.load_pipeline import (
    _run_importer_in_background,
    _run_origin_load_in_background,
    clear_dataset,
)
from vtscore.datasets.registry import remove_loaded_id as _reg_remove_loaded
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


def _parse_clipper_params(raw: str) -> dict | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("clipper_params must be a JSON object")
        return parsed
    except (ValueError, TypeError) as exc:
        abort(400, message=f"Invalid clipper_params: {exc}")


def _save_uploaded_files_to_temp(files, upload_dir: Path) -> int:
    """Stream each uploaded file into *upload_dir*; return saved count."""
    saved = 0
    for f in files:
        rel = _safe_relative_upload_path(f.filename or "")
        if rel is None:
            continue
        dest = upload_dir / Path(*rel.parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        f.save(dest)
        saved += 1
    return saved


def _build_local_folder_field_values(form, upload_dir: Path, clipper_params: dict | None) -> dict:
    """Translate the multipart form into the importer's field_values dict."""
    field_values: dict = {
        "path": str(upload_dir),
        "media_type": (form.get("media_type") or "").strip(),
        "recursive": (form.get("recursive") or "true").strip().lower() not in ("false", "0", "no", "off"),
    }
    for key in ("embedder", "converters", "source_specs"):
        val = (form.get(key) or "").strip()
        if val:
            field_values[key] = val
    clipper = (form.get("clipper") or "").strip()
    if clipper:
        field_values["clipper"] = clipper
        if clipper_params is not None:
            field_values["clipper_params"] = clipper_params
    return field_values


def _extract_clipper_config(field_values: dict) -> tuple[str, dict | None, list[dict] | None]:
    """Pop clipper-related keys; mutate *field_values* in place.

    Returns ``(clipper_name, clipper_params, chain_steps)``.
    """
    from vtscore.datasets.load_pipeline import _parse_chain_field

    clipper_name = field_values.pop("clipper", "") or ""
    clipper_params = field_values.pop("clipper_params", None)
    chain_steps = _parse_chain_field(field_values.pop("clipper_chain", None))
    field_values["clipper"] = clipper_name
    return clipper_name, clipper_params, chain_steps


def _make_local_folder_loader(importer, field_values: dict, upload_dir: Path, media_type: str):
    """Build the ``target_medias -> None`` task that the importer will run."""
    from vtscore.datasets.load_pipeline import auto_chunk_size, consume_chunks_into

    use_chunked = getattr(importer, "supports_chunked", False)
    chunk_size = auto_chunk_size(media_type) if use_chunked else 0

    def _load(target_medias):
        try:
            if use_chunked:
                consume_chunks_into(target_medias, importer.run_chunked(field_values, chunk_size))
            else:
                importer.run(field_values, target_medias)
        finally:
            shutil.rmtree(upload_dir, ignore_errors=True)

    return _load


@datasets_load_bp.route("/api/dataset/import-local-folder", methods=["POST"])
@datasets_load_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_load_bp.alt_response(
    400,
    description=(
        "Multipart body is missing the ``files`` field, has no valid files, is "
        "missing ``media_type``, or carries malformed ``clipper_params``."
    ),
)
@datasets_load_bp.alt_response(500, description="The server_folder importer is unavailable.")
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
        abort(400, message="No files uploaded")

    importer = get_importer("server_folder")
    if importer is None:
        abort(500, message="server_folder importer not available")

    media_type = (request.form.get("media_type") or "").strip()
    if not media_type:
        abort(400, message="Missing required field: 'media_type'")

    clipper_params = _parse_clipper_params(request.form.get("clipper_params") or "")

    LOCAL_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload_dir = Path(tempfile.mkdtemp(prefix="local_folder_", dir=LOCAL_UPLOADS_DIR))

    try:
        saved = _save_uploaded_files_to_temp(files, upload_dir)
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise
    if saved == 0:
        shutil.rmtree(upload_dir, ignore_errors=True)
        abort(400, message="No valid files in upload")

    field_values = _build_local_folder_field_values(request.form, upload_dir, clipper_params)

    # Origin is intentionally synthetic — the on-disk path is a temp dir we
    # are about to delete, so storing it on each media would be misleading
    # and ``can_reload_from_origin`` would (correctly) refuse to reload.
    origin = {
        "importer": "server_folder",
        "params": {"path": "<browser_upload>", "media_type": media_type},
    }

    clipper_name, clipper_params_out, chain_steps = _extract_clipper_config(field_values)
    loader = _make_local_folder_loader(importer, field_values, upload_dir, media_type)

    from vtsearch.auth import get_current_user
    from vtscore.datasets.load_pipeline import _normalize_media_type

    dataset_name = (request.form.get("dataset_name") or "").strip() or "Local folder upload"
    task_id = _run_origin_load_in_background(
        loader,
        origin,
        name=dataset_name,
        clipper=clipper_name,
        clipper_params=clipper_params_out,
        chain_steps=chain_steps,
        embedder=field_values.get("embedder", ""),
        created_by=get_current_user(),
        media_type=_normalize_media_type(media_type),
    )
    return {"ok": True, "message": "Loading started", "task_id": str(task_id) if task_id else ""}


@datasets_load_bp.route("/api/dataset/import-local-files", methods=["POST"])
@datasets_load_bp.response(200, DatasetLoadStartedResponseSchema)
@datasets_load_bp.alt_response(
    400,
    description=(
        "Multipart body is missing the ``paths_file`` field, is missing "
        "``media_type``, or carries malformed ``clipper_params``."
    ),
)
@datasets_load_bp.alt_response(500, description="The server_files importer is unavailable.")
def import_local_files():
    """Import a paths file uploaded from the user's *browser* machine.

    The browser picks a single file (a ``.txt`` / ``.list`` with one
    server-side media path per line, or a ``.npz`` archive that also
    supplies pre-computed embedding vectors) and POSTs it as the
    multipart field ``paths_file``.  We stream it to a server-side
    temporary directory and then delegate to the regular
    :mod:`server_files` importer for resolution and embedding.  The
    temp directory is removed once the importer finishes (success or
    failure).
    """
    storage = request.files.get("paths_file")
    if not storage or not storage.filename:
        abort(400, message="No paths file uploaded")

    importer = get_importer("server_files")
    if importer is None:
        abort(500, message="server_files importer not available")

    media_type = (request.form.get("media_type") or "").strip()
    if not media_type:
        abort(400, message="Missing required field: 'media_type'")

    clipper_params = _parse_clipper_params(request.form.get("clipper_params") or "")

    LOCAL_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    upload_dir = Path(tempfile.mkdtemp(prefix="local_files_", dir=LOCAL_UPLOADS_DIR))
    # Preserve the suffix so the importer picks the right reader (.txt vs .npz).
    suffix = Path(storage.filename).suffix
    paths_file = upload_dir / f"paths_file{suffix}"
    try:
        storage.save(paths_file)
    except Exception:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise

    field_values: dict = {
        "media_type": media_type,
        "paths_file": str(paths_file),
    }
    for key in ("embedder", "source_specs"):
        val = (request.form.get(key) or "").strip()
        if val:
            field_values[key] = val
    clipper = (request.form.get("clipper") or "").strip()
    if clipper:
        field_values["clipper"] = clipper
        if clipper_params is not None:
            field_values["clipper_params"] = clipper_params

    # Origin is intentionally synthetic — the on-disk paths file is in a temp
    # dir we are about to delete, so storing it on each media would be
    # misleading and ``can_reload_from_origin`` would (correctly) refuse to
    # reload.
    origin = {
        "importer": "server_files",
        "params": {"paths_file": "<browser_upload>", "media_type": media_type},
    }

    clipper_name, clipper_params_out, chain_steps = _extract_clipper_config(field_values)
    loader = _make_local_folder_loader(importer, field_values, upload_dir, media_type)

    from vtsearch.auth import get_current_user
    from vtscore.datasets.load_pipeline import _normalize_media_type

    dataset_name = (request.form.get("dataset_name") or "").strip() or "Local files upload"
    task_id = _run_origin_load_in_background(
        loader,
        origin,
        name=dataset_name,
        clipper=clipper_name,
        clipper_params=clipper_params_out,
        chain_steps=chain_steps,
        embedder=field_values.get("embedder", ""),
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
        from vtscore.converters import get_converter  # noqa: PLC0415

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
