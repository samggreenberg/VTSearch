"""Dashboard, demo dataset listing, and other UI helper routes.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``. Schema-level
validation surfaces as 422; handler-level rejects (unknown demo, missing
source on disk, path-traversal) use ``abort()`` with the standard
``message`` envelope.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from flask_smorest import Blueprint, abort

from vtsearch.config import DATA_DIR, EMBEDDINGS_DIR
from vtsearch.datasets import DEMO_DATASETS
from vtsearch.datasets.loader import read_pkl_clipper, read_pkl_embedder
from vtsearch.datasets.load_pipeline import _origin_to_str
from vtsearch.routes._shared import format_mtime
from vtsearch.schemas.datasets import (
    BrowseMediaFilesQuerySchema,
    BrowseMediaFilesResponseSchema,
    BrowseMediaFilesSelectRequestSchema,
    BrowseMediaFilesSelectResponseSchema,
    DashboardDatasetInfoResponseSchema,
    DashboardDatasetRenameRequestSchema,
    DashboardDatasetRenameResponseSchema,
    DashboardDiskUsageResponseSchema,
    DemoCategoriesResponseSchema,
    DemoDatasetListQuerySchema,
    DemoDatasetListResponseSchema,
    DetectMediaTypeQuerySchema,
    DetectMediaTypeResponseSchema,
)
from vtsearch.state import (
    get_dataset_display_name,
    get_dupe_count,
    set_dataset_display_name,
    snapshot_medias,
)

datasets_ui_bp = Blueprint(
    "datasets_ui",
    __name__,
    description="Dashboard, demo dataset listing, browse-media-files picker, and disk usage.",
)


# ---------------------------------------------------------------------------
# Demo dataset listing
# ---------------------------------------------------------------------------


def _folder_has_content(folder) -> bool:
    """Return True if *folder* exists and contains at least one entry."""
    return folder is not None and folder.exists() and any(folder.iterdir())


@datasets_ui_bp.route("/api/dataset/demo-list")
@datasets_ui_bp.arguments(DemoDatasetListQuerySchema, location="query")
@datasets_ui_bp.response(200, DemoDatasetListResponseSchema)
def demo_dataset_list(query: dict):  # noqa: C901
    """List available demo datasets.

    Each dataset has a ``status`` field with one of three values:

    * ``"ready"`` – embeddings are cached and source data is present.
    * ``"needs_embedding"`` – source data is downloaded but not yet embedded.
    * ``"needs_download"`` – source data must be downloaded (and then embedded).
    """
    # Only include demo datasets whose media type is currently registered.
    from vtsearch.converters import list_converters_for_source
    from vtsearch.media import get as media_get

    # Optional embedder/clipper filters: when the caller specifies an embedder
    # or clipper, a cached pkl is only considered "ready" if it was produced by
    # those same values.
    requested_embedder = query.get("embedder", "").strip()
    requested_clipper = query.get("clipper", "").strip()

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

        # If the pkl exists but was embedded with a different embedder than
        # the one the user selected, downgrade from "ready" to "needs_embedding".
        pkl_embedder: str | None = None
        if status == "ready" and requested_embedder:
            pkl_embedder = read_pkl_embedder(pkl_file)
            if pkl_embedder is not None and pkl_embedder != requested_embedder:
                status = "needs_embedding"

        # Same check for clipper: if the pkl was created with a different
        # clipper, it needs re-clipping (and re-embedding of the clips).
        pkl_clipper: str | None = None
        if status == "ready" and requested_clipper:
            pkl_clipper = read_pkl_clipper(pkl_file)
            if pkl_clipper is not None and pkl_clipper != requested_clipper:
                status = "needs_embedding"

        # Calculate number of files from slice parameters
        num_categories = len(dataset_info["categories"])
        items_per_category = dataset_info.get("items_per_category") or 0
        slice_frac_start = dataset_info.get("slice_frac_start")
        slice_frac_end = dataset_info.get("slice_frac_end")
        if slice_frac_start is not None:
            frac_start = slice_frac_start
            frac_end = slice_frac_end if slice_frac_end is not None else 1.0
            # Fall back to 40 only when the dataset didn't declare its size.
            base_per_cat = items_per_category if items_per_category > 0 else 40
            per_cat = int(base_per_cat * (frac_end - frac_start))
            num_files = num_categories * per_cat
        else:
            slice_start = dataset_info.get("slice_start", 0)
            slice_end = dataset_info.get("slice_end")
            if slice_end is not None:
                per_cat = slice_end - slice_start
            elif items_per_category > 0:
                per_cat = items_per_category - slice_start
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

        # Converters that consume this demo's media type (M→N converters).
        available_converters = [c.to_dict() for c in list_converters_for_source(media_type)]

        # Resolve pkl_embedder if not already read above.
        if pkl_embedder is None and has_pkl:
            pkl_embedder = read_pkl_embedder(pkl_file)

        # Resolve pkl_clipper if not already read above.
        if pkl_clipper is None and has_pkl:
            pkl_clipper = read_pkl_clipper(pkl_file)

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
                "available_converters": available_converters,
                "pkl_embedder": pkl_embedder or "",
                "pkl_clipper": pkl_clipper or "",
            }
        )
    return {"datasets": demos}


@datasets_ui_bp.route("/api/dataset/demo-categories/<name>")
@datasets_ui_bp.response(200, DemoCategoriesResponseSchema)
@datasets_ui_bp.alt_response(404, description="No demo dataset with that name is registered.")
def demo_dataset_categories(name: str):
    """List the categories within a specific demo dataset.

    Returns ``{"categories": ["cat1", "cat2", ...]}`` for the named demo.
    """
    dataset_info = DEMO_DATASETS.get(name)
    if dataset_info is None:
        abort(404, message=f"Unknown demo dataset: {name}")

    categories = dataset_info.get("categories", [])
    return {"categories": categories}


# ---------------------------------------------------------------------------
# File browsing for the media example picker
# ---------------------------------------------------------------------------

# Collect all media-file extensions from every registered media type.
_MEDIA_EXTENSIONS: set[str] | None = None


def _media_extensions() -> set[str]:
    """Lazily build the set of known media-file extensions (lowercase, with dot)."""
    global _MEDIA_EXTENSIONS
    if _MEDIA_EXTENSIONS is None:
        from vtsearch.media import all_types

        exts: set[str] = set()
        for mt in all_types():
            for pattern in mt.file_extensions:
                # pattern looks like "*.wav" — extract ".wav"
                ext = pattern.lstrip("*").lower()
                exts.add(ext)
        _MEDIA_EXTENSIONS = exts
    return _MEDIA_EXTENSIONS


def _resolve_browse_root(source: str) -> Path | None:
    """Map a browse source identifier to an absolute directory path.

    Supported source values:

    * ``demo:<name>`` — the ``required_folder`` of the named demo dataset.
    * ``folder`` — the configured ``saved_datasets_dir``.

    Returns ``None`` if the source is unrecognised or the directory does not
    exist.
    """
    if source.startswith("demo:"):
        demo_name = source[5:]
        info = DEMO_DATASETS.get(demo_name)
        if info is None:
            return None
        folder = info.get("required_folder")
        if folder is None or not folder.is_dir():
            return None
        return folder.resolve()

    if source == "folder":
        from vtsearch.settings import get_saved_datasets_dir

        ds_dir = get_saved_datasets_dir()
        ds_dir.mkdir(parents=True, exist_ok=True)
        return ds_dir.resolve()

    return None


@datasets_ui_bp.route("/api/browse-media-files")
@datasets_ui_bp.arguments(BrowseMediaFilesQuerySchema, location="query")
@datasets_ui_bp.response(200, BrowseMediaFilesResponseSchema)
@datasets_ui_bp.alt_response(400, description="Path escapes the source root, or other invalid path.")
@datasets_ui_bp.alt_response(404, description="Source / directory not found on disk.")
def browse_media_files(query: dict):
    """List files and subdirectories within an allowed root.

    Query parameters:

    * ``source`` — one of ``demo:<name>`` or ``folder``.
    * ``path`` — relative sub-path within the root (default ``""``).

    Returns::

        {
            "directories": [{"name": "dog"}, {"name": "cat"}, ...],
            "files": [
                {"name": "bark.wav", "path": "dog/bark.wav", "size_bytes": 12345},
                ...
            ]
        }
    """
    source = query.get("source", "").strip()
    subpath = query.get("path", "").strip()

    root = _resolve_browse_root(source)
    if root is None:
        abort(404, message="Source not found or not available on disk")

    # Resolve the target directory, preventing traversal.
    target = (root / subpath).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        abort(400, message="Invalid path")

    if not target.is_dir():
        abort(404, message="Directory not found")

    known_exts = _media_extensions()
    directories: list[dict] = []
    files: list[dict] = []

    for entry in sorted(target.iterdir()):
        if entry.name.startswith("."):
            continue
        rel = str(entry.relative_to(root))
        if entry.is_dir():
            directories.append({"name": entry.name, "path": rel, "modified_at": format_mtime(entry)})
        elif entry.is_file() and entry.suffix.lower() in known_exts:
            files.append(
                {
                    "name": entry.name,
                    "path": rel,
                    "size_bytes": entry.stat().st_size,
                    "modified_at": format_mtime(entry),
                }
            )

    return {"directories": directories, "files": files, "root_path": str(root)}


@datasets_ui_bp.route("/api/dataset/detect-media-type")
@datasets_ui_bp.arguments(DetectMediaTypeQuerySchema, location="query")
@datasets_ui_bp.response(200, DetectMediaTypeResponseSchema)
@datasets_ui_bp.alt_response(400, description="Path escapes the source root.")
@datasets_ui_bp.alt_response(404, description="Source not available, or directory not found.")
def detect_media_type(query: dict):
    """Sample a folder and report which media type dominates by extension.

    Powers the auto-detect hint in the import modal: rather than making
    the user pick ``media_type`` blindly, the modal calls this after the
    user selects a folder and pre-fills the dropdown with whichever type
    has the most matching files in the first ``limit`` files of the tree.

    The 404 returned when the source is unavailable on disk is
    intercepted by the app-level ``NotFound`` errorhandler and keeps the
    legacy ``{"error": "Not Found", "request_id": ...}`` envelope.
    """
    from vtsearch.datasets.media_type_detection import detect_media_types_in_folder

    source = query["source"].strip() or "folder"
    subpath = query["path"].strip()
    recursive = query["recursive"]
    limit = max(1, min(query["limit"], 500))

    root = _resolve_browse_root(source)
    if root is None:
        abort(404, message="Source not found or not available on disk")

    target = (root / subpath).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        abort(400, message="Invalid path")

    if not target.is_dir():
        abort(404, message="Directory not found")

    return detect_media_types_in_folder(target, recursive=recursive, limit=limit)


@datasets_ui_bp.route("/api/browse-media-files/select", methods=["POST"])
@datasets_ui_bp.arguments(BrowseMediaFilesSelectRequestSchema)
@datasets_ui_bp.response(201, BrowseMediaFilesSelectResponseSchema)
@datasets_ui_bp.alt_response(400, description="Path escapes the source root.")
@datasets_ui_bp.alt_response(404, description="Source or file not found on disk.")
def select_browsed_file(body: dict):
    """Copy a file from a browse source into ``data/example_media/``.

    Expects JSON::

        {"source": "demo:esc50_s", "path": "dog/1-100032-A-0.wav"}

    Returns::

        {"filename": "<safe-name>", "original_name": "1-100032-A-0.wav"}
    """
    import shutil
    import uuid

    from vtsearch.config import DATA_DIR

    source = body["source"].strip()
    file_path = body["path"].strip()

    root = _resolve_browse_root(source)
    if root is None:
        abort(404, message="Source not found")

    # Resolve and validate the file path within the root.
    abs_path = (root / file_path).resolve()
    try:
        abs_path.relative_to(root)
    except ValueError:
        abort(400, message="Invalid path")

    if not abs_path.is_file():
        abort(404, message="File not found")

    # Copy to data/example_media/ with a unique prefix to avoid collisions.
    dest_dir = DATA_DIR / "example_media"
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex[:8]}_{abs_path.name}"
    dest = dest_dir / safe_name
    shutil.copy2(abs_path, dest)

    return {"filename": safe_name, "original_name": abs_path.name}


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@datasets_ui_bp.route("/api/dashboard/dataset-info")
@datasets_ui_bp.response(200, DashboardDatasetInfoResponseSchema)
@datasets_ui_bp.alt_response(404, description="No dataset is currently loaded.")
def dashboard_dataset_info():
    """Return metadata about the currently loaded dataset for the dashboard.

    Returns a JSON object with ``name``, ``num_medias``, ``media_type``, and
    ``origin`` extracted from the first media that has origin info.
    """
    snap = snapshot_medias()
    if not snap:
        abort(404, message="No dataset loaded")

    first = next(iter(snap.values()))
    media_type = first.get("type", "audio")
    num_medias = len(snap)

    # Determine origin from the first media that has one
    origin = None
    for m in snap.values():
        o = m.get("origin")
        if o:
            origin = _origin_to_str(o)
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
    for m in snap.values():
        o = m.get("origin")
        if isinstance(o, dict):
            source = o
            break

    return {
        "name": name,
        "num_medias": num_medias,
        "num_dupes": get_dupe_count(),
        "media_type": media_type,
        "origin": origin or "unknown",
        "source": source,
    }


@datasets_ui_bp.route("/api/dashboard/dataset-rename", methods=["PUT"])
@datasets_ui_bp.arguments(DashboardDatasetRenameRequestSchema)
@datasets_ui_bp.response(200, DashboardDatasetRenameResponseSchema)
def dashboard_dataset_rename(body: dict):
    """Set a custom display name for the currently loaded dataset."""
    new_name = body["name"].strip()
    if not new_name:
        abort(400, message="name is required")

    set_dataset_display_name(new_name)
    return {"success": True, "name": new_name}


@datasets_ui_bp.route("/api/dashboard/disk-usage")
@datasets_ui_bp.response(200, DashboardDiskUsageResponseSchema)
def dashboard_disk_usage():
    """Return free / used / total bytes for the partition holding ``DATA_DIR``."""
    probe = DATA_DIR if DATA_DIR.exists() else DATA_DIR.parent
    usage = shutil.disk_usage(str(probe))
    return {
        "total": usage.total,
        "used": usage.used,
        "free": usage.free,
        "path": str(probe),
    }
