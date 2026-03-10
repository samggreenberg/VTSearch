"""Dashboard, demo dataset listing, and other UI helper routes."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from vtsearch.config import EMBEDDINGS_DIR
from vtsearch.datasets import DEMO_DATASETS
from vtsearch.datasets.loader import read_pkl_embedder
from vtsearch.routes.datasets_loading import _origin_to_str
from vtsearch.routes.helpers import get_json_or_400
from vtsearch.utils import (
    get_dataset_display_name,
    get_dupe_count,
    set_dataset_display_name,
    snapshot_medias,
)

datasets_ui_bp = Blueprint("datasets_ui", __name__)


# ---------------------------------------------------------------------------
# Demo dataset listing
# ---------------------------------------------------------------------------


def _folder_has_content(folder) -> bool:
    """Return True if *folder* exists and contains at least one entry."""
    return folder is not None and folder.exists() and any(folder.iterdir())


@datasets_ui_bp.route("/api/dataset/demo-list")
def demo_dataset_list():
    """List available demo datasets.

    Each dataset has a ``status`` field with one of three values:

    * ``"ready"`` – embeddings are cached and source data is present.
    * ``"needs_embedding"`` – source data is downloaded but not yet embedded.
    * ``"needs_download"`` – source data must be downloaded (and then embedded).
    """
    # Only include demo datasets whose media type is currently registered.
    from vtsearch.converters import list_converters_for_source
    from vtsearch.media import get as media_get

    # Optional embedder filter: when the caller specifies an embedder, a cached
    # pkl is only considered "ready" if it was produced by that same embedder.
    requested_embedder = request.args.get("embedder", "").strip()

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

        # Converters that consume this demo's media type (M→N converters).
        available_converters = [c.to_dict() for c in list_converters_for_source(media_type)]

        # Resolve pkl_embedder if not already read above.
        if pkl_embedder is None and has_pkl:
            pkl_embedder = read_pkl_embedder(pkl_file)

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
            }
        )
    return jsonify({"datasets": demos})


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------


@datasets_ui_bp.route("/api/dashboard/dataset-info")
def dashboard_dataset_info():
    """Return metadata about the currently loaded dataset for the dashboard.

    Returns a JSON object with ``name``, ``num_medias``, ``media_type``, and
    ``origin`` extracted from the first media that has origin info.
    """
    snap = snapshot_medias()
    if not snap:
        return jsonify({"error": "No dataset loaded"}), 404

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

    return jsonify(
        {
            "name": name,
            "num_medias": num_medias,
            "num_dupes": get_dupe_count(),
            "media_type": media_type,
            "origin": origin or "unknown",
            "source": source,
        }
    )


@datasets_ui_bp.route("/api/dashboard/dataset-rename", methods=["PUT"])
def dashboard_dataset_rename():
    """Set a custom display name for the currently loaded dataset."""
    data = get_json_or_400()
    if not isinstance(data, dict):
        return data

    new_name = data.get("name", "").strip()
    if not new_name:
        return jsonify({"error": "name is required"}), 400

    set_dataset_display_name(new_name)
    return jsonify({"success": True, "name": new_name})
