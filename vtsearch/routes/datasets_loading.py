"""Dataset loading helpers: background threading, origin management, staging."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from uuid import uuid4

from vtsearch.config import DATA_DIR
from vtsearch.datasets import export_dataset_to_file
from vtsearch.datasets.registry import (
    SAVED_DATASETS_DIR,
    register_dataset as _reg_register,
    set_loaded_id as _reg_set_loaded,
)
from vtsearch.utils import (
    build_diversity_tree,
    clear_all,
    collapse_duplicates,
    get_dataset_display_name,
    medias,
    update_progress,
)


def clear_dataset():
    """Clear the current dataset, votes, and all related state."""
    clear_all()


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


def _origin_to_str(origin: dict | None) -> str:
    """Convert an origin dict to a human-readable string."""
    if not origin:
        return "unknown"
    importer = origin.get("importer", "")
    params = origin.get("params", {})
    if importer == "demo":
        return f"demo:{params.get('name', '')}"
    elif importer == "pickle":
        return f"file:{params.get('filename', '')}"
    elif importer == "folder":
        return f"folder:{params.get('path', '')}"
    elif importer:
        return importer
    return "unknown"


def _auto_register_dataset(name: str = "", origin_str: str = "unknown", source: dict | None = None) -> None:
    """Save the current medias as a pkl and register in the dataset registry.

    Called at the end of every successful dataset load.  Skips if medias is
    empty or if a registry entry with the same pkl path already exists (to
    avoid duplicating on reload).
    """
    if not medias:
        return

    first = next(iter(medias.values()))
    media_type = first.get("type", "audio")
    num_items = len(medias)

    # Derive name from display-name override, origin, or fallback
    if not name:
        name = get_dataset_display_name() or origin_str or "Untitled"
        if ":" in name:
            name = name.split(":", 1)[1] or name

    # Save to a pkl file in saved_datasets/
    SAVED_DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path = str(SAVED_DATASETS_DIR / f"ds_{uuid4().hex}.pkl")
    try:
        data_bytes = export_dataset_to_file(medias)
        Path(pkl_path).write_bytes(data_bytes)
        del data_bytes
    except Exception:
        return

    entry = _reg_register(
        name=name,
        media_type=media_type,
        num_items=num_items,
        pkl_path=pkl_path,
        origin=origin_str,
        source=source,
    )
    _reg_set_loaded(entry["id"])


def _run_importer_in_background(importer, field_values: dict) -> None:
    """Start *importer*.run() in a daemon thread after clearing the dataset."""

    def load_task():
        try:
            clear_dataset()
            gc.collect()
            importer.run(field_values, medias)
            origin = importer.build_origin(field_values)
            _set_clip_origins(medias, origin)
            collapse_duplicates(medias)
            build_diversity_tree()
            # Auto-register in the dataset registry before signalling idle
            # so the frontend sees the new entry when it refreshes the grid.
            origin_str = _origin_to_str(origin)
            _auto_register_dataset(
                name=importer.display_name,
                origin_str=origin_str,
                source=origin,
            )
            _load_embedder_for_clips()
        except MemoryError:
            medias.clear()
            gc.collect()
            update_progress(
                "idle",
                "",
                0,
                0,
                "Out of memory — this dataset is too large. Try a smaller dataset or free up system RAM.",
            )
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    # Signal "loading" before the thread starts so frontend polling never
    # sees a stale "idle" from a previous load and prematurely stops.
    update_progress("loading", "Preparing to load dataset…", 0, 0)

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
                "idle",
                f"Staged: {name} ({count} medias)",
                100,
                100,
                staging_result={"path": str(staging_path), "name": name, "count": count, "media_type": media_type},
            )
        except MemoryError:
            gc.collect()
            update_progress(
                "idle",
                "",
                0,
                0,
                "Out of memory — this dataset is too large. Try a smaller dataset or free up system RAM.",
            )
        except Exception as e:
            update_progress("idle", "", 0, 0, str(e))

    thread = threading.Thread(target=stage_task, daemon=True)
    thread.start()
