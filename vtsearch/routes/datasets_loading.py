"""Dataset loading helpers: background threading, origin management, staging."""

from __future__ import annotations

import gc
import threading
from pathlib import Path
from uuid import uuid4

from vtsearch.config import DATA_DIR
from vtsearch.datasets import export_dataset_to_file
from vtsearch.datasets.registry import (
    get_saved_datasets_dir,
    register_dataset as _reg_register,
    set_loaded_id as _reg_set_loaded,
)
from vtsearch.utils import (
    build_diversity_tree,
    clear_all,
    collapse_duplicates,
    get_dataset_display_name,
    get_dupe_count,
    medias,
    snapshot_medias,
    update_progress,
)


def clear_dataset():
    """Clear the current dataset, votes, and all related state."""
    clear_all()


def _get_embedder_for_clips():
    """Return the embedder for the current dataset, or None."""
    snap = snapshot_medias()
    if not snap:
        return None
    first = next(iter(snap.values()))
    embedder_name = first.get("embedder", "")
    media_type = first.get("type", "audio")

    from vtsearch.media import embedders_for_type, get_embedder

    if embedder_name:
        try:
            return get_embedder(embedder_name)
        except KeyError:
            pass

    avail = embedders_for_type(media_type)
    return avail[0] if avail else None


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
    emb = _get_embedder_for_clips()
    if emb is None:
        update_progress("idle", "Ready")
        return
    emb.load_models()
    # Warm up the text encoder so the first text sort is instant.
    update_progress("loading", "Warming up text encoder…", 0, 0)
    try:
        emb.embed_text("warmup")
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
    """Convert an origin dict to a human-readable string.

    Delegates to the importer's :meth:`origin_display` when available,
    falling back to a generic ``"<importer_name>:<first_param>"`` format.
    """
    if not origin:
        return "unknown"
    importer_name = origin.get("importer", "")
    if not importer_name:
        return "unknown"

    from vtsearch.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is not None:
        return importer.origin_display(origin)

    # Unknown importer — generic fallback
    params = origin.get("params", {})
    if params:
        first_val = next(iter(params.values()))
        return f"{importer_name}:{first_val}"
    return importer_name


def _apply_clipper(clips_dict: dict, clipper_name: str) -> None:
    """Apply a clipper to all medias in *clips_dict*, replacing them in-place.

    Each media is run through the clipper's ``clip()`` method.  The resulting
    clips are assigned fresh sequential IDs and their origins are annotated
    with the clipper name and clip index.
    """
    if not clipper_name:
        return
    from vtsearch.media import get_clipper

    try:
        clipper = get_clipper(clipper_name)
    except KeyError:
        return

    all_clipped: list[dict] = []
    for media in list(clips_dict.values()):
        clipped = clipper.clip(media)
        for idx, clip in enumerate(clipped):
            # Annotate origin with clipper info
            orig = clip.get("origin")
            if isinstance(orig, dict):
                clip["origin"] = dict(orig)
                clip["origin"]["params"] = dict(clip["origin"].get("params", {}))
                clip["origin"]["params"]["clipper"] = clipper_name
                if len(clipped) > 1:
                    clip["origin"]["params"]["clip_index"] = str(idx)
            all_clipped.append(clip)

    clips_dict.clear()
    for new_id, clip in enumerate(all_clipped, 1):
        clip["id"] = new_id
        clips_dict[new_id] = clip


def _auto_register_dataset(
    name: str = "",
    origin_str: str = "unknown",
    source: dict | None = None,
    clipper: str = "",
) -> None:
    """Save the current medias as a pkl and register in the dataset registry.

    Called at the end of every successful dataset load.  Skips if medias is
    empty or if a registry entry with the same pkl path already exists (to
    avoid duplicating on reload).
    """
    snap = snapshot_medias()
    if not snap:
        return

    first = next(iter(snap.values()))
    media_type = first.get("type", "audio")
    num_items = len(snap)

    # Derive name from display-name override, origin, or fallback
    if not name:
        name = get_dataset_display_name() or origin_str or "Untitled"
        if ":" in name:
            name = name.split(":", 1)[1] or name

    # Save to a pkl file in saved_datasets/
    ds_dir = get_saved_datasets_dir()
    ds_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = str(ds_dir / f"ds_{uuid4().hex}.pkl")
    try:
        data_bytes = export_dataset_to_file(snap)
        Path(pkl_path).write_bytes(data_bytes)
        del data_bytes
    except Exception:
        return

    entry = _reg_register(
        name=name,
        media_type=media_type,
        num_items=num_items,
        num_dupes=get_dupe_count(),
        pkl_path=pkl_path,
        origin=origin_str,
        source=source,
        clipper=clipper,
    )
    _reg_set_loaded(entry["id"])


def _run_origin_load_in_background(load_fn, origin: dict, *, name: str = "", clipper: str = "") -> None:
    """Run a dataset load in a background thread with standard error handling.

    *load_fn* is called after ``clear_dataset()`` / ``gc.collect()`` and should
    populate ``medias``.  Everything after (origin tagging, clipping, dedup,
    diversity tree, registry, embedder warm-up) is handled automatically.

    Parameters
    ----------
    load_fn:
        Callable that loads data into *medias*.
    origin:
        Origin dict (``{"importer": ..., "params": ...}``).
    name:
        Display name for the dataset registry.  Falls back to
        ``_origin_to_str(origin)`` when empty.
    clipper:
        Name of the clipper to apply after loading.  Empty string means
        no clipping.
    """

    # Set progress to "loading" synchronously so the frontend never sees a
    # stale "idle" status from a previous operation before the thread starts.
    update_progress("loading", "Preparing dataset...")

    def task():
        try:
            clear_dataset()
            gc.collect()
            load_fn()
            # Suppress any premature "idle" that load_fn may have emitted
            # (e.g. load_demo_dataset signals idle before returning).
            # The frontend must not see "idle" until registration and
            # embedder warm-up are complete, otherwise the dashboard grid
            # renders before the new entry exists in the registry.
            update_progress("loading", "Finalizing…")
            _set_clip_origins(medias, origin)
            if clipper:
                _apply_clipper(medias, clipper)
            collapse_duplicates(medias)
            build_diversity_tree()
            origin_str = _origin_to_str(origin)
            _auto_register_dataset(
                name=name,
                origin_str=origin_str,
                source=origin,
                clipper=clipper,
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

    threading.Thread(target=task, daemon=True).start()


def _run_importer_in_background(importer, field_values: dict) -> None:
    """Start *importer*.run() in a daemon thread after clearing the dataset."""
    origin = importer.build_origin(field_values)
    clipper_name = field_values.pop("clipper", "")
    _run_origin_load_in_background(
        lambda: importer.run(field_values, medias),
        origin,
        name=importer.display_name,
        clipper=clipper_name,
    )


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
