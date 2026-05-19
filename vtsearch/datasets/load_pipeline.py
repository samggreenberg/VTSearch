"""Dataset loading helpers: background threading, origin management, staging."""

from __future__ import annotations

import gc
import time
import traceback
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from vtsearch.config import CoreConfig


class ConcurrencyGate:
    """A semaphore-like gate whose limit is read fresh on every acquisition.

    Unlike :class:`threading.Semaphore`, the cap is a callable evaluated at
    each ``acquire()`` attempt, so changes to the underlying setting take
    effect immediately for queued and future tasks (already-running tasks
    are never preempted).
    """

    def __init__(self, get_limit: Callable[[], int]) -> None:
        self._get_limit = get_limit
        self._cv = threading.Condition()
        self._active = 0

    def _limit(self) -> int:
        return max(1, int(self._get_limit()))

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        with self._cv:
            if not blocking:
                if self._active >= self._limit():
                    return False
                self._active += 1
                return True

            if timeout is None:
                while self._active >= self._limit():
                    self._cv.wait()
                self._active += 1
                return True

            deadline = time.monotonic() + timeout
            while self._active >= self._limit():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(timeout=remaining)
            self._active += 1
            return True

    def release(self) -> None:
        with self._cv:
            self._active -= 1
            self._cv.notify_all()

    @property
    def active(self) -> int:
        with self._cv:
            return self._active


# Two independent gates control how many dataset loads can run concurrently
# in each phase.  The download/import phase is bandwidth- and disk-bound;
# the embedding phase is CPU/GPU- and RAM-bound.  Splitting the gates lets
# one dataset download while another is still embedding, instead of forcing
# strict end-to-end serialisation.  Limits are user-configurable via the
# ``max_concurrent_dataset_downloads`` and ``max_concurrent_dataset_embeddings``
# settings; defaults derive from the host's CPU/GPU counts (see
# :func:`vtsearch.embedding.loader.default_concurrent_downloads` and
# :func:`vtsearch.embedding.loader.default_concurrent_embeddings`).
_download_gate = ConcurrencyGate(lambda: CoreConfig.from_settings().max_concurrent_dataset_downloads)
_embed_gate = ConcurrencyGate(lambda: CoreConfig.from_settings().max_concurrent_dataset_embeddings)


# ---------------------------------------------------------------------------
# App-side persistence hook
# ---------------------------------------------------------------------------
# The library remembers the user's per-media-type embedder pick by calling
# whatever the app installs here.  Default is a no-op so this module doesn't
# need to import ``vtsearch.settings`` (Phase 2 of
# ``docs/plans/extract-library.md``).  ``vtsearch/shim/`` registers the
# real implementation — ``vtsearch.settings.set_last_embedder_for_media_type``
# — at app startup.
_last_embedder_persistence_hook: Callable[[str, str], None] | None = None


def register_last_embedder_persistence_hook(fn: Callable[[str, str], None]) -> None:
    """Install the callback used to persist the user's per-media-type embedder pick.

    The Flask app installs ``vtsearch.settings.set_last_embedder_for_media_type``
    as the hook at startup so library callers don't have to know about the
    user-pref persistence layer.  Library-only callers can leave the default
    in place (no persistence).
    """
    global _last_embedder_persistence_hook
    _last_embedder_persistence_hook = fn

from vtsearch.auth import get_current_user
from vtsearch.config import DATA_DIR
from vtsearch.datasets import export_dataset_to_file
from vtsearch.datasets.loader import apply_custom_metadata_md5
from vtsearch.datasets.registry import (
    get_saved_datasets_dir,
    register_dataset as _reg_register,
    add_loaded_id as _reg_add_loaded,
)
from vtsearch.state import (
    DatasetContext,
    build_diversity_tree_for_context,
    clear_all,
    collapse_duplicates,
    register_context,
    snapshot_medias,
)
from vtsearch.concurrency.progress import update_progress
from vtsearch.concurrency.progress import (
    CancelledError,
    clear_thread_progress,
    dataset_progress,
    loading_tasks,
    set_thread_progress,
)


# ---------------------------------------------------------------------------
# Step-aware progress wrapper
# ---------------------------------------------------------------------------

# Maps the status strings emitted by inner functions to step numbers.
# "downloading" covers both download and extraction.
# "loading" covers model loading and pickle loading.
# "embedding" covers per-file embedding.
_STATUS_TO_STEP = {
    "downloading": 1,
    "loading": 2,
    "embedding": 3,
}
_TOTAL_LOAD_STEPS = 4  # download, load model, embed, finalize


def clear_dataset():
    """Clear the current dataset, votes, and all related state."""
    clear_all()


def _get_embedder_for_medias(media_dict: dict):
    """Resolve the embedder for *media_dict*.

    Imported lazily to avoid a circular dependency: this module sits under
    ``vtsearch.datasets`` but ``vtsearch.routes._shared`` lives in the
    routes layer, which itself imports from this module.
    """
    from vtsearch.routes._shared import get_embedder_for_medias as _impl

    return _impl(media_dict)


def _get_embedder_for_clips():
    """Return the embedder for the current dataset, or None."""
    return _get_embedder_for_medias(snapshot_medias())


def _load_embedder_with_progress(
    media_dict: dict | None,
    progress_fn,
    step: int | None = None,
    total_steps: int | None = None,
) -> None:
    """Eagerly load the embedder and warm up its text encoder.

    *progress_fn* is called as ``progress_fn(status, message, current, total, **kw)``
    to report progress.  When *media_dict* is passed it is used to determine
    the embedder; otherwise falls back to the active dataset.
    """
    if step is None:
        step = _TOTAL_LOAD_STEPS
    if total_steps is None:
        total_steps = _TOTAL_LOAD_STEPS

    emb = _get_embedder_for_medias(media_dict) if media_dict else _get_embedder_for_clips()
    if emb is None:
        progress_fn("idle", "Ready", step=None, total_steps=None)
        return

    def _model_load_progress(status, message, current, total):
        progress_fn(status, message, current, total, step=step, total_steps=total_steps)

    progress_fn("loading", "Loading embedding model…", 0, 0, step=step, total_steps=total_steps)
    original_cb = emb._on_progress
    emb._on_progress = _model_load_progress
    try:
        emb.load_models()
    finally:
        emb._on_progress = original_cb

    # Warm up the text encoder so the first text sort is instant.
    progress_fn("loading", "Warming up text encoder…", 0, 0, step=step, total_steps=total_steps)
    try:
        emb.embed_text("warmup")
    except Exception:
        pass
    progress_fn("idle", "Ready", step=None, total_steps=None)


def _load_embedder_for_clips(step: int | None = None, total_steps: int | None = None) -> None:
    """Load embedder using the global progress tracker."""
    _load_embedder_with_progress(None, update_progress, step=step, total_steps=total_steps)


def _origin_to_str(origin: dict | None) -> str:
    """Convert an origin dict to a human-readable string."""
    if not origin:
        return "unknown"
    importer_name = origin.get("importer", "")
    if not importer_name:
        return "unknown"

    from vtsearch.datasets.importers import get_importer

    importer = get_importer(importer_name)
    if importer is not None:
        return importer.origin_display(origin)

    params = origin.get("params", {})
    if params:
        first_val = next(iter(params.values()))
        return f"{importer_name}:{first_val}"
    return importer_name


def _normalize_media_type(value: str) -> str:
    """Normalize a media type string (folder_import_name or type_id) to a canonical type_id."""
    value = (value or "").strip()
    if not value:
        return ""
    try:
        from vtsearch.media import get_by_folder_name, normalize_type_id  # noqa: PLC0415

        try:
            return get_by_folder_name(value).type_id
        except KeyError:
            return normalize_type_id(value)
    except Exception:
        return value


def _apply_clipper(  # noqa: C901
    clips_dict: dict,
    clipper_name: str,
    clipper_params: dict | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Apply a clipper to all medias in *clips_dict*, replacing them in-place.

    After clipping, each clip gets:
    - A recomputed MD5 based on its actual content (so that dedup doesn't
      collapse distinct clips from the same parent).
    - Clip boundaries stored in ``origin["params"]`` (``clip_start``,
      ``clip_end``, ``clip_box``) so they survive label export/import.
    - A fresh embedding computed from the clipped content (audio/image/text)
      instead of inheriting the parent's embedding.

    Any importer-provided MD5 or embedding on the *parent* media is
    discarded for clips produced by a non-trivial clipper, since those
    values describe the full media item, not the sub-item.

    Args:
        on_progress: Optional callback ``(current, total, phase)`` invoked
            during clipping and re-embedding so callers can report progress.
    """
    if not clipper_name:
        return
    from vtsearch.media import get_clipper

    try:
        clipper = get_clipper(clipper_name)
    except KeyError:
        return

    if clipper_params:
        clipper = clipper.with_params(clipper_params)

    # Per-dataset resolution hook.  Default base implementation returns
    # self; reserved for clippers that need a dataset-level decision.
    # Auto routing now happens per-item via resolve_for_media() below.
    durations = [float(m.get("duration", 0) or 0) for m in clips_dict.values()]
    clipper = clipper.resolve_for_durations(durations)

    _base_keys = {"name", "display_name", "media_type", "parameters", "description", "creation_questions"}

    all_clipped: list[dict] = []
    # Track which clips need MD5/embedding recomputation.  Any clip that
    # came from a multi-output clipper call is a genuine sub-item whose
    # inherited MD5 and embedding describe the *parent*, not the clip.
    needs_recompute: list[bool] = []

    media_list = list(clips_dict.values())
    total_medias = len(media_list)
    media_type = clipper.media_type
    for media_idx, media in enumerate(media_list):
        if on_progress:
            on_progress(media_idx, total_medias, "clipping")
        # Per-media resolution.  Non-auto clippers return self; auto
        # clippers branch on the item's own duration so different items
        # can take different routes.  Each clip records the resolved
        # concrete clipper's name and parameters in its origin, so
        # cross-dataset replay is deterministic regardless of the
        # original auto policy.
        resolved = clipper.resolve_for_media(media)
        resolved_dict = resolved.to_dict()
        effective_params = {k: v for k, v in resolved_dict.items() if k not in _base_keys}
        clipped = resolved.clip(media)
        is_real_clip = len(clipped) > 1
        for idx, clip in enumerate(clipped):
            orig = clip.get("origin")
            if isinstance(orig, dict):
                clip["origin"] = dict(orig)
                clip["origin"]["params"] = dict(clip["origin"].get("params", {}))
                clip["origin"]["params"]["clipper"] = resolved.name
                # Store the clipper's effective parameter values so that
                # cross-dataset resolution can reconstruct the exact same
                # clipper configuration.
                for pk, pv in effective_params.items():
                    clip["origin"]["params"][f"clipper_{pk}"] = str(pv)
                if is_real_clip:
                    clip["origin"]["params"]["clip_index"] = str(idx)
                # Persist clip boundaries in origin so they survive label
                # export/import and can be used for cross-dataset resolution.
                if clip.get("clip_start") is not None:
                    clip["origin"]["params"]["clip_start"] = str(clip["clip_start"])
                if clip.get("clip_end") is not None:
                    clip["origin"]["params"]["clip_end"] = str(clip["clip_end"])
                if clip.get("clip_box") is not None:
                    clip["origin"]["params"]["clip_box"] = ",".join(str(v) for v in clip["clip_box"])
            all_clipped.append(clip)
            needs_recompute.append(is_real_clip)

    # Recompute MD5 and re-embed every genuine sub-item.
    _fixup_clip_md5_and_embeddings(all_clipped, needs_recompute, media_type, on_progress=on_progress)

    # Regenerate thumbnails so audio/video clips show their own range
    # rather than the parent's full waveform / mid-frame.
    _regenerate_clip_thumbnails(all_clipped, needs_recompute, media_type)

    clips_dict.clear()
    for new_id, clip in enumerate(all_clipped, 1):
        clip["id"] = new_id
        clips_dict[new_id] = clip


def _regenerate_clip_thumbnails(  # noqa: C901
    clips: list[dict],
    needs_recompute: list[bool],
    media_type: str,
) -> None:
    """Refresh ``thumbnail_bytes`` on clipped audio/video sub-items.

    Audio and video clippers copy ``thumbnail_bytes`` verbatim from the parent
    media; without this fixup, every sub-item would render the parent's
    waveform or middle-frame thumbnail in the find/label list.

    Image clips don't go through this path — their thumbnail is the cropped
    ``media_bytes`` itself, served directly by the media-image route.
    """
    if media_type == "audio":
        from vtsearch.media.audio.media_type import (
            generate_waveform_thumbnail,
            generate_waveform_thumbnail_from_file,
        )

        for clip, recompute in zip(clips, needs_recompute):
            if not recompute:
                continue
            wav = clip.get("media_bytes")
            thumb: bytes | None = None
            if wav is not None:
                thumb = generate_waveform_thumbnail(wav)
            else:
                path = clip.get("media_path")
                if path:
                    thumb = generate_waveform_thumbnail_from_file(Path(path))
            if thumb is not None:
                clip["thumbnail_bytes"] = thumb
        return

    if media_type == "video":
        from vtsearch.media.video.media_type import (
            generate_video_thumbnail_at,
            generate_video_thumbnail_from_file_at,
        )

        for clip, recompute in zip(clips, needs_recompute):
            if not recompute:
                continue
            t0 = clip.get("clip_start")
            t1 = clip.get("clip_end")
            if t0 is None or t1 is None:
                continue
            mid = (float(t0) + float(t1)) / 2.0
            video_bytes = clip.get("media_bytes")
            thumb: bytes | None = None
            if video_bytes is not None:
                thumb = generate_video_thumbnail_at(video_bytes, mid)
            else:
                path = clip.get("media_path")
                if path:
                    thumb = generate_video_thumbnail_from_file_at(Path(path), mid)
            if thumb is not None:
                clip["thumbnail_bytes"] = thumb


def _fixup_clip_md5_and_embeddings(  # noqa: C901
    clips: list[dict],
    needs_recompute: list[bool],
    media_type: str,
    on_progress: Callable[[int, int, str], None] | None = None,
) -> None:
    """Recompute MD5 and embeddings for clips that need it.

    A clip needs recomputation when:
    - ``needs_recompute`` is ``True`` (genuine sub-item from a multi-output
      clipper — the parent's MD5 and embedding are stale), **or**
    - the clip has no embedding at all (import phase was skipped because a
      clipper was going to re-embed anyway).

    Without the MD5 fix, all clips from the same parent would share the
    parent's MD5 (causing ``collapse_duplicates`` to merge them).

    Embeddings are batched through ``embed_media_bulk`` in a single call
    per invocation so GPU-backed embedders can fuse the forward pass.
    Failures fall back to keeping the clip's existing embedding.
    """
    import hashlib

    total_clips = len(clips)
    embed_indices: list[int] = []
    embed_inputs: list[dict] = []
    for clip_idx, (clip, recompute) in enumerate(zip(clips, needs_recompute)):
        # Also embed clips that have no embedding (e.g. when the import
        # phase skipped embedding because a clipper was specified).
        needs_embed = recompute or clip.get("embedding") is None
        if not needs_embed:
            continue

        content_bytes = _clip_content_bytes(clip, media_type)
        if content_bytes is None:
            if recompute:
                # Metadata-only clips (e.g. video): bytes unchanged but
                # boundaries differ — create a unique MD5 by hashing the
                # parent bytes + clip boundaries so dedup doesn't collapse
                # distinct clips.
                parent_bytes = clip.get("media_bytes", b"")
                boundary_tag = f"|clip_start={clip.get('clip_start')}|clip_end={clip.get('clip_end')}"
                combined = hashlib.md5(parent_bytes).hexdigest() + boundary_tag
                clip["md5"] = hashlib.md5(combined.encode()).hexdigest()
            continue

        if recompute:
            clip["md5"] = hashlib.md5(content_bytes).hexdigest()
        embed_indices.append(clip_idx)
        embed_inputs.append(_build_clip_embed_input(clip, media_type))

    if not embed_indices:
        return

    if on_progress:
        on_progress(0, total_clips, "embedding")

    embedder = _resolve_clip_embedder(media_type)
    if embedder is None:
        return

    def _clip_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        if on_progress:
            # Map the embedder's batch-level progress back to clip-list
            # coordinates so the existing on_progress(current, total, phase)
            # contract keeps reporting against total_clips.
            scaled = min(total_clips, int(current * len(embed_indices) / max(1, total)))
            on_progress(scaled, total_clips, "embedding")

    original_cb = embedder._on_progress
    embedder._on_progress = _clip_progress
    try:
        vectors = embedder.embed_media_bulk(embed_inputs)
    except Exception:
        import logging as _logging

        _logging.getLogger(__name__).exception(
            "Bulk clip re-embed failed for media_type=%s (%d clips)", media_type, len(embed_indices)
        )
        return
    finally:
        embedder._on_progress = original_cb

    for slot, vec in zip(embed_indices, vectors):
        if vec is not None:
            clips[slot]["embedding"] = vec

    if on_progress:
        on_progress(total_clips, total_clips, "embedding")


def _clip_content_bytes(clip: dict, media_type: str) -> bytes | None:
    """Return the embeddable content bytes for a clip.

    For media types where the clipper produces new bytes (audio, image) or
    a new string (text), return those bytes so the caller can hash and
    re-embed them.  For metadata-only clips (video), return ``None`` —
    the caller will use a boundary-based hash instead.
    """
    if media_type == "video":
        # Video clippers store boundaries as metadata without slicing
        # the underlying bytes, so there is nothing new to hash/embed.
        return None
    if clip.get("media_bytes") is not None and media_type != "text":
        return clip["media_bytes"]
    if clip.get("media_string") is not None and media_type == "text":
        return clip["media_string"].encode("utf-8")
    return None


def _build_clip_embed_input(clip: dict, media_type: str) -> dict:
    """Build the minimal media dict a bulk embedder needs for a clip.

    Hands the embedder the in-memory ``media_bytes`` (audio/image) or
    ``media_string`` (text) so the bulk surface never has to round-trip
    the content through a tempfile.  Preserves ``origin_name`` and
    ``filename`` so embedders that surface diagnostic paths still log
    something useful.
    """
    base: dict = {
        "origin_name": clip.get("origin_name", ""),
        "filename": clip.get("filename", ""),
    }
    if media_type == "text":
        base["media_string"] = clip.get("media_string", "")
    else:
        base["media_bytes"] = clip.get("media_bytes")
    return base


def _resolve_clip_embedder(media_type: str):
    """Pick the default embedder for *media_type* used by clip re-embed.

    Mirrors the legacy ``embed_file`` fallback chain: first registered
    embedder for the media type, or ``None`` if none are registered.
    """
    try:
        from vtsearch.media import embedders_for_type  # noqa: PLC0415
    except ImportError:
        return None
    avail = embedders_for_type(media_type)
    if not avail:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "clip re-embed: no embedders registered for media_type=%r — skipping bulk call",
            media_type,
        )
        return None
    return avail[0]


def _auto_register_dataset(
    media_dict: dict,
    name: str = "",
    origin_str: str = "unknown",
    source: dict | None = None,
    clipper: str = "",
    embedder: str = "",
    created_by: str = "",
    display_name: str | None = None,
    ingest_started_at: float | None = None,
) -> dict | None:
    """Save *media_dict* as a pkl and register in the dataset registry.

    Unlike the old version, this accepts an explicit *media_dict* instead
    of reading from the global ``medias`` proxy, enabling parallel loads.

    Returns the registry entry dict on success, or ``None`` on failure/skip.
    """
    if not media_dict:
        return None

    first = next(iter(media_dict.values()))
    media_type = first.get("type", "audio")
    num_items = len(media_dict)

    if not embedder:
        embedder = first.get("embedder", "")

    if not name:
        name = display_name or origin_str or "Untitled"
        if ":" in name:
            name = name.split(":", 1)[1] or name

    # Count dupes
    num_dupes = sum(
        1
        for m in media_dict.values()
        if isinstance(m.get("origin"), dict) and m["origin"].get("importer") == "dupe_set"
    )

    # Count file types by extension
    from collections import Counter

    ext_counter: Counter[str] = Counter()
    for m in media_dict.values():
        fn = m.get("filename", "")
        if fn and "." in fn:
            ext_counter[fn.rsplit(".", 1)[-1].lower()] += 1
        else:
            ext_counter["(no extension)"] += 1
    file_type_counts = dict(ext_counter.most_common())

    ds_dir = get_saved_datasets_dir()
    ds_dir.mkdir(parents=True, exist_ok=True)
    pkl_path = str(ds_dir / f"ds_{uuid4().hex}.pkl")
    try:
        data_bytes = export_dataset_to_file(media_dict)
        Path(pkl_path).write_bytes(data_bytes)
        del data_bytes
    except Exception:
        traceback.print_exc()
        return None

    entry = _reg_register(
        name=name,
        media_type=media_type,
        num_items=num_items,
        num_dupes=num_dupes,
        pkl_path=pkl_path,
        origin=origin_str,
        source=source,
        clipper=clipper,
        embedder=embedder,
        created_by=created_by,
        file_type_counts=file_type_counts,
        ingest_started_at=ingest_started_at,
    )
    _reg_add_loaded(entry["id"])
    return entry


# ---------------------------------------------------------------------------
# Parallel-safe background loading
# ---------------------------------------------------------------------------


def _run_origin_load_in_background(  # noqa: C901
    load_fn,
    origin: dict,
    *,
    name: str = "",
    clipper: str = "",
    clipper_params: dict | None = None,
    embedder: str = "",
    created_by: str = "",
    media_type: str = "",
) -> str:
    """Run a dataset load in a background thread with standard error handling.

    *load_fn* is called with a single argument — the target medias dict —
    and should populate it in-place.  Everything after (origin tagging,
    clipping, dedup, diversity tree, registry, embedder warm-up) is handled
    automatically.

    The dataset context is NOT activated during loading.  It is activated
    only upon successful completion, and only if no other dataset is
    currently active.

    Returns the task_id that can be used to poll progress or cancel.
    """

    # Reset the legacy cancellation flag so a previous cancel does not
    # immediately abort this new operation — but only when no other
    # parallel loads are running (otherwise we would clear cancellation
    # that might still be intended for those in-flight tasks).
    if not loading_tasks.has_active_tasks():
        dataset_progress.reset_cancel()

    # Remember the user's embedder pick per media type so the next dataset
    # importer modal can pre-select it even when no loaded dataset is
    # around to supply the same hint via ``guessedMediaEmbedder``.
    if media_type and embedder and _last_embedder_persistence_hook is not None:
        try:
            _last_embedder_persistence_hook(media_type, embedder)
        except Exception:
            pass

    import time as _time

    task_id = f"_loading_{uuid4().hex[:8]}"
    ingest_started_at = _time.time()
    tracker = loading_tasks.create_task(
        task_id, name or _origin_to_str(origin), media_type=media_type, embedder=embedder
    )

    # Set initial progress synchronously so the first poll sees it.
    tracker.update("loading", "Preparing dataset...", step=1, total_steps=_TOTAL_LOAD_STEPS)

    # Snapshot the user that triggered the load so background per-user
    # state (settings writes, settings_source sync) resolves correctly.
    _request_user = created_by or get_current_user()

    def task():  # noqa: C901
        from vtsearch.auth import set_thread_user

        set_thread_user(_request_user)
        ctx = DatasetContext(task_id)
        context_id = task_id  # tracks the current context key (may change after migration)

        # Tracks which gate (if any) we currently hold.  The download gate
        # covers the import/download phase; the embed gate covers all
        # CPU/GPU-bound embedding work (importer-side embedding plus the
        # post-load clipper, dedup, diversity-tree, and embedder warm-up).
        # We swap from download → embed when the importer first signals an
        # "embedding" status, freeing the download slot so another dataset
        # can start downloading in parallel.
        held_gate: dict[str, str | None] = {"name": None}

        def _acquire(gate: ConcurrencyGate, name: str, wait_msg: str) -> None:
            if gate.acquire(blocking=False):
                held_gate["name"] = name
                return
            tracker.update("loading", wait_msg, 0, 0, step=1, total_steps=_TOTAL_LOAD_STEPS)
            while not gate.acquire(timeout=0.5):
                tracker.check_cancelled()
            held_gate["name"] = name

        def _swap_to_embed() -> None:
            if held_gate["name"] == "embed":
                return
            if held_gate["name"] == "download":
                _download_gate.release()
                held_gate["name"] = None
            _acquire(_embed_gate, "embed", "Waiting for other datasets to finish embedding…")

        def stepped(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
            tracker.check_cancelled()
            if status == "idle":
                return
            # Importer transitioned into per-file embedding — swap gates so
            # another download can start while we do the GPU-bound work.
            if status == "embedding" and held_gate["name"] != "embed":
                _swap_to_embed()
            step = _STATUS_TO_STEP.get(status)
            tracker.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)

        try:
            # Acquire the download gate up front: every load starts with the
            # importer's download/import phase.  The task is already visible
            # in the UI, so the user sees the wait message while queued.
            _acquire(
                _download_gate,
                "download",
                "Waiting for other datasets to finish downloading…",
            )

            tracker.update("loading", "Preparing new dataset…", 0, 0, step=1, total_steps=_TOTAL_LOAD_STEPS)
            register_context(ctx)
            gc.collect()

            # Set thread-local progress so that loader/downloader callbacks
            # route to this task's tracker instead of the global singleton.
            set_thread_progress(stepped)
            try:
                import inspect

                sig = inspect.signature(load_fn)
                if sig.parameters:
                    load_fn(ctx.medias)
                else:
                    load_fn()
            finally:
                clear_thread_progress()

            tracker.check_cancelled()

            # Post-load steps (apply_md5, clipping, dedup, diversity tree,
            # registry save, embedder warm-up) are all CPU/GPU-bound and
            # touch embeddings — gate them on the embed semaphore.  This is
            # a no-op if the importer already swapped mid-load.
            _swap_to_embed()

            apply_custom_metadata_md5(ctx.medias)

            # Tag medias that don't already have an origin.
            for media in ctx.medias.values():
                if media.get("origin") is None:
                    media["origin"] = origin
                if not media.get("origin_name"):
                    media["origin_name"] = media.get("filename", "")

            if clipper:

                def _clipper_progress(current: int, total: int, phase: str) -> None:
                    tracker.check_cancelled()
                    if phase == "clipping":
                        msg = "Clipping media…"
                    else:
                        msg = "Embedding clips…"
                    tracker.update(
                        "loading",
                        msg,
                        current=current,
                        total=total,
                        step=_TOTAL_LOAD_STEPS,
                        total_steps=_TOTAL_LOAD_STEPS,
                    )

                _clipper_progress(0, 0, "clipping")
                _apply_clipper(ctx.medias, clipper, clipper_params, on_progress=_clipper_progress)

            def _dedup_progress(current: int, total: int) -> None:
                tracker.check_cancelled()
                tracker.update(
                    "loading",
                    "Removing duplicates…",
                    current=current,
                    total=total,
                    step=_TOTAL_LOAD_STEPS,
                    total_steps=_TOTAL_LOAD_STEPS,
                )

            _dedup_progress(0, 0)
            collapse_duplicates(ctx.medias, on_progress=_dedup_progress)

            def _diversity_progress(current: int, total: int) -> None:
                tracker.check_cancelled()
                tracker.update(
                    "loading",
                    "Building diversity index…",
                    current=current,
                    total=total,
                    step=_TOTAL_LOAD_STEPS,
                    total_steps=_TOTAL_LOAD_STEPS,
                )

            _diversity_progress(0, 0)
            build_diversity_tree_for_context(ctx, on_progress=_diversity_progress)
            tracker.check_cancelled()
            tracker.update("loading", "Saving to registry…", step=_TOTAL_LOAD_STEPS, total_steps=_TOTAL_LOAD_STEPS)
            origin_str = _origin_to_str(origin)
            entry = _auto_register_dataset(
                ctx.medias,
                name=name,
                origin_str=origin_str,
                source=origin,
                clipper=clipper,
                embedder=embedder,
                created_by=created_by,
                display_name=name,
                ingest_started_at=ingest_started_at,
            )

            # Migrate the context from the temp task_id to the real registry ID.
            if entry is not None:
                _migrate_context_id(task_id, entry["id"])
                context_id = entry["id"]
                ctx.dataset_display_name = entry.get("name", name)
                # Associate the loading task with the real dataset ID so the
                # frontend can show the embedder-warmup progress inline on the
                # dataset row instead of as an orphan loading task.
                loading_tasks.set_dataset_id(task_id, entry["id"])

            # Warm up the embedder.  Use a progress wrapper that updates the
            # task tracker, not the global singleton.
            def _task_progress(status, message="", current=0, total=0, **kw):
                tracker.update(status, message, current, total, **kw)

            _load_embedder_with_progress(ctx.medias, _task_progress)

            # Achievement: dataset load succeeded (demos/synthetic excluded).
            from vtsearch.achievements import record_dataset_load

            record_dataset_load(str(origin.get("importer", "")))
        except CancelledError:
            from vtsearch.state.core import unregister_context

            unregister_context(context_id)
            gc.collect()
            tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
        except ImportError as e:
            traceback.print_exc()
            from vtsearch.state.core import unregister_context

            unregister_context(context_id)
            gc.collect()
            tracker.update(
                "idle",
                "",
                0,
                0,
                error=(f"Missing dependency: {e}. Install all required packages with: pip install -e '.[cpu,dev]'"),
                step=None,
                total_steps=None,
            )
        except MemoryError:
            from vtsearch.state.core import unregister_context

            unregister_context(context_id)
            gc.collect()
            tracker.update(
                "idle",
                "",
                0,
                0,
                error="Out of memory — this dataset is too large. Try a smaller dataset or free up system RAM.",
                step=None,
                total_steps=None,
            )
        except Exception as e:
            traceback.print_exc()
            from vtsearch.state.core import unregister_context

            unregister_context(context_id)
            gc.collect()
            error_msg = str(e) or repr(e) or "Unknown error during dataset loading"
            tracker.update("idle", "", 0, 0, error=error_msg, step=None, total_steps=None)
        finally:
            if held_gate["name"] == "download":
                _download_gate.release()
            elif held_gate["name"] == "embed":
                _embed_gate.release()
            held_gate["name"] = None
            clear_thread_progress()
            loading_tasks.mark_finished(task_id)
            from vtsearch.auth import set_thread_user as _clear_thread_user

            _clear_thread_user(None)

    threading.Thread(target=task, daemon=True).start()
    return task_id


def _migrate_context_id(old_id: str, new_id: str) -> None:
    """Re-key a context from *old_id* to *new_id* in the store."""
    from vtsearch.state.core import _contexts, _state_lock

    with _state_lock:
        ctx = _contexts.get(old_id)
        if ctx is None:
            return
        ctx.dataset_id = new_id
        # Insert under the new key before removing the old, so the context
        # is never briefly invisible to concurrent lookups.
        _contexts[new_id] = ctx
        if old_id != new_id:
            _contexts.pop(old_id, None)


def consume_chunks_into(
    target: dict[int, dict[str, Any]],
    chunks: Iterable[dict[int, dict[str, Any]]],
) -> None:
    """Drain *chunks* into *target* with sequential IDs.

    Each chunk yielded by an importer's ``run_chunked()`` re-uses IDs
    starting at 1, so naive ``target.update(chunk)`` would overwrite
    earlier chunks.  Renumber every media to a unique ID continuing from
    whatever IDs are already present in *target*.
    """
    next_id = max(target.keys(), default=0) + 1
    for chunk in chunks:
        for media in chunk.values():
            media["id"] = next_id
            target[next_id] = media
            next_id += 1


_CHUNK_SIZE_BY_MEDIA_TYPE: dict[str, int] = {
    "text": 5000,
    "image": 500,
    "audio": 100,
    "video": 25,
    "document": 50,
}


def auto_chunk_size(media_type: str) -> int:
    """Pick a chunk size for *media_type* that bounds peak memory.

    Tuned roughly so a single in-flight chunk's raw bytes + embeddings stay
    below ~1 GB on typical inputs.  Returns a positive int.  Importers that
    do not support chunked loading silently ignore the value.
    """
    return _CHUNK_SIZE_BY_MEDIA_TYPE.get(_normalize_media_type(media_type), 100)


def _run_importer_in_background(importer, field_values: dict) -> str:
    """Start *importer*.run() in a daemon thread.

    When the importer reports ``supports_chunked``, the loader streams
    medias in via ``run_chunked`` to bound peak memory during the
    import/embedding phase.  The chunk size is auto-selected from the
    field's ``media_type`` (see :func:`auto_chunk_size`); there is no
    user-facing knob.

    Returns the task_id for progress tracking.
    """
    created_by = get_current_user()
    origin = importer.build_origin(field_values)
    clipper_name = field_values.pop("clipper", "") or ""
    clipper_params = field_values.pop("clipper_params", None)
    # Keep clipper in field_values for importers that need it (e.g. demo
    # importer writes a .clipper sidecar for readiness tracking).
    field_values["clipper"] = clipper_name
    embedder_name = field_values.get("embedder", "")

    # When a multi-output clipper is selected, skip embedding during the
    # import phase.  The clipper step will re-embed every clip anyway, so
    # computing parent embeddings up front is wasted work.  Default
    # clippers (pass-through, single output) still need the parent
    # embedding since it won't be recomputed.
    if clipper_name and not clipper_name.endswith("_default"):
        field_values["skip_embedding"] = True

    # Extract media_type from field_values so in-progress tasks can expose it
    # to the frontend (used for guessing the type in subsequent add dialogs).
    media_type_hint = _normalize_media_type(field_values.get("media_type", ""))

    use_chunked = getattr(importer, "supports_chunked", False)
    chunk_size = auto_chunk_size(media_type_hint) if use_chunked else 0

    def _load(target_medias):
        if use_chunked:
            consume_chunks_into(target_medias, importer.run_chunked(field_values, chunk_size))
        else:
            importer.run(field_values, target_medias)

    return _run_origin_load_in_background(
        _load,
        origin,
        name=importer.resolve_display_name(field_values),
        clipper=clipper_name,
        clipper_params=clipper_params,
        embedder=embedder_name,
        created_by=created_by,
        media_type=media_type_hint,
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

    _request_user = get_current_user()

    def stage_task():
        from vtsearch.auth import set_thread_user

        set_thread_user(_request_user)
        try:
            temp_medias: dict = {}
            importer.run(field_values, temp_medias)
            apply_custom_metadata_md5(temp_medias)

            if not temp_medias:
                update_progress("idle", "", 0, 0, error="Import produced no medias.")
                return

            first = next(iter(temp_medias.values()))
            media_type = first.get("type", "audio")
            count = len(temp_medias)
            name = label or importer.resolve_display_name(field_values)

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
        except ImportError as e:
            traceback.print_exc()
            gc.collect()
            update_progress(
                "idle",
                "",
                0,
                0,
                error=f"Missing dependency: {e}. Install all required packages with: pip install -e '.[cpu,dev]'",
            )
        except MemoryError:
            gc.collect()
            update_progress(
                "idle",
                "",
                0,
                0,
                error="Out of memory — this dataset is too large. Try a smaller dataset or free up system RAM.",
            )
        except Exception as e:
            traceback.print_exc()
            error_msg = str(e) or repr(e) or "Unknown error during staging"
            update_progress("idle", "", 0, 0, error=error_msg)
        finally:
            from vtsearch.auth import set_thread_user as _clear_thread_user

            _clear_thread_user(None)

    thread = threading.Thread(target=stage_task, daemon=True)
    thread.start()
