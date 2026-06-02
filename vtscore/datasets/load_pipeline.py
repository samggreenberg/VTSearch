"""Dataset loading helpers: background threading, origin management, staging."""

from __future__ import annotations

import gc
import logging
import time
import traceback
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from vtscore.config import CoreConfig


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
# :func:`vtscore.embedding.loader.default_concurrent_downloads` and
# :func:`vtscore.embedding.loader.default_concurrent_embeddings`).
_download_gate = ConcurrencyGate(lambda: CoreConfig.from_settings().max_concurrent_dataset_downloads)
_embed_gate = ConcurrencyGate(lambda: CoreConfig.from_settings().max_concurrent_dataset_embeddings)


# ---------------------------------------------------------------------------
# App-side persistence hook
# ---------------------------------------------------------------------------
# The library remembers the user's per-media-type embedder pick by calling
# whatever the app installs here.  Default is a no-op so this module doesn't
# need to import ``vtsearch.settings`` (Phase 2 of
# ``../docs/architecture.md``).  ``vtsearch/shim/`` registers the
# real implementation; ``vtsearch.settings.set_last_embedder_for_media_type``
# is wired at app startup.
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
from vtscore.config import DATA_DIR
from vtscore.datasets import export_dataset_to_file
from vtscore.datasets.loader import apply_custom_metadata_md5
from vtscore.datasets.registry import (
    get_saved_datasets_dir,
    register_dataset as _reg_register,
    add_loaded_id as _reg_add_loaded,
    unregister_dataset as _reg_unregister,
)
from vtsearch.state import (
    DatasetContext,
    build_diversity_tree_for_context,
    clear_all,
    collapse_duplicates,
    register_context,
)
from vtscore.embedding.matrix import invalidate_embedding_matrix
from vtscore.concurrency.progress import update_progress
from vtscore.concurrency.progress import (
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
    ``vtscore.datasets`` but ``vtsearch.routes._shared`` lives in the
    routes layer, which itself imports from this module.
    """
    from vtsearch.routes._shared import get_embedder_for_medias as _impl

    return _impl(media_dict)


def _origin_to_str(origin: dict | None) -> str:
    """Convert an origin dict to a human-readable string."""
    if not origin:
        return "unknown"
    importer_name = origin.get("importer", "")
    if not importer_name:
        return "unknown"

    from vtscore.datasets.importers import get_importer

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
        from vtscore.media import get_by_folder_name, normalize_type_id  # noqa: PLC0415

        try:
            return get_by_folder_name(value).type_id
        except KeyError:
            return normalize_type_id(value)
    except Exception:
        return value


def _parse_chain_field(raw: Any) -> list[dict] | None:
    """Decode a ``clipper_chain`` importer field value into a step list.

    The field may arrive as a JSON string (typical client encoding) or as
    a native list (programmatic callers). Returns ``None`` for missing /
    malformed values so the legacy single-clipper path stays in effect.
    """
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        import json as _json

        try:
            decoded = _json.loads(raw)
        except (TypeError, ValueError):
            return None
        if isinstance(decoded, list):
            return decoded
        return None
    return None


def _apply_clipper(  # noqa: C901
    clips_dict: dict,
    clipper_name: str,
    clipper_params: dict | None = None,
    on_progress: Callable[[int, int, str], None] | None = None,
    chain_steps: list[dict] | None = None,
) -> None:
    """Apply a clipper (or full chain) to all medias in *clips_dict*, in place.

    Either *clipper_name* (single-step legacy path) or *chain_steps*
    (ordered list of converter/clipper steps, see
    ``docs/plans/clipper-chain.md``) may be supplied.  When both are
    given, *chain_steps* wins; the single name/params get folded into
    the chain by callers that want both encodings on the same clip.

    After running the chain, each clip gets:
    - A recomputed MD5 based on its actual content (so that dedup doesn't
      collapse distinct clips from the same parent).
    - The full chain trail in ``origin.params['clipper_chain']`` plus
      legacy single-clipper keys (``clipper``, ``clipper_<param>``,
      ``clip_start``/``clip_end``/``clip_box``/``clip_index``) describing
      the last clipper step.
    - A fresh embedding computed from the clipped content (audio/image/text)
      instead of inheriting the parent's embedding.

    Any importer-provided MD5 or embedding on the *parent* media is
    discarded for clips produced by a non-trivial chain, since those
    values describe the full media item, not the sub-item.

    Args:
        on_progress: Optional callback ``(current, total, phase)`` invoked
            during clipping and re-embedding so callers can report progress.
    """
    from vtscore.datasets.clipper_chain import apply_chain_to_clips, normalise_chain

    # Resolve the effective step list. A non-empty `chain_steps` takes
    # precedence; otherwise build a length-1 chain from the legacy
    # single-clipper args (if any).
    steps = normalise_chain(chain_steps)
    legacy_default = not steps and not chain_steps and clipper_name and clipper_name.endswith("_default")
    if legacy_default:
        # Legacy fast path: a single ``*_default`` clipper is a no-op on
        # the data but stamps ``clipper`` / ``clipper_<key>`` in every
        # origin. We don't put it in the chain (so ``clipper_chain``
        # isn't written for default-only loads) but we do preserve the
        # legacy stamp so existing readers continue to see it.
        from vtscore.media import get_clipper  # noqa: PLC0415

        try:
            clipper = get_clipper(clipper_name)
        except KeyError:
            return
        if clipper_params:
            clipper = clipper.with_params(clipper_params)
        resolved_dict = clipper.to_dict()
        base_keys = {"name", "display_name", "media_type", "parameters", "description", "creation_questions"}
        effective_params = {k: v for k, v in resolved_dict.items() if k not in base_keys}
        for media in clips_dict.values():
            orig = media.get("origin")
            if isinstance(orig, dict):
                media["origin"] = dict(orig)
                media["origin"]["params"] = dict(orig.get("params", {}))
                media["origin"]["params"]["clipper"] = clipper.name
                for pk, pv in effective_params.items():
                    media["origin"]["params"][f"clipper_{pk}"] = str(pv)
        return

    if not steps and clipper_name:
        # Resolve the legacy single-clipper args into a length-1 chain.
        # Catch unknown names here so we match the legacy no-op semantics.
        from vtscore.media import get_clipper  # noqa: PLC0415

        try:
            get_clipper(clipper_name)
        except KeyError:
            return
        steps = [{"kind": "clipper", "name": clipper_name, "params": dict(clipper_params or {})}]
    if not steps:
        return

    result = apply_chain_to_clips(clips_dict, steps, on_progress=on_progress)
    if result is None:
        return
    final_type, needs_recompute = result

    clips_list = list(clips_dict.values())
    _fixup_clip_md5_and_embeddings(clips_list, needs_recompute, final_type, on_progress=on_progress)
    _regenerate_clip_thumbnails(clips_list, needs_recompute, final_type)

    # Update `media_type` on every clip so downstream code sees the final type
    # (converter chains change the media_type of the carriers).
    for clip in clips_dict.values():
        clip["media_type"] = final_type


def _regenerate_clip_thumbnails(  # noqa: C901
    clips: list[dict],
    needs_recompute: list[bool],
    media_type: str,
) -> None:
    """Refresh ``thumbnail_bytes`` on clipped audio/video sub-items.

    Audio and video clippers copy ``thumbnail_bytes`` verbatim from the parent
    media; without this fixup, every sub-item would render the parent's
    waveform or middle-frame thumbnail in the find/label list.

    Image clips don't go through this path; their thumbnail is the cropped
    ``media_bytes`` itself, served directly by the media-image route.
    """
    if media_type == "audio":
        from vtscore.media.audio.media_type import (
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
        from vtscore.media.video.media_type import (
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
      clipper (the parent's MD5 and embedding are stale), **or**
    - the clip has no embedding at all (import phase was skipped because a
      clipper was going to re-embed anyway).

    For any clip reaching this fixup with new content bytes (audio/image/
    text), the MD5 is always rehashed from those final bytes (including
    single-output clippers that copy the parent dict via ``dict(media)``
    and would otherwise carry the parent's stale MD5 forward.

    Without the MD5 fix, clips from the same parent would share the
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
        metadata_only = content_bytes is None and media_type == "video"
        if content_bytes is None and not metadata_only:
            continue

        # Always recompute MD5 from the final clip bytes whenever we're
        # also recomputing the embedding. Any single-output clipper that
        # rewrites media_bytes (e.g. ImageObjectClipper with one detection,
        # ImageBboxClipper) would otherwise inherit the parent's MD5 via
        # ``dict(media)`` and cause dedup to merge distinct crops.
        if content_bytes is not None:
            clip["md5"] = hashlib.md5(content_bytes).hexdigest()
        elif recompute:
            # Metadata-only clips (e.g. video): bytes unchanged but
            # boundaries differ; create a unique MD5 by hashing the
            # parent bytes + clip boundaries so dedup doesn't collapse
            # distinct clips.
            parent_bytes = clip.get("media_bytes", b"")
            boundary_tag = f"|clip_start={clip.get('clip_start')}|clip_end={clip.get('clip_end')}"
            combined = hashlib.md5(parent_bytes).hexdigest() + boundary_tag
            clip["md5"] = hashlib.md5(combined.encode()).hexdigest()
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
    re-embed them.  For metadata-only clips (video), return ``None``;
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

    Hands the embedder the in-memory ``media_bytes`` (audio/image/video) or
    ``media_string`` (text) so the bulk surface never has to round-trip
    the content through a tempfile.  Preserves ``origin_name`` and
    ``filename`` so embedders that surface diagnostic paths still log
    something useful.

    Video clips additionally carry ``clip_start`` / ``clip_end`` (and
    ``media_path`` when available) because the underlying parent bytes
    are shared across every tile; the embedder uses the boundary
    metadata to sample distinct frame ranges per tile.
    """
    base: dict = {
        "origin_name": clip.get("origin_name", ""),
        "filename": clip.get("filename", ""),
    }
    if media_type == "text":
        base["media_string"] = clip.get("media_string", "")
    else:
        base["media_bytes"] = clip.get("media_bytes")
    if media_type == "video":
        if clip.get("media_path"):
            base["media_path"] = clip["media_path"]
        if "clip_start" in clip:
            base["clip_start"] = clip["clip_start"]
        if "clip_end" in clip:
            base["clip_end"] = clip["clip_end"]
    return base


def _resolve_clip_embedder(media_type: str):
    """Pick the default embedder for *media_type* used by clip re-embed.

    Mirrors the legacy ``embed_file`` fallback chain: first registered
    embedder for the media type, or ``None`` if none are registered.
    """
    try:
        from vtscore.media import embedders_for_type  # noqa: PLC0415
    except ImportError:
        return None
    avail = embedders_for_type(media_type)
    if not avail:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "clip re-embed: no embedders registered for media_type=%r; skipping bulk call",
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
    media_type = first.get("media_type", "audio")
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
    import time as _time

    from vtscore.config import CoreConfig

    now = _time.time()
    try:
        config = CoreConfig.from_settings()
        max_age = config.dataset_max_age_days
    except RuntimeError:
        max_age = None
    expires_at = now + max_age * 86400 if max_age is not None else None

    try:
        data_bytes = export_dataset_to_file(
            media_dict,
            embedder=embedder,
            clipper=clipper,
            media_type=media_type,
            name=name,
            created_at=now,
            expires_at=expires_at,
        )
        Path(pkl_path).write_bytes(data_bytes)
        del data_bytes
    except Exception:
        traceback.print_exc()
        return None

    try:
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
            expires_at=expires_at,
        )
    except Exception:
        # Registry write failed; clean up the orphaned pkl so we don't
        # leave a stale file behind with nothing pointing at it.
        traceback.print_exc()
        Path(pkl_path).unlink(missing_ok=True)
        return None
    _reg_add_loaded(entry["id"])
    return entry


# ---------------------------------------------------------------------------
# Parallel-safe background loading
# ---------------------------------------------------------------------------


class _LoadGateController:
    """Tracks which load-pipeline gate (download / embed) is currently held.

    Splits gate-acquisition concerns out of the task body: the importer
    runs under the download gate (bandwidth-bound), and we swap to the
    embed gate as soon as the importer signals it's started embedding so
    another dataset can begin downloading in parallel.
    """

    def __init__(self, tracker) -> None:
        self._tracker = tracker
        self._held: str | None = None

    @property
    def held(self) -> str | None:
        return self._held

    def acquire(self, gate: ConcurrencyGate, name: str, wait_msg: str) -> None:
        if gate.acquire(blocking=False):
            self._held = name
            return
        self._tracker.update("loading", wait_msg, 0, 0, step=1, total_steps=_TOTAL_LOAD_STEPS)
        while not gate.acquire(timeout=0.5):
            self._tracker.check_cancelled()
        self._held = name

    def acquire_download(self) -> None:
        self.acquire(_download_gate, "download", "Waiting for other datasets to finish downloading…")

    def swap_to_embed(self) -> None:
        if self._held == "embed":
            return
        if self._held == "download":
            _download_gate.release()
            self._held = None
        self.acquire(_embed_gate, "embed", "Waiting for other datasets to finish embedding…")

    def release(self) -> None:
        if self._held == "download":
            _download_gate.release()
        elif self._held == "embed":
            _embed_gate.release()
        self._held = None


def _make_stepped_progress(controller: _LoadGateController, tracker):
    """Build the importer-side progress callback.

    Routes status updates into *tracker* with the right step number, and
    triggers the download→embed gate swap on the first ``"embedding"``
    status so a queued download can start.
    """

    def stepped(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        tracker.check_cancelled()
        if status == "idle":
            return
        if status == "embedding" and controller.held != "embed":
            controller.swap_to_embed()
        step = _STATUS_TO_STEP.get(status)
        tracker.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)

    return stepped


def _run_importer(load_fn, ctx: DatasetContext, stepped) -> None:
    """Invoke *load_fn* under thread-local progress, populating ctx.medias."""
    import inspect  # noqa: PLC0415

    set_thread_progress(stepped)
    try:
        sig = inspect.signature(load_fn)
        if sig.parameters:
            load_fn(ctx.medias)
        else:
            load_fn()
    finally:
        clear_thread_progress()


def _tag_origins(media_dict: dict, origin: dict) -> None:
    """Stamp *origin* onto medias that don't already carry one.

    Each media gets its own fresh copy of the origin dict (including a
    fresh ``params``).  Sharing one dict by reference across siblings
    means any later mutation of ``media["origin"]["params"]`` on one
    media silently corrupts every other media stamped by the same load;
    and that aliasing also survives pickle round-trips via backreferences.
    """
    for media in media_dict.values():
        if media.get("origin") is None:
            media["origin"] = {
                "importer": origin.get("importer", ""),
                "params": dict(origin.get("params", {})),
            }
        if not media.get("origin_name"):
            media["origin_name"] = media.get("filename", "")


def _apply_clipper_stage(
    ctx: DatasetContext, tracker, clipper: str, clipper_params: dict | None, chain_steps: list[dict] | None
) -> None:
    """Run the clipper / chain stage with tracker-routed progress."""
    if not (clipper or chain_steps):
        return

    def _clipper_progress(current: int, total: int, phase: str) -> None:
        tracker.check_cancelled()
        if phase == "clipping":
            msg = "Clipping media…"
        elif phase == "converting":
            msg = "Converting media…"
        else:
            msg = "Embedding clips…"
        tracker.update(
            "loading", msg, current=current, total=total, step=_TOTAL_LOAD_STEPS, total_steps=_TOTAL_LOAD_STEPS
        )

    _clipper_progress(0, 0, "clipping")
    _apply_clipper(ctx.medias, clipper, clipper_params, on_progress=_clipper_progress, chain_steps=chain_steps)
    # Clipping reassigns media IDs starting from 1 and rewrites embeddings
    # in place; the cached matrix's id list can collide with the new one
    # while pointing at stale rows. Drop the cache so the next reader
    # rebuilds against the live medias dict.
    invalidate_embedding_matrix(ctx)


def embed_missing(  # noqa: C901
    medias: dict[int, dict[str, Any]],
    embedder_name: str = "",
    on_progress: Callable[[str, str, int, int], None] | None = None,
) -> None:
    """Embed media items in *medias* that don't already have an embedding.

    Importers are not responsible for calling the embedder; they emit
    media dicts and optionally pre-populate ``embedding`` from
    ``content_vectors`` / ``custom_metadata_map``.  Items still at
    ``None`` after the importer finishes go through this function, which
    resolves a single embedder (the user's pick when given, otherwise the
    default for the media type of the unembedded items) and bulk-embeds
    them in one ``embed_media_bulk`` call.

    Items whose bulk-embed call returns ``None`` stay at ``None``; the
    load pipeline drops them via :func:`_drop_none_embeddings_stage`.
    Patch-region tensors are also attached here for embedders that
    report ``supports_patch_regions``.
    """
    missing = [(mid, m) for mid, m in medias.items() if m.get("embedding") is None]
    if not missing:
        return

    media_type = ""
    for _, m in missing:
        mt = m.get("media_type")
        if mt:
            media_type = mt
            break
    if not media_type:
        return

    from vtscore.media import embedders_for_type, get_embedder  # noqa: PLC0415

    emb = None
    if embedder_name:
        try:
            emb = get_embedder(embedder_name)
        except KeyError:
            emb = None
    if emb is None:
        avail = embedders_for_type(media_type)
        emb = avail[0] if avail else None
    if emb is None:
        return

    if on_progress is None:

        def _noop_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
            return None

        on_progress = _noop_progress

    if getattr(emb, "_model", None) is None:
        on_progress("loading", "Loading embedding model…", 0, 0)
        emb.load_models()

    total = len(missing)
    on_progress("embedding", f"Embedding {total} item(s)…", 0, total)

    inputs = [m for _, m in missing]
    original_cb = emb._on_progress
    emb._on_progress = on_progress
    try:
        vectors = emb.embed_media_bulk(inputs)
    except Exception:
        logging.getLogger(__name__).exception("Bulk embed failed for media_type=%s (%d items)", media_type, total)
        return
    finally:
        emb._on_progress = original_cb

    embedder_id = emb.name
    for (mid, _), vec in zip(missing, vectors):
        if vec is None:
            continue
        media = medias.get(mid)
        if media is None:
            continue
        media["embedding"] = vec
        if not media.get("embedder"):
            media["embedder"] = embedder_id

    # Patch-region pass for embedders that support it (DINOv2/v3/EUPE).
    if getattr(emb, "supports_patch_regions", False) is not True:
        return

    patch_inputs = [m for _, m in missing if m.get("patch_regions") is None]
    if not patch_inputs:
        return

    emb._on_progress = on_progress
    try:
        outputs = emb.patch_forward_bulk(patch_inputs)
    except Exception:
        logging.getLogger(__name__).exception(
            "Bulk patch-forward failed for media_type=%s (%d items)", media_type, len(patch_inputs)
        )
        outputs = [None] * len(patch_inputs)
    finally:
        emb._on_progress = original_cb

    for media, patch_out in zip(patch_inputs, outputs):
        if patch_out is None:
            continue
        _attach_patch_regions_to_media(media, patch_out)


def _embed_missing_stage(
    ctx: DatasetContext,
    tracker,
    embedder_name: str,
) -> None:
    """Pipeline wrapper around :func:`embed_missing` with tracker-routed progress."""

    def _emb_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        tracker.check_cancelled()
        step = _STATUS_TO_STEP.get(status, _STATUS_TO_STEP["embedding"])
        tracker.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)

    embed_missing(ctx.medias, embedder_name, on_progress=_emb_progress)
    invalidate_embedding_matrix(ctx)


def _attach_patch_regions_to_media(media: dict, patch_out) -> None:
    """Attach HAC patch-region tree to *media* (mirrors ``loader_folder._attach_patch_regions``)."""
    import numpy as np  # noqa: PLC0415

    from vtscore.media.patch_embed import build_region_tree, to_fp16  # noqa: PLC0415

    regions = build_region_tree(patch_out, k=12, alpha=0.5)
    media["patch_regions"] = to_fp16(regions)
    media["patch_grid"] = patch_out.patch_grid.astype(np.float16, copy=False)


def _drop_none_embeddings_stage(ctx: DatasetContext, tracker) -> None:
    """Drop any media that finished the clipper stage without an embedding.

    ``_fixup_clip_md5_and_embeddings`` is best-effort: when its bulk
    re-embed call fails (no embedder, vector ``None``, exception) the
    clip is left in ``ctx.medias`` with ``embedding=None``.  Letting
    those through poisons every downstream consumer; the matrix builder
    in ``vtscore/embedding/matrix.py`` raises (M11 fix); sort/score
    aggregations get wrong-length lists.  Drop them here so the rest of
    the load pipeline (dedup, diversity tree, registry) sees a clean
    dict, and surface the count to the progress tracker so the user
    knows N is lower than the importer reported.
    """
    none_ids = [cid for cid, media in ctx.medias.items() if media.get("embedding") is None]
    if not none_ids:
        return

    for cid in none_ids:
        del ctx.medias[cid]

    import logging  # noqa: PLC0415

    logging.getLogger(__name__).warning(
        "Dropped %d media item(s) with embedding=None (importer or re-embed step failed)",
        len(none_ids),
    )
    tracker.update(
        "loading",
        f"Dropped {len(none_ids)} item(s) with failed embedding…",
        current=0,
        total=0,
        step=_TOTAL_LOAD_STEPS,
        total_steps=_TOTAL_LOAD_STEPS,
    )
    invalidate_embedding_matrix(ctx)


def _collapse_duplicates_stage(ctx: DatasetContext, tracker) -> None:
    def _progress(current: int, total: int) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            "Removing duplicates…",
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _progress(0, 0)
    collapse_duplicates(ctx.medias, on_progress=_progress)
    invalidate_embedding_matrix(ctx)


def _build_diversity_tree_stage(ctx: DatasetContext, tracker) -> None:
    def _progress(current: int, total: int) -> None:
        tracker.check_cancelled()
        tracker.update(
            "loading",
            "Building diversity index…",
            current=current,
            total=total,
            step=_TOTAL_LOAD_STEPS,
            total_steps=_TOTAL_LOAD_STEPS,
        )

    _progress(0, 0)
    build_diversity_tree_for_context(ctx, on_progress=_progress)


def _register_and_migrate(
    ctx: DatasetContext,
    tracker,
    task_id: str,
    origin: dict,
    name: str,
    clipper: str,
    embedder: str,
    created_by: str,
    ingest_started_at: float,
) -> tuple[str, str | None]:
    """Save to registry, migrate the context from task_id to its real id.

    Returns ``(context_id, registry_entry_id)``; *context_id* is the
    (possibly migrated) context id, and *registry_entry_id* is the id of
    the newly created registry entry (or ``None`` if registration was
    skipped).  Callers should retain *registry_entry_id* so a later
    failure in the surrounding pipeline can roll the entry back.

    If the registry entry is created successfully but the subsequent
    in-memory migration steps raise, the entry is rolled back before the
    exception propagates so we never leave an orphan on disk.
    """
    tracker.update("loading", "Saving to registry…", step=_TOTAL_LOAD_STEPS, total_steps=_TOTAL_LOAD_STEPS)
    entry = _auto_register_dataset(
        ctx.medias,
        name=name,
        origin_str=_origin_to_str(origin),
        source=origin,
        clipper=clipper,
        embedder=embedder,
        created_by=created_by,
        display_name=name,
        ingest_started_at=ingest_started_at,
    )
    if entry is None:
        return task_id, None
    entry_id = entry["id"]
    try:
        _migrate_context_id(task_id, entry_id)
        ctx.dataset_display_name = entry.get("name", name)
        # Associate the loading task with the real dataset ID so the
        # finished-task tick is attributed to the right dashboard row.
        loading_tasks.set_dataset_id(task_id, entry_id)
    except Exception:
        # Migration failed after the registry entry was written; roll
        # it back so the dashboard doesn't show a half-built dataset.
        try:
            _reg_unregister(entry_id)
        except Exception:
            traceback.print_exc()
        raise
    return entry_id, entry_id


def _warmup_embedder_async(media_dict: dict) -> None:
    """Warm up the embedder (model load + text-encoder prime) in a daemon thread.

    Fire-and-forget: the caller doesn't wait, and there is no progress
    surface; the dataset is usable for grid-browsing immediately, and
    text sort waits behind its own ``_embedder_load_lock`` (see
    ``vtsearch/routes/sorting.py:_load_embedder_with_progress``) on first
    use.  ``MediaEmbedder.load_models`` is idempotent and serialised by
    a per-class lock, so racing this thread against an on-demand sort
    load is safe.
    """

    def _run() -> None:
        emb = _get_embedder_for_medias(media_dict)
        if emb is None:
            return
        try:
            emb.load_models()
            emb.embed_text("warmup")
        except Exception:
            pass

    threading.Thread(target=_run, name="warmup-embedder", daemon=True).start()


def _handle_load_failure(
    exc: BaseException,
    context_id: str,
    tracker,
    registry_entry_id: str | None = None,
) -> None:
    """Unregister the context and write the failure into *tracker*.

    If *registry_entry_id* is set, the on-disk registry entry (and its
    backing pkl) is also removed; this prevents an orphaned dashboard
    row when a load fails after :func:`_register_and_migrate` has
    already written the entry.
    """
    from vtscore.state.core import unregister_context  # noqa: PLC0415

    if isinstance(exc, CancelledError):
        error = "Cancelled"
    elif isinstance(exc, ImportError):
        traceback.print_exc()
        error = f"Missing dependency: {exc}. Install all required packages with: pip install -e '.[cpu,dev]'"
    elif isinstance(exc, MemoryError):
        error = "Out of memory: this dataset is too large. Try a smaller dataset or free up system RAM."
    else:
        traceback.print_exc()
        error = str(exc) or repr(exc) or "Unknown error during dataset loading"

    unregister_context(context_id)
    if registry_entry_id:
        try:
            _reg_unregister(registry_entry_id)
        except Exception:
            traceback.print_exc()
    gc.collect()
    tracker.update("idle", "", 0, 0, error=error, step=None, total_steps=None)


def _run_origin_load_in_background(
    load_fn,
    origin: dict,
    *,
    name: str = "",
    clipper: str = "",
    clipper_params: dict | None = None,
    chain_steps: list[dict] | None = None,
    embedder: str = "",
    created_by: str = "",
    media_type: str = "",
) -> str:
    """Run a dataset load in a background thread with standard error handling.

    *load_fn* is called with a single argument (the target medias dict);
    and should populate it in-place.  Everything after (origin tagging,
    clipping, dedup, diversity tree, registry, embedder warm-up) is handled
    automatically.

    The dataset context is NOT activated during loading.  It is activated
    only upon successful completion, and only if no other dataset is
    currently active.

    Returns the task_id that can be used to poll progress or cancel.
    """
    # Reset the legacy cancellation flag so a previous cancel does not
    # immediately abort this new operation; but only when no other parallel
    # loads are running (otherwise we would clear cancellation that might
    # still be intended for those in-flight tasks).
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

    task_id = f"_loading_{uuid4().hex[:8]}"
    ingest_started_at = time.time()
    tracker = loading_tasks.create_task(
        task_id, name or _origin_to_str(origin), media_type=media_type, embedder=embedder
    )
    tracker.update("loading", "Preparing dataset...", step=1, total_steps=_TOTAL_LOAD_STEPS)

    # Snapshot the user that triggered the load so background per-user
    # state (settings writes, settings_source sync) resolves correctly.
    request_user = created_by or get_current_user()

    def task():
        from vtsearch.auth import thread_user  # noqa: PLC0415
        from vtscore.state.core import thread_dataset_context  # noqa: PLC0415

        ctx = DatasetContext(task_id)
        # Pin the in-flight context to this thread so importers, clippers,
        # dedup, diversity-tree, and label-sync helpers that resolve via
        # ``get_active_context()`` see the dataset being built, not the
        # empty fallback context.  Without this, mutations addressed at
        # the active context (e.g. label restoration, vote replay) land
        # on ``_empty_dataset_context`` and are silently lost.
        #
        # ``thread_user`` / ``thread_dataset_context`` snapshot the prior
        # thread-local values on entry and restore them on exit, so a
        # future pooled / reused worker thread cannot leak identity or
        # context across jobs.  ``mark_finished`` runs in the outer
        # ``finally`` (after the scopes exit) so callers waiting on
        # ``has_active_tasks() == False`` see fully cleaned-up worker
        # state.
        context_id = task_id
        registry_entry_id: str | None = None
        controller = _LoadGateController(tracker)
        stepped = _make_stepped_progress(controller, tracker)

        try:
            with thread_user(request_user), thread_dataset_context(ctx):
                try:
                    controller.acquire_download()
                    tracker.update("loading", "Preparing new dataset…", 0, 0, step=1, total_steps=_TOTAL_LOAD_STEPS)
                    register_context(ctx)
                    gc.collect()

                    _run_importer(load_fn, ctx, stepped)
                    tracker.check_cancelled()

                    # Backstop: an importer that completes without raising but
                    # produces zero medias would otherwise sail through clipping,
                    # dedup, and registry steps and surface as a green dashboard
                    # row with 0 items.  Fail loudly instead, mirroring the
                    # staging-flow guard at ``_stage_importer_in_background``.
                    if not ctx.medias:
                        raise ValueError("Import produced no medias.")

                    # Post-load stages are CPU/GPU-bound and touch embeddings;
                    # gate them on the embed semaphore.  Calling swap here
                    # unconditionally is also the safety net for minimalist
                    # importers that complete without firing an ``"embedding"``
                    # status: ``_make_stepped_progress``'s callback-driven swap
                    # never fires for them, so without this call the download
                    # gate would stay held through every post-load stage.  The
                    # ``finally: controller.release()`` below is a second-line
                    # backstop that releases whichever gate is held on any
                    # error path.  No-op if the importer already swapped
                    # mid-load.
                    controller.swap_to_embed()

                    apply_custom_metadata_md5(ctx.medias)
                    _tag_origins(ctx.medias, origin)
                    _apply_clipper_stage(ctx, tracker, clipper, clipper_params, chain_steps)
                    _embed_missing_stage(ctx, tracker, embedder)
                    _drop_none_embeddings_stage(ctx, tracker)
                    _collapse_duplicates_stage(ctx, tracker)
                    _build_diversity_tree_stage(ctx, tracker)
                    tracker.check_cancelled()
                    context_id, registry_entry_id = _register_and_migrate(
                        ctx, tracker, task_id, origin, name, clipper, embedder, created_by, ingest_started_at
                    )
                    # Embedder warm-up is fire-and-forget so the dashboard row goes
                    # green immediately.  Text sort waits behind its own progress
                    # bar on first use if the model isn't ready yet.
                    _warmup_embedder_async(ctx.medias)

                    from vtsearch.achievements import record_dataset_load  # noqa: PLC0415

                    record_dataset_load(str(origin.get("importer", "")))
                except Exception as exc:
                    _handle_load_failure(exc, context_id, tracker, registry_entry_id=registry_entry_id)
                finally:
                    controller.release()
                    clear_thread_progress()
        finally:
            loading_tasks.mark_finished(task_id)

    threading.Thread(target=task, daemon=True).start()
    return task_id


def _migrate_context_id(old_id: str, new_id: str) -> None:
    """Re-key a context from *old_id* to *new_id* in the store."""
    from vtscore.state.core import _contexts, _state_lock

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
    from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

    # Normalize ``field_type="file"`` values to UploadedFile.  The
    # request path supplies a FileStorage / BytesIOUploadedFile already;
    # the reload-from-origin path supplies a server path string that
    # needs CliUploadedFile wrapping so ``run()`` doesn't have to
    # branch on the input shape.
    field_values = wrap_cli_file_fields(importer.fields, field_values)
    created_by = get_current_user()
    origin = importer.build_origin(field_values)
    clipper_name = field_values.pop("clipper", "") or ""
    clipper_params = field_values.pop("clipper_params", None)
    chain_steps = _parse_chain_field(field_values.pop("clipper_chain", None))
    # Keep clipper in field_values for importers that need it (e.g. demo
    # importer writes a .clipper sidecar for readiness tracking).
    field_values["clipper"] = clipper_name
    embedder_name = field_values.get("embedder", "")

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
        chain_steps=chain_steps,
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
    from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

    field_values = wrap_cli_file_fields(importer.fields, field_values)
    _request_user = get_current_user()

    def stage_task():
        from vtsearch.auth import thread_user

        with thread_user(_request_user):
            try:
                temp_medias: dict = {}
                importer.run(field_values, temp_medias)
                apply_custom_metadata_md5(temp_medias)
                embed_missing(temp_medias, field_values.get("embedder", "") or "", on_progress=update_progress)
                temp_medias = {mid: m for mid, m in temp_medias.items() if m.get("embedding") is not None}

                if not temp_medias:
                    update_progress("idle", "", 0, 0, error="Import produced no medias.")
                    return

                first = next(iter(temp_medias.values()))
                media_type = first.get("media_type", "audio")
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
                    error="Out of memory: this dataset is too large. Try a smaller dataset or free up system RAM.",
                )
            except Exception as e:
                traceback.print_exc()
                error_msg = str(e) or repr(e) or "Unknown error during staging"
                update_progress("idle", "", 0, 0, error=error_msg)

    thread = threading.Thread(target=stage_task, daemon=True)
    thread.start()
