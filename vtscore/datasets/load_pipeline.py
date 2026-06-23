"""Dataset loading orchestration: background threading, gate handoff, staging.

This module strings the post-import :mod:`vtscore.datasets.stages` together
into a background-threaded dataset load: it acquires the download/embed
concurrency gates, runs the importer, then drives the clipper, embed, dedup,
diversity-tree, registry, and (optional) projection stages while routing each
stage's progress into the shared loading-task tracker. The per-stage work
itself lives under :mod:`vtscore.datasets.stages`; the
:class:`~vtscore.concurrency.gate.ConcurrencyGate` primitive lives in
:mod:`vtscore.concurrency.gate`.
"""

from __future__ import annotations

import gc
import json
import time
import traceback
import threading
from collections.abc import Iterable
from typing import Any, Callable
from uuid import uuid4

from vtscore.config import CoreConfig, DATA_DIR
from vtscore.concurrency.gate import ConcurrencyGate
from vtscore.concurrency.progress import (
    CancelledError,
    clear_thread_progress,
    dataset_progress,
    loading_tasks,
    set_thread_progress,
    update_progress,
)
from vtscore.datasets import export_dataset_to_file
from vtscore.datasets.loader import apply_custom_metadata_md5
from vtscore.datasets.registry import unregister_dataset as _reg_unregister
from vtscore.state import DatasetContext, clear_all, register_context

from vtscore.datasets.stages._common import (
    _LOAD_STEP_WEIGHTS,
    _STATUS_TO_STEP,
    _TOTAL_LOAD_STEPS,
    _origin_to_str,
)
from vtscore.datasets.stages.clipper import _apply_clipper_stage
from vtscore.datasets.stages.embedding import embed_missing, _embed_missing_stage
from vtscore.datasets.stages.finalize import (
    _build_diversity_tree_stage,
    _collapse_duplicates_stage,
    _drop_none_embeddings_stage,
)
from vtscore.datasets.stages.projection import _build_projection_stage
from vtscore.datasets.stages.registry import _register_and_migrate


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


def _parse_bool(value: Any) -> bool:
    """Coerce a request-supplied flag to ``bool``.

    Accepts native bools, the ``"true"``/``"false"`` strings that
    checkbox fields serialize to (per :class:`PluginField`), and ``None``
    (treated as ``False``).
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _parse_embedder_list(value: Any) -> list[str] | None:
    """Coerce a request-supplied embedder list to ``list[str]`` (or ``None``).

    The v3 create-time three-role picker sends the bound trio (text / patch /
    structural picks, deduped) under the ``embedders`` field.  It arrives as a
    native list on JSON-body routes and as a string on multipart routes (a JSON
    array string, or a comma-separated fallback).  ``None`` / empty in →
    ``None`` (the caller falls back to the single ``embedder`` field — the
    pre-trio path).  Order is preserved and blanks/dupes are dropped.
    """
    if value is None:
        return None
    items: list[Any]
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            items = parsed if isinstance(parsed, list) else [parsed]
        except (TypeError, ValueError):
            items = text.split(",")
    out: list[str] = []
    for item in items:
        name = str(item).strip()
        if name and name not in out:
            out.append(name)
    return out or None


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
    embedders: list[str] | None = None,
    created_by: str = "",
    media_type: str = "",
    build_projection: bool = False,
) -> str:
    """Run a dataset load in a background thread with standard error handling.

    *load_fn* is called with a single argument (the target medias dict);
    and should populate it in-place.  Everything after (origin tagging,
    clipping, dedup, diversity tree, registry, embedder warm-up) is handled
    automatically.

    *embedder* is the primary create-time embedder (recorded as each media's
    primary and used for the per-media-type persistence hint / task display).
    *embedders* is the optional v3 trio (text / patch / structural create-time
    picks): when set, every name is embedded during ingest so a multi-embedder
    dataset is produced.  ``None`` falls back to the single *embedder* — the
    pre-trio create path, unchanged.

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
        task_id,
        name or _origin_to_str(origin),
        media_type=media_type,
        embedder=embedder,
        step_weights=_LOAD_STEP_WEIGHTS,
    )
    tracker.update("loading", "Preparing dataset...", step=1, total_steps=_TOTAL_LOAD_STEPS)

    # Snapshot the user that triggered the load so background per-user
    # state (settings writes, settings_source sync) resolves correctly.
    from vtsearch.auth import get_current_user  # noqa: PLC0415

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
                    _embed_missing_stage(ctx, tracker, embedders if embedders else [embedder])
                    _drop_none_embeddings_stage(ctx, tracker)
                    _collapse_duplicates_stage(ctx, tracker)
                    _build_diversity_tree_stage(ctx, tracker)
                    tracker.check_cancelled()
                    context_id, registry_entry_id = _register_and_migrate(
                        ctx, tracker, task_id, origin, name, clipper, embedder, created_by, ingest_started_at
                    )
                    # Opt-in: compute + persist the 2-D Browse projection now,
                    # so the Browse canvas opens instantly instead of building
                    # UMAP lazily on first visit.  Best-effort and runs after
                    # registration: the dataset is already saved and usable, so
                    # a failure (or a cancel during the fit) leaves it intact
                    # and just defers the projection to the lazy Browse path.
                    if build_projection:
                        try:
                            _build_projection_stage(ctx, tracker, context_id)
                        except Exception:
                            traceback.print_exc()
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
    from vtsearch.auth import get_current_user  # noqa: PLC0415

    field_values = wrap_cli_file_fields(importer.fields, field_values)
    created_by = get_current_user()
    origin = importer.build_origin(field_values)
    clipper_name = field_values.pop("clipper", "") or ""
    clipper_params = field_values.pop("clipper_params", None)
    chain_steps = _parse_chain_field(field_values.pop("clipper_chain", None))
    # Keep clipper in field_values for importers that need it (e.g. demo
    # importer stores it in the container metadata for readiness tracking).
    field_values["clipper"] = clipper_name
    embedder_name = field_values.get("embedder", "")
    embedders = _parse_embedder_list(field_values.pop("embedders", None))
    # The primary picker's choice always leads the embed order (it becomes each
    # media's recorded primary); the trio's patch/structural picks ride behind
    # it.  Defensive: include the primary even if the client omitted it.
    if embedders and embedder_name and embedder_name not in embedders:
        embedders = [embedder_name, *embedders]
    build_projection = _parse_bool(field_values.pop("build_projection", None))

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
        embedders=embedders,
        created_by=created_by,
        media_type=media_type_hint,
        build_projection=build_projection,
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
    from vtsearch.auth import get_current_user  # noqa: PLC0415
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
                from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

                temp_medias = {mid: m for mid, m in temp_medias.items() if media_embedding(m) is not None}

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
