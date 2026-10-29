"""A dataset load from an origin: the importer, then every post-import stage, on a worker thread.

:func:`_run_origin_load_in_background` is the single-dataset load every
importer run, reload-from-origin and demo load ends in.  It acquires the
download gate, runs the importer, swaps to the embed gate, then
:func:`_finish_dataset_load` drives the clipper, embed, dedup, coverage-atlas,
registry and (optional) projection stages from :mod:`vtscore.datasets.stages`,
routing each stage's progress into the load's
:class:`~vtscore.datasets.stages._common.AdaptiveLoadPacer`.  The multi-dataset
load (:mod:`vtscore.datasets.load_multi`) reuses :class:`_DatasetLoadSpec`,
:class:`_LoadIds`, :func:`_finish_dataset_load` and the two hook runners.

The gates, the task harness and the failure handling are
:mod:`vtscore.datasets.load_workers`'.  The persistence hook is read through
that module at call time (``_workers._last_embedder_persistence_hook``): the
app installs it after this module is imported, so a ``from … import`` copy
would stay ``None``.
"""

from __future__ import annotations

import copy
import gc
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from vtscore.datasets import load_workers as _workers
from vtscore.datasets.import_event import FAILED, SUCCEEDED, DatasetImported
from vtscore.datasets.load_workers import (
    _CANCELLED_MESSAGE,
    _LoadGateController,
    _describe_source,
    _handle_load_failure,
    _make_stepped_progress,
    _park_load_terminal,
    _recorded_embedder_name,
    _run_importer,
    _spawn_import_worker,
    _start_import_task,
    _tag_origins,
    _warmup_embedder_async,
)
from vtscore.datasets.loader import apply_custom_metadata_md5
from vtscore.state import DatasetContext, register_context

from vtscore.datasets.stages._common import (
    AdaptiveLoadPacer,
    FinalizeProgress,
    _TOTAL_LOAD_STEPS,
    _origin_to_str,
    load_cost_terms,
    load_step_weights,
)
from vtscore.datasets.stages._load_profiler import start_profiler
from vtscore.datasets.stages.clipper import _apply_clipper_stage, _relazify_reference_clips_stage
from vtscore.datasets.stages.embedding import _embed_missing_stage
from vtscore.datasets.stages.finalize import (
    _build_coverage_atlas_stage,
    _collapse_duplicates_stage,
    _collapse_near_duplicates_stage,
    _drop_none_embeddings_stage,
    _follow_source_merges_stage,
)
from vtscore.datasets.stages.projection import _build_projection_stage, _maybe_signpost_texts_stage
from vtscore.datasets.stages.registry import _register_and_migrate
from vtscore.datasets.thumbnail_warm import start_archive_thumbnail_warm


@dataclass
class _DatasetLoadSpec:
    """The per-dataset half of a load's parameters: what the post-import stages need.

    Everything :func:`_finish_dataset_load` reads about *one* dataset, gathered
    so a multi-dataset import (:mod:`vtscore.datasets.load_multi`) can carry
    one of these per output while the single-dataset load builds one from its
    keyword arguments.
    """

    task_id: str
    origin: dict
    name: str
    clipper: str
    clipper_params: dict | None
    chain_steps: list[dict] | None
    embedder: str
    embedders: list[str] | None
    created_by: str
    media_type: str
    build_projection: bool
    ingest_started_at: float
    #: The id every dataset of one multi-dataset import shares, and the
    #: ``OutputSpec.category`` this one stands for; ``None`` / ``""`` for a
    #: single-dataset load.  Recorded on the registry entry.
    import_group: str | None = None
    output_category: str = ""
    #: The finished medias of the sibling datasets this one's converter rows
    #: read from (the Image dataset, for a Face dataset), so the outputs follow
    #: the merges made there.  Filled in by the multi-dataset load just before
    #: it finalizes this dataset; empty for a single-dataset load.
    source_siblings: list[dict[int, dict[str, Any]]] = field(default_factory=list)


@dataclass
class _LoadIds:
    """The ids :func:`_finish_dataset_load` learns as it goes.

    Mutable on purpose: the caller's failure handler needs the registry entry
    id even when the stages raise *after* registration, so the helper writes
    it here the moment it exists rather than returning it at the end.
    """

    context_id: str
    registry_entry_id: str | None = None


def _finish_dataset_load(ctx: DatasetContext, tracker, pacer, spec: _DatasetLoadSpec, ids: _LoadIds) -> None:
    """Run every post-import stage on *ctx*: origin tagging through registration and warm-ups.

    The importer has populated ``ctx.medias``; this stamps origins, clips,
    embeds, finalizes (drops failed embeds, dedups, builds the coverage atlas),
    saves and registers the dataset, then kicks off the opt-in projection and
    the fire-and-forget warm-ups.  Progress goes through *pacer* (the load's
    :class:`AdaptiveLoadPacer`) and *tracker*; ``ids.context_id`` /
    ``ids.registry_entry_id`` are updated as soon as the registry stage has
    migrated the context, so a caller catching an exception from here can roll
    the entry back.  Runs with *ctx* pinned as the thread's dataset context by
    the caller.
    """
    apply_custom_metadata_md5(ctx.medias)
    _tag_origins(ctx.medias, spec.origin)
    _apply_clipper_stage(ctx, pacer, spec.clipper, spec.clipper_params, spec.chain_steps)
    embed_failures = _embed_missing_stage(ctx, pacer, spec.embedders if spec.embedders else [spec.embedder])
    # Step 4 (finalize) bundles several sub-stages. Route them
    # through a FinalizeProgress proxy so each maps into its own
    # ordered slice of the step-4 bar instead of independently
    # filling (and pinning at 100%) the whole slice — keeps the
    # bar advancing and the ETA self-correcting through the
    # serialize/disk-write window. See FinalizeProgress.
    fin = FinalizeProgress(pacer, spec.media_type)
    fin.begin("cleanup")
    dropped_groups = _drop_none_embeddings_stage(ctx, fin, embed_failures)
    # Re-lazify clips from reference (thin) parents now that
    # embedding is done: strip their materialized bytes so the
    # dataset stores recipes, not duplicated clip payloads.
    _relazify_reference_clips_stage(ctx, fin)
    fin.begin("dedup")
    _collapse_duplicates_stage(ctx, fin)
    _collapse_near_duplicates_stage(ctx, fin)
    _follow_source_merges_stage(ctx, spec.source_siblings)
    fin.begin("coverage")
    _build_coverage_atlas_stage(ctx, fin)
    tracker.check_cancelled()
    # Opt-in (rides the projection opt-in): cache a signpost
    # text per media BEFORE the registry save, so the texts —
    # the sign pipeline's only full-corpus model cost — are
    # pickled with the dataset and later browse / Find→Browse
    # re-fits skip the text models entirely.
    _maybe_signpost_texts_stage(ctx, fin, spec.build_projection)
    fin.begin("registry")
    ids.context_id, ids.registry_entry_id = _register_and_migrate(
        ctx,
        fin,
        spec.task_id,
        spec.origin,
        spec.name,
        spec.clipper,
        spec.embedder,
        spec.created_by,
        spec.ingest_started_at,
        import_group=spec.import_group,
        output_category=spec.output_category,
        dropped_groups=dropped_groups,
    )
    # Opt-in: compute + persist the 2-D Browse projection now,
    # so the Browse canvas opens instantly instead of building
    # UMAP lazily on first visit.  Best-effort and runs after
    # registration: the dataset is already saved and usable, so
    # a failure (or a cancel during the fit) leaves it intact
    # and just defers the projection to the lazy Browse path.
    if spec.build_projection:
        fin.begin("projection")
        try:
            _build_projection_stage(ctx, fin)
        except Exception:
            traceback.print_exc()
    # Embedder warm-up is fire-and-forget so the dashboard row goes
    # green immediately.  Text sort waits behind its own progress
    # bar on first use if the model isn't ready yet.
    _warmup_embedder_async(ctx.medias)
    # Same deal for archive-member thumbnails: the importer reads
    # no member bytes by design, so those media land with no
    # thumbnail and every browse tile would stream a tar member
    # and decode it on the request thread.  Warm them off the
    # request path now that the dataset is registered and
    # browsable; a no-op for every other import path.  Kicked on
    # reload too (this runs for pickle loads as well), since the
    # save above necessarily predates the pass.
    start_archive_thumbnail_warm(ctx)

    from vtscore.achievements_hooks import record_achievement  # noqa: PLC0415

    record_achievement("dataset_load", str(spec.origin.get("importer", "")))


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
    merge_near_duplicates: bool = False,
    dataset_id: str = "",
    n_hint: int | None = None,
    download_size_mb_hint: float | None = None,
    post_load: Callable[[DatasetContext], None] | None = None,
    on_finished: Callable[[DatasetImported], None] | None = None,
) -> str:
    """Run a dataset load in a background thread with standard error handling.

    *load_fn* is called with a single argument (the target medias dict);
    and should populate it in-place.  Everything after (origin tagging,
    clipping, dedup, coverage atlas, registry, embedder warm-up) is handled
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

    *post_load*, when given, is called once with the new dataset's
    :class:`DatasetContext` after a load that **succeeded** (never after a
    failure or a cancel).  It runs on the worker thread, with that context
    pinned as the thread's dataset context and the requesting user replayed,
    but only once the load's own tracker has parked terminal and its timing
    profiler has finished - so the dataset row is already live, and nothing
    the hook does is billed to the load's measured phases.  An exception it raises
    is logged and swallowed: the dataset is saved either way.  The app uses it
    to start the importing user's AutoFind detectors on the new dataset.

    *on_finished*, when given, is called once with a
    :class:`~vtscore.datasets.import_event.DatasetImported` saying how the load
    ended: after a load that succeeded (following *post_load*) and after one
    that failed, but never after a cancel.  It runs on the worker thread on the
    same terms as *post_load*, except that no dataset context is pinned (the
    event carries everything the hook is given), and an exception it raises is
    swallowed the same way.  The app uses it to run the functions a server
    admin names with ``--on-dataset-imported``.

    Returns the task_id that can be used to poll progress or cancel.
    """
    # Remember the user's embedder pick per media type so the next dataset
    # importer modal can pre-select it even when no loaded dataset is
    # around to supply the same hint via ``guessedMediaEmbedder``.  Read off
    # the workers module at call time: the app installs the hook after import.
    hook = _workers._last_embedder_persistence_hook
    if media_type and embedder and hook is not None:
        try:
            hook(media_type, embedder)
        except Exception:
            pass

    ingest_started_at = time.time()
    # ``_start_import_task`` mints the id, creates the per-task tracker, and
    # snapshots the user that triggered the load so background per-user state
    # (settings writes, settings_source sync) resolves correctly.
    task = _start_import_task(
        prefix="_loading_",
        display_name=name or _origin_to_str(origin),
        total_steps=_TOTAL_LOAD_STEPS,
        media_type=media_type,
        embedder=embedder,
        weights=load_step_weights(media_type, n=n_hint, download_size_mb=download_size_mb_hint, embedder=embedder),
        created_by=created_by,
        source=_describe_source(origin),
    )
    task_id, tracker = task.task_id, task.tracker
    # Env-gated per-phase timing recorder (VTSEARCH_PROFILE_LOAD); a no-op
    # stand-in and zero-cost when off. It feeds the developer-side refit of the
    # shipped load cost model, splitting cold/warm model loads and finalize
    # sub-slots. Subscribed before the first phase fires — hence before the
    # step-1 update below, not inside ``_start_import_task``. See
    # scripts/profiling/README.md.
    profiler = start_profiler(tracker, media_type, embedder)
    tracker.update("loading", "Preparing dataset...", step=1, total_steps=_TOTAL_LOAD_STEPS)

    def load_task():
        from vtscore.state.core import thread_dataset_context  # noqa: PLC0415

        ctx = DatasetContext(task_id)
        ctx.merge_near_duplicates = merge_near_duplicates
        # Pin the in-flight context to this thread so importers, clippers,
        # dedup, coverage-atlas, and label-sync helpers that resolve via
        # ``get_active_context()`` see the dataset being built, not the
        # empty fallback context.  Without this, mutations addressed at
        # the active context (e.g. label restoration, vote replay) land
        # on ``_empty_dataset_context`` and are silently lost.
        #
        # ``thread_dataset_context`` snapshots the prior thread-local value on
        # entry and restores it on exit, so a future pooled / reused worker
        # thread cannot leak context across jobs; ``_spawn_import_worker`` does
        # the same for the user identity, and runs ``mark_finished`` after both
        # scopes have exited so callers waiting on ``has_active_tasks() ==
        # False`` see fully cleaned-up worker state.
        ids = _LoadIds(context_id=task_id)
        spec = _DatasetLoadSpec(
            task_id=task_id,
            origin=origin,
            name=name,
            clipper=clipper,
            clipper_params=clipper_params,
            chain_steps=chain_steps,
            embedder=embedder,
            embedders=embedders,
            created_by=created_by,
            media_type=media_type,
            build_projection=build_projection,
            ingest_started_at=ingest_started_at,
        )
        controller = _LoadGateController(tracker, task.total_steps)
        # Pace the unified bar from the per-phase cost terms, rebasing on what
        # actually happens (cached archives, observed bandwidth, skipped
        # phases). All stage progress below routes through the pacer.
        cost_terms, terms_calibrated = load_cost_terms(
            media_type, n=n_hint, download_size_mb=download_size_mb_hint, embedder=embedder
        )
        pacer = AdaptiveLoadPacer(tracker, cost_terms, calibrated=terms_calibrated)
        stepped = _make_stepped_progress(controller, pacer)
        profiler.bind_thread()  # so FinalizeProgress.begin stamps land here (no-op when off)

        try:
            with thread_dataset_context(ctx):
                try:
                    controller.acquire_download()
                    pacer.update("loading", "Preparing new dataset…", 0, 0, step=1, total_steps=_TOTAL_LOAD_STEPS)
                    register_context(ctx)
                    gc.collect()

                    _run_importer(load_fn, ctx, stepped)
                    tracker.check_cancelled()

                    # Backstop: an importer that completes without raising but
                    # produces zero medias would otherwise sail through clipping,
                    # dedup, and registry steps and surface as a green dashboard
                    # row with 0 items.  Fail loudly instead, mirroring the
                    # staging-flow guard in ``load_staging._stage_importer_in_background``.
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

                    # Everything after the importer — origin tagging, clipping,
                    # embedding, finalize, registry, warm-ups — is the per-dataset
                    # helper, shared with the multi-dataset import.
                    _finish_dataset_load(ctx, tracker, pacer, spec, ids)
                except Exception as exc:
                    _handle_load_failure(
                        exc, ids.context_id, tracker, registry_entry_id=ids.registry_entry_id, task_id=task_id
                    )
                finally:
                    controller.release()
        finally:
            # Pass the demo dataset id (empty for non-demo loads) so profiler
            # rows carry it and can resolve the archive size via
            # ``download_size_mb_for`` — otherwise app-recorded rows land with
            # ``dataset_id: ""`` and can't feed fit_load_weights.py (see #2614).
            # The profiler learns the resolved embedder here rather than at
            # construction: it is only known once the medias exist (#3345).
            # Writes JSONL + unbinds (no-op when off).
            profiler.finish(len(ctx.medias), dataset_id, embedder=_recorded_embedder_name(ctx.medias, embedder))
            _park_load_terminal(tracker, len(ctx.medias))

        # Outside the ``finally``: a load that raised past it never reaches
        # here, and one that failed inside it stamped ``error`` on the tracker.
        # AutoFind's hook goes first: it only spawns a thread, while an
        # ``on_finished`` hook may block on, say, a mail server.
        _run_post_load(post_load, ctx, tracker)
        _report_finished(
            on_finished,
            ctx,
            tracker,
            dataset_id=ids.context_id,
            name=name or _origin_to_str(origin),
            user=task.request_user,
            media_type=media_type,
            origin=origin,
        )

    return _spawn_import_worker(task, load_task)


def _run_post_load(post_load: Callable[[DatasetContext], None] | None, ctx: DatasetContext, tracker) -> None:
    """Call a load's *post_load* hook if the load succeeded; see :func:`_run_origin_load_in_background`."""
    if post_load is None or tracker.get().get("error"):
        return
    from vtscore.state.core import thread_dataset_context  # noqa: PLC0415

    with thread_dataset_context(ctx):
        try:
            post_load(ctx)
        except Exception:
            traceback.print_exc()


def _report_finished(
    on_finished: Callable[[DatasetImported], None] | None,
    ctx: DatasetContext,
    tracker,
    *,
    dataset_id: str,
    name: str,
    user: str,
    media_type: str,
    origin: dict,
) -> None:
    """Tell a load's *on_finished* hook how it ended; see :func:`_run_origin_load_in_background`.

    *dataset_id* and *name* are what the load was asked for; on success they
    are replaced by what the registry saved.
    """
    if on_finished is None:
        return
    error = tracker.get().get("error") or ""
    if error == _CANCELLED_MESSAGE:
        return
    try:
        if error:
            event = DatasetImported(
                outcome=FAILED,
                dataset_id="",
                name=name,
                user=user,
                media_type=media_type,
                n_media=0,
                origin=copy.deepcopy(origin),
                error=error,
            )
        else:
            first = next(iter(ctx.medias.values()), {})
            event = DatasetImported(
                outcome=SUCCEEDED,
                dataset_id=dataset_id,
                name=ctx.dataset_display_name or name,
                user=user,
                media_type=media_type or first.get("media_type", ""),
                n_media=len(ctx.medias),
                origin=copy.deepcopy(origin),
            )
        on_finished(event)
    except Exception:
        traceback.print_exc()
