"""Multi-dataset imports: one importer run, several datasets (#4703).

A single-dataset load (:func:`~vtscore.datasets.load_pipeline._run_origin_load_in_background`)
is one importer run feeding one :class:`~vtscore.state.DatasetContext` that
one chain of post-import stages turns into one registry entry.  A
**multi-dataset** load keeps the one importer run — the expensive acquire:
downloading an archive, walking a folder, listing a bucket — and fans its
output into N contexts, one per :class:`~vtscore.datasets.importers.base.OutputSpec`,
each of which then goes through the same post-import stages
(:func:`~vtscore.datasets.load_pipeline._finish_dataset_load`) on its own and
lands as its own dataset, with its own dashboard row, its own
:class:`~vtscore.datasets.import_event.DatasetImported` event and its own
AutoFind run.

What is shared and what is per output:

* **Shared:** the importer's acquire phase, run once under the download gate
  through :meth:`~vtscore.datasets.importers.base.core.ImporterBase.run_outputs`
  (or its chunked twin); the ``reference_files`` / ``build_projection`` /
  ``merge_near_duplicates`` toggles; the requesting user.
* **Per output:** the dataset's media type and source rows (what the importer
  reads), and the embedders, clipper, cleanup gates and name (what the
  pipeline applies).  Each output's origin is
  :meth:`~ImporterBase.build_origin` of the *narrowed* field values, so it
  records exactly one dataset's ``media_type`` / ``source_specs`` and a later
  reload-from-origin rebuilds that one dataset through the ordinary
  single-dataset path.

The N datasets stay linked: every registry entry records the run's
``import_group`` (one id minted per run) and the ``output_category`` it stands
for, so a Face dataset can find the Image sibling its crops were cut from.

Progress and cancellation: every output has its own loading task from the
moment the request returns, so the dashboard shows N rows at once.  While the
shared acquire runs, its progress is mirrored onto every row; a cancel on one
row during that phase drops *that* output (its context is released, its row
parks ``Cancelled``) and the acquire carries on for the rest, aborting only
when no output is left.  Afterwards the outputs are finalized one at a time
under the embed gate — a cancel or a failure in one is that output's alone.

The env-gated load profiler is not bound here: its rows describe a single
load's phases, and a shared acquire phase has no single dataset to bill to.
"""

from __future__ import annotations

import gc
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from vtscore.concurrency.progress import CancelledError, clear_thread_progress, loading_tasks, set_thread_progress
from vtscore.datasets.clipper_chain import append_cleaner_steps
from vtscore.datasets.import_event import DatasetImported
from vtscore.datasets.importers.base import OutputSpec, output_dataset_name
from vtscore.datasets import load_pipeline as _pipeline
from vtscore.datasets.load_pipeline import (
    _DatasetLoadSpec,
    _LoadGateController,
    _LoadIds,
    _finish_dataset_load,
    _handle_load_failure,
    _parse_bool,
    _parse_chain_field,
    _park_load_terminal,
    _report_finished,
    _run_post_load,
    _start_import_task,
    auto_chunk_size,
    consume_chunks_into,
)
from vtscore.datasets.stages._common import (
    _STATUS_TO_STEP,
    _TOTAL_LOAD_STEPS,
    AdaptiveLoadPacer,
    load_cost_terms,
    load_step_weights,
)
from vtscore.state import DatasetContext, register_context

#: Shared form keys a multi-dataset request may still carry at the top level
#: but that only mean anything per dataset: every output brings its own, so a
#: stray top-level copy is dropped rather than applied to all of them.
_PER_OUTPUT_KEYS = ("clipper", "clipper_params", "clipper_chain", "cleaners", "embedder", "embedders", "source_specs")


@dataclass
class _OutputJob:
    """One output's worth of load state: its spec, task, context and pacer."""

    output: OutputSpec
    narrowed: dict[str, Any]
    label: str
    spec: _DatasetLoadSpec
    task: Any
    ctx: DatasetContext
    pacer: AdaptiveLoadPacer
    ids: _LoadIds
    #: Set once the job has reached a terminal state on its own (a cancel during
    #: the shared acquire); the shared phases then leave it alone.
    done: bool = False

    @property
    def tracker(self):
        return self.task.tracker


class _FanoutTracker:
    """The tracker surface the shared acquire phase writes to: every live job at once.

    :class:`_LoadGateController` and the importer's progress callback both
    expect one object with ``update`` / ``check_cancelled``; during the shared
    phase that object is this fan-out, which mirrors each update onto every
    output still in flight and treats a cancel on one row as that row's exit
    rather than everyone's.
    """

    def __init__(self, jobs: list[_OutputJob]) -> None:
        self._jobs = jobs
        #: The gate controller whose waiting messages this fan-out carries;
        #: bound right after construction (each needs the other).
        self.controller: _LoadGateController | None = None

    @property
    def live(self) -> list[_OutputJob]:
        return [j for j in self._jobs if not j.done]

    def update(self, status: str, message: str = "", current: int = 0, total: int = 0, **kwargs: Any) -> None:
        for job in self.live:
            job.pacer.update(status, message, current, total, **kwargs)

    def check_cancelled(self) -> None:
        """Retire every job whose row was cancelled; raise once none is left."""
        for job in self.live:
            if job.tracker.is_cancelled:
                self.retire(job, CancelledError("Operation cancelled by user"))
        if not self.live:
            raise CancelledError("Operation cancelled by user")

    def retire(self, job: _OutputJob, exc: BaseException) -> None:
        """Fail *job* with *exc* now (its context released, its row parked) and drop it from the live set."""
        _handle_load_failure(exc, job.ids.context_id, job.tracker)
        job.done = True

    def stepped(self, status: str, message: str = "", current: int = 0, total: int = 0) -> None:
        """Importer-side progress callback for the shared acquire phase.

        The multi-dataset twin of
        :func:`~vtscore.datasets.load_pipeline._make_stepped_progress`: maps the
        status onto a load step, swaps the download gate for the embed gate on
        the first ``"embedding"`` status, and mirrors the update onto every
        live output's pacer.
        """
        self.check_cancelled()
        if status == "idle":
            return
        if status == "embedding" and self.controller is not None and self.controller.held != "embed":
            self.controller.swap_to_embed()
        step = _STATUS_TO_STEP.get(status)
        self.update(status, message, current, total, step=step, total_steps=_TOTAL_LOAD_STEPS)


def _build_jobs(
    importer,
    field_values: dict[str, Any],
    outputs: list[OutputSpec],
    *,
    base_name: str,
    created_by: str,
    build_projection: bool,
    merge_near_duplicates: bool,
    ingest_started_at: float,
    origin_for: Callable[[OutputSpec, dict[str, Any]], dict[str, Any]] | None,
) -> list[_OutputJob]:
    """Mint one loading task, context and pacer per output; nothing runs yet.

    Also mints the run's import group: one id stamped on every output's spec,
    so the registry entries they land as can name each other.
    """
    import_group = uuid4().hex
    jobs: list[_OutputJob] = []
    for output in outputs:
        narrowed = output.narrow(field_values)
        name = output_dataset_name(base_name, output)
        chain_steps = append_cleaner_steps(_parse_chain_field(output.clipper_chain), output.cleaners)
        embedders = list(output.embedders) if output.embedders else None
        # The primary leads the embed order (it becomes each media's recorded
        # primary); defensive, as in the single-dataset path.
        if embedders and output.embedder and output.embedder not in embedders:
            embedders = [output.embedder, *embedders]
        if getattr(importer, "handles_own_clipping", False):
            if output.clipper_params is not None:
                narrowed["clipper_params"] = output.clipper_params
            if chain_steps is not None:
                narrowed["clipper_chain"] = chain_steps
            clipper, clipper_params, pipeline_chain = "", None, None
        else:
            clipper, clipper_params, pipeline_chain = output.clipper, output.clipper_params, chain_steps
        origin = origin_for(output, narrowed) if origin_for is not None else importer.build_origin(narrowed)

        # Remember the pick per media type, as the single path does, so the
        # next Add Dataset form pre-selects it.  Read off the pipeline module
        # at call time: the app installs the hook after import.
        hook = _pipeline._last_embedder_persistence_hook
        if output.media_type and output.embedder and hook is not None:
            try:
                hook(output.media_type, output.embedder)
            except Exception:
                pass

        task = _start_import_task(
            prefix="_loading_",
            display_name=name,
            total_steps=_TOTAL_LOAD_STEPS,
            media_type=output.media_type,
            embedder=output.embedder,
            weights=load_step_weights(output.media_type, embedder=output.embedder),
            created_by=created_by,
        )
        task.tracker.update("loading", "Preparing dataset...", step=1, total_steps=_TOTAL_LOAD_STEPS)
        ctx = DatasetContext(task.task_id)
        ctx.merge_near_duplicates = merge_near_duplicates
        cost_terms, calibrated = load_cost_terms(output.media_type, embedder=output.embedder)
        pacer = AdaptiveLoadPacer(task.tracker, cost_terms, calibrated=calibrated)
        spec = _DatasetLoadSpec(
            task_id=task.task_id,
            origin=origin,
            name=name,
            clipper=clipper,
            clipper_params=clipper_params,
            chain_steps=pipeline_chain,
            embedder=output.embedder,
            embedders=embedders,
            created_by=created_by,
            media_type=output.media_type,
            build_projection=build_projection,
            ingest_started_at=ingest_started_at,
            import_group=import_group,
            output_category=output.category,
        )
        label = output.category_label()
        jobs.append(
            _OutputJob(
                output=output,
                narrowed=narrowed,
                label=label,
                spec=spec,
                task=task,
                ctx=ctx,
                pacer=pacer,
                ids=_LoadIds(context_id=task.task_id),
            )
        )
    return jobs


def _job_for(jobs: list[_OutputJob], output: OutputSpec) -> _OutputJob | None:
    """The job an importer's yielded *output* belongs to.

    Importers are asked to yield the very ``OutputSpec`` objects they were
    given, so identity is the lookup; an importer that copied them is matched
    by equality instead, first match wins.
    """
    for job in jobs:
        if job.output is output:
            return job
    for job in jobs:
        if job.output == output:
            return job
    return None


def _drain_importer(
    importer, field_values: dict[str, Any], jobs: list[_OutputJob], fanout: _FanoutTracker, thin: bool
) -> None:
    """Run the importer once and route what it yields into each output's context.

    Uses :meth:`~ImporterBase.run_outputs_chunked` when the importer supports
    chunked loading (renumbering each chunk into the output's dataset as it
    arrives, as the single path's chunk consumer does) and
    :meth:`~ImporterBase.run_outputs` otherwise.  Media yielded for an output
    that has since been cancelled are dropped on the floor.
    """
    outputs = [job.output for job in jobs if not job.done]
    if getattr(importer, "supports_chunked", False):
        chunk_size = min(auto_chunk_size(job.output.media_type) for job in jobs if not job.done)
        stream = importer.run_outputs_chunked(field_values, outputs, chunk_size, thin=thin)
    else:
        stream = importer.run_outputs(field_values, outputs, thin=thin)

    set_thread_progress(fanout.stepped)
    try:
        for output, chunk in stream:
            fanout.check_cancelled()
            job = _job_for(jobs, output)
            if job is None:
                raise ValueError(
                    f"{type(importer).__name__} yielded media for an output it was not asked for: {output!r}"
                )
            if job.done or not chunk:
                continue
            consume_chunks_into(job.ctx.medias, [chunk])
    finally:
        clear_thread_progress()


def _spawn_multi_worker(jobs: list[_OutputJob], request_user: str, body: Callable[[], None]) -> list[str]:
    """Run *body* on one daemon thread that owns every job's task; return the task ids.

    The multi-dataset twin of
    :func:`~vtscore.datasets.load_pipeline._spawn_import_worker`: the same
    user replay and progress clean-up, but the one thread is registered as the
    worker of *every* output's task (so a cancel on any row can tell "still
    stopping" from "nobody here"), and every task is marked finished once the
    thread has fully unwound.
    """

    def run() -> None:
        from vtscore.state.current_user import thread_user  # noqa: PLC0415

        try:
            with thread_user(request_user):
                body()
        finally:
            clear_thread_progress()
            for job in jobs:
                loading_tasks.mark_finished(job.task.task_id)

    worker = threading.Thread(target=run, daemon=True)
    for job in jobs:
        loading_tasks.set_worker(job.task.task_id, worker)
    worker.start()
    return [job.task.task_id for job in jobs]


@dataclass
class _SharedOptions:
    """The request-level half of a multi-dataset import: what every output shares."""

    thin: bool
    build_projection: bool
    merge_near_duplicates: bool
    base_name: str


def _pop_shared_options(importer, field_values: dict[str, Any]) -> _SharedOptions:
    """Strip the shared toggles (and any stray per-dataset keys) off *field_values*.

    Mirrors the pops at the top of
    :func:`~vtscore.datasets.load_pipeline._run_importer_in_background`.  The
    base name is resolved before ``dataset_name`` goes: what the user typed,
    else the importer's own derivation from its URL / path / upload fields;
    each output appends its category label (see
    :func:`~vtscore.datasets.importers.base.outputs.output_dataset_name`).
    """
    thin = _parse_bool(field_values.pop("reference_files", None))
    build_projection = _parse_bool(field_values.pop("build_projection", None))
    merge_near_duplicates = _parse_bool(field_values.pop("merge_near_duplicates", None))
    field_values.pop("outputs", None)
    for key in _PER_OUTPUT_KEYS:
        field_values.pop(key, None)
    base_name = importer.resolve_display_name(field_values)
    field_values.pop("dataset_name", None)
    return _SharedOptions(thin, build_projection, merge_near_duplicates, base_name)


def _acquire_all(
    importer,
    field_values: dict[str, Any],
    jobs: list[_OutputJob],
    fanout: _FanoutTracker,
    controller: _LoadGateController,
    thin: bool,
    cleanup: Callable[[], None] | None,
) -> None:
    """The shared phase: take the download gate, run the importer once, swap to the embed gate.

    A failure here (or every row being cancelled) fails each output still in
    flight the same way, so the caller finds nothing live to finalize.
    """
    try:
        controller.acquire_download()
        for job in fanout.live:
            job.pacer.update("loading", "Preparing new dataset…", 0, 0, step=1, total_steps=_TOTAL_LOAD_STEPS)
            register_context(job.ctx)
        gc.collect()
        try:
            _drain_importer(importer, field_values, jobs, fanout, thin)
        finally:
            if cleanup is not None:
                try:
                    cleanup()
                except Exception:
                    traceback.print_exc()
        fanout.check_cancelled()
        # Post-import stages are CPU/GPU-bound; gate them on the embed
        # semaphore.  No-op if the importer already swapped mid-run.
        controller.swap_to_embed()
    except Exception as exc:
        for job in fanout.live:
            fanout.retire(job, exc)


def _finalize_each(jobs: list[_OutputJob]) -> None:
    """Run the post-import stages on every live job in turn, each failure its own.

    The rows still queued say which dataset they are waiting behind, so a user
    watching three rows sees one of them working and the others explain the
    pause.
    """
    from vtscore.state.core import thread_dataset_context  # noqa: PLC0415

    pending = [job for job in jobs if not job.done]
    for index, job in enumerate(pending):
        for later in pending[index + 1 :]:
            if not later.done:
                later.tracker.update(
                    "loading", f"Waiting for {job.spec.name} to finish…", 0, 0, step=2, total_steps=_TOTAL_LOAD_STEPS
                )
        with thread_dataset_context(job.ctx):
            try:
                job.tracker.check_cancelled()
                if not job.ctx.medias:
                    raise ValueError(f"Import produced no {job.label} media.")
                _finish_dataset_load(job.ctx, job.tracker, job.pacer, job.spec, job.ids)
            except Exception as exc:
                _handle_load_failure(exc, job.ids.context_id, job.tracker, registry_entry_id=job.ids.registry_entry_id)
            finally:
                job.done = True


def _report_each(
    jobs: list[_OutputJob],
    request_user: str,
    post_load: Callable[[DatasetContext], None] | None,
    on_finished: Callable[[DatasetImported], None] | None,
) -> None:
    """Fire the per-dataset hooks: AutoFind first, then the import-finished event, per output."""
    for job in jobs:
        _run_post_load(post_load, job.ctx, job.tracker)
        _report_finished(
            on_finished,
            job.ctx,
            job.tracker,
            dataset_id=job.ids.context_id,
            name=job.spec.name,
            user=request_user,
            media_type=job.output.media_type,
            origin=job.spec.origin,
        )


def _run_multi_output_load_in_background(
    importer,
    field_values: dict[str, Any],
    outputs: list[OutputSpec],
    *,
    post_load: Callable[[DatasetContext], None] | None = None,
    on_finished: Callable[[DatasetImported], None] | None = None,
    origin_for: Callable[[OutputSpec, dict[str, Any]], dict[str, Any]] | None = None,
    cleanup: Callable[[], None] | None = None,
) -> list[str]:
    """Run *importer* once and save one dataset per entry of *outputs*, in the background.

    The multi-dataset twin of
    :func:`~vtscore.datasets.load_pipeline._run_importer_in_background`.
    *field_values* are the shared form values (the importer's own fields plus
    the shared toggles); each :class:`OutputSpec` carries one dataset's media
    type, source rows and per-dataset load options.  Returns one task id per
    output, in the outputs' order; each names that dataset's row on the
    ``loading-tasks`` channel and can be cancelled on its own.

    *post_load* and *on_finished* are called per output on the terms
    :func:`_run_origin_load_in_background` documents: *post_load* after each
    dataset that saved, *on_finished* once per dataset that saved or failed
    (never for a cancelled one).

    *origin_for*, when given, builds an output's origin from ``(output,
    narrowed field values)`` in place of the importer's own
    :meth:`~ImporterBase.build_origin`; the browser-upload routes use it to
    record their synthetic ``<browser_upload>`` origin.  *cleanup*, when given,
    runs once the importer has finished (or failed), before any dataset is
    finalized — the moment a staged upload directory can go.
    """
    from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415
    from vtscore.state.current_user import get_current_user  # noqa: PLC0415

    if not outputs:
        raise ValueError("A multi-dataset import needs at least one output.")

    field_values = wrap_cli_file_fields(importer.fields, field_values)
    created_by = get_current_user()
    shared = _pop_shared_options(importer, field_values)
    jobs = _build_jobs(
        importer,
        field_values,
        outputs,
        base_name=shared.base_name,
        created_by=created_by,
        build_projection=shared.build_projection,
        merge_near_duplicates=shared.merge_near_duplicates,
        ingest_started_at=time.time(),
        origin_for=origin_for,
    )
    request_user = jobs[0].task.request_user

    def load_task() -> None:
        fanout = _FanoutTracker(jobs)
        controller = _LoadGateController(fanout, _TOTAL_LOAD_STEPS)
        fanout.controller = controller
        try:
            _acquire_all(importer, field_values, jobs, fanout, controller, shared.thin, cleanup)
            _finalize_each(jobs)
        finally:
            controller.release()
            for job in jobs:
                _park_load_terminal(job.tracker, len(job.ctx.medias))
        # Outside the ``finally``, as in the single path: a job that failed has
        # its error on the tracker, so the hooks see each outcome.
        _report_each(jobs, request_user, post_load, on_finished)

    return _spawn_multi_worker(jobs, request_user, load_task)
