"""Staging imports for the combine flow: an importer run saved to a temporary pkl, not registered.

:func:`_stage_importer_in_background` runs an importer and the embed stage on
a worker thread, under the same download / embed gates and on the same task
harness as a dataset load (both from :mod:`vtscore.datasets.load_workers`),
then writes the medias to a ``.pkl`` under :data:`STAGING_DIR` and publishes
its path as the task's terminal ``staging_result``.  It touches no dataset
context and no registry entry; the combine route merges the staged files.
"""

from __future__ import annotations

import gc
from uuid import uuid4

from vtscore.config import DATA_DIR
from vtscore.concurrency.progress import set_thread_progress
from vtscore.datasets import export_dataset_to_file
from vtscore.datasets.load_workers import (
    _LoadGateController,
    _describe_source,
    _failure_message,
    _normalize_media_type,
    _spawn_import_worker,
    _start_import_task,
)
from vtscore.datasets.loader import apply_custom_metadata_md5
from vtscore.datasets.stages.embedding import embed_missing, _check_named_vector_widths
from vtscore.timing import step_weights


# ---------------------------------------------------------------------------
# Staging – import datasets to temporary pkl files for the combine flow
# ---------------------------------------------------------------------------

STAGING_DIR = DATA_DIR / "staging"

#: Task family and step count for a staging import (see vtscore.timing.tasks).
_STAGE_TASK = "dataset_stage"
_TOTAL_STAGE_STEPS = 3  # acquire, embed, serialize

#: Maps the status strings an importer emits onto ``dataset_stage``'s steps, the
#: way :data:`~vtscore.datasets.stages._common._STATUS_TO_STEP` does for a load.
#: Staging folds the load's four steps into three, so every pre-embed status
#: shares the acquire slice.
#:
#: This map is what makes the ``embed`` step measure an embed. An importer that
#: embeds *inside* ``run()`` — every demo source does, and the demo importer is
#: what the combine flow stages — used to report stepless, so the tracker kept
#: step 1 and the whole embedding leg was recorded as ``acquire``. #3521 §5
#: fitted that step at ``b = 0.0136 s/item, r² 0.9995`` — an embed curve under
#: the wrong name — beside an ``embed`` at ``b = 7.2e-07``, on a sweep that
#: cleared the embeddings cache before every rep. Clearing the cache moved
#: 11–40 s of real embedding into the run and ``embed`` did not move, because
#: the boundary, not the cache, was what put it there (#3593).
_STAGE_STATUS_TO_STEP = {
    "downloading": 1,
    "extracting": 1,
    "loading": 1,
    "converting": 1,
    "embedding": 2,
}


def _make_staging_progress(controller: _LoadGateController, tracker):
    """Build the importer-side progress callback for a staging run.

    Mirrors :func:`~vtscore.datasets.load_workers._make_stepped_progress`: it
    stamps the step each status belongs to — so the bar paces against the
    phase that actually ran — and
    swaps the download gate for the embed gate on the first
    ``"embedding"``, so a queued import can start fetching while this one holds
    only the embed slot. Staging used to swap only after ``run()`` returned,
    which meant a demo staging did its embedding under the *download* gate.

    Two deliberate differences from the load pipeline's version, both because
    staging has no :class:`~vtscore.datasets.stages._common.AdaptiveLoadPacer`
    between the importer and the tracker:

    - a status the map does not know leaves ``step`` **unset** rather than
      passing ``None``, so the tracker keeps the step it was last told instead
      of nulling the whole-job fraction for that update;
    - the step never moves backwards. A demo's clipper reports its clip
      embedding under a plain ``"loading"`` status, which would otherwise walk
      the bar back to acquire after the embed slice had already started.
    """

    def stepped(status: str, message: str = "", current: int = 0, total: int = 0, **kwargs) -> None:
        # An importer signalling *its* completion is not the staging job's:
        # serialization still has to run, and the terminal update is the one at
        # the bottom of ``stage_task``. A failure riding along is another
        # matter and is forwarded. (``load_demo_dataset`` ends with exactly such
        # an ``"idle"``, which used to park the whole staging task at idle
        # mid-run.)
        if status == "idle" and "error" not in kwargs:
            return
        if status == "embedding" and controller.held != "embed":
            controller.swap_to_embed()
        if "step" not in kwargs:
            step = _STAGE_STATUS_TO_STEP.get(status)
            if step is not None and step >= (tracker.get().get("step") or 0):
                kwargs["step"] = step
        kwargs.setdefault("total_steps", _TOTAL_STAGE_STEPS)
        tracker.update(status, message, current, total, **kwargs)

    return stepped


def _stage_importer_in_background(importer, field_values: dict, label: str = "") -> str:
    """Run *importer*.run() in a daemon thread, saving the result to a staging pkl.

    Unlike :func:`~vtscore.datasets.load_pipeline._run_importer_in_background`,
    this does **not** modify the global ``medias`` dict and does **not**
    register a dataset.  Instead it writes a
    temporary ``.pkl`` file to :data:`STAGING_DIR` and publishes the terminal
    ``staging_result`` on this operation's own per-task progress tracker.

    Each call gets a dedicated :class:`ProgressTracker` (via ``loading_tasks``),
    keyed by the returned ``task_id``, mirroring
    :func:`~vtscore.datasets.load_origin._run_origin_load_in_background`, so
    two concurrent stagings never interleave one channel and their terminal
    ``staging_result``s cannot
    collide.  The shared setup/teardown around the body — tracker, user
    replay, worker registration — is
    :func:`_start_import_task` / :func:`_spawn_import_worker`, the same harness
    the load pipeline runs on.

    Staging is gated exactly like a load: the importer runs under the download
    gate and the embed under the embed gate.  It has the same appetite for
    bandwidth, RAM, and GPU as a regular import, and the combine flow stages
    several datasets at once — so without this, N stagings ran fully in
    parallel with each other *and* with gated loads, defeating the
    ``max_concurrent_dataset_downloads`` / ``max_concurrent_dataset_embeddings``
    limits whose whole purpose is bounding that pressure.

    Returns the ``task_id`` a caller can poll (via the ``loading-tasks`` SSE
    channel) for progress and the final ``staging_result``.
    """
    from vtscore.plugins.uploads import wrap_cli_file_fields  # noqa: PLC0415

    field_values = wrap_cli_file_fields(importer.fields, field_values)

    staged_media_type = _normalize_media_type(field_values.get("media_type", ""))
    staged_embedder = field_values.get("embedder", "") or ""
    # Staging reports the same step structure every other long-running family
    # does, which is what earns it a whole-job bar.
    # The boundaries stamped below mark the steps this function drives; the
    # importer's own progress calls are mapped onto the same three steps by
    # ``_make_staging_progress``, because an importer that embeds inside
    # ``run()`` is embedding whatever step number the last stamp left behind
    # (#3593).
    task = _start_import_task(
        prefix="_staging_",
        display_name=label or importer.resolve_display_name(field_values),
        total_steps=_TOTAL_STAGE_STEPS,
        media_type=staged_media_type,
        embedder=staged_embedder,
        weights=step_weights(_STAGE_TASK, media_type=staged_media_type, embedder=staged_embedder),
        extra_fields={"staging_result": None},
        source=_describe_source(lambda: importer.build_origin(field_values)),
    )
    tracker = task.tracker
    tracker.update("loading", "Preparing dataset…", 0, 0, step=1, total_steps=_TOTAL_STAGE_STEPS)

    def stage_task():
        controller = _LoadGateController(tracker, task.total_steps)
        # Route the importer's own progress calls (and embedding progress)
        # into this task's tracker instead of the global singleton, mapped onto
        # this task's steps on the way.
        set_thread_progress(_make_staging_progress(controller, tracker))
        try:
            controller.acquire_download()
            temp_medias: dict = {}
            importer.run(field_values, temp_medias)
            apply_custom_metadata_md5(temp_medias)
            # Backstop for the handoff ``_make_staging_progress`` normally makes
            # on the importer's first ``"embedding"`` status: an importer that
            # embeds nothing itself (every non-demo source) never fires one, and
            # would otherwise hold the download gate through the embed below.
            # No-op when the swap already happened mid-run.
            controller.swap_to_embed()
            tracker.update("embedding", "Embedding…", 0, 0, step=2, total_steps=_TOTAL_STAGE_STEPS)
            _check_named_vector_widths(temp_medias)
            embed_missing(temp_medias, field_values.get("embedder", "") or "", on_progress=tracker.update)
            from vtscore.embedding.media_vectors import media_embedding  # noqa: PLC0415

            temp_medias = {mid: m for mid, m in temp_medias.items() if media_embedding(m) is not None}

            if not temp_medias:
                tracker.update("idle", "", 0, 0, error="Import produced no medias.")
                return

            first = next(iter(temp_medias.values()))
            media_type = first.get("media_type", "audio")
            count = len(temp_medias)
            name = label or importer.resolve_display_name(field_values)

            tracker.update("loading", "Writing staged file…", 0, 0, step=3, total_steps=_TOTAL_STAGE_STEPS)
            data_bytes = export_dataset_to_file(temp_medias)
            del temp_medias
            gc.collect()

            STAGING_DIR.mkdir(parents=True, exist_ok=True)
            staging_path = STAGING_DIR / f"stage_{uuid4().hex}.pkl"
            staging_path.write_bytes(data_bytes)
            del data_bytes
            gc.collect()

            tracker.update(
                "idle",
                f"Staged: {name} ({count} medias)",
                100,
                100,
                step=_TOTAL_STAGE_STEPS,
                total_steps=_TOTAL_STAGE_STEPS,
                staging_result={"path": str(staging_path), "name": name, "count": count, "media_type": media_type},
            )
        except Exception as exc:
            gc.collect()
            tracker.update("idle", "", 0, 0, error=_failure_message(exc, "Unknown error during staging"))
        finally:
            controller.release()

    return _spawn_import_worker(task, stage_task)
