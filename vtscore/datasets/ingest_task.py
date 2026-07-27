"""Run :func:`~vtscore.datasets.ingest.ingest_missing_medias` as a background task.

Ingesting a labelset's missing media fetches and embeds one file per label
from its origin, so for a labelset of any real size it is far too slow to sit
inside a request handler: the caller hangs with no progress and a long enough
ingest trips a proxy/gateway timeout with the import already half-applied
(issue #2703).

:func:`start_ingest_task` moves that work onto the same
``detector_loading_tasks`` tracker the detector load already uses, so it
streams progress over the existing ``/api/events`` SSE feed and the caller
gets a task id back immediately.  Both import paths use it — the
detector-from-labelset route and the in-session label importer — so neither
one can reintroduce the blocking shape.

The one app-tier dependency, ``vtsearch.threading.spawn`` (which replays the
request's user / dataset / detector thread-locals into the worker), is
injected by the caller so this module stays Flask-free, matching
:mod:`vtscore.detectors.embedder_sync`.
"""

from __future__ import annotations

from typing import Any, Callable

# A ``spawn``-style callable: ``spawn(target, *, name=...) -> Thread``.
SpawnFn = Callable[..., Any]

#: Called on the worker thread once the ingest itself has finished, with the
#: number of medias ingested.  Whatever dict it returns is merged into the
#: task's terminal ``ingest_result`` payload, so a caller can publish the
#: downstream numbers (labels re-applied, entries still unresolved) that only
#: become known after the ingest.
AfterIngest = Callable[[int], "dict[str, Any] | None"]


def start_ingest_task(
    entries: list[dict[str, Any]],
    medias: Any,
    *,
    task_id: str,
    name: str,
    spawn: SpawnFn,
    detector_id: str = "",
    media_type: str = "",
    after_ingest: AfterIngest | None = None,
) -> str:
    """Ingest *entries* into *medias* on a background thread; return *task_id*.

    Registers a ``detector_loading_tasks`` entry **before** returning, so the
    caller can hand the id straight back to the client and the SSE feed has
    already published the row by the time the response lands.

    The task publishes its terminal numbers as an ``ingest_result`` dict on
    the tracker (``{"ingested": n, ...}`` plus anything *after_ingest*
    contributes), mirroring how combine-datasets staging publishes
    ``staging_result``.  A cancel surfaces as ``error="Cancelled"``; any other
    failure surfaces its message, matching the detector-load task.

    *medias* is the live dataset dict to extend, usually the request-scoped
    ``medias`` proxy — ``spawn`` replays the dataset context, so the proxy
    resolves to the same dataset inside the worker.
    """
    from vtscore.concurrency.progress import detector_loading_tasks

    tracker = detector_loading_tasks.create_task(
        task_id,
        name,
        detector_id=detector_id,
        media_type=media_type,
        extra_fields={"ingest_result": None},
    )
    base_msg = f"Fetching {len(entries)} missing media…"
    tracker.update("loading", base_msg, 0, len(entries), step=1, total_steps=1)

    def ingest_task() -> None:
        from vtscore.concurrency.progress import (
            CancelledError,
            clear_thread_progress,
            set_thread_progress,
        )
        from vtscore.datasets.ingest import ingest_missing_medias

        def _on_progress(status: str, message: str, current: int, total: int) -> None:
            tracker.check_cancelled()
            tracker.update("loading", message or base_msg, current, total, step=1, total_steps=1)

        try:
            # Bind the thread-progress hook too: the legacy ingest path runs a
            # whole dataset importer, and importers report through
            # ``get_thread_progress()``.  Without this their frames would land
            # on the global dataset tracker and light up an unrelated bar.
            set_thread_progress(_on_progress)
            try:
                ingested = ingest_missing_medias(entries, medias, on_progress=_on_progress)
                result: dict[str, Any] = {"ingested": ingested}
                if after_ingest is not None:
                    result.update(after_ingest(ingested) or {})
            finally:
                clear_thread_progress()
            tracker.update("idle", "", 0, 0, ingest_result=result, step=None, total_steps=None)
        except CancelledError:
            tracker.update("idle", "", 0, 0, error="Cancelled", step=None, total_steps=None)
        except Exception as e:
            import traceback as _tb

            _tb.print_exc()
            tracker.update(
                "idle",
                "",
                0,
                0,
                error=str(e) or repr(e) or "Unknown error ingesting missing media",
                step=None,
                total_steps=None,
            )
        finally:
            detector_loading_tasks.mark_finished(task_id)

    spawn(ingest_task, name=f"labelset-ingest-{task_id[-12:]}")
    return task_id
