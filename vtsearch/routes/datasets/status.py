"""Dataset status / cancellation endpoints.

Per-operation progress is streamed via the unified ``/api/events`` SSE
endpoint; see ``vtsearch/routes/events.py``.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. Schema-level
validation is unused (all bodies are empty); handler-level rejects
(unknown task id) surface as 404 with the standard ``message`` envelope.
"""

from flask_smorest import Blueprint, abort

from vtscore.concurrency.progress import (
    cancel_dataset_progress,
    cancel_dataset_task,
)
from vtsearch.schemas.datasets import (
    CancelDatasetLoadResponseSchema,
    DatasetStatusResponseSchema,
)
from vtsearch.state import (
    bad_votes,
    get_dataset_display_name,
    get_dupe_count,
    good_votes,
    snapshot_medias,
)

datasets_status_bp = Blueprint(
    "datasets_status",
    __name__,
    description="Dataset status (loaded / num medias / display name) and load-cancellation.",
)


@datasets_status_bp.route("/api/dataset/status")
@datasets_status_bp.response(200, DatasetStatusResponseSchema)
def dataset_status():
    """Return the current dataset status."""
    snap = snapshot_medias()
    media_type = None
    if snap:
        media_type = next(iter(snap.values())).get("media_type", "audio")
    return {
        "loaded": len(snap) > 0,
        "num_medias": len(snap),
        "has_votes": len(good_votes) + len(bad_votes) > 0,
        "media_type": media_type,
        "display_name": get_dataset_display_name(),
        "num_dupes": get_dupe_count(),
    }


@datasets_status_bp.route("/api/dataset/cancel", methods=["POST"])
@datasets_status_bp.response(200, CancelDatasetLoadResponseSchema)
@datasets_status_bp.alt_response(
    409,
    schema=CancelDatasetLoadResponseSchema,
    description="The cancel reached nothing: no operation was running, or the progress it claimed was stale.",
)
def cancel_dataset_load():
    """Cancel dataset load/import operations.

    Cancels all active loading tasks and the legacy global tracker, then waits
    briefly for one of them to act on the flag.  Cancellation is cooperative,
    so a flag set with no worker left to observe it stops nothing; answering
    ``ok`` in that case is what let a *finished* import keep looking like a
    wedged one (#3167).  A cancel that reached nothing answers ``409``, and any
    stale progress it found is cleared on the way out.
    """
    report = cancel_dataset_progress()
    return (report, 409) if not report["ok"] else report


@datasets_status_bp.route("/api/dataset/cancel/<task_id>", methods=["POST"])
@datasets_status_bp.response(200, CancelDatasetLoadResponseSchema)
@datasets_status_bp.alt_response(404, description="No task with the supplied id.")
@datasets_status_bp.alt_response(
    409,
    schema=CancelDatasetLoadResponseSchema,
    description="The task was not running, or its progress was stale (no live worker).",
)
def cancel_dataset_load_task(task_id: str):
    """Cancel a specific dataset loading task.

    Same honesty contract as ``POST /api/dataset/cancel``, narrowed to one
    task.
    """
    report = cancel_dataset_task(task_id)
    if report is None:
        abort(404, message="Task not found")
    return (report, 409) if not report["ok"] else report
