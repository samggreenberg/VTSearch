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
    loading_tasks,
)
from vtsearch.auth import get_current_user
from vtsearch.schemas.datasets import (
    CancelDatasetLoadResponseSchema,
    DatasetRegistryOkResponseSchema,
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
    description="Dataset status (loaded / num medias / display name), load-cancellation, and dismissing finished tasks.",
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

    Cancels every active loading task, then waits briefly for one of them to
    act on the flag.  Cancellation is cooperative, so a flag set with no worker
    left to observe it stops nothing.  A cancel that reached nothing answers
    ``409``, and any stale progress it found is cleared on the way out.
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


@datasets_status_bp.route("/api/dataset/dismiss/<task_id>", methods=["POST"])
@datasets_status_bp.response(200, DatasetRegistryOkResponseSchema)
@datasets_status_bp.alt_response(404, description="No task with the supplied id that the caller can see.")
@datasets_status_bp.alt_response(409, description="The task is still running; cancel it instead.")
def dismiss_dataset_task(task_id: str):
    """Dismiss a finished dataset task, removing its row from the dashboard.

    A failed import stays on the ``loading-tasks`` channel until this is called
    (#4973), so the user has something to read after the failure toast is gone.
    Only the user who started the import can see that failure, so only they can
    dismiss it; to anyone else it does not exist.  Any other finished task can
    be dismissed early too; it would have lapsed on its own.
    """
    visible = loading_tasks.visible_to(loading_tasks.list_tasks(), get_current_user())
    task = next((t for t in visible if t["task_id"] == task_id), None)
    if task is None:
        abort(404, message="Task not found")
    if task.get("status") != "idle":
        abort(409, message="Task is still running; cancel it instead")
    loading_tasks.dismiss_task(task_id)
    return {"ok": True}
