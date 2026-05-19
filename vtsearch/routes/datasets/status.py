"""Dataset status / cancellation endpoints.

Per-operation progress is streamed via the unified ``/api/events`` SSE
endpoint — see ``feature-brainstorm.md`` §12.5 and
``vtsearch/routes/events.py``.

Migrated to ``flask_smorest`` so these routes appear in
``/api/openapi.json``. See ``docs/plans/openapi-schema.md``. Schema-level
validation is unused (all bodies are empty); handler-level rejects
(unknown task id) surface as 404 with the standard ``message`` envelope.
"""

from flask_smorest import Blueprint, abort

from vtscore.concurrency.progress import (
    cancel_dataset_progress,
    loading_tasks as _loading_tasks,
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
        media_type = next(iter(snap.values())).get("type", "audio")
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
def cancel_dataset_load():
    """Cancel dataset load/import operations.

    Cancels all active loading tasks and the legacy global tracker.
    """
    _loading_tasks.cancel_all()
    cancel_dataset_progress()
    return {"ok": True}


@datasets_status_bp.route("/api/dataset/cancel/<task_id>", methods=["POST"])
@datasets_status_bp.response(200, CancelDatasetLoadResponseSchema)
@datasets_status_bp.alt_response(404, description="No active task with the supplied id.")
def cancel_dataset_load_task(task_id: str):
    """Cancel a specific dataset loading task."""
    ok = _loading_tasks.cancel_task(task_id)
    if not ok:
        abort(404, message="Task not found")
    return {"ok": True}
