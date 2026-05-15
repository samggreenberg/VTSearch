"""Dataset status / cancellation endpoints.

Per-operation progress is streamed via the unified ``/api/events`` SSE
endpoint — see ``feature-brainstorm.md`` §12.5 and
``vtsearch/routes/events.py``.
"""

from flask import Blueprint, jsonify

from vtsearch.concurrency.progress import (
    cancel_dataset_progress,
    loading_tasks as _loading_tasks,
)
from vtsearch.state import (
    bad_votes,
    get_dataset_display_name,
    get_dupe_count,
    good_votes,
    snapshot_medias,
)

datasets_status_bp = Blueprint("datasets_status", __name__)


@datasets_status_bp.route("/api/dataset/status")
def dataset_status():
    """Return the current dataset status."""
    snap = snapshot_medias()
    media_type = None
    if snap:
        media_type = next(iter(snap.values())).get("type", "audio")
    return jsonify(
        {
            "loaded": len(snap) > 0,
            "num_medias": len(snap),
            "has_votes": len(good_votes) + len(bad_votes) > 0,
            "media_type": media_type,
            "display_name": get_dataset_display_name(),
            "num_dupes": get_dupe_count(),
        }
    )


@datasets_status_bp.route("/api/dataset/cancel", methods=["POST"])
def cancel_dataset_load():
    """Cancel dataset load/import operations.

    Cancels all active loading tasks and the legacy global tracker.
    """
    _loading_tasks.cancel_all()
    cancel_dataset_progress()
    return jsonify({"ok": True})


@datasets_status_bp.route("/api/dataset/cancel/<task_id>", methods=["POST"])
def cancel_dataset_load_task(task_id: str):
    """Cancel a specific dataset loading task."""
    ok = _loading_tasks.cancel_task(task_id)
    if not ok:
        return jsonify({"error": "Task not found"}), 404
    return jsonify({"ok": True})
