"""Dataset status / progress / cancellation endpoints."""

from flask import Blueprint, jsonify

from vtsearch.concurrency.progress import (
    cancel_dataset_progress,
    get_progress,
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


@datasets_status_bp.route("/api/dataset/progress")
def dataset_progress():
    """Return the current progress of long-running operations.

    For backward compatibility this returns a single progress dict.
    Prefers the first active loading task if any, otherwise falls back
    to the legacy global tracker (used by staging operations).
    """
    tasks = _loading_tasks.list_tasks()
    active = [t for t in tasks if t.get("status") != "idle"]
    if active:
        return jsonify(active[0])
    # Check if any just-finished task has an error to report
    errored = [t for t in tasks if t.get("error")]
    if errored:
        return jsonify(errored[0])
    return jsonify(get_progress())


@datasets_status_bp.route("/api/dataset/loading-tasks")
def dataset_loading_tasks():
    """Return all active dataset loading tasks with their progress."""
    return jsonify({"tasks": _loading_tasks.list_tasks()})


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
