"""SSE endpoint streaming all progress channels to connected clients.

Replaces per-tracker REST polling (``/api/dataset/progress``,
``/api/sort/progress``, etc.) with a single push channel.
"""

from __future__ import annotations

from flask import Blueprint, Response, stream_with_context

from vtscore.concurrency.events import stream_progress_events

events_bp = Blueprint("events", __name__)


@events_bp.route("/api/events")
def progress_events() -> Response:
    """Stream progress updates for every channel as Server-Sent Events.

    The client connects with ``new EventSource('/api/events')`` and listens
    for ``dataset``, ``sort``, ``find``, ``eval``, ``loading-tasks``, and
    ``detector-loading-tasks`` events. The first frame on every channel is
    the current snapshot; clients do not need a separate REST call to
    bootstrap state.
    """
    response = Response(
        stream_with_context(stream_progress_events()),
        mimetype="text/event-stream",
    )
    # Long-lived; bypass the @after_request no-store header (already set,
    # which is fine for SSE) and disable proxy buffering so events flush
    # immediately instead of pooling behind nginx / gunicorn.
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response
