"""SSE endpoint streaming all progress channels to connected clients.

Replaces per-tracker REST polling (``/api/dataset/progress``,
``/api/sort/progress``, etc.) with a single push channel.
"""

from __future__ import annotations

from typing import Generator

from flask import Blueprint, Response, jsonify, stream_with_context

from vtscore.concurrency.events import acquire_sse_slot, release_sse_slot, stream_progress_events
from vtsearch.hooks import state_sync_exempt

events_bp = Blueprint("events", __name__)


@events_bp.route("/api/events")
@state_sync_exempt
def progress_events() -> Response:
    """Stream progress updates for every channel as Server-Sent Events.

    The client connects with ``new EventSource('/api/events')`` and listens
    for ``server`` (per-connect identity frame carrying ``boot_id``),
    ``dataset``, ``sort``, ``find``, ``eval``, ``loading-tasks``,
    ``detector-loading-tasks``, and ``heartbeat`` (periodic liveness ping)
    events. The first frame on every channel is the current snapshot; clients
    do not need a separate REST call to bootstrap state.

    Each open connection pins a ``gthread`` worker thread for its lifetime,
    so connections are capped (``MAX_SSE_CONNECTIONS``) to leave headroom for
    ordinary requests; once the cap is hit new connects get a 503 instead of
    starving the pool. ``EventSource`` treats a non-2xx response as a fatal
    error and stops auto-reconnecting, so the frontend's own ``onerror``
    handler schedules a manual reconnect (see ``progress-events.service.ts``);
    it also probes ``/healthz`` to classify the error, so a cap rejection —
    proof the backend is alive — never counts toward the offline circuit
    breaker (#2816). The dev server (``app.run(threaded=True)``) has no
    thread pool to protect and runs uncapped (``uncap_sse_connections``).

    ``@state_sync_exempt`` keeps the (re)connect off the global
    ``_state_lock``: the stream is read-only and subscribes only to the
    *global* progress trackers, and gating it on the very lock a long
    Find/load contends would stop progress events reaching the client
    during exactly the operations the progress bar exists to report.
    """
    if not acquire_sse_slot():
        response = jsonify({"message": "Too many live event streams open; retry shortly."})
        response.status_code = 503
        response.headers["Retry-After"] = "5"
        return response

    def generate() -> Generator[str, None, None]:
        try:
            yield from stream_progress_events()
        finally:
            release_sse_slot()

    response = Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
    )
    # Long-lived stream: set our own Cache-Control so the global @after_request
    # hook defers instead of stamping no-store, and disable proxy buffering so
    # events flush immediately instead of pooling behind nginx / gunicorn.
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response
