"""Server-Sent Events stream for progress channels.

Maintains a registry of named channels (each backed by a
:class:`ProgressTracker` or :class:`LoadingTasksTracker`) and produces SSE
event strings that ``/api/events`` streams to connected clients.

Replaces the per-tracker REST polling endpoints (``/api/dataset/progress``,
``/api/sort/progress``, etc.) with a single push channel — see
``docs/plans/feature-brainstorm.md`` §12.5.
"""

from __future__ import annotations

import json
import queue
import time
from typing import Any, Callable, Iterator

from vtsearch.concurrency.progress import (
    LoadingTasksTracker,
    ProgressTracker,
    dataset_progress,
    detector_loading_tasks,
    eval_progress,
    find_progress,
    loading_tasks,
    sort_progress,
)

#: Snapshot interval used as a fallback when no events arrive — also doubles
#: as the SSE heartbeat (comment line) cadence so proxies / load balancers
#: don't drop idle connections.
HEARTBEAT_SECONDS = 15.0

#: Single-channel trackers (key = SSE event name).
_TRACKER_CHANNELS: dict[str, ProgressTracker] = {
    "dataset": dataset_progress,
    "sort": sort_progress,
    "find": find_progress,
    "eval": eval_progress,
}

#: Task-list trackers (key = SSE event name).
_TASK_CHANNELS: dict[str, LoadingTasksTracker] = {
    "loading-tasks": loading_tasks,
    "detector-loading-tasks": detector_loading_tasks,
}


def _format_sse(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def initial_snapshot() -> list[str]:
    """Return the SSE frames a freshly-connected client should receive first."""
    frames: list[str] = []
    for name, tracker in _TRACKER_CHANNELS.items():
        frames.append(_format_sse(name, tracker.get()))
    for name, tasks_tracker in _TASK_CHANNELS.items():
        frames.append(_format_sse(name, tasks_tracker.list_tasks()))
    return frames


def stream_progress_events(
    *, heartbeat_seconds: float = HEARTBEAT_SECONDS, max_queue: int = 1024
) -> Iterator[str]:
    """Yield SSE-formatted strings for every progress channel until disconnect.

    Each connected client gets a private bounded queue; tracker subscriptions
    push snapshots into the queue, and the generator drains it. On client
    disconnect (the generator is closed by Flask/Werkzeug) the ``finally``
    block unsubscribes so we don't leak callbacks.
    """
    q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=max_queue)

    subscriptions: list[tuple[Any, Callable[..., None]]] = []

    def _make_tracker_handler(channel: str) -> Callable[[dict[str, Any]], None]:
        def handler(snapshot: dict[str, Any]) -> None:
            try:
                q.put_nowait((channel, snapshot))
            except queue.Full:
                pass

        return handler

    def _make_tasks_handler(channel: str) -> Callable[[list[dict[str, Any]]], None]:
        def handler(snapshot: list[dict[str, Any]]) -> None:
            try:
                q.put_nowait((channel, snapshot))
            except queue.Full:
                pass

        return handler

    for name, tracker in _TRACKER_CHANNELS.items():
        h = _make_tracker_handler(name)
        tracker.subscribe(h)
        subscriptions.append((tracker, h))
    for name, tasks_tracker in _TASK_CHANNELS.items():
        h = _make_tasks_handler(name)
        tasks_tracker.subscribe(h)
        subscriptions.append((tasks_tracker, h))

    try:
        # Send the SSE handshake comment immediately so the browser fires
        # `onopen` before any real event; some proxies buffer until the
        # first byte arrives.
        yield ": connected\n\n"

        for frame in initial_snapshot():
            yield frame

        last_heartbeat = time.monotonic()
        while True:
            timeout = max(0.0, heartbeat_seconds - (time.monotonic() - last_heartbeat))
            try:
                name, snapshot = q.get(timeout=timeout)
                yield _format_sse(name, snapshot)
            except queue.Empty:
                yield ": heartbeat\n\n"
                last_heartbeat = time.monotonic()
    finally:
        for subject, handler in subscriptions:
            subject.unsubscribe(handler)
