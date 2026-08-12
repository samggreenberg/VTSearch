"""Server-Sent Events stream for progress channels.

Maintains a registry of named channels (each backed by a
:class:`ProgressTracker` or :class:`LoadingTasksTracker`) and produces SSE
event strings that ``/api/events`` streams to connected clients.

Replaces the per-tracker REST polling endpoints (``/api/dataset/progress``,
``/api/sort/progress``, etc.) with a single push channel.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import uuid
from typing import Any, Callable, Generator

from vtscore.concurrency.progress import (
    LoadingTasksTracker,
    ProgressTracker,
    dataset_progress,
    detector_loading_tasks,
    eval_progress,
    find_progress,
    loading_tasks,
    sort_progress,
)

#: SSE heartbeat cadence. Also drives the periodic re-emit of every
#: channel's snapshot: task channels so finished tasks vanish from clients
#: once they pass the ``LoadingTasksTracker`` stale-prune window without us
#: having to schedule a background timer for every finish, and tracker
#: channels so a client that dropped a terminal frame on queue overflow
#: recovers within one heartbeat (#2960).
#:
#: The heartbeat is the frontend's authoritative "backend is alive" signal
#: (see ``ConnectionStateService``): as long as a frame — heartbeat or real
#: progress — keeps arriving, the client's offline circuit breaker stays
#: online even while a long, busy operation (dataset ingest, training) makes
#: the backend slow to answer unrelated background pollers. So the cadence
#: must stay comfortably below the breaker's tolerance for consecutive poller
#: misses; 5s is well under that.
HEARTBEAT_SECONDS = 5.0

#: Idle socket-probe cadence. With ``stream_with_context`` a client that
#: vanished abruptly (page reload, killed tab) is only detected when a write
#: to the dead socket fails — and the slot it holds via
#: :func:`acquire_sse_slot` is only released then. If the only idle write
#: were the heartbeat, a reloading page could collide with its *own*
#: previous connection's slot for a full heartbeat period (#2816). So when
#: no frame has been written for this long, the stream emits a bare SSE
#: comment: invisible to the browser's ``EventSource`` (it must NOT reset
#: the client's liveness breaker — only real events do), but enough to make
#: the write fail fast and free the slot.
KEEPALIVE_SECONDS = 1.0

#: Per-process identifier emitted as the first SSE frame on every new
#: connection. Clients compare it against the last value they saw; when
#: it changes, the backend has restarted and any frontend state keyed
#: on ``task_id``s from the previous process must be discarded
#: (otherwise stale ids leak forever — see audit M27).
BOOT_ID = uuid.uuid4().hex

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


def _default_max_connections() -> int:
    """Derive the SSE connection cap from the deployed thread pool size.

    Each open ``/api/events`` connection pins a ``gthread`` worker thread
    for its whole lifetime (the generator blocks in ``queue.get()`` between
    updates). ``gunicorn.conf.py`` runs a single worker with
    ``VTSEARCH_THREADS`` (default 8) threads, so an unbounded number of
    open tabs can starve the pool of threads needed to serve ordinary REST
    requests — a stalled request plus a live heartbeat then reads as an app
    hang rather than a clear error (comprehensive-audit-2026-07 #1).
    Reserve a fixed headroom so SSE can never claim every thread.
    """
    pool = int(os.environ.get("VTSEARCH_THREADS", "8"))
    reserve = 2
    return max(1, pool - reserve)


#: Hard cap on concurrent ``/api/events`` connections this process accepts.
#: Override directly with ``VTSEARCH_SSE_MAX_CONNECTIONS``; otherwise derived
#: from ``VTSEARCH_THREADS`` via :func:`_default_max_connections`.
MAX_SSE_CONNECTIONS = int(os.environ.get("VTSEARCH_SSE_MAX_CONNECTIONS", str(_default_max_connections())))

_connection_lock = threading.Lock()
_active_connections = 0


def uncap_sse_connections() -> None:
    """Remove the SSE connection cap for servers without a bounded thread pool.

    The cap exists to keep long-lived streams from starving gunicorn's
    ``gthread`` worker pool. The Flask dev server (``app.run(threaded=True)``,
    see ``vtsearch/cli_main.py``) spawns a thread per connection and has no
    pool to protect, so there the cap is pure downside: extra tabs get 503s
    while the server has capacity to spare (#2816). An explicit
    ``VTSEARCH_SSE_MAX_CONNECTIONS`` override still wins.
    """
    global MAX_SSE_CONNECTIONS
    if "VTSEARCH_SSE_MAX_CONNECTIONS" in os.environ:
        return
    MAX_SSE_CONNECTIONS = sys.maxsize


def acquire_sse_slot() -> bool:
    """Reserve one of the bounded SSE connection slots.

    Returns ``False`` (without side effects) if the process is already at
    :data:`MAX_SSE_CONNECTIONS`; the caller should reject the connection
    instead of starting a stream that would pin a thread indefinitely.
    Pair every successful call with :func:`release_sse_slot`.
    """
    global _active_connections
    with _connection_lock:
        if _active_connections >= MAX_SSE_CONNECTIONS:
            return False
        _active_connections += 1
        return True


def release_sse_slot() -> None:
    """Release a slot reserved by :func:`acquire_sse_slot`."""
    global _active_connections
    with _connection_lock:
        _active_connections = max(0, _active_connections - 1)


def active_sse_connections() -> int:
    """Current count of open SSE connections (test/introspection helper)."""
    with _connection_lock:
        return _active_connections


def _format_sse(event: str, data: Any) -> str:
    payload = json.dumps(data, default=str)
    return f"event: {event}\ndata: {payload}\n\n"


def initial_snapshot() -> list[str]:
    """Return the SSE frames a freshly-connected client should receive first.

    The first frame is always the ``server`` channel carrying the
    process's :data:`BOOT_ID`; subsequent frames are the per-channel
    progress snapshots.
    """
    frames: list[str] = [_format_sse("server", {"boot_id": BOOT_ID})]
    for name, tracker in _TRACKER_CHANNELS.items():
        frames.append(_format_sse(name, tracker.get()))
    for name, tasks_tracker in _TASK_CHANNELS.items():
        frames.append(_format_sse(name, tasks_tracker.list_tasks()))
    return frames


def stream_progress_events(  # noqa: C901
    *,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    keepalive_seconds: float = KEEPALIVE_SECONDS,
    max_queue: int = 1024,
) -> Generator[str, None, None]:
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
        last_write = time.monotonic()
        while True:
            deadline = min(last_heartbeat + heartbeat_seconds, last_write + keepalive_seconds)
            try:
                name, snapshot = q.get(timeout=max(0.0, deadline - time.monotonic()))
                yield _format_sse(name, snapshot)
            except queue.Empty:
                if time.monotonic() - last_heartbeat >= heartbeat_seconds:
                    # Emit the heartbeat as a real, named ``heartbeat`` event
                    # rather than an SSE comment (``: heartbeat``). Comments are
                    # invisible to the browser's EventSource API, so the client
                    # could not use them as a liveness signal; a named event fires
                    # a listener, letting the frontend treat every heartbeat as
                    # proof the backend is still alive (and keeping idle proxies
                    # open just as a comment would).
                    yield _format_sse("heartbeat", {"ts": time.time()})
                    # Re-emit every channel's current snapshot. For task
                    # channels this is what makes finished tasks disappear
                    # once their stale-prune window elapses. For tracker
                    # channels it is a self-heal: a client whose bounded
                    # queue overflowed (see the ``queue.Full`` drops above)
                    # can lose a channel's single terminal ``idle``/``error``
                    # frame, which would otherwise leave its progress bar
                    # stuck at the last percentage until some later operation
                    # happened to fire that channel again (#2960). Snapshots
                    # are tiny, so re-emitting them makes every channel
                    # eventually consistent after any drop.
                    for name, tracker in _TRACKER_CHANNELS.items():
                        yield _format_sse(name, tracker.get())
                    for name, tasks_tracker in _TASK_CHANNELS.items():
                        yield _format_sse(name, tasks_tracker.list_tasks())
                    last_heartbeat = time.monotonic()
                else:
                    # Socket probe only (see KEEPALIVE_SECONDS): a comment is
                    # invisible to EventSource, so it deliberately does not
                    # feed the client's liveness breaker — its sole job is to
                    # make a write happen so a dead connection raises and
                    # releases its slot promptly.
                    yield ": ka\n\n"
            last_write = time.monotonic()
    finally:
        for subject, handler in subscriptions:
            subject.unsubscribe(handler)
