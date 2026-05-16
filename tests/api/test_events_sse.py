"""Tests for the /api/events SSE endpoint and ProgressTracker subscriptions."""

import json
import threading
import time

from vtsearch.concurrency.events import (
    _TASK_CHANNELS,
    _TRACKER_CHANNELS,
    initial_snapshot,
    stream_progress_events,
)
from vtsearch.concurrency.progress import (
    LoadingTasksTracker,
    ProgressTracker,
    dataset_progress,
    loading_tasks,
    sort_progress,
)


# ---------------------------------------------------------------------------
# Subscribe / notify primitive tests
# ---------------------------------------------------------------------------


class TestProgressTrackerSubscriptions:
    def test_subscribe_receives_snapshot_on_update(self):
        tracker = ProgressTracker()
        received: list[dict] = []
        tracker.subscribe(received.append)

        tracker.update("loading", "Working", 3, 10)

        assert len(received) == 1
        assert received[0]["status"] == "loading"
        assert received[0]["current"] == 3
        assert received[0]["total"] == 10

    def test_unsubscribe_stops_notifications(self):
        tracker = ProgressTracker()
        received: list[dict] = []
        tracker.subscribe(received.append)
        tracker.unsubscribe(received.append)

        tracker.update("loading", "", 1, 2)
        assert received == []

    def test_unsubscribe_missing_is_noop(self):
        tracker = ProgressTracker()
        tracker.unsubscribe(lambda _: None)  # never subscribed; should not raise

    def test_subscriber_exception_does_not_break_producer(self):
        tracker = ProgressTracker()
        good: list[dict] = []

        def bad(_snapshot):
            raise RuntimeError("intentional")

        tracker.subscribe(bad)
        tracker.subscribe(good.append)

        tracker.update("loading", "", 1, 1)

        # The good subscriber still ran even though the bad one raised.
        assert len(good) == 1

    def test_extra_fields_round_trip_through_subscription(self):
        tracker = ProgressTracker(extra_fields={"error": None, "step": None})
        received: list[dict] = []
        tracker.subscribe(received.append)

        tracker.update("error", "boom", 0, 0, error="kapow", step=2)

        assert received[0]["error"] == "kapow"
        assert received[0]["step"] == 2


class TestLoadingTasksTrackerSubscriptions:
    def test_create_task_notifies(self):
        tracker = LoadingTasksTracker()
        received: list[list[dict]] = []
        tracker.subscribe(received.append)

        tracker.create_task("t1", name="One")
        try:
            assert len(received) == 1
            assert received[0][0]["task_id"] == "t1"
        finally:
            tracker.remove_task("t1")

    def test_inner_tracker_update_propagates_to_outer(self):
        tracker = LoadingTasksTracker()
        received: list[list[dict]] = []
        inner = tracker.create_task("t1", name="One")
        tracker.subscribe(received.append)
        try:
            inner.update("loading", "Working", 5, 10)
            assert len(received) == 1
            assert received[0][0]["status"] == "loading"
            assert received[0][0]["current"] == 5
        finally:
            tracker.remove_task("t1")

    def test_remove_notifies(self):
        tracker = LoadingTasksTracker()
        tracker.create_task("t1")
        received: list[list[dict]] = []
        tracker.subscribe(received.append)

        tracker.remove_task("t1")
        assert len(received) == 1
        assert received[0] == []


# ---------------------------------------------------------------------------
# /api/events endpoint
# ---------------------------------------------------------------------------


def _read_sse_event(stream, *, timeout=2.0):
    """Pull one event:/data: pair from the SSE generator."""
    deadline = time.monotonic() + timeout
    buf = ""
    while time.monotonic() < deadline:
        chunk = next(stream, None)
        if chunk is None:
            break
        buf += chunk
        if "\n\n" in buf:
            frame, _, buf = buf.partition("\n\n")
            return frame, buf
    return None, buf


class TestEventsRoute:
    def test_initial_snapshot_includes_every_channel(self):
        frames = initial_snapshot()
        # One frame per known channel.
        names = {f.split("\n", 1)[0].replace("event: ", "") for f in frames}
        expected = set(_TRACKER_CHANNELS.keys()) | set(_TASK_CHANNELS.keys())
        assert names == expected

    def test_endpoint_returns_sse_mimetype(self, client):
        # We can't keep the generator open easily through the test client,
        # so just hit it with a streaming GET and assert the response shape.
        resp = client.get("/api/events", buffered=False)
        try:
            assert resp.status_code == 200
            assert resp.mimetype == "text/event-stream"
            assert resp.headers.get("X-Accel-Buffering") == "no"
        finally:
            resp.close()

    def test_stream_yields_update_after_subscribe(self):
        """Pump the generator directly and assert a live update arrives."""
        gen = stream_progress_events(heartbeat_seconds=60.0)
        try:
            # Drain initial connect comment + the snapshot frames so the
            # generator is parked on the queue.get() call.
            seen_channels: set[str] = set()
            expected = set(_TRACKER_CHANNELS.keys()) | set(_TASK_CHANNELS.keys())
            deadline = time.monotonic() + 2.0
            while seen_channels != expected and time.monotonic() < deadline:
                chunk = next(gen)
                if chunk.startswith("event: "):
                    name = chunk.split("\n", 1)[0].removeprefix("event: ")
                    seen_channels.add(name)
            assert seen_channels == expected

            # Now publish an update on a fresh tracker and read it back.
            #
            # The generator is blocked inside queue.get() in the SSE
            # thread; we trigger an update from this thread, then pull
            # the next frame.
            def push_update():
                # Small wait so we know the generator has parked on
                # queue.get(). Without this, the publish can race the
                # subscribe.
                time.sleep(0.05)
                sort_progress.update("running", "tick", 1, 10)

            t = threading.Thread(target=push_update, daemon=True)
            t.start()
            frame = next(gen)
            t.join(timeout=1.0)
            assert frame.startswith("event: sort\n")
            data_line = [ln for ln in frame.splitlines() if ln.startswith("data: ")][0]
            payload = json.loads(data_line.removeprefix("data: "))
            assert payload["status"] == "running"
            assert payload["current"] == 1
        finally:
            gen.close()

    def test_initial_dataset_snapshot_reflects_current_state(self):
        dataset_progress.update("loading", "snap", 7, 8)
        try:
            frames = initial_snapshot()
            ds_frame = next(f for f in frames if f.startswith("event: dataset\n"))
            data_line = [ln for ln in ds_frame.splitlines() if ln.startswith("data: ")][0]
            payload = json.loads(data_line.removeprefix("data: "))
            assert payload["current"] == 7
            assert payload["total"] == 8
        finally:
            dataset_progress.update("idle", "", 0, 0)

    def test_initial_loading_tasks_snapshot(self):
        loading_tasks.create_task("evt_test", name="Evt DS")
        try:
            frames = initial_snapshot()
            lt_frame = next(f for f in frames if f.startswith("event: loading-tasks\n"))
            data_line = [ln for ln in lt_frame.splitlines() if ln.startswith("data: ")][0]
            payload = json.loads(data_line.removeprefix("data: "))
            assert any(t["task_id"] == "evt_test" for t in payload)
        finally:
            loading_tasks.remove_task("evt_test")
