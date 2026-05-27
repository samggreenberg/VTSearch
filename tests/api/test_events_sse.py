"""Tests for the /api/events SSE endpoint and ProgressTracker subscriptions."""

import json
import threading
import time

from vtscore.concurrency.events import (
    _TASK_CHANNELS,
    _TRACKER_CHANNELS,
    BOOT_ID,
    initial_snapshot,
    stream_progress_events,
)
from vtscore.concurrency.progress import (
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


class TestProgressTrackerEta:
    """``ProgressTracker`` should fill in a smoothed ``eta_seconds`` once a
    bar has been running >5s, reset its phase clock when the status or total
    changes, and stay silent otherwise. We patch :func:`time.monotonic` for
    determinism; the clock advance values matter, not wall time."""

    def _make_tracker(self) -> ProgressTracker:
        return ProgressTracker(extra_fields={"eta_seconds": None})

    def test_eta_is_none_until_min_elapsed(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("vtscore.concurrency.progress.time.monotonic", lambda: now[0])
        tracker = self._make_tracker()

        tracker.update("loading", "", 0, 100)
        assert tracker.get()["eta_seconds"] is None

        now[0] += 2.0  # <5s elapsed
        tracker.update("loading", "", 5, 100)
        assert tracker.get()["eta_seconds"] is None

    def test_eta_appears_after_min_elapsed(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("vtscore.concurrency.progress.time.monotonic", lambda: now[0])
        tracker = self._make_tracker()

        tracker.update("loading", "", 0, 100)
        now[0] += 10.0
        tracker.update("loading", "", 10, 100)
        eta = tracker.get()["eta_seconds"]
        # 10s elapsed for 10/100 done → 90 units remain at 1 unit/sec → 90s.
        assert eta is not None
        assert 89.0 <= eta <= 91.0

    def test_eta_smoothed_via_ema(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("vtscore.concurrency.progress.time.monotonic", lambda: now[0])
        tracker = self._make_tracker()

        tracker.update("loading", "", 0, 100)
        now[0] += 10.0
        tracker.update("loading", "", 10, 100)
        first = tracker.get()["eta_seconds"]
        assert first is not None

        # Pretend the rate suddenly doubles: 10 more units in just 1s.
        now[0] += 1.0
        tracker.update("loading", "", 20, 100)
        smoothed = tracker.get()["eta_seconds"]
        # Raw new ETA = (11 / 20) * 80 = 44s. Smoothed with alpha=0.3 against
        # the previous ~90s sample sits well above 44 and below the old 90.
        assert smoothed is not None
        assert 44.0 < smoothed < first

    def test_eta_resets_when_status_changes(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("vtscore.concurrency.progress.time.monotonic", lambda: now[0])
        tracker = self._make_tracker()

        tracker.update("downloading", "", 0, 100)
        now[0] += 10.0
        tracker.update("downloading", "", 50, 100)
        assert tracker.get()["eta_seconds"] is not None

        # New phase: clock should restart, ETA hidden until the next 5s elapse.
        tracker.update("embedding", "", 0, 100)
        assert tracker.get()["eta_seconds"] is None
        now[0] += 1.0
        tracker.update("embedding", "", 5, 100)
        assert tracker.get()["eta_seconds"] is None

    def test_eta_resets_when_total_changes(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("vtscore.concurrency.progress.time.monotonic", lambda: now[0])
        tracker = self._make_tracker()

        tracker.update("loading", "", 0, 100)
        now[0] += 10.0
        tracker.update("loading", "", 50, 100)
        assert tracker.get()["eta_seconds"] is not None

        # A new bar with a different total resets the phase clock.
        tracker.update("loading", "", 0, 200)
        assert tracker.get()["eta_seconds"] is None

    def test_eta_none_for_indeterminate_bars(self, monkeypatch):
        now = [1000.0]
        monkeypatch.setattr("vtscore.concurrency.progress.time.monotonic", lambda: now[0])
        tracker = self._make_tracker()

        tracker.update("loading", "", 0, 0)
        now[0] += 20.0
        tracker.update("loading", "", 42, 0)
        assert tracker.get()["eta_seconds"] is None

    def test_eta_field_absent_when_extra_not_declared(self):
        """Trackers created without an ``eta_seconds`` extra get no key;
        the feature is opt-in via :data:`_PROGRESS_COMMON_EXTRAS`."""
        tracker = ProgressTracker()
        tracker.update("loading", "", 1, 2)
        assert "eta_seconds" not in tracker.get()


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


class TestEventsRoute:
    def test_initial_snapshot_includes_every_channel(self):
        frames = initial_snapshot()
        # One frame per known channel plus the leading `server` identity
        # frame carrying boot_id.
        names = {f.split("\n", 1)[0].replace("event: ", "") for f in frames}
        expected = set(_TRACKER_CHANNELS.keys()) | set(_TASK_CHANNELS.keys()) | {"server"}
        assert names == expected

    def test_initial_snapshot_first_frame_is_server_boot_id(self):
        frames = initial_snapshot()
        assert frames[0].startswith("event: server\n")
        data_line = [ln for ln in frames[0].splitlines() if ln.startswith("data: ")][0]
        payload = json.loads(data_line.removeprefix("data: "))
        assert payload == {"boot_id": BOOT_ID}

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
            expected = set(_TRACKER_CHANNELS.keys()) | set(_TASK_CHANNELS.keys()) | {"server"}
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
