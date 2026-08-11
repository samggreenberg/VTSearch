"""Tests for the /api/events SSE endpoint and ProgressTracker subscriptions."""

import json
import threading
import time

import vtscore.concurrency.events as events_mod
from vtscore.concurrency.events import (
    _TASK_CHANNELS,
    _TRACKER_CHANNELS,
    BOOT_ID,
    acquire_sse_slot,
    active_sse_connections,
    initial_snapshot,
    release_sse_slot,
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
        first = tracker._smoothed_eta
        assert first is not None

        # Pretend the rate suddenly doubles: 10 more units in just 1s.
        now[0] += 1.0
        tracker.update("loading", "", 20, 100)
        smoothed = tracker._smoothed_eta
        # Raw new ETA = (11 / 20) * 80 = 44s. Smoothed with alpha=0.3 against
        # the previous ~90s sample sits well above 44 and below the old 90.
        # Asserted on the *internal* estimate: what gets published is snapped to
        # the coarse ETA ladder (see TestHumbleEta), and a step this small is
        # precisely what the ladder exists to absorb.
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

    def test_heartbeat_is_a_real_named_event(self):
        """An idle stream emits a named ``heartbeat`` event (not an SSE
        comment) so the browser's EventSource fires a listener and the client
        can use it as a liveness signal."""
        gen = stream_progress_events(heartbeat_seconds=0.01)
        try:
            # Drain the connect comment + initial snapshot, then keep pulling
            # until the idle branch fires a heartbeat (no updates published, so
            # the queue.get() times out almost immediately).
            heartbeat = None
            deadline = time.monotonic() + 2.0
            while heartbeat is None and time.monotonic() < deadline:
                chunk = next(gen)
                if chunk.startswith("event: heartbeat\n"):
                    heartbeat = chunk
            assert heartbeat is not None, "no heartbeat frame arrived"
            # It is a real event with a JSON data line, never a `: heartbeat`
            # comment.
            assert not heartbeat.startswith(": ")
            data_line = [ln for ln in heartbeat.splitlines() if ln.startswith("data: ")][0]
            payload = json.loads(data_line.removeprefix("data: "))
            assert "ts" in payload
        finally:
            gen.close()

    def test_idle_stream_emits_keepalive_comment_between_heartbeats(self):
        """An idle stream probes the socket with an SSE comment well before
        the next heartbeat, so a dead client's slot is released in ~1s
        instead of a full heartbeat period (#2816). The probe must be a
        comment (invisible to EventSource), never a named event."""
        gen = stream_progress_events(heartbeat_seconds=60.0, keepalive_seconds=0.01)
        try:
            assert next(gen).startswith(": connected")
            for _ in initial_snapshot():
                next(gen)
            # The next idle wakeup is the socket-probe comment, not a heartbeat.
            chunk = next(gen)
            assert chunk.startswith(": ")
            assert "event:" not in chunk
        finally:
            gen.close()

    def test_heartbeat_still_fires_with_keepalive_enabled(self):
        """Keepalive probes must not starve the named heartbeat the client's
        liveness breaker depends on."""
        gen = stream_progress_events(heartbeat_seconds=0.05, keepalive_seconds=0.01)
        try:
            heartbeat = None
            deadline = time.monotonic() + 2.0
            while heartbeat is None and time.monotonic() < deadline:
                chunk = next(gen)
                if chunk.startswith("event: heartbeat\n"):
                    heartbeat = chunk
            assert heartbeat is not None, "no heartbeat frame arrived"
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

    def test_heartbeat_reemits_tracker_channels(self):
        """A client whose bounded queue overflowed can lose a tracker
        channel's single terminal frame, which would leave its progress bar
        stuck at the last percentage until some later operation happened to
        fire that channel again. The heartbeat re-emits every tracker
        snapshot, so the client self-heals within one heartbeat (#2960)."""
        # max_queue=1 makes the overflow deterministic: the first update
        # fills the queue, the terminal one is dropped just as it would be
        # for a stalled client behind a burst of per-item frames.
        gen = stream_progress_events(heartbeat_seconds=0.01, max_queue=1)
        try:
            assert next(gen).startswith(": connected")
            for _ in initial_snapshot():
                next(gen)

            sort_progress.update("running", "Scoring", 97, 100)
            sort_progress.update("idle", "Done", 100, 100)  # dropped: queue full

            # The only queued frame is the stale `running` one.
            frame = next(gen)
            assert frame.startswith("event: sort\n")
            payload = json.loads(
                [ln for ln in frame.splitlines() if ln.startswith("data: ")][0].removeprefix("data: ")
            )
            assert payload["status"] == "running"

            # The heartbeat repairs it without any further tracker update.
            repaired = None
            deadline = time.monotonic() + 2.0
            while repaired is None and time.monotonic() < deadline:
                chunk = next(gen)
                if chunk.startswith("event: sort\n"):
                    repaired = json.loads(
                        [ln for ln in chunk.splitlines() if ln.startswith("data: ")][0].removeprefix("data: ")
                    )
            assert repaired is not None, "no re-emitted sort frame arrived"
            assert repaired["status"] == "idle"
            assert repaired["current"] == 100
        finally:
            gen.close()
            sort_progress.update("idle", "", 0, 0)

    def test_heartbeat_reemits_every_tracker_channel(self):
        """The re-emit covers all of ``_TRACKER_CHANNELS``, not just the one
        that happened to move last."""
        gen = stream_progress_events(heartbeat_seconds=0.01)
        try:
            assert next(gen).startswith(": connected")
            for _ in initial_snapshot():
                next(gen)

            seen: set[str] = set()
            expected = set(_TRACKER_CHANNELS) | set(_TASK_CHANNELS)
            saw_heartbeat = False
            deadline = time.monotonic() + 2.0
            while seen != expected and time.monotonic() < deadline:
                chunk = next(gen)
                if chunk.startswith("event: heartbeat\n"):
                    saw_heartbeat = True
                elif saw_heartbeat and chunk.startswith("event: "):
                    seen.add(chunk.split("\n", 1)[0].removeprefix("event: "))
            assert seen == expected
        finally:
            gen.close()

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


class TestEventsStateSyncExemption:
    """`/api/events` must skip the lock-taking ``before_request`` state-sync.

    The SSE stream is read-only — it only subscribes to the *global*
    progress trackers and yields their snapshots — so it never needs the
    per-request vote rehydration. Gating its (re)connect on ``_state_lock``
    is self-defeating: while a long Find/load holds the worker busy, the
    EventSource reconnect would block on the very lock the long job
    contends, and the progress events the bar needs would never arrive.
    So it is exempt exactly like the jobs/active spinner poll.
    """

    def _spy_state_sync(self, monkeypatch) -> list[str]:
        """Replace the two before_request state-sync helpers with recorders.

        ``_set_request_context`` imports them from
        ``vtscore.detectors.dataset_sync`` at call time, so patching the
        source module is what the hook actually sees.
        """
        calls: list[str] = []
        import vtscore.detectors.dataset_sync as ds

        monkeypatch.setattr(ds, "ensure_votes_match_active_dataset", lambda: calls.append("votes"))
        monkeypatch.setattr(ds, "ensure_detector_model_matches_active_embedder", lambda: calls.append("embedder"))
        return calls

    def test_events_skips_lock_taking_state_sync(self, client, monkeypatch):
        calls = self._spy_state_sync(monkeypatch)
        resp = client.get("/api/events", buffered=False)
        try:
            assert resp.status_code == 200
        finally:
            resp.close()
        # Exempt prefix → the lock-taking rehydrate never ran for this request.
        assert calls == []

    def test_non_exempt_endpoint_still_runs_state_sync(self, client, monkeypatch):
        """Control: a normal (non-exempt) endpoint still triggers the sync,
        proving the empty result above is the exemption, not a dead spy."""
        calls = self._spy_state_sync(monkeypatch)
        resp = client.get("/api/version")
        assert resp.status_code == 200
        assert "votes" in calls and "embedder" in calls


# ---------------------------------------------------------------------------
# SSE connection cap (comprehensive-audit-2026-07 open follow-up #1)
# ---------------------------------------------------------------------------


class TestSseConnectionCapPrimitive:
    def test_acquire_up_to_cap_then_rejects(self, monkeypatch):
        monkeypatch.setattr(events_mod, "MAX_SSE_CONNECTIONS", 2)
        assert active_sse_connections() == 0
        assert acquire_sse_slot() is True
        assert acquire_sse_slot() is True
        assert active_sse_connections() == 2
        # Third acquire is over the cap: rejected without incrementing.
        assert acquire_sse_slot() is False
        assert active_sse_connections() == 2

        release_sse_slot()
        assert active_sse_connections() == 1
        assert acquire_sse_slot() is True
        assert active_sse_connections() == 2

        release_sse_slot()
        release_sse_slot()
        assert active_sse_connections() == 0

    def test_release_below_zero_is_clamped(self, monkeypatch):
        monkeypatch.setattr(events_mod, "MAX_SSE_CONNECTIONS", 2)
        assert active_sse_connections() == 0
        release_sse_slot()
        assert active_sse_connections() == 0

    def test_default_max_connections_derives_from_thread_pool(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_THREADS", "8")
        assert events_mod._default_max_connections() == 6

    def test_default_max_connections_floors_at_one(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_THREADS", "1")
        assert events_mod._default_max_connections() == 1

    def test_uncap_removes_the_cap(self, monkeypatch):
        """The dev server (`app.run(threaded=True)`) has no thread pool to
        protect, so `_run_server` uncaps SSE connections (#2816)."""
        monkeypatch.delenv("VTSEARCH_SSE_MAX_CONNECTIONS", raising=False)
        monkeypatch.setattr(events_mod, "MAX_SSE_CONNECTIONS", 2)
        events_mod.uncap_sse_connections()
        assert events_mod.MAX_SSE_CONNECTIONS > 10**6
        for _ in range(10):
            assert acquire_sse_slot() is True
        for _ in range(10):
            release_sse_slot()

    def test_uncap_respects_explicit_env_override(self, monkeypatch):
        monkeypatch.setenv("VTSEARCH_SSE_MAX_CONNECTIONS", "4")
        monkeypatch.setattr(events_mod, "MAX_SSE_CONNECTIONS", 4)
        events_mod.uncap_sse_connections()
        assert events_mod.MAX_SSE_CONNECTIONS == 4


class TestSseConnectionCapRoute:
    def test_route_returns_503_with_retry_after_when_saturated(self, client, monkeypatch):
        monkeypatch.setattr(events_mod, "MAX_SSE_CONNECTIONS", 0)
        resp = client.get("/api/events", buffered=False)
        try:
            assert resp.status_code == 503
            assert resp.headers.get("Retry-After") == "5"
            body = resp.get_json()
            assert "message" in body
        finally:
            resp.close()
        # Rejection never touched the counter.
        assert active_sse_connections() == 0

    def test_route_rejects_when_already_saturated(self, client, monkeypatch):
        """Simulate another connection already holding the sole slot (via the
        primitive, not a second nested test-client request — the Werkzeug
        test client's request-context stack isn't reentrant across two
        concurrently-open streams on the same client)."""
        monkeypatch.setattr(events_mod, "MAX_SSE_CONNECTIONS", 1)
        assert acquire_sse_slot() is True
        try:
            resp = client.get("/api/events", buffered=False)
            try:
                assert resp.status_code == 503
            finally:
                resp.close()
        finally:
            release_sse_slot()
        assert active_sse_connections() == 0

    def test_route_releases_slot_on_disconnect(self, client, monkeypatch):
        monkeypatch.setattr(events_mod, "MAX_SSE_CONNECTIONS", 1)
        resp = client.get("/api/events", buffered=False)
        try:
            assert resp.status_code == 200
            assert active_sse_connections() == 1
        finally:
            resp.close()
        # Closing the connection frees its slot.
        assert active_sse_connections() == 0
