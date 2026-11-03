"""The :class:`LoadingTasksTracker`: one progress row per concurrent load.

Each dataset or detector load gets its own :class:`ProgressTracker` row,
listed until it finishes (or, for a kept failure, until it is dismissed).  The
``loading_tasks`` / ``detector_loading_tasks`` singletons are defined here,
beside their class, so each exists exactly once:
:mod:`vtscore.concurrency.progress_cancel` reads ``loading_tasks`` and cannot
import it from the facade that re-exports it.

Import these names from :mod:`vtscore.concurrency.progress`, which re-exports
every public one.
"""

import threading
import time
from typing import Any, Callable

from vtscore.concurrency.progress_tracker import PROGRESS_COMMON_EXTRAS, ProgressTracker


def ingest_eta_hidden() -> bool:
    """Deprecated: always ``True``.  Ingest progress bars never publish an ETA.

    An *ingest* is work that brings new media in — a dataset import, a
    staging import, a labelset's missing-media fetch. Its cost is set by the
    network, the source's disks and the files themselves, so its estimate
    swung too wildly to be worth showing. It used to be an operator switch
    (``CoreConfig.hide_ingest_eta``, issue #4233); since issue #4667 the
    ingest paths build their trackers with ``publish_eta=False``
    unconditionally and nothing calls this. Kept so out-of-tree callers keep
    importing it.
    """
    return True


# ---------------------------------------------------------------------------
# Loading tasks tracker - manages multiple concurrent loading operations
# ---------------------------------------------------------------------------


#: The ``error`` a task carries when the user cancelled it rather than it failing.
#: A string contract, not decoration: the frontend tests for this exact value to
#: tell a requested stop from a failure, and :class:`LoadingTasksTracker` never
#: keeps a cancel around as a failure.
CANCELLED_ERROR = "Cancelled"

#: How many kept failures (see ``create_task(keep_failure=)``) a tracker holds at
#: once; the oldest go first.
DEFAULT_MAX_KEPT_FAILURES = 20


def _is_kept_failure(snapshot: dict[str, Any]) -> bool:
    """True for a finished ``keep_failure`` task that failed (a cancel is not a failure)."""
    error = snapshot.get("error")
    return (
        bool(snapshot.get("keep_failure"))
        and snapshot.get("status") == "idle"
        and bool(error)
        and error != CANCELLED_ERROR
    )


class LoadingTasksTracker:
    """Manages multiple concurrent dataset loading tasks.

    Each task has its own :class:`ProgressTracker`, a display name, and a
    creation timestamp.  The dashboard polls :meth:`list_tasks` to show
    one progress row per loading dataset.

    A task created with ``keep_failure=True`` that finishes with an error stays
    listed until :meth:`dismiss_task` removes it, rather than aging out.  Imports
    ask for this: a failed import never produces a dataset, so its row is the
    only thing left to say it happened, and letting it lapse after a few
    seconds left the user waiting on a dataset that was never coming (#4973).
    At most *max_kept_failures* are kept, oldest dropped first, so failures
    nobody dismisses cannot pile up without bound.  A kept failure is private
    to whoever started the task: see :meth:`visible_to`.
    """

    def __init__(self, *, max_kept_failures: int = DEFAULT_MAX_KEPT_FAILURES) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._subscribers: list[Callable[[list[dict[str, Any]]], None]] = []
        self._subscribers_lock = threading.Lock()
        self.max_kept_failures = max_kept_failures

    def subscribe(self, callback: Callable[[list[dict[str, Any]]], None]) -> None:
        """Register a callback fired with the task list after every change.

        Same semantics as :meth:`ProgressTracker.subscribe`: invoked
        synchronously, outside locks, exceptions swallowed.
        """
        with self._subscribers_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[list[dict[str, Any]]], None]) -> None:
        """Remove a previously-registered subscriber. No-op if not present."""
        with self._subscribers_lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def _notify(self) -> None:
        snapshot = self.list_tasks()
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(snapshot)
            except Exception:
                pass

    def create_task(
        self,
        task_id: str,
        name: str = "",
        dataset_id: str = "",
        media_type: str = "",
        detector_id: str = "",
        embedder: str = "",
        step_weights: list[float] | None = None,
        extra_fields: dict[str, Any] | None = None,
        publish_eta: bool = True,
        created_by: str | None = None,
        source: str = "",
        keep_failure: bool = False,
    ) -> ProgressTracker:
        """Create and register a new loading task.

        *step_weights* (one weight per step) tunes how the whole-job ``overall``
        bar paces across phases; omit for equal weighting. *extra_fields* adds
        task-specific tracked keys (e.g. ``staging_result``) on top of the
        shared progress extras. *publish_eta* ``False`` keeps the task's
        ``eta_seconds`` at ``None`` (see :class:`ProgressTracker`). Returns the
        per-task :class:`ProgressTracker` instance, already reporting
        ``status="loading"``: the task is running from the moment it exists,
        until an ``update("idle", ...)`` ends it.

        *keep_failure* keeps the task listed if it fails, until
        :meth:`dismiss_task` (see the class doc); use it for work whose failure
        leaves nothing else behind to report it.  *created_by* names the user
        the work is for, defaulting to the current user; :meth:`visible_to`
        uses it to keep a kept failure private. *source* is a human-readable
        description of what the task reads (an import's origin, say), shown
        beside a failure so the user can tell what broke.
        """
        if created_by is None:
            from vtscore.state.current_user import get_current_user  # noqa: PLC0415

            created_by = get_current_user()
        fields = dict(PROGRESS_COMMON_EXTRAS)
        if extra_fields:
            fields.update(extra_fields)
        # Born working, not ``"idle"``: the ``_notify`` below publishes this
        # row before the caller's first ``update``, and ``"idle"`` is the
        # terminal status every reader of these channels treats as "done".  A
        # client that caught that first frame took a load that had not started
        # for one that had finished — the Find route guard then opened an
        # unloaded detector to a stream of 409s (issue #4187).
        tracker = ProgressTracker(extra_fields=fields, initial_status="loading", publish_eta=publish_eta)
        if step_weights is not None:
            tracker.set_step_weights(step_weights)
        tracker.subscribe(lambda _snapshot: self._notify())
        with self._lock:
            self._tasks[task_id] = {
                "tracker": tracker,
                "name": name,
                "created_at": time.time(),
                "finished_at": None,
                # Filled in by :meth:`set_worker` once the caller has its thread.
                "worker": None,
                "dataset_id": dataset_id,
                "media_type": media_type,
                "detector_id": detector_id,
                "embedder": embedder,
                "created_by": created_by,
                "source": source,
                "keep_failure": keep_failure,
            }
        self._notify()
        return tracker

    def mark_finished(self, task_id: str) -> None:
        """Record the time a task finished (for deferred cleanup).

        ``list_tasks()`` prunes stale finished entries the next time it is
        called; SSE streams call it on every heartbeat tick so a
        background timer here is unnecessary.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry["finished_at"] = time.time()
        self._notify()

    def is_finished(self, task_id: str) -> bool:
        """Return ``True`` once :meth:`mark_finished` ran for *task_id*.

        The task body calls ``mark_finished`` from its outermost ``finally``,
        so a ``True`` here means the worker thread has fully unwound (gates
        released, contexts restored).  Tests use this to wait for a
        cancelled/finished task deterministically instead of sleeping.
        ``False`` for unknown (never created or already removed) task ids.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
        return bool(entry and entry.get("finished_at") is not None)

    def get_tracker(self, task_id: str) -> ProgressTracker | None:
        """Return the ProgressTracker for *task_id*, or ``None``."""
        with self._lock:
            entry = self._tasks.get(task_id)
        return entry["tracker"] if entry else None

    def cancel_task(self, task_id: str) -> bool:
        """Signal a specific task to cancel.  Returns ``True`` if found."""
        tracker = self.get_tracker(task_id)
        if tracker is not None:
            tracker.cancel()
            return True
        return False

    def cancel_all(self) -> None:
        """Signal all active tasks to cancel."""
        with self._lock:
            tasks = list(self._tasks.values())
        for entry in tasks:
            entry["tracker"].cancel()

    def set_worker(self, task_id: str, thread: threading.Thread) -> None:
        """Record the thread running *task_id*, so cancellation can be honest.

        Cancellation is cooperative: :meth:`ProgressTracker.cancel` only sets a
        flag, and *something has to be running* to observe it.  Knowing whether
        a worker is still alive is what lets a cancel distinguish "it will stop
        shortly" from "there is nobody here to stop" instead of answering both
        with the same cheerful ``ok`` (#3167).

        Register the thread *before* starting it; a task with no registered
        worker reads as not running.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry["worker"] = thread

    def worker_alive(self, task_id: str) -> bool:
        """Return ``True`` if *task_id* has a registered worker still running."""
        with self._lock:
            entry = self._tasks.get(task_id)
        worker = entry.get("worker") if entry else None
        return bool(worker is not None and worker.is_alive())

    def active_task_ids(self) -> list[str]:
        """Return the ids of tasks whose tracker still claims to be working.

        "Claims" is deliberate: a stale tracker looks exactly like a running
        one from here, which is why :func:`cancel_dataset_progress` cross-checks
        each id against :meth:`worker_alive` before reporting what it did.
        """
        with self._lock:
            entries = list(self._tasks.items())
        return [tid for tid, e in entries if e["tracker"].get()["status"] != "idle"]

    def set_dataset_id(self, task_id: str, dataset_id: str) -> None:
        """Associate a loading task with its final registry dataset ID."""
        with self._lock:
            entry = self._tasks.get(task_id)
            if entry:
                entry["dataset_id"] = dataset_id
        self._notify()

    def remove_task(self, task_id: str) -> None:
        """Remove a completed/cancelled task from the tracker."""
        with self._lock:
            self._tasks.pop(task_id, None)
        self._notify()

    def dismiss_task(self, task_id: str) -> bool:
        """Remove a *finished* task, typically a kept failure the user has read.

        Unlike :meth:`remove_task`, this refuses a task that is still running:
        dropping its entry would orphan the worker, leaving nothing to cancel it
        through.  Returns ``True`` if a finished task was removed, ``False`` if
        *task_id* is unknown or still running.
        """
        with self._lock:
            entry = self._tasks.get(task_id)
        if entry is None or entry["tracker"].get()["status"] != "idle":
            return False
        with self._lock:
            # Identity check: a task re-created under the same id in the
            # meantime (a retried registry load reuses its id) is not ours to drop.
            if self._tasks.get(task_id) is not entry:
                return False
            del self._tasks[task_id]
        self._notify()
        return True

    @staticmethod
    def visible_to(tasks: list[dict[str, Any]], viewer: str | None) -> list[dict[str, Any]]:
        """Filter a :meth:`list_tasks` snapshot down to what *viewer* should see.

        A kept failure stays until its owner dismisses it, so it is shown only
        to the user who started the task; anyone else would be left looking at
        a row that was not theirs to read or clear.  Everything else stays
        visible to everyone (running tasks, and failures that lapse on their
        own), as do kept failures with no real owner: a task created outside any
        user's request carries the ``"default"`` fallback.  *viewer* ``None``
        means unfiltered.
        """
        if viewer is None:
            return tasks
        from vtscore.state.current_user import DEFAULT_USER  # noqa: PLC0415

        return [
            t for t in tasks if not _is_kept_failure(t) or t.get("created_by", DEFAULT_USER) in (viewer, DEFAULT_USER)
        ]

    @staticmethod
    def _build_snapshot(task_id: str, entry: dict[str, Any]) -> dict[str, Any]:
        """Build the public snapshot dict for one task entry.

        Reads the task's :class:`ProgressTracker` and stamps on the
        ``task_id``, ``name``, ``created_at`` identity plus any optional
        fields (``dataset_id``, ``detector_id``, ``media_type``, ``embedder``,
        ``created_by``, ``source``, ``keep_failure``, ``finished_at``) that are set.
        """
        snapshot = entry["tracker"].get()
        snapshot["task_id"] = task_id
        snapshot["name"] = entry["name"]
        snapshot["created_at"] = entry["created_at"]
        for key in ("dataset_id", "detector_id", "media_type", "embedder", "created_by", "source", "keep_failure"):
            if entry.get(key):
                snapshot[key] = entry[key]
        if entry.get("finished_at") is not None:
            snapshot["finished_at"] = entry["finished_at"]
        return snapshot

    def list_tasks(self) -> list[dict[str, Any]]:
        """Return a snapshot of all active loading tasks.

        Each entry includes: ``task_id``, ``name``, ``created_at``, and
        all fields from the task's :class:`ProgressTracker`.

        Finished tasks without errors are removed after 5 seconds.
        Finished tasks *with* errors are kept for 30 seconds so that
        the polling frontend has time to display them — unless the task was
        created with ``keep_failure``, which keeps a failure (though not a
        cancel) until :meth:`dismiss_task`, up to ``max_kept_failures`` of them.
        """
        now = time.time()
        stale: list[str] = []
        with self._lock:
            entries = list(self._tasks.items())
        result = []
        kept_failures: list[tuple[float, str]] = []
        for task_id, entry in entries:
            finished = entry.get("finished_at")
            if finished is not None:
                snapshot = self._build_snapshot(task_id, entry)
                if _is_kept_failure(snapshot):
                    kept_failures.append((finished, task_id))
                    result.append(snapshot)
                    continue
                max_age = 30 if snapshot.get("error") else 5
                if (now - finished) > max_age:
                    stale.append(task_id)
                    continue
                result.append(snapshot)
                continue
            result.append(self._build_snapshot(task_id, entry))
        if len(kept_failures) > self.max_kept_failures:
            kept_failures.sort(reverse=True)
            overflow = {tid for _, tid in kept_failures[self.max_kept_failures :]}
            stale.extend(overflow)
            result = [t for t in result if t["task_id"] not in overflow]
        if stale:
            with self._lock:
                for tid in stale:
                    self._tasks.pop(tid, None)
        return result

    def has_active_tasks(self) -> bool:
        """Return ``True`` if any loading task is still running (not idle)."""
        with self._lock:
            entries = list(self._tasks.values())
        return any(e["tracker"].get()["status"] != "idle" for e in entries)

    def reset_for_tests(self, join_timeout: float = 5.0) -> None:
        """Cancel every task, wait for its worker, then clear the registry.

        Clearing alone is not isolation: the entry it drops is the only handle
        anything has on the worker thread, so a load still running at the end of
        a test became an *unreachable* thread that carried on mutating shared
        state — contexts, progress, the load concurrency gates — underneath the
        next test (issue #3613).  Cancelling first gives the worker its
        cooperative exit; the join then makes "the previous test's threads are
        gone" true rather than hoped-for.

        *join_timeout* is a budget for the whole sweep, not per worker, so a
        thread parked on something no test will ever set costs a bounded delay
        once instead of stalling the run.  It is spent only when a worker is
        actually alive, which is the exceptional case.
        """
        with self._lock:
            entries = list(self._tasks.values())

        for entry in entries:
            entry["tracker"].cancel()

        deadline = time.monotonic() + join_timeout
        for entry in entries:
            worker = entry.get("worker")
            if worker is None or not worker.is_alive():
                continue
            worker.join(timeout=max(0.0, deadline - time.monotonic()))

        with self._lock:
            self._tasks.clear()


#: Application-wide loading tasks tracker (for datasets).
loading_tasks = LoadingTasksTracker()

#: Application-wide loading tasks tracker (for detectors).
detector_loading_tasks = LoadingTasksTracker()
