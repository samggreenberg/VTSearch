"""Progress tracking for long-running operations."""

import time
import threading
from typing import Any, Callable, Optional

_UNSET = object()  # sentinel for "caller did not provide this argument"


class CancelledError(Exception):
    """Raised when an operation is cancelled via :meth:`ProgressTracker.cancel`."""


class ProgressTracker:
    """Thread-safe progress tracker for long-running operations.

    Each instance manages its own lock and data dict. The *extra_fields*
    parameter lets callers declare additional keys (e.g. ``"error"``,
    ``"staging_result"``) that are tracked alongside the base fields.

    A :class:`threading.Event` is used for cooperative cancellation: call
    :meth:`cancel` to set the flag and :meth:`check_cancelled` from inside
    the background thread to raise :class:`CancelledError` when the flag is
    set.

    Subscribers (registered via :meth:`subscribe`) are invoked with a
    snapshot of the data dict on every :meth:`update`. The SSE event
    endpoint uses this to push progress to connected clients without
    polling.

    Args:
        extra_fields: Mapping of extra field names to their default values.
            These fields can be set via keyword arguments in :meth:`update`
            and are returned by :meth:`get`.
    """

    #: Minimum elapsed time (seconds) before an ETA is computed. Below this we
    #: don't have enough samples to extrapolate reliably and the number jitters
    #: wildly, so the snapshot's ``eta_seconds`` stays ``None``.
    _ETA_MIN_ELAPSED = 5.0

    #: Smoothing factor for the EMA over the raw ETA. ``0.3`` weights the new
    #: sample lightly enough to dampen noise while still tracking real slowdowns.
    _ETA_SMOOTHING_ALPHA = 0.3

    def __init__(self, extra_fields: Optional[dict[str, Any]] = None) -> None:
        self._lock = threading.Lock()
        self._extra_defaults = dict(extra_fields) if extra_fields else {}
        self._cancel_event = threading.Event()
        self._data: dict[str, Any] = {
            "status": "idle",
            "message": "",
            "current": 0,
            "total": 0,
            **{k: v for k, v in self._extra_defaults.items()},
        }
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._subscribers_lock = threading.Lock()
        self._phase_key: tuple[str, int] | None = None
        self._phase_start: float | None = None
        self._phase_current_start: int = 0
        self._smoothed_eta: float | None = None

    def _compute_eta(self, status: str, current: int, total: int) -> Optional[float]:
        """Compute the smoothed ETA in seconds for the current bar.

        Resets the phase clock when ``status`` changes, ``total`` changes, or
        ``current`` resets backwards (a new bar is starting). Returns ``None``
        until at least :data:`_ETA_MIN_ELAPSED` seconds of work have elapsed
        with a known total and ``current > 0``.
        """
        now = time.monotonic()
        phase_key = (status, total)
        if self._phase_key != phase_key or self._phase_start is None or current < self._phase_current_start:
            self._phase_key = phase_key
            self._phase_start = now
            self._phase_current_start = current
            self._smoothed_eta = None
            return None

        if total <= 0 or current <= 0 or current >= total:
            return None
        elapsed = now - self._phase_start
        if elapsed < self._ETA_MIN_ELAPSED:
            return None
        completed = current - self._phase_current_start
        if completed <= 0:
            return None
        raw_eta = (elapsed / completed) * (total - current)
        if self._smoothed_eta is None:
            self._smoothed_eta = raw_eta
        else:
            alpha = self._ETA_SMOOTHING_ALPHA
            self._smoothed_eta = alpha * raw_eta + (1.0 - alpha) * self._smoothed_eta
        return self._smoothed_eta

    def update(
        self,
        status: str,
        message: str = "",
        current: int = 0,
        total: int = 0,
        **kwargs: Any,
    ) -> None:
        """Update progress in a thread-safe manner.

        Args:
            status: Current operation phase (e.g. ``"idle"``, ``"loading"``).
            message: Human-readable description of what is happening.
            current: Number of units completed so far.
            total: Total number of units expected (0 if unknown).
            **kwargs: Values for any extra fields declared at construction.
                Unrecognised keys are silently ignored.
        """
        with self._lock:
            self._data["status"] = status
            self._data["message"] = message
            self._data["current"] = current
            self._data["total"] = total
            for key in self._extra_defaults:
                if key in kwargs:
                    self._data[key] = kwargs[key]
            if "eta_seconds" in self._extra_defaults:
                self._data["eta_seconds"] = self._compute_eta(status, current, total)
            snapshot = dict(self._data)
        self._notify(snapshot)

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Register a callback fired with a snapshot after every update.

        The callback runs synchronously on the thread that called
        :meth:`update`, *outside* the tracker's lock. Subscribers must be
        non-blocking and exception-safe; any exception they raise is
        swallowed so a misbehaving subscriber cannot break the producer.
        """
        with self._subscribers_lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[dict[str, Any]], None]) -> None:
        """Remove a previously-registered subscriber. No-op if not present."""
        with self._subscribers_lock:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

    def _notify(self, snapshot: dict[str, Any]) -> None:
        with self._subscribers_lock:
            subs = list(self._subscribers)
        for cb in subs:
            try:
                cb(snapshot)
            except Exception:
                pass

    def cancel(self) -> None:
        """Signal the background operation to stop.

        This sets the internal cancel event.  The background thread must
        cooperatively check it via :meth:`check_cancelled`.
        """
        self._cancel_event.set()

    def check_cancelled(self) -> None:
        """Raise :class:`CancelledError` if :meth:`cancel` has been called.

        Call this periodically from inside the background thread (e.g. once
        per loop iteration) to allow cooperative cancellation.
        """
        if self._cancel_event.is_set():
            raise CancelledError("Operation cancelled by user")

    @property
    def is_cancelled(self) -> bool:
        """Return ``True`` if cancellation has been requested."""
        return self._cancel_event.is_set()

    def reset_cancel(self) -> None:
        """Clear the cancellation flag.

        Called at the beginning of a new operation so that a previous
        cancellation does not immediately abort the next run.
        """
        self._cancel_event.clear()

    def get(self) -> dict[str, Any]:
        """Return a snapshot of the current progress data.

        Returns:
            A shallow copy of the internal data dict.
        """
        with self._lock:
            return dict(self._data)


# ---------------------------------------------------------------------------
# Thread-local progress callback
# ---------------------------------------------------------------------------
# Background loading threads set a per-thread progress callback via
# set_thread_progress().  The _default_progress() functions in loader.py,
# downloader.py, etc. check this first, falling back to the global
# update_progress() when no per-thread callback is set.  This avoids
# monkey-patching module-level defaults and allows parallel loads to each
# report to their own ProgressTracker.

_thread_progress = threading.local()


def set_thread_progress(callback) -> None:
    """Set the progress callback for the current thread."""
    _thread_progress.callback = callback


def get_thread_progress():
    """Return the per-thread progress callback, or ``None``."""
    return getattr(_thread_progress, "callback", None)


def clear_thread_progress() -> None:
    """Remove the per-thread progress callback."""
    _thread_progress.callback = None


# ---------------------------------------------------------------------------
# Shared progress extras
# ---------------------------------------------------------------------------
#: Extras shared by every long-running operation: an optional sub-step counter
#: (``step``/``total_steps`` - used when a single operation has multiple phases
#: like load→embed→stage), an ``error`` string, and a smoothed ``eta_seconds``
#: filled in automatically by :meth:`ProgressTracker._compute_eta`. Every
#: singleton tracker - and every per-task tracker created by
#: :class:`LoadingTasksTracker` - exposes these so the frontend can render any
#: progress payload with the same ``ProgressEvent`` interface (see
#: ``frontend/src/app/models/api.models.ts``).
_PROGRESS_COMMON_EXTRAS: dict[str, Any] = {
    "step": None,
    "total_steps": None,
    "error": None,
    "eta_seconds": None,
}


# ---------------------------------------------------------------------------
# Loading tasks tracker - manages multiple concurrent loading operations
# ---------------------------------------------------------------------------


class LoadingTasksTracker:
    """Manages multiple concurrent dataset loading tasks.

    Each task has its own :class:`ProgressTracker`, a display name, and a
    creation timestamp.  The dashboard polls :meth:`list_tasks` to show
    one progress row per loading dataset.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._subscribers: list[Callable[[list[dict[str, Any]]], None]] = []
        self._subscribers_lock = threading.Lock()

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
    ) -> ProgressTracker:
        """Create and register a new loading task.

        Returns the per-task :class:`ProgressTracker` instance.
        """
        tracker = ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))
        tracker.subscribe(lambda _snapshot: self._notify())
        with self._lock:
            self._tasks[task_id] = {
                "tracker": tracker,
                "name": name,
                "created_at": time.time(),
                "finished_at": None,
                "dataset_id": dataset_id,
                "media_type": media_type,
                "detector_id": detector_id,
                "embedder": embedder,
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

    def list_tasks(self) -> list[dict[str, Any]]:  # noqa: C901
        """Return a snapshot of all active loading tasks.

        Each entry includes: ``task_id``, ``name``, ``created_at``, and
        all fields from the task's :class:`ProgressTracker`.

        Finished tasks without errors are removed after 5 seconds.
        Finished tasks *with* errors are kept for 30 seconds so that
        the polling frontend has time to display them.
        """
        now = time.time()
        stale: list[str] = []
        with self._lock:
            entries = list(self._tasks.items())
        result = []
        for task_id, entry in entries:
            finished = entry.get("finished_at")
            if finished is not None:
                snapshot = entry["tracker"].get()
                has_error = bool(snapshot.get("error"))
                max_age = 30 if has_error else 5
                if (now - finished) > max_age:
                    stale.append(task_id)
                    continue
                snapshot["task_id"] = task_id
                snapshot["name"] = entry["name"]
                snapshot["created_at"] = entry["created_at"]
                if entry.get("dataset_id"):
                    snapshot["dataset_id"] = entry["dataset_id"]
                if entry.get("detector_id"):
                    snapshot["detector_id"] = entry["detector_id"]
                if entry.get("media_type"):
                    snapshot["media_type"] = entry["media_type"]
                if entry.get("embedder"):
                    snapshot["embedder"] = entry["embedder"]
                result.append(snapshot)
            else:
                snapshot = entry["tracker"].get()
                snapshot["task_id"] = task_id
                snapshot["name"] = entry["name"]
                snapshot["created_at"] = entry["created_at"]
                if entry.get("dataset_id"):
                    snapshot["dataset_id"] = entry["dataset_id"]
                if entry.get("detector_id"):
                    snapshot["detector_id"] = entry["detector_id"]
                if entry.get("media_type"):
                    snapshot["media_type"] = entry["media_type"]
                if entry.get("embedder"):
                    snapshot["embedder"] = entry["embedder"]
                result.append(snapshot)
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

    def reset_for_tests(self) -> None:
        """Clear all tasks.  For test isolation."""
        with self._lock:
            self._tasks.clear()


#: Application-wide loading tasks tracker (for datasets).
loading_tasks = LoadingTasksTracker()

#: Application-wide loading tasks tracker (for detectors).
detector_loading_tasks = LoadingTasksTracker()


# ---------------------------------------------------------------------------
# Application-wide singleton trackers
# ---------------------------------------------------------------------------

#: Dataset / import progress (used by dataset loading, downloading, embedding).
#: Adds ``staging_result`` on top of the common extras for the combine-datasets
#: staging flow.
dataset_progress = ProgressTracker(extra_fields={**_PROGRESS_COMMON_EXTRAS, "staging_result": None})

#: Sort-specific progress (used by text-sort operations).
sort_progress = ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))

#: Eval progress (used by train-and-score / voting-iterations analysis).
eval_progress = ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))

#: Find progress (used by the /api/find multi-dataset×model scoring operation).
find_progress = ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))


# ---------------------------------------------------------------------------
# Backward-compatible free-function API
# ---------------------------------------------------------------------------


def _common_extras_kwargs(
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
    error: Any = _UNSET,
) -> dict[str, Any]:
    """Build the kwargs dict for the shared ``step``/``total_steps``/``error`` extras.

    Only fields explicitly supplied by the caller are forwarded so omitted
    fields are left unchanged (true update/merge semantics).
    """
    kwargs: dict[str, Any] = {}
    if step is not _UNSET:
        kwargs["step"] = step
    if total_steps is not _UNSET:
        kwargs["total_steps"] = total_steps
    if error is not _UNSET:
        kwargs["error"] = error
    return kwargs


def update_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    error: Any = _UNSET,
    staging_result: Any = _UNSET,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
) -> None:
    """Update the global dataset progress tracker.

    All write access is serialised internally so that background threads
    can safely report progress while the Flask request thread polls
    :func:`get_progress`.

    Only extra fields explicitly supplied by the caller are forwarded;
    omitted fields are left unchanged (true update/merge semantics).
    """
    kwargs = _common_extras_kwargs(step, total_steps, error)
    if staging_result is not _UNSET:
        kwargs["staging_result"] = staging_result
    dataset_progress.update(status, message, current, total, **kwargs)


def get_progress() -> dict[str, Any]:
    """Return a snapshot of the current dataset progress data.

    Checks per-task loading trackers first (used by parallel dataset
    loading) and falls back to the legacy global singleton.
    """
    tasks = loading_tasks.list_tasks()
    active = [t for t in tasks if t.get("status") != "idle"]
    if active:
        return active[0]
    # Check if any just-finished task has an error to report
    errored = [t for t in tasks if t.get("error")]
    if errored:
        return errored[0]
    return dataset_progress.get()


def cancel_dataset_progress() -> None:
    """Signal the current dataset operation(s) to cancel.

    Cancels all active per-task loading trackers as well as the legacy
    global singleton (used by staging operations).
    """
    loading_tasks.cancel_all()
    dataset_progress.cancel()


def check_dataset_cancelled() -> None:
    """Raise :class:`CancelledError` if the dataset operation was cancelled."""
    dataset_progress.check_cancelled()


def update_sort_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
    error: Any = _UNSET,
) -> None:
    """Update the sort progress tracker in a thread-safe manner."""
    sort_progress.update(status, message, current, total, **_common_extras_kwargs(step, total_steps, error))


def get_sort_progress() -> dict[str, Any]:
    """Return a snapshot of the current sort progress data."""
    return sort_progress.get()


def update_eval_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
    error: Any = _UNSET,
) -> None:
    """Update the eval progress tracker in a thread-safe manner."""
    eval_progress.update(status, message, current, total, **_common_extras_kwargs(step, total_steps, error))


def get_eval_progress() -> dict[str, Any]:
    """Return a snapshot of the current eval progress data."""
    return eval_progress.get()


def update_find_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
    error: Any = _UNSET,
) -> None:
    """Update the find progress tracker in a thread-safe manner."""
    find_progress.update(status, message, current, total, **_common_extras_kwargs(step, total_steps, error))


def get_find_progress() -> dict[str, Any]:
    """Return a snapshot of the current find progress data."""
    return find_progress.get()
