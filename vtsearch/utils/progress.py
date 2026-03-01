"""Progress tracking for long-running operations."""

import threading
from typing import Any, Optional


class ProgressTracker:
    """Thread-safe progress tracker for long-running operations.

    Each instance manages its own lock and data dict. The *extra_fields*
    parameter lets callers declare additional keys (e.g. ``"error"``,
    ``"staging_result"``) that are tracked alongside the base fields.

    Args:
        extra_fields: Mapping of extra field names to their default values.
            These fields can be set via keyword arguments in :meth:`update`
            and are returned by :meth:`get`.
    """

    def __init__(self, extra_fields: Optional[dict[str, Any]] = None) -> None:
        self._lock = threading.Lock()
        self._extra_defaults = dict(extra_fields) if extra_fields else {}
        self._data: dict[str, Any] = {
            "status": "idle",
            "message": "",
            "current": 0,
            "total": 0,
            **{k: v for k, v in self._extra_defaults.items()},
        }

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
            for key, default in self._extra_defaults.items():
                self._data[key] = kwargs.get(key, default)

    def get(self) -> dict[str, Any]:
        """Return a snapshot of the current progress data.

        Returns:
            A shallow copy of the internal data dict.
        """
        with self._lock:
            return dict(self._data)


# ---------------------------------------------------------------------------
# Application-wide singleton trackers
# ---------------------------------------------------------------------------

#: Dataset / import progress (used by dataset loading, downloading, embedding).
dataset_progress = ProgressTracker(extra_fields={"error": None, "staging_result": None})

#: Sort-specific progress (used by text-sort operations).
sort_progress = ProgressTracker()


# ---------------------------------------------------------------------------
# Backward-compatible free-function API
# ---------------------------------------------------------------------------


def update_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    error: Optional[str] = None,
    staging_result: Optional[dict] = None,
) -> None:
    """Update the global dataset progress tracker.

    All write access is serialised internally so that background threads
    can safely report progress while the Flask request thread polls
    :func:`get_progress`.
    """
    dataset_progress.update(status, message, current, total, error=error, staging_result=staging_result)


def get_progress() -> dict[str, Any]:
    """Return a snapshot of the current dataset progress data."""
    return dataset_progress.get()


def update_sort_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
) -> None:
    """Update the sort progress tracker in a thread-safe manner."""
    sort_progress.update(status, message, current, total)


def get_sort_progress() -> dict[str, Any]:
    """Return a snapshot of the current sort progress data."""
    return sort_progress.get()
