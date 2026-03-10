"""Progress tracking for long-running operations."""

import threading
from typing import Any, Optional

_UNSET = object()  # sentinel for "caller did not provide this argument"


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
            for key in self._extra_defaults:
                if key in kwargs:
                    self._data[key] = kwargs[key]

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
dataset_progress = ProgressTracker(
    extra_fields={"error": None, "staging_result": None, "step": None, "total_steps": None}
)

#: Sort-specific progress (used by text-sort operations).
sort_progress = ProgressTracker()

#: Eval progress (used by train-and-score / voting-iterations analysis).
eval_progress = ProgressTracker()


# ---------------------------------------------------------------------------
# Backward-compatible free-function API
# ---------------------------------------------------------------------------


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
    kwargs: dict[str, Any] = {}
    if error is not _UNSET:
        kwargs["error"] = error
    if staging_result is not _UNSET:
        kwargs["staging_result"] = staging_result
    if step is not _UNSET:
        kwargs["step"] = step
    if total_steps is not _UNSET:
        kwargs["total_steps"] = total_steps
    dataset_progress.update(status, message, current, total, **kwargs)


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


def update_eval_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
) -> None:
    """Update the eval progress tracker in a thread-safe manner."""
    eval_progress.update(status, message, current, total)


def get_eval_progress() -> dict[str, Any]:
    """Return a snapshot of the current eval progress data."""
    return eval_progress.get()
