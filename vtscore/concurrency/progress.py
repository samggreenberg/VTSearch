"""Progress tracking for long-running operations.

The public face of the progress modules: the per-thread progress sink
(:func:`set_thread_progress`, :func:`resolve_progress_callback`,
:func:`update_progress`) and the free-function API over the shared trackers
live here, and every public name of the modules below is re-exported, as the
same object, so ``vtscore.concurrency.progress.X`` resolves as it did when
this was one module.  Each module reads only from those above it:

* :mod:`~vtscore.concurrency.progress_tracker` - :class:`ProgressTracker`,
  :class:`CancelledError`, ``PROGRESS_COMMON_EXTRAS`` and the ``sort_progress``
  / ``eval_progress`` / ``find_progress`` singletons.
* :mod:`~vtscore.concurrency.loading_tracker` - :class:`LoadingTasksTracker`,
  ``CANCELLED_ERROR`` and the ``loading_tasks`` / ``detector_loading_tasks``
  singletons.
* :mod:`~vtscore.concurrency.progress_cancel` -
  :func:`cancel_dataset_progress` and :func:`cancel_dataset_task`.

Private helpers live in the module that uses them: a test that patches one
patches that module, not this one.
"""

import inspect
import threading
from typing import Any, Callable

from vtscore.concurrency.loading_tracker import (
    CANCELLED_ERROR,
    DEFAULT_MAX_KEPT_FAILURES,
    LoadingTasksTracker,
    detector_loading_tasks,
    ingest_eta_hidden,
    loading_tasks,
)
from vtscore.concurrency.progress_cancel import (
    CANCEL_ACK_GRACE_SECONDS,
    cancel_dataset_progress,
    cancel_dataset_task,
)
from vtscore.concurrency.progress_tracker import (
    PROGRESS_COMMON_EXTRAS,
    CancelledError,
    ProgressTracker,
    eval_progress,
    find_progress,
    sort_progress,
)

__all__ = [
    "CANCEL_ACK_GRACE_SECONDS",
    "CANCELLED_ERROR",
    "DEFAULT_MAX_KEPT_FAILURES",
    "PROGRESS_COMMON_EXTRAS",
    "CancelledError",
    "LoadingTasksTracker",
    "ProgressCallback",
    "ProgressTracker",
    "cancel_dataset_progress",
    "cancel_dataset_task",
    "clear_thread_progress",
    "detector_loading_tasks",
    "eval_progress",
    "find_progress",
    "get_eval_progress",
    "get_find_progress",
    "get_sort_progress",
    "get_thread_progress",
    "ingest_eta_hidden",
    "loading_tasks",
    "noop_progress",
    "resolve_progress_callback",
    "set_thread_progress",
    "sort_progress",
    "update_eval_progress",
    "update_find_progress",
    "update_progress",
    "update_sort_progress",
]

_UNSET = object()  # sentinel for "caller did not provide this argument"


# ---------------------------------------------------------------------------
# Thread-local progress callback
# ---------------------------------------------------------------------------
# Background loading threads set a per-thread progress callback via
# set_thread_progress(); every library module that reports progress without an
# explicit callback resolves one through resolve_progress_callback().  This
# avoids monkey-patching module-level defaults and allows parallel loads to
# each report to their own ProgressTracker.
#
# There is deliberately no process-wide fallback sink.  One used to exist (the
# ``dataset_progress`` singleton behind the SSE ``dataset`` channel), and it
# was a standing source of phantom progress bars: work that finished, died, or
# never had a watcher still left the channel parked on its last message, and
# nothing downstream could tell that apart from a wedged import (#3167).  Work
# that wants to be seen binds a tracker for its thread; work that binds nothing
# is, by construction, work nobody is watching, so it reports into a no-op.

#: The shape every progress sink in the library implements:
#: ``(status, message, current, total) -> None``.
ProgressCallback = Callable[[str, str, int, int], None]

_thread_progress = threading.local()


def noop_progress(status: str, message: str = "", current: int = 0, total: int = 0) -> None:
    """Progress sink that discards everything.  The default when none is bound."""
    return None


def set_thread_progress(callback) -> None:
    """Set the progress callback for the current thread."""
    _thread_progress.callback = callback


def get_thread_progress():
    """Return the per-thread progress callback, or ``None``."""
    return getattr(_thread_progress, "callback", None)


def clear_thread_progress() -> None:
    """Remove the per-thread progress callback."""
    _thread_progress.callback = None


def resolve_progress_callback() -> ProgressCallback:
    """Return the progress sink the calling thread should report into.

    The per-thread callback installed by :func:`set_thread_progress` when one
    is bound, else :func:`noop_progress`.  Library code that takes an optional
    ``on_progress`` argument calls this to fill in the ``None`` case, so a
    caller that supplied nothing still reaches whichever tracker the enclosing
    load bound for its thread.
    """
    cb = get_thread_progress()
    return cb if cb is not None else noop_progress


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


def _accepts_extras(cb: Callable[..., None], kwargs: dict[str, Any]) -> bool:
    """Return ``True`` when *cb* can be called with every key in *kwargs*.

    A sink that declares ``**kwargs`` takes anything; otherwise every key must
    name a real keyword parameter.  A callable whose signature cannot be read
    (a C builtin, an exotic wrapper) is assumed not to accept them, so the
    four-argument call is what runs.
    """
    try:
        params = inspect.signature(cb).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return True
    return all(key in params and params[key].kind is not inspect.Parameter.POSITIONAL_ONLY for key in kwargs)


def update_progress(
    status: str,
    message: str = "",
    current: int = 0,
    total: int = 0,
    error: Any = _UNSET,
    step: Any = _UNSET,
    total_steps: Any = _UNSET,
) -> None:
    """Report dataset/import progress into the calling thread's sink.

    The public free-function spelling of :func:`resolve_progress_callback` for
    plugin authors (see ``vtscore/docs/extending/dataset-importers.md``): an
    importer that calls this from inside a running load reports onto that
    load's own :class:`ProgressTracker`, exactly as if it had accepted an
    ``on_progress`` argument.  Called with no tracker bound to the thread it is
    a no-op.

    ``error``, ``step`` and ``total_steps`` are forwarded only when supplied,
    and only when the bound sink accepts them (a :class:`ProgressTracker`'s
    ``update`` does; the plain four-argument callbacks the load pipeline
    installs do not, and would raise on them).  Acceptance is decided by
    inspecting the sink's signature rather than by catching :class:`TypeError`
    from the call, so a ``TypeError`` raised *inside* a sink still propagates
    instead of being retried as an arity mismatch.
    """
    cb = resolve_progress_callback()
    kwargs = _common_extras_kwargs(step, total_steps, error)
    if kwargs and _accepts_extras(cb, kwargs):
        cb(status, message, current, total, **kwargs)  # type: ignore[call-arg]
        return
    cb(status, message, current, total)


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
