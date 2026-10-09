"""Deprecated (#4667): the retired per-step timing recorder, now a no-op.

Arming ``VTSEARCH_TIMING_RECORD`` used to make every task wrapped in
:func:`record_task` append one JSONL row per step, which the tuning script
fitted into the per-environment profile :mod:`vtscore.timing.profile` read
back. That loop existed mainly to steady the remaining-time estimate on dataset
imports, and that estimate was removed (#4667), so the recorder, the fitter and
the tuning script went with it.

These names are kept so out-of-tree code keeps importing and calling them:
:func:`record_task` always returns a recorder whose methods do nothing,
:func:`recording_enabled` is always ``False``, and :func:`note_branch` /
:func:`note_no_encoder_load` do nothing. Nothing in this repository calls them.

The dataset-load pipeline's own env-gated profiler
(:mod:`vtscore.datasets.stages._load_profiler`, ``VTSEARCH_PROFILE_LOAD``) is
unaffected: it feeds the developer-side refit of the shipped load cost model.
"""

from __future__ import annotations

from typing import Any, Optional

#: Deprecated (#4667): nothing reads this environment variable any more.
RECORD_ENV_VAR = "VTSEARCH_TIMING_RECORD"


def note_branch(step: str, branch: str) -> None:
    """Deprecated (#4667): does nothing."""


def note_no_encoder_load() -> None:
    """Deprecated (#4667): does nothing."""


def reset_seen_models_for_tests() -> None:
    """Deprecated (#4667): does nothing; there is no residency ledger left."""


def recording_enabled() -> bool:
    """Deprecated (#4667): always ``False``."""
    return False


class TaskTimingRecorder:
    """Deprecated (#4667): a recorder whose every method does nothing.

    Keeps the surface the live recorder had (context manager, ``start``,
    ``bind_thread``, ``mark_branch``, ``disclaim_encoder``, ``set_scale``,
    ``finish``) so a call site written against it keeps running.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __enter__(self) -> "TaskTimingRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def start(self) -> None:
        pass

    def bind_thread(self) -> None:
        pass

    def mark_branch(self, step: str, branch: str) -> None:
        pass

    def disclaim_encoder(self) -> None:
        pass

    def set_scale(
        self,
        n: Optional[float] = None,
        size_mb: Optional[float] = None,
        embedder: Optional[str] = None,
    ) -> None:
        pass

    def finish(
        self,
        n: Optional[float] = None,
        size_mb: Optional[float] = None,
        ok: bool = True,
        embedder: Optional[str] = None,
    ) -> None:
        pass


def record_task(
    tracker: Any,
    task: str,
    *,
    media_type: str = "",
    embedder: str = "",
    status_phases: Optional[dict[str, str]] = None,
    auto_finish: bool = False,
    only_phases: Optional[tuple[str, ...]] = None,
) -> TaskTimingRecorder:
    """Deprecated (#4667): returns a :class:`TaskTimingRecorder` that does nothing."""
    return TaskTimingRecorder()
