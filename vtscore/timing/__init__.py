"""Per-task timing model: how each long-running task's bar splits across its steps.

Every long-running VTSearch operation — a dataset load, a detector load, a text
sort, a Find, a train-and-score, a promote — reports progress as
``step``/``total_steps`` and paces its unified bar with a per-step **weight
vector** (see :meth:`vtscore.concurrency.progress.ProgressTracker.set_step_weights`).
A vector that is wrong in the same direction for the whole job is exactly what
makes a progress bar race one phase and crawl the next.

Each task declares its ordered step names and shipped default terms in
:mod:`vtscore.timing.tasks`; :func:`step_weights` normalizes those into the
vector the tracker takes. ``dataset_load`` is the exception: its shipped model
is the measured, ``n``-aware affine table in
:mod:`vtscore.datasets.stages._load_cost_model`. An unknown task gets the
caller's own fallback (usually equal weighting).

There used to be a third, most specific layer: an admin-measured
**per-environment profile** (``VTSEARCH_TIMING_PROFILE``), written by a tuning
script from rows the ``VTSEARCH_TIMING_RECORD`` recorder collected. It was
retired with the dataset-import ETA it chiefly existed to steady (#4667). The
names it exported survive as deprecated no-ops — see
:mod:`vtscore.timing.profile` and :mod:`vtscore.timing.recorder`.

Nothing here touches disk.

See ``../docs/packages/timing.md`` for the package reference.
"""

from __future__ import annotations

from vtscore.timing.profile import (
    StepCoeffs,
    TimingProfile,
    active_profile,
    cell_keys,
    known_tasks,
    normalize_device,
    profile_covers,
    reload_profile,
    slot_shares,
    step_terms,
    step_weights,
)
from vtscore.timing.recorder import note_branch, note_no_encoder_load, record_task, recording_enabled
from vtscore.timing.tasks import CHEAP_BRANCHES, TASKS, DEAR_BRANCHES, TaskSpec, task_spec

__all__ = [
    "CHEAP_BRANCHES",
    "DEAR_BRANCHES",
    "TASKS",
    "StepCoeffs",
    "TaskSpec",
    "TimingProfile",
    "active_profile",
    "cell_keys",
    "known_tasks",
    "normalize_device",
    "note_branch",
    "note_no_encoder_load",
    "profile_covers",
    "record_task",
    "recording_enabled",
    "reload_profile",
    "slot_shares",
    "step_terms",
    "step_weights",
    "task_spec",
]
