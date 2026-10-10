# `vtscore.timing`

How each long-running task's progress bar splits across its steps.

Every long-running VTSearch operation - a dataset load, a detector load,
a text sort, a Find, a train-and-score, a promote - reports progress as
`step` / `total_steps` and paces its unified bar with a per-step
**weight vector** (`ProgressTracker.set_step_weights`). A weight vector
that is wrong in the same direction for a whole job makes a progress bar
race one phase and crawl the next.

This package is where those vectors come from: each task declares its
ordered steps and shipped default terms once, and asks for its vector at
its entry point instead of carrying a literal one.

Related docs: [`concurrency.md`](concurrency.md) for `ProgressTracker`
and the bar these weights drive.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/timing/tasks.py` | `TASKS` / `TaskSpec` - the canonical registry of task families, their ordered steps and default terms |
| `vtscore/timing/profile.py` | Turn the defaults into a weight vector: `step_weights`, `step_terms`; device helpers; the deprecated profile shims |
| `vtscore/timing/recorder.py` | Deprecated no-op shims for the retired timing recorder |

The package root re-exports the `profile`, `recorder` and `tasks` API
listed below (see its `__all__`).

---

## Where the weights come from

1. **The shipped defaults** in `vtscore/timing/tasks.py`. Default terms
   are **pseudo-seconds**: only their ratios are meaningful.
2. **`dataset_load`** is the exception: it ships no flat terms, because
   its model is the measured affine table in
   `vtscore/datasets/stages/_load_cost_model.py`, which is already
   `n`-aware per `(device, media_type, embedder)` cell. That table is
   refitted by developers with the scripts under `scripts/profiling/`.
3. **The caller's fallback** (usually equal weighting), if the task is
   unknown entirely.

Nothing here touches disk.

---

## Task registry

Every task driving a `step`/`total_steps` bar registers a `TaskSpec` in
`TASKS`.

| Field | Meaning |
|-------|---------|
| `name` | Stable identifier - the key callers pass to `step_weights` |
| `steps` | Ordered cost-*phase* names |
| `step_index` | 1-based tracker step each phase reports against, parallel to `steps` |
| `tracker_steps` | How many step numbers the task reports - the length of the weight vector |
| `scale` | Human description of what `n` counts |
| `default_terms` | Shipped pseudo-seconds, parallel to `steps` (may be empty) |
| `media_default_terms` | Per-media-type overrides of `default_terms` (`{"audio": (...)}`), each parallel to `steps`; read through `defaults_for(media_type)`, which falls back to `default_terms` for any media type not named |
| `byte_scaled` | Which phases scale with archive bytes rather than item count (descriptive) |
| `loads_encoder` | Whether a run can pay a cold encoder load (descriptive) |

Registered today: `dataset_load`, `dataset_open`, `dataset_promote`,
`dataset_stage`, `detector_load`, `text_sort`, `find`,
`train_and_score`.

**Phases versus tracker steps.** Usually they are the same and
`step_index` is just `(1, 2, 3, …)`. A task may model one step as
several phases - `dataset_load`'s step 1 covers both the network
transfer and the archive unpack - in which case the phases share a
tracker step and `step_weights` sums their terms back into that slot.

`dataset_open` is the one task with a per-media override: reading an
audio pickle is a much larger share of an open than reading an image
pickle, so audio ships `(0.40, 0.60)` beside the task-wide `(0.15, 0.85)`
(#4105).

Adding a long-running task means adding a `TaskSpec` here, then calling
`step_weights(...)` at the task's entry point instead of writing a
literal vector.

---

## Using it

```python
from vtscore.timing import step_weights

weights = step_weights("text_sort", media_type="image", fallback=[0.2, 0.8])
if weights:
    tracker.set_step_weights(weights)
```

The vector has one entry per tracker step and sums to 1, ready for
`set_step_weights`. Pass a `fallback` - `step_weights` returns it when
the task is unknown or ships no terms.

### Steps this run will skip

A default answers "how big a share of the job is this step". It cannot
answer "does this step happen at all", and where a step forks on process
state the second question is the one that decides the bar. A text sort's
`load_model` is seconds on a process's first sort and **exactly zero**
on every later one (#3596).

The caller usually knows which branch it is on before it starts. Name
the steps that will not run and they are priced at zero for this run:

```python
weights = step_weights(
    "text_sort",
    media_type="image",
    skip_steps=() if encoder_is_cold else ("load_model",),
)
```

| Function | Description |
|----------|-------------|
| `step_weights(task, *, media_type, skip_steps, fallback, ...)` | Normalised per-tracker-step weights, or *fallback* |
| `step_terms(task, *, media_type, skip_steps, ...)` | The same before normalisation - the shipped terms keyed by step name, or `None` |
| `known_tasks()` / `task_spec(name)` | Registry lookups |
| `cell_keys(device, media_type, embedder)` / `normalize_device(device)` | `device|media|embedder` key resolution, most specific first, and the coarse `cuda`/`cpu` device key |

Both lookup functions still accept `device`, `embedder`, `n`, `size_mb`
and `branch`, which selected and scaled the retired profile's cells
(below). The shipped defaults have none of those axes, so they are
ignored.

---

## Deprecated: the per-environment profile

The admin-measured profile (`VTSEARCH_TIMING_PROFILE`, fed by the
`VTSEARCH_TIMING_RECORD` recorder) was retired with the dataset-import ETA
it chiefly existed to steady (#4667); neither environment variable is read.
The public names are kept as **deprecated no-ops** so out-of-tree code
keeps importing them:

| Name | Now |
|------|-----|
| `active_profile()` / `reload_profile(path=None)` / `parse_profile(raw, source="")` | Always `EMPTY_PROFILE` (falsy) |
| `TimingProfile`, `EMPTY_PROFILE`, `StepCoeffs` | Kept as types; `StepCoeffs` still evaluates `a + b·n + per_mb·archive_mb` |
| `profile_covers(task)` | Always `False` |
| `slot_shares(task, step, ...)` | Always `None` |
| `record_task(...)` | A `TaskTimingRecorder` whose every method does nothing |
| `recording_enabled()` | Always `False` |
| `note_branch(step, branch)` / `note_no_encoder_load()` | Do nothing |
| `CHEAP_BRANCHES` / `DEAR_BRANCHES` | Unchanged constants; nothing reads them |
| `profile.PROFILE_ENV_VAR`, `profile.SCHEMA_NAME`, `profile.SCHEMA_VERSION`, `recorder.RECORD_ENV_VAR` | Unchanged constants; nothing reads them |

The studies that measured the profile are written up under
[`docs/experiments/`](../../../docs/experiments/), for example
[the r² study](../../../docs/experiments/2026-09-02-timing-r2-3345/REPORT.md).
