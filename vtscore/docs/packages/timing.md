# `vtscore.timing`

How long each step of a long-running task will take, measured rather
than guessed.

Every long-running VTSearch operation - a dataset load, a detector load,
a text sort, a Find, a train-and-score, a promote - reports progress as
`step` / `total_steps` and paces its unified bar with a per-step
**weight vector** (`ProgressTracker.set_step_weights`). Those vectors
used to be hand-guessed constants sitting next to each task's code. A
guess that is wrong in the same direction for a whole job is exactly
what makes a progress bar race one phase, crawl the next, and walk its
ETA *upward* while the user watches.

This package replaces the guesses with a per-environment cost model:
a deployment measures itself once, and every instance in that
environment predicts its own timings thereafter.

Related docs: [`concurrency.md`](concurrency.md) for `ProgressTracker`
and the bar these weights drive.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/timing/tasks.py` | `TASKS` / `TaskSpec` - the canonical registry of task families and their ordered steps |
| `vtscore/timing/profile.py` | Load, resolve and apply a profile: `step_weights`, `step_terms`, `slot_shares`, `active_profile` |
| `vtscore/timing/recorder.py` | Env-gated recorder that measures what each step really took |
| `vtscore/timing/fit.py` | Turn recorded timings into a profile document (the writer for the format `profile.py` reads) |

---

## The cost model

Each step gets an affine cost:

```
T_step ≈ a + b · n + per_mb · archive_mb
```

`n` is the task's natural scale variable - items to embed, labels to
train on, medias to score. `archive_mb` covers the byte-scaled phases of
a download.

Coefficients are keyed by a **cell**: `(device, media_type, embedder)`.
That granularity is the point. The same step costs wildly different
amounts on a V100 versus a laptop CPU, and on 200-character texts versus
30-second videos; a single global constant cannot be right for both.

## The three-layer resolution

A cell resolves most-specific-first:

1. **The admin profile** - a JSON file named by
   `VTSEARCH_TIMING_PROFILE`, produced by
   `scripts/profiling/tune_timing_profile.py` on the hardware that will
   actually serve the app.
2. **The shipped defaults** in `vtscore/timing/tasks.py` (and, for
   `dataset_load`, the calibrated table in
   `vtscore/datasets/stages/_load_cost_model.py`). These reproduce the
   pre-profile hand-tuned weights *exactly*, so an instance with no
   profile paces as it always did.
3. **Equal weighting**, if the task is unknown entirely.

Default terms are **pseudo-seconds**: only their ratios are meaningful,
because nobody measured them. A profile replaces them with real seconds,
which is what makes the ETA stop drifting.

Nothing persists at runtime and nothing is cached across processes: the
profile is read once per process at first use, and `reload_profile()`
re-reads it.

---

## Task registry

Every task driving a `step`/`total_steps` bar registers a `TaskSpec` in
`TASKS`. The registry is the shared vocabulary between three parties
that would otherwise drift apart: the **task code**, which needs one
weight per tracker step; the **recorder**, which must label a measured
duration with a step name; and the **tuning script**, which fits per
step and writes the profile JSON keyed by those same names.

| Field | Meaning |
|-------|---------|
| `name` | Stable identifier - the profile JSON's task key, the label in recorded rows, and the `--tasks` selector. **Never rename one** without migrating the profiles admins have already generated |
| `steps` | Ordered cost-*phase* names; profile coefficients are keyed by these |
| `step_index` | 1-based tracker step each phase reports against, parallel to `steps` |
| `tracker_steps` | How many step numbers the task reports - the length of the weight vector |
| `scale` | Human description of what `n` counts |
| `byte_scaled` | Which phases get a per-MB rate instead of a per-item slope |

Registered today: `dataset_load`, `dataset_open`, `dataset_promote`,
`dataset_stage`, `detector_load`, `text_sort`, `find`,
`train_and_score`.

**Phases versus tracker steps.** Usually they are the same and
`step_index` is just `(1, 2, 3, …)`. A task may model one step as
several phases that scale differently - `dataset_load`'s step 1 covers
both the network transfer and the archive unpack, both byte-scaled but
at very different rates - in which case the phases share a tracker step
and `step_weights` sums their predicted seconds back into that slot.

`dataset_load` deliberately carries **no** default terms: its shipped
model is the measured affine table in `_load_cost_model`, which is
already `n`-aware per cell and better than any flat vector.

Adding a long-running task means adding a `TaskSpec` here, then calling
`step_weights(...)` at the task's entry point instead of writing a
literal vector.

---

## Using it

```python
from vtscore.timing import step_weights

weights = step_weights(
    "text_sort",
    device=device, media_type="image", embedder="siglip",
    n=len(medias),
    fallback=[0.2, 0.8],
)
if weights:
    tracker.set_step_weights(weights)
```

The vector has one entry per tracker step and sums to 1, ready for
`set_step_weights`. Pass a `fallback` - `step_weights` returns it when
the task is unknown or nothing resolves.

| Function | Description |
|----------|-------------|
| `step_weights(task, *, device, media_type, embedder, n, size_mb, fallback)` | Normalised per-tracker-step weights, or *fallback* |
| `step_terms(...)` | The same prediction before normalisation - raw predicted seconds per phase |
| `slot_shares(task, step, ...)` | Measured sub-stage shares *within* one step, for steps that pace several ordered sub-stages behind one number (today only the dataset load's `finalize`). Raw weights; the consumer normalises |
| `profile_covers(task)` | Whether the active profile has any measured cell for *task*. Public API with no in-repo caller - for out-of-tree callers that want to branch on coverage before asking for weights |
| `active_profile()` / `reload_profile(path=None)` | The parsed profile; re-read it |
| `known_tasks()` / `task_spec(name)` | Registry lookups |
| `cell_keys(device, media_type, embedder)` / `normalize_device(device)` | Cell-key resolution, most specific first |

---

## Recording

Arm the recorder by pointing `VTSEARCH_TIMING_RECORD` at a JSONL path.
Each task wrapped in `record_task` then appends one row per step:

```json
{"task": "text_sort", "device": "cuda", "cuml": true, "media_type": "image",
 "embedder": "siglip", "n": 12403, "size_mb": 0.0, "step": "score",
 "seconds": 1.83, "ok": true}
```

Because the recorder sits behind an env var, an admin has two ways to
gather data and both produce the same file:

- **Drive it.** Run the tuning script, which exercises each task family
  against exemplar datasets with the recorder armed.
- **Watch it.** Set `VTSEARCH_TIMING_RECORD` on the real server and let
  real users generate the timings. This measures the production mix
  directly - the datasets people actually load, at the sizes they
  actually are - which no synthetic sweep reproduces.

When disarmed the cost is one `os.environ` lookup per task and a couple
of no-op method calls: no tracker subscription, no file handle.

### Two recorders, side by side

The dataset-load pipeline carries an older, richer recorder
(`vtscore/datasets/stages/_load_profiler.py`) that additionally
distinguishes cold from warm model loads, cold from cached downloads,
and the sub-slots inside finalize. Both run on the same load. They
answer different questions, are armed by different env vars, and write
different files - and the fitter reads both row shapes, so a
pre-existing dataset-load calibration sweep folds into a new profile
rather than being re-measured.

---

## Fitting

`vtscore/timing/fit.py` is the writer for the format `profile.py` reads.
It lives next to the reader so the two cannot drift: a schema change has
to be made in one directory or it will not round-trip.
`normalize_row` flattens both recorder shapes into one.

The fit is deliberately plain. Per `(task, cell, step)`:

- **Byte-scaled steps** (a download and its unpack) get a per-MB rate:
  the median of `seconds / archive_mb`. Regressing these against item
  count would ask `n` to explain something it cannot see - 500 videos
  and 500 text files are the same `n` and two orders of magnitude apart
  in bytes.
- **Everything else** gets ordinary least squares against `n`: the
  intercept is what the step costs at all (loading an encoder, opening a
  file) and the slope is what each additional item adds.
- A fit with no spread in `n`, or one that comes back with a **negative**
  slope (noise beating signal on a short step), collapses to the median
  seconds with no slope. A confidently wrong slope extrapolates badly at
  sizes the sweep never visited; a flat median merely stops improving.
