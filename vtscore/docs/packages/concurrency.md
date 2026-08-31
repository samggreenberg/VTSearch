# `vtscore.concurrency`

The runtime plumbing for background work: independent layers for running
a task off-request, reporting how far it has got, letting the user cancel
it, and capping how many run at once.

Related docs: [`state.md`](state.md) for the contexts these jobs and
trackers operate against; [`security.md`](security.md) for the
safe-load helpers used during dataset import; [`timing.md`](timing.md)
for the per-step duration model that turns a step index into an ETA.

**Import from the defining module.** `vtscore/concurrency/` has no
`__init__.py` - it is a PEP 420 implicit namespace package, so it exports
nothing of its own and `from vtscore.concurrency import JobManager` raises
`ImportError`. Every snippet below imports from the module that defines the
name: `async_jobs`, `progress`, `memory_budget`, or `events`. The same is
true of `vtscore.security`.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/concurrency/async_jobs.py` | Single-slot async job manager: coalescing background tasks with a result slot |
| `vtscore/concurrency/progress.py` | `ProgressTracker`, `LoadingTasksTracker`, the module-level singletons, cooperative cancellation |
| `vtscore/concurrency/events.py` | Server-Sent Events stream over the progress channels |
| `vtscore/concurrency/notifications.py` | `notify()` - one-off user-facing messages, rendered as toasts by the app |
| `vtscore/concurrency/gate.py` | `ConcurrencyGate` - a semaphore whose limit is re-read on every acquisition |
| `vtscore/concurrency/memory_budget.py` | Cap fan-out so peak per-worker memory fits a fraction of available RAM |

- [Two kinds of "progress"](#two-kinds-of-progress)
- [Async jobs](#async-jobs)
- [`ProgressTracker`](#progresstracker)
- [`LoadingTasksTracker`](#loadingtaskstracker)
- [Module-level singletons](#module-level-singletons)
- [Per-thread progress callback](#per-thread-progress-callback)
- [Cancellation contract](#cancellation-contract)
- [Memory budget](#memory-budget)
- [`ConcurrencyGate`](#concurrencygate)
- [`events.py`](#eventspy)
- [User notifications](#user-notifications)

---

## Two kinds of "progress"

The word "progress" shows up in two unrelated parts of `vtscore`:

| Layer | Module | What it tracks |
|-------|--------|----------------|
| Long-running operation progress | `vtscore.concurrency.progress` | Dataset load / sort / eval / find - coarse-grained `(current, total, status, message)` for UI bars and cancel buttons |
| Labeling-session analyzer | `vtscore.detectors.labeling_progress` | Per-step trained model cache + stopping-condition metrics |

The labeling-session analyzer is **not** in this package. If you see
`clear_progress_cache()` or `recreate_model_at_time()` referenced
anywhere, those are the *analyzer*, not the long-running-operation
tracker.

---

## Async jobs

`vtscore/concurrency/async_jobs.py` exposes a single-slot background-job
manager. Flask runs with `gthread` workers; endpoints doing GIL-bound
Python work starve unrelated requests like `/api/votes` polls. The job
manager moves the heavy work onto a daemon thread so the request handler
returns immediately with a job ID.

### `AsyncJob` (dataclass)

State container for one background job (`async_jobs.py`). Fields:
`job_id` (UUID4 hex), `signature` (caller-supplied fingerprint),
`status` (`"pending"` / `"running"` / `"done"` / `"error"` /
`"cancelled"`), `result`, `error`, `current` / `total` / `message`
(progress counters), `started_at`, `cancel_event` / `done_event`
(`threading.Event`), `user` (captured at `start()` for per-user settings
resolution), `dataset_id` / `detector_id` (captured at `start()`;
consumed by `list_active_pairs()`).

Methods: `cancel()` sets `cancel_event`; `is_cancelled` reads it;
`update_progress(current, total, message="")` updates the three progress
fields (single-writer per job).

### `JobManager`

One background job runs at a time; one **pending slot** coalesces
follow-up requests. A second `start()` while a job is in flight stashes
the new `(signature, target)` in the pending slot; further `start()`
calls overwrite the slot (latest wins) and return the same pending
`AsyncJob`. When the running job finishes, the pending job is promoted
and spawned automatically.

```python
from vtscore.concurrency.async_jobs import JobManager

mgr = JobManager("my-task", max_history=8)

def my_target(job):
    for i in range(100):
        if job.is_cancelled:
            return
        job.update_progress(i, 100, f"step {i}")
    job.result = {"hits": 42}

job = mgr.start(
    signature=("dataset_a", "detector_x"),
    target=my_target,
    dataset_id="dataset_a",
    detector_id="detector_x",
)
job.done_event.wait(timeout=60)
if job.status == "done":
    print(job.result)
```

Result caching: the most recent successfully-completed job is kept by
signature. `mgr.cached_for(signature)` returns it if its signature
matches - the "re-sort without new votes is free" fast path.

Cancellation: `job.cancel()` sets the event; the target must check
`job.is_cancelled` cooperatively. If the target returns while the event
is set, the manager records `status = "cancelled"`. If the target
raises, `status = "error"` with `job.error` set - and the pending slot
is still promoted so a queued follow-up runs. The manager is
thread-safe (internal `RLock`) but **not** re-entrant - the target must
not call `mgr.start()` on the same manager.

When a job spawns its daemon thread, the manager enters the Flask-free
`vtscore.state.current_user.thread_user(job.user)` context manager
before invoking the target so per-user settings resolution works from
inside the worker; the context manager clears the thread-local on exit.
`JobManager` never imports from `vtsearch.auth` — the whole point of the
`current_user` extraction is that `start()` stands alone. Library-only
consumers leave `user=None`.

### Module-level managers and `JOB_MANAGERS`

```python
# vtscore/concurrency/async_jobs.py
learned_sort_jobs = JobManager("learned-sort")
eval_jobs = JobManager("eval-train-score")

JOB_MANAGERS: dict[str, JobManager] = {
    "learned-sort": learned_sort_jobs,
    "eval": eval_jobs,
}
```

`JOB_MANAGERS` is what `/api/jobs/active` reads. The string keys are
public job-type names exposed in the response, so they're stable across
releases. Library consumers can add their own manager and register it
here for the active-jobs surface to pick up.

`list_active_pairs()` returns
`[{dataset_id, detector_id, job_types}, ...]` for every (dataset,
detector) pair with at least one running or pending job across every
registered manager. Jobs missing `dataset_id` / `detector_id` are
dropped. `reset_all_async_jobs_for_tests()` walks `JOB_MANAGERS` and
clears state.

---

## `ProgressTracker`

Thread-safe progress tracker for a single long-running operation
(`progress.py`). Each instance holds its own lock, data dict, cancel
event, and optional subscriber callbacks.

```python
from vtscore.concurrency.progress import ProgressTracker

tracker = ProgressTracker(extra_fields={"error": None, "eta_seconds": None})
tracker.update("loading", "Reading file", current=10, total=100)
snapshot = tracker.get()
# {"status": "loading", "message": "...", "current": 10, "total": 100,
#  "error": None, "eta_seconds": None}
```

| Method | Description |
|--------|-------------|
| `update(status, message, current, total, **kwargs)` | Atomic update; only declared extra fields in `kwargs` are honoured |
| `get()` | Shallow copy of the internal data dict |
| `subscribe(cb)` / `unsubscribe(cb)` | Register a callback fired with each update snapshot (outside the lock) |
| `cancel()` | Set the cancel event |
| `check_cancelled()` | Raise `CancelledError` if cancel event is set |
| `is_cancelled` | Read the event |
| `reset_cancel()` | Clear the event - call at the start of each new operation |

`extra_fields` declares keys `update()` will honour beyond the base
`(status, message, current, total)` tuple. Unrecognised keys are
silently dropped, so a single `update_progress()` call site can supply
kwargs some trackers care about and others don't. Every shipped tracker
uses `_PROGRESS_COMMON_EXTRAS` (`progress.py`):

```python
_PROGRESS_COMMON_EXTRAS = {
    "step": None, "total_steps": None,      # sub-step counter
    "error": None, "eta_seconds": None,     # eta_seconds auto-populated
}
```

`dataset_progress` still declares `"staging_result"` (retained for the
legacy free-function API and its test), but the staging flow itself now
runs through per-task trackers created via
`LoadingTasksTracker.create_task(..., extra_fields={"staging_result": None})`,
so two concurrent staging imports no longer collide on one channel.

**ETA:** when `extra_fields` includes `"eta_seconds"`, every `update()`
recomputes a smoothed ETA. The tracker keeps a phase key
`(status, total)`; whenever it changes or `current` resets backwards
the clock resets. After 5 seconds of elapsed work the raw ETA is
`(elapsed / completed) * (total - current)` smoothed with an EMA
(`alpha = 0.3`). Before then, `eta_seconds` stays `None`.

**Subscribers:** `subscribe(cb)` registers a callback fired with a
snapshot after every `update()`, synchronously on the producer thread
**outside** the lock. Non-blocking and exception-safe; exceptions are
swallowed. This is what `events.py` uses.

---

## `LoadingTasksTracker`

A bag of named `ProgressTracker`s, each with a creation timestamp
(`progress.py`). Used to multiplex concurrent dataset / detector
loads - the dashboard polls `list_tasks()` to show one row per loading
operation.

```python
from vtscore.concurrency.progress import LoadingTasksTracker

bag = LoadingTasksTracker()
tracker = bag.create_task(
    task_id="ds_load_42", name="loading my-dataset.pkl",
    dataset_id="my-dataset", media_type="audio", embedder="clap",
)
tracker.update("downloading", "Fetching ...", 0, 100)
bag.mark_finished("ds_load_42")
```

Methods: `create_task(task_id, ...)` (register and return tracker),
`get_tracker(task_id)`, `mark_finished(task_id)` (schedules pruning),
`remove_task(task_id)`, `cancel_task(task_id)` / `cancel_all()`,
`set_dataset_id(task_id, ds_id)` (late-bind once known), `list_tasks()`
(snapshot; prunes stale finished entries), `has_active_tasks()`,
`subscribe(cb)` / `unsubscribe(cb)`.

Stale-prune policy: finished tasks without errors are removed 5 seconds
after `mark_finished()`; tasks with errors are kept for 30 seconds so
the polling frontend can display them. The prune runs lazily inside
`list_tasks()`.

---

## Module-level singletons

```python
# vtscore/concurrency/progress.py
dataset_progress = ProgressTracker(extra_fields={**_PROGRESS_COMMON_EXTRAS, "staging_result": None})
sort_progress    = ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))
eval_progress    = ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))
find_progress    = ProgressTracker(extra_fields=dict(_PROGRESS_COMMON_EXTRAS))

loading_tasks          = LoadingTasksTracker()
detector_loading_tasks = LoadingTasksTracker()
```

Each single-channel tracker has a thin module-level update/get pair so
callers don't have to import the tracker itself:

| Tracker | Update | Get | Cancel |
|---------|--------|-----|--------|
| `dataset_progress` | `update_progress(...)` | `get_progress()` | `cancel_dataset_progress()` |
| `sort_progress` | `update_sort_progress(...)` | `get_sort_progress()` | - |
| `eval_progress` | `update_eval_progress(...)` | `get_eval_progress()` | - |
| `find_progress` | `update_find_progress(...)` | `get_find_progress()` | - |

```python
from vtscore.concurrency.progress import update_progress, get_progress, check_dataset_cancelled

update_progress("loading", "Starting", 0, 100, step=1, total_steps=3)
check_dataset_cancelled()   # raises CancelledError if cancel was requested
update_progress("loading", "Halfway", 50, 100)
update_progress("done", "Finished", 100, 100)
```

`get_progress()` is asymmetric: it checks the per-task `loading_tasks`
first (used by parallel dataset loading) and only falls back to
`dataset_progress` when no per-task entry is active. A caller that wants
progress shown on the dashboard should prefer `loading_tasks.create_task()`
over the singleton when parallel loads are possible.

The free functions honour a sentinel default (`_UNSET = object()`) for
optional extras: only fields explicitly supplied by the caller are
forwarded, so omitted fields are left unchanged - true update/merge
semantics rather than "every call clobbers every field".

---

## Per-thread progress callback

Background loading threads inside `vtscore.datasets` and
`vtscore.embedding.loader` write progress through a per-thread callback,
not directly to a tracker. Module-level helpers check the thread-local
first and fall back to a global default when nothing is set.

```python
# vtscore/concurrency/progress.py
def set_thread_progress(callback) -> None: ...
def get_thread_progress(): ...
def clear_thread_progress() -> None: ...
```

This mirrors `vtscore.media.set_thread_progress_callback` (a separate
channel for the media-registry's callback). Both exist so multi-threaded
consumers (e.g. scoring two detectors in parallel) don't have one
thread clobber another's callback.

```python
from vtscore.concurrency.progress import set_thread_progress, clear_thread_progress, loading_tasks

def worker(task_id):
    tracker = loading_tasks.get_tracker(task_id)
    set_thread_progress(lambda status, msg, cur, tot: tracker.update(status, msg, cur, tot))
    try:
        # ... library code that calls update_progress internally ...
    finally:
        clear_thread_progress()
```

---

## Cancellation contract

Cancellation is **cooperative**. Setting the event does not preempt the
running thread; the thread must check periodically and return early.

`ProgressTracker.cancel()` sets the event; `check_cancelled()` raises
`CancelledError`. The canonical pattern inside a loop:

```python
from vtscore.concurrency.progress import check_dataset_cancelled, update_progress

for i, item in enumerate(items):
    check_dataset_cancelled()           # raises CancelledError if cancelled
    update_progress("loading", f"item {i}", i, len(items))
    # ... do work ...
```

`AsyncJob` cancellation is the analogous pattern but uses the job's own
event: the target function checks `job.is_cancelled` and returns.

**Cancellation is one-shot per operation.** Call `reset_cancel()` at the
start of each new operation so a previous cancellation doesn't
immediately abort the next run.

`cancel_dataset_progress()` is a convenience that cancels every active
task in `loading_tasks` (which now includes staging imports) **and** the
legacy `dataset_progress` singleton, so a clean abort touches both
surfaces regardless of which one a given operation reports through.

---

## Memory budget

`vtscore/concurrency/memory_budget.py` exposes one function:

```python
def cap_workers_by_memory(
    n_items: int, embed_dim: int, *, max_workers: int,
    bytes_per_element: int = 4, budget_fraction: float = 0.25,
) -> int:
    """Cap max_workers so peak per-worker memory fits in the budget."""
```

Auto-detect and auto-process endpoints fan out across all medias inside
a `ThreadPoolExecutor`. Each worker materialises an `(N, D)` fp32
tensor, which at `N=100k, D=1152` is ~450 MB per worker. Eight
unconstrained workers can push multi-GB transient allocations.

The helper computes per-worker bytes as
`n_items * embed_dim * bytes_per_element`, takes 25% of currently
available memory as the budget (default), and returns
`max(1, min(max_workers, budget // per_worker))`.

```python
from vtscore.concurrency.memory_budget import cap_workers_by_memory

n_workers = cap_workers_by_memory(
    n_items=len(medias), embed_dim=512, max_workers=8,
)  # 1..8 depending on free RAM
```

Available-memory detection prefers `/proc/meminfo`'s `MemAvailable:`,
falls back to `sysconf(SC_AVPHYS_PAGES) * SC_PAGE_SIZE`, and lastly
uses a 1 GiB floor. The function always returns at least 1.

---

## `ConcurrencyGate`

`vtscore/concurrency/gate.py` is a semaphore whose cap is a **callable
re-read on every acquisition**, not a number fixed at construction:

```python
class ConcurrencyGate:
    def __init__(self, get_limit: Callable[[], int]) -> None: ...
    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool: ...
    def release(self) -> None: ...
    @property
    def active(self) -> int: ...
```

That is the whole point of it existing next to `threading.Semaphore`.
The dataset-load pipeline gates concurrent downloads and concurrent
embeddings on settings the user can change mid-run; with a plain
semaphore the new value would apply only to the next process. Here it
takes effect immediately for queued and future tasks. Already-running
tasks are never preempted, and the limit is floored at 1 so a
misconfigured setting cannot deadlock the pipeline.

```python
from vtscore.concurrency.gate import ConcurrencyGate

gate = ConcurrencyGate(lambda: CoreConfig.from_settings().concurrent_downloads)
if gate.acquire(timeout=30):
    try:
        ...
    finally:
        gate.release()
```

`acquire()` is not a context manager - pair it with `release()` in a
`finally`. The `waiter_parked` event on the instance is a test hook (it
lets a test wait for "a waiter is queued" instead of sleeping a fixed
race window); production code should not read it.

---

## `events.py`

`vtscore/concurrency/events.py` is the SSE wiring layer: it owns
`_TRACKER_CHANNELS` (`dataset`, `sort`, `find`, `eval`) and
`_TASK_CHANNELS` (`loading-tasks`, `detector-loading-tasks`), subscribes
to each tracker's notify stream, and emits
`event: <channel>\ndata: <json>\n\n` frames suitable for an SSE response.

| Function | Description |
|----------|-------------|
| `initial_snapshot() -> list[str]` | SSE frames a freshly-connected client should receive first |
| `stream_progress_events(*, heartbeat_seconds=5.0, max_queue=1024)` | Generator yielding SSE strings until disconnect |

The generator subscribes to each tracker, drains a private bounded
queue per client, and unsubscribes in a `finally` block on disconnect.
Heartbeats every 5 seconds keep the connection alive and re-emit the
task channels so clients see finished tasks vanish once the stale-prune
window elapses. This module is thin wiring - most library consumers
don't need to touch it.

It also carries one channel that is *not* a tracker: `NOTIFICATION_CHANNEL`
(`"notification"`), fed by the broker below. It is absent from
`initial_snapshot()` and never re-emitted on a heartbeat, because a
notification is something that happened once rather than a state to
converge on.

---

## User notifications

`vtscore/concurrency/notifications.py` answers a question the trackers
can't: how does code tell the user something *without stopping*?

A tracker publishes state, and an exception ends the operation. Neither
fits "we skipped 3 unreadable files but the other 900 imported fine" -
raising throws away the 900, and a log line is written where nobody is
looking. `notify()` is the third option: keep going, and put the message
in front of the user (a toast in the app; a stderr line under the CLI).

```python
from vtscore.concurrency.notifications import notify

notify(
    "Skipped 3 unreadable files",
    level="warning",                       # info | success | warning | error
    detail="page_2.pdf, page_9.pdf, notes.pdf",
    source="Server Folder",
)
```

| Name | Description |
|------|-------------|
| `notify(message, *, level="info", detail=None, source=None) -> Notification` | Publish and log one message. Never raises |
| `Notification` | Frozen dataclass: `id`, `level`, `message`, `detail`, `source`, `timestamp`, `.to_dict()` |
| `NotificationBroker` | `subscribe` / `unsubscribe` / `publish` / `subscriber_count` / `clear_subscribers` |
| `notifications` | The process-wide broker singleton |

Plugin authors get the same thing with `source` prefilled - see
[`plugins.md`](plugins.md) and `PluginBase.notify`.

**Delivery is live-only.** Every client with an open `/api/events` stream
at the moment of the call receives the frame; one that connects a moment
later does not, and there is no replay buffer. Notifications narrate work
the user is currently watching. Anything that has to survive a page reload
belongs in a task's terminal payload or in persisted state instead.

**It cannot fail.** An unknown `level` degrades to `"info"`, an over-long
message is truncated, a blank message is logged but not shown, and a
subscriber that raises is swallowed. A call whose whole purpose is to
*avoid* interrupting the caller must not become the interruption.

**It always logs**, at a severity matching the level, so a headless run
still has a record when no browser is attached.
