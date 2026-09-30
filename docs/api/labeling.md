# Labeling & Diversity

[← Back to API index](../API.md)

> These endpoints read/mutate the active dataset and detector via the
> [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).

---

## The Precision Floor

### Get / set the precision floor

```
GET /api/min-precision
```

→ `{"min_precision": 0.5, "status": "unchecked", "count": 32, "range": null, "schedule": {"candidate": 32, "rounds": 1, "picks": 5}, "threshold": 0.5123, "n_returned": 41}`

```
POST /api/min-precision
```

**Body:** `{"min_precision": 0.75}`

The precision floor is the share of what the detector returns that should be
right. The line always keeps a **set**: the top `count` unvoted items of the
ranking the detector last scored. Before any check that set is the floor's
**starting candidate** - the top 128 at 10%, the top 64 at 25%, the top 32 at
50% and above - and nothing has measured how much of it is right. A **spot
check** ([below](#the-spot-check)) measures it: the user votes on random picks
from the set, and a Clopper-Pearson bound on those picks either confirms the
floor or, round by round, halves the set down to the top 32 and says how close
it got. The floor is kept per detector and seeded from the user's
`min_precision` setting, which is `0.5` until the user changes it. A number is
clamped to `[0.01, 1]`; a boolean, non-number or `null` is a 422: every
detector has a floor. It is a pure cutoff knob - the active detector's line
moves to the set the new floor keeps without retraining and, in Find mode, the
unverified items re-split - and the same value is settable as `min_precision`
on `PUT /api/settings`. Both verbs return the new line in the same round trip,
so the app's floor control moves its line without re-scoring. The control
offers three floors, 10%, 50% and 90%, as unnumbered radios along a False
Positives to False Negatives spectrum. The API takes any value in range, but
when the app shows a detector whose floor is off that list, the control moves
it to the nearest of the three.

The floor replaced the Inclusion knob, and `/api/inclusion` is gone. Inclusion
survives only as the internal unit Autopilot's acquisition cut and the Smart
indicator are measured in.

| Field | Meaning |
|---|---|
| `min_precision` | The active detector's floor. |
| `status` | `unchecked` (no spot check has run at this floor; the line keeps the starting candidate), `confirmed` (the last check's likely range clears the floor), or `short` (it ended below the floor, and the line keeps the top 32 it ended on). |
| `count` | How many unvoted items the line keeps: the starting candidate (capped by the corpus), the confirmed set, or 32 after a short check. |
| `range` | The check's **likely range** for how much of the kept set is right - `{"lo", "hi", "labelled", "right", "stale"}` - or `null` while unchecked. A Clopper-Pearson interval from the check's `labelled` picks (`right` of them right), each tail at the level the check's rounds were tested at, and exact once the picks cover the set. `stale` is `true` once later votes moved the list under the result: the range describes the list as it was when checked. |
| `schedule` | What a check at this floor costs: `candidate` (its starting set), `rounds` and `picks` a round. |
| `threshold` | The line: the last item of the kept set. `null` when no detector is active or none has computed a threshold yet. |
| `n_returned` | Items at or above `threshold` in the ranking the detector last scored, voted items included. `null` before a retrain has scored one. |

The range comes only from the check's picks, never from a model: model-chosen
votes break most of an estimator's promises once the reference pool is
consistent, while a uniform pick has no such bias.

### The floor state

Every response that carries a detector's line carries a `floor` object beside
its `threshold`, so a client can say what set the line keeps and how close the
check got: the [learned sort](medias.md#learned-sort),
[`/api/find-label`](find.md#find-label-score--label-the-active-dataset), each
detector of [`/api/auto-detect`](find.md#auto-detect), and the CLI's
autodetect results.

```json
{"min_precision": 0.5, "status": "short", "count": 32, "range": {"lo": 0.11, "hi": 0.73, "labelled": 5, "right": 2, "stale": false}, "schedule": {"candidate": 32, "rounds": 1, "picks": 5}}
```

The fields mean what they do on `/api/min-precision` above. The line keeps a
set in every state: every match, count and action keeps working on it, and
the app draws it the same in all three, with the state and its range in the
floor control and on the Find Stats chart. It is never
`null` for want of a check. A headless run (AutoRun, the CLI) has nobody to
vote, so it exports the `unchecked` starting candidate and records it as such.

### The spot check

```
GET  /api/precision-check
POST /api/precision-check/start
POST /api/precision-check/votes
POST /api/precision-check/cancel
```

Every verb returns `{"floor": <floor state>, "check": <check> | null}`, the
check being the running one, else the last finished one.

**`start`** fixes the candidate off the active detector's current ranking -
the top `schedule.candidate` unvoted items of the ranking its last learned sort
or Find pass scored - and deals the first round's picks. The candidate's ids
never change after this, so every round samples one list however the model
retrains behind it. **409** when there is no ranking yet, nothing in it is
unvoted, or the candidate is the one the last finished check already
measured: there is no redraw on the same candidate (any vote, the check's own
included, changes it). A check already running is replaced.

**`votes`** takes `{"votes": [{"id": 12, "label": "good"}, ...]}` on the round's
picks. Each is an ordinary vote on the item - it trains the model, persists to
the labelset, and in Find mode verifies the item - tagged with provenance
`{"flow": "check"}`. A partial round waits for the rest; an id that is not one
of the round's picks is a **400**; no running check is a **409**. Once the
round is complete it is decided: the floor is **confirmed** when the range's
lower end clears it; otherwise the candidate halves (down to 32) and the next
round's picks are dealt; a round that fails at 32 ends the check **short**.
The floor's line then moves to the set the check ended on, and the result is
kept on the detector: later votes retrain the model and the line follows the
new ranking at the same count, with the range reported `stale`.

**`cancel`** abandons a running check; its votes so far stay ordinary votes
and the floor's state is as it was.

```json
{
  "floor": {"min_precision": 0.1, "status": "unchecked", "count": 128, "range": null, "schedule": {"candidate": 128, "rounds": 3, "picks": 5}},
  "check": {
    "status": "running", "min_precision": 0.1,
    "round": 2, "rounds": 3, "picks_per_round": 5,
    "candidate": 64, "start_candidate": 128,
    "picks": [811, 42, 300, 57, 129],
    "labelled": 2, "right": 0,
    "range": {"lo": 0.0, "hi": 0.777, "labelled": 2, "right": 0}
  }
}
```

| Field | Meaning |
|---|---|
| `status` | `running`, `confirmed`, `short`, or `cancelled`. |
| `round` / `rounds` | The round being voted on (1-based) and how many the check can run: 3 at 10%, 2 at 25%, 1 at 50% and above. |
| `picks_per_round` | Fresh picks each round draws: 5 at 10-50%, 11 at 75%, 29 at 90%. |
| `candidate` / `start_candidate` | The current candidate's size and the size it started at; a failed round halves it, down to 32. |
| `picks` | The picks awaiting a vote this round, in draw order (random). They are a check, not the ranking: a client must not show them as the top of the sort. |
| `labelled` / `right` | Labels inside the current candidate so far (labels already seen inside a halved candidate are kept), and how many were right. |
| `range` | The current candidate's likely range from those labels; `null` before any. |

---

## Labeling Progress

### Analyze progress

```
POST /api/labeling-progress
```

Requires at least one good vote, one bad vote, and label history (400
otherwise). Synchronous: it first advances the per-step model cache over the
whole label history (training any steps not yet cached), so it can be slow on a
long history. The error-cost and stability series cover only the steps a
detector was trained for (see the note under
[Indicator score history](#indicator-score-history)); diversity covers every
step. Each `error_cost` is `fpr + fnr`, measured at the line that detector
would draw at Inclusion 0, whatever the precision floor is.

→
```json
{
  "total_labels": 40,
  "total_medias": 500,
  "error_cost_over_time": [{"num_labels": 10, "error_cost": 0.3, "fpr": 0.1, "fnr": 0.2, "time_index": 9}],
  "stability_over_time": [{"num_labels": 11, "num_flips": 12, "num_confident_flips": 3, "num_unlabeled": 470, "num_pool": 470, "time_index": 10}],
  "diversity_level_over_time": [{"num_labels": 10, "diversity_level": 2, "depth": 3}]
}
```

### Labeling status indicators

```
GET /api/labeling-status
```

→
```json
{
  "smart": {"status": "green", "reason": "..."},
  "stable": {"status": "yellow", "reason": "..."},
  "span": {"status": "red", "reason": "...", "level": 13, "depth": 121, "target": 40},
  "good_count": 12,
  "bad_count": 9,
  "total_count": 500,
  "stale": false
}
```

Each metric has a `status` of `"red"`, `"yellow"`, or `"green"` and an
optional human-readable `reason`. `span` also carries its coverage-atlas
counts: `level` (consecutive covered nodes, also sent as `diversity_level`),
`depth` (total nodes, also `max_level`), and `target`, the level that turns it
green (the diversity goal capped at `depth`; the Autopilot panel paces its
Explore Diversity light against it). The frontend polls this every ~2 s while
labeling, so it never blocks on training: when the per-step cache already
covers the label history the status is computed inline (`stale: false`);
otherwise the last snapshot is returned at once with `stale: true` (counts and
`span` live, `smart` / `stable` lagging) and the cache is advanced by a
background worker for a later poll.

> **Metric-id naming note.** The third indicator is keyed **`span`** in this
> `labeling-status` response, but the `metric` query/body parameter on
> `indicator-score-history` and `eval/train-and-score` (below) uses
> **`diverse`** for the same concept. Both spellings are intentional in the
> current code: use `span` when reading a status object, and `diverse` when
> requesting the diversity metric. (`smart` and `stable` are spelled the same
> in both places.)

### Indicator score history

```
GET /api/indicator-score-history
```

**Query params:** `metric`: one of `"smart"`, `"stable"`, `"diverse"`.

→ `{"metric": "smart", "history": [...], "complete": true}`

Returns per-step indicator data straight from the cache that the
`labeling-status` background worker advances. This route is **read-only**: it
never advances the cache and never trains a model, so it returns promptly
whatever the dataset size or label-history length.

`smart` carries one point per label step the app actually trained a detector
for — that is, per step whose label set a learned sort ran against. `stable`
carries one per such step after the first, since each entry compares a detector
against the one trained before it. Steps in between are absent from both, so the
series are shorter than the label history and their `num_labels` values are not
contiguous. `diverse` measures the votes rather than a detector and does have a
point per step.

A `complete: true` response with an empty `smart` history is therefore a real
answer, not a miss: it means nothing has been trained yet.

When the cache does not yet cover the whole label history — the normal state
while the user is actively labeling, since `labeling-status` defers the advance
to a background worker — the response is `{"metric": ..., "history": [],
"complete": false}`. Clients should then fall back to
`POST /api/eval/train-and-score` (below), which computes the same series on a
background thread with live progress and cancellation. `complete: false` is
also returned if the cache is momentarily locked by an in-flight refresh, so
the read never waits on a build.

### Evaluate metric (train-and-score)

```
POST /api/eval/train-and-score
```

**Body:** `{"metric": "smart"}` (or `"stable"` / `"diverse"`; optional `"wait": true`)

The work retrains the detector head at every step of the label history, so it runs
on a background daemon thread. The route returns immediately with a `job_id`;
poll `GET /api/eval/train-and-score/result` for the metric data and subscribe
to the `eval` SSE channel for live progress. A signature cache short-circuits
identical re-runs. Tests can pass `{"wait": true}` to block until done and get
the data inline.

→ `{"job_id": "...", "status": "running", "current": 0, "total": 10}`, or
(cached / `wait=true`) `{"job_id": "...", "status": "done", "metric": "smart",
"error_cost": [...]}`. The metric-specific key is `error_cost` (smart),
`stability` (stable), or `diversity` (diverse).

### Poll train-and-score result

```
GET /api/eval/train-and-score/result
```

**Query params:** `job_id`: the id returned by `train-and-score`.

→ `running`: `{"job_id": "...", "status": "running", "current": 5, "total": 10}`;
`done`: same shape as the cached/`wait` response above; `cancelled`:
`{"job_id": "...", "status": "cancelled"}`. 404 if the job is unknown; 500 if
the background job failed.

### Cancel train-and-score

```
POST /api/eval/train-and-score/cancel/{job_id}
```

Sets the cancel flag on the background job; the per-step retrain loop polls it
cooperatively. Returns 200 even when the job has already finished. 404 if the
job is unknown.

→ `{"ok": true}`

### Evaluation progress (SSE)

Eval progress streams on the `eval` channel of
[`/api/events`](events.md):

```json
{"status": "running", "message": "Computing smart...", "current": 5, "total": 10}
```

`status` is `"idle"` or `"running"`; the operation is done once `status`
is back to `"idle"` and `current >= total`.

---

## Coverage Atlas

### Get next diverse sample

```
GET /api/coverage-atlas/next
POST /api/coverage-atlas/next
```

POST accepts an optional body with sort scores to influence selection:

**Body:** `{"scores": {"0": 0.9, "1": 0.2}, "threshold": 0.5}`

Malformed score keys/values or a non-numeric `threshold` are a 400. Without
scores (or on GET) the next node's most typical element is returned.

→ `{"id": 42, "coverage_level": 3, "exhausted": false}`

`id` is `null` when the atlas is not built or exhausted. `coverage_level` is the
number of consecutive evidence-bearing nodes in BFS order (0 when nothing is
labeled, up to the total number of atlas nodes when fully covered). `exhausted`
is `true` when every node carries labeled evidence. Sibling nodes are visited
largest-first, so each suggestion covers the biggest unexplored region.

With scores and a `threshold`, the pick is a surprise probe: a presumed-good node (median score
at or above the threshold) yields its lowest-scored element, a presumed-bad
node its highest-scored one. In nodes with a concentrated direction the
extremum is drawn from the node's typical half, so a flip signals a real
hidden pocket rather than a lone oddball.
