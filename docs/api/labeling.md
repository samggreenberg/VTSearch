# Labeling & Diversity

[← Back to API index](../API.md)

> These endpoints read/mutate the active dataset and detector via the
> [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).

---

## Inclusion & Thresholds

### Get / set inclusion

```
GET /api/inclusion
```

→ `{"inclusion": 0, "threshold": 0.5123}`

```
POST /api/inclusion
```

**Body:** `{"inclusion": 3}`

Any number is accepted: it is truncated toward zero, then clamped to −10..+10
(a boolean or non-number is a 422). Inclusion is a pure cutoff knob — it
re-derives the active detector's threshold from its cached fold orderings
without retraining, and in Find mode re-splits the unverified items over the
frozen scores. The same value is also settable as `inclusion` on
`PUT /api/settings`.

→ `{"inclusion": 3, "threshold": 0.4471}`

Both verbs return the cutoff the inclusion resolves to on the **active
detector** (`X-Detector-Id`), so the Find slider can move its line without
re-scoring. `threshold` is `null` when no detector is active or none has
computed a threshold yet.

Inclusion draws the line only while the detector has **no precision floor**:
a set floor wins (see below), and `threshold` is then the floor's line.

### Get / set the precision floor

```
GET /api/min-precision
```

→ `{"min_precision": 0.5, "status": "insufficient_evidence", "threshold": 0.5123, "n_returned": 412, "calibration_positives": 3}`

```
POST /api/min-precision
```

**Body:** `{"min_precision": 0.75}` — or `{"min_precision": null}` to clear it.

The precision floor is the fraction of what the detector returns that should
be right: the line returns as much as it can while at least that share of it
is estimated right (a bootstrap lower bound, not a point estimate). It is kept
per detector and seeded from the user's `min_precision` setting, which is
`0.5` until the user changes it. A number is clamped to `[0.01, 1]`; a boolean
or non-number is a 422. Like Inclusion it is a pure cutoff knob — the active
detector re-cuts without retraining and, in Find mode, re-splits the
unverified items — and the same value is settable as `min_precision` on
`PUT /api/settings`.

| Field | Meaning |
|---|---|
| `min_precision` | The active detector's floor, or `null` when none is set and Inclusion draws the line. |
| `status` | `promised` (the line keeps the floor), `unreachable` (enough evidence, but no cut reaches it), `insufficient_evidence` (fewer than 10 positives among the calibration votes), or `null` with no floor. |
| `threshold` | The line: the floor's cut when promised, the **Inclusion 0** cut when the floor promises nothing, Inclusion's cut when no floor is set. |
| `n_returned` | Items at or above `threshold` in the corpus the cut decides — the dataset the detector last trained against, less its voted items when those are excluded from the population estimate. `null` before a retrain has fitted one. |
| `calibration_positives` | Positives among the held-out calibration votes that may serve as evidence. Only votes drawn off the learned sort's own ranking count (Autopilot's Hard picks, or the top / cutoff band of a learned-sorted list): votes from the text sort, the coverage atlas, Find verification or bulk actions train the detector but not the promise. |

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
would draw at Inclusion 0, whatever the `inclusion` setting is.

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
  "span": {"status": "red", "reason": "..."},
  "good_count": 12,
  "bad_count": 9,
  "total_count": 500,
  "stale": false
}
```

Each metric has a `status` of `"red"`, `"yellow"`, or `"green"` and an
optional human-readable `reason`. The frontend polls this every ~2 s while
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
