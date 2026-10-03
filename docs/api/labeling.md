# Labeling & Diversity

[← Back to API index](../API.md)

> These endpoints read/mutate the active dataset and detector via the
> [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).

---

## The Balance

### Get / set the balance

```
GET /api/balance
```

→ `{"beta": 1.0, "status": "unchecked", "count": 32, "precision": null, "recall": null, "fbeta": null, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}, "threshold": 0.5123, "n_returned": 32}`

```
POST /api/balance
```

**Body:** `{"beta": 1}`

The balance is the user's precision/recall preference, as F-beta's beta: what
the line should make the most of (#4413, the owner's ruling of 2026-10-01,
priced by #4411 in
[`REPORT.md`](../experiments/2026-10-01-fbeta-line-4411/REPORT.md)). The
app's Threshold control offers three presets as unnumbered radios along a
False Positives to False Negatives spectrum: **2** (recall-leaning, the left
radio: the line returns the most, with more wrong ones in it), **1**
(balanced, the default) and **0.5** (precision-leaning, the right radio:
only the surest, missing more). The API takes any beta in `[0.25, 4]`; a
number outside it is clamped, and a boolean, non-number or `null` is a 422.
The balance is kept per detector and seeded from the user's `beta` setting.

The line always keeps a **set**: the top `count` unvoted items of the ranking
the detector last scored.

- **Unchecked** (before any spot check at this beta; AutoRun, the CLI and a
  cold Find, which have nobody to vote): the count at which the
  vote-anchored mixture's F-beta peaks over the unvoted ranking
  (`fbeta_count`: a 2-component mixture fitted on the ranking's scores and
  anchored on the votes, its high-component posterior summed down the
  ranking for the positives in each top *k* and in all), **capped** at 128
  items for beta 2 and 32 for beta 1 and 0.5 (`balance_schedule`, #4389).
  The cap is what holds the mixture on a large sparse corpus, where it
  over-counts the positives and would return 2-4x too much (#4411).
- **Checked**: a spot check has walked the ranking ([below](#the-spot-check)).
  What that does to the line follows the preset (#4427): at beta 1 and below
  the check is **advisory** - the walk's ranges inform the line, its votes
  are ordinary votes, and the count stays the unchecked rule's; above 1 the
  check **trims** - the walk may only step shallower from the bands holding
  the line, and the line keeps the band edge it ended on.

A `POST` is a pure cutoff move: the active detector's line moves to the set
the new beta keeps without retraining and, in Find mode, the unverified items
re-split. Both verbs return the new line in the same round trip, so the app's
control moves its line without re-scoring. The same value is settable as
`beta` on `PUT /api/settings`.

| Field | Meaning |
|---|---|
| `beta` | The active detector's balance: F-beta's beta. |
| `status` | `unchecked` (no spot check has run at this beta; the line keeps the mixture's F-beta peak under the cap) or `checked` (a walk has run: under `trim` the line keeps its end, under `advisory` the walk informs it). |
| `shape` | How a check treats the line at this beta (#4427): `advisory` (beta ≤ 1: the walk's ranges inform the line, the count stays the unchecked rule's) or `trim` (above 1: the walk may only step shallower, and the line takes its end). |
| `audited` | The set the last walk ended on, a band edge; what `precision`, `recall` and `fbeta` describe. Under `advisory` it is not the set the line keeps. `null` while unchecked. |
| `count` | How many unvoted items the line keeps: the unchecked count (capped by the corpus), or under `trim` the set the walk ended on (a band edge: 8, 16, 32, 64, ...). |
| `precision` | The check's **likely range** for how much of the kept set is right - `{"lo", "hi", "labelled", "right", "stale"}` - or `null` while unchecked. Each audited band's Clopper-Pearson interval from its picks, each tail at `alpha / bands` over the set's bands, weighted by band size; exact where the picks cover a band. `stale` is `true` once later votes moved the list under the result: the range describes the list as it was when checked. |
| `recall` | The check's likely range for how much of the corpus's positives the kept set found: the bands' intervals times their sizes, over the mixture's count of positives in the unvoted ranking (fixed when the walk started). The same `labelled`, `right` and `stale`; `null` while unchecked. The rougher of the two ranges: the picks cannot measure its denominator. |
| `fbeta` | The walk's F-beta estimate for the kept set - `(1 + beta²) · tp / (beta² · n_pos + count)`, `tp` the band-weighted positives among the picks, `n_pos` the mixture's - or `null` while unchecked. |
| `schedule` | The cap and what a walk from it costs: `candidate` (the cap, 32 or 128, which is where the walk starts), `rounds` (the bands it audits before its first verdict: 3 for 32, 5 for 128) and `picks` a band (5). |
| `threshold` | The line: the last item of the kept set. `null` when no detector is active or none has computed a threshold yet. |
| `n_returned` | Items at or above `threshold` in the ranking the detector last scored, voted items included. `null` before a retrain has scored one. |

The precision range comes only from the check's picks, never from a model:
model-chosen votes break most of an estimator's promises once the reference
pool is consistent, while a uniform pick has no such bias. The recall range
and the F-beta estimate lean on the mixture's count of positives, so they are
wider and rougher (#4411: the walk's own F-beta estimate is off by
0.15-0.21).

The balance replaced the Inclusion knob (`/api/inclusion` is gone).
Inclusion survives only as the internal unit Autopilot's acquisition cut and
the Smart indicator are measured in.

### The line state

Every response that carries a detector's line carries a `balance` object
beside its `threshold`, so a client can say what set the line keeps and what
the check found on it: the
[learned sort](medias.md#learned-sort),
[`/api/find-label`](find.md#find-label-score--label-the-active-dataset), the
[Find stats](find.md#find-stats-detector-evaluation), each detector of
[`/api/auto-detect`](find.md#auto-detect), the CLI's autodetect results, and
every `/api/precision-check` verb.

```json
{"beta": 1.0, "status": "checked", "count": 48, "precision": {"lo": 0.55, "hi": 0.8, "labelled": 15, "right": 10, "stale": false}, "recall": {"lo": 0.3, "hi": 0.6, "labelled": 15, "right": 10, "stale": false}, "fbeta": 0.61, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}}
```

The fields mean what they do on `/api/balance` above, less `threshold` and
`n_returned`. The line keeps a set in both states: every match, count and
action keeps working on it, and the app draws it the same in both, with the
state and its ranges under the Threshold control and in the Find Stats
chart's legend. It is never `null` for want of a check. A
headless run (AutoRun, the CLI) has nobody to vote, so it exports the
`unchecked` line and records it as such.

### The spot check

```
GET  /api/precision-check
POST /api/precision-check/start
POST /api/precision-check/votes
POST /api/precision-check/cancel
```

Every verb returns `{"balance": <balance state>, "check": <check> | null}`,
the check being the running one, else the last finished one.

**`start`** fixes the unvoted ranking off the active detector's current one
(the ranking its last learned sort or Find pass scored, in rank order), cuts
it into bands (the top 8, the next 8, then 16, 32, ...), and deals the first
band's picks: 5 drawn uniformly from the band, a census of a band smaller
than that. The ids never change after this, so every band samples one list
however the model retrains behind it. The walk also fixes `n_pos`, the
mixture's count of positives in the unvoted ranking, which its recall is
read against; when no mixture fits the ranking, or the fit collapsed onto a
few near-duplicate scores (#4419), `n_pos` is the balance's cap
(`schedule.candidate`) lowered to the unvoted count - the set the unchecked
line would have kept - so the check still starts. The walk starts at the
bands that hold `schedule.candidate` items. **409** when there is no ranking
yet, nothing in it is unvoted, or the list is the one the last finished check
already walked (there is no redraw on the same list: any vote, the check's
own included, changes it). A check already running is replaced.

**`votes`** takes `{"votes": [{"id": 12, "label": "good"}, ...]}` on the round's
picks. Each is an ordinary vote on the item - it trains the model, persists to
the labelset, and in Find mode verifies the item - tagged with provenance
`{"flow": "check"}`. A partial round waits for the rest; an id that is not one
of the band's picks is a **400**; no running check is a **409**. Once the
band is audited the walk moves on: the next band the set under test still owes
is dealt, or the set is decided. The walk estimates the set's F-beta -
`tp` from the picks (each band's share of right picks times the band's size),
`n_pos` from the mixture - and goes one band **deeper** while a deeper set's
estimate rises (its picks are dealt); the first time it falls, the walk ends
on the peak. From a start whose first deeper step falls, it goes one band
**shallower** while the estimate does not fall (no new picks: a shallower set
is a subset of an audited one) and ends on the peak. A tie keeps the smaller
set. The check ends **checked** on the peak's band edge. The line then moves
to the set the check ended on, and the result is kept on the detector: later
votes retrain the model and the line follows the new ranking at the same
count, with the ranges reported `stale`. A result belongs to its beta: another
beta is `unchecked` until it is checked itself, and the earlier result shows
again if the beta moves back.

**`cancel`** abandons a running check; its votes so far stay ordinary votes
and the line's state is as it was.

```json
{
  "balance": {"beta": 1.0, "status": "unchecked", "count": 32, "precision": null, "recall": null, "fbeta": null, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}},
  "check": {
    "status": "running", "beta": 1.0,
    "fbeta": null, "recall": {"lo": 0.25, "hi": 0.7, "labelled": 15, "right": 10},
    "round": 4, "rounds": 12, "picks_per_round": 5,
    "candidate": 64, "start_candidate": 32,
    "bands": 4, "band": {"index": 3, "lo": 33, "hi": 64}, "direction": "deeper", "estimate": null,
    "picks": [811, 42, 300, 57, 129],
    "labelled": 15, "right": 10,
    "range": {"lo": 0.4, "hi": 0.85, "labelled": 15, "right": 10}
  }
}
```

| Field | Meaning |
|---|---|
| `status` | `running`; `checked` once the walk finished; or `cancelled`. |
| `beta` | The beta the walk is at. |
| `fbeta` / `recall` | The set under test's F-beta estimate (`null` while a band of it is still unaudited) and its likely recall range from the labels so far (`null` before any). |
| `round` / `rounds` | The round being voted on (1-based; one round per band audited) and how many bands the ranking has in all. |
| `picks_per_round` | Fresh picks each band is audited with: 5 (a census of a band smaller than that). |
| `candidate` / `start_candidate` | The set under test's size (the top `bands` bands; the kept set's once the check has finished) and the count the walk started from. |
| `bands` / `band` | How many bands from the top the set under test spans, and the band whose picks are pending: `{"index", "lo", "hi"}`, its index from the top and its rank positions (1-based, inclusive); `null` between bands. |
| `direction` | Which way the walk last moved: `start` (still auditing the starting bands), `deeper` (the last set scored better) or `shallower` (it did not). |
| `estimate` | The band-weighted share of the set under test that its picks say is right; `null` while a band of it is still unaudited. |
| `picks` | The picks awaiting a vote this round, in draw order (random). They are a check, not the ranking: a client must not show them as the top of the sort. |
| `labelled` / `right` | Labels inside the set under test so far, and how many were right. |
| `range` | The set's likely precision range from those labels; `null` before any. |

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
would draw at Inclusion 0, whatever the balance is.

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
