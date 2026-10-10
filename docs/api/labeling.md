# Labeling & Diversity

[← Back to API index](../API.md)

> These endpoints read/mutate the active dataset and detector via the
> [`X-Dataset-Id` / `X-Detector-Id` context headers](../API.md#context-headers-x-dataset-id--x-detector-id).
>
> How the line is drawn and checked is in [`docs/ML.md`](../ML.md#threshold-calibration);
> this page covers what the endpoints return. Exact fields are in the OpenAPI
> spec ([Machine-readable schema](../API.md#machine-readable-schema)).

---

## The Balance

### Get / set the balance

```
GET /api/balance
```

→ The [balance fields](#balance-fields) below.

```
POST /api/balance          Body: {"beta": 1}
```

The balance is the user's precision/recall preference, as F-beta's beta: what
the line should make the most of. The app's Threshold control offers three
presets as unnumbered radios along a False Positives to False Negatives
spectrum: **4** (recall-leaning: the line returns the most, with more wrong
ones in it), **1** (balanced, the default) and **1/4** (precision-leaning: only
the surest, missing more). A stored balance off that list shows on its nearest
radio in log space, a tie going to the radio that leans further. The API takes
any beta in `[0.25, 4]`; a number outside it is clamped, and a boolean,
non-number or `null` is a 422.

The balance is kept **per detector**, on the detector's JSON: it is asked for
when the detector is created, and every `POST` here stores the new value there,
so Autopilot, which has no Threshold control of its own, runs at it. A detector
that keeps none (one made by AutoFind or the CLI) is seeded on first read from
the user's `beta` setting, which every pick also updates: it is the user's last
pick, and what the New Detector form starts on.

The line comes **from the labels alone**, so an exported labelset draws it again
on any corpus: the threshold is the cut where the expected F-beta of what it
would return from the corpus being decided peaks. There is no count and no
cap: the line keeps every unvoted item at or above the threshold, which on a
corpus with nothing like the target can be none. A detector whose calibration
folds support no class model (too few votes, one class) keeps its retrain's
fallback cut.

- **Unchecked**: no spot check has run at this beta.
- **Checked**: a spot check has walked the ranking ([below](#the-spot-check)).
  The check is **advisory** at every preset: its ranges describe the set it
  audited, its picks are ordinary votes the next retrain learns from, and
  the line stays where the labels put it.

A `POST` is a pure cutoff move: the active detector's line moves to the new
beta's cut of the same labels' line without retraining and, in Find mode, the
unverified items re-split. Both verbs return the new line in the same round
trip, so the app's control moves its line without re-scoring. The same value
is settable as `beta` on `PUT /api/settings`, which also sets the active
detector's.

#### Balance fields

| Field | Meaning |
|---|---|
| `beta` | The active detector's balance: F-beta's beta. |
| `status` | `unchecked` (no spot check has run at this beta), `checked` (a walk has run; it informs the line and never moves it), or `gate`: a structural (`sift_vlad` / `sift_vlad_doc`) detector's line, which is the verification gate's boundary rather than a cut on a ranking. No check applies there, so `gate` carries no ranges and `checkable` is `false`. |
| `shape` | How a check treats the line: always `advisory` (the walk's ranges inform the line, its picks train, and the line stays where the labels put it). |
| `audited` | The set the last walk ended on, a band edge; what `precision`, `recall` and `fbeta` describe - not the set the line keeps. `null` while unchecked. |
| `count` | How many unvoted items of the ranking the detector last scored are at or above `threshold`: what the line keeps there, possibly 0. No cap bounds it. Under `gate`, how many unvoted items the verification gate passed on the last re-rank. |
| `precision` | The check's **likely range** for how much of the kept set is right - `{"lo", "hi", "labelled", "right", "stale"}` - or `null` while unchecked. Each audited band's Clopper-Pearson interval from its picks, each tail at `alpha / bands` over the set's bands, weighted by band size; exact where the picks cover a band. `stale` is `true` once later votes moved the list under the result: the range describes the list as it was when checked. |
| `recall` | The check's likely range for how much of the corpus's positives the kept set found: the bands' intervals times their sizes, over the mixture's count of positives in the unvoted ranking (fixed when the walk started). The same `labelled`, `right` and `stale`; `null` while unchecked. The rougher of the two ranges: the picks cannot measure its denominator. |
| `fbeta` | The walk's F-beta estimate for the kept set - `(1 + beta²) · tp / (beta² · n_pos + count)`, `tp` the band-weighted positives among the picks, `n_pos` the mixture's - or `null` while unchecked. |
| `checkable` | Whether a spot check can start on this line: `false` with no ranking to walk - a structural (`sift_vlad` / `sift_vlad_doc`) detector, whose line is the verification gate's boundary rather than a cut on a ranking, or a detector not yet trained on this dataset - or with nothing in it left unvoted. `POST /api/precision-check/start` refuses those with a **409**, so the app offers no check there. |
| `separation` | The labels' d': how many spreads apart the labels line's Good and Bad score components sit, the spread the line was cut with. `null` before a retrain has drawn the labels' line. |
| `check_due` | `true` when the labels separate weakly enough that a spot check is due (`weak_check_due`): `separation` below 1.5, at least 10 votes, and 25 votes since the last check ended (finished or closed). Autopilot runs the check; the Train tab's Check button calls for one. Never `true` in Find or while a check runs. |
| `schedule` | Where a walk starts and what it costs: `candidate` (32 at beta ≤ 1, 128 above: the bands a walk audits first; no longer a cap on the line), `rounds` (the bands it audits before its first verdict: 3 for 32, 5 for 128) and `picks` a band (5). |
| `threshold` | The line: the labels' cut at this beta. `null` when no detector is active or none has computed a threshold yet. |
| `n_returned` | Items at or above `threshold` in the ranking the detector last scored, voted items included. `null` before a retrain has scored one. |

The precision range comes only from the check's picks, never from a model:
model-chosen votes break most of an estimator's promises once the reference
pool is consistent, while a uniform pick has no such bias. The recall range
and the F-beta estimate lean on the mixture's count of positives, so they are
wider and rougher.

*Inclusion* is an internal unit only (Autopilot's acquisition cut and the
Smart indicator are measured in it); no endpoint takes or returns it.

### The line state

Every response that carries a detector's line carries a `balance` object
beside its `threshold`, so a client can say what set the line keeps and what
the check found on it: the
[learned sort](medias.md#learned-sort),
[`/api/find-label`](find.md#find-label-score--label-the-active-dataset), the
[Find stats](find.md#find-stats-detector-evaluation), each detector of
[`/api/auto-detect`](find.md#auto-detect), the CLI's autodetect results, and
every `/api/precision-check` verb.

The fields mean what they do on `/api/balance` above, less `threshold` and
`n_returned`. The line keeps a set in both states: every match, count and
action keeps working on it, and the app draws it the same in both, with the
state and its ranges under the Threshold control in Train (in Find the
control reads the [test of the line](find.md#test-the-line) instead). It is
never `null` for want of a check. A
headless run (AutoFind, the CLI) has nobody to vote, so it exports the
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
few near-duplicate scores, `n_pos` is the balance's cap
(`schedule.candidate`) lowered to the unvoted count - the bands a walk starts
from - so the check still starts. The walk starts at the
bands that hold `schedule.candidate` items. **409** when there is no ranking
(none yet, or a structural detector's, which never has one), nothing in it is
unvoted - the two cases the balance's `checkable` reports - or the list is the one the last finished check
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
set. The check ends **checked** on the peak's band edge. The line does not
move: the check is advisory, its result is kept on the detector for
its ranges, the walk's picks retrain the model like any vote, and the ranges
are reported `stale` once later votes move the list. A result belongs to its
beta: another beta is `unchecked` until it is checked itself, and the earlier
result shows again if the beta moves back.

**`cancel`** abandons a running check; its votes so far stay ordinary votes
and the line's state is as it was.

The `check` object:

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

→ `{total_labels, total_medias, error_cost_over_time, stability_over_time,
diversity_level_over_time}`, each series a list of per-step points keyed by
`num_labels`.

### Labeling status indicators

```
GET /api/labeling-status
```

→ `{stop_rule, smart, stable, span, good_count, bad_count, total_count, stale}`.

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

`stop_rule` names what ends labeling: `"lights"` (the three metrics above) or,
on a document collection whose pages carry tiles (`sift_vlad_doc`), `"dry_run"`.
A document collection's response is computed from the vote order alone, so it
is never stale; `smart`, `stable` and `span` read `"off"`, and a `dry_run`
object (`status`, `reason`, `run`, `target`) carries the stop. `run` counts the standing non-Good votes cast since the last Good: each media's
last vote, in vote order. It is red until half of `target`, yellow from there,
and green at `target` once the labelset holds a Good.

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
[`/api/events`](events.md) as a
[progress object](events.md#progress-object-shape). `status` is `"idle"` or `"running"`; the operation is done once `status`
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
