# Find, Auto-Detect & Scoring

[← Back to API index](../API.md)

Endpoints for running detectors against data: multi-dataset **Find**, the
active-dataset **Find Label** / **Auto-Detect** flows, and their evaluation
stats and cancel companions.

Several endpoints here read or mutate the active dataset / detector context via
the [`X-Dataset-Id` / `X-Detector-Id` headers](../API.md#context-headers-x-dataset-id--x-detector-id);
the required ones are marked below.

---

## Multi-dataset Find

Runs each selected detector against each selected dataset (loaded from its
pickle) and returns a merged table. Detector and dataset ids come from the
request **body**, so these routes do not require context headers.

### Check label resolution (pre-flight)

```
POST /api/find/check-labels
```

**Body:** `{"dataset_ids": ["id1", "id2"], "detector_ids": ["m1"]}`

Reports, per detector, how many labels can be resolved against the chosen
datasets so the UI can warn before an expensive Find. Call before `POST
/api/find`.

→
```json
{
  "warnings": [
    {
      "detector_name": "Mammals",
      "total_labels": 82,
      "resolved_labels": 60,
      "failed_labels": 22
    }
  ]
}
```

`warnings` only contains entries for detectors with at least one unresolved
label; an empty list means everything resolves.

### Run find

```
POST /api/find
```

**Body:** `{"dataset_ids": ["id1", "id2"], "detector_ids": ["m1"]}`

→
```json
{
  "results": [
    {
      "id": 0,
      "filename": "dog.wav",
      "md5": "...",
      "origin_name": "...",
      "origin": {"...": "..."},
      "dataset_name": "ESC-50",
      "detector_verdicts": {"Dog Barks": {"verdict": "Good"}}
    }
  ],
  "negative_results": [...],
  "datasets": ["ESC-50", "Speech Commands"],
  "detectors": ["Dog Barks"],
  "media_type": "audio",
  "multiple_datasets": true,
  "multiple_detectors": false,
  "total_hits": 42
}
```

Each verdict is one of `Good`, `Bad`, `Error`, `N/A`. Errors: **400** (empty
id lists, or a detector has no labels), **404** (unknown dataset/detector id),
**409** (cancelled via `POST /api/find/cancel`), **500** (pickle load failed).

### Cancel find

```
POST /api/find/cancel
```

Sets the shared `find_progress` cancel flag so any in-flight scoring path
(find / find-label / auto-detect) stops cooperatively; the cancelled request
then answers **409**. Always **200**, no-op when idle.

→ `{"ok": true}`

### Find progress (SSE)

Find progress streams on the `find` channel of [`/api/events`](events.md):

```json
{
  "status": "running",
  "message": "Scoring with \"ModelName\" on \"DatasetName\"...",
  "current": 150,
  "total": 300,
  "step": 2,
  "total_steps": 3,
  "error": null
}
```

`status` is `"idle"` or `"running"`. `step` / `total_steps` track the high-level
Find phases (prepare detectors, load data, score); `overall` (0..1) and
`eta_seconds` give a single whole-job progress fraction and ETA (see
[Events](events.md#progress-object-shape)).

---

## Active-dataset scoring

These operate on the **loaded** dataset (the in-memory snapshot), not pickles.

### Find Label (score + label the active dataset)

```
POST /api/find-label
```

**Requires** `X-Dataset-Id` **and** `X-Detector-Id`.
**Body:** `{"detector_id": "abc123"}`

Scores every loaded media with the given detector and applies Good/Bad labels
to **all** elements by threshold, freezing scores and initial labels for the
Find verification workflow. If no trained head is cached in the detector
context, it trains on the fly from the detector's labelset (resolving label
origins as needed).

Items the human has **verified** (see `verified` on [`GET
/api/votes`](medias.md)) keep their existing vote and click-time: re-scoring is
the normal fold-corrections → retrain → re-score loop, and it must not invert a
recorded human decision. Their machine call still seeds `find_initial_labels`,
so a disagreement surfaces as a correction in Find stats. `good_count` /
`bad_count` therefore count the labels actually *adopted* — the threshold split
everywhere except those held votes.

→
```json
{
  "ok": true,
  "results": [{"id": 0, "score": 0.9812}, ...],
  "threshold": 0.5,
  "balance": {"beta": 1.0, "status": "unchecked", "count": 32, "precision": null, "recall": null, "fbeta": null, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}},
  "floor": {"min_precision": 0.5, "status": "unchecked", "count": 32, "range": null, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}},
  "good_count": 42,
  "bad_count": 458,
  "detector_name": "Dog Barks"
}
```

`balance` and `floor` are the [line state](labeling.md#the-line-state) of
`threshold`: the set the Good/Bad split keeps, and what a spot check found
on it. The `line_preference` setting says which of the two drew the line
(the balance by default; the floor is deprecated and rides along for one
release). Under the balance the threshold is the labels' line (#4452): the
class model the detector's labels give its head, with the corpus side (how
many positives, each item's chance) re-fitted on this dataset's scores - what a Train on a dataset like this
one would draw. Nothing is counted on the scored corpus, so a dataset with
nothing like the target can come back with no Good split at all. A fresh pass
is `unchecked` until a check runs. On
patch-region-aware datasets each result additionally carries `best_region`.
Errors: **400** (no medias loaded, or detector has no labels), **404**
(detector not found), **409** (active dataset can't supply the detector's
embedder type, or the run was cancelled).

### Auto-Detect

```
POST /api/auto-detect
```

**Body:** `{"detector_name": ""}` — omit / empty to run **every** detector
flagged for Auto-Find on the active dataset's media type, or name a single one.

Scores the active dataset with each Auto-Find detector, training each head on
demand, and returns one result column per detector.

→
```json
{
  "media_type": "audio",
  "detectors_run": 2,
  "results": {
    "Dog Barks": {
      "detector_name": "Dog Barks",
      "threshold": 0.5,
      "balance": {"beta": 1.0, "status": "unchecked", "count": 32, "precision": null, "recall": null, "fbeta": null, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}},
      "floor": {"min_precision": 0.5, "status": "unchecked", "count": 32, "range": null, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}},
      "total_hits": 42,
      "hits": [{"id": 0, "score": 0.98}, ...],
      "negative_hits": [{"id": 7, "score": 0.02}, ...]
    }
  },
  "missing_detectors": []
}
```

Each detector's `balance` and `floor` are the
[line state](labeling.md#the-line-state) of its `threshold` (`null` only when
there was no trained context to ask). Nobody can vote in a headless run, so
every detector exports its preference's `unchecked` line - under the balance,
the labels' line with the corpus side fitted on the active dataset, the
same line a Find there draws (#4452); under the deprecated floor, the smaller
of the starting candidate and the mixture's count (#4389) - and the server
logs that the set was never checked.

When an exporter is configured for Auto-Find, an `auto_export` object
(`{exporter, success, message?/error?, open_url?}` plus any exporter-specific
extras such as `filepath`) is added. Errors: **400** (no medias loaded, or no
AutoRun detectors for the media type), **404** (named detector not on the
caller's AutoRun list), **409** (cancelled).

This is the synchronous, scripted form. The Dashboard runs the same detectors
in the **background** instead - after a web import (see the `autorun` flag
under [Loading Datasets](datasets.md#loading-datasets)) and from a dataset's
⋯ **Run AutoRun**
([`POST /api/datasets/registry/{dataset_id}/autorun`](datasets.md#run-autorun-on-a-registered-dataset)),
whose big **AutoRun** button runs the ticked detectors in their place -
and keeps each run's results for the user who started it:

### AutoRun results

```
GET /api/autorun/runs/{run_id}
```

`run_id` is the background run's `task_id`. Returns the body above
(`auto_export` included when an exporter ran) plus `run_id`, `dataset_id`,
`dataset_name`, `trigger` (`"import"` or `"manual"`) and `created_at`.
Runs live in memory only, and only the most recent few, so **404** covers an
unknown run, another user's, one that has aged out, and any from before a
restart alike.

### Test the line

```
GET  /api/line-test
POST /api/line-test/start
POST /api/line-test/votes
POST /api/line-test/unvote
POST /api/line-test/cancel
```

**All but `GET` require** `X-Detector-Id`.

Test mode's test of the line the Find pass drew (#4524; the design is
`docs/plans/test-mode.md`, the statistics
`vtscore/training/thresholds/line_test.py`). The question is *if this line
went to AutoRun, what share of what it ships would be right, and what share of
the real matches would it ship?* The answer comes from **uniform picks within
rank bands** on both sides of the line, never from the ranking's top or from
the model: the user votes each pick, and the picks say, as likely ranges, the
line's precision and recall on this corpus and F-beta at the balance. Unlike
the [spot check](labeling.md#the-spot-check), a test never moves the line and
never trains: the ranking is frozen for the whole test, which is what makes the
band design valid.

Every verb returns the same body:

```json
{
  "balance": {"beta": 1.0, "status": "unchecked", "count": 64, "...": "..."},
  "threshold": 0.43, "line_count": 64,
  "test": {
    "phase": "matches",
    "report": {"phase": "matches", "matches_stop": null, "misses_stop": null,
               "matches_width": 0.41, "misses_width": 0.38, "picks_above": 5, "picks_below": 0},
    "beta": 1.0, "line_count": 64, "size": 1200,
    "round": 2, "picks_per_round": 5,
    "band": {"index": 2, "side": "above", "lo": 17, "hi": 32},
    "picks": [412, 77, 903, 15, 260],
    "labelled": 5,
    "bands": [{"index": 0, "side": "above", "lo": 1, "hi": 8, "labelled": 0, "right": 0, "range": null}, "..."],
    "estimates": {
      "beta": 1.0,
      "precision": {"point": 0.78, "lo": 0.58, "hi": 0.96},
      "recall": {"point": 0.52, "lo": 0.31, "hi": 0.74},
      "fbeta": {"point": 0.62, "lo": 0.45, "hi": 0.79},
      "found": "about half of them found",
      "positives_above": {"point": 50.1, "lo": 37, "hi": 61},
      "positives_below": {"point": 46.3, "lo": 20, "hi": 80},
      "tail_positives": 12.4, "tail_from_model": true,
      "labelled": 5,
      "at_edges": [{"count": 8, "side": "above", "precision": {"...": "..."}, "recall": {"...": "..."}, "fbeta": {"...": "..."}, "found": "..."}, "..."]
    },
    "budgets": {"matches_width": 0.2, "misses_width": 0.25, "matches_picks": 40, "misses_picks": 40,
                "picks_per_round": 5, "dry_run_share": 0.05, "model_weight": 5, "alpha": 0.05}
  },
  "stale": false, "moved": false,
  "presets": [{"beta": 0.25, "count": 31, "precision": {"...": "..."}, "recall": {"...": "..."}, "fbeta": {"...": "..."}, "found": "..."}, "..."]
}
```

- **`start`** freezes the ranking off the pass's frozen scores (best first)
  and the line at the detector's threshold (`line_count` items at or above
  it), takes the labels line's chance per item as the auxiliary below the line
  when the detector has a class model, and deals the first round. A test
  already running is replaced; a finished test of the same line is returned
  as it is. **409** with no Find pass to test.
- **`votes`** takes `{"votes": [{"id": 412, "label": "good"}, ...]}` on the
  round's `picks`. Each vote is a **session vote** with provenance `test`:
  verified in Find mode (it lands in the Review tab's piles) and kept out of
  the labelset like every Find vote, never a training label. The test records
  it against the pick's band and the ranges move; a partial round waits for
  the rest, and a whole round deals the next. **400** for an id that is not
  one of the round's picks; **409** with no test running.
- **`unvote`** takes `{"id": 412}` and takes one vote of the current round
  back (the ↓ key): the pick returns to `picks` and its session vote is
  lifted. **400** for a vote not of this round.
- **`cancel`** drops a running test; its votes so far stay session votes. A
  finished test is left as it is. Always **200**.

`test.phase` is derived from the sample on every read
(`vtscore.training.thresholds.line_phase`): `matches` (precision, picks from
the bands above the line, the band holding the line first, then the band
whose round would narrow the F-beta range most), `misses` (recall, a walk
down the bands below the line with a dry-run stop), `done`, or `nothing`
(the line keeps fewer items than one round). `report` says why each finished
phase ended (`width`, `budget`, `exhausted`, `dry_run`) and carries the
ranges' current widths for the app's phase lights. `estimates` is every
number from one set of joint draws, and `null` when there is nothing to test;
`at_edges` re-estimates the line at every band edge from the same draws.

`presets` is the line each balance preset (beta 1/4, 1, 4) would draw on this
corpus and what the picks already taken say it would ship - the verdict's
**Lean the Threshold** - and is empty before a test or when the detector has
no class model to draw a line at another balance with. `stale` is `true`
once corrections have been folded into the detector since the test (the
detector has now seen the test set); `moved` once the line no longer keeps the
set the test measured (the balance changed, or the ranking did). A fresh
`/api/find-label` pass, a vote clear and a dataset switch all drop the test.

### Find stats (detector evaluation)

```
GET /api/find/stats
```

Pure-read detector-evaluation stats over the adopted Find label set: a 2×2
confusion of the adopted label vs. the detector's original call, the Kept rate,
what the balance (and the deprecated floor) says about the line, and the
checked-precision curve. The app's result pane reads the 2×2 once a
[test of the line](#test-the-line) is done; the curve it draws is the test's,
not this one.

→
```json
{
  "total_good": 42, "total_bad": 458,
  "verified_count": 30,
  "confirmed_good": 25, "confirmed_bad": 3,
  "culled_false_pos": 3, "rescued_false_neg": 2,
  "agreements": 28, "corrections": 2,
  "agreement_rate": 0.93,
  "verified_precision": 0.82, "verified_called_good": 17, "verified_kept_good": 14,
  "threshold": 0.5, "n_scored": 500, "n_returned": 45, "stale": false,
  "balance": {"beta": 1.0, "status": "checked", "count": 48, "precision": {"lo": 0.55, "hi": 0.8, "labelled": 15, "right": 10, "stale": false}, "recall": {"lo": 0.3, "hi": 0.6, "labelled": 15, "right": 10, "stale": false}, "fbeta": 0.61, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}},
  "floor": {"min_precision": 0.5, "status": "unchecked", "count": 32, "range": null, "schedule": {"candidate": 32, "rounds": 3, "picks": 5}},
  "precision_curve": [
    {"n_returned": 1, "threshold": 0.98, "checked": 1, "checked_good": 1,
     "verified_precision": 1.0}, ...
  ]
}
```

- `verified_precision` (the Stats **Kept rate**) is taken over the checked items
  the detector called Good only: `verified_kept_good / verified_called_good`,
  `null` when none is checked. Unchecked matches are not counted as right.
- `precision_curve` reads down the ranking (score descending): each point is the
  top `n_returned` items, sampled at about 40 log-spaced counts plus the current
  cut's (`n_returned` at the top level). `verified_precision` is
  `checked_good / checked` over the items in it the user verified (`null` when
  none). The curve carries no model-based estimate: the #4220 estimator breaks
  most of its "at least" promises once its reference pool is consistent
  (#4256), so the only range the chart shows for unchecked items is the spot
  check's, in `balance` (#4360, #4413).
- `balance` and `floor` are the [line state](labeling.md#the-line-state) of
  the line at `threshold`: the set the line keeps (possibly none), and the
  spot check's likely ranges for the set it audited. The chart's legend says which it is (`Line: checked
  (48 kept)` or `Line: the top 32, unchecked`); it draws no floor line any
  more. The `floor` object is the deprecated preference's reading of the same
  line, carried for one release.
- `stale` is `true` once corrections have been folded into the detector since
  this Find run scored.

### Find work queues

These compute Find's working sets **server-side** from the frozen scores, the
live cutoff, and the verified set, so a client holding only a window of a large
ranking can still act on every matching item. Both **require** `X-Detector-Id`.
Neither has a frontend caller yet: they were built ahead of the Find-view
windowing work that switches the client onto them.

```
GET /api/find/queue-ids?filter=unverified_good
```

`filter`: `unverified_good` (default — the left work queue: above-cutoff items
not yet verified) or `good` (verified-good plus unverified positives).

→ `{"ids": [12, 7, 40], "count": 3}` in rank order; empty outside Find mode or
before a scoring pass.

```
GET /api/find/boundary-next?side=above&exclude=12
```

The next unverified item on the boundary walk, which steps outward from the
cutoff alternating faces, so "just sit and vote" samples marginal positives
and marginal negatives. `side` (`above` default / `below`) is the preferred
face, falling back to the other; `exclude` skips one id (the item just voted,
whose verification may not be visible yet).

→ `{"id": 57, "side": "below"}`, or `{"id": null, "side": null}` when both sides
are exhausted.

### Evidence coverage

```
GET /api/find/evidence-coverage
```

How much of the active dataset the active detector is calling **without
labeled evidence behind the call**. For each scored item it compares the
distance to the predicted class's labeled examples against the labelset's own
leave-one-out distances (a conformal support p-value) and computes a trust-score
ratio against the other class. It needs only the detector's labelset
(re-embedded in memory at load), not the dataset it was trained on, so it works
for a detector handed over from another user — the complement to the
[domain-shift report](datasets.md#domain-shift-report), which needs the
training dataset's atlas. Pure read.

→
```json
{
  "available": true,
  "n_items": 5000, "n_pos_labels": 40, "n_neg_labels": 35,
  "k": 1, "alpha": 0.05,
  "frac_unsupported": 0.21, "expected_unsupported": 0.05, "z_score": 51.9,
  "median_support": 0.34,
  "frac_low_trust": 0.12, "median_trust": 1.6,
  "unsupported": true
}
```

`frac_unsupported` is the share of items whose support p-value falls below
`alpha` (it sits near `expected_unsupported` when the labels cover the data);
`frac_low_trust` the share closer to the other class's evidence than their
own. `unsupported` is the headline verdict (z > 3 **and** `frac_unsupported ≥
2·alpha`). `available: false` (with zeroed fields) — never a 4xx — when there
is no scored Find run or no resolvable labelset.

### Fold corrections into the detector

```
POST /api/find/corrections-to-detector
```

**Requires** `X-Dataset-Id` **and** `X-Detector-Id`.

Writes the Find corrections (adopted labels that differ from the detector's
original call) into the active detector's on-disk labelset for future scoring,
leaving the current Find session frozen and marking it stale.

→ `{"ok": true, "name": "Dog Barks", "corrections_added": 2, "num_labels": 84}`

Errors: **400** (no Find run yet), **404** (no active detector), **409**
(detector vote state not aligned with the active dataset).

### End the Find session

```
POST /api/find/end-session
```

**Requires** `X-Dataset-Id` **and** `X-Detector-Id`.

Discards the active detector's live Find session — the whole-dataset
presumptions `find-label` wrote into its vote dicts, the frozen scores, and the
verified set — and re-derives its votes from its on-disk labelset.

Find and training share one set of per-detector vote dicts, so this is what
separates the two: the Train window calls it on entry, before it reads
[`GET /api/votes`](medias.md). Without it a user who ran Find and went back to
training saw every item in the collection already voted (Autopilot lands in a
terminal phase on arrival), and the find-mode write-back guard kept each new
training vote out of the labelset.

Find-session state is in-memory only and is already dropped on a dataset
switch; nothing durable is lost. Corrections folded in via
`/api/find/corrections-to-detector` live in the labelset and survive.

Idempotent — with no session to end it is a no-op reporting `ended: false`.

→ `{"ok": true, "ended": true}`
