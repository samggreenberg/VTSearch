# Find, Auto-Detect & Scoring

[← Back to API index](../API.md)

Endpoints for running detectors against data: multi-dataset **Find**, the
active-dataset **Find Label** / **Auto-Detect** flows, and their evaluation
stats and cancel companions.

**Naming.** The app's **Test** view (`/test/:datasetId/:detectorId`) is the
caller of Find Label, the find stats, corrections and queues, and the test of
the line below. These routes, their fields (`find_mode`, `find_scores`) and
the `find` SSE channel keep the view's former name, so "Find mode" and "the
Find pass" on this page mean the Test view's session.
The Dashboard's **Find** button is something else: it starts a background
AutoFind per ticked dataset
([`POST /api/datasets/registry/{dataset_id}/autofind`](datasets.md#run-autofind-on-a-registered-dataset)).

Several endpoints here read or mutate the active dataset / detector context via
the [`X-Dataset-Id` / `X-Detector-Id` headers](../API.md#context-headers-x-dataset-id--x-detector-id);
the required ones are marked below. Exact response fields are in the OpenAPI
spec ([Machine-readable schema](../API.md#machine-readable-schema)).

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

→ `{warnings: [{detector_name, total_labels, resolved_labels, failed_labels}]}`.
`warnings` only contains entries for detectors with at least one unresolved
label; an empty list means everything resolves.

### Run find

```
POST /api/find
```

**Body:** `{"dataset_ids": ["id1", "id2"], "detector_ids": ["m1"]}`

→ `{results, negative_results, datasets, detectors, media_type,
multiple_datasets, multiple_detectors, total_hits}`. Each row names its
`dataset_name` and carries `detector_verdicts` keyed by detector name.

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

Find progress streams on the `find` channel of [`/api/events`](events.md) as a
[progress object](events.md#progress-object-shape).

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
Find verification workflow. If no current head is cached in the detector
context, it builds one on the fly from the detector's labelset (resolving label
origins as needed). Which one follows the **label quota**: under
3 Goods or 4 Bads, unless a Good has 16 Bads beside it, the detector is the Goods' centroid (every item ranked by its
cosine to the average of the Goods, cut at the midpoint of a two-Gaussian fit
to those cosines on this dataset), and from there the trained head. One Good is
enough; a labelset with no Good is a **400** that says so. `label_quota` says
which detector the pass got and what it still owes.

Items the human has **verified** (see `verified` on [`GET
/api/votes`](medias.md)) keep their existing vote and click-time: re-scoring is
the normal fold-corrections → retrain → re-score loop, and it must not invert a
recorded human decision. Their machine call still seeds `find_initial_labels`,
so a disagreement surfaces as a correction in Find stats. `good_count` /
`bad_count` therefore count the labels actually *adopted* — the threshold split
everywhere except those held votes.

→ `{ok, results, threshold, balance, good_count, bad_count, detector_name,
label_quota}`; `results` rows are `{id, score}`.

`label_quota.tier` is `centroid` for the Goods' centroid, `trained` for the
trained head; `n_good` / `n_bad` are the detector's labels and `goods_owed` /
`bads_owed` what a trained head still needs. The centroid's line is its own
midpoint (its `threshold` is always 0.5 on its own scale) and does not take
the balance: `balance` then counts what that line keeps, and a balance change
leaves it where it is.

`balance` is the [line state](labeling.md#the-line-state) of `threshold`:
the set the Good/Bad split keeps, and what a spot check found on it. The
threshold is the labels' line: the class model the detector's labels
give its head, with the corpus side (how many positives, each item's chance)
re-fitted on this dataset's scores - what a Train on a dataset like this one
would draw. Nothing is counted on the scored corpus, so a dataset with
nothing like the target can come back with no Good split at all. A fresh pass
is `unchecked` until a check runs. On
patch-region-aware datasets each result additionally carries `best_region`.
Errors: **400** (no medias loaded, the detector has no labels, or none of
them is a Good), **404** (detector not found), **409** (active dataset can't
supply the detector's embedder type, or the run was cancelled).

### Auto-Detect

```
POST /api/auto-detect
```

**Body:** `{"detector_name": ""}` — omit / empty to run **every** detector
flagged for AutoFind on the active dataset's media type, or name a single one.

Scores the active dataset with each AutoFind detector, training each head on
demand, and returns one result column per detector.

→ `{media_type, detectors_run, results, missing_detectors}`; `results` is keyed
by detector name, each with its `threshold`, `balance`, `label_quota`,
`total_hits`, `hits` and `negative_hits` (`{id, score}` rows).

Each detector's `label_quota` is as in [find-label](#find-label-score--label-the-active-dataset): a detector
under the label quota runs as the Goods' centroid, and its hits are the
centroid's. A detector with no Good cannot be scored and is left out of
`results`.

Each detector's `balance` is the [line state](labeling.md#the-line-state) of
its `threshold` (`null` only when there was no trained context to ask).
Nobody can vote in a headless run, so every detector exports its `unchecked`
line - the labels' line with the corpus side fitted on the active dataset, the
same line a Find there draws - and the server logs that the set was
never checked.

When an exporter is configured for AutoFind, an `auto_export` object
(`{exporter, success, message?/error?, open_url?}` plus any exporter-specific
extras such as `filepath`) is added. Errors: **400** (no medias loaded, or no
AutoFind detectors for the media type), **404** (named detector not on the
caller's AutoFind list), **409** (cancelled).

This is the synchronous, scripted form. The Dashboard runs the same detectors
in the **background** instead - after a web import (see the `autofind` flag
under [Loading Datasets](datasets.md#loading-datasets)) and from a dataset's
⋯ **Run AutoFind**
([`POST /api/datasets/registry/{dataset_id}/autofind`](datasets.md#run-autofind-on-a-registered-dataset)),
whose big **Find** button runs the ticked detectors in their place -
and keeps each run's results for the user who started it:

### AutoFind results

```
GET /api/autofind/runs/{run_id}
```

`run_id` is the background run's `task_id`. Returns the body above
(`auto_export` included when an exporter ran) plus `run_id`, `dataset_id`,
`dataset_name`, `trigger` (`"import"`, `"manual"` for the ⋯ **Run AutoFind**,
or `"find"` for the **Find** button's picked detectors) and `created_at`.
Runs live in memory only, and only the most recent few, so **404** covers an
unknown run, another user's, one that has aged out, and any from before a
restart alike.

### Prepare Browse for AutoFind results

```
POST /api/autofind/runs/{run_id}/browse-prep
```

Starts laying out run `run_id`'s Good results (every detector's `hits`, once
each) as the subset map the Find Results dialog's **Browse** builds, on the
run's own dataset rather than the request's active one, so that Browse, when
pressed, finds it ready or part-way there. The dialog sends it while
open. It starts nothing while other work is in flight: a dataset or detector
load, a background AutoFind or Find, a projection build, a learned sort or
eval, or a Find, sort or eval bar mid-run.

→ `{status, ...}`. `status` is `building` (the fit is under way; `job_id` names it), `ready`
(the map is built; `projection_id`), `busy` (nothing started: ask again
later), or `skipped` with a `reason` (the dataset is no longer loaded, or the
run found nothing Good). A layout already built for these ids, or the fit
already running for them, is reported whatever the server is doing. **404**
as for [AutoFind results](#autofind-results).

### Test the line

```
GET  /api/line-test
POST /api/line-test/start
POST /api/line-test/votes
POST /api/line-test/unvote
POST /api/line-test/cancel
POST /api/line-test/forget
```

**All but `GET` require** `X-Detector-Id`.

Test mode's test of the line the Find pass drew (the design is
[*The test sample*](../../vtscore/docs/packages/training.md#the-test-sample-linetest-linebudgets-line_phase-found_words), the statistics
`vtscore/training/thresholds/line_test.py`). The question is *if this line
went to AutoFind, what share of what it ships would be right, and what share of
the real matches would it ship?* The answer comes from **uniform picks within
rank bands** on both sides of the line, never from the ranking's top or from
the model: the user votes each pick, and the picks say, as likely ranges, the
line's precision and recall on this corpus and F-beta at the balance. Unlike
the [spot check](labeling.md#the-spot-check), a test never moves the line and
never trains: the ranking is frozen for the whole test, which is what makes the
band design valid.

Every verb returns the same body: `{balance, threshold, line_count, test,
stale, moved, presets}`. `test` holds the `phase` and its `report`, the round's
`band` and `picks`, every band's tallies in `bands`, the joint `estimates`
(precision, recall and F-beta as `{point, lo, hi}` ranges, the plain-language
`found` line, and `at_edges`), the `budgets` the phases run to, `kept_at` and
`class_model`.

- **`start`** freezes the ranking off the pass's frozen scores (best first)
  and the line at the detector's threshold (`line_count` items at or above
  it), takes the labels line's chance per item as the auxiliary below the line
  when the detector has a class model, and deals the first round. A test
  already running is replaced; a finished test of the same line is returned
  as it is. So is a [kept verdict](#the-kept-verdict) for this dataset that is
  not stale and was drawn from this very ranking and line: the test resumes
  from its picks (`test.kept_at` is when they were taken, `null` for a test
  begun afresh), and they are session votes again. **409** with no Find pass
  to test.
- **`votes`** takes `{"votes": [{"id": 412, "label": "good"}, ...]}` on the
  round's `picks`. Each vote is a **session vote** with provenance `test`:
  verified in Find mode (it lands in the Review tab's piles) and kept out of
  the labelset like every Find vote, never a training label. The test records
  it against the pick's band and the ranges move; a partial round waits for
  the rest, and a whole round deals the next. The vote that finishes the test
  keeps its verdict on the detector. **400** for an id that is not one of the
  round's picks; **409** with no test running.
- **`unvote`** takes `{"id": 412}` and takes one vote of the current round
  back (the ↓ key): the pick returns to `picks` and its session vote is
  lifted. **400** for a vote not of this round.
- **`cancel`** drops a running test; its votes so far stay session votes. A
  finished test is left as it is. Always **200**.
- **`forget`** drops the [kept verdict](#the-kept-verdict) for this dataset,
  and a finished test in memory with it, so the next `start` deals a fresh
  test even over a ranking a kept verdict was drawn from. It is a reset, which
  the screenshot harness calls between shots; the app has no button for it,
  because a retrain already marks a verdict stale, a changed ranking already
  deals a fresh test, and a second test of an unchanged ranking measures
  nothing the first did not. A running test is left as it is, and the
  forgotten test's picks stay session votes. **200** whether or not anything
  was kept.

`test.phase` is derived from the sample on every read
(`vtscore.training.thresholds.line_phase`): `matches` (precision, picks from
the bands above the line, the band holding the line first, then the band
whose round would narrow the F-beta range most), `misses` (recall, a walk
down the bands below the line, a band a round), `done`, or `nothing` (the
line keeps fewer items than one round). With a class model the walk runs to
`budgets.misses_picks`; without one (`test.class_model` is `false`: a
structural or document detector) it stops at the first band with no match,
and the recall below the bands it reached is unmeasured, so the app reads it
in words only. `report` says why each finished phase ended (`width`,
`budget`, `exhausted`, `dry_run`; the misses phase ends on `width` or
`dry_run` only without a class model) and carries the ranges' current widths
for the app's phase lights. `estimates` is every number from one set of joint
draws, and `null` when there is nothing to test; `at_edges` re-estimates the
line at every band edge from the same draws.

`presets` is the line each balance preset (beta 1/4, 1, 4) would draw on this
corpus and what the picks already taken say it would ship - the verdict's
**Lean the Threshold** - and is empty before a test or when the detector has
no class model to draw a line at another balance with. `stale` is `true`
once corrections have been folded into the detector since the test (the
detector has now seen the test set); `moved` once the line no longer keeps the
set the test measured (the balance changed, or the ranking did). A fresh
`/api/find-label` pass, a vote clear and a dataset switch all drop the test.

#### The kept verdict

A finished test outlives the session
(`vtscore/detectors/line_verdicts.py`). The vote that brings a test to `done`
writes its verdict into the detector's JSON beside the labelset, under
`test_verdicts`, one per tested dataset (a newer test of the same dataset
replaces the older): the dataset's id and name, when it finished, the balance
and the line it ran at, the picks' ids and labels with the band each came
from, and the precision, recall and F-beta ranges. Ids, labels and numbers
only, never a score vector or a model. `nothing` leaves no verdict.

A verdict is **stale** once the detector has been retrained: it records a
digest of the training labels' signature, and any change to the labels (a
Train vote, Add Corrections, an import) no longer matches it. A stale verdict
stays, flagged. One finished after Add Corrections already folded the test set
in is stale from the start. Two readers show it: the detector's
[stats](detectors.md#detector-statistics) (`test_verdicts`, every verdict, newest first) and the
[detector listing](detectors.md#list-registered-detectors), where an AutoFind detector
carries `test_verdict`, its newest.

### Find stats (detector evaluation)

```
GET /api/find/stats
```

Pure-read detector-evaluation stats over the adopted Find label set: a 2×2
confusion of the adopted label vs. the detector's original call, the Kept rate,
what the balance says about the line, and the checked-precision curve. The
app's result pane reads the 2×2 once a [test of the line](#test-the-line) is
done; the curve it draws is the test's, not this one.

→ The adopted-vs-original 2×2 counts (`confirmed_good`, `confirmed_bad`,
`culled_false_pos`, `rescued_false_neg`, `agreements`, `corrections`,
`agreement_rate`), the Kept rate, `threshold` / `n_scored` / `n_returned` /
`stale`, `balance`, and `precision_curve`.

- `verified_precision` (the Stats **Kept rate**) is taken over the checked items
  the detector called Good only: `verified_kept_good / verified_called_good`,
  `null` when none is checked. Unchecked matches are not counted as right.
- `precision_curve` reads down the ranking (score descending): each point is the
  top `n_returned` items, sampled at about 40 log-spaced counts plus the current
  cut's (`n_returned` at the top level). `verified_precision` is
  `checked_good / checked` over the items in it the user verified (`null` when
  none). The curve carries no model-based estimate: model-based estimators
  break their promises once the reference pool is consistent, so the only
  range the chart shows for unchecked items is the spot check's, in `balance`.
- `balance` is the [line state](labeling.md#the-line-state) of the line at
  `threshold`: the set the line keeps (possibly none), and the spot check's
  likely ranges for the set it audited. The chart's legend says which it is
  (`Line: checked (48 kept)` or `Line: the top 32, unchecked`).
- `stale` is `true` once corrections have been folded into the detector since
  this Find run scored.

### Find work queues

These compute Find's working sets **server-side** from the frozen scores, the
live cutoff, and the verified set, so a client holding only a window of a large
ranking can still act on every matching item. Both **require** `X-Detector-Id`.
Neither has a frontend caller yet (the Test view computes its queues
client-side).

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

→ `{available, n_items, n_pos_labels, n_neg_labels, k, alpha,
frac_unsupported, expected_unsupported, z_score, median_support,
frac_low_trust, median_trust, unsupported}`.

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

→ `{ok, name, corrections_added, num_labels}`

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
[`GET /api/votes`](medias.md#get-votes). Without it a user returning from Find
to training would see every item already voted, and the find-mode write-back
guard would keep each new training vote out of the labelset.

Find-session state is in-memory only and is already dropped on a dataset
switch; nothing durable is lost. Corrections folded in via
`/api/find/corrections-to-detector` live in the labelset and survive.

Idempotent — with no session to end it is a no-op reporting `ended: false`.

→ `{"ok": true, "ended": true}`
