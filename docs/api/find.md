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
**500** (pickle load failed).

### Cancel find

```
POST /api/find/cancel
```

Sets the shared `find_progress` cancel flag so any in-flight scoring path
(find / find-label / auto-detect) stops cooperatively. Always **200**, no-op
when idle.

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
  "good_count": 42,
  "bad_count": 458,
  "detector_name": "Dog Barks"
}
```

On patch-region-aware datasets each result additionally carries `best_region`.
Errors: **400** (no medias loaded, or detector has no labels), **404**
(detector not found), **409** (active dataset can't supply the detector's
embedder type).

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
      "total_hits": 42,
      "hits": [{"id": 0, "score": 0.98}, ...],
      "negative_hits": [{"id": 7, "score": 0.02}, ...]
    }
  },
  "missing_detectors": []
}
```

When an exporter is configured for Auto-Find, an `auto_export` object
(`{exporter, success, message?/error?}`) is added. Errors: **400** (no medias
loaded, or no Auto-Find detectors for the media type), **404** (named detector
not flagged for Auto-Find).

### Find stats (detector evaluation)

```
GET /api/find/stats
```

Pure-read detector-evaluation stats over the adopted Find label set: a 2×2
confusion of the adopted label vs. the detector's original call, plus an FP/FN
threshold sweep.

→
```json
{
  "total_good": 42, "total_bad": 458,
  "verified_count": 30,
  "confirmed_good": 25, "confirmed_bad": 3,
  "culled_false_pos": 3, "rescued_false_neg": 2,
  "agreements": 28, "corrections": 2,
  "agreement_rate": 0.93, "precision": 0.89,
  "inclusion": 0, "threshold": 0.5, "stale": false,
  "sweep": [{"inclusion": -10, "threshold": 0.7, "false_pos": 1, "false_neg": 9}, ...]
}
```

`sweep` covers inclusion −10..10.

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
