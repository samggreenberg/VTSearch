# `vtscore.state`

The runtime state package. Every loaded dataset and every loaded detector
gets its own context object that bundles the mutable data structures
operating on it - votes, label history, click times, the cached embedding
matrix, the coverage atlas, the trained head, the calibrated threshold.
The package also owns the resolution chain that picks which context is
"active" for the current call (Flask request, background thread, or test
harness), the helpers that operate on the active context, and the
process-wide registries that map IDs to contexts. Everything is
protected by a single re-entrant lock; there is no implicit global
"active" pointer.

Related docs: [`concurrency.md`](concurrency.md) for the job/progress
infrastructure that drives long-running operations on these contexts.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/state/core.py` | `DatasetContext`, `DetectorContext`, `MediasDict`, `_state_lock`, the resolution chain, both registries |
| `vtscore/state/votes.py` | Vote / label operations, label history, text-sort suggestions, learned scores, the Find queue |
| `vtscore/state/clicks.py` | Click-time ordinals (`assign_click_time`, `remove_click_time`, `get_vote_click_times`) |
| `vtscore/state/coverage.py` | Build / restore / mutate the active dataset's coverage atlas |
| `vtscore/state/media_lookup.py` | Origin / MD5 / name lookup tables, cached per context; exact-duplicate collapsing |
| `vtscore/state/sort_results_cache.py` | Process-global cache of full sorted result lists for windowed paging |
| `vtscore/state/current_user.py` | Flask-free half of user identity: resolver hook + thread-local user |
| `vtscore/state/__init__.py` | Public re-exports, the setting-persistence hooks, and the cross-cutting `snapshot_medias` / `clear_all` helpers |

- [The two context classes](#the-two-context-classes)
- [The resolution chain](#the-resolution-chain)
- [Context registries](#context-registries)
- [Resolver hooks (app integration)](#resolver-hooks-app-integration)
- [Scoped overrides and `with_*_context`](#scoped-overrides-and-with__context)
- [Vote operations](#vote-operations)
- [Click times](#click-times)
- [Coverage atlas](#coverage-atlas)
- [Media lookup](#media-lookup)
- [Sorted-result window cache](#sorted-result-window-cache)
- [Current user](#current-user)
- [Setting-persistence hooks](#setting-persistence-hooks)
- [Thread-safety invariants](#thread-safety-invariants)

---

## The two context classes

`DatasetContext` and `DetectorContext` live in `vtscore/state/core.py`
and `vtscore/state/core.py`. Both use `__slots__` so accidentally
shadowing an attribute raises immediately. Every field is in-memory
only; datasets persist via pickle, detectors via labelset JSON (origins
+ labels only, never model weights - see the "No Persisted Vectors or
MLPs" rule in `CLAUDE.md`).

### `DatasetContext`

Per-dataset mutable state. One context per loaded dataset.

| Attribute | Type | Description |
|-----------|------|-------------|
| `dataset_id` | `str` | Registration key |
| `medias` | `MediasDict` | `cid -> media dict` for every loaded item; a `dict` subclass that bumps `media_revision` on every structural mutation |
| `media_revision` | `int` (read-only) | Monotonic counter advanced on every `medias` mutation; the matrix caches key on it |
| `coverage_atlas` | `CoverageAtlas \| None` | Hierarchical partition of the embeddings with per-class evidence counts (built at load, or lazily for large datasets) |
| `dataset_display_name` | `str \| None` | UI-overridable name |
| `merge_near_duplicates` | `bool` | Transient create-time flag: run the extra near-duplicate collapse after MD5 dedup |
| `_emb_matrix_ids` / `_emb_matrix` / `_emb_matrix_revision` | | Cached `(N, D)` contiguous float32 embedding matrix, the sorted media-ID list it corresponds to, and the `media_revision` it was built at |
| `_emb_sidecar_disabled` | `bool` | One-way latch: once an in-place vector rewrite happens, this context never reads the on-disk matrix sidecar again |
| `_region_matrix*` | | The same cache expanded to one row per `(media, region)` pair, for patch-region embedders |
| `_origin_key_index` / `_md5_index` / `_name_index` / `_lookup_index_revision` | | Cached `build_media_lookup` tables, keyed on `media_revision` |
| `_projection` / `_pyramids` / `_full_job_id` | | VTSBrowse: the frozen UMAP layout, per-bin-shape tile pyramids, and the in-flight build job |
| `_subset_*` | | The same three, for an ephemeral subset layout (e.g. a Find run's positives) |
| `_region_labels` / `_subset_region_labels` / `_relabel_job_id` | | VTSBrowse region signposts for each layout, plus the in-flight relabel job |
| `_text_embedder` / `_patch_embedder` / `_structural_embedder` / `_binding_explicit` | | Role-typed embedder binding - embedder **names** only, never vectors |

The private fields are caches and bookkeeping; treat the first block of
the table as the supported surface. See
[`projection.md`](projection.md) for the VTSBrowse fields and
[`embedding.md`](embedding.md) for the binding.

The cached matrix is rebuilt lazily by
`vtscore.embedding.matrix.get_embedding_matrix` and reused across
cosine sort, detector scoring, and coverage-atlas construction so the library
doesn't rebuild an `(N, D)` matrix per call. Cache validity is a single
`media_revision` compare, so a mutation that changes vectors without
changing the id set still invalidates it (root-cause Pattern #4). An
in-place vector rewrite must bump the counter via
`invalidate_embedding_matrix` / `bump_media_revision`. `clear_medias()`
drops both the matrix and the coverage atlas.

### `DetectorContext`

Per-detector mutable state. Vote state, training artefacts, and the
trained head live here - never on `DatasetContext`. One detector can be
trained on labels collected against many datasets; the labelset is the
canonical persisted form.

| Attribute | Type | Description |
|-----------|------|-------------|
| `detector_id` / `name` / `media_type` / `embedder` | `str` | Identity fields |
| `embedder_type` | `str` | Immutable embedder *type* (single-vector / patch / structural) the detector was created against |
| `good_votes` / `bad_votes` | `dict[int, None]` | Set-shaped vote dicts |
| `label_history` | `list[tuple[int, str, float]]` | `(media_id, label, timestamp)` events |
| `vote_click_times` | `dict[int, int]` | Ordinal assigned at vote time |
| `vote_region_boxes` | `dict[int, tuple[float, float, float, float]]` | Optional box per good vote |
| `click_counter` | `int` | Monotonically-increasing ordinal source |
| `find_mode` | `bool` | The in-memory votes are Find scoring output, not training labels - blocks label sync |
| `last_learned_scores` | `dict[int, float]` | Most recent `train_and_score` output |
| `textsort_suggestions` | `list[str]` | LRU of recent text-sort queries |
| `find_initial_labels` | `dict[int, str]` | Labels the detector applied during a Find run |
| `verified_ids` | `dict[int, None]` | IDs the human explicitly verified this Find session |
| `find_scores` | `dict[int, float]` | Frozen per-item score, so an Inclusion change re-thresholds without re-scoring |
| `find_eval_stale` | `bool` | The labelset changed since this Find evaluation was scored |
| `inclusion` | `int \| None` | Per-detector inclusion fraction override |
| `training_medias` | `dict[int, dict[str, Any]]` | Voted medias with embeddings |
| `label_embeddings` | `dict[str, np.ndarray]` | `stable_element_id -> embedding`, built from origins |
| `label_embedding_regions` | `dict[str, tuple \| None]` | The region each cached `label_embeddings` entry was pooled from - detects a region edit |
| `label_local_features` | `dict[str, StructuralFeatures]` | Cross-dataset local features for structural detectors |
| `label_negative_regions` / `label_score_regions` | `dict[str, list[np.ndarray]]` | Per-element patch stacks on patch datasets (flooded negatives; full score rows) |
| `model` | `torch.nn.Sequential \| None` | Trained head (process-lifetime only) |
| `verification_classifier` | `torch.nn.Sequential \| None` | Structural detectors' second head: match-statistic verification |
| `threshold` | `float` | Calibrated decision threshold |
| `labelset_good_count` / `labelset_bad_count` | `int` | Counts from the on-disk labelset (cross-dataset) |
| `votes_dataset_id` | `str` | Dataset ID the cid-keyed dicts are valid against |
| `cached_labelset` | `LabelSet \| None` | Parsed labelset, reused across requests |
| `cached_labelset_mtime` / `cached_labelset_media_type` | `float` / `str` | Mtime and media type of the JSON the cache was built from |
| `labelset_source` | `dict \| None` | Active sync target |
| `calibration_cache` | `tuple[Any, CalibrationFolds] \| None` | Fingerprint → per-fold held-out scores and models. Deliberately *excludes* inclusion, so an Inclusion change re-runs only the quantile rule |
| `anchored_cut_cache` | `FoldAnchoredCut \| None` | The fold-anchored population estimator behind the current threshold |

Everything in this table is in-memory only. `model`,
`verification_classifier`, `label_embeddings`, `label_local_features`
and the region stacks are all re-derived from the labelset's origins on
the next process start - see the "No Persisted Vectors or MLPs" rule in
`CLAUDE.md`.

The two-tier `(good_votes, bad_votes)` + `(labelset_good_count,
labelset_bad_count)` arrangement is what lets a detector trained on
dataset A stay trainable when the user switches to dataset B: the cid
dicts only count labels for media in the currently-loaded dataset, while
the labelset counts include every labelled element regardless of which
dataset's origins they came from.

### Votes are `dict[int, None]`, not sets

`good_votes` and `bad_votes` look like sets but are typed as
`dict[int, None]`. Add a vote with `votes[cid] = None`, remove with
`votes.pop(cid, None)`, test with `cid in votes`. Dicts preserve
insertion order so the vote sequence round-trips through JSON without a
separate ordering field. **Never** call `votes.add()` or
`votes.remove()` - that will `AttributeError` and indicates the caller
is treating the dict as a set.

---

## The resolution chain

The library never carries a global "currently active" pointer. Helpers
that operate on "the" active context walk the chain in
`get_active_detector_context` (`vtscore/state/core.py`); the dataset side
(`get_active_context`) is identical minus the forced-override tier.
Highest precedence wins.

| Tier | Source | How it's set | Lifetime |
|------|--------|--------------|----------|
| 1 | Forced override (detector only) | `override_detector_context(ctx)` context manager | Block scope |
| 2 | Request resolver | `register_dataset_context_resolver(fn)` / `register_detector_context_resolver(fn)` | Process lifetime; usually reads `flask.g` |
| 3 | Thread-local | `set_thread_dataset_context(ctx)` / `set_thread_detector_context(ctx)` | Thread lifetime |
| 4 | Request-missing sentinel | Returned when `register_request_context_predicate(fn)`'s predicate is true | Per call |
| 5 | Empty fallback | The module-level `_empty_*_context` | Always present |

Neither accessor ever returns `None`. The bottom two tiers differ in what
they do on a **write**:

- The **empty fallback** is the library/CLI case - no host app, no
  request. Reads see empty containers rather than a `NoneType` error, and
  writes land in it harmlessly.
- The **request-missing sentinel** is the "inside a request that never
  identified a dataset/detector" case. The Flask shim registers
  `flask.has_request_context` as the predicate, so a request that sends
  no `X-Dataset-Id` / `X-Detector-Id` gets a *frozen* context: reads see
  empty containers, and any mutation raises `RequestMissingContextError`
  instead of silently polluting the process-wide empty context for every
  other caller. Library-only consumers never see this tier, because the
  default predicate returns `False`.

```python
from vtscore.state import get_active_context, get_active_detector_context

ds_ctx = get_active_context()             # never returns None
det_ctx = get_active_detector_context()   # never returns None
```

`is_request_missing_context(ctx)` (and its per-side variants
`is_request_missing_dataset_context` / `is_request_missing_detector_context`)
tells the two apart when a caller wants to degrade gracefully rather than
let the write raise.

---

## Context registries

Two process-wide registries map each loaded ID to its context. These are
different from the on-disk dataset registry in
`vtscore.datasets.registry`, which tracks `.pkl` files available on disk
- a dataset can be on-disk-registered but not loaded (no context).

| Function | Side | Description |
|----------|------|-------------|
| `register_context(ctx)` | dataset | Add by `ctx.dataset_id` |
| `unregister_context(id)` | dataset | Remove and return, or `None` |
| `get_context(id)` | dataset | Look up, or `None` |
| `list_loaded_dataset_ids()` | dataset | All registered IDs |
| `clear_all_contexts()` | dataset | Drop every context (tests) |
| `register_detector_context(ctx)` | detector | Add by `ctx.detector_id`; also clears labeling-progress cache |
| `unregister_detector_context(id)` | detector | Remove and clear the progress cache |
| `get_detector_context(id)` | detector | Look up, or `None` |
| `list_loaded_detector_ids()` | detector | All registered IDs |
| `clear_all_detector_contexts()` | detector | Drop every context (tests) |

```python
from vtscore.state import DatasetContext, register_context, get_context

register_context(DatasetContext("ds_a"))
assert get_context("ds_a").dataset_id == "ds_a"
assert get_context("nope") is None
```

`unregister_*` also clears the thread-local pointer if it happened to
point at the context being removed.

---

## Resolver hooks (app integration)

`vtscore` is a library - it must not import Flask. The app layer
(`vtsearch/shim/`) installs a Flask-aware resolver that reads the
request's `X-Dataset-Id` / `X-Detector-Id` header out of `flask.g`. The
hooks are in `vtscore/state/core.py`:

```python
from vtscore.state.core import (
    register_dataset_context_resolver,
    register_detector_context_resolver,
)
import flask

def _flask_dataset_resolver():
    if not flask.has_request_context():
        return None
    return getattr(flask.g, "dataset_context", None)

register_dataset_context_resolver(_flask_dataset_resolver)
```

Library-only consumers leave the default resolver in place
(`_default_context_resolver()` returns `None`) and use the thread-local
tier. A consumer running library code from a worker thread calls
`set_thread_dataset_context(ctx)` / `set_thread_detector_context(ctx)` at
the start and `set_thread_*_context(None)` at the end. There is no
"unregister resolver" - installing a new one replaces the previous.

---

## Scoped overrides and `with_*_context`

### `override_detector_context(ctx)`

A context manager that forces `get_active_detector_context()` to return
`ctx` regardless of resolver / thread-local. Lives at the top of the
chain (tier 1). Used by `vtscore/detectors/workflow.py` so the
apply-and-retrain flow works whether the caller is inside a Flask
request or a background thread.

```python
from vtscore.state.core import override_detector_context

with override_detector_context(other_detector_ctx):
    apply_label(media_id=42, label="good")
# Resolution chain restored on exit.
```

There is no analogous override for datasets - the thread-local plus
`with_dataset_context` has been sufficient.

### `with_dataset_context(dataset_id)` / `with_detector_context(detector_id)`

Lookup-by-ID context managers that read from the registry, push onto the
thread-local, and restore on exit. They raise `ValueError` if the ID is
not registered.

```python
from vtscore.state import with_dataset_context, with_detector_context

with with_dataset_context("ds_a"):
    with with_detector_context("det_x"):
        # both contexts are active inside the block
        ...
```

These are **not** thread-safe - they mutate the calling thread's
thread-local only.

---

## Vote operations

All vote helpers operate on the active `DetectorContext`. They resolve
the context themselves via `get_active_detector_context()` so the caller
does not need to thread it through every call. Each helper acquires
`_state_lock` for its whole check-then-modify sequence.

Everything below lives in `vtscore/state/votes.py`.

| Function | Description |
|----------|-------------|
| `set_vote(cid, target, region_box=None, *, count_streak=True)` | **The primary entry point.** Set the vote to an absolute `"good"` / `"bad"` / `"none"` |
| `toggle_vote(cid, vote, region_box=None)` | Thin wrapper over `set_vote` that computes the target from the current state |
| `clear_votes()` | Drop all votes, history, click times, region boxes |
| `apply_label(cid, label, *, silent=False, region_box=None, record_achievement=True, count_streak=True)` | Set label unconditionally (for imports). `silent=True` skips history + atlas evidence |
| `apply_label_with_click_time(cid, label)` | Same as `apply_label` but assigns a click-time ordinal |
| `apply_labels_bulk_with_click_time(labels, replace_all=False, *, record_achievement=True, preserve_verified=False)` | Apply many labels under a single lock acquisition; returns the set of applied IDs |
| `add_label_to_history(cid, label)` | Append `(cid, label, time)` to `label_history` |
| `add_textsort_suggestion(text)` / `get_textsort_suggestions()` | LRU of recent queries |
| `update_learned_scores(scores)` / `get_learned_scores()` | Cache last `train_and_score` output |
| `set_find_initial_labels(labels)` / `get_find_initial_labels()` | Snapshot of detector-assigned labels |
| `set_find_scores(scores)` / `get_find_scores()` | The raw scores a Find run produced |
| `find_queue_ids(label_filter)` | Find-run IDs in score order, filtered by `"good"` / `"bad"` / `"all"` |
| `find_boundary_next(side, exclude=None)` | The next unverified item just above / below the Find cut |
| `rethreshold_unverified_find_items(...)` | Re-apply the detector's cut to Find items the user hasn't verified |

### Absolute targets, not toggles

`set_vote(cid, target)` is the one to call. It is **idempotent**: setting
the target equal to the current state appends nothing to `label_history`,
credits no achievement counter, and assigns no new click-time. That
matters because two browser tabs holding a stale view would otherwise
each send a "toggle" and inflate the counters; an absolute target makes
the stale duplicate collapse into a no-op on the server.

`toggle_vote(cid, vote)` is kept for in-process callers that want the old
"clicking the same button un-votes" affordance. It computes the target
and delegates, so all the correctness rules are shared. The HTTP
`POST /api/medias/<id>/vote` endpoint uses `set_vote`, not this.

State transitions, whichever entry point you use:

- Same label as existing → **remove** the vote (treated as "unlabel");
  atlas evidence dropped unless the opposite vote is still present.
- Opposite label → flip; the labeling-progress cache is invalidated
  **from the point where this media first appeared in the training data**.
  Earlier cached steps (whose models never included it) are preserved.
- No prior vote → apply, assign click-time, count as atlas evidence.

```python
from vtscore.state import set_vote, toggle_vote

set_vote(42, "good")                                    # good, whatever it was
set_vote(42, "good")                                    # idempotent no-op
set_vote(42, "good", region_box=(0.1, 0.1, 0.5, 0.5))   # good, with a region
set_vote(42, "none")                                    # un-vote

toggle_vote(42, "good")                                 # add good
toggle_vote(42, "good")                                 # remove (toggle off)
```

The `region_box` argument is normalised `(x0, y0, x1, y1)` in [0, 1] and
is only honoured when the vote is being added and the label is `"good"`.
This is the patch-embedder v2 hook used by image detectors with region
prompts.

### Bulk apply with `replace_all`

`apply_labels_bulk_with_click_time(labels, replace_all=True)` clears any
pre-existing votes for IDs **outside** the bulk list before applying. This
is what `/api/find-label` wants: a detector trained on dataset A holds
A's media IDs in its context, and switching to B must not leak those
stale IDs into B's right-scroll Goods/Bads list.

---

## Click times

A thin wrapper around `DetectorContext.click_counter` and
`DetectorContext.vote_click_times`. `assign_click_time(cid)` bumps the
counter and stores the new value under `cid`; `remove_click_time` drops
the entry; `get_vote_click_times` returns a copy.

```python
from vtscore.state import assign_click_time, get_vote_click_times

assign_click_time(42)   # returns 1
assign_click_time(99)   # returns 2
get_vote_click_times()  # {42: 1, 99: 2}
```

Click times are used by the labeling-session analyzer
(`vtscore.detectors.labeling_progress`) to recreate the model at each
point in the labeling history.

---

## Coverage atlas

A hierarchical k-means partition of a dataset's embeddings, used to pick
the next maximally-diverse training sample and to answer "has this
region been exercised?". The atlas lives on
`DatasetContext.coverage_atlas`, and `vtscore/state/coverage.py` holds the
helpers below that build and mutate it.

The **structure itself** is a separate package,
[`vtscore.coverage`](coverage.md) - a pure algorithm that holds no state and
takes no lock. What it keeps per node (evidence channels, mean-centered
geometry, vMF moments, calibrated typicality) is documented there. One
consequence shows up in this API: `coverage_atlas_label(cid, good=...)` takes
a polarity, where the diversity tree's `diversity_tree_label(cid)` did not.

| Function | Description |
|----------|-------------|
| `build_coverage_atlas(media_dict=None, on_progress=None)` | Build over the active dataset (or *media_dict*); replays the active detector's votes |
| `build_coverage_atlas_for_context(ctx, on_progress=None)` | Build on a specific (possibly inactive) context; replays nothing |
| `build_coverage_atlas_serializable(medias, on_progress=None)` | Build and return the cache payload only; touches no context |
| `restore_coverage_atlas_from_cache(ctx, cached)` | Adopt a pickle-cached atlas onto *ctx*; `False` when it doesn't match |
| `resync_coverage_atlas_to_detector(ds_ctx, det_ctx)` | Clear evidence and replay *det_ctx*'s votes (caller holds `_state_lock`) |
| `get_coverage_atlas()` | Active dataset's atlas, or `None` |
| `coverage_atlas_next_sample(scores=None, threshold=None)` | Surprise-maximising element of the first evidence-free node |
| `coverage_atlas_label(cid, good)` / `coverage_atlas_unlabel(cid)` | Count / uncount one item's evidence |
| `should_auto_build_coverage_atlas(n)` | Whether a dataset of *n* items builds its atlas at load |

`build_coverage_atlas_for_context` is the variant parallel dataset
loading uses - the new dataset is not yet active, so the helper must
operate on the passed-in `ctx` directly. A fresh context has no votes,
so unlike `build_coverage_atlas` it replays nothing.

`coverage_atlas_next_sample` accepts an optional `scores` dict
(typically `DetectorContext.last_learned_scores`). When supplied, the
picker looks at the node's median score: above the threshold it returns
the **lowest**-scored element (likely a surprise miss in a mostly-good
region); below, the **highest**-scored element (likely a surprise hit in
a mostly-bad region).

### Build cost and the auto-build threshold

Hierarchical k-means over millions of vectors costs minutes and
gigabytes, so datasets larger than `COVERAGE_ATLAS_AUTO_THRESHOLD`
(50,000 items) do **not** build the atlas at load;
`should_auto_build_coverage_atlas(n)` is the predicate. Everything
downstream degrades gracefully to score-only sampling when
`get_coverage_atlas()` returns `None`, and the app exposes an on-demand
build (`POST /api/datasets/registry/<id>/coverage-atlas`).

The atlas is *derivable* state - a pure function of the medias' vectors -
so caching it inside the dataset pickle is allowed under the same
reasoning that lets the pickle hold the vectors themselves. That is what
`build_coverage_atlas_serializable` / `restore_coverage_atlas_from_cache`
are for. Restore is refused unless the cached vector-id set matches the
loaded medias exactly, so a remapped, deduplicated or partially-dropped
media set falls through to a rebuild rather than adopting a stale
partition. Old diversity-tree caches fail that format check and rebuild
the same way.

---

## Media lookup

Pure helpers in `vtscore/state/media_lookup.py` for mapping label
entries onto media IDs. Each function takes the media dict explicitly
except `get_dupe_count`, which falls back to the active context.

| Function | Description |
|----------|-------------|
| `build_media_lookup(media_dict)` | Build `(origin_lookup, md5_lookup, name_lookup)` tables |
| `resolve_media_ids(entry, origin_lookup, md5_lookup, name_lookup=None)` | Union of media IDs matched by origin+name and MD5 |
| `find_missing_entries(entries, origin_lookup, md5_lookup, name_lookup=None)` | Labels with no matching media |
| `build_md5_lookup(media_dict)` | The MD5 table alone, when the other two aren't needed |
| `cached_media_lookups(ctx=None)` / `cached_md5_lookup(ctx=None)` | The same tables, memoised on the context and keyed on `media_revision` |
| `collapse_duplicates(media_dict, on_progress=None)` | Merge same-MD5 media into a single `dupe_set` entry |
| `get_dupe_count(media_dict=None)` | Count `dupe_set`-origin medias |
| `next_media_id(media_dict)` | One past the current max cid |

Prefer the `cached_*` variants on any request path. Building the tables
is O(N) with a `json.dumps` per origin, and many routes resolve label
entries against the active dataset on every call (label import/export,
find-stats, add-to-pile, learned sort). The cache invalidates itself
whenever `media_revision` advances, so it can never serve a table built
against a different media set.

The lookup tables are origin-keyed first, MD5-keyed second, name-keyed
third. The fallback to name-keyed matching **only** triggers when the
entry has neither an origin+name pair nor an MD5 - matching by basename
alone on a partial origin would falsely apply a label whenever an
unrelated dataset happens to contain a file with the same basename.

```python
from vtscore.state import build_media_lookup, resolve_media_ids, get_active_context

origin_lu, md5_lu, name_lu = build_media_lookup(get_active_context().medias)
matches = resolve_media_ids(
    {"origin": {...}, "origin_name": "foo.wav", "md5": "abc..."},
    origin_lu, md5_lu, name_lu,
)
```

`collapse_duplicates` rewrites the representative media's `origin` to a
synthetic `"importer": "dupe_set"` origin whose `members` list records
each duplicate's original provenance, then deletes the non-representative
entries.

*Near*-duplicate collapsing - the perceptual-hash pass that catches
re-encoded and reformatted copies - is a pure algorithm and lives with the
media types whose bytes it hashes: see
[`media.md`](media.md#near-duplicate-collapsing).

---

## Sorted-result window cache

`vtscore/state/sort_results_cache.py` holds a process-global,
LRU-bounded `SortResultsCache`. A sort ranks the *whole* dataset, but at
100k / 1M items the ordered list must not ship to the browser in one
response. The route stashes the full descending list, hands back an
opaque `sort_token`, and the client pages deeper windows through
`GET /api/sort/page?token=…`.

| Name | Description |
|------|-------------|
| `SORT_WINDOW_THRESHOLD` (20,000) | Below this the full ranking is transmitted unchanged; at or above it, windowing engages |
| `SORT_WINDOW_HEAD` (500) / `SORT_WINDOW_TAIL` (200) | Above-threshold rows, and past-boundary context rows, in the initial window |
| `initial_window_end(total, above_threshold)` | End index (exclusive) of that initial window |
| `result_score(result)` / `count_above_threshold(results, threshold)` | Read a row's score under either key; count rows above a cut |
| `SortResultsCache.store(...)` / `.page(...)` | Stash a full list and mint a token; page into a stored list |

Only the lightweight ranking rows (`{"id", "score"}` or
`{"id", "similarity"[, "best_region"]}`) are cached - never embeddings
or weights, and nothing is serialised, so this stays inside the "No
Persisted Vectors or MLPs" rule.

The token doubles as a **sort-generation** token: a re-sort mints a fresh
one, so a client holding an old token either reads its own consistent
(if stale) list or gets a 404 and refetches from the top. It can never
page into a *newer* ranking by accident.

---

## Current user

`vtscore/state/current_user.py` is the Flask-free half of "who is this
work being done for?". Same shape as the context resolvers: the library
defines the hook, the app installs a request-aware implementation.

| Function | Description |
|----------|-------------|
| `get_current_user()` | Registered resolver (`g.user` under Flask) → thread-local → `DEFAULT_USER` (`"default"`) |
| `register_request_user_resolver(fn)` / `reset_request_user_resolver()` | Install / restore the request-scoped resolver (the app reads the session) |
| `set_thread_user(username)` / `get_thread_user()` / `thread_user(username)` | Thread-local tier, including a context-manager form |

---

## Setting-persistence hooks

Some library helpers - `get_inclusion`, `set_inclusion`,
`set_calibrate_count`, `set_calibration_fraction`
- read user-pref values that the **app** owns. The library exposes the
hook surface; the app installs the persistence callbacks. Library-only
consumers see purely in-memory mutation.

```python
# vtscore/state/__init__.py
KNOWN_SETTING_KEYS = frozenset({"inclusion", "calibrate_count",
                                "calibration_fraction"})

def register_setting_persister(key: str, fn: Callable[[Any], None]) -> None:
    """Install the persister for *key*, which must be in
    KNOWN_SETTING_KEYS; anything else raises ValueError."""
```

The app side (`vtsearch/shim/`) wires this at startup:

```python
from vtscore.state import register_setting_persister
from vtsearch.settings import set_inclusion, set_calibrate_count

register_setting_persister("inclusion", set_inclusion)
register_setting_persister("calibrate_count", set_calibrate_count)
```

The reads route through `CoreConfig.from_settings()` so the value comes
from the live config snapshot.

Cross-cutting helpers shipped at the package level: `snapshot_medias()`
(shallow copy under the lock), `get_media(cid)`, `clear_medias()`,
`clear_all()`, `get_dataset_display_name()` / `set_dataset_display_name()`.

---

## Thread-safety invariants

- **One re-entrant lock** (`vtscore/state/core.py`,
  `_state_lock = threading.RLock()`) guards every mutation across both
  context registries, every per-context dict, the coverage atlas, and
  the setting cache. RLock so that compound operations like
  `clear_all()` don't deadlock.
- **`__slots__` on both contexts** prevents accidental attribute
  shadowing. Adding a new field requires editing `__slots__`.
- **No global "active" pointer.** The resolver/thread-local/empty
  fallback chain means there is no module-level "current" variable that
  one thread could clobber for another.
- **Empty fallback never raises.** `get_active_context()` with nothing
  registered returns an empty `DatasetContext`, not `None`.
- **Vote dicts are `dict[int, None]`, not sets.** Use
  `votes[cid] = None`; `votes.pop(cid, None)`; `cid in votes`.

For the long-running-operation progress trackers used while these
contexts are being mutated, see [`concurrency.md`](concurrency.md).
