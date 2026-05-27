# `vtscore.state`

The runtime state package. Every loaded dataset and every loaded detector
gets its own context object that bundles the mutable data structures
operating on it - votes, label history, click times, the cached embedding
matrix, the diversity tree, the trained MLP, the calibrated threshold.
The package also owns the resolution chain that picks which context is
"active" for the current call (Flask request, background thread, or test
harness), the helpers that operate on the active context, and the
process-wide registries that map IDs to contexts. Everything is
protected by a single re-entrant lock; there is no implicit global
"active" pointer.

Related docs: [`concurrency.md`](concurrency.md) for the job/progress
infrastructure that drives long-running operations on these contexts.

## Contents

- [The two context classes](#the-two-context-classes)
- [The resolution chain](#the-resolution-chain)
- [Context registries](#context-registries)
- [Resolver hooks (app integration)](#resolver-hooks-app-integration)
- [Scoped overrides and `with_*_context`](#scoped-overrides-and-with__context)
- [Vote operations](#vote-operations)
- [Click times](#click-times)
- [Diversity tree](#diversity-tree)
- [Media lookup](#media-lookup)
- [Setting-persistence hooks](#setting-persistence-hooks)
- [Thread-safety invariants](#thread-safety-invariants)

---

## The two context classes

`DatasetContext` and `DetectorContext` live in `vtscore/state/core.py:91`
and `vtscore/state/core.py:122`. Both use `__slots__` so accidentally
shadowing an attribute raises immediately. Every field is in-memory
only; datasets persist via pickle, detectors via labelset JSON (origins
+ labels only, never MLP weights - see the "No Persisted Vectors or
MLPs" rule in `CLAUDE.md`).

### `DatasetContext`

Per-dataset mutable state. One context per loaded dataset.

| Attribute | Type | Description |
|-----------|------|-------------|
| `dataset_id` | `str` | Registration key |
| `medias` | `dict[int, dict[str, Any]]` | `cid -> media dict` for every loaded item |
| `diversity_tree` | `DiversityTree \| None` | Hierarchical clustering of the embeddings (built lazily) |
| `dataset_display_name` | `str \| None` | UI-overridable name |
| `_emb_matrix_ids` | `list[int] \| None` | Sorted media-ID list the matrix corresponds to |
| `_emb_matrix` | `np.ndarray \| None` | Cached `(N, D)` contiguous float32 embedding matrix |

The cached matrix is rebuilt lazily by
`vtscore.embedding.matrix.get_embedding_matrix` and reused across
cosine sort, MLP scoring, and diversity-tree construction so the library
doesn't rebuild an `(N, D)` matrix per call. `clear_medias()` drops both
the matrix and the diversity tree.

### `DetectorContext`

Per-detector mutable state. Vote state, training artefacts, and the
trained MLP live here - never on `DatasetContext`. One detector can be
trained on labels collected against many datasets; the labelset is the
canonical persisted form.

| Attribute | Type | Description |
|-----------|------|-------------|
| `detector_id` / `name` / `media_type` / `embedder` | `str` | Identity fields |
| `good_votes` / `bad_votes` | `dict[int, None]` | Set-shaped vote dicts |
| `label_history` | `list[tuple[int, str, float]]` | `(media_id, label, timestamp)` events |
| `vote_click_times` | `dict[int, int]` | Ordinal assigned at vote time |
| `vote_region_boxes` | `dict[int, tuple[float, float, float, float]]` | Optional box per good vote |
| `click_counter` | `int` | Monotonically-increasing ordinal source |
| `last_learned_scores` | `dict[int, float]` | Most recent `train_and_score` output |
| `textsort_suggestions` | `list[str]` | LRU of recent text-sort queries |
| `find_initial_labels` | `dict[int, str]` | Labels the detector applied during a Find run |
| `inclusion` | `int \| None` | Per-detector inclusion fraction override |
| `training_medias` | `dict[int, dict[str, Any]]` | Voted medias with embeddings |
| `label_embeddings` | `dict[str, np.ndarray]` | `stable_element_id -> embedding`, built from origins |
| `model` | `torch.nn.Sequential \| None` | Trained MLP (process-lifetime only) |
| `threshold` | `float` | Calibrated decision threshold |
| `labelset_good_count` / `labelset_bad_count` | `int` | Counts from the on-disk labelset (cross-dataset) |
| `votes_dataset_id` | `str` | Dataset ID the cid-keyed dicts are valid against |
| `cached_labelset` | `LabelSet \| None` | Parsed labelset, reused across requests |
| `cached_labelset_mtime` | `float` | Mtime of the JSON the cache was built from |
| `labelset_source` | `dict \| None` | Active sync target |
| `calibration_cache` | `tuple[Any, float] \| None` | Memoised cross-calibration threshold |

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
that operate on "the" active context walk a four-tier chain (see
`vtscore/state/core.py:331` for the detector side; the dataset side is
identical minus the forced-override tier). Highest precedence wins.

| Tier | Source | How it's set | Lifetime |
|------|--------|--------------|----------|
| 1 | Forced override (detector only) | `override_detector_context(ctx)` context manager | Block scope |
| 2 | Request resolver | `register_dataset_context_resolver(fn)` / `register_detector_context_resolver(fn)` | Process lifetime; usually reads `flask.g` |
| 3 | Thread-local | `set_thread_dataset_context(ctx)` / `set_thread_detector_context(ctx)` | Thread lifetime |
| 4 | Empty fallback | The module-level `_empty_*_context` | Always present |

The empty fallback exists so library code can read `medias`, `good_votes`,
etc. without a context registered and see empty containers rather than a
`NoneType` error.

```python
from vtscore.state import get_active_context, get_active_detector_context

ds_ctx = get_active_context()             # never returns None
det_ctx = get_active_detector_context()   # never returns None
```

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
hooks are at `vtscore/state/core.py:66`:

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
chain (tier 1). Used by `vtscore/detectors/workflow.py:38` so the
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

| Function | File | Description |
|----------|------|-------------|
| `clear_votes()` | `votes.py:25` | Drop all votes, history, click times, region boxes |
| `toggle_vote(cid, vote, region_box=None)` | `votes.py:122` | Toggle semantics: same vote removes; opposite flips |
| `apply_label(cid, label, *, silent=False, region_box=None)` | `votes.py:209` | Set label unconditionally (for imports). `silent=True` skips history + diversity tree |
| `apply_label_with_click_time(cid, label)` | `votes.py:257` | Same as `apply_label` but assigns a click-time ordinal |
| `apply_labels_bulk_with_click_time(labels, replace_all=False)` | `votes.py:283` | Apply many labels under a single lock acquisition |
| `add_label_to_history(cid, label)` | `votes.py:49` | Append `(cid, label, time)` to `label_history` |
| `add_textsort_suggestion(text)` / `get_textsort_suggestions()` | `votes.py:62` | LRU of recent queries |
| `update_learned_scores(scores)` / `get_learned_scores()` | `votes.py:85` | Cache last `train_and_score` output |
| `set_find_initial_labels(labels)` / `get_find_initial_labels()` | `votes.py:99` | Snapshot of detector-assigned labels |

### Toggle semantics

`toggle_vote` matches the UI's click behaviour:

- Same label as existing → **remove** the vote (treated as "unlabel");
  diversity tree marker dropped unless the opposite vote is still present.
- Opposite label → flip; the labeling-progress cache is invalidated
  **from the point where this media first appeared in the training data**.
  Earlier cached steps (whose models never included it) are preserved.
- No prior vote → apply, assign click-time, mark in diversity tree.

```python
from vtscore.state import toggle_vote

toggle_vote(42, "good")                                    # add good
toggle_vote(42, "good")                                    # remove (toggle off)
toggle_vote(42, "good", region_box=(0.1, 0.1, 0.5, 0.5))   # add with region
toggle_vote(42, "bad")                                     # flip good→bad
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

## Diversity tree

A hierarchical k-means clustering over a dataset's embeddings, used to
pick the next maximally-diverse training sample. The tree lives on
`DatasetContext.diversity_tree`; the helpers in `state/diversity.py`
manipulate the active dataset's tree. The implementation is in
`vtscore/state/diversity_tree.py:34` (`DiversityTree`).

The tree partitions the embedding space recursively, tracks which leaves
have been "seen" (had a label applied to one of their members), and
provides a "next unseen sample" query. The frontend uses this to walk
through diverse parts of the dataset before homing in on a category.

| Function | Description |
|----------|-------------|
| `build_diversity_tree(media_dict=None, on_progress=None)` | Build over the active dataset (or `media_dict`); replays votes |
| `build_diversity_tree_for_context(ctx, on_progress=None)` | Build on a specific (possibly inactive) context |
| `get_diversity_tree()` | Active dataset's tree, or `None` |
| `diversity_tree_next_sample(scores=None, threshold=None)` | First unseen-node element |
| `diversity_tree_label(cid)` / `diversity_tree_unlabel(cid)` | Mark / unmark a leaf |

`build_diversity_tree_for_context` is the variant parallel dataset
loading uses - the new dataset is not yet active and the helper must
operate on the passed-in `ctx` directly.

`diversity_tree_next_sample` accepts an optional `scores` dict
(typically `DetectorContext.last_learned_scores`). When supplied, the
picker looks at the node's median score: above the threshold it returns
the **lowest**-scored element (likely a surprise miss in a mostly-good
region); below, the **highest**-scored element (likely a surprise hit in
a mostly-bad region).

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
| `collapse_duplicates(media_dict, on_progress=None)` | Merge same-MD5 media into a single `dupe_set` entry |
| `get_dupe_count(media_dict=None)` | Count `dupe_set`-origin medias |
| `next_media_id(media_dict)` | One past the current max cid |

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

---

## Setting-persistence hooks

Some library helpers - `get_inclusion`, `set_inclusion`,
`set_calibrate_count`, `set_calibration_fraction`, `set_safe_thresholds`
- read user-pref values that the **app** owns. The library exposes the
hook surface; the app installs the persistence callbacks. Library-only
consumers see purely in-memory mutation.

```python
# vtscore/state/__init__.py:159
def register_setting_persister(key: str, fn: Callable[[Any], None]) -> None:
    """Recognised keys: inclusion, calibrate_count,
    calibration_fraction, safe_thresholds."""
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

- **One re-entrant lock** (`vtscore/state/core.py:36`,
  `_state_lock = threading.RLock()`) guards every mutation across both
  context registries, every per-context dict, the diversity tree, and
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
