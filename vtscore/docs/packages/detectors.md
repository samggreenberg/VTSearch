# `vtscore.detectors` — Detector lifecycle, training, and labelset materialisation

A **detector** in vtscore is the trained classifier you search with: a
small MLP plus a calibrated threshold plus a `LabelSet`. This package
owns the resolve → embed → train pipeline that turns origin trails into
a trained model, the on-disk JSON store that persists the labelset
(never weights), the registry that tracks every detector the user has
created, and a separate labeling-session analyzer that caches per-step
MLPs so the stopping-condition UI can answer "should I keep voting?"
without retraining.

The generic ML primitives this package builds on live in
[`vtscore.training`](training.md). The embedding façades it uses to turn
origins into vectors live in `vtscore.embedding`.

## Contents

| Module                                       | Concern                                                             |
|----------------------------------------------|---------------------------------------------------------------------|
| `vtscore/detectors/registry.py`              | Persistent registry of detector entries (one JSON manifest)         |
| `vtscore/detectors/store.py`                 | Per-detector on-disk JSON labelset file I/O                         |
| `vtscore/detectors/training.py`              | `train_and_threshold`, `train_and_score`, `train_detector_from_origins` |
| `vtscore/detectors/workflow.py`              | `apply_and_retrain` — combined "apply labels + retrain MLP" entry   |
| `vtscore/detectors/resolver.py`              | Origin → file → embedding pipeline + pluggable resolvers            |
| `vtscore/detectors/labelset_training.py`     | Train from the saved labelset (cross-dataset)                       |
| `vtscore/detectors/labelset_elements.py`     | Stable element IDs, per-element views                               |
| `vtscore/detectors/label_sync.py`            | Sync current votes back into the on-disk labelset                   |
| `vtscore/detectors/label_restoration.py`     | Restore saved labels into the active dataset                        |
| `vtscore/detectors/dataset_sync.py`          | Rehydrate cid-keyed vote state on dataset switch                    |
| `vtscore/detectors/media_seeding.py`         | Seed votes from a detector's example media files                    |
| `vtscore/detectors/labeling_progress.py`     | Per-step model cache + stopping-condition metrics                   |
| `vtscore/detectors/input_spec.py`            | Clipper input-spec extraction and matching                          |

The package `__init__.py` is intentionally minimal — every public name
is imported from the submodule it lives in.

---

## What a detector is

A detector on disk is a JSON file at
`<CoreConfig.from_settings().detectors_dir>/<slug>.json` containing:

```json
{
  "name": "Crying babies",
  "media_type": "audio",
  "input_spec": {"clipper": "sound_tiling", "clipper_params": {"duration": "2.0"}},
  "labelset": {
    "elements": [
      {
        "label": "good",
        "origin": {"importer": "server_folder", "params": {"folder": "/data/esc50"}},
        "origin_name": "1-187207-A-20.wav",
        "filename": "1-187207-A-20.wav",
        "md5": "5d41402abc4b2a76b9719d911017c592",
        "region_box": null
      },
      ...
    ]
  }
}
```

It contains **no weights, no embeddings, no scores**. On every load,
weights are re-derived: each `LabeledElement.origin` is resolved to a
file via an importer or media source, embedded with the active media
type's embedder, and the resulting `(X_list, y_list)` is fed into
`train_and_threshold`. The trained MLP + threshold live in
`DetectorContext.model` / `.threshold` until the process ends or the
labelset changes. This is the invariant the
[CLAUDE.md "No Persisted Vectors or MLPs"](../../../CLAUDE.md) rule
enforces, and it's why `_PICKLE_SAFE_CLASSES` in
`vtscore.security.pickle` does not include any torch types — the only
sanctioned persisted form is the labelset.

---

## Registry

`vtscore/detectors/registry.py` maintains a JSON manifest at
`vtscore.config.DATA_DIR / "detector_registry.json"`. Each entry is a
flat dict with `id` (uuid hex), `name`, `media_type`, `num_training`,
`text_query`, `media_example`, `created_by`, `created_at`. Mutation is
guarded by a module-level `RLock`; reads return deep copies so callers
can mutate freely without races.

```python
from vtscore.detectors import registry

entry = registry.register_detector(name="Crying babies", media_type="audio",
                                    text_query="a baby crying", created_by="alice")
registry.update_detector(entry["id"], num_training=12)
registry.unregister_detector(entry["id"])
```

CRUD: `list_detectors`, `get_detector`, `register_detector`,
`unregister_detector`, `rename_detector`, `update_detector`,
`find_by_name`. Loaded-set tracking:
`add_loaded_detector_id` / `remove_loaded_detector_id` /
`is_detector_loaded` / `get_loaded_detector_ids`. Mode flag:
`is_find_mode` / `set_find_mode`. Test reset: `reset_for_tests`.

### Find mode

When the user runs Find on a detector against a different dataset
(`is_find_mode() == True`), the global vote dicts contain scoring hits,
not real training labels. `vtscore/detectors/label_sync.py` checks
`is_find_mode()` and silently skips the labelset write so the
detector's saved training data is not overwritten.

---

## Store

`vtscore/detectors/store.py` is the low-level file I/O layer.
`get_detectors_dir()` reads `CoreConfig.from_settings().detectors_dir`
(Phase 2 seam — library callers will pass a `CoreConfig` directly after
Phase 8). `_detector_path(name)` slugifies via
`re.sub(r"[^a-z0-9_-]+", "_", name.lower())` and appends `.json`.
`_read_detector(path)` returns the parsed dict or `None`.
`_write_detector(path, data)` writes atomically via tempfile +
`os.fsync` + `os.replace`. The leading underscores are historical;
every other module in the package calls these names.

---

## Training

`vtscore/detectors/training.py` consolidates the canonical detector
training pipeline. Three public entry points cover three different
contexts.

### `train_and_threshold(X_list, y_list, snap=None)`

`vtscore/detectors/training.py:37`. The canonical pipeline used by
every detector route:

1. Cross-calibration threshold via
   `calculate_cross_calibration_threshold` (respects `calibrate_count`
   and `calibration_fraction` settings).
2. Full-data model via `train_model` (respects `inclusion`).
3. Optional safe-threshold blend via `calculate_safe_threshold` when
   `get_safe_thresholds()` is on and a media snapshot is provided.

Returns `(model, threshold)`. Today this function consults
`vtsearch.state` to read `get_inclusion`, `get_calibrate_count`,
`get_calibration_fraction`, and `get_safe_thresholds` — a Phase 3 seam.
Library consumers running outside an app should pass these via
`CoreConfig` (the existing `state` shims read from there already).

### `train_and_score(...)`

`vtscore/detectors/training.py:273`. Vote-aware online trainer. Takes
the current `clips_dict`, `good_votes` / `bad_votes`, threshold-
related settings, and an optional `vote_region_boxes` map and returns
`(results, threshold, model)`:

- `results` — list of `{"id": cid, "score": rounded_float, "best_region": [...]?}`
  dicts, sorted by raw score descending.
- `threshold` — the operating point (cross-cal or safe-blended).
- `model` — the trained `nn.Sequential` (or `None` when training was
  not possible).

The function is the per-vote hot path:

- `_build_vote_tensors` (line 142) gathers training vectors from
  `good_votes` / `bad_votes`. When a vote on an image has a `region_box`
  and the source media has a stored `patch_grid`, the training vector
  is pooled on the fly via
  `vtscore.media.patch_embed.box_to_vote_vector` — image-level voting
  on a patch-aware media gets the same vector as in v1.
- Below 6 labels the cross-cal trainings are skipped (threshold defaults
  to 0.5) because the safe-threshold blend would discard the result
  anyway.
- `_score_all_media` (line 187) takes one of two paths. Region-aware
  datasets flatten all `(media, region)` vectors into one tensor, run
  a single forward pass, and max-pool per media so the winning region
  index can be surfaced. Plain datasets use the cached
  `(N, D)` embedding matrix from
  `vtscore.embedding.matrix.get_embedding_matrix_for_snap`.

```python
from vtscore.detectors.training import train_and_score

results, threshold, model = train_and_score(
    clips_dict=snap,
    good_votes={1: None, 5: None},
    bad_votes={2: None, 7: None},
    inclusion_value=0,
    safe_thresholds=False,
    calibrate_count=2,
    calibration_fraction=0.5,
)
```

### Origin-based helpers

| Function                                                  | Behaviour                                                            |
|-----------------------------------------------------------|----------------------------------------------------------------------|
| `train_detector_from_origins(good_origins, bad_origins, inclusion, media_type, embedder_name, ...)` (line 405) | Resolve every entry to a file, embed with the named embedder, train. `embedder_name` is required — pass the embedder the detector was originally trained with so re-derived vectors don't drift onto the media type's default. Returns `(weights_dict, threshold)` or `(None, 0.5)` on insufficient data. |
| `collect_media_origins(media_ids, snap)` (line 373) | Extract `origin / origin_name / filename / md5` for every cid that appears in `snap`. |
| `serialize_weights(model)` (line 111) | Pickle-safe `state_dict` dump via `tensor.tolist()`. Round-trips through `build_model_from_weights`. |
| `validate_good_bad_split(y_list)` (line 24) | Precondition; raises `ValueError` when either class is empty. |

---

## Workflow

`vtscore/detectors/workflow.py` is a single function:

```python
def apply_and_retrain(
    detector_id: str,
    det_ctx: DetectorContext,
    new_entries: list[dict],
    detector_name: str,
) -> tuple[int, bool]:
```

It is the combined "apply external labels and retrain the detector"
entry point. Used by the label-import route and the labelset-source
sync flow:

1. Resolves every label entry against the current dataset's medias
   (via `build_media_lookup` + `resolve_media_ids`).
2. Calls `apply_label(cid, label)` for each match — votes show up in
   the UI like the user just cast them.
3. Calls `sync_labels_to_loaded_detector()` so the on-disk labelset
   captures the new votes.
4. Retrains the MLP via `train_and_score` when both polarities have
   labels, stores the model and threshold on `det_ctx`, and snapshots
   the voted medias into `det_ctx.training_medias`.

Returns `(resolved_count, trained_bool)`.

The key implementation detail is the **context-override pattern** at
the top:

```python
with override_detector_context(det_ctx):
    snap = snapshot_medias()
    ...
```

`override_detector_context` lives in `vtscore.state.core` and pushes
`det_ctx` to the top of the detector-context resolution chain for the
duration of the block. Inside, every call that reads the "active"
detector context (the `good_votes` / `bad_votes` proxies, the label-
sync helpers) sees `det_ctx`, regardless of whether the caller is
inside a Flask request, a background thread, or a test. Library
consumers that don't have Flask installed get this for free — the
proxies fall through to the context-override stack before they look at
`flask.g`.

---

## Origin resolution

`vtscore/detectors/resolver.py` turns an `Origin` (importer name +
params) into a file on disk and then an embedding. Two pluggable
resolver protocols sit at the top:

```python
@runtime_checkable
class SourceResolver(Protocol):
    def __call__(self, stack: ExitStack, origin: dict, origin_name: str,
                 filename: str) -> Path | None: ...

@runtime_checkable
class ImporterResolver(Protocol):
    def __call__(self, origin: dict, origin_name: str,
                 filename: str) -> Path | None: ...
```

`register_source_resolver(fn)` and `register_importer_resolver(fn)`
replace the default implementations. Library callers that want to
extend resolution (e.g. fetch from a private CDN before falling
through to importers) plug in here. On first use,
`_auto_wire_resolvers` installs defaults that delegate to
`vtscore.datasets.sources.get_source_for_origin` (registering
`source.cleanup` on the caller's `ExitStack`) and
`vtscore.datasets.importers.get_importer` +
`importer.resolve_file`.

### Public surface

```python
from vtscore.detectors.resolver import (
    resolve_file_context, resolve_file_from_origin,
    embed_file, resolve_label_embeddings, ResolvedLabels,
)

with resolve_file_context(origin, origin_name, filename) as path:
    if path is not None:
        emb = embed_file(path, media_type="audio")
```

| Function                                                             | Behaviour                                                          |
|----------------------------------------------------------------------|--------------------------------------------------------------------|
| `resolve_file_context(origin, origin_name, filename)` (line 195)     | **Context manager** — must wrap any code that reads the file. Some sources (PullWrest, http_archive cache misses) materialise files in a tempdir they own; the `ExitStack` keeps that tempdir alive until the `with` block exits. |
| `resolve_file_from_origin(origin, origin_name, filename)` (line 223) | One-shot convenience. Safe for `path.exists()` checks; unsafe for any call that may garbage-collect the source. |
| `embed_file(file_path, media_type, embedder_name="")` (line 376)     | Pick the embedder for the media type (named, else first registered) and call `embedder.embed_media(media_from_path(...))`. |
| `resolve_label_embeddings(labels, media_type, progress_callback=None)` (line 691) | Batch entry point. Returns `ResolvedLabels`.            |

`ResolvedLabels` is a dataclass with `embeddings`, `labels`,
`resolved_count`, `total_count`, `missing_entries` plus the
`available_fraction` and `has_good_and_bad` properties.

Two synthetic origin types are special-cased: `dupe_set` (tries each
member origin in turn) and `converter` (reconstructs a parent origin
from `parent_importer` / `parent_path` / `parent_url` params and
recurses). Clipped origins — audio `clip_start`/`clip_end`, image
`clip_box`, text `clip_index`, and origin-stored clipper chains — are
handled transparently: the chain is replayed on the resolved source
file via `vtscore.datasets.clipper_chain.replay_chain_on_file` before
embedding.

---

## Labelset materialisation

The detector's saved labelset is dataset-agnostic — every element is
keyed by origin / md5. At load time, the labelset has to be resolved
into a concrete embedding cache so the MLP can train. This is what
`labelset_training.py` does, and it's why one labelset trained on
dataset A can score dataset B.

### `populate_label_embeddings(det_ctx, labelset, *, media_type, snap, on_progress=None)`

`vtscore/detectors/labelset_training.py:127`. For each element in the
labelset, ensure `det_ctx.label_embeddings[stable_element_id(elem)]` is
populated:

1. If the element resolves to a cid in the active `snap`, reuse
   `snap[cid]["embedding"]` (zero I/O). Region-voted elements re-pool
   from the source `patch_grid` every pass — the cache is keyed by
   stable element id (origin/md5), intentionally stable across region
   edits.
2. Otherwise, resolve the element's origin to a file via
   `resolve_file_context` and embed it with the active dataset's
   embedder.
3. Cache the result on `det_ctx.label_embeddings`.

If the active dataset's embedder name differs from
`det_ctx.embedder`, the cache is cleared first — mixing vectors from
two embedders into one MLP produces garbage.

### `build_xy_from_labelset(det_ctx, labelset)`

`vtscore/detectors/labelset_training.py:185`. Walk the labelset
elements (filtering to `good` / `bad`), look up each cached embedding,
and return `(X_list, y_list)`.

### `train_from_labelset(det_ctx, labelset, *, media_type, snap, on_progress=None)`

`vtscore/detectors/labelset_training.py:207`. Populate the cache,
build `(X, y)`, run `train_and_threshold`, store the result on
`det_ctx.model` / `det_ctx.threshold`. Returns `True` on success,
`False` when fewer than 2 cached vectors exist or one class is
missing.

### `labelset_train_and_score(det_ctx, labelset, *, media_type, clips_dict, ...)`

`vtscore/detectors/labelset_training.py:241`. Like `train_and_score`
but trains on the full labelset (cross-dataset labels) and scores only
the active `clips_dict`. Returns the same `(results, threshold, model)`
tuple.

### `update_cache_for_cid(det_ctx, labelset, cid, snap)`

`vtscore/detectors/labelset_training.py:316`. Refresh the cache entry
for whichever labelset element corresponds to one cid. Called after a
vote toggles a media item in the active dataset, so the labelset's
training vector tracks the live embedding.

---

## Element identity (cross-dataset label restore)

`vtscore/detectors/labelset_elements.py` handles per-element identity
in the labelset detail view. The right pane in label/train mode is
*labelset-driven* — each row is a `LabeledElement` from the JSON
file, not a cid — so the frontend needs a stable id per element plus
a way to map elements back to the currently-loaded dataset.

| Function                                                       | Behaviour                                                            |
|----------------------------------------------------------------|----------------------------------------------------------------------|
| `stable_element_id(elem)` (line 22)                            | SHA-1 of `element_key(elem)` (origin / md5) truncated to 16 hex chars; stable across label flips |
| `find_element_by_id(elements, target_id)` (line 38)            | Linear scan; returns `(idx, elem)` or `None`                         |
| `resolve_current_dataset_cid(elem)` (line 46)                  | Match against the active dataset by origin + name, then md5; never triggers file resolution |
| `resolve_element_to_path(elem)` (line 67)                      | Context manager yielding the on-disk path via `resolve_file_context`; required for previewing labelset entries from datasets that aren't loaded |
| `build_element_view(elem, *, media_type, click_times, learned_scores)` (line 87) | Serialise one element to the right-pane row shape   |
| `build_labels_detail(detector_data)` (line 123)                | Response body for the labels-detail route; splits into `good` / `bad` lists |
| `apply_element_vote_in_data(detector_data, target_id, vote)` (line 153) | Toggle semantics mirror `toggle_vote`: same vote → remove, opposite → flip. Returns `(changed, updated_element_or_None, action)` |

---

## Sync helpers

### `label_sync.sync_labels_to_loaded_detector()` (line 110)

Persist current votes into the loaded detector's labelset on disk.
Called automatically after each vote. The sync is **non-destructive
across datasets**: existing labelset entries whose origin doesn't
match anything in the current dataset are preserved verbatim;
entries matching current-dataset media are reconciled against the
active votes (replaced, flipped, or removed). Skipped entirely when
`is_find_mode()` is True — find-mode votes are scoring hits on a
different dataset and don't belong in the training set.

### `label_restoration.restore_labels_from_detector(det_data)` (line 11)

Take a detector-JSON dict, resolve every labelset element against the
active dataset, and apply matching votes silently
(`apply_label(..., silent=True)`). Two-pass matching:

1. **Fast** — `(origin, origin_name)`, then md5, then origin_name
   fallback (via `build_media_lookup` + `resolve_media_ids`).
2. **Slow** — for unresolved entries, materialise the origin file via
   `resolve_file_context`, compute its md5, and check the md5 lookup
   again. This is what makes cross-dataset label restore work: a
   detector trained on dataset A still restores labels when you load
   dataset B if both share the same underlying files.

Returns the number of labels restored.

### `dataset_sync.ensure_votes_match_active_dataset()`

Rehydrate per-dataset detector state on dataset switch. No-op unless
the dataset id or labelset file mtime has changed since the detector
last saw it. When triggered, clears `good_votes` / `bad_votes` /
`label_history` / etc., replays `restore_labels_from_detector`, and
caches the parsed labelset + mtime so subsequent requests within the
same `(dataset_id, file_mtime)` tuple are no-ops.

Also calls `invalidate_model_on_embedder_switch` on every request as
its first step (cheap O(1) check) so a stale MLP can't survive a
dataset swap to a different embedder.

### `dataset_sync.invalidate_model_on_embedder_switch(det_ctx, new_embedder)`

Drop a cached MLP whose `det_ctx.embedder` stamp doesn't match
*new_embedder*. The MLP is dimensionally and semantically tied to its
embedder: scoring a CLAP-trained MLP against SigLIP vectors yields
either a tensor-shape error or — when the two embedders share a
dimensionality — silent garbage. Clears `model`, `threshold`,
`calibration_cache`, `label_embeddings`, and the stamp itself; the
next training pass re-stamps via `populate_label_embeddings` /
`apply_and_retrain`. No-op when either stamp is empty (the invariant
fires only on a confident mismatch). Used by
`ensure_votes_match_active_dataset` (per-request invalidation) and
`scoring._resolve_or_train_detector` / `find._score_dataset`
(defence-in-depth at each cached-MLP read site).

### `dataset_sync.first_media_embedder(medias)`

Return the embedder name of the first media in *medias*, or `""` when
the dict is empty or the field is missing. Convenience accessor used
by the invalidator and the scoring routes — every media in a single
dataset shares an embedder, so the first entry is representative.

### `media_seeding.seed_good_votes_from_examples(examples)` (line 10)

Seed good votes from a detector's `media_example` list. Each
`{"type": "media", "value": filename}` entry is read from
`DATA_DIR / "example_media"`; matching media (by md5) get a good vote
in place, non-matching files are embedded, inserted with an
`example_media` origin, then voted. Path traversal is guarded by
`file_path.resolve().relative_to(server_media_dir.resolve())`.

---

## Labeling-session analyzer

`vtscore/detectors/labeling_progress.py` is a separate cache from
`vtscore.concurrency.progress`. The two used to share the
`progress.py` filename; this one was renamed to make the distinction
obvious.

- **`vtscore.concurrency.progress`** — long-running-operation
  progress and cancellation (`ProgressTracker`, `dataset_progress`,
  `sort_progress`, etc.).
- **`vtscore.detectors.labeling_progress`** — per-step MLP cache and
  stopping-condition metrics. Used by the labeling-progress UI to
  answer "should I keep voting?" without retraining.

A single `threading.RLock` (`_progress_lock`) protects all module
state: `_cache_inclusion` (rebuild trigger), `_cached_steps` (one entry
per label-history step with `model` / `threshold` / `good_ids` /
`bad_ids` / `stability` / `diversity`), `_cache_good_ids` /
`_cache_bad_ids` (running label sets), `_cache_prev_predictions`
(stability baseline), `_cache_diversity_tree`, and `_live_models`
(models injected by `train_and_score` during sorting, keyed by
`(frozenset(good), frozenset(bad))`).

### Public API

| Function                                        | Behaviour                                                              |
|-------------------------------------------------|------------------------------------------------------------------------|
| `clear_progress_cache()`                        | Drop everything. Call when votes are cleared, medias change, etc.      |
| `invalidate_progress_cache_from(media_id)`      | Truncate the cache to just before `media_id` first appeared (vote-flip case) |
| `inject_live_model(good, bad, model, threshold)`| Register a model produced by `train_and_score` so the cache can reuse it |
| `recreate_model_at_time(snap, history, t, inclusion)` | Return the model + threshold + good/bad ids for step `t`           |
| `calculate_error_cost_over_time(...)`           | Per-step FPR/FNR-weighted cost on current votes                        |
| `calculate_prediction_stability_over_time(...)` | Per-step flip-count on unlabeled medias                                |
| `calculate_diversity_level_over_time(...)`      | Per-step diversity-tree coverage                                       |
| `compute_labeling_status(...)`                  | Aggregate red / yellow / green status for the Smart / Stable / Span indicators |
| `analyze_labeling_progress(...)`                | Run the three "over-time" functions and bundle the result              |

### Smart / Stable / Span statuses

`compute_labeling_status` returns three indicators:

- **Smart** — fits a linear regression slope over the most recent 10
  error-cost values; green when the relative slope is above
  `-0.015` (cost has leveled off).
- **Stable** — fraction of unlabeled predictions that flipped between
  successive steps; green when the recent 10-step average is below
  0.5% and no single step exceeded 1%.
- **Span** — diversity-tree coverage; green at
  `CoreConfig.from_settings().autopilot_goal_diversity` nodes (default
  40), yellow at 10, red below.

Each color comes with a `reason` string the UI displays as a tooltip.

---

## Input spec

`vtscore/detectors/input_spec.py` is a small helper module that
records, exports, and compares the clipper a detector was trained on
— so the CLI can warn when a dataset's clipping doesn't match.

```python
def extract_input_spec_from_medias(medias) -> dict | None: ...
def build_detector_meta(detector_data, *, threshold=None) -> dict: ...
def apply_detector_meta(detector_data, detector_meta) -> bool: ...
def clipper_matches(detector_spec, dataset_spec) -> bool: ...
```

An `input_spec` is `{"clipper": "sound_tiling", "clipper_params": {"duration": "2.0"}}`
or `None`. Detectors with no input spec accept any dataset (legacy
behaviour). The `detector_meta` block is what travels with a labelset
through a `LabelsetSource`, so a downstream consumer can reproduce the
training clipper without reading the detector JSON.

---

## Invariants worth restating

- **Labelset is the only persisted form.** Detector JSON files store
  `LabeledElement` lists; never weights, never embeddings, never
  scores. Every load re-derives the MLP from origins.
- **Origins are stable across datasets.** `stable_element_id` is
  computed from origin / md5 fields, so the same training label
  identifies the same source file no matter which dataset is loaded.
- **Resolver is pluggable.** `register_source_resolver` and
  `register_importer_resolver` let library consumers extend file
  resolution without modifying the package.
- **Threading.** The registry, store, and labeling-progress cache all
  use `RLock`. `train_and_score` and `train_from_labelset` honour the
  thread-safe RNG behaviour of `train_model`. The
  `override_detector_context` pattern in `workflow.apply_and_retrain`
  is the canonical way to bind a context for the duration of a
  background-thread operation.
