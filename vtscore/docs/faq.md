# FAQ & Common Gotchas

Questions that come up often when working with `vtscore`, plus pitfalls
that are easy to walk into without warning. If you hit something not
covered here, check the per-package docs under [packages/](packages/) or
the architecture overview in [architecture.md](architecture.md).

## Contents

- [General](#general)
- [Loading data](#loading-data)
- [Embedders and embeddings](#embedders-and-embeddings)
- [Training and scoring](#training-and-scoring)
- [Detectors on disk](#detectors-on-disk)
- [State and contexts](#state-and-contexts)
- [Plugins](#plugins)
- [Performance](#performance)
- [Threading and concurrency](#threading-and-concurrency)

---

## General

### How is `vtscore` different from `vtsearch`?

`vtscore` is the **library**: dataset loaders, embedders, detector training,
detector lifecycle, evaluation. It has no Flask, no Angular, no settings
JSON. `vtsearch` is the **application** that wraps `vtscore` with a
Flask + Angular UI, a per-user settings system, and authentication.

The two ship from the same git repository today. See
[architecture.md](architecture.md) for the dependency direction
(`vtsearch` depends on `vtscore`, never the reverse).

### Can I use `vtscore` without `vtsearch`?

Yes. The whole point of the extraction was to make `vtscore` a standalone
library. See [integration.md](integration.md) for what installing it in
your own app looks like.

### Why don't I see `vtscore` on PyPI?

It hasn't been published yet. Both packages ship from the single root
`pyproject.toml` (which declares the `vtsearch` distribution). A
standalone `vtscore` PyPI release is deferred until a real external
consumer asks for it; until then, install the repository in editable
mode.

### How do I check the version?

```python
import vtscore
print(vtscore.__version__)  # '0.1.0'
```

`vtscore` uses **independent semver** - the constant is manually bumped
in `vtscore/__init__.py` per release. (`vtsearch.__version__` is
different: it's the UTC timestamp of the git HEAD commit. Don't confuse
the two.)

## Loading data

### Why does my `load_dataset_from_folder` call raise `ValueError`?

`ValueError: Invalid media type: <name>` means `media_type` names no
registered type. You do **not** need to import the media-type plugin first:
importing `vtscore.datasets.loader` pulls in `vtscore.media`, whose
sub-package scan registers all six shipped types (`audio`, `image`,
`text`, `video`, `document`, `face`). So check for a typo, or - for a
plugin type - for a warning that its sub-package failed to import
(`python -W default -c "import vtscore.media"` shows it).

`from vtscore.media import audio` in the examples is documentation of
intent, not a registration step.

### Why doesn't `load_dataset_from_folder` return the medias dict?

It populates the `medias` dict you pass in place and returns `None`, the
same shape as every importer's `run(field_values, medias)`. Pass an empty
dict, get a populated dict.

### Why are my loaded medias' embeddings empty?

Because loading and embedding are separate stages, by design. A loader
emits media dicts with `embeddings == {}` unless an importer supplied
pre-computed vectors; the embed stage fills the rest in:

```python
from vtscore.datasets.stages.embedding import embed_missing

embed_missing(medias)                  # media type's default embedder
embed_missing(medias, "clap_music")    # or a named one
```

Call it once after the loader (the app's load pipeline does this for you).
It is keyed per embedder, so running it a second time with a different
embedder name adds that embedder's vectors alongside the first's rather
than replacing them.

### Why is `media["origin"]` `None` after a folder load?

`load_dataset_from_folder` only stamps an origin when you pass one:

```python
from vtscore.datasets.origin import Origin

load_dataset_from_folder(
    folder_path=folder, media_type="audio", medias=medias,
    origin=Origin(importer="server_folder",
                  params={"path": str(folder), "media_type": "audio"}).to_dict(),
)
```

Importers do this for you. It matters because origins are what a saved
detector re-resolves at load time - labels built from origin-less medias
cannot be re-derived.

### What's the shape of a media item?

A plain dict keyed by sequential integer IDs; the type key is
`media_type`, and vectors live in the plural `embeddings` dict keyed by
embedder name (read them with
`vtscore.embedding.media_vectors.media_embedding(media)` - there is no
singular `media["embedding"]`). The full annotated shape is in
[concepts.md §1](concepts.md#1-media).

### Can I load datasets from URLs?

Use `vtscore.datasets.importers.http_archive` - it accepts a URL,
downloads the archive into the staging area, and runs a normal folder
import on the extracted contents. The URL is recorded in the `origin`
so the labelset can be re-resolved later.

### How do I add a new media type (3D mesh, point cloud, …)?

See [extending/media-types.md](extending/media-types.md). The contract
is small: a `MediaType` subclass, an `EMBEDDER` for it, a `CLIPPERS`
list, and a sub-package under `vtscore/media/<your-type>/`. Media types
have no entry-point group; to ship one out-of-tree, symlink the
sub-package into `vtscore/media/`.

## Embedders and embeddings

### What dimension are the embeddings?

Depends on the embedder — each one declares its dimensionality via the
`embedding_dim` property:

<!-- BEGIN GENERATED: embedder-dims -->
<!-- Generated by scripts/gen-docs-inventories.py; do not edit by hand. Refresh with: python scripts/gen-docs-inventories.py -->

| Embedder | Media type | D |
|---|---|---|
| `clap_general` | `audio` (default) | 512 |
| `ast` | `audio` | 768 |
| `beats` | `audio` | 768 |
| `clap` | `audio` | 512 |
| `clap_music` | `audio` | 512 |
| `paraspeechclap` | `audio` | 768 |
| `whisper_encoder` | `audio` | 512 |
| `face` | `face` | 512 |
| `siglip` | `image` (default) | 768 |
| `clip` | `image` | 512 |
| `clip_l` | `image` | 768 |
| `dinov2_patch` | `image` | 768 |
| `dinov2_single` | `image` | 768 |
| `dinov3_patch` | `image` | 768 |
| `dinov3_single` | `image` | 768 |
| `eupe_patch` | `image` | 768 |
| `eupe_single` | `image` | 768 |
| `sift_vlad` | `image` | 8192 |
| `sift_vlad_doc` | `image` | 8192 |
| `siglip2` | `image` | 768 |
| `siglip2_l` | `image` | 1152 |
| `siglip_l` | `image` | 1152 |
| `e5` | `text` (default) | 768 |
| `bge` | `text` | 768 |
| `xclip` | `video` (default) | 768 |
| `languagebind` | `video` | 768 |
| `videomae` | `video` | 768 |

<!-- END GENERATED: embedder-dims -->

Verify in your own code via `media_embedding(medias[1]).shape[0]`
(`from vtscore.embedding.media_vectors import media_embedding`).

### Are embeddings normalised?

Yes, always. `MediaEmbedder.embed_media` and `embed_text` L2-normalize
whatever the subclass's `_embed_media_impl` / `_embed_text_impl` returns,
so every vector stored in `media["embeddings"]` and every text-query vector
is unit-norm regardless of the underlying model. A dot product between
them is their cosine similarity.

### How does `embed_text_query` know which model to use?

`embed_text_query(text, media_type, enrich=False, embedder_name="")`
uses `embedder_name` when you pass one, and otherwise the default
embedder for `media_type`. If your dataset was embedded with a
non-default embedder (e.g. `clap_music`), pass that name - a query
embedded in a different model's space ranks garbage. Embedders without a
text tower (`supports_text is False`) return `None`.

### Where do model weights live?

In `vtscore.config.MODELS_CACHE_DIR`: `$VTSEARCH_MODELS_DIR` if set,
otherwise `<data dir>/models`, where the data dir is `$VTSEARCH_DATA_DIR`
or the repository's `data/`. The cache uses the standard HuggingFace
layout; you can pre-warm it offline by downloading the relevant model IDs
(see `vtscore/config/models.py` for constants like `CLAP_MODEL_ID`).

### Can I disable model downloads?

Set `HF_HUB_OFFLINE=1` in your environment. Embedders will fail to load
if their weights aren't already cached, but that's the point - you
catch the missing-cache error explicitly instead of accidentally
fetching at runtime.

## Training and scoring

### How many labels do I need?

The detector works from about 4 labels (2 good, 2 bad) and improves through
about 50. Above 100 the gains are mostly noise. The production head is linear
(a single `Linear(D, 1)` fitted as a maximum-margin SVM), which is what makes
it usable that low: with 3-5 positives an MLP is under-determined and its
scores wobble from retrain to retrain. See
[docs/ML.md](../../docs/ML.md#the-three-heads-which-one-is-shipped-and-why).

### Is training deterministic?

Yes, given a fixed seed. `train_model` takes a `seed: int = 42` keyword
argument; the production SVM head passes it to liblinear (which touches no
global RNG), and the BCE heads additionally isolate theirs via
`torch.random.fork_rng()`. The same seed always produces bit-identical weights.

### What does `inclusion_value` do?

It moves the **decision threshold**, not the model. Inclusion never enters
training: `train_model` is class-balanced regardless, so every item's score
is identical at every inclusion, and only the cut moves. Range `[-10, +10]`.

Under the shipped cut (`FoldAnchoredCut.threshold_at`, see the next answer),
`k` is a cost ratio: `inclusion_cost_weights(k)` prices a miss at `2**k`
times a false alarm (or a false alarm at `2**-k` times a miss, below zero).
The `mid_tilt` rule starts from the fold-anchored midpoint cut at `k = 0`
and shifts its quantile by however far the rate-optimal cut moves at those
weights. How far one step moves the line therefore depends on the fitted
mixtures; it is not a fixed share of misses, and on measured data the steps
come out shorter than their nominal size (`docs/ML.md`, "The knob
under-delivers its own steps").

When a detector has no fitted estimator, the cut falls back to
`vtscore.training.thresholds.conformal_threshold`, where `k` has a budget
meaning: positive `k` caps the threshold at the `alpha(k) = min(1, 0.25 *
2**-k)` quantile of the calibration positives, and negative `k` walks it up
toward the 0.75 quantile of the positives at -10, above a false-positive
guard.

Both rules are monotone non-increasing in `k`, so the included sets are
nested: everything included at `k` is still included at `k + 1`. That is
what makes "cut off at Inclusion 1, verify up to Inclusion 4" well-defined.

Inclusion is no longer a user preference (#4269): the app's control is the
**balance** (#4413), F-beta's beta, where the user says which way to lean
between precision and recall: the line keeps the top of the ranking where the
estimated F-beta peaks, and a spot check of random picks measures how much of
it is right and how much it found
(`vtscore.training.thresholds.spot_check`, #4272). `k` stays underneath, as
the unit Autopilot's acquisition offset and the Smart indicator's pricing are
measured in. The entry points that used to take
the user's Inclusion (`train_and_score(inclusion_value=...)` and its
siblings, `vtscore.state.set_inclusion`, `CoreConfig(inclusion=...)`) are
deprecated: they accept only `0`, with a `DeprecationWarning`. The
lower-level functions that take `inclusion_value` as a cut position
(`conformal_threshold`, `threshold_from_folds`, `train_svm`, ...) are
unchanged.

### How is the decision threshold chosen?

By fusing the labels with the haystack's own score distribution rather
than picking between them. Per calibration fold, a 2-component mixture
is fitted to that fold model's scores over the whole collection with the
fold's *held-out* votes clamped to their labeled component; each fold's
midpoint cut is carried to the final model as a quantile, the folds are
averaged in quantile space, and a cut at another inclusion shifts that
quantile by the rate-optimal cut's own displacement from Inclusion 0. See
`vtscore/training/thresholds/anchored.py:fold_anchored_gmm_threshold`. It is
unconditional - there used to be a `safe_thresholds` toggle, but the
fused estimator measured better at every label count, so the toggle was
removed. `calculate_safe_threshold` remains the fallback for label sets
too small to form calibration folds.

## Detectors on disk

### What's actually saved when I `save_detector`?

A JSON file under `CoreConfig.detectors_dir / "<slug>.json"` (the name is
slugified: lowercased, anything outside `[a-z0-9_-]` collapsed to `_`). It
contains:

- Detector metadata: `name`, `media_type`, and optionally `embedder_type`.
- The `LabelSet` under a `"labelset"` key, serialised as
  `{"labels": [...]}`: one entry per `LabeledElement`, each with `md5`,
  `label`, `origin_name`, `origin`, and optional `filename`, `category`,
  `region_box` and `metadata`.
- Nothing else - **no embeddings, no model weights**.

The embedder *name* is not in this file; it is recorded on the detector's
registry entry (`vtscore.detectors.registry`), stamped the first time
training runs. That is why the reload path takes `embedder_name` as an
explicit argument.

On load, the library re-derives every embedding from the origin and
re-trains the head. This is by design - see
[architecture.md §The no-persisted-vectors rule](architecture.md#the-no-persisted-vectors-rule).

### Doesn't re-deriving embeddings on every load take forever?

For a few hundred labels, no - it's seconds. For thousands of labels
across remote storage, yes. The hand-roll mitigations:

1. **Cache files locally.** Don't pull from S3 on every load if you can
   help it. The `vtscore.datasets.sources.http_archive` source already
   does this.
2. **Train from a loaded dataset instead of from origins.**
   `train_detector_from_origins` always resolves and re-embeds every
   origin. If a dataset pickle (which *does* persist embeddings) already
   holds the labelled items, load it, match labels to medias by `md5`,
   and hand those vectors to
   `vtscore.detectors.training.train_and_threshold` - no file access, no
   embedding.

### Can I import an existing detector from another tool?

Build a `LabelSet` of `LabeledElement`s and call
`vtscore.detectors.store.save_detector(name, labelset, media_type=...)`.
Or implement a custom `LabelImporter` plugin - see
[extending/label-importers.md](extending/label-importers.md).

### How do I delete a detector?

```python
from vtscore.detectors.store import _detector_path
_detector_path("name").unlink(missing_ok=True)
from vtscore.detectors.registry import unregister_detector
unregister_detector("detector_id")
```

(The `_detector_path` underscore is mildly intentional - direct
filesystem manipulation is escape-hatch territory. `save_detector` /
`load_detector` are the public pair; there is deliberately no
`delete_detector`, because deleting the file is only half the job - the
registry entry has to go too, and only the caller knows its id. For
app-style flows the route layer handles both.)

## State and contexts

### What's the difference between `DatasetContext` and `DetectorContext`?

`DatasetContext` holds dataset-intrinsic state: the medias dict, the
coverage atlas, the cached embedding matrix. `DetectorContext` holds
detector-intrinsic state: votes, the trained head, the threshold, the
labelset source. One dataset can be open with multiple detectors active
against it simultaneously; both contexts are independently registered.

### How does the library know which context to operate on?

Through a resolution chain - explicit override, installed resolver hook,
thread-local binding, then a shared empty fallback - spelled out in
[architecture.md §Resolution chain](architecture.md#resolution-chain-for-active-context).

The part that bites: `get_active_context()` / `get_active_detector_context()`
never return `None`. Outside an app they return the shared empty fallback,
so two library callers that both skip binding will write into the same
context. Bind one explicitly with `set_thread_dataset_context(ctx)` /
`set_thread_detector_context(ctx)`.

### Why does my background thread see no active context?

Thread-local context doesn't inherit. If you start a thread, you must
call `set_thread_dataset_context(ctx)` / `set_thread_detector_context(ctx)`
inside the thread function - the calling thread's binding doesn't
follow.

The library's own workers (the dataset-load pipeline and
`vtscore.concurrency.async_jobs`) bind the context for you; if you're
rolling your own thread pool, you do it.

### Are `medias`, `good_votes`, etc. importable?

Not from `vtscore`. Those proxy objects are app-tier - they live in
`vtsearch.state_proxies` and are re-exported by `vtsearch.state`.
Library code uses `DatasetContext.medias` and `DetectorContext.good_votes`
directly via a resolved context.

If you need to write to vote dicts, use the public ops in
`vtscore.state` - `toggle_vote(media_id, vote)`,
`apply_label(media_id, label)`, `clear_votes()`, etc. They take no context
argument: they act on the *active* detector context, so bind one first
(`set_thread_detector_context(ctx)` or `override_detector_context(ctx)`).

## Plugins

### How do plugins get discovered?

Every `PluginRegistry` walks its target package on construction,
imports every module, and harvests the sentinel attribute (`IMPORTER`,
`EXPORTER`, `EMBEDDER`, `CLIPPERS`, `CONVERTER`, …). It also scans the
matching `importlib.metadata` entry-point group (`vtscore.<family>`)
and adds any plugins declared there.

Discovery is **eager by default**: by the time
`vtscore.datasets.importers.__init__` returns, every importer is
registered. If you need pre-discovery state (rare; mostly for tests),
construct the registry with `eager=False`.

### My third-party plugin isn't being discovered. Why not?

Check, in order:

1. Did you declare it in `pyproject.toml` under the right group?
   ```toml
   [project.entry-points."vtscore.importers"]
   my_importer = "my_pkg.my_module:IMPORTER"
   ```
   The target must be the **instantiated** plugin (the module's sentinel),
   not the class. The group name is `vtscore.<family>` for library
   plugins, `vtsearch.<family>` for app-tier (settings) plugins; media
   types, embedders, clippers and cleaners have no group at all.
2. Did you actually `pip install -e .` your package after editing
   `pyproject.toml`?
3. Is `name` unique among registered plugins? Built-ins win on name
   clashes; the clashing third-party plugin is skipped with a warning.
4. Does the entry-point target raise on import? The error is logged as a
   warning and the plugin becomes a tombstone: absent from `list_*()`,
   re-raising the original error when `get_*()` hands it out. Check
   `python -W default -c "import vtscore.datasets.importers"` (or the
   family's package) for the warning.

### How are built-ins different from entry-point plugins?

Mechanically: built-ins are modules inside the `vtscore/` package tree
with a sentinel; entry-point plugins are external packages declared via
`importlib.metadata`. Semantically: built-ins are bundled with the
library; entry-point plugins live elsewhere. The runtime behaviour is
identical.

## Performance

### Why is my first call slow?

Embedder model weights are downloaded on first use (~30s–2min depending
on size) and cached for next time. After that, first-use within a
process loads from disk (~5s) and subsequent uses are instant.

### How big is the embedding matrix in memory?

`(N * D * 4)` bytes for a float32 matrix. 100,000 audio items × 512
dimensions × 4 bytes = ~200 MB. The cached matrix on `DatasetContext`
lives only in process memory; clearing it is `invalidate_embedding_matrix(ctx)`.

### My detector training is using only one CPU core. Why?

`VTSEARCH_TORCH_THREADS` defaults to `1` as a memory-saving default (each
extra thread allocates its own scratch buffers), and
`vtscore.media.torch_setup.ensure_torch_configured()` applies it with
`torch.set_num_threads()` when the first embedder loads. Set
`$VTSEARCH_TORCH_THREADS` before importing `vtscore` to raise it. (The
VTSearch app additionally exports `OMP_NUM_THREADS` / `MKL_NUM_THREADS`
from the same value before importing torch; a library script that wants
the same effect on OpenMP/MKL sets them itself.)

### How do I parallelise dataset loading?

The pipeline already does - `vtscore.datasets.load_pipeline` runs
embedding under a `ConcurrencyGate` capped by
`CoreConfig.max_concurrent_dataset_embeddings`. Bumping that value lets
multiple datasets embed in parallel.

Within a single dataset, embedder fan-out is automatic: each importer's
`run()` decides the fan-out. The folder importer fans out per-file using
a pool sized by `cap_workers_by_memory()`.

## Threading and concurrency

### Is `train_model` thread-safe?

Yes. The production linear-SVM head is fitted by liblinear, which reads
no global RNG; the BCE heads use a local `torch.Generator` for weight init
and `torch.random.fork_rng()` to isolate the dropout RNG. Two parallel
`train_model` calls produce the same results as two serial calls.

### Is the embedder cache thread-safe?

Yes. `MediaEmbedder.load_models()` takes a per-class lock, so when several
threads hit a cold embedder at once, one performs the load and the rest
wait for it and return. Forward passes are serialised across all embedders
by a shared embed lock. If you're spawning many workers from cold, pre-warm
with `vtscore.embedding.preload_predicted_embedders(extra_media_types=None,
extra_embedders=None)` - with no arguments it warms whatever the registries
predict this deployment will need.

### Can I cancel a long-running operation?

Yes, with `vtscore.concurrency.progress.cancel_dataset_progress()`
(import from the module - `vtscore.concurrency` is an implicit namespace
package with no `__init__.py`, so there is nothing to import *from* it
directly), or `cancel_dataset_task(task_id)` for one operation. Both set
the cancel event on the loading task's own `ProgressTracker`; the
operation polls `tracker.check_cancelled()` at chunk boundaries and
raises `CancelledError` when set. Worker functions that want to be
cancellable should call it periodically themselves (not from inside a
bounded loop - use a `while True` loop with the cancel check at the top;
bounded loops can run to completion before the cancel signal arrives in
some race conditions). Reporting progress through the thread's own sink
usually does this for you: the callbacks the load pipeline binds check
cancellation before recording each tick.
