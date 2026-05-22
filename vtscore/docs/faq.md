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

`vtscore` is the **library**: dataset loaders, embedders, MLP training,
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

`vtscore` uses **independent semver** — the constant is manually bumped
in `vtscore/__init__.py` per release. (`vtsearch.__version__` is
different: it's the UTC timestamp of the git HEAD commit. Don't confuse
the two.)

## Loading data

### Why does my `load_dataset_from_folder` call do nothing?

You probably forgot to import the media-type plugin:

```python
# WRONG — KeyError: "audio"
from vtscore.datasets.loader import load_dataset_from_folder

medias = {}
load_dataset_from_folder(Path("/data"), media_type="audio", medias=medias)
```

```python
# RIGHT
from vtscore.media import audio  # noqa: F401 — registers MediaType + embedders
from vtscore.datasets.loader import load_dataset_from_folder

medias = {}
load_dataset_from_folder(Path("/data"), media_type="audio", medias=medias)
```

Importing `vtscore.media.audio` triggers the `MEDIA_TYPE`, `EMBEDDER`,
and `CLIPPERS` sentinels. The library doesn't import every media-type
plugin by default — you pay only for what you use.

### Why doesn't `load_dataset_from_folder` return the medias dict?

Historical reason: the in-place population pattern predates the
functional refactor. `medias` is mutated in place; the function returns
`None`. Pass an empty dict, get a populated dict.

### What's the shape of a media item?

```python
medias[42] = {
    "id": 42, "media_type": "audio", "embedder": "audio_clap",
    "file_size": 318456,
    "md5": "5d41402abc4b2a76b9719d911017c592",
    "embedding": np.ndarray,            # shape (D,), float32
    "filename": "barks/poodle.wav",
    "category": "custom",
    "origin": {"importer": "server_folder", "params": {...}},
    "origin_name": "barks/poodle.wav",
    "media_bytes": None, "media_string": None,
    "media_path": "/data/audio/barks/poodle.wav",
    "duration": 2.41,
}
```

See [concepts.md §1](concepts.md#1-media) for the full description.

### Can I load datasets from URLs?

Use `vtscore.datasets.importers.http_archive` — it accepts a URL,
downloads the archive into the staging area, and runs a normal folder
import on the extracted contents. The URL is recorded in the `origin`
so the labelset can be re-resolved later.

### How do I add a new media type (3D mesh, point cloud, …)?

See [extending/media-types.md](extending/media-types.md). The contract
is small: a `MediaType` subclass, an `EMBEDDER` for it, a `CLIPPERS`
list, and a sub-package under `vtscore/media/<your-type>/` (or
distributed via entry points).

## Embedders and embeddings

### What dimension are the embeddings?

Depends on the embedder:

| Embedder | D |
|----------|---|
| LAION-CLAP (audio default) | 512 |
| SigLIP (image default) | 768 |
| X-CLIP (video default) | 768 |
| E5-base-v2 (text default) | 768 |
| BGE-base-en-v1.5 (text alt) | 768 |

Verify in your own code via `medias[1]["embedding"].shape[0]`.

### Are embeddings normalised?

Depends on the embedder. LAION-CLAP and SigLIP return L2-normalised
vectors out of the box; E5 doesn't (it requires explicit normalisation
for cosine sim, which the helpers in `vtscore.training.region_similarity`
do for you). Don't assume — check `np.linalg.norm(embedding)`.

### How does `embed_text_query` know which model to use?

The `media_type` parameter routes the query to the default embedder for
that media type. If your dataset uses a non-default embedder
(e.g. CLAP Music instead of CLAP), pass the explicit embedder name via
`vtscore.media.get_embedder(name).embed_text(query)`.

### Where do model weights live?

`CoreConfig.data_dir / "models"` by default (which resolves to
`./data/models/` unless `$VTSEARCH_DATA_DIR` is set). The cache uses the
standard HuggingFace layout; you can pre-warm it offline by downloading
the relevant model IDs (see `vtscore/config.py` for the constants like
`CLAP_MODEL_ID`).

### Can I disable model downloads?

Set `HF_HUB_OFFLINE=1` in your environment. Embedders will fail to load
if their weights aren't already cached, but that's the point — you
catch the missing-cache error explicitly instead of accidentally
fetching at runtime.

## Training and scoring

### How many labels do I need?

The MLP works from about 4 labels (2 good, 2 bad) and improves through
about 50. Above 100 the gains are mostly noise. The MLP hidden layer
auto-sizes from the training-set count (see `_auto_hidden_dim` in
`vtscore/training/mlp.py`).

### Is training deterministic?

Yes, given a fixed seed. `train_model` takes a `seed: int = 42` keyword
argument and isolates its RNG via `torch.random.fork_rng()`. Different
seeds produce slightly different models (mostly within noise); the same
seed always produces bit-identical weights.

### What does `inclusion_value` do?

It biases the BCE loss toward predicting more positives (positive
`inclusion_value`) or more negatives (negative value). Range `[-10,
+10]`; 0 means class-balanced. Each step doubles the class weight in
that direction.

### What's the "safe threshold" mode?

When `safe_thresholds=True`, the library blends the cross-calibration
threshold with a more conservative GMM-derived fallback. Useful when
you're scoring noisy data and would rather miss a hit than over-include
false positives. See `vtscore/training/thresholds.py:calculate_safe_threshold`.

### Why is `train_and_threshold` calling `vtsearch.state`?

It's not — that was a Phase-2 seam. Today `train_and_threshold` lives in
`vtscore.detectors.training` and reads its knobs from `CoreConfig`. If
you see an old reference somewhere, please update it.

## Detectors on disk

### What's actually saved when I `save_detector`?

A JSON file under `CoreConfig.detectors_dir / "<name>.json"`. It
contains:

- Detector metadata (`name`, `media_type`, `embedder`).
- The `LabelSet`: a list of `LabeledElement`s, each with `md5`, `label`,
  `origin_name`, `origin`, and optional `region_box` and `metadata`.
- Nothing else — **no embeddings, no model weights**.

On load, the library re-derives every embedding from the origin and
re-trains the MLP. This is by design — see
[architecture.md §The no-persisted-vectors rule](architecture.md#the-no-persisted-vectors-rule).

### Doesn't re-deriving embeddings on every load take forever?

For a few hundred labels, no — it's seconds. For thousands of labels
across remote storage, yes. The hand-roll mitigations:

1. **Cache files locally.** Don't pull from S3 on every load if you can
   help it. The `vtscore.datasets.sources.http_archive` source already
   does this.
2. **Pre-warm the dataset.** Load the relevant dataset pickle (which
   *does* persist embeddings) before constructing the detector; the
   `train_detector_from_origins` resolver can short-circuit to the
   cached embedding when origins match.

### Can I import an existing detector from another tool?

Convert it to the `LabelSet` JSON shape and call
`vtscore.detectors.store.save_detector(name, labelset)`. Or implement a
custom `LabelImporter` plugin — see
[extending/label-importers.md](extending/label-importers.md).

### How do I delete a detector?

```python
from vtscore.detectors.store import _detector_path
_detector_path("name").unlink(missing_ok=True)
from vtscore.detectors.registry import unregister_detector
unregister_detector("detector_id")
```

(The `_detector_path` underscore is mildly intentional — direct
filesystem manipulation is escape-hatch territory. For app-style flows
the route layer handles this.)

## State and contexts

### What's the difference between `DatasetContext` and `DetectorContext`?

`DatasetContext` holds dataset-intrinsic state: the medias dict, the
diversity tree, the cached embedding matrix. `DetectorContext` holds
detector-intrinsic state: votes, the trained MLP, the threshold, the
labelset source. One dataset can be open with multiple detectors active
against it simultaneously; both contexts are independently registered.

### How does the library know which context to operate on?

Resolution chain, highest precedence first:

1. `override_*_context()` context manager
2. Installed resolver hook
3. Thread-local set via `set_thread_*_context()`
4. `None` (the library has no implicit global default)

See [architecture.md §Resolution chain](architecture.md#resolution-chain-for-active-context).

### Why does my background thread see no active context?

Thread-local context doesn't inherit. If you start a thread, you must
call `set_thread_dataset_context(ctx)` / `set_thread_detector_context(ctx)`
inside the thread function — the calling thread's binding doesn't
follow.

The app's `vtsearch.shim` does this automatically when spawning
worker threads from the job manager; if you're rolling your own thread
pool, you do it.

### Are `medias`, `good_votes`, etc. importable?

Not from `vtscore`. Those proxy objects are app-tier — they live in
`vtsearch.shim.state_proxies` and are re-exported by `vtsearch.state`.
Library code uses `DatasetContext.medias` and `DetectorContext.good_votes`
directly via a resolved context.

If you need to write to vote dicts, use the public ops:
`vtscore.state.toggle_vote(ctx, cid, label)`, `apply_label(ctx, cid, label)`,
`clear_votes(ctx)`, etc.

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
   my_importer = "my_pkg.my_module:MyImporter"
   ```
   The group name is `vtscore.<family>` for library plugins,
   `vtsearch.<family>` for app-tier (settings) plugins.
2. Did you actually `pip install -e .` your package after editing
   `pyproject.toml`?
3. Is `name` unique among registered plugins? Built-ins win on name
   clashes; third-party plugins with the same name are silently
   shadowed.
4. Does the entry-point target raise on import? Errors are logged as
   warnings; check `python -W default -c "import vtscore.datasets"`
   for the warning.

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

### My MLP training is using only one CPU core. Why?

`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` are set at import time as a
memory-optimisation default (the project runs in many low-memory
containers). Override by setting `$VTSEARCH_TORCH_THREADS` before
importing `vtscore`, or by calling `torch.set_num_threads()` after
import.

### How do I parallelise dataset loading?

The pipeline already does — `vtscore.datasets.load_pipeline` runs
embedding under a `ConcurrencyGate` capped by
`CoreConfig.max_concurrent_dataset_embeddings`. Bumping that value lets
multiple datasets embed in parallel.

Within a single dataset, embedder fan-out is automatic: each importer's
`run()` decides the fan-out. The folder importer fans out per-file using
a pool sized by `cap_workers_by_memory()`.

## Threading and concurrency

### Is `train_model` thread-safe?

Yes. It uses a local `torch.Generator` for weight init and
`torch.random.fork_rng()` to isolate the dropout RNG. Two parallel
`train_model` calls produce bit-identical results to two serial calls.

### Is the embedder cache thread-safe?

Reads are; concurrent first-time loads of the same backbone may
download the weights twice (the second download overwrites the first
with identical bytes). The cost is one wasted HTTP request; the result
is correct. If you're spawning many workers from cold, pre-warm with
`vtscore.embedding.preload_predicted_embedders(names)`.

### Can I cancel a long-running operation?

Yes, with `vtscore.concurrency.progress.cancel_dataset_progress()`. The
operation polls `check_dataset_cancelled()` at chunk boundaries and
raises `CancelledError` when set. Worker functions that want to be
cancellable should call `check_dataset_cancelled()` periodically
themselves (not from inside a bounded loop — use a `while True` loop
with the cancel check at the top; bounded loops can run to completion
before the cancel signal arrives in some race conditions).
