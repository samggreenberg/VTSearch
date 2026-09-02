# `vtscore.embedding`

A thin façade over the `MediaEmbedder` registry in
[`vtscore.media`](media.md) plus the runtime-management glue every
caller needs: lazy device resolution, model preload prediction, the
text-query LRU, and the cached `(N, D)` matrix used by every sort and
learned-sort path. No `vtscore.embedding` module contains an embedder
implementation - those live next to their `MediaType` in
`vtscore/media/<type>/embedder_*.py`. What this package owns is the
calling convention: a small set of well-known functions that hide
"look up the default embedder for media type X, lazy-load it, embed
the file, return the vector".

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/embedding/helpers.py` | The `embed_*_file` façades and `embed_text_query` (with its LRU) |
| `vtscore/embedding/loader.py` | Model loading / initialisation, preload prediction, concurrency defaults |
| `vtscore/embedding/matrix.py` | The lazy cached `(N, D)` matrix, its region-expanded twin, and the on-disk sidecar |
| `vtscore/embedding/media_vectors.py` | Per-media vector access out of the `media["embeddings"]` dict |
| `vtscore/embedding/normalize.py` | Canonical L2 normalisation - the single ingest chokepoint |
| `vtscore/embedding/binding.py` | Role-typed (text / patch / structural) embedder binding for a dataset |
| `vtscore/embedding/__init__.py` | Re-exports the public façade |

---

## What's an embedder, in this package's vocabulary?

A *embedder* is a `MediaEmbedder` instance registered under
`embedders_for_type(media_type)`. The first entry in that list is
the **default embedder** for that media type; the helpers below all
pick it implicitly. To use a non-default embedder, call
`vtscore.media.get_embedder(name)` directly or pass `embedder_name`
to `embed_text_query`.

Models are **lazy-loaded**. The first call to any `embed_*` helper
that hits a fresh embedder triggers `MediaEmbedder.load_models()`,
which downloads weights (or reads them from
`vtscore.config.MODELS_CACHE_DIR`) and stores the model on
`self._model`. Subsequent calls reuse the in-memory model. There is
no eviction.

---

## File embedders

`vtscore/embedding/helpers.py`. Four one-liners that pick the
default embedder for a media type, build a minimal media dict from a
file path, and run a forward pass.

```python
from pathlib import Path
from vtscore.embedding import (
    embed_audio_file,
    embed_image_file,
    embed_video_file,
    embed_paragraph_file,
)

vec = embed_audio_file(Path("clip.wav"))    # (D,) float32, or None
vec = embed_image_file(Path("photo.jpg"))
vec = embed_video_file(Path("scene.mp4"))
vec = embed_paragraph_file(Path("doc.txt"))
```

Each returns `Optional[np.ndarray]` - `None` when no embedder is
registered for that media type, or when the embedder fails on the
input. The vector dimension depends on the default embedder for the
type (CLAP → 512, SigLIP → 768, X-CLIP → 512, E5 → 768).

These wrappers exist mainly so older code can keep calling
`embed_audio_file(p)` without knowing about the embedder registry.
For new code, prefer:

```python
from vtscore.media import embedders_for_type
from vtscore.media.embedder import media_from_path

emb = embedders_for_type("audio")[0]    # default
vec = emb.embed_media(media_from_path(Path("clip.wav")))
```

which exposes the embedder so you can also call `embed_text`,
`embed_media_bulk`, or `patch_forward` against the same instance.

---

## Text queries (and their LRU)

```python
def embed_text_query(
    text: str,
    media_type: str,
    enrich: bool = False,
    embedder_name: str = "",
) -> Optional[np.ndarray]: ...
```

(`vtscore/embedding/helpers.py`.) Embed a free-form text query
into the vector space of a media type, so it can be cosine-scored
against media embeddings. When `embedder_name` is set, that specific
embedder is used; otherwise it falls back to the default for the
given `media_type`. `enrich=True` runs `embed_text_enriched`, which
averages the query across the embedder's `description_wrappers`
(e.g. `"a photo of {text}"`, `"a video of {text}"`). Most embedders
declare no wrappers - the ensemble was measured to *lose* to the typed
query on `siglip`, `clap`, `e5` and `bge` (#3127/#3341) - so on those
`enrich=True` is simply `embed_text`.

Results are cached in a small process-wide LRU
(`_query_cache` in `vtscore/embedding/helpers.py`):

| Aspect       | Detail                                                 |
|--------------|--------------------------------------------------------|
| Max size     | 32 entries                                             |
| Key          | `(embedder_name, media_type, enrich, text)`            |
| Eviction     | LRU (oldest evicted on insert past max)                |
| Thread-safe  | Yes; guarded by `_query_cache_lock`                    |
| Persisted    | **No.** Process-lifetime only.                         |

The embedder name is in the key because the same query embedded by
CLIP vs. SigLIP lands in different vector spaces - caching by
`(media_type, text)` alone would silently return wrong-space vectors
when the dataset uses a non-default embedder.

```python
from vtscore.embedding import embed_text_query, clear_text_query_cache

q1 = embed_text_query("a dog barking", "audio")
q2 = embed_text_query("a dog barking", "audio")   # cache hit
clear_text_query_cache()                          # drop everything
```

> **No persisted vectors.** Per the project invariant - and the LRU's
> 32-entry cap - query embeddings live for the process lifetime only.
> Never serialise them to disk or to settings.

---

## Device resolution

```python
def get_torch_device() -> torch.device: ...
```

(`vtscore/embedding/loader.py`.) Resolves
`vtscore.config.DEVICE` (env `$VTSEARCH_DEVICE`, default `"auto"`)
to a concrete `torch.device`. `"auto"` picks `cuda` when available,
`mps` on Apple silicon, otherwise `cpu`. Torch is imported lazily
here - calling `get_torch_device()` before any embedder runs is
safe.

```python
import torch
from vtscore.embedding import get_torch_device

dev = get_torch_device()   # torch.device("cuda") on a GPU host
model.to(dev)
```

---

## Runtime initialisation

```python
def initialize_models(on_progress: ProgressCallback | None = None) -> None: ...
```

(`vtscore/embedding/loader.py`.) Pass `on_progress` to render console
progress bars for the two heavy first-time imports it triggers (scikit-learn
and transformers, ~10s combined on a cold start); omit it (the default, used
by tests and the eval CLI) to run silently. Sets up the runtime environment:

1. Creates `vtscore.config.MODELS_CACHE_DIR` on disk.
2. Calls `ensure_torch_configured()` which applies
   `torch.set_num_threads(TORCH_THREADS)` **if** torch is already
   imported (otherwise defers to the first code path that imports it).
3. Runs `gc.collect()`.

It does **not** load any embedder models. Call it once at process
start so the model cache directory exists before the first embedder
download.

---

## Smart preload

The smart-preload functions walk the dataset and detector registries
and warm the embedders the user is most likely to need next. The
predict-step is dataset/detector-driven:

- For each registered dataset: `entry["embedder"]` if set, else the
  default for `entry["media_type"]`.
- For each registered detector: the default for `entry["media_type"]`.

Unknown names are dropped. Order is deterministic (datasets first,
then detectors) so identical registries produce identical preload
lists across runs.

```python
def predict_embedders_to_preload() -> list[str]: ...
def preload_predicted_embedders() -> list[str]: ...
def smart_preload_in_background() -> None: ...
def predict_embedder_for_dataset(dataset_id: str) -> str: ...
def preload_embedder_for_dataset(dataset_id: str) -> str: ...
```

(`vtscore/embedding/loader.py`.)

| Function                            | Sync? | Returns                              |
|-------------------------------------|-------|--------------------------------------|
| `predict_embedders_to_preload()`    | sync  | List of embedder names to warm.      |
| `preload_predicted_embedders()`     | sync  | List of names that were warmed.      |
| `smart_preload_in_background()`     | thread | Spawns a daemon thread; idempotent.  |
| `predict_embedder_for_dataset(id)`  | sync  | One embedder name (or `""`).         |
| `preload_embedder_for_dataset(id)`  | thread | Spawns a daemon; returns target name.|

`preload_predicted_embedders()` is the foreground variant used at
startup; it prints intermediate status to stdout with a console
progress bar. `smart_preload_in_background()` is the same predictor
but quiet - used when a new dataset is registered, so the
implied embedder is warmed without blocking the registration call.

```python
from vtscore.embedding import (
    initialize_models,
    smart_preload_in_background,
    preload_predicted_embedders,
)

initialize_models()
preload_predicted_embedders()           # foreground at startup
# ... later, when a new dataset registers ...
smart_preload_in_background()           # daemon thread, idempotent
```

---

## Backbone accessors

Three legacy helpers expose the underlying torch model + processor
of the most common embedders, for callers that need to drive the
backbone directly (custom forward passes, intermediate-layer probes,
etc.):

```python
def get_clap_model():   # (model, processor) for the "clap" embedder
def get_xclip_model():  # (model, processor) for the "xclip" embedder
def get_e5_model():     # SentenceTransformer for the "e5" embedder
```

(`vtscore/embedding/loader.py`.) Each calls
`get_embedder(name).loaded_backbone()`, the public accessor on
`MediaEmbedder` - which loads the model if it is not already resident
and returns `(model, processor)`. `get_e5_model` returns just the
first element, since a `SentenceTransformer` needs no processor.
Prefer the public `embed_media` / `embed_text` API for new code -
these helpers exist because parts of the legacy app reach into the
backbone.

`loaded_backbone()` is the supported way to reach any embedder's raw
model, not just these three. Its default implementation reads the
`_model` / `_processor` attribute convention that `load_models()` itself
relies on, so it works for embedders built the usual way; an embedder
holding its backbone elsewhere overrides it (as `languagebind` does, to
return its tokenizer). It raises `RuntimeError` rather than returning
`None` when no backbone is resident after loading.

---

## Cached embedding matrix

For any sort or learned-sort over a loaded dataset, you need every
media's embedding as a contiguous `(N, D) float32` array.
`vtscore/embedding/matrix.py` keeps that array on `DatasetContext`
and rebuilds it only when the context's `media_revision` counter
changes (bumped on every `medias` mutation).

```python
def get_embedding_matrix(ctx: DatasetContext) -> tuple[list[int], np.ndarray]: ...
def invalidate_embedding_matrix(ctx: DatasetContext) -> None: ...
def get_embedding_matrix_for_snap(snap: dict) -> tuple[list[int], np.ndarray]: ...
def scoreable_snapshot(snap: dict, embedder_name: str | None = None) -> tuple[dict, list[int]]: ...
```

(`vtscore/embedding/matrix.py`.)

### How the cache works

`DatasetContext._emb_matrix`, `DatasetContext._emb_matrix_ids`, and
`DatasetContext._emb_matrix_revision` are the cache.
`get_embedding_matrix(ctx, embedder_name=None)`
(`vtscore/embedding/matrix.py`) runs in four phases, and only the two
cheap ones hold the lock - a large numpy stack must not stall every
other request's state-sync:

1. **Locked.** Snapshot `sorted_ids` and `revision = ctx.media_revision`.
   If `ctx._emb_matrix_revision == revision` and a matrix is present,
   return the cached `(ids, matrix)` immediately. Otherwise take a
   shallow ref-copy of `ctx.medias` and release the lock.
2. **Unlocked.** Try the on-disk sidecar (below); on a miss, allocate an
   `(N, D) float32` array and fill it row by row from
   `media_embedding(...)` (the per-embedder `media["embeddings"]` dict).
3. **Locked.** Re-check that `ctx.media_revision` still equals the
   snapshotted `revision` before storing - a mutation during the
   unlocked build must not cache a stale matrix - and populate the
   cache if it does.
4. **Unlocked, best-effort.** Persist a freshly-built matrix as a
   sidecar, unless the sidecar is where it came from or phase 3 lost
   the race.

Pass an explicit *embedder_name* (one of a multi-embedder dataset's
bound slots) to build from that embedder's vectors instead. The named
path builds fresh every call and never touches the cache or the
sidecar - both are reserved for the hot primary path. A name that
happens to equal the primary collapses back onto the cached path.

An empty dataset returns `([], np.empty((0, 0), dtype=np.float32))`; a
media missing the requested vector raises `ValueError`, and one whose
vector is the wrong width raises `MismatchedVectorError`.

### Skipping what can't be scored

Those raises are right for the dataset's *own* matrix: the load
pipeline's drop-none stage has already removed vector-less media, so a
raise there is a real invariant break. They are wrong for an arbitrary
snapshot handed to the scorer, which carries no such guarantee — the CLI
scores importer output that never went through that stage, and one bound
embedder of a multi-embedder dataset can have failed on media another
succeeded on.

`scoreable_snapshot(snap, embedder_name=None, *, region_rows=False)`
answers the two questions the builders raise on — is there a vector, and
is it a 1-D row of the same width as the rest — and returns
`(scoreable, dropped_ids)`. Scoring paths filter first and score what is
left, so one unembeddable image costs one skipped item and a log line
rather than the whole run. Set `region_rows` to key the check on the
snapshot's patch-slot embedder, which is what the region-row matrix
reads.

Convert to torch with `torch.from_numpy(matrix)` for a zero-copy
view. The matrix is contiguous, so it's safe to slice without
copying.

```python
from vtscore.embedding import get_embedding_matrix, invalidate_embedding_matrix
from vtscore.state import get_active_context

ctx = get_active_context()
ids, matrix = get_embedding_matrix(ctx)   # (N, D) float32
# ... mutate ctx.medias somehow ...
invalidate_embedding_matrix(ctx)          # next call rebuilds
```

Callers that add / remove / replace entries in `ctx.medias` don't need
to invalidate: `ctx.medias` is a `MediasDict` that bumps
`media_revision` on every structural mutation, so the next access sees
the new revision and rebuilds. The one case that *does* need an explicit
call is an **in-place** rewrite of an existing media's vector
(`ctx.medias[cid]["embeddings"][name] = vec` during re-embed / clip): a
dict subclass can't observe a mutation to a value's internals, so those
stages call `invalidate_embedding_matrix(ctx)` (which bumps the counter)
afterwards (logical-bug-audit root-cause Pattern #4).

`get_embedding_matrix_for_snap(snap)` (`vtscore/embedding/matrix.py`)
is for callers that hold a media-dict snapshot (typically from
`snapshot_medias()`): when `snap`'s key set matches the active
dataset's medias, the cached matrix is reused; otherwise a fresh
non-cached matrix is built. Cross-dataset Find takes the fresh path.

### Cache lifetime and persistence

- **In-memory cache:** lives on the `DatasetContext` instance, in
  `_emb_matrix` / `_emb_matrix_ids` / `_emb_matrix_revision`, for as
  long as `ctx` is alive. See [state](state.md) for `DatasetContext`.
- **On-disk mmap sidecar:** for a dataset that is *registered against a
  saved `.pkl`*, a freshly-built primary matrix is also written next to
  that pickle as `<stem>.embmat.npy` + `<stem>.embids.npy`, and a later
  cold load `mmap`s it back instead of restacking the matrix from
  per-item embeddings. Details below.
- **Thread safety:** every read/write of the in-memory cache goes
  through `vtscore.state.core._state_lock`. Concurrent callers serialise
  on the same lock, so the rebuild happens exactly once when the key
  set changes. The sidecar read/write happens *outside* the lock.

#### The `.npy` sidecar, and why it isn't a "persisted vector"

CLAUDE.md forbids persisting embeddings, with an exception for dataset
pickle files. The sidecar sits inside that exception rather than beside
it: it is a **derived cache of data the pickle already stores durably**,
regenerable from `ctx.medias` at any time, deterministic, and swept
alongside the pkl by `registry.unregister_dataset` (both files share the
pkl's stem, so the stem-glob delete catches them). Nothing reaches disk
that was not already on disk. See S1 in `docs/plans/scalability.md`.

The guard rails that make it safe to trust:

- **Only for registered datasets.** `_registered_pkl_path` returns
  `None` for in-memory datasets (tests, ephemeral browse contexts,
  positives-map previews), and those get no sidecar at all.
- **Validated on read.** `_try_load_matrix_sidecar` adopts the file only
  when both halves exist, the persisted id list matches the current
  sorted ids exactly, and the width matches a probe embedding.
  Otherwise it returns `None` and the caller restacks.
- **Crash-safe.** Each half is written to a temp file and atomically
  renamed, matrix first. A crash between the two renames leaves a
  mismatched pair, which the read-side checks reject.
- **Latched off after an in-place rewrite.** An id set cannot detect a
  same-id vector rewrite (re-embed, re-clip), so the first
  `invalidate_embedding_matrix(ctx)` sets `ctx._emb_sidecar_disabled`
  permanently for that context's lifetime. A fresh load gets a fresh
  context, and so a fresh unset latch.
- **Best-effort.** A read-only filesystem, a full disk or a concurrent
  writer is logged and swallowed. The sidecar is an optimisation, never
  a dependency.

Persisting *per-item* vectors anywhere outside the dataset pickle - to
settings, to a detector JSON, to a new cache file of your own - is still
a project-invariant violation.

---

## Per-media vector access

`vtscore/embedding/media_vectors.py` is how you read and write the
`media["embeddings"]` dict. A media carries a `{embedder_name: vector}`
map, not a single `embedding` field, because one media can hold vectors
from several bound embedders at once.

```python
def primary_embedder_name(media) -> str | None: ...
def media_embedder_names(media) -> list[str]: ...           # primary first
def media_embedding(media, embedder_name=None) -> Any: ...  # primary when unset
def init_embeddings(embedder_name, vec) -> dict: ...        # for media-dict literals
def set_media_embedding(media, embedder_name, vec) -> None: ...
def ensure_embeddings_dict(media) -> None: ...
```

The primary is `media["embedder"]` when recorded, else the first key of
the dict. `media_embedder_names` puts it first so role derivation and
primary-vector reads agree on which embedder leads.

`UNKNOWN_EMBEDDER_KEY` (the empty string) is the sentinel for a
pre-computed, externally-supplied vector with no embedder name - an NPZ
import, `content_vectors`, `custom_metadata_map`. Such a vector is
stored rather than dropped: a nameless vector binds no role, and the
embed stage's *named* missing-vector check still treats the media as
lacking any named embedder's vector, while `media_embedding` still
resolves the sole entry as primary.

Use `init_embeddings` in media-dict literals at creation time, and
`set_media_embedding` afterwards. Passing `vec=None` yields `{}`, the
deferred-embed placeholder the embed stage fills in later.

---

## Normalisation

`vtscore/embedding/normalize.py` is one function:

```python
def l2_normalize(vec: object) -> np.ndarray: ...
```

VTSearch is direction-only nearly everywhere - every similarity
comparison treats embeddings as points on the unit sphere. Rather than
re-normalising at each comparison, **the unit vector is the stored
form**. This module is the single place that performs it, applied at
every point a vector enters the system: fresh embeds
(`MediaEmbedder.embed_media` / `embed_media_bulk`, so every subclass is
covered), text queries (`MediaEmbedder.embed_text`), pickle-loaded
stored vectors, and re-ingested-from-origin vectors.

Because the invariant holds at the store, downstream consumers - the
cached matrix, the coverage atlas's k-means, MLP training, region
similarity, the VTSBrowse projection - all consume unit vectors without
re-normalising.

Two properties make applying it at several chokepoints safe: it is
**idempotent** (re-normalising a unit vector returns it unchanged up to
float32 rounding, so an embedder that already normalises costs nothing),
and a **zero or non-finite-norm vector is returned unchanged** rather
than divided, so it can't mint `inf` / `nan` rows that would poison every
downstream consumer.

---

## Role-typed embedder binding

`vtscore/embedding/binding.py` implements the v3 "three-slot" model: a
dataset binds up to one **text**-capable embedder, up to one
**patch**-capable embedder, and up to one **structural** (geometric
verification) embedder. All three can coexist on one dataset, and each
drives a different path - text sort against the text slot; region
similarity, region voting and the detector head against the patch slot;
instance retrieval and geometric re-rank against the structural slot.

The three immutable detector *embedder types* partition the registry
(no embedder advertises more than one capability flag), classified in
precedence order **structural ▸ patch_semantic ▸ semantic** so that
typing is a total function matching score-routing precedence:

| Constant | Label |
|----------|-------|
| `EMBEDDER_TYPE_STRUCTURAL` | Structural |
| `EMBEDDER_TYPE_PATCH_SEMANTIC` | Patch Semantic |
| `EMBEDDER_TYPE_SEMANTIC` | Semantic |

| Function | Description |
|----------|-------------|
| `embedder_type(name)` | Classify one embedder into its type |
| `derive_binding(name)` / `derive_binding_from_names(names)` | Map embedder name(s) onto the `(text, patch, structural)` triple |
| `validate_binding(...)` | Reject a slot pointing at an embedder lacking that role's capability |
| `embedder_of_type(names, target_type)` | The first name of a given type, or `None` |
| `dataset_supplied_types(names)` / `detector_dataset_compatible(det_type, names)` | Which types a dataset supplies; whether a detector can run on it |
| `score_marker_embedder(media)` / `..._for_snap(snap)` / `keying_embedder_for_snap(det_ctx, snap)` | Which embedder the score / cache key routes through |

`derive_binding` is how a **pre-v3 dataset** - one `embedder` name and
nothing else - resolves into the three-slot model on load, by
role-typing that name against the embedder's declared capabilities. An
explicit binding (`bind_embedders`) overrides the derivation for
genuinely multi-embedder datasets.

Neither function holds state; the binding lives on `DatasetContext` as
embedder **names**, never vectors. Capability lookups go through the
embedder registry, so an unknown name resolves to "no capabilities" and
is ineligible for every slot.

A detector locks one type at create time and is compatible with any
dataset binding an embedder of that same type - the labels (and, for
`patch_semantic`, the region boxes) re-derive against whichever concrete
embedder that dataset supplies.

---

## Concurrency defaults

```python
def default_concurrent_downloads() -> int: ...
def default_concurrent_embeddings() -> int: ...
```

(`vtscore/embedding/loader.py`.) Heuristics for the dataset-
load `ConcurrencyGate`s in `vtscore.datasets.load_pipeline`:

- **Downloads** - defaults to `min(4, os.cpu_count())`. Bandwidth and
  disk-bound; a handful of concurrent downloads saturates a home
  connection without thrashing FDs.
- **Embeddings** - defaults to `1` on CPU-only boxes; on multi-GPU
  rigs, `min(2, num_cuda_devices)` so two datasets can embed in
  parallel without overcommitting a single device's VRAM.

These are *defaults*. The actual limits read through
`vtscore.config.CoreConfig` so they can be overridden per-deployment.

---

## Gotchas

- **No persisted vectors.** Every result of `embed_*_file` and
  `embed_text_query` is in-memory only. The text-query LRU caps at
  32 entries. Persisting vectors to disk, settings, or a detector
  JSON is a project-invariant violation - see CLAUDE.md.
- **The default-embedder choice is registry-order-dependent.**
  `embedders_for_type(t)[0]` sorts `is_default=True` first. If two
  embedders both claim default, the second-registered wins by
  insertion order - fix by setting `is_default=False` on the loser.
- **Reach the backbone through `loaded_backbone()`, not attributes.**
  The three getters (`get_clap_model`, etc.) used to reach into each
  subclass's private `_get_model_and_processor()` via
  `typing.cast(Any, emb)`, which broke silently if an embedder was
  reimplemented. They now go through the ABC method, so a custom
  embedder either works via the default or overrides one documented
  hook. New code should still prefer `embed_media` / `embed_text`.
- **Matrix invalidation is implicit, but explicit is faster.** If you
  mutate `ctx.medias`, calling `invalidate_embedding_matrix(ctx)`
  costs nothing and avoids the next `get_embedding_matrix` doing a
  full key-set comparison before rebuilding.
- **`initialize_models()` does *not* load embedder weights.** It only
  sets the cache dir and configures torch threads. Use
  `preload_predicted_embedders()` if you want the predicted set
  warmed eagerly at startup.
- **Cross-context matrix reuse.** `get_embedding_matrix_for_snap`
  reads `get_active_context()`. If your call site has a specific
  `ctx` in hand and you don't want the active-context behaviour,
  call `get_embedding_matrix(ctx)` directly with a snapshot whose
  key set matches `ctx.medias`.

---

## Cross-references

- [media](media.md) - the embedder ABC, registry, and per-type
  inventory.
- [converters](converters.md) - when you want to run an image
  embedder over audio (via spectrogram), an audio embedder over
  speech (via ASR), etc.
- [state](state.md) - `DatasetContext` is where the matrix cache
  lives; lock semantics come from `_state_lock`.
- `vtscore.training.region_similarity` - patch-region cosine
  scoring built on top of the matrix.
