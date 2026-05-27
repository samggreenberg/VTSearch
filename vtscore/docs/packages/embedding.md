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

`vtscore/embedding/helpers.py:28`–`49`. Four one-liners that pick the
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

(`vtscore/embedding/helpers.py:86`.) Embed a free-form text query
into the vector space of a media type, so it can be cosine-scored
against media embeddings. When `embedder_name` is set, that specific
embedder is used; otherwise it falls back to the default for the
given `media_type`. `enrich=True` runs `embed_text_enriched`, which
averages the query across the embedder's `description_wrappers`
(e.g. `"a photo of {text}"`, `"a video of {text}"`).

Results are cached in a small process-wide LRU
(`_query_cache` at `vtscore/embedding/helpers.py:60`):

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

(`vtscore/embedding/loader.py:22`.) Resolves
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
def initialize_models() -> None: ...
```

(`vtscore/embedding/loader.py:76`.) Sets up the runtime environment:

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

(`vtscore/embedding/loader.py:131`–`287`.)

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

(`vtscore/embedding/loader.py:298`–`320`.) Each calls
`get_embedder(name).load_models()` and returns the private
`_get_model_and_processor()` / `_get_model()` accessor. Prefer the
public `embed_media` / `embed_text` API for new code - these helpers
exist because parts of the legacy app reach into the backbone.

---

## Cached embedding matrix

For any sort or learned-sort over a loaded dataset, you need every
media's embedding as a contiguous `(N, D) float32` array.
`vtscore/embedding/matrix.py` keeps that array on `DatasetContext`
and rebuilds it only when the set of loaded media IDs changes.

```python
def get_embedding_matrix(ctx: DatasetContext) -> tuple[list[int], np.ndarray]: ...
def invalidate_embedding_matrix(ctx: DatasetContext) -> None: ...
def get_embedding_matrix_for_snap(snap: dict) -> tuple[list[int], np.ndarray]: ...
```

(`vtscore/embedding/matrix.py:29`–`104`.)

### How the cache works

`DatasetContext._emb_matrix` and `DatasetContext._emb_matrix_ids` are
the cache. `get_embedding_matrix(ctx)` (`vtscore/embedding/matrix.py:29`):

1. Acquires `vtscore.state.core._state_lock`.
2. Computes `sorted_ids = sorted(ctx.medias.keys())`.
3. If `sorted_ids == ctx._emb_matrix_ids`, returns the cached
   `(ids, matrix)` directly - no rebuild.
4. Otherwise, allocates an `(N, D) float32` array, fills it row by
   row from `ctx.medias[cid]["embedding"]`, and stores it on the
   context.

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

Callers that mutate `ctx.medias` directly don't strictly need to
invalidate - the next access detects the new key set and rebuilds -
but explicit invalidation is cheaper than the implicit "did the key
set change?" walk and removes the ambiguity.

`get_embedding_matrix_for_snap(snap)` (`vtscore/embedding/matrix.py:70`)
is for callers that hold a media-dict snapshot (typically from
`snapshot_medias()`): when `snap`'s key set matches the active
dataset's medias, the cached matrix is reused; otherwise a fresh
non-cached matrix is built. Cross-dataset Find takes the fresh path.

### Cache lifetime and persistence

- **Cache lifetime:** as long as `ctx` is alive. The matrix is **not**
  written to disk and **never** persists across process restarts.
- **Where it lives:** on the `DatasetContext` instance, in
  `_emb_matrix` / `_emb_matrix_ids`. This is by design - the
  matrix is keyed to one dataset's contents and shouldn't outlive
  them. See [state](state.md) for `DatasetContext`.
- **Thread safety:** every read/write goes through
  `vtscore.state.core._state_lock`. Concurrent callers serialise on
  the same lock, so the rebuild happens exactly once when the key
  set changes.

---

## Concurrency defaults

```python
def default_concurrent_downloads() -> int: ...
def default_concurrent_embeddings() -> int: ...
```

(`vtscore/embedding/loader.py:51`–`73`.) Heuristics for the dataset-
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
- **`_get_model_and_processor()` and `_get_model()` are private.** The
  three backbone accessors (`get_clap_model`, etc.) reach in via
  `typing.cast(Any, emb)` because those methods are subclass-private
  and not on the ABC. New code should prefer the public surface.
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
