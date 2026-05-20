# Core Concepts

Eight ideas show up in every `vtscore` package. Read this once and the
package docs read smoothly.

1. [Media](#1-media) — what the library actually moves around.
2. [Embedding](#2-embedding) — the fixed-dimensional vector for a media item.
3. [Origin](#3-origin) — provenance: how to find a media item again.
4. [LabelSet / LabeledElement](#4-labelset--labeledelement) — voted labels with provenance.
5. [Embedder](#5-embedder) — the model that turns a file into an embedding.
6. [Detector](#6-detector) — the MLP + threshold that scores embeddings.
7. [Context](#7-context) — per-dataset / per-detector mutable state holder.
8. [Plugin](#8-plugin) — auto-discovered, swappable component of any family.

---

## 1. Media

A **media item** is a Python dict — not a class, not a dataclass — keyed by
sequential integer IDs starting at 1. The dict shape is the library's
internal interchange format; every importer, embedder, exporter, and
detector reads and writes media dicts.

```python
medias[42] = {
    "id": 42,                              # int, matches dict key
    "type": "audio",                       # MediaType.type_id
    "embedder": "audio_clap",              # MediaEmbedder.name
    "file_size": 318456,                   # bytes
    "md5": "5d41402abc4b2a76b9719d911017c592",  # content hash
    "embedding": np.ndarray,               # shape (D,), float32
    "filename": "barks/poodle.wav",        # relative path from origin root
    "category": "custom",                  # demo-dataset slot
    "origin": {"importer": "server_folder",
               "params": {"path": "/data/audio"}},
    "origin_name": "barks/poodle.wav",
    "media_bytes": None,                   # populated lazily for embedding
    "media_string": None,                  # text-only media
    "media_path": "/data/audio/barks/poodle.wav",
    "duration": 3.142,                     # type-specific (sec for audio/video)
}
```

The shape is defined by `vtscore/datasets/loader_folder.py` (the
`_build_base_media_dict` helper) and consistent across every loader. Plugin
authors who add a new media type extend the shape with their own keys but
must preserve the base contract.

The dict-of-dicts representation is deliberate:

- Cheap to serialise (it's already JSON-shaped except for the embedding).
- Easy to project — `[m["embedding"] for m in medias.values()]` builds the
  training matrix in one line.
- No risk of accidental method calls or stale class instances after a
  pickle round-trip.

## 2. Embedding

An **embedding** is a contiguous `np.ndarray` of shape `(D,)` and dtype
`float32`. Every media item carries one. The dimensionality `D` depends on
the embedder:

| Embedder | D | Notes |
|----------|---|-------|
| LAION-CLAP (audio) | 512 | Default for audio. |
| CLAP Music | 512 | Alternative for music collections. |
| SigLIP (image) | 768 | Default for images. |
| X-CLIP (video) | 768 | Default for video. |
| E5-base-v2 (text) | 768 | Default for text. |
| BGE-base-en-v1.5 (text) | 768 | Alternative text embedder. |

The library also maintains a **cached embedding matrix** per
`DatasetContext`: a contiguous `(N, D)` float32 array built lazily by
`vtscore.embedding.matrix.get_embedding_matrix(ctx)` and reused across
cosine sort, MLP scoring, and diversity-tree construction. The cache lives
in process memory only — it's never persisted — and is invalidated when
the underlying `medias` dict changes.

## 3. Origin

An **`Origin`** is a `(importer, params)` tuple that uniquely identifies the
source of a media item:

```python
from vtscore.datasets.origin import Origin

origin = Origin(
    importer="server_folder",
    params={"path": "/data/audio", "media_type": "audio"},
)
origin.to_dict()
# {"importer": "server_folder", "params": {"path": "/data/audio", "media_type": "audio"}}
```

Origins are **the canonical persisted form** for media references in
vtscore. A detector JSON file stores origins (inside `LabeledElement`s);
when the detector loads, the library resolves each origin back to a file,
re-embeds it, and rebuilds the training matrix. No vector or weight is
ever stored on disk outside of dataset pickles.

The reverse-resolve job lives in `vtscore/detectors/resolver.py`:

```python
from vtscore.detectors.resolver import resolve_file_from_origin

path = resolve_file_from_origin(origin)  # → Path("/data/audio/...")
```

Different importers produce different `params` shapes. Plugin authors must
ensure their importer's origins can be re-resolved by the matching
`MediaSource` plugin (or by a fallback in the importer itself). See
[extending/dataset-importers.md](extending/dataset-importers.md) and
[extending/media-types.md](extending/media-types.md).

## 4. LabelSet / LabeledElement

A **`LabeledElement`** binds one origin to one label (`"good"` or
`"bad"`) plus optional metadata:

```python
from vtscore.datasets.labelset import LabeledElement

elem = LabeledElement(
    md5="5d41402abc4b2a76b9719d911017c592",
    label="good",
    origin_name="barks/poodle.wav",
    origin={"importer": "server_folder", "params": {"path": "/data/audio"}},
    region_box=None,                       # optional (x, y, w, h)
    metadata={"reviewer": "alice", "ts": 1700000000},  # free-form
)
```

A **`LabelSet`** is an ordered collection of `LabeledElement`s plus
optional detector-level metadata:

```python
from vtscore.datasets.labelset import LabelSet

labelset = LabelSet(
    labels=[elem1, elem2, …],
    metadata={"detector_name": "dog barks", "media_type": "audio"},
)
labelset.merge(other_labelset)             # union by md5
json_str = labelset.to_json()              # round-trip-safe
restored = LabelSet.from_json(json_str)
```

`LabelSet` round-trips through JSON for detector storage and through
plain dicts for the labelset-source / labelset-exporter plugin contracts.
`LabeledElement.metadata` is the recommended escape hatch for per-label
data that doesn't fit the core fields — it survives every serialisation
boundary.

## 5. Embedder

A **`MediaEmbedder`** is a plugin that turns a media file into an
embedding. Every concrete embedder subclasses `vtscore.media.MediaEmbedder`
and implements at least two methods:

```python
class MediaEmbedder:
    name: str
    display_name: str
    media_type: str

    def embed_files(self, paths: list[Path]) -> np.ndarray:
        """Return (N, D) float32 array, one row per file."""

    def embed_text(self, query: str) -> np.ndarray:
        """Return (D,) float32 — the text-query embedding in the same space."""

    def forward_patches(self, path: Path) -> np.ndarray:
        """Optional. Return (P, D) for patch-level scoring."""
```

Embedders are **lazy** — the model is downloaded and constructed on first
use, then cached process-wide. The cache lives in `CoreConfig.data_dir /
"models"` by default. Embedders that wrap multimodal models implement
`embed_text` so a text query can seed a sort or a detector in the same
vector space as the media; embedders that don't (e.g. an audio-only
embedder with no text encoder) raise `NotImplementedError` from
`embed_text`.

See [packages/media.md](packages/media.md) for the full embedder contract
and [extending/embedders.md](extending/embedders.md) to write your own.

## 6. Detector

A **detector** in `vtscore` is a small MLP classifier plus a calibrated
decision threshold, both derived from a `LabelSet`.

Architecture (`vtscore/training/mlp.py`):

```
input  (D,)
  → Linear(D, H)        # H auto-sized from training-set count
  → ReLU
  → Dropout(p)          # default 0.3
  → Linear(H, 1)
  → (no activation; train with BCEWithLogitsLoss)
```

Training (`vtscore.training.train_model`):

- Class-weighted BCE-with-logits — weight controlled by
  `inclusion_value ∈ [-10, +10]` (positive biases toward more recalled
  positives, negative biases toward fewer false positives).
- Local `torch.Generator` seeded with the caller-supplied seed
  (default 42) for deterministic weight init.
- `torch.random.fork_rng()` around `nn.Dropout` so concurrent training
  calls don't race on the global RNG.

Thresholding (`vtscore/training/thresholds.py`):

- `calculate_cross_calibration_threshold` runs k-fold cross-cal (k =
  `calibrate_count`) and returns the median optimal threshold.
- `calculate_safe_threshold` blends in a conservative fallback when the
  user has `safe_thresholds=True`.
- `calculate_gmm_threshold` is a 2-component GMM midpoint, used as the
  ultimate fallback when cross-cal can't converge.

The combined pipeline is `vtscore.detectors.training.train_and_threshold`:
one call gives you `(model, threshold)` from `(X_list, y_list)`. The
detector lifecycle (load labelset → re-derive embeddings → train) lives
in `vtscore.detectors.training.train_detector_from_origins`; the caller
must pass `embedder_name` so re-derivation uses the same embedder the
detector was trained with, not whatever the media type's current default
happens to be.

## 7. Context

A **`DatasetContext`** holds all mutable state belonging to one loaded
dataset:

- `medias: dict[int, dict]` — the media items.
- `diversity_tree` — hierarchical clustering for max-diversity sampling.
- `dataset_display_name: str | None` — human-readable name.
- `_emb_matrix_ids` / `_emb_matrix` — the cached `(N, D)` matrix and the
  sorted media-id list it corresponds to.

A **`DetectorContext`** holds all mutable state belonging to one loaded
detector:

- Vote state: `good_votes`, `bad_votes`, `label_history`, `vote_click_times`,
  `vote_region_boxes`.
- Cached training artefacts: `training_medias`, `label_embeddings`,
  `model`, `threshold`, `calibration_cache`.
- Cached labelset: `cached_labelset`, `cached_labelset_mtime`,
  `cached_labelset_media_type` (so a re-train doesn't re-parse JSON).
- Cross-dataset counts: `labelset_good_count`, `labelset_bad_count` —
  enable "Sort by Learned" even when the active dataset isn't the one
  the detector was trained on.
- Sync state: `labelset_source` (dict of `{source_name, field_values}`).

Both are `RLock`-protected by the single module-level lock in
`vtscore/state/core.py`. Both use `__slots__` for memory efficiency
(typical apps load dozens to hundreds of context objects).

Crucially, **neither is a global.** Library code never reaches for "the
active context" implicitly; every call goes through the resolution chain
described in [architecture.md](architecture.md#resolution-chain-for-active-context).

## 8. Plugin

A **plugin** is any swappable component of one of vtscore's eleven plugin
families: dataset importers, results exporters, label importers, labelset
sources, media types, embedders, clippers, media converters, media
sources, processor importers, and (app-side) settings importers /
exporters / sources.

Every plugin is:

- A subclass of the family's base ABC (e.g. `DatasetImporter`).
- Declared at module level via a **sentinel** attribute (`IMPORTER`,
  `EXPORTER`, `EMBEDDER`, `CLIPPERS`, …).
- Auto-discovered by the family's `PluginRegistry` at construction time.
- Optionally exposed to third parties via an `importlib.metadata` entry
  point under `vtscore.<family>`.

The full mechanics are documented in [packages/plugins.md](packages/plugins.md)
and the per-family authoring guides under [extending/](extending/).
