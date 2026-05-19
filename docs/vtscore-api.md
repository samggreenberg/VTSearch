# `vtscore` Public API Sketch

> **Status:** Phase 0 deliverable for [`docs/plans/extract-library.md`](plans/extract-library.md).
> This document is the **contract the refactor must preserve** — the set of symbols an
> external `vtscore` consumer is expected to import, and the one-line semantics each one
> guarantees. Exact signatures, types, and parameter names will be pinned down during
> Phases 1–4 as call sites are converted from globals/Flask to explicit arguments;
> what this sketch fixes is the **shape**: what exists, where it lives, and what it does.

## Methodology

The inventory was assembled by walking the current VTSearch tree under `vtsearch/` and
identifying public symbols (no leading underscore) in each candidate library subpackage,
cross-referenced against each package's `__init__.py` re-exports. The mapping from
current import path → future library path is one-to-one:

| Today                        | After Phase 8                |
|------------------------------|------------------------------|
| `vtsearch.datasets.X`        | `vtscore.datasets.X`         |
| `vtsearch.media.X`           | `vtscore.media.X`            |
| `vtsearch.converters.X`      | `vtscore.converters.X`       |
| `vtsearch.labels.X`          | `vtscore.labels.X`           |
| `vtsearch.embedding.X`       | `vtscore.embedding.X`        |
| `vtsearch.training.X`        | `vtscore.training.X`         |
| `vtsearch.detectors.X`       | `vtscore.detectors.X`        |
| `vtsearch.eval.X`            | `vtscore.eval.X`             |
| `vtsearch.state.X`           | `vtscore.state.X` (contexts only — proxies stay app-side) |
| `vtsearch.plugins.X`         | `vtscore.plugins.X`          |
| `vtsearch.sync.X`            | `vtscore.sync.X`             |
| `vtsearch.concurrency.X`     | `vtscore.concurrency.X`      |
| `vtsearch.security.X`        | `vtscore.security.X`         |
| `vtsearch.utils.X`           | `vtscore.utils.X`            |
| `vtsearch.exporters.X`       | `vtscore.exporters.X`        |
| `vtsearch.cli` etc.          | `vtscore.cli` etc.           |
| `vtsearch.config`            | `vtscore.config`             |

The plan's "final shape" in extract-library.md was written before the codebase split
`models/` into `detectors/`/`embedding/`/`training/`, and before `concurrency/` and
`security/` graduated into their own packages. The mapping above is canonical; the
plan's tree should be updated to match it during Phase 1.

Symbols flagged **[SEAM]** import `flask` or `vtsearch.settings` today and must be
detangled before they can ship in `vtscore`. They are listed here because they belong
in the library by intent — what changes is the implementation, not the public name.

---

## `vtscore.config`

Constants and environment-driven knobs that library code reads at import time. The
mutable, per-user pref equivalents stay in the app's `vtsearch.settings` module.

```python
DATA_DIR: Path
"""Base directory for runtime artefacts. Honours $VTSEARCH_DATA_DIR; defaults to ./data."""

EMBEDDINGS_DIR: Path
"""Embedding cache root (DATA_DIR / 'embeddings')."""

MODELS_CACHE_DIR: Path
"""HuggingFace / torch model cache root (DATA_DIR / 'models')."""

TORCH_THREADS: int
"""Native thread count cap. Honours $VTSEARCH_TORCH_THREADS; defaults to 1."""

DEVICE: str
"""Compute device hint: 'auto' | 'cuda' | 'cpu' | 'mps'. Honours $VTSEARCH_DEVICE."""

def resolve_device() -> str:
    """Resolve DEVICE='auto' to the concrete torch device string for this host."""

SERVER_ROOTS: tuple[Path, ...]
"""Allowed filesystem roots for server-path importers. Honours $VTSEARCH_SERVER_ROOTS."""

MAX_UPLOAD_MB: int
"""HTTP request size cap in MB; 0 = unlimited. Honours $VTSEARCH_MAX_UPLOAD_MB."""

TRAIN_EPOCHS: int
"""Upper bound on MLP training epochs. Honours $VTSEARCH_TRAIN_EPOCHS; default 200."""

TRAIN_PATIENCE: int
"""Early-stop patience in epochs. Honours $VTSEARCH_TRAIN_PATIENCE; default 10."""

DEFAULT_CALIBRATE_COUNT: int
"""Default count of held-out items used for threshold calibration."""

MLP_HIDDEN_MIN: int
MLP_HIDDEN_MAX: int
MLP_DROPOUT: float
"""MLP architecture bounds used by the auto-sizing heuristic in vtscore.training.mlp."""

CLAP_MODEL_ID: str           # HuggingFace ID for LAION CLAP audio embedder
CLAP_MUSIC_MODEL_ID: str     # CLAP variant trained on music
CLAP_SAMPLE_RATE: int        # 48000
XCLIP_MODEL_ID: str          # X-CLIP video embedder
E5_MODEL_ID: str             # Multilingual E5 text embedder
BGE_MODEL_ID: str            # BGE text embedder variant
SIGLIP_MODEL_ID: str
SIGLIP2_MODEL_ID: str
CLIP_MODEL_ID: str
DINOV2_MODEL_ID: str
DINOV3_MODEL_ID: str
EUPE_MODEL_ID: str
LANGUAGEBIND_VIDEO_MODEL_ID: str
"""Embedder model identifiers. Each lazy-loaded by vtscore.embedding.loader.get_*."""
```

---

## `vtscore.datasets`

Dataset domain objects, on-disk loaders, importer registry, and the metadata helpers
each demo dataset needs. The current `vtsearch.datasets.__init__` already re-exports
the full surface; this is faithful to that contract.

### Domain objects

```python
class Origin:
    """Provenance of one media element: the importer that produced it plus the params
    needed to reproduce its file. Pickled into datasets and detector labelsets."""

@dataclass
class LabeledElement:
    """A single (md5, label, origin, optional region box, optional metadata) tuple.
    Round-trips through JSON and pickle. metadata is free-form per-label data."""

class LabelSet:
    """Ordered collection of LabeledElement plus optional detector metadata.
    Supports merge, JSON ser/de, and is what labelset sources/exporters consume."""
```

### Importer registry

```python
class DatasetImporter(PluginBase):
    """Abstract base for a dataset importer. Subclasses declare `fields` (a list of
    PluginField) and implement `run(field_values, progress_callback) -> iterable of
    (Media, embedding)`. Multi-media importers set `multi_media = True` and iterate
    `self.effective_source_specs(field_values)` to fan out across source types."""

ImporterField = PluginField
"""Alias kept for clarity at import sites; identical to vtscore.plugins.PluginField."""

def get_importer(name: str) -> DatasetImporter:
    """Look up a registered importer by its `name` attribute; raises KeyError."""

def list_importers() -> list[DatasetImporter]:
    """Return all registered importers, including third-party entry-point plugins,
    minus those with hidden_from_picker=True (these are scaffolds, not selectable)."""
```

### Loaders / exporters

```python
def load_dataset_from_folder(
    folder: Path | str,
    media_type: str,
    *,
    progress: ProgressCallback | None = None,
    **opts,
) -> tuple[list[Media], list[Embedding]]:
    """Walk `folder`, embed each file with the default embedder for `media_type`,
    return (medias, embeddings). Honours filetype filters declared by the MediaType."""

def load_dataset_from_pickle(path: Path | str) -> tuple[list[Media], list[Embedding]]:
    """Restore a (media, embedding) snapshot previously written by export_dataset_to_file.
    Uses vtscore.security.safe_pickle_load — no arbitrary code execution risk."""

def load_demo_dataset(dataset_id: str, *, progress=None) -> tuple[list[Media], list[Embedding]]:
    """Download (if needed) and load one of the built-in demo datasets keyed by
    `dataset_id` (e.g. 'esc50', 'cifar10', 'gtzan'). See DEMO_DATASETS for the keys."""

def export_dataset_to_file(
    medias: list[Media],
    embeddings: list[Embedding],
    path: Path | str,
) -> None:
    """Pickle (medias, embeddings) to disk. The only sanctioned way to persist
    embeddings (see CLAUDE.md "No Persisted Vectors")."""
```

### Demo metadata loaders

```python
def load_esc50_metadata(root: Path) -> list[dict]: ...
def load_cifar10_batch(path: Path) -> tuple[list[bytes], list[str]]: ...
def load_video_metadata_from_folders(roots: list[Path]) -> list[dict]: ...
def load_image_metadata_from_folders(roots: list[Path]) -> list[dict]: ...
def load_paragraph_metadata_from_folders(roots: list[Path]) -> list[dict]: ...
def load_places365_metadata(root: Path) -> list[dict]: ...
"""Per-dataset metadata helpers. Each returns the rows the corresponding
demo dataset's importer needs to materialise Media objects."""
```

### Downloaders

```python
def download_file_with_progress(
    url: str,
    dest: Path,
    *,
    progress: ProgressCallback | None = None,
) -> Path:
    """HTTP GET with chunked streaming and progress events. Used by every download_*."""

def download_esc50(dest_dir: Path, *, progress=None) -> Path: ...
def download_cifar10(dest_dir: Path, *, progress=None) -> Path: ...
def download_ucf101_subset(dest_dir: Path, *, progress=None) -> Path: ...
def download_20newsgroups(dest_dir: Path, *, progress=None) -> Path: ...
"""One-shot demo-dataset fetchers. Each is idempotent: returns immediately if the
expected target already exists."""
```

### Split / utility

```python
def split_dataset(
    medias: Sequence[Media],
    *,
    test_fraction: float,
    seed: int,
) -> tuple[list[Media], list[Media]]:
    """Stratified shuffle on origin keys; reproducible given the same seed."""

DEMO_DATASETS: dict[str, DemoDataset]
"""Flat mapping {dataset_id -> DemoDataset metadata}. Populated lazily from every
registered MediaType.demo_datasets at first access."""
```

### Seams to cut [SEAM]

```python
# vtscore.datasets.load_pipeline
#   currently reads vtsearch.settings.get_max_concurrent_dataset_downloads/
#   get_max_concurrent_dataset_embeddings via the module-level ConcurrencyGate setup.
#   Phase 2: these limits move onto CoreConfig and are passed into the pipeline.

# vtscore.datasets.registry
#   currently reads vtsearch.settings.get_saved_datasets_dir to locate dataset pkls.
#   Phase 2: takes a Path argument; the app builds it from settings.
```

---

## `vtscore.media`

The plugin framework for everything media-type-specific: per-format `MediaType` plugins,
their embedders, clippers, and processor ABCs. No Flask or settings imports today.

### Core ABCs

```python
class MediaType(PluginBase):
    """One per media format (audio, image, text, video, document). Bundles
    file-extension filters, demo datasets, MIME serving, and the loader that turns
    a path into a Media object."""

class MediaEmbedder(PluginBase):
    """Embed media into a fixed-D vector space. Subclasses implement
    `embed_files(paths) -> np.ndarray` and `embed_text(query) -> np.ndarray`.
    Patch-level embeddings via `forward_patches` are optional."""

class MediaClipper(PluginBase):
    """Split one media into sub-items of the same type (e.g. a long audio file into
    fixed-length windows). Subclasses implement `clip(media) -> list[Media]`."""

class Processor(PluginBase):
    """Base for autorun processors that run after a dataset loads."""

class Detector(Processor):
    """Boolean 'is this good?' processor. Returns a float score per media."""

class Localizer(Processor):
    """Returns a list of (box, confidence) per media."""

class Extractor(Processor):
    """Returns a list of free-form metadata records per media."""
```

### Support types

```python
@dataclass
class DemoDataset:
    """Metadata for one demo dataset: id, label, description, category slug list,
    optional category->slice bounds, optional per-category item count."""

@dataclass
class MediaResponse:
    """Framework-agnostic HTTP response wrapper (bytes/file, mimetype, optional
    download_name). The app converts this to a Flask Response in the route layer."""

ProgressCallback = Callable[[str, str, int, int], None]
"""(status, message, current, total) -> None. Threaded through every long-running
library op so the caller can report progress upstream."""
```

### MediaType registry

```python
def register(media_type: MediaType) -> None: ...
def get(type_id: str) -> MediaType: ...
def get_by_folder_name(name: str) -> MediaType: ...
def get_by_extension(ext: str) -> MediaType: ...
def all_types() -> list[MediaType]: ...
def all_folder_names() -> list[str]: ...
def all_type_ids() -> list[str]: ...
def all_types_dict() -> list[dict]: ...
def all_demo_datasets() -> dict[str, DemoDataset]: ...
def normalize_type_id(type_id: str) -> str: ...
def set_progress_callback(cb: ProgressCallback) -> None:
    """Global default progress callback for media-registry operations.
    A per-thread override via set_thread_progress_callback takes priority when set."""

def set_thread_progress_callback(cb: ProgressCallback | None) -> None:
    """Set a progress callback for the current thread only. Multi-threaded library
    consumers (e.g. parallel evaluation harnesses) should use this to avoid one
    thread clobbering another's callback. None clears the thread override and
    falls back to the global. Mirrors vtscore.concurrency.progress.set_thread_progress."""

def get_thread_progress_callback() -> ProgressCallback | None: ...
```

### Embedder registry

```python
def register_embedder(emb: MediaEmbedder) -> None: ...
def get_embedder(name: str) -> MediaEmbedder: ...
def embedders_for_type(type_id: str) -> list[MediaEmbedder]: ...
def all_embedders() -> list[MediaEmbedder]: ...
def all_embedders_dict() -> list[dict]: ...
```

### Clipper registry

```python
def register_clipper(c: MediaClipper) -> None: ...
def get_clipper(name: str) -> MediaClipper: ...
def clippers_for_type(type_id: str) -> list[MediaClipper]: ...
def all_clippers() -> list[MediaClipper]: ...
def all_clippers_dict() -> list[dict]: ...
```

### Per-media-type plugin contract

Each `vtscore.media.{audio,image,text,video,document}` subpackage exports:

- `MEDIA_TYPE: MediaType` — sentinel; auto-registered at import time.
- `CLIPPERS: list[MediaClipper]` — sentinel; auto-registered at import time.
- `EMBEDDER` sentinels in `embedder_*.py` modules — auto-registered.

---

## `vtscore.converters`

Cross-format converters (audio→spectrogram, video→keyframes, document→image, ASR, OCR).
Used by multi-media importers and the legacy `converters` field on per-importer specs.

```python
class MediaConverter(PluginBase):
    """Abstract base. Each instance declares `source_type` and `target_type` strings,
    optional `fields` for user-configurable params, and implements
    `convert(media, params) -> list[Media]`."""

class Audio2ImageMediaConverter(MediaConverter): ...   # mel-spectrogram
class Audio2TextMediaConverter(MediaConverter): ...    # Whisper ASR
class Document2ImageMediaConverter(MediaConverter): ...
class Document2TextMediaConverter(MediaConverter): ...
class Image2TextMediaConverter(MediaConverter): ...    # OCR
class Video2AudioMediaConverter(MediaConverter): ...
class Video2ImageMediaConverter(MediaConverter): ...   # keyframes

def list_converters() -> list[MediaConverter]: ...
def get_converter(name: str) -> MediaConverter: ...
def list_converters_for_target(target_type: str) -> list[MediaConverter]: ...
def list_converters_for_source(source_type: str) -> list[MediaConverter]: ...
```

Sentinel: each concrete converter module exports `CONVERTER: MediaConverter` for
auto-discovery (entry-point group `vtsearch.converters`, which becomes
`vtscore.converters` after the rename).

---

## `vtscore.labels`

Label-side plugins: importers that pull labels in from external systems and
sources that sync bidirectionally with them.

```python
class LabelImporter(PluginBase):
    """Pull labels from somewhere external. run(field_values) yields
    [{'md5': ..., 'label': 'good'|'bad'}, ...]."""

class LabelsetSource(SyncSource[LabelSet, LabelSet]):
    """Bidirectional sync target for a detector's labelset. Implements
    `load(field_values) -> LabelSet` and `save(labelset, field_values) -> None`,
    with optional `load_full()` for richer detector metadata."""

LabelImporterField = PluginField
LabelsetSourceField = PluginField

def get_label_importer(name: str) -> LabelImporter: ...
def list_label_importers() -> list[LabelImporter]: ...
def get_labelset_source(name: str) -> LabelsetSource: ...
def list_labelset_sources() -> list[LabelsetSource]: ...
```

### Seams to cut [SEAM]

```python
# vtscore.labels.sync
#   sync_to_labelset_source / sync_from_labelset_source today reach into the active
#   DetectorContext via thread/Flask globals. Phase 3: they take the DetectorContext
#   explicitly. Public names stay; signatures gain an explicit `ctx` param.
def sync_to_labelset_source(ctx: DetectorContext) -> None: ...
def sync_from_labelset_source(ctx: DetectorContext) -> LabelSet: ...
```

Sentinels: `LABEL_IMPORTER` (label_importers) and `LABELSET_SOURCE` (labelset_sources).

---

## `vtscore.embedding`

Thin façade over the MediaEmbedder registry plus runtime-management helpers (device
selection, lazy model loading, the cached `(N, D)` matrix used by sorting).

```python
def embed_audio_file(path: Path | str) -> np.ndarray: ...
def embed_image_file(path: Path | str) -> np.ndarray: ...
def embed_video_file(path: Path | str) -> np.ndarray: ...
def embed_paragraph_file(path: Path | str) -> np.ndarray: ...
def embed_text_query(query: str, media_type: str) -> np.ndarray:
    """Embed a text query into the vector space of the given media_type.
    LRU-cached per (query, media_type). clear_text_query_cache() drops the cache."""

def clear_text_query_cache() -> None: ...

def get_torch_device() -> torch.device:
    """Resolve config.DEVICE to a concrete torch.device."""

def initialize_models() -> None:
    """Set up the torch runtime: cache dir, thread limits, optional CUDA visibility."""

def predict_embedders_to_preload(
    datasets: Iterable[DatasetRegistryEntry],
    detectors: Iterable[DetectorRegistryEntry],
) -> set[str]:
    """Return the set of embedder names worth warming for the given registries."""

def preload_predicted_embedders(names: set[str]) -> None: ...
def smart_preload_in_background(names: set[str]) -> None: ...

def get_clap_model(): ...
def get_xclip_model(): ...
def get_e5_model(): ...
"""Lazy accessors for the most common embedder backbones. Each loads on first
call, caches process-wide, and returns the underlying torch model + processor."""

def default_concurrent_downloads() -> int: ...
def default_concurrent_embeddings() -> int: ...
"""Defaults for the dataset-load ConcurrencyGates."""

def get_embedding_matrix(ctx: DatasetContext) -> np.ndarray:
    """Lazily build and cache an (N, D) contiguous float32 matrix on ctx for
    every loaded media. Used by sorting and learned-sort scoring."""

def invalidate_embedding_matrix(ctx: DatasetContext) -> None: ...
def get_embedding_matrix_for_snap(snap: dict, ctx: DatasetContext) -> np.ndarray:
    """Return the matrix for a snapshot of medias; reuses ctx's cache when applicable."""
```

---

## `vtscore.training`

Generic learned-sort primitives. Media-agnostic: every function operates on numpy
arrays of embeddings and labels. No Flask, no settings, no globals.

### MLP

```python
def build_model(input_dim: int, *, hidden_dim: int | None = None) -> torch.nn.Module:
    """Construct a small MLP classifier. hidden_dim=None auto-sizes from input_dim."""

def build_model_from_weights(weights: dict) -> torch.nn.Module:
    """Reconstruct a previously serialised MLP from its weights dict."""

def train_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = config.TRAIN_EPOCHS,
    patience: int = config.TRAIN_PATIENCE,
) -> torch.nn.Module:
    """Train in-place; returns the same model. Early-stops on patience."""
```

### Thresholds

```python
def calculate_gmm_threshold(scores: np.ndarray) -> float:
    """Fit a 2-component GMM to scores; return the midpoint between component means."""

def find_optimal_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Grid-search the score axis for the threshold maximising F1 on (scores, y)."""

def calculate_cross_calibration_threshold(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_folds: int = 5,
) -> float:
    """Stratified k-fold cross-validation; return the median optimal threshold."""

def cross_calibration_threshold_cached(key: Hashable, *fn_args, **fn_kwargs) -> float:
    """Memoised wrapper around calculate_cross_calibration_threshold. Used during
    interactive sorting to avoid recomputing on every vote."""

def calculate_safe_threshold(scores: np.ndarray, y: np.ndarray) -> float:
    """Blend the cross-cal threshold with a conservative fallback; used when the
    safe_thresholds user pref is on."""
```

### SVM (prototype)

```python
@dataclass
class SVMClassifier:
    """Trained SVM plus optional probability calibrator. Picklable."""

def train_svm(X: np.ndarray, y: np.ndarray, *, calibrate: bool = True) -> SVMClassifier:
    """Fit a linear SVM; if calibrate=True, also fit a Platt calibrator."""
```

### Region similarity (patch-level scoring)

```python
def score_against_query(
    query_embedding: np.ndarray,
    patch_embeddings: np.ndarray,
) -> tuple[float, tuple[int, int, int, int]]:
    """Return (max cosine score, best (x, y, w, h) region) for one media."""

def cosine_sort_with_boxes(
    medias_snap: dict,
    query_embedding: np.ndarray,
    matrix: np.ndarray,
) -> list[dict]:
    """Score every media in snap; return per-media hit dicts with region info."""
```

---

## `vtscore.detectors`

Detector lifecycle and the resolve → embed → train pipeline. The current
`vtsearch/detectors/workflow.py` is **Flask-aware** today and needs Phase 1 surgery
before it ships — but the public name and intent stay.

### Registry

```python
def list_detectors() -> list[DetectorRegistryEntry]: ...
def get_detector(detector_id: str) -> DetectorRegistryEntry: ...
def register_detector(entry: DetectorRegistryEntry) -> None: ...
def unregister_detector(detector_id: str) -> None: ...
def rename_detector(detector_id: str, new_name: str) -> None: ...
def update_detector(detector_id: str, **fields) -> None: ...
def find_by_name(name: str) -> DetectorRegistryEntry | None: ...
def add_loaded_detector_id(detector_id: str) -> None: ...
def remove_loaded_detector_id(detector_id: str) -> None: ...
def is_detector_loaded(detector_id: str) -> bool: ...
def get_loaded_detector_ids() -> set[str]: ...
def is_find_mode(detector_id: str) -> bool: ...
def set_find_mode(detector_id: str, enabled: bool) -> None: ...
```

### Training / scoring

```python
def validate_good_bad_split(good: int, bad: int) -> None:
    """Raise if either count is zero. Used as a precondition by every trainer."""

def train_and_threshold(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: CoreConfig | None = None,
) -> tuple[torch.nn.Module, float]:
    """Train an MLP and compute its calibrated threshold in one call."""

def serialize_weights(model: torch.nn.Module) -> dict:
    """Export MLP weights as a pickle-safe dict (state_dict + architecture metadata)."""

def train_and_score(ctx: DetectorContext, dataset: DatasetContext) -> dict[int, float]:
    """Vote-aware online trainer. Reads ctx.good_votes/bad_votes, trains, scores all
    medias in dataset, returns {media_id -> score}. Caches model on ctx."""

def collect_media_origins(medias: dict) -> list[tuple[int, Origin]]:
    """Extract (cid, Origin) pairs for every labelled media. Used when persisting
    votes to a detector JSON."""

def train_detector_from_origins(
    ctx: DetectorContext,
    *,
    config: CoreConfig | None = None,
) -> None:
    """Load-time retrainer: resolves every LabeledElement in ctx.labelset to a
    file → embedding, builds (X, y), trains, stores the model on ctx."""
```

### Origin resolution

```python
class SourceResolver(Protocol):
    """Callable: (origin: Origin) -> Path. Implemented by every MediaSource."""

class ImporterResolver(Protocol):
    """Callable: (origin: Origin) -> Path. Implemented by importers that can
    re-fetch by origin (e.g. demo importers, http_archive)."""

def register_source_resolver(name: str, resolver: SourceResolver) -> None: ...
def register_importer_resolver(name: str, resolver: ImporterResolver) -> None: ...

@dataclass
class ResolvedLabels:
    """Grouped (X, y, label_keys) ready to feed train_and_threshold."""

@contextmanager
def resolve_file_context(origin: Origin) -> Iterator[Path]:
    """Yield a Path to the origin's file; cleans up any tmp file on exit."""

def resolve_file_from_origin(origin: Origin) -> Path: ...
def embed_file(path: Path, media_type: str) -> np.ndarray: ...
def resolve_label_embeddings(labelset: LabelSet) -> ResolvedLabels:
    """Resolve every LabeledElement in `labelset` to its file, embed it, group by
    label, return the (X, y) ready for training."""
```

### Labelset materialisation

```python
def populate_label_embeddings(ctx: DetectorContext) -> None:
    """Rebuild ctx.label_embeddings from ctx.labelset (resolve + embed every entry).
    Idempotent; used after labels load or sync."""

def build_xy_from_labelset(ctx: DetectorContext) -> tuple[np.ndarray, np.ndarray]: ...
def train_from_labelset(ctx: DetectorContext, *, config=None) -> None: ...
def labelset_train_and_score(
    ctx: DetectorContext,
    dataset: DatasetContext,
) -> dict[int, float]:
    """Like train_and_score but uses ctx.labelset, not the live vote dicts."""

def update_cache_for_cid(ctx: DetectorContext, cid: int) -> None:
    """Refresh the label embedding for one media after a vote change."""
```

### Labelset element identity (for cross-dataset label restore)

```python
def stable_element_id(elem: LabeledElement) -> str: ...
def find_element_by_id(labelset: LabelSet, element_id: str) -> LabeledElement | None: ...
def resolve_current_dataset_cid(elem: LabeledElement, dataset: DatasetContext) -> int | None:
    """Find the cid in the live dataset that matches this labelset element, by origin."""

@contextmanager
def resolve_element_to_path(elem: LabeledElement) -> Iterator[Path]: ...
def build_element_view(elem: LabeledElement) -> dict: ...
def build_labels_detail(labelset: LabelSet) -> dict: ...
def apply_element_vote_in_data(labelset: LabelSet, element_id: str, label: str | None) -> None:
    """Apply (or, with label=None, remove) a vote on the labelset by stable id.
    Used by the detector-detail page when the dataset isn't loaded."""
```

### Labeling-session analyzer

```python
def clear_progress_cache() -> None: ...
def invalidate_progress_cache_from(media_id: int) -> None: ...
def inject_live_model(ctx: DetectorContext, model: torch.nn.Module) -> None: ...
def recreate_model_at_time(ctx: DetectorContext, t: int) -> torch.nn.Module: ...
def calculate_error_cost_over_time(ctx: DetectorContext) -> list[float]: ...
def calculate_prediction_stability_over_time(ctx: DetectorContext) -> list[float]: ...
def compute_labeling_status(ctx: DetectorContext) -> dict: ...
def calculate_diversity_level_over_time(ctx: DetectorContext) -> list[float]: ...
def analyze_labeling_progress(ctx: DetectorContext) -> dict:
    """Run all of the above and aggregate into one report."""
```

### Seams to cut [SEAM]

```python
# vtscore.detectors.workflow
#   apply_and_retrain() currently imports flask.g to discover the active contexts.
#   Phase 1: take DatasetContext and DetectorContext as explicit args.
def apply_and_retrain(dataset: DatasetContext, ctx: DetectorContext, *, config=None) -> None: ...

# vtscore.detectors.store, .label_sync, .label_restoration, .dataset_sync, .media_seeding
#   each imports vtsearch.settings (saved_datasets_dir, detectors_dir, etc.) directly.
#   Phase 2: these accept CoreConfig (or the specific Path) as an argument.
```

---

## `vtscore.eval`

Offline evaluation: text-sort, learned-sort, voting-iteration simulation, and plotting.

```python
@dataclass
class EvalQuery:
    """One (query text, target category) pair for text-sort evaluation."""

@dataclass
class QueryMetrics:
    """Metrics for one text-sort query (AP, P@k, R@k, etc.)."""

@dataclass
class LearnedSortMetrics:
    """Metrics for one learned-sort fold (binary classification stats + ranking AP)."""

@dataclass
class DatasetResult:
    """Aggregated results for one eval dataset.
    Has .to_dict(), .mean_average_precision, .mean_learned_f1 properties."""

def compute_average_precision(ranked_ids: list[int], relevant_ids: set[int]) -> float: ...
def compute_precision_recall_at_k(
    ranked_ids: list[int],
    relevant_ids: set[int],
    k: int,
) -> tuple[float, float]: ...
def compute_metrics(ranked_ids: list[int], relevant_ids: set[int]) -> QueryMetrics: ...
def compute_binary_classification_metrics(y_true, y_pred) -> dict[str, float]: ...

def eval_text_sort(
    dataset: DatasetContext,
    queries: list[EvalQuery],
) -> list[QueryMetrics]: ...

def eval_learned_sort(
    dataset: DatasetContext,
    *,
    test_fraction: float,
    n_folds: int,
) -> list[LearnedSortMetrics]: ...

def run_eval(...) -> list[DatasetResult]:
    """Full evaluation pipeline across datasets. Returns one DatasetResult per dataset."""

def format_results_json(results: list[DatasetResult]) -> str: ...

def simulate_voting_iterations(
    dataset: DatasetContext,
    *,
    n_seeds: int,
    target_category: str,
) -> dict:
    """Simulate casting votes in random order; record cost trajectory.
    Returns one trajectory per seed."""

def run_voting_iterations_eval(
    datasets: list[DatasetContext],
    categories: list[str],
    *,
    n_seeds: int,
) -> list[dict]: ...

def run_voting_iterations_eval_from_pickles(pickles: list[Path]) -> list[dict]: ...
```

> `plot_eval_results` and `plot_voting_iterations` are **not** part of `vtscore`.
> Plotting is presentation, not computation — those helpers move to `vtsearch/` and
> import matplotlib app-side. The library exports the data (`DatasetResult`,
> `QueryMetrics`, `LearnedSortMetrics`, `format_results_json`); rendering it is
> the app's job.

---

## `vtscore.state`

The library exports **context objects**. Module-level proxies (`medias`, `good_votes`,
`label_history`, etc.) stay app-side per the plan — they read `flask.g` and are
inherently a web-app affordance.

### Contexts (the library primitive)

```python
class DatasetContext:
    """Per-dataset state: medias dict, diversity tree, display name, cached embedding
    matrix. Mutable, RLock-protected."""

class DetectorContext:
    """Per-detector state: votes, labelset, label_embeddings cache, click times,
    trained MLP, threshold, labelset_source binding. Mutable, RLock-protected."""

@contextmanager
def with_dataset_context(ctx: DatasetContext) -> Iterator[None]:
    """Bind `ctx` as the active dataset for the current thread inside the block."""

@contextmanager
def with_detector_context(ctx: DetectorContext) -> Iterator[None]:
    """Bind `ctx` as the active detector for the current thread inside the block."""
```

### Dataset / detector registries

```python
def register_context(ctx: DatasetContext) -> None: ...
def unregister_context(dataset_id: str) -> None: ...
def get_context(dataset_id: str) -> DatasetContext: ...
def get_active_context() -> DatasetContext | None: ...
def set_thread_dataset_context(ctx: DatasetContext | None) -> None: ...
def get_thread_dataset_context() -> DatasetContext | None: ...
def list_loaded_dataset_ids() -> list[str]: ...
def clear_all_contexts() -> None: ...

def register_detector_context(ctx: DetectorContext) -> None: ...
def unregister_detector_context(detector_id: str) -> None: ...
def get_detector_context(detector_id: str) -> DetectorContext: ...
def get_active_detector_context() -> DetectorContext | None: ...
def set_thread_detector_context(ctx: DetectorContext | None) -> None: ...
def get_thread_detector_context() -> DetectorContext | None: ...
def list_loaded_detector_ids() -> list[str]: ...
def clear_all_detector_contexts() -> None: ...
```

### Operations on the active (or explicitly-passed) context

These take a DetectorContext / DatasetContext explicitly post-Phase 3. Today many of
them reach for the active context implicitly.

```python
def snapshot_medias(ctx: DatasetContext) -> dict[int, Media]: ...
def get_media(ctx: DatasetContext, cid: int) -> Media: ...
def clear_medias(ctx: DatasetContext) -> None: ...
def clear_all(dataset: DatasetContext, detector: DetectorContext) -> None: ...

def clear_votes(ctx: DetectorContext) -> None: ...
def toggle_vote(ctx: DetectorContext, cid: int, label: str) -> None: ...
def apply_label(ctx: DetectorContext, cid: int, label: str) -> None: ...
def apply_label_with_click_time(ctx: DetectorContext, cid: int, label: str) -> None: ...
def apply_labels_bulk_with_click_time(
    ctx: DetectorContext,
    labels: list[tuple[int, str]],
    *,
    replace_all: bool = False,
) -> None: ...
def add_label_to_history(ctx: DetectorContext, cid: int, label: str) -> None: ...
def add_textsort_suggestion(ctx: DetectorContext, text: str) -> None: ...
def get_textsort_suggestions(ctx: DetectorContext) -> list[str]: ...
def set_find_initial_labels(ctx: DetectorContext, labels: dict[int, str]) -> None: ...
def get_find_initial_labels(ctx: DetectorContext) -> dict[int, str]: ...
def update_learned_scores(ctx: DetectorContext, scores: dict[int, float]) -> None: ...
def get_learned_scores(ctx: DetectorContext) -> dict[int, float]: ...

def assign_click_time(ctx: DetectorContext, cid: int) -> None: ...
def remove_click_time(ctx: DetectorContext, cid: int) -> None: ...
def get_vote_click_times(ctx: DetectorContext) -> dict[int, float]: ...
```

### Diversity tree

```python
def build_diversity_tree(root_embedding, *, ctx: DatasetContext) -> DiversityTree: ...
def build_diversity_tree_for_context(ctx: DatasetContext) -> DiversityTree: ...
def get_diversity_tree(ctx: DatasetContext) -> DiversityTree | None: ...
def diversity_tree_next_sample(ctx: DatasetContext, *, exclude: set[int]) -> int | None: ...
def diversity_tree_label(ctx: DatasetContext, cid: int) -> None: ...
def diversity_tree_unlabel(ctx: DatasetContext, cid: int) -> None: ...
```

### Media lookup (origin → cid)

```python
def build_media_lookup(medias: dict, origin: dict | None, origin_name: str) -> dict: ...
def resolve_media_ids(lookup: dict, origin: dict, origin_name: str) -> list[int]: ...
def find_missing_entries(medias: dict, origin: dict, origin_name: str) -> list[dict]: ...
def collapse_duplicates(medias: dict, dupe_map: dict) -> None: ...
def get_dupe_count(medias: dict | None = None) -> int: ...
def next_media_id(medias: dict) -> int: ...
```

### What stays in the app, NOT in vtscore

The following names appear in `vtsearch.state` today but are app-side concerns and
**do not move** to the library:

- `medias`, `good_votes`, `bad_votes`, `label_history`, `vote_click_times`,
  `find_initial_labels`, `last_learned_scores`, `textsort_suggestions`, `inclusion`,
  `dataset_display_name`, `safe_thresholds`, `calibrate_count`, `calibration_fraction`,
  `enrich_descriptions` — these are `_ProxyDict` / `_ProxyList` objects that read
  `flask.g`. They stay in the app as a thin shim that delegates to library contexts.
- `autorun_detectors`, `autorun_extractors`, `autorun_localizers` and their CRUD
  (`add_autorun_extractor`, `remove_autorun_extractor`, `rename_autorun_extractor`,
  `get_autorun_extractors`, `get_autorun_extractors_by_media`, and the matching
  `_autorun_localizer` set) — *policy* about which processors to run automatically,
  which is an app concern. The library keeps the `Processor` / `Detector` /
  `Localizer` / `Extractor` ABCs and the code that applies them; the app keeps
  the list of which ones to apply on load. Phase 3 wires the resolved list
  through as an explicit argument to whichever library entry point needs it.
- `get_inclusion`, `set_inclusion`, `get_calibrate_count`, `set_calibrate_count`,
  `get_calibration_fraction`, `set_calibration_fraction`, `get_safe_thresholds`,
  `set_safe_thresholds`, `get_dataset_display_name`, `set_dataset_display_name`,
  `add_autorun_extractor`/`add_autorun_localizer` etc. — each of these wraps a
  `vtsearch.settings.get_*` / `set_*` accessor. The library never persists user prefs;
  it accepts the relevant value through a `CoreConfig` or a context field.

---

## `vtscore.plugins`

The discovery and field-declaration scaffolding shared by every plugin family.

```python
@dataclass
class PluginField:
    """One configurable input for a plugin. key, label, field_type, description,
    accept, options, default, required, placeholder, dynamic_options, depends_on,
    min, max, step. Validated by the plugin framework at registration time."""

class PluginBase:
    """Mixin providing CLI-arg parsing, JSON serialisation, and the `name`,
    `display_name`, `description`, `icon`, `fields`, `ui_mode`, `hidden_from_picker`
    class attributes that every plugin declares."""

class PluginRegistry(Generic[T]):
    """Auto-discovering registry. Constructed with (package, sentinel, label,
    discover_modules=..., entry_point_group=...). At first access it walks
    `package` for `sentinel` attributes and also scans the matching
    importlib.metadata entry_point_group; built-ins win on name clashes;
    broken entry points warn and are skipped."""

def make_plugin_registry(
    package: ModuleType | str,
    sentinel: str,
    label: str,
    *,
    discover_modules: list[str] | None = None,
    entry_point_group: str | None = None,
) -> tuple[Callable[[str], T], Callable[[], list[T]]]:
    """Convenience factory: returns (get, list) accessor pair, the standard shape
    every plugin-family module re-exports."""

FieldType = Literal[
    "file", "folder", "url", "text", "password", "email", "number",
    "select", "server_path", "checkbox",
]
```

---

## `vtscore.sync`

Generic bidirectional-sync ABC shared by settings sources (app-side) and labelset
sources (library-side).

```python
class SyncSource(PluginBase, Generic[LoadT, SaveT]):
    """Bidirectional sync target. Subclasses implement
    `load(field_values) -> LoadT` and `save(value: SaveT, field_values) -> None`.
    Field-driven and discoverable via the plugin registry."""
```

---

## `vtscore.concurrency`

Async job manager, progress tracking, and the memory-aware worker cap. The progress
infrastructure here is the **long-running operation** layer; do not confuse it with
the labeling-session analyzer in `vtscore.detectors.labeling_progress`.

### Async jobs

```python
@dataclass
class AsyncJob:
    """State container for one background job: job_id, signature, status, result,
    error, progress counters, cancel/done Events, user, dataset_id, detector_id."""

class JobManager:
    """Single-slot background job manager with one pending slot. Constructed with
    (name, max_history). Used by the learned-sort and eval routes today."""

learned_sort_jobs: JobManager
eval_jobs: JobManager
JOB_MANAGERS: dict[str, JobManager]
"""The built-in managers and the lookup table that exposes them by name."""

def list_active_pairs() -> list[tuple[str, str]]:
    """Enumerate (dataset_id, detector_id) pairs that currently have an active job."""
```

### Progress tracking

```python
class CancelledError(Exception):
    """Raised by ProgressTracker.check_cancelled() after .cancel()."""

class ProgressTracker:
    """Thread-safe single-operation progress tracker. extra_fields are stored on the
    snapshot alongside (status, current, total, message)."""

class LoadingTasksTracker:
    """A bag of named ProgressTrackers, each with a creation timestamp. Used to
    multiplex concurrent dataset/detector loads."""

loading_tasks: LoadingTasksTracker
detector_loading_tasks: LoadingTasksTracker
dataset_progress: ProgressTracker
sort_progress: ProgressTracker
eval_progress: ProgressTracker
find_progress: ProgressTracker

def set_thread_progress(cb: ProgressCallback) -> None: ...
def get_thread_progress() -> ProgressCallback | None: ...
def clear_thread_progress() -> None: ...

def update_progress(status, message, current, total, error=None, *,
                    staging_result=None, step=None, total_steps=None) -> None: ...
def get_progress() -> dict: ...
def cancel_dataset_progress() -> None: ...
def check_dataset_cancelled() -> None:
    """Raise CancelledError if cancel_dataset_progress() has been called."""

def update_sort_progress(...) -> None: ...
def get_sort_progress() -> dict: ...
def update_eval_progress(...) -> None: ...
def get_eval_progress() -> dict: ...
def update_find_progress(...) -> None: ...
def get_find_progress() -> dict: ...
```

### Memory budget

```python
def cap_workers_by_memory(
    n_items: int,
    embed_dim: int,
    *,
    max_workers: int,
    bytes_per_element: int = 4,
    budget_fraction: float = 0.25,
) -> int:
    """Cap worker count so the working set fits inside budget_fraction of free RAM.
    Used by dataset-load fan-out."""
```

---

## `vtscore.security`

Defensive helpers. Used at every external-input boundary; no app coupling.

```python
def get_file_access_base_dir() -> Path:
    """The single base directory under which validate_server_filepath permits paths."""

def validate_server_filepath(filepath_str: str, base_dir: Path) -> Path:
    """Resolve `filepath_str` and assert it stays inside `base_dir`.
    Raises on escape attempts."""

def sanitize_template_value(value: str) -> str:
    """Sanitise a value before substituting it into a filesystem-path template."""

def rglob_follow_symlinks(root: Path, pattern: str) -> Iterator[Path]: ...
def glob_top_level(root: Path, pattern: str) -> list[Path]: ...

def validate_url(url: str) -> str:
    """SSRF guard for outbound HTTP. Rejects private IPs, link-local, and metadata
    endpoints. Returns the validated URL."""

class RestrictedUnpickler(pickle.Unpickler):
    """Allowlist-based unpickler used for every untrusted pickle load."""

def safe_pickle_load(f: IO[bytes], **kwargs) -> Any: ...
def peek_pickle_dataset_summary(f: IO[bytes]) -> dict:
    """Read just enough of a dataset pickle to summarise it (count, media type, etc.)
    without materialising embeddings."""
```

---

## `vtscore.utils`

The leftover module. Most live helpers landed in their topical packages; this is the
small set that doesn't fit elsewhere.

```python
def build_media_hit(cid: int, media: Media, score: float, **extra) -> dict:
    """The single source of truth for the scored-media hit dict. Shared by CLI
    autodetect and /api/labels/fill-from-sort."""

# vtscore.utils.synthetic
def generate_audio_dataset(out_dir: Path, n: int, *, seed: int = 0) -> None: ...
def generate_image_dataset(out_dir: Path, n: int, *, seed: int = 0) -> None: ...
def generate_video_dataset(out_dir: Path, n: int, *, seed: int = 0) -> None: ...
"""Offline media synthesis used by the SyntheticDatasetImporter. Deterministic
given the same seed."""
```

---

## `vtscore.exporters`

Results exporters: each takes a `LabelSet` (or the equivalent ranking) and writes /
posts / mails it somewhere. They're registered exactly like importers, and the
sentinel module attribute is `EXPORTER`.

```python
class LabelsetExporter(PluginBase):
    """Abstract base. Subclasses declare `fields` and implement
    `export(labelset: LabelSet, field_values: dict) -> None`."""

ExporterField = PluginField  # backwards-compat alias

def get_exporter(name: str) -> LabelsetExporter: ...
def list_exporters() -> list[LabelsetExporter]: ...

# Built-ins (each registered via an `EXPORTER` sentinel in its module):
#   server_json_file, server_csv_file, webhook, email_smtp, gui (display-only,
#   hidden_from_picker), holder (scaffold, hidden_from_picker until API client lands).
```

---

## `vtscore.cli`

The CLI entry points an external scripting consumer would call directly. None of
these touch Flask; they each take an explicit settings path and an exporter name.

```python
def autodetect_main(
    dataset_path: str | Path,
    settings_path: str | Path,
    exporter_name: str,
    exporter_field_values: dict,
    *,
    dry_run: bool = False,
) -> None:
    """Load a pickled dataset, run every autorun detector, export the hits."""

def autodetect_importer_main(
    importer_name: str,
    field_values: dict,
    settings_path: str | Path,
    exporter_name: str,
    exporter_field_values: dict,
    *,
    dry_run: bool = False,
) -> None:
    """Same as autodetect_main but the dataset comes from an importer + field values
    instead of an on-disk pickle."""

def autodetect_main_chunked(
    dataset_path: str | Path,
    chunk_size: int,
    settings_path: str | Path,
    exporter_name: str,
    exporter_field_values: dict,
    *,
    dry_run: bool = False,
) -> None:
    """Chunked variant of autodetect_main for datasets too large to fit in memory."""

def autodetect_importer_main_chunked(
    importer_name: str,
    field_values: dict,
    chunk_size: int,
    settings_path: str | Path,
    exporter_name: str,
    exporter_field_values: dict,
    *,
    dry_run: bool = False,
) -> None: ...

# vtscore.cli_pipeline
def load_pipeline_file(path: str | Path) -> dict:
    """Parse a YAML pipeline file and return the validated config dict."""

# vtscore.cli_progress — progress output formatting (text vs NDJSON)
FORMATS: tuple[str, ...]              # ("text", "json")
def set_format(fmt: str) -> None: ...
def get_format() -> str: ...
def emit(event: str, *, text: str = "", stream=None, **fields) -> None: ...
def emit_error(message: str, *, stream=None) -> None: ...
```

---

## Known seams to cut before Phase 8

A single-source list of everything the inventory flagged as importing `flask` or
`vtsearch.settings`. Each must be detangled in Phases 1–4 before the `git mv`.

| Module                                      | Imports                       | Plan phase |
|---------------------------------------------|-------------------------------|-----------|
| `vtsearch/detectors/workflow.py`            | `flask.g`                     | Phase 1   |
| `vtsearch/state/core.py` (proxies only)     | `flask` (request context)     | Phase 1 (proxies stay app-side, not migrated) |
| `vtsearch/datasets/load_pipeline.py`        | `vtsearch.settings`           | Phase 2   |
| `vtsearch/datasets/registry.py`             | `vtsearch.settings`           | Phase 2   |
| `vtsearch/detectors/store.py`               | `vtsearch.settings`           | Phase 2   |
| `vtsearch/detectors/label_sync.py`          | `vtsearch.settings`           | Phase 2   |
| `vtsearch/detectors/label_restoration.py`   | `vtsearch.settings`           | Phase 2   |
| `vtsearch/detectors/dataset_sync.py`        | `vtsearch.settings`           | Phase 2   |
| `vtsearch/detectors/media_seeding.py`       | `vtsearch.settings`           | Phase 2   |
| `vtsearch/embedding/loader.py`              | `vtsearch.settings` (defaults helpers) | Phase 2 |
| `vtsearch/detectors/labeling_progress.py`   | `vtsearch.settings`           | Phase 2   |
| `vtsearch/labels/sync.py`                   | implicit Flask via state proxies | Phase 1/3 — takes explicit ctx after detangling |

Two earlier seams the plan called out have already been resolved by the codebase split:

- The plan refers to `vtsearch/models/training_workflow.py` (Flask-aware); that module
  is today `vtsearch/detectors/workflow.py` — same seam, new location.
- The plan refers to `vtsearch/models/loader.py`; that module is today
  `vtsearch/embedding/loader.py` — same settings imports.

The plan's "final shape" tree should be updated during Phase 1 to reflect the
current packages: `detectors/`, `embedding/`, `training/`, `concurrency/`,
`security/` are now distinct subpackages, not subdirectories of `models/`.

## Out of scope for `vtscore`

Explicit non-goals — these are app concerns that stay in `vtsearch/`:

- `vtsearch/app.py`, `vtsearch/routes/**` — every Flask blueprint.
- `vtsearch/auth/` — `LoginProvider` ABC and the default single-user impl.
  (External library consumers don't have "users".)
- `vtsearch/settings.py`, `vtsearch/settings_factory.py`,
  `vtsearch/settings_models.py`, `vtsearch/settings_io/` — auto-saving JSON user prefs.
  `SettingsSource` (the bidirectional-sync wrapper) stays app-side; the underlying
  `SyncSource[L,S]` ABC ships in `vtscore.sync`.
- `vtsearch/logging_config.py` — Flask-aware structured-log setup.
- `vtsearch/state/` proxy objects (`medias`, `good_votes`, etc.) — these read
  `flask.g`. The contexts they delegate to ship in `vtscore.state`.
- `vtsearch/medias.py` (test-media generator) — already lives under
  `tests/fixtures/medias.py`; it's not part of the runtime app.
- `vtsearch/achievements.py` — pure user-pref / gamification, no library consumer.
- `vtsearch/schemas/` — flask-smorest marshmallow schemas for the HTTP API.

## Phase 1 decisions

The four open questions surfaced at Phase 0 review are settled. The principle
that resolves them: **library = "the ability to do X"; app = "the policy that
decides when X happens, what gets persisted, and where it shows up."**

1. **`labelset_source` plugin family — ships in `vtscore.labels`.** Pulling labels
   in from external systems is a core consumer affordance (you can't use the
   library without it), so the ABC, the registry (`get_labelset_source` /
   `list_labelset_sources`), and the discovery sentinel all live library-side.
   The shared `SyncSource[L,S]` base in `vtscore.sync` is unchanged. Contrast
   with `SettingsSource`, which is a user-pref concern and stays in `vtsearch/`.
2. **Data visualization stays in the app.** Plotting eval results is presentation,
   not computation — it belongs next to the routes that render charts, not in
   the library that produces the numbers. `vtscore.eval` exports the dataclasses,
   metric functions, and runners (`run_eval`, `eval_text_sort`, `eval_learned_sort`,
   `simulate_voting_iterations`, `format_results_json`); the `plot_*` helpers move
   to `vtsearch/` and import matplotlib there. This also drops the need for a
   `vtscore[viz]` extra.
3. **`autorun_*` lists are app-side; processor execution is library-side.** The
   `Processor` / `Detector` / `Localizer` / `Extractor` ABCs and the code that
   applies them to media stay in `vtscore` — the *ability* to run a processor is
   a library concern. The registry of which processors to autorun
   (`autorun_detectors`, `autorun_extractors`, `autorun_localizers`) plus its
   CRUD (`add_autorun_extractor`, `remove_autorun_localizer`, etc.) is *policy*
   the app owns; it stays in `vtsearch/`. The library accepts the resolved list
   as an argument when a caller wants to run it.
4. **Mirror the concurrency progress pattern on `vtscore.media`: per-thread override
   plus global default.** Add `set_thread_progress_callback()` that takes priority
   when set on the calling thread, and keep `set_progress_callback()` as the
   global fallback for single-threaded consumers. Cost is ~10 lines
   (`threading.local` or a `ContextVar`); benefit is that a library consumer
   running multi-threaded ingestion (e.g. scoring two detectors in parallel)
   doesn't have one thread clobber another's callback. The shape mirrors
   `vtscore.concurrency.progress.set_thread_progress` so the two surfaces are
   consistent.
