# `vtscore.datasets`

Dataset ingestion: what gets loaded into memory, where it came from, and
how it gets persisted. Everything that converts external bytes (a folder of
WAVs, an HTTP archive, a `.pkl` snapshot, a demo download) into the
in-memory media dict that the rest of `vtscore` operates on lives here.
The package owns three things: the domain objects (`Origin`, `LabeledElement`,
`LabelSet`), the loaders / exporter that move data in and out of memory,
and the auto-discovered registries of importers and media sources that
plug new bytes-providers into the system.

## Contents

The largest package in the library. Every module, grouped by role.

**Domain objects and provenance**

| Module | Concern |
|--------|---------|
| `vtscore/datasets/origin.py` | `Origin` - importer name + the params that reproduce one element |
| `vtscore/datasets/labelset.py` | `LabeledElement`, `LabelSet`, `element_key` |
| `vtscore/datasets/metadata.py` | Per-media metadata extraction |
| `vtscore/datasets/file_types.py` | Best-effort file-type labelling for media dicts |
| `vtscore/datasets/config.py` | Dataset configurations, built from the media-type registry |
| `vtscore/datasets/demo_counts.py` | Exact demo-dataset media counts, measured once and written down |

**Loading and persistence**

| Module | Concern |
|--------|---------|
| `vtscore/datasets/loader.py` | `export_dataset_to_file` and the shared loader surface |
| `vtscore/datasets/loader_folder.py` | Folder loaders (whole and chunked) |
| `vtscore/datasets/loader_pickle.py` | Pickle loaders (whole and chunked) |
| `vtscore/datasets/loader_demo.py` | Demo-dataset loader |
| `vtscore/datasets/load_pipeline.py` | Load orchestration: background threading, gate handoff, staging |
| `vtscore/datasets/stages/` | The post-import stages the pipeline runs - see below |
| `vtscore/datasets/container.py` | ZIP-based dataset container format |
| `vtscore/datasets/registry.py` | Persistent on-disk registry of `.pkl` datasets |
| `vtscore/datasets/ingest.py` / `ingest_task.py` | Re-run origins to ingest missing medias, foreground and background |
| `vtscore/datasets/split.py` | Deterministic category-stratified splits for evaluation |

**Bytes providers**

| Module | Concern |
|--------|---------|
| `vtscore/datasets/importers/` | Dataset-importer registry, ABCs, and the built-in importers |
| `vtscore/datasets/sources/` | `MediaSource` registry and the built-in low-level media sources |
| `vtscore/datasets/downloader/` | Demo-dataset downloaders, per media type |
| `vtscore/datasets/archive.py` | Extract zip/tar/rar archives and load their media |
| `vtscore/datasets/archive_stream.py` | Stream one member out of a tar/zip **without** extracting |

**Load-time transforms and helpers**

| Module | Concern |
|--------|---------|
| `vtscore/datasets/clipper_chain.py` | Ordered converter / clipper / cleaner steps applied at load |
| `vtscore/datasets/media_type_detection.py` | Sample a folder and guess which media type dominates |
| `vtscore/datasets/pdf.py` | PDF page rendering |
| `vtscore/datasets/thumbnail_warm.py` | Background thumbnail warm-up for archive-member datasets |

### `vtscore/datasets/stages/`

The post-import pipeline, one module per stage, run in this order:

| Module | Stage |
|--------|-------|
| `clipper.py` | Clipper / converter chain, plus per-clip MD5 + embedding fixup |
| `embedding.py` | Embed items the importer left unembedded |
| `finalize.py` | Drop failed embeds, collapse duplicates, build the coverage atlas |
| `projection.py` | Optional 2-D UMAP projection build + persist |
| `registry.py` | Save to the dataset registry, migrate the context id |

`_common.py` holds shared constants and helpers; `_load_profiler.py` and
`_load_cost_model.py` are the env-gated per-phase timing recorder and
the affine cost coefficients fit from its measurements, which is how the
loader predicts a remaining-time estimate.

---

## What you load into

Every loader populates a single dict, conventionally bound to the name
`medias`, keyed by sequential integer IDs starting at 1. Each value is a
plain `dict[str, Any]` (there is no `Media` dataclass); with the shape
below. The same shape round-trips through `export_dataset_to_file` /
`load_dataset_from_pickle`.

| Key                  | Type                          | Notes                                                                                  |
|----------------------|-------------------------------|----------------------------------------------------------------------------------------|
| `id`                 | `int`                         | Sequential media ID, same as the dict key.                                             |
| `media_type`         | `str`                         | Media type id: `"audio"`, `"image"`, `"text"`, `"video"`, `"document"`, `"face"`.       |
| `embedder`           | `str`                         | Name of the registered `MediaEmbedder` that produced the **primary** vector.           |
| `md5`                | `str`                         | Hex digest of the **source bytes** (streamed, constant memory). See "MD5" gotcha.      |
| `embeddings`         | `dict[str, np.ndarray]`       | `embedder_name -> vector`. One media can carry a vector per bound embedder; read it through `vtscore.embedding.media_vectors`. On pickle round-trip the vectors become plain lists and back. |
| `filename`           | `str`                         | Basename of the source file, e.g. `"clip_123.wav"`.                                    |
| `category`           | `str`                         | Category derived from the parent folder name (or `"unknown"`).                         |
| `origin`             | `dict \| None`                | Serialised `Origin.to_dict()`: `{"importer": ..., "params": {...}}`.                  |
| `origin_name`        | `str`                         | Unique name within the origin (typically the relative path or filename).               |
| `media_bytes`        | `bytes \| None`               | Inline raw bytes (default). `None` when `thin=True`.                                   |
| `media_path`         | `str \| None`                 | Filesystem path. Set by thin loaders instead of `media_bytes`.                         |
| `media_string`       | `str \| None`                 | Text payload for `type == "text"` (the loaded paragraph).                              |
| `duration`           | `float \| None`               | Seconds, for audio / video.                                                            |
| `file_size`          | `int`                         | Bytes on disk.                                                                         |
| `width` / `height`   | `int \| None`                 | Image / video dimensions.                                                              |
| `word_count` / `character_count` | `int \| None`     | Text statistics.                                                                       |
| `custom_metadata`    | `dict[str, Any] \| None`      | Arbitrary per-media display metadata attached by importers.                            |
| `media_url`          | `str \| None`                 | Optional URL for lazy-fetch flows (set by URL-based importers).                        |

Importers may attach additional keys; the schema above is the durable
subset that `export_dataset_to_file` preserves (see
`vtscore/datasets/loader.py`).

**MD5 gotcha:** `md5` is the hash of the **raw source file bytes**,
computed via `file_md5` (`vtscore/utils/hashing.py`). It is
*not* the hash of the embedding, of `media_bytes` after any in-memory
transformation, or of a clipped sub-region. Folder-importer subclasses
can short-circuit this calculation by populating `content_md5s` on the
importer instance.

---

## Domain objects

### `Origin`

`vtscore/datasets/origin.py`: the importer name plus the params
needed to reproduce one media element. Every media dict carries one
(serialised via `Origin.to_dict()`); every `LabeledElement` carries
one. Hashable, equal by value, and the canonical persisted form of
"where this thing came from"; the *no persisted vectors* rule means
an Origin plus a fresh embedder must be sufficient to re-derive the
embedding.

```python
from vtscore.datasets import Origin

o = Origin("server_folder", {"path": "/data/audio", "media_type": "audio"})
o.to_dict()           # {'importer': 'server_folder', 'params': {...}}
o.display()           # 'server_folder(/data/audio)'
Origin.from_dict(o.to_dict()) == o      # True
```

### `LabeledElement`

`vtscore/datasets/labelset.py`: one `(md5, label, origin, …)` tuple,
the unit of a labelset. The optional `metadata` field is a free-form
`dict[str, Any]` that round-trips through `to_dict()` / `from_dict()`,
so an importer can attach external system identifiers that survive
re-export.

| Field         | Type                                          | Notes                                                              |
|---------------|-----------------------------------------------|--------------------------------------------------------------------|
| `md5`         | `str`                                         | Content hash (matches the corresponding media dict's `md5`).       |
| `label`       | `str`                                         | `"good"` or `"bad"`.                                               |
| `origin`      | `dict \| None`                                | Serialised `Origin`; `None` for legacy label files with no origin. |
| `origin_name` | `str`                                         | Unique name within the origin.                                     |
| `filename`    | `str`                                         | Original basename.                                                 |
| `category`    | `str`                                         | Class label from the source dataset structure.                     |
| `metadata`    | `dict[str, Any] \| None`                      | Free-form per-element metadata (round-trips).                      |
| `region_box`  | `tuple[float, float, float, float] \| None`   | Normalised `(x0, y0, x1, y1)` for region-level yes-votes.          |

`to_dict()` only emits non-empty optional fields, keeping the
serialised form compact for legacy consumers that only look at
`md5` and `label`.

### `LabelSet`

`vtscore/datasets/labelset.py`: ordered list of `LabeledElement`
plus an optional `detector_meta` block (`media_type`, `input_spec`,
`threshold`). The serialised form
`{"labels": [...], "detector_meta": {...}}` is a strict superset of
the legacy `{"labels": [{"md5", "label"}, …]}` shape.

```python
from vtscore.datasets import LabelSet, LabeledElement

ls = LabelSet([
    LabeledElement(md5="abc", label="good", filename="dog1.wav"),
    LabeledElement(md5="def", label="bad",  filename="cat1.wav"),
])
ls2 = LabelSet.from_dict(ls.to_dict())
len(ls2)                  # 2

# Construction from live state:
ls3 = LabelSet.from_clips_and_votes(medias, good_votes, bad_votes)
ls4 = LabelSet.from_results(autodetect_results)

# Merge across sources; elements dedup by Origin when present, else md5.
merged = ls_a.merge(ls_b, conflict_policy="drop")
```

`"drop"` is the only supported `conflict_policy` today; entries with
disagreeing labels across inputs are silently removed. See
`element_key` in `vtscore/datasets/labelset.py` for the dedup key.

---

## Loaders and exporter

The public surface in `vtscore/datasets/loader.py` is a re-export façade
over three sibling modules:

| Function                        | Module                    | Populates / returns                                  |
|---------------------------------|---------------------------|------------------------------------------------------|
| `load_dataset_from_folder`      | `loader_folder.py`    | Populates `medias` in-place; returns `None`.         |
| `load_dataset_from_folder_chunked` | `loader_folder.py` | Iterator of chunk dicts.                             |
| `load_dataset_from_pickle`      | `loader_pickle.py`    | Populates `medias` in-place; returns `None`.         |
| `load_dataset_from_pickle_chunked` | `loader_pickle.py` | Iterator of chunk dicts.                             |
| `load_demo_dataset`             | `loader_demo.py`       | Populates `medias` in-place; returns `None`.         |
| `export_dataset_to_file`        | `loader.py`           | Returns pickle **bytes** (caller writes to disk).    |

All three primary loaders **mutate** the `medias` dict the caller passes
in; they clear it first, then populate it with sequential int IDs
starting at 1. They do **not** return the dict; treat the in-place
mutation as the result.

### Folder loader

```python
from vtscore.datasets import load_dataset_from_folder

medias: dict[int, dict] = {}
load_dataset_from_folder(
    folder_path=Path("/data/sounds"),
    media_type="audio",
    medias=medias,
    embedder_name="clap",         # "" picks the first registered embedder
    thin=False,                   # True stores media_path instead of media_bytes
    recursive=True,
)
```

Extra hooks short-circuit work the caller has already done:

- `content_vectors: dict[str, np.ndarray]`: reuse a pre-computed embedding per filename.
- `content_md5s: dict[str, str]`: reuse a pre-computed MD5.
- `custom_metadata_map: dict[str, dict[str, Any]]`: attach `custom_metadata`; nested `"md5"` and `"embedding"` keys take priority over both the above.
- `skip_embedding=True`: load metadata only; files without a pre-computed vector get `embedding=None`.

`media_type` is looked up in the `vtscore.media` registry by
`MediaType.folder_import_name`, so a new media type registered through
the media-tier plugin path becomes loadable without touching this
function.

### Pickle loader

```python
from vtscore.datasets import load_dataset_from_pickle, export_dataset_to_file

medias: dict[int, dict] = {}
load_dataset_from_pickle(Path("dataset.pkl"), medias, thin=False)

# round-trip: exporter returns bytes; caller writes them:
Path("snapshot.pkl").write_bytes(export_dataset_to_file(medias))
```

`export_dataset_to_file` returns the pickle as a `bytes` blob and
converts any `np.ndarray` embeddings to plain lists so the result
deserialises cleanly under
[`vtscore.security.safe_pickle_load`](../../security/pickle.py).
Pickle files are the **only sanctioned vector store** in the project
(CLAUDE.md "No Persisted Vectors"); detector labelsets, settings, and
sync source files all persist origins and re-derive embeddings on
demand.

### Demo loader

```python
from vtscore.datasets import load_demo_dataset, DEMO_DATASETS

medias: dict[int, dict] = {}
load_demo_dataset("esc50", medias, embedder_name="clap")
```

`load_demo_dataset` is a cache-aware wrapper: it loads a previously
embedded `<dataset_name>.pkl` from `vtscore.config.EMBEDDINGS_DIR`
when the cached embedder matches, and otherwise downloads + embeds
fresh. The actual download / embedding is delegated to each
`MediaType.load_demo_source` implementation, so adding a new demo
dataset is a media-tier concern.

### Progress

Every loader takes an optional `on_progress: Callable[[str, str, int,
int], None]`. When omitted it falls back to the per-thread or global
progress callback in `vtscore.concurrency.progress`. Multi-threaded
consumers should pass an explicit callback per call to avoid one
thread clobbering another's reporter (`vtscore/datasets/loader.py`).

---

## Demo metadata loaders

Per-format helpers in `vtscore/datasets/metadata.py` that turn an
on-disk demo dataset's metadata sidecar (CSV, MAT, CIFAR pickle
batch, folder tree) into a `dict[str, dict[str, Any]]` keyed by
filename: `load_esc50_metadata`, `load_urbansound8k_metadata`,
`load_audio_metadata_from_folders`, `load_oxford_flowers_metadata`,
`load_places365_metadata`, `load_cifar10_batch`,
`load_video_metadata_from_folders`,
`load_image_metadata_from_folders`,
`load_paragraph_metadata_from_folders`. Library consumers normally
don't call these directly; they're plumbed into the demo loaders.
They're public because external scripts re-use them when staging
their own copies of the same datasets.

---

## Importers

`DatasetImporter` (`vtscore/datasets/importers/base/dataset_importer.py`) is the ABC
for "convert external bytes into a `medias` dict". Subclasses declare
`fields: list[PluginField]`, implement `run(field_values, medias,
thin=False)` (or the higher-level `list_records` + `fetch_record`
hooks), and expose a module-level `IMPORTER` sentinel.

```python
from vtscore.datasets import get_importer, list_importers

imp = get_importer("server_folder")    # KeyError if absent
for i in list_importers():             # hidden_from_picker excluded
    print(i.name, i.display_name)
```

Third-party importers can register via the `vtscore.importers`
entry-point group; built-ins win on name clashes. See
[plugins](plugins.md) for the registry mechanism in general.

### Built-in importers

| Name               | Display name           | Notes                                                                             |
|--------------------|------------------------|-----------------------------------------------------------------------------------|
| `server_folder`    | Server                 | Server-side folder scan.                                                          |
| `server_files`     | Files                  | Hidden from picker. Server-side file list.                                        |
| `local`            | Local                  | Browser-upload placeholder; re-enters `server_folder`.                            |
| `pickle`           | Upload Saved Dataset   | Hidden. `.pkl` round-trip path.                                                   |
| `http_archive`     | Import from URL        | Hidden. Downloads + extracts an archive.                                          |
| `demo`             | Downloaded Media       | Wraps `load_demo_dataset`.                                                        |
| `synthetic`        | Synthetic Media        | Generates deterministic media via `vtscore.utils.synthetic`.                      |
| `combine_datasets` | Combined Datasets      | Hidden. Internal: merges two loaded datasets.                                     |

**Multi-media imports.** Every importer accepts a `source_specs` form
value: a list of `SourceSpec(source_type, converter, params)` rows
that fan one import out across several source media types, each
optionally run through a named `MediaConverter`. The framework owns
conversion and ingestion; subclasses never call `get_converter()`
themselves.

**Importer override points** (pick one, simplest first):

1. `list_records()` + `fetch_record()`: single-source-type service
   importer. Default `fetch_source_media()` delegates here.
2. `fetch_source_media(spec, ...)`: multi-source-type service
   importer where the backend serves one type per query. Framework
   calls you once per spec; you yield raw source-type media dicts.
3. `fetch_all_source_media(specs, ...)`: multi-source-type service
   importer where one upstream call returns mixed types. Framework
   calls you once with the full spec list; you yield
   `(spec, raw_media)` pairs. Default delegates to
   `fetch_source_media()` per spec.
4. `run()`: folder-shaped importers that own the medias dict
   directly (typical body: stage files, call
   `load_dataset_from_folder()` + `run_converters_on_folder()`).

For hooks 1–3 the framework also assigns sequential IDs, fills
`media["origin"]` from `build_origin()`, and runs each spec's
converter on the yielded raw media before storing the result.

**Per-record hooks (`list_records` / `fetch_record`).** Hook 1's
bulk variant `_fetch_records_bulk_impl` lets you replace the per-item
loop with batched / concurrent I/O.

**Resolving back to a file.** Importers whose media is reachable on
disk **must** override `resolve_file(origin, origin_name, filename) ->
Path | None` (`vtscore/datasets/importers/base/core.py`). Cross-dataset
features (applying a saved detector to a different dataset via Find,
re-embedding a labelset after switching embedders) depend on it. The
default returns `None`, which is only correct when the media genuinely
cannot be relocated (e.g. browser-uploaded pickles with no server path).

---

## Media sources

`MediaSource` (`vtscore/datasets/sources/base.py`) is a thin
abstraction *below* the importer: it knows how to enumerate items at a
location (`list_items`), fetch one by key (`fetch_item`), and resolve
a stored origin back to a `Path` (`resolve_path`). Importers that deal
with individual files (server folder, HTTP archive) compose a
`MediaSource`; importers that don't (pickle, combine_datasets) skip
this layer entirely.

```python
from vtscore.datasets.sources import get_source_for_origin

origin = {"importer": "server_folder", "params": {"path": "/data/audio"}}
src = get_source_for_origin(origin)
if src is not None:
    for item in src.list_items(extensions=[".wav"]):
        path = src.fetch_item(item.key)
    src.cleanup()                 # archive sources may have a temp dir
```

Auto-discovery uses the `SOURCE` sentinel; matching is by the
factory's `name` against `origin["importer"]`. Sources are
**stateful** (an HTTP archive may unpack a zip on first access), so
each `get_source_for_origin` call returns a fresh instance; call
`cleanup()` when done.

### Built-in media sources

| Name                  | Module                                          | Use case                                  |
|-----------------------|-------------------------------------------------|-------------------------------------------|
| `server_folder`       | `vtscore/datasets/sources/local_folder.py`      | Local filesystem folder.                  |
| `http_archive`        | `vtscore/datasets/sources/http_archive.py`      | Downloaded zip / tar; unpacks on demand.  |

---

## On-disk dataset registry

`vtscore/datasets/registry.py` maintains a JSON manifest at
`<DATA_DIR>/dataset_registry.json` listing every dataset the user has
*saved* to disk (one `.pkl` per dataset, location resolved via
`CoreConfig.from_settings().saved_datasets_dir`).

```python
from vtscore.datasets.registry import (
    list_datasets, get_dataset, register_dataset, unregister_dataset,
    add_loaded_id, remove_loaded_id, is_loaded,
)

register_dataset(
    name="My Sounds", media_type="audio", num_items=1234,
    pkl_path="/data/saved_datasets/ds_xxx.pkl",
    origin="server_folder", embedder="clap",
)
```

This is **not the same** as the in-memory dataset *context* registry
in [`vtscore.state`](state.md), which tracks `DatasetContext`s
currently live in RAM. The two coordinate via `_loaded_ids` (the set
of saved-dataset IDs whose `.pkl` is currently materialised into a
context) but their lifecycles are independent: a dataset can be saved
without being loaded, and a context can exist for a one-shot in-memory
load that was never registered.

Per-entry fields include `created_by`, `readers` (list of usernames or
`"*"`), and `file_type_counts`. Access-control helpers
(`can_user_access`, `is_owner`, `list_datasets_for_user`,
`set_readers`) implement multi-user visibility rules; library-only
consumers without users can ignore them and treat every dataset as
visible.

---

## Splits

`split_dataset(medias, test_fraction, seed)` does category-stratified
train/test splits with reproducible per-category RNGs derived from
`SHA-256("<seed>:<category>")` (`vtscore/datasets/split.py`).
Adding or removing categories does not perturb the split of other
categories. Categories of size ≥ 2 are guaranteed at least one item
in each split; clip IDs are preserved (not renumbered).

```python
from vtscore.datasets import split_dataset

simulate, test = split_dataset(medias, test_fraction=0.2, seed=42)
```

---

## Concurrency gates

`vtscore/datasets/load_pipeline.py` defines `ConcurrencyGate`: a
semaphore whose limit is re-read on every `acquire()`, so changes to
the underlying setting affect queued and future tasks without
preempting in-flight ones. Two module-level gates drive dataset
loading:

| Gate              | Limit source                                              | Phase covered                                       |
|-------------------|-----------------------------------------------------------|-----------------------------------------------------|
| `_download_gate`  | `CoreConfig.max_concurrent_dataset_downloads`             | Importer's download / import step (bandwidth-bound) |
| `_embed_gate`     | `CoreConfig.max_concurrent_dataset_embeddings`            | Embedding + post-load (CPU/GPU/RAM-bound)           |

A load acquires the download gate first, switches to the embed gate
on the importer's first `"embedding"` progress event, and runs
post-load steps (clipping, dedup, diversity tree, embedder warm-up)
under the embed gate. One dataset can start downloading while another
is still embedding. Library consumers normally don't touch these
gates directly; they're driven by the load orchestrator and read
their limits from `CoreConfig`. Default limits are 1/1, preserving
serialised behaviour out of the box.

---

## `DEMO_DATASETS`

```python
from vtscore.datasets import DEMO_DATASETS

DEMO_DATASETS["esc50"]
# {'label': 'ESC-50 Animals', 'description': ..., 'category': [...], 'media_type': 'audio', ...}
```

A flat `dict[str, DemoDataset]` populated lazily from every registered
`MediaType.demo_datasets` at import time
(`vtscore/datasets/config.py`). Adding a media type plugin
automatically adds its demos here; there is no central registration
step. `load_demo_dataset(name, medias)` keys into this dict; the
`demo` importer enumerates it for its picker.
