# `vtscore.media`

Everything media-format-specific lives here: the `MediaType` plugin that
describes a content kind (audio, image, text, video, document, face), the
`MediaEmbedder` that turns one media item into a vector, the `MediaClipper`
that splits one media into sub-items of the same type, and the `Processor`
ABCs (`Detector` / `Localizer` / `Extractor`) that score or annotate media.
Each media-type sub-package self-registers at import time through sentinel
attributes - adding a new format is "drop a folder into `vtscore/media/`",
not "edit an `__init__.py`". This package has no Flask or settings imports
and is the foundation every other `vtscore` subsystem builds on.

## Contents

**Shared surface** - the top-level modules, which every media type builds on.

| Module | Concern |
|--------|---------|
| `vtscore/media/__init__.py` | The registries and their accessors (`get`, `get_embedder`, `all_types`, …), plus sentinel discovery |
| `vtscore/media/base.py` | The `MediaType` ABC |
| `vtscore/media/embedder.py` | The `MediaEmbedder` ABC and its capability flags |
| `vtscore/media/load_progress.py` | Model-load progress interception (tqdm, weight tensors) and resilient HF fetching |
| `vtscore/media/torch_ops.py` | Torch tensor/device adapters shared by every embedder |
| `vtscore/media/clipper.py` | The `MediaClipper` ABC (1 → N, same type) |
| `vtscore/media/cleaner.py` | The `MediaCleaner` ABC (1 → 1 cleanup gate, run before embedding) |
| `vtscore/media/processors.py` | The `Processor` / `Detector` / `Localizer` / `Extractor` ABCs |
| `vtscore/media/cropping.py` | Shared normalised-box cropping helpers |
| `vtscore/media/patch_embed.py` | Raw patch grids for image embedders (the MaxPatch region geometry) |
| `vtscore/media/lazy_clip.py` | Derive a clip's bytes from its source on demand |
| `vtscore/media/clip_recipe.py` | Parse the `origin.params` clip dialects, shared by both replay paths |
| `vtscore/media/provenance.py` | Human-readable provenance for converter- and clipper-derived media |
| `vtscore/media/structural.py` | Structural (instance-matching) features and geometric verification |
| `vtscore/media/structural_geometry.py` | Parametrised geometric verification models |
| `vtscore/media/structural_splg.py` | SuperPoint + LightGlue structural backend |
| `vtscore/media/near_dupes.py` | Near-duplicate detection (image pHash, text SimHash) and collapsing |
| `vtscore/media/torch_setup.py` | Runtime torch configuration for embedder code |

**Media types** - one self-registering sub-package each. Every one holds a
`media_type.py` (the `MediaType`), plus whichever of `clipper.py`,
`cleaner.py`, `decode.py`, `embedder_*.py` and processor modules it needs.

| Sub-package | Embedders | Also holds |
|-------------|-----------|------------|
| `vtscore/media/audio/` | CLAP (general / music / base), AST, BEATs, ParaSpeechCLAP, Whisper | clipper, cleaner, ffmpeg + decode helpers, WAV byte slicing (`wav.py`), silence detection, speech extractor, synthetic audio generator |
| `vtscore/media/image/` | SigLIP, SigLIP-L, SigLIP 2, SigLIP2-L, CLIP, CLIP-L (eval-only), DINOv2 (patch / single), DINOv3 (patch / single), EUPE (patch / single), SIFT-VLAD (+ `_doc` variant) | clipper, cleaner, decode, edge trim, thumbnail, YOLO extractor, OCR extractor, face localizer, demo sources |
| `vtscore/media/text/` | E5, BGE | clipper (paragraph / sentence), cleaner |
| `vtscore/media/video/` | X-CLIP, LanguageBind, VideoMAE v2 | clipper (auto / tile / scene-detect), cleaner, ffmpeg decode, frame sampling, `clip_box` crop |
| `vtscore/media/document/` | none - a *convert-out* half type (converted to image or text before embedding) | clipper |
| `vtscore/media/face/` | FaceNet | clipper - a *convert-in* half type: never imported, only produced by the `image2face` converter |

Underscore-prefixed modules (`_clap_shared.py`, `_dinov3_shared.py`,
`_frame_sampling.py`, `vtscore/media/_toponymy_demo.py`, …) are internal
helpers, not public surface. `vtscore/media/assets/` holds bundled data
files (the SIFT-VLAD codebook).

The package root re-exports the ABCs (`MediaType`, `MediaEmbedder`,
`MediaClipper`, `MediaCleaner`, `Processor`, `Detector`, `Localizer`,
`Extractor`), the support types (`MediaResponse`, `DemoDataset`,
`ProgressCallback`) and every registry accessor below.

---

## Quick start

```python
from vtscore.media import get, get_embedder, embedders_for_type, get_clipper

audio = get("audio")                     # AudioMediaType instance
print(audio.file_extensions)             # ["*.wav", "*.mp3", "*.flac", ...]

default_embedder = embedders_for_type("image")[0]   # SigLIP (default sorts first)
clip = get_embedder("clip")              # by name; KeyError if unknown

tiling = get_clipper("sound_tiling")
sub_medias = tiling.clip(audio_media_dict)
```

---

## Core ABCs

The concept-level picture (media dicts, embedders) is in
[concepts.md](../concepts.md#1-media); this section is the member
reference. Authoring walkthroughs are in
[extending/media-types.md](../extending/media-types.md),
[extending/embedders.md](../extending/embedders.md) and
[extending/clippers.md](../extending/clippers.md).

### `MediaType` - one per content kind

`vtscore/media/base.py::MediaType` bundles the file-extension filter,
demo-dataset list, serving helper, and "load one file into a media dict"
loader for a single content format.

| Abstract member | Purpose |
|-----------------|---------|
| `type_id` | Internal identifier - `"audio"`, `"image"`, etc. |
| `name` | Human-readable label for pickers |
| `icon` | SVG icon key |
| `file_extensions` | List of glob patterns (e.g. `["*.wav", "*.mp3"]`); empty for `face` |
| `loops` | Whether the player loops (audio/video → `True`) |
| `demo_datasets` | List of `DemoDataset` records |
| `load_media_data(file_path, media_bytes=None)` | Returns type-specific fields to merge into the media dict; must include `"duration"` |
| `media_response(media)` | Returns a framework-agnostic `MediaResponse` |

Non-abstract members worth knowing:

| Member | Purpose |
|--------|---------|
| `importable` (property) | `True` for every type a user can import; `False` for a convert-in half type (`face`). |
| `embeddable` (property) | `True` iff `embedders_for_type(type_id)` is non-empty; `False` for `document`. |
| `converts_to` | Embeddable `type_id`s a non-embeddable type converts into (`document` → `["image", "text"]`). |
| `has_thumbnail` | Whether items have a still-image thumbnail (drives VTSBrowse bin shape). |
| `folder_import_name`, `dir_key`, `pickle_extra_fields`, `display_metadata(media)` | Import / pickle / display plumbing with sensible defaults. |
| `load_thin_media_data(file_path)` | `load_media_data` minus the payload keys, for thin (by-reference) loads. |
| `ensure_thumbnail_bytes(media)`, `image_response(media)` | Thumbnail helpers. |
| `load_models()` | Legacy no-op kept for subclasses that still own model loading. |

A `MediaType` is **not** an embedder: embedding is a separate plugin, and a
media type may have zero, one, or many embedders.

`_resolve_media_bytes(media)` / `_resolve_media_string(media)` read content
in this order: `media_bytes` / `media_string` in memory → a lazy-clip recipe
in `origin.params` (see `vtscore/media/lazy_clip.py`) → an archive member →
`media_path` on disk → `media_url` (fetched through the SSRF guard). So
implementations transparently handle in-memory, thin, derived and
URL-backed items.

### `MediaEmbedder` - media/text → vector

`vtscore/media/embedder.py::MediaEmbedder` takes one media dict and produces
a fixed-D `np.ndarray`. Each embedder is bound to exactly one `MediaType`
via `media_type_id`. Public methods are framework wrappers (locking,
L2-normalisation, progress); **subclasses override the `_…_impl` hooks**.

| Member | Required | Purpose |
|--------|----------|---------|
| `name` | yes | Unique registry key (e.g. `"clap"`) |
| `media_type_id` | yes | Which `MediaType.type_id` it targets |
| `_load_models_impl()` | yes | Load weights from disk / Hub (called by `load_models()`) |
| `_embed_media_impl(media)` | yes | Forward pass for one item (called by `embed_media()`) |
| `_embed_text_impl(text)` | optional | Embed a query into the same space (called by `embed_text()`); default `None` = no text search |
| `_embed_media_bulk_impl(medias)` | optional | Batched forward; default loops `embed_media` |
| `_patch_forward_impl(media)` / `_patch_forward_bulk_impl` | patch embedders | Return a `vtscore.media.patch_embed.PatchEmbedOutput` (CLS vector, `(H, W, D)` patch grid, saliency) |
| `_local_features_forward_impl(media)` / `…_bulk_impl` | structural embedders | Return structural features for geometric verification |
| `description_wrappers` | optional | Prompts used by `embed_text_enriched`; `[]` (default) makes it plain `embed_text` |
| `loaded_backbone()` | optional | `(model, processor)` for the raw backbone; default reads `_model` / `_processor` |

`embed_media_bulk`, `embed_text` and `embed_text_enriched` return
**L2-normalised** vectors, so subclasses need not normalise.

Capability flags (properties, all overridable) gate features elsewhere in
the stack: `is_default`, `eval_only` (withheld from pickers and
`embedders_for_type`), `supports_text`, `supports_patch_regions`,
`supports_geometric_verification`, `embed_batch_size`, `license_notice`,
plus the descriptive `display_name`, `model_id`, `embedding_dim`.

Threading and progress contract:

- `MediaEmbedder._embed_lock` is a **class-level** `threading.Lock` shared
  by every embedder subclass. `embed_media()`, `patch_forward()` and
  `local_features_forward()` acquire it, so at most one forward pass runs at
  a time process-wide. The default bulk path acquires it per item, so two
  parallel callers interleave.
- `_model_load_lock` is **per-class** (created in `__init_subclass__`).
  `load_models()` returns immediately once `self._model is not None` and
  otherwise serialises concurrent first loads.
- `_on_progress` is **per-thread** over a process-wide default. Set the
  default with `set_default_progress_callback(cb)` (what
  `vtscore.media.set_progress_callback` does); route one thread's progress
  with the `progress_scope(cb)` context manager, or silence it with
  `silent_progress()`. Two concurrent loads sharing one singleton embedder
  therefore never cross-report.

Shared building blocks are **re-exported from `vtscore.media.embedder`**
(the column names the defining module):

| Helper | Defined in | Purpose |
|--------|------------|---------|
| `media_from_path(file_path, origin=None)` | `embedder.py` | Wrap a path in a minimal media dict. |
| `resolve_embed_batch_size(default=32)` | `embedder.py` | Read `$VTSEARCH_EMBED_BATCH_SIZE`, falling back to `default`. |
| `embedder_load_setup(on_progress, message)` | `load_progress.py` | Configure torch threads, report progress, return the model cache dir. |
| `load_pretrained_local_first(fn, *args, **kwargs)` | `load_progress.py` | Prefer cached weights; retry transient HF errors. |
| `hf_token()` | `load_progress.py` | The HF auth token to pass to `from_pretrained`. |
| `intercept_tqdm_progress(callback)` | `load_progress.py` | Forward HF tqdm bars to a progress callback. |
| `intercept_weight_loading_progress(callback, label=...)` | `load_progress.py` | Tensor-level progress for weight loading. |
| `timed_progress(on_progress, status, message, ...)` | `load_progress.py` | Append an elapsed `(Ns)` to a long-running progress message. |
| `extract_tensor`, `to_compute_device`, `to_model_inputs`, `to_float32`, `embed_autocast` | `torch_ops.py` | Torch tensor / device adapters. |

Image embedders additionally share `vtscore/media/image/_image_bulk.py`,
which decodes each batch on a thread pool **one batch ahead** of the
forward pass - see
[`resolve_decode_workers()`](config.md#allocated_cpus--resolve_decode_workers)
for sizing.

### `MediaClipper` - split one media into sub-medias of the same type

`vtscore/media/clipper.py::MediaClipper`: given one media dict, return one
or more media dicts of the **same** type (tile audio into windows, split a
paragraph into sentences, crop an image). Every produced sub-media is what
gets embedded and labelled. Abstract members: `name`, `media_type`,
`clip(media) -> list[dict]`.

Optional members: `display_name`, `description`, `summary_template`,
`parameters` (UI-tunable knobs), `creation_questions`, `with_params(params)`
(return a re-parameterised copy), and `resolve_for_media(media)` - the
per-item hook the load pipeline
(`vtscore/datasets/clipper_chain.py::_run_clipper_step`) calls before
`clip`, used by auto-routing clippers such as `video_auto`. The
**resolved** clipper's name and parameters are recorded in each clip's
origin, so replay is deterministic.

`resolve_for_durations(durations)` is **reserved and inert**: nothing calls
it. It stays because it is published contract; put routing logic in
`resolve_for_media`.

`DefaultClipper(name, media_type, description)` is the concrete pass-through
base every `*DefaultClipper` subclasses. The module also exports tiling
helpers (`clip_with_bounds`, `validate_tiling_params`, `tile_starts`,
`tiling_parameters`).

### `MediaCleaner` - 1 → 1 cleanup before embedding

`vtscore/media/cleaner.py::MediaCleaner` subclasses `MediaClipper`.
Implement `clean(media) -> dict` (return the media unchanged when there is
nothing to do; a cleaner never aborts a load); `clip` wraps it as a
single-output chain step. `default_enabled` (default `False`) sets whether
the import UI pre-checks it. Cleaners have their own registry so they never
appear in a clipper chooser; registration order is run order.

### `Processor` / `Detector` / `Localizer` / `Extractor`

`vtscore/media/processors.py`. Every processor declares `name` and
`media_type`, may override `load_model()` (default no-op), and implements
`process(media)` via a typed method:

| ABC | Typed method | Returns |
|-----|--------------|---------|
| `Detector` | `detect(media) -> bool` | "does this media match?" |
| `Localizer` | `localize(media) -> list[dict]` | regions; each **must** include `"confidence"` (float in `[0, 1]`) and `"bbox"` (media-specific) |
| `Extractor` | `extract(media) -> list[dict]` | per-occurrence metadata; each **must** include `"confidence"` |

Which processors run automatically on load is decided by the host
application; the library only ships the ABCs and the in-tree
implementations. See [../../docs/EXTENDING-processors.md](../../../docs/EXTENDING-processors.md).

---

## Support types

| Name | Defined in | Description |
|------|------------|-------------|
| `MediaResponse(data, mimetype, download_name="")` | `base.py` | Dataclass: bytes (or a dict) of a MIME type, with a download name. Framework-agnostic return of `MediaType.media_response`. |
| `DemoDataset` | `base.py` | Dataclass describing one downloadable demo dataset: `id`, `label`, `description`, `categories`, `source`, `required_folder`, slicing bounds (`slice_start` / `slice_end` / `slice_frac_*`), `items_per_category`, `download_size_mb`. |
| `ProgressCallback` | `vtscore/concurrency/progress.py` (re-exported) | `Callable[[str, str, int, int], None]` - `(status, message, current, total)`; `0, 0` means indeterminate. See [concurrency.md](concurrency.md). |
| `crop_file_bytes(file_path, media_type, params)` | `cropping.py` | Crop one file without a clipper pipeline. `"audio"`: `{"start", "end"}` seconds; `"image"`: `{"box": [x1, y1, x2, y2]}` in original-image pixels. `ValueError` for other types. |

---

## Registry API

Populated at import time by `vtscore/media/__init__.py::_discover_media_plugins`,
which scans every sub-package of `vtscore/media/` (symlinked directories
included) for sentinels:

| Sentinel | Location | Type |
|----------|----------|------|
| `MEDIA_TYPE` | media-type package `__init__.py` | `MediaType` |
| `CLIPPERS` | media-type package `__init__.py` | `list[MediaClipper]` |
| `CLEANERS` | media-type package `__init__.py` | `list[MediaCleaner]` |
| `EMBEDDER` | an `embedder*.py` module or `embedder*/` sub-package inside a media-type package | `MediaEmbedder` |

Symlinked embedder files / directories are loaded via
`importlib.util.spec_from_file_location`, so an out-of-tree embedder can be
wired in by symlinking it into the media-type folder. A module that fails
to import emits `warnings.warn(...)` and is skipped. There is no
entry-point group for this family; out-of-tree code can also call the
`register*` functions directly.

All `get*` lookups raise `KeyError` for an unknown key.

| Media types | Embedders | Clippers | Cleaners |
|-------------|-----------|----------|----------|
| `register(mt)` | `register_embedder(emb)` | `register_clipper(c)` | `register_cleaner(c)` |
| `get(type_id)` | `get_embedder(name)` | `get_clipper(name)` | `get_cleaner(name)` |
| `all_types()` | `all_embedders()` | `all_clippers()` | `all_cleaners()` |
| `all_types_dict()` | `all_embedders_dict()` | `all_clippers_dict()` | `all_cleaners_dict()` |
| `all_type_ids()` | `embedders_for_type(type_id)` | `clippers_for_type(type_id)` | `cleaners_for_type(type_id)` |
| `get_by_folder_name(name)`, `get_by_extension(ext)` (`None` on miss), `all_folder_names()`, `all_demo_datasets()`, `normalize_type_id(type_id)` (identity passthrough) | `embedder_for_medias(media_dict)` | | |

`embedders_for_type(t)` returns the **user-selectable** embedders for `t`,
default first (`embedders_for_type(t)[0]` is the default) and
`eval_only` embedders withheld; `all_embedders_dict()` applies the same
filter. `get_embedder` and `all_embedders` are unfiltered.
`embedder_for_medias(media_dict)` reads the `"embedder"` name on the first
media, falling back to the type's default.

`set_progress_callback(cb)` wires `cb` into every registered `MediaType`
and, via `set_default_progress_callback`, every embedder. Call it once at
startup.

---

## In-tree media types

| Type | Default embedder | Other embedders | Clippers | Cleaners |
|------|------------------|-----------------|----------|----------|
| `audio` | `clap_general` (`laion/larger_clap_general`) | `clap`, `clap_music`, `ast`, `beats`, `paraspeechclap`, `whisper_encoder` | `sound_tiling`, `sound_default`, `sound_silence`, `sound_speech_activity` | `audio_silence_trim` |
| `image` | `siglip` (`google/siglip-base-patch16-224`) | `clip`, `siglip_l`, `siglip2`, `siglip2_l`, `dinov2_single`, `dinov2_patch`, `dinov3_single`, `dinov3_patch`, `eupe_single`, `eupe_patch`, `sift_vlad`, `sift_vlad_doc`; `clip_l` is `eval_only` | `image_default`, `image_tiling`, `image_object` | `image_exif_orient`, `image_edge_trim` |
| `text` | `e5` (`intfloat/e5-base-v2`) | `bge` | `text_default`, `text_paragraph`, `text_sentence` | `text_markup_strip`, `text_whitespace` |
| `video` | `xclip` (`microsoft/xclip-base-patch32`) | `languagebind`, `videomae` | `video_auto`, `video_default`, `video_tiling`, `video_scene` | `video_letterbox_crop`, `video_blank_trim` |
| `document` | none (convert-out: `converts_to = ["image", "text"]`) | - | `document_default` | - |
| `face` | none flagged default; `face` (FaceNet) is the only embedder | - | `face_default` | - |

`face` declares no file extensions and `importable = False`: face crops
come only from the `image2face` converter (see
[converters.md](converters.md)). `document` has no embedder and must be
converted first.

Patch-region image embedders (`dinov2_patch`, `dinov3_patch`, `eupe_patch`)
set `supports_patch_regions = True` and implement `_patch_forward_impl`; the
dataset loader gates the patch pipeline on that flag.

Each media-type package's `__init__.py` exposes the sentinels, e.g.
`vtscore/media/audio/__init__.py`:

```python
MEDIA_TYPE = AudioMediaType()
CLIPPERS = [
    SoundTilingClipper(10.0, 1.0),
    SoundDefaultClipper(),
    SoundSilenceClipper(),
    SoundSpeechActivityClipper(),
]
CLEANERS = [AudioSilenceTrimCleaner()]
```

---

## Implementing a new media type

Full walkthroughs:
[extending/media-types.md](../extending/media-types.md) (library tier) and
[../../docs/EXTENDING-media.md](../../../docs/EXTENDING-media.md) (app tier).
In short: create `vtscore/media/<type>/` with `media_type.py` (subclass
`MediaType`), `__init__.py` exposing `MEDIA_TYPE` / `CLIPPERS` (and
optionally `CLEANERS`), and one `embedder_<name>.py` per embedder ending in
`EMBEDDER = MyEmbedder()`. The registries pick everything up on the next
import.

---

## Gotchas

- **One forward pass at a time, process-wide.** `_embed_lock` is shared by
  every `MediaEmbedder` subclass. An embedder that wants its own threading
  story should override `_embed_media_bulk_impl` and respect the contract.
- **Prefer `progress_scope` over assigning `_on_progress`.** Assignment is
  thread-scoped; a process-wide default needs `set_default_progress_callback`.
- **Embedders have no `to_disk` / `from_disk`.** Vectors and trained
  weights are in-memory only; see
  [architecture.md](../architecture.md#the-no-persisted-vectors-rule).
- **`load_models()` may do network I/O** on first call to fetch weights
  (`load_pretrained_local_first` prefers the local cache and retries
  transient HF errors).
- **`supports_patch_regions = True` is a promise.** The ABC's
  `_patch_forward_impl` returns `None`; if you set the flag without
  overriding it, the loader stores empty region data.
- **Torch threading.** `vtscore/media/torch_setup.py::ensure_torch_configured`
  applies `vtscore.config.TORCH_THREADS` (env `$VTSEARCH_TORCH_THREADS`,
  default `1`) once, if torch is already imported. `embedder_load_setup`
  calls it for you.

---

## Near-duplicate collapsing

`vtscore.state.collapse_duplicates` matches on **exact** MD5 (see [`state.md`](state.md#media-lookup)). `vtscore/media/near_dupes.py`
catches the rest: a re-encoded JPEG and a reformatted copy of the same
article are not byte-identical but are the same item for labeling
purposes, and leaving both in inflates the dataset and wastes votes.

| Function | Description |
|----------|-------------|
| `phash_image(thumbnail_bytes)` | 64-bit DCT perceptual hash, or `None` for missing / undecodable bytes |
| `simhash_text(text)` | 64-bit SimHash over word 4-shingles, or `None` for empty text |
| `collapse_near_duplicates(media_dict, on_progress=None)` | Group by Hamming distance and collapse each group; returns the group count |

Only `image` and `text` media are considered; every other type is left
untouched, and any item whose hash comes back `None` is simply left
ungrouped. Grouping is banded LSH over the 64-bit hashes plus a
union-find, so it does not pay the O(N²) pairwise comparison.

This runs *after* the exact-MD5 pass, so some group members are already
`dupe_set` representatives; their members are merged into the near-dup
group rather than nested. The representative keeps a `dupe_set` origin
listing every member's provenance, exactly as in the exact case. It is
opt-in per dataset via `DatasetContext.merge_near_duplicates`.

---

## Cross-references

- [embedding](embedding.md) - façade over the registry plus the
  matrix cache and smart preload.
- [converters](converters.md) - cross-type bridges (audio → image / text,
  document → image / text, image → face / text, video → audio / image).
- [extending/](../extending/README.md) - library-tier authoring guides for
  media types, embedders and clippers.
