# `vtscore.media`

Everything media-format-specific lives here: the `MediaType` plugin that
describes a content kind (audio, image, text, video, document), the
`MediaEmbedder` that turns one media item into a vector, the `MediaClipper`
that splits one media into sub-items of the same type, and the `Processor`
ABCs (`Detector` / `Localizer` / `Extractor`) that score or annotate media.
Each media-type sub-package self-registers at import time through sentinel
attributes — adding a new format is "drop a folder into `vtscore/media/`",
not "edit an `__init__.py`". This package has no Flask or settings imports
and is the foundation every other `vtscore` subsystem builds on.

---

## Quick start

```python
from vtscore.media import (
    get,
    get_embedder,
    embedders_for_type,
    all_types,
    clippers_for_type,
)

# Look up a registered media type by id.
audio = get("audio")                     # AudioMediaType instance
print(audio.file_extensions)             # ["*.wav", "*.mp3", ...]

# The default embedder for a media type sorts first.
default_embedder = embedders_for_type("image")[0]   # SigLIP
clip = get_embedder("clip")              # OpenAI CLIP, by name

# Default-clipper-per-type pattern. The first hit is always a no-op
# clipper that returns the media unchanged, e.g. ImageDefaultClipper.
tiling = next(c for c in clippers_for_type("audio") if c.name == "sound_tiling")
sub_medias = tiling.clip(audio_media_dict)
```

---

## Core ABCs

### `MediaType` — one per content kind

`vtscore/media/base.py:171` — an ABC bundling the file-extension filter,
demo-dataset list, HTTP-serving helper, and "load one file into a media
dict" loader for a single content format.

A concrete `MediaType` declares:

| Abstract member                     | Purpose                                           |
|-------------------------------------|---------------------------------------------------|
| `type_id` (`vtscore/media/base.py:204`)  | Internal identifier — `"audio"`, `"image"`, etc.  |
| `name`                              | Human-readable label for pickers                  |
| `icon`                              | SVG icon key                                      |
| `file_extensions`                   | List of glob patterns (e.g. `["*.wav", "*.mp3"]`) |
| `loops`                             | Whether the player loops (audio/video → `True`)   |
| `demo_datasets` (`vtscore/media/base.py:317`) | List of `DemoDataset` records                  |
| `load_media_data(path, bytes)` (`vtscore/media/base.py:391`) | Returns `{"duration": ..., "media_bytes": ...}` and any type-specific fields |
| `media_response(media)` (`vtscore/media/base.py:482`) | Returns a framework-agnostic `MediaResponse` |

A `MediaType` is **not** an embedder. Embedding is a separate plugin
(see below), and one media type may have zero, one, or many embedders
attached to it. `MediaType.load_models()` is a legacy no-op kept for
subclasses that still own their model loading.

The class also provides `_resolve_media_bytes(media)` /
`_resolve_media_string(media)` helpers
(`vtscore/media/base.py:418`, `vtscore/media/base.py:441`) that read
content from `media_bytes` → `media_path` → `media_url` in that order, so
implementations transparently handle in-memory, on-disk, and lazy-fetched
items.

### `MediaEmbedder` — file/text → vector

`vtscore/media/embedder.py:416` — an ABC for "take one media dict and
produce a fixed-D `np.ndarray`". Each embedder is bound to exactly one
`MediaType` via `media_type_id`, but a media type can have many
embedders. Subclasses implement four things:

| Member                                        | Required | Purpose                              |
|-----------------------------------------------|----------|--------------------------------------|
| `name`                                        | yes      | Unique registry key (e.g. `"clap"`)  |
| `media_type_id`                               | yes      | Which `MediaType.type_id` it targets |
| `_load_models_impl()`                         | yes      | Load weights from disk / Hub         |
| `_embed_media_impl(media)`                    | yes      | Forward pass for one item            |
| `embed_text(text)`                            | optional | Embed a query into the same space    |
| `_embed_media_bulk_impl(medias)`              | optional | Batched forward; default loops       |
| `description_wrappers`                        | optional | Prompts used by `embed_text_enriched`|

Threading and lock contract:

- `MediaEmbedder._embed_lock` (`vtscore/media/embedder.py:438`) is a
  **class-level** `threading.Lock` shared across every embedder
  subclass. `embed_media()` and `patch_forward()` both acquire it, so
  at most one forward pass runs at a time process-wide. Bulk callers
  acquire it per-item, not once per batch, so two parallel callers
  interleave smoothly.
- `_model_load_lock` (`vtscore/media/embedder.py:434`) is **per-class**.
  `load_models()` is idempotent and lock-protected — concurrent callers
  serialise on the first load, subsequent callers return immediately
  once `self._model is not None`.
- `_on_progress` is a per-instance attribute set by
  `set_progress_callback()`; default is a no-op so direct
  instantiation in tests doesn't require wiring.

Optional capability flags (`supports_text`, `supports_patch_regions`,
`license_notice`) describe what an embedder can do; the rest of the
stack reads these to gate features (text-search affordances, the
patch-region pipeline).

The helper module ships several shared building blocks every embedder
implementation uses:

| Helper                                                 | Purpose                                       |
|--------------------------------------------------------|-----------------------------------------------|
| `media_from_path(path)` (`vtscore/media/embedder.py:56`) | Wrap a `Path` in a minimal media dict.       |
| `embedder_load_setup(cb, msg)` (`vtscore/media/embedder.py:140`) | Wire torch threads, return cache dir. |
| `load_pretrained_local_first(fn, *)` (`vtscore/media/embedder.py:177`) | Prefer cached weights, retry transient HF errors. |
| `intercept_tqdm_progress(cb)` (`vtscore/media/embedder.py:215`) | Forward HF tqdm bars to your progress callback. |
| `intercept_weight_loading_progress(cb, label)` (`vtscore/media/embedder.py:378`) | Tensor-level progress for weight loading. |
| `extract_tensor(out)` (`vtscore/media/embedder.py:81`)  | Normalise the assorted shapes HF returns.    |
| `timed_progress(cb, status, msg)` (`vtscore/media/embedder.py:103`) | Append `(Ns)` to a stuck progress message. |
| `resolve_embed_batch_size(default)` (`vtscore/media/embedder.py:38`) | Read `$VTSEARCH_EMBED_BATCH_SIZE`.      |

### `MediaClipper` — split one media into sub-medias of the same type

`vtscore/media/clipper.py:9` — given one media dict, return one or more
media dicts of the **same** type. Used to tile a long audio clip into
fixed-length windows, slice a paragraph into sentences, crop an image
into a fixed bbox, etc. Clippers are how you bound the unit of
recall — every produced sub-media is what gets embedded and labelled.

```python
class MediaClipper(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    @abstractmethod
    def media_type(self) -> str: ...
    @abstractmethod
    def clip(self, media: dict) -> list[dict]: ...
```

Two optional hooks let the dataset-load pipeline tune a clipper at
load time:

- `resolve_for_durations(durations)` (`vtscore/media/clipper.py:145`) —
  dataset-level: decide once, given every item's duration.
- `resolve_for_media(media)` (`vtscore/media/clipper.py:156`) — per-item:
  auto-route to a different concrete clipper based on each item
  (e.g. pass-through for short audio, tiling for long).

Clippers may declare `parameters` (UI-tunable knobs) and override
`with_params()` to return a copy with overridden values. The **resolved**
clipper's `name` and parameter values are recorded in each output
clip's origin, so cross-dataset replay is deterministic regardless of
the original auto policy.

### `Processor` / `Detector` / `Localizer` / `Extractor`

`vtscore/media/processors.py:9`–`235`. Four ABCs in a hierarchy. The
generic `process(media)` method delegates to a typed method on each
subclass:

| ABC          | Typed method               | Return type                  |
|--------------|----------------------------|------------------------------|
| `Detector`   | `detect(media) -> bool`    | "does this media match?"     |
| `Localizer`  | `localize(media) -> list[dict]` | bounding boxes + confidence |
| `Extractor`  | `extract(media) -> list[dict]`  | structured per-occurrence metadata |

Every concrete processor is bound to one `media_type` and may override
`load_model()` for one-time weight loading (default: no-op).

Localizer outputs **must** include `"confidence"` (float in `[0, 1]`)
and `"bbox"` (format is media-specific). Extractor outputs must include
`"confidence"`; the rest of the schema is extractor-specific.

> Which processors run automatically when a dataset loads is an
> *app-side* concern, not a library concern. `vtscore.media` ships the
> ABCs and the application code calls them; the registry of "autorun
> these on every newly-loaded media" lives in `vtsearch/`. The library
> accepts the resolved processor list as an argument when a caller
> wants execution.

---

## Support types

### `MediaResponse`

`vtscore/media/base.py:69` — a `dataclass` for "this is bytes of MIME
type X, name it Y on download". Framework-agnostic so the same
`MediaType.media_response(media)` works whether the caller is a Flask
route, a Jupyter notebook, or a CLI exporter. The app converts it to a
`flask.Response` route-side.

```python
@dataclass
class MediaResponse:
    data: bytes | dict
    mimetype: str
    download_name: str = ""
```

### `DemoDataset`

`vtscore/media/base.py:88` — metadata for one downloadable demo dataset
attached to a media type. Bundles the id, label, description, category
slugs, optional slicing bounds, and the `required_folder` used both as
a staleness check on the pickle cache and as the browsable root for
the "Select Media Example" file picker.

### `ProgressCallback`

```python
ProgressCallback = Callable[[str, str, int, int], None]
# (status, message, current, total) -> None
```

(`vtscore/media/base.py:34`.) Threaded through every long-running
library op so the caller can report progress upstream. `current == 0`
and `total == 0` means indeterminate.

### `crop_file_bytes`

`vtscore/media/cropping.py:16` — apply a single-clip bounded clipper to
an arbitrary file. Used by upload/example-sort callers that need to
materialise a sub-region of one item without round-tripping through a
full clipper pipeline. Supported types: `"audio"` (start/end seconds)
and `"image"` (bbox in original-image pixel coords).

---

## Registry API

The registry is a small set of module-level dicts populated at import
time by `_discover_media_plugins()` (`vtscore/media/__init__.py:320`).
Three sentinel names drive discovery:

| Sentinel     | Location                          | Type                 |
|--------------|-----------------------------------|----------------------|
| `MEDIA_TYPE` | media-type package `__init__.py`  | `MediaType`          |
| `CLIPPERS`   | media-type package `__init__.py`  | `list[MediaClipper]` |
| `EMBEDDER`   | `embedder_*.py` inside a media-type package | `MediaEmbedder` |

Sub-packages **and** `.py` files are both scanned, and symlinks are
followed via `importlib.util.spec_from_file_location` — a custom
embedder living outside the VTSearch tree can be wired in by
symlinking a single file (or directory) into the relevant media-type
folder. No `__init__.py` edits required.

### Media-type accessors

| Function (`vtscore/media/__init__.py`)    | Use                                          |
|-------------------------------------------|----------------------------------------------|
| `register(mt)` (`:68`)                    | Manually register (for tests / third-party). |
| `get(type_id)` (`:73`)                    | Look up by `type_id`; raises `KeyError`.     |
| `get_by_folder_name(name)` (`:84`)        | Look up by `folder_import_name`.             |
| `get_by_extension(ext)` (`:101`)          | Match a file extension (`".wav"`).           |
| `all_types()` (`:96`)                     | Every registered `MediaType`.                |
| `all_type_ids()` (`:126`)                 | Just the ids.                                |
| `all_types_dict()` (`:135`)               | JSON-safe summaries.                         |
| `all_demo_datasets()` (`:144`)            | Flat `{id: info}` across every type.         |
| `normalize_type_id(type_id)` (`:60`)      | Legacy alias passthrough.                    |

### Embedder accessors

| Function                                                   | Use                                          |
|------------------------------------------------------------|----------------------------------------------|
| `register_embedder(emb)` (`vtscore/media/__init__.py:219`) | Manual registration.                         |
| `get_embedder(name)` (`vtscore/media/__init__.py:224`)     | Look up by `name`.                           |
| `embedders_for_type(type_id)` (`vtscore/media/__init__.py:234`) | All embedders for a media type, **default first**. |
| `all_embedders()` (`vtscore/media/__init__.py:245`)        | Every registered embedder.                   |
| `all_embedders_dict()` (`vtscore/media/__init__.py:250`)   | JSON-safe summaries.                         |

`embedders_for_type(t)[0]` is the default embedder for `t` (the one
whose `is_default` returns `True`). Exactly one embedder per media
type should override `is_default`.

### Clipper accessors

| Function                                                   | Use                                  |
|------------------------------------------------------------|--------------------------------------|
| `register_clipper(c)` (`vtscore/media/__init__.py:182`)    | Manual registration.                 |
| `get_clipper(name)` (`vtscore/media/__init__.py:187`)      | Look up by `name`.                   |
| `clippers_for_type(type_id)` (`vtscore/media/__init__.py:197`) | All clippers for a media type.   |
| `all_clippers()` (`vtscore/media/__init__.py:202`)         | Every registered clipper.            |
| `all_clippers_dict()` (`vtscore/media/__init__.py:207`)    | JSON-safe summaries.                 |

### Progress callback

```python
def set_progress_callback(cb: ProgressCallback) -> None: ...
```

(`vtscore/media/__init__.py:363`) — wires `cb` into every registered
`MediaType._on_progress` and `MediaEmbedder._on_progress`. Call this
once at startup. A thread-local override
(`set_thread_progress_callback`) is in the public-API sketch but lives
in `vtscore.concurrency.progress.set_thread_progress` today; see the
[concurrency](concurrency.md) doc.

---

## In-tree media types

Each lives under `vtscore/media/<type>/` and self-registers:

| Type     | Folder                       | Default embedder                | Other embedders ship in-tree                            |
|----------|------------------------------|---------------------------------|---------------------------------------------------------|
| audio    | `vtscore/media/audio/`       | `clap` (`laion/clap-htsat-unfused`) | `clap_general`, `clap_music`, `ast`, `whisper_encoder` |
| image    | `vtscore/media/image/`       | `siglip` (`google/siglip-base-patch16-224`) | `clip`, `siglip2`, `dinov2_single`, `dinov2_patch`, `dinov3_single`, `dinov3_patch`, `eupe_single`, `eupe_patch`, `face` |
| text     | `vtscore/media/text/`        | `e5` (`intfloat/multilingual-e5-base`) | `bge`                                            |
| video    | `vtscore/media/video/`       | `xclip` (`microsoft/xclip-base-patch32`) | `videomae`, `languagebind`                       |
| document | `vtscore/media/document/`    | — (uses converters; see below)  | —                                                       |

Each media-type package's `__init__.py` exposes the sentinels. For
example, `vtscore/media/audio/__init__.py`:

```python
from vtscore.media.audio.clipper import (
    SoundAutoClipper,
    SoundDefaultClipper,
    SoundSilenceClipper,
    SoundSpeechActivityClipper,
    SoundTilingClipper,
)
from vtscore.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
CLIPPERS = [
    SoundAutoClipper(),
    SoundDefaultClipper(),
    SoundTilingClipper(2.0),
    SoundSilenceClipper(),
    SoundSpeechActivityClipper(),
]
```

Each `embedder_<name>.py` module inside the folder ends with an
`EMBEDDER = MyEmbedder()` line, which is what `_discover_embedders_in`
picks up.

Patch-region image embedders (`dinov2_patch`, `dinov3_patch`,
`eupe_patch`) return `supports_patch_regions = True` and implement
`_patch_forward_impl`, producing a
`vtscore.media.patch_embed.PatchEmbedOutput` per image (CLS vector +
patch grid + saliency map). The dataset loader gates the patch
pipeline on this flag.

The `document` type is special: it has no native embedder, and is
intended to be embedded indirectly via converters
(document → image → image-embedder, or document → text →
text-embedder). See [converters](converters.md).

---

## Implementing a new media type

Sketch — the full walkthrough is in
[../../docs/EXTENDING-media.md](../../../docs/EXTENDING-media.md).

1. Create `vtscore/media/<type>/__init__.py`, `media_type.py`,
   `clipper.py`, and one or more `embedder_<name>.py` modules.
2. In `media_type.py`, subclass `MediaType` and implement every
   abstract property/method.
3. In `__init__.py`, expose `MEDIA_TYPE = MyMediaType()` and
   `CLIPPERS = [...]`.
4. In each `embedder_<name>.py`, subclass `MediaEmbedder`, implement
   `_load_models_impl` and `_embed_media_impl`, and expose
   `EMBEDDER = MyEmbedder()` at module top level.
5. Restart the process; the registries pick everything up.

The same sentinel pattern works for embedders shipped as
sub-packages (`embedder_<name>/__init__.py`) and for symlinked
out-of-tree implementations.

---

## Gotchas

- **The model load lock is per-class, but the embed lock is
  class-level shared across every `MediaEmbedder` subclass.** That
  means two different embedders cannot run forward passes
  concurrently; the global lock is intentional (one GPU at a time).
  If you spin up a third-party embedder that wants its own threading
  story, override `embed_media_bulk` carefully and respect the
  contract.
- **`_on_progress` is per-instance**, set by
  `set_progress_callback`. Cloning an embedder via deep-copy will
  carry the old callback; prefer re-registering or calling
  `set_progress_callback` again.
- **Embedders do not own a `to_disk` / `from_disk`.** Vectors and
  trained MLP weights are in-memory artefacts only. Re-derive on
  demand from origins (`Origin → file → embedding`). The single
  exception is dataset pickles, which snapshot media + embeddings as
  one unit (see [datasets](datasets.md)).
- **`load_models()` performs network I/O.** It hits the HuggingFace
  Hub on first call to download weights, even with `local_files_only`
  set (the helper retries transient 5xx / timeout errors with
  exponential backoff). Wrap calls in `intercept_tqdm_progress` /
  `intercept_weight_loading_progress` if you want progress reported
  to a UI.
- **Patch-region embedders must override `_patch_forward_impl`.** If
  `supports_patch_regions = True` but the implementation defaults to
  `None`, the dataset loader will store empty region data. This is
  not enforced by the ABC — the `True` flag is treated as a promise.
- **Discovery is eager and silent on import errors.** A broken
  embedder module emits a `warnings.warn(...)` and the registry skips
  it. Check `all_embedders()` after import if a custom embedder
  doesn't show up.
- **Torch threading.** `vtscore.media.torch_setup.ensure_torch_configured`
  (`vtscore/media/torch_setup.py:16`) reads `vtscore.config.TORCH_THREADS`
  (env `$VTSEARCH_TORCH_THREADS`, default `1`) and calls
  `torch.set_num_threads` the first time torch is imported. Every
  code path that touches torch (embedders, MLP training, scoring)
  must call this first. `embedder_load_setup` does it for you.

---

## Cross-references

- [embedding](embedding.md) — façade over the registry plus the
  matrix cache and smart preload.
- [converters](converters.md) — cross-type bridges (audio →
  spectrogram, OCR, ASR, video → keyframes).
- [../../docs/EXTENDING-media.md](../../../docs/EXTENDING-media.md) — the
  full walkthrough for adding a media type, embedder, or clipper.
- [../../docs/EXTENDING-processors.md](../../../docs/EXTENDING-processors.md) —
  adding detectors / localizers / extractors.
