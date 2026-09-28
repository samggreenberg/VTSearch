# `vtscore.converters`

A *converter* turns a media dict of one type into one or more media
dicts of a **different** type. They're how you embed audio with an
image model (audio → spectrogram → SigLIP), search OCR'd images with a
text model (image → text → E5), or run a video through a
frame-image pipeline. Every converter is auto-discovered via the
`CONVERTER` sentinel, so adding one is a one-file change.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/converters/base.py` | The `MediaConverter` ABC and the `resolve_media_bytes` helper |
| `vtscore/converters/__init__.py` | The auto-discovering registry and its accessors |
| `vtscore/converters/runner.py` | `run_converters_on_folder` / `apply_converter_to_demo` - wire conversion into dataset import |
| `vtscore/converters/audio2image.py` | Render audio as a mel-spectrogram / CQT image |
| `vtscore/converters/audio2text.py` | Transcribe speech via Whisper (ASR) |
| `vtscore/converters/image2text.py` | OCR an image for embedded text |
| `vtscore/converters/image2face.py` | Localise faces and emit one crop per detection |
| `vtscore/converters/document2image.py` | Render document pages as images |
| `vtscore/converters/document2text.py` | Extract a document's embedded text |
| `vtscore/converters/video2image.py` | Extract frames as images |
| `vtscore/converters/video2audio.py` | Extract the audio track |

**See also:** [`../extending/converters.md`](../extending/converters.md)
for writing a converter (contract, parameters, packaging, tests).

---

## When to use a converter

A converter is the right tool when the embedder you want to apply
isn't directly compatible with the source format:

| You have      | You want                                        | Use                                  |
|---------------|-------------------------------------------------|--------------------------------------|
| Audio files   | Embed them with an image model (SigLIP, DINOv3) | `audio2image` (spectrogram)          |
| Audio files   | Transcribe speech and embed with a text model   | `audio2text` (Whisper ASR)           |
| Image files   | Search them by the text they contain            | `image2text` (OCR)                   |
| Image files   | Embed each face in FaceNet identity space       | `image2face`                         |
| Video files   | Embed individual frames with an image model     | `video2image`                        |
| Video files   | Embed the audio track with an audio model       | `video2audio`                        |
| Documents (PDF) | Run an image embedder over each page          | `document2image`                     |
| Documents     | Embed extracted body text                       | `document2text`                      |

Converters are not embedders - they don't produce vectors. They
produce media dicts, which the framework embed stage then embeds with
the **target** media type's embedder.

---

## `MediaConverter` ABC

`vtscore/converters/base.py`. A `PluginBase` subclass - same
field-driven configuration system every other plugin family uses.

```python
class MediaConverter(PluginBase, ABC):
    display_name: str = ""
    description: str = ""
    summary_template: str = ""               # "{key}" placeholders for the UI preview
    fields: list[PluginField] = []           # user-configurable params

    @property
    def name(self) -> str:                   # default: f"{source}2{target}"
        return f"{self.source_type}2{self.target_type}"

    @property
    @abstractmethod
    def source_type(self) -> str: ...        # type_id of input

    @property
    @abstractmethod
    def target_type(self) -> str: ...        # type_id of output

    @abstractmethod
    def convert(self, media: dict, params: dict | None = None) -> list[dict]: ...

    def convert_normalized(self, media: dict, params: dict | None = None) -> list[dict]: ...
    def normalize_params(self, params: dict | None) -> dict: ...
    def validate_params(self, params: dict | None) -> dict: ...
    def get_param(self, params: dict | None, key: str) -> Any: ...
```

The contract:

- `convert(media, params)` returns a **list** of new media dicts.
  Empty list means "could not convert" (an empty document, a decode
  failure).
- Each returned dict must contain at minimum `"filename"` and the
  data fields expected by the target media type (e.g. `"media_bytes"`
  + `"duration"` for image/audio/video, `"media_string"` for text). It
  does **not** include `"id"`, `"embedding"`, or `"md5"` - the runner
  and embed stage assign those.
- The *source* media carries `media_bytes` in a full import but only
  `{filename, media_path}` in reference (*thin*) mode. Read binary
  input with `resolve_media_bytes(media)` (bytes, else the file at
  `media_path`, else `None`), not `media["media_bytes"]`.
- **Framework call sites use `convert_normalized`**, never `convert`
  directly. It runs `normalize_params` (drops empty-string values for
  fields with a default or that are optional, validates against the
  `fields` schema, fills defaults; raises `ValueError` on invalid
  input), calls `convert`, and copies the source's `custom_metadata`
  onto each output that lacks one. So `convert` may index
  `params[key]` directly. `get_param` remains as a shim for
  converters called with raw params.

---

## Built-in converters

Eight ship in-tree, all in `vtscore/converters/`. Each module ends
with `CONVERTER = MyConverter()`, which the registry picks up
automatically.

| Name | Class | Source → Target | Fields | Backend (lazy import) |
|------|-------|-----------------|--------|-----------------------|
| `audio2image`    | `Audio2ImageMediaConverter`    | audio → image    | `spectrogram_type`, `n_mels`, `time_window_s`, `colormap` | `librosa` + `matplotlib` |
| `audio2text`     | `Audio2TextMediaConverter`     | audio → text     | `model_size`, `language` | `openai-whisper` (`import whisper`) |
| `image2text`     | `Image2TextMediaConverter`     | image → text     | `language`, `threshold` | PaddleOCR |
| `image2face`     | `Image2FaceMediaConverter`     | image → face     | `threshold`, `padding`, `min_size` | `facenet-pytorch` (MTCNN), opt-in |
| `video2image`    | `Video2ImageMediaConverter`    | video → image    | `n_clips` / `seconds_per_frame` (mutually clearing) | `vtscore.media.video.decode` |
| `video2audio`    | `Video2AudioMediaConverter`    | video → audio    | `ffmpeg_timeout` | FFmpeg |
| `document2image` | `Document2ImageMediaConverter` | document → image | - | PyMuPDF (`agpl` extra) |
| `document2text`  | `Document2TextMediaConverter`  | document → text  | - | PyMuPDF (`agpl` extra) |

All but `Image2FaceMediaConverter` are re-exported from
`vtscore.converters`; import that one from
`vtscore.converters.image2face`, or look any of them up by name:

```python
from vtscore.converters import get_converter

v2i = get_converter("video2image")
outputs = v2i.convert_normalized(media_dict, {"n_clips": "20"})
```

---

## Registry & discovery

`vtscore/converters/__init__.py` builds the registry with the standard
plugin machinery ([`plugins.md`](plugins.md)):

```python
get_converter, list_converters = make_plugin_registry(
    package=__name__,
    sentinel="CONVERTER",
    label="media converter",
    discover_modules=True,
    entry_point_group="vtscore.converters",
)
```

Every module under `vtscore.converters` exposing `CONVERTER`, plus
anything registered under the `vtscore.converters` entry-point group,
is discovered eagerly at import time.

| Function                                  | Purpose                                    |
|-------------------------------------------|--------------------------------------------|
| `list_converters()`                       | Every registered converter.                |
| `get_converter(name)`                     | Look up by `name`; `None` on miss.         |
| `list_converters_for_target(target_type)` | All converters producing `target_type`.    |
| `list_converters_for_source(source_type)` | All converters consuming `source_type`.    |

```python
from vtscore.converters import list_converters_for_target

for c in list_converters_for_target("image"):
    print(c.name, "<-", c.source_type)
# audio2image <- audio
# document2image <- document
# video2image <- video
```

Out-of-tree converters register through the entry-point group:

```toml
[project.entry-points."vtscore.converters"]
my_converter = "my_pkg.my_converter:CONVERTER"
```

---

## Running a converter - the runner

`vtscore/converters/runner.py`. Importers don't call `convert()`
directly; they call:

```python
def run_converters_on_folder(
    folder_path: Path,
    target_media_type: str = "",
    medias: dict[int, dict] | None = None,
    thin: bool = False,
    on_progress: ProgressCallback | None = None,
    base_origin: dict | None = None,
    recursive: bool = True,
    converter_specs: list | None = None,
) -> None: ...
```

For each spec in *converter_specs* - `SourceSpec(converter, params, ...)`
objects or `{"converter": ..., "params": ...}` dicts; specs with no
converter (the "include directly" rows) and unknown names are skipped,
as are converters whose `target_type` isn't *target_media_type* - it:

1. Scans *folder_path* for files matching the converter's **source**
   media type's extensions.
2. Calls `converter.convert_normalized(source_media, params)` on each
   (a converter that raises is logged and skipped).
3. Appends each output to *medias* with sequential IDs starting after
   the current max, **unembedded** (`embeddings={}`, `embedder=""`).
   The framework embed stage (`vtscore.datasets.stages.embedding.embed_missing`)
   embeds them after the importer returns.
4. Records a replayable `converter` origin on each output (below).

In *thin* mode each output also carries a `_lazy_source` marker; its
bytes are stripped after embedding and re-derived on demand by
`vtscore.media.lazy_clip`.

```python
from pathlib import Path
from vtscore.converters.runner import run_converters_on_folder

medias: dict[int, dict] = {}
run_converters_on_folder(
    folder_path=Path("/data/recordings"),
    target_media_type="image",
    medias=medias,
    base_origin={"importer": "server_folder", "params": {"path": "/data/recordings"}},
    converter_specs=[{"converter": "audio2image", "params": {}}],
)
# `medias` now holds unembedded spectrogram media; embed them before scoring.
```

`apply_converter_to_demo(converter_name, dataset_name, medias,
embedder_name="", on_progress=None)` converts every media of a demo
dataset **in place** (`medias` ends up holding only the converted
outputs, renumbered from 1). Raises `ValueError` for an unknown
converter. `embedder_name` is accepted and ignored: the embed stage
resolves the target type's embedder itself.

### Converted media dicts and their origin

Each output becomes (`_build_converted_media_dict`):

```python
{
    "id": <assigned>, "media_type": <converter.target_type>,
    "embedder": "", "embeddings": {},
    "file_size": <len of media_bytes / media_string>,
    "md5": <content md5 of the output>,
    "filename": origin_name, "origin_name": f"{source_rel}→{output_filename}",
    "category": "custom",
    "origin": {"importer": "converter", "params": {...}},
    "media_path": <resolved source path>,
    "duration": <output.get("duration", 0)>,
    # plus whichever the output carries: media_bytes, media_string,
    # width, height, word_count, character_count
}
```

The origin `params` (all values stored as strings):

| Key | Meaning |
|-----|---------|
| `converter` | The converter's name (`video2image`, …) |
| `source_file` | The source filename **relative to the scanned folder** |
| `source_path` | The **resolved absolute path** of the source file (authoritative when the folder is a staging area of symlinks, as with `server_files`) |
| `converter_param_<key>` | Every converter param |
| `converter_out_index` / `converter_n_out` | This output's position in the returned list, and the list's length at import time |
| `converter_content_hash` | Short md5 of the output payload - the authoritative replay disambiguator |
| `parent_importer` | The importer that supplied the source corpus (`"demo"` for `apply_converter_to_demo`) |
| `parent_<key>` (`parent_path`, `parent_url`, `parent_paths_file`, `parent_manifest`, `parent_name`) | That importer's own locator param, prefixed, so the corpus itself is recoverable |

`vtscore.media.provenance` renders these as the human-readable
**Source** / **Derived Via** / **Imported Via** metadata lines.

---

## Gotchas

- **Converters don't produce vectors.** If you call
  `convert_normalized()` outside the runner, embedding the outputs is
  your job.
- **Heavy deps are imported lazily inside `convert()`**, and a missing
  one yields `[]` rather than an exception - install the relevant
  extras before relying on a converter. PyMuPDF is AGPL-3.0 and sits in
  the opt-out `agpl` extra; the `document2*` converters log the
  message from `vtscore.utils.optional_deps.agpl_unavailable_message`
  when it is absent.
- **Empty strings count as unset.** `normalize_params` drops `""` for
  any field that has a default or is optional, so a blank UI input
  falls back to the declared default.
- **`name` defaults to `f"{source}2{target}"`.** Two converters with
  the same source/target pair must override `name` on one of them, or
  one shadows the other in the registry.
- **`apply_converter_to_demo` mutates `medias` in place.** Snapshot
  first if you need the originals.

---

## Cross-references

- [media](media.md) - the `MediaType` / `MediaEmbedder` / `MediaClipper`
  ABCs and registry; converter outputs are media of those types.
- [datasets](datasets.md) - the import pipeline and embed stage that
  run converters and embed their outputs.
- [plugins](plugins.md) - `PluginField` / `PluginBase` / registry
  scaffolding shared by every plugin family.
- [`../extending/converters.md`](../extending/converters.md) and
  [`docs/EXTENDING-media.md`](../../../docs/EXTENDING-media.md#adding-a-media-converter) -
  writing a converter (library and app-side views).
