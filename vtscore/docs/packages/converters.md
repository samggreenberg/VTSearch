# `vtscore.converters`

A *converter* turns a media dict of one type into one or more media
dicts of a **different** type. They're how you embed audio with an
image model (audio → spectrogram → SigLIP), search OCR'd documents
with a text model (image → text → E5), or run a video through a
keyframe-image pipeline. The package ships seven concrete
converters, all auto-discovered via the `CONVERTER` sentinel, plus a
runner (`run_converters_on_folder`) that wires conversion into the
dataset-import path. Adding a new converter is a one-file change.

---

## When to use a converter

A converter is the right tool when the embedder you want to apply
isn't directly compatible with the source format. Three common
shapes:

| You have      | You want                                        | Use                                  |
|---------------|-------------------------------------------------|--------------------------------------|
| Audio files   | Embed them with an image model (SigLIP, DINOv3) | `audio2image` (mel-spectrogram)       |
| Image files   | Search them by transcribed text                 | `image2text` (OCR)                    |
| Video files   | Embed individual frames with an image model     | `video2image` (keyframe extraction)   |
| Video files   | Embed the audio track with an audio model       | `video2audio`                         |
| Audio files   | Transcribe speech and embed with a text model   | `audio2text` (Whisper ASR)            |
| Document files (PDF) | Run an image embedder over each page     | `document2image`                      |
| Document files | Embed extracted body text                      | `document2text`                       |

Converters are not embedders - they don't produce vectors. They
produce media dicts, which then get embedded by the **target** media
type's embedder. The runner handles that handoff.

---

## `MediaConverter` ABC

`vtscore/converters/base.py:11`. A `PluginBase` subclass - same
field-driven configuration system every other plugin family uses.

```python
class MediaConverter(PluginBase, ABC):
    display_name: str = ""
    description: str = ""
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
    def convert(
        self,
        media: dict,
        params: dict | None = None,
    ) -> list[dict]: ...
```

The implementation contract:

- `convert(media, params)` returns a **list** of new media dicts.
  Empty list means "skipped - could not convert" (e.g. an empty
  document, an audio decode failure). The caller treats empty
  lists as "no output for this source".
- Each returned dict must contain at minimum `"filename"` and the
  data fields expected by the target media type's
  `load_media_data` (e.g. `"media_bytes"` + `"duration"` for image,
  `"media_string"` for text). The dict does **not** include `"id"`,
  `"embedding"`, or `"md5"` - the runner assigns those.
- `params` follows the same shape every plugin family uses:
  `{field.key: value}`. The default is `None`, meaning "use declared
  defaults". Implementations should always read params through
  `self.get_param(params, key)` (`vtscore/converters/base.py:95`)
  so missing or empty values fall back to `field.default`.

### Declaring parameters

Converters declare user-tunable knobs via `fields`:

```python
from vtscore.plugins import PluginField

class Video2ImageMediaConverter(MediaConverter):
    fields = [
        PluginField(
            key="n_clips",
            label="Frames per video",
            field_type="number",
            description="Number of evenly-spaced frames to extract.",
            default="10",
            required=False,
            min="1",
            max="1000",
            step="1",
        ),
    ]
```

(`vtscore/converters/video2image.py:29`–`41`.) The frontend reads
these fields off `converter.to_dict()` and renders matching inputs.

---

## Built-in converters

Seven ship in-tree, all in `vtscore/converters/`. Each module ends
with `CONVERTER = MyConverter()`, which the registry picks up
automatically.

| Module                                   | Class                            | Source → Target | Notes                                         |
|------------------------------------------|----------------------------------|-----------------|-----------------------------------------------|
| `vtscore/converters/audio2image.py`      | `Audio2ImageMediaConverter`      | audio → image   | Mel-spectrogram or CQT PNG via librosa + matplotlib. Configurable: `spectrogram_type`, `n_mels`, `time_window_s`, `colormap`. |
| `vtscore/converters/audio2text.py`       | `Audio2TextMediaConverter`       | audio → text    | Whisper ASR (HF `openai/whisper-*`).          |
| `vtscore/converters/video2image.py`      | `Video2ImageMediaConverter`      | video → image   | OpenCV keyframe extraction. Configurable: `n_clips`. |
| `vtscore/converters/video2audio.py`      | `Video2AudioMediaConverter`      | video → audio   | FFmpeg-backed audio track demux.              |
| `vtscore/converters/document2image.py`   | `Document2ImageMediaConverter`   | document → image | Render PDF pages to PNG (PyMuPDF).           |
| `vtscore/converters/document2text.py`    | `Document2TextMediaConverter`    | document → text | Extract embedded text from documents.         |
| `vtscore/converters/image2text.py`       | `Image2TextMediaConverter`       | image → text    | OCR via Tesseract.                            |

Each is importable directly:

```python
from vtscore.converters import (
    Audio2ImageMediaConverter,
    Audio2TextMediaConverter,
    Document2ImageMediaConverter,
    Document2TextMediaConverter,
    Image2TextMediaConverter,
    Video2AudioMediaConverter,
    Video2ImageMediaConverter,
)
```

Or via the registry:

```python
from vtscore.converters import get_converter
v2i = get_converter("video2image")
outputs = v2i.convert(media_dict, {"n_clips": "20"})
```

---

## Registry & discovery

`vtscore/converters/__init__.py:25`. The registry is a
`PluginRegistry[MediaConverter]` built on the standard discovery
machinery - exactly like importers, exporters, settings sources, etc.

```python
_registry: PluginRegistry[MediaConverter] = PluginRegistry(
    package="vtscore.converters",
    sentinel="CONVERTER",
    label="media converter",
    discover_modules=True,
    entry_point_group="vtscore.converters",
)
```

The registry scans `vtscore.converters` for modules that expose a
module-level `CONVERTER` attribute, and also imports anything
registered under the `vtscore.converters` Python entry-point group.
Built-ins win on name clashes; broken third-party entries warn and
are skipped.

### Public accessors

| Function (`vtscore/converters/__init__.py`)                | Purpose                                       |
|------------------------------------------------------------|-----------------------------------------------|
| `list_converters()` (`:34`)                                | Every registered converter.                   |
| `get_converter(name)` (`:39`)                              | Look up by `name`; returns `None` on miss.    |
| `list_converters_for_target(target_type)` (`:44`)          | All converters producing `target_type`.       |
| `list_converters_for_source(source_type)` (`:49`)          | All converters consuming `source_type`.       |

`list_converters_for_target("image")` is the typical query for "give
me every way to produce an image, regardless of source". This is
what dataset importers call to populate their "convert via..."
dropdowns.

```python
from vtscore.converters import list_converters_for_target

for c in list_converters_for_target("image"):
    print(c.name, "<-", c.source_type)
# audio2image <- audio
# document2image <- document
# video2image <- video
```

---

## The `CONVERTER` sentinel

Every concrete converter module ends with a module-level
`CONVERTER = MyConverter()` assignment:

```python
# vtscore/converters/audio2image.py
class Audio2ImageMediaConverter(MediaConverter):
    display_name = "Audio → Image (spectrogram)"
    description = "Render audio as a mel-spectrogram or CQT image"
    fields = [...]
    @property
    def source_type(self) -> str: return "audio"
    @property
    def target_type(self) -> str: return "image"
    def convert(self, media, params=None): ...

CONVERTER = Audio2ImageMediaConverter()
```

(`vtscore/converters/audio2image.py:261`.) That's the only
boilerplate needed for discovery. Out-of-tree converters can either:

1. Drop the module into `vtscore/converters/` (or symlink it there).
2. Register via the `vtscore.converters` entry-point group in
   `pyproject.toml`:

```toml
[project.entry-points."vtscore.converters"]
my_converter = "my_pkg.my_converter:CONVERTER"
```

The registry's `eager=True` (default) means discovery happens at
import time, so by the time `list_converters()` returns, every
in-tree and entry-point converter is already known.

---

## Running a converter - the runner

`vtscore/converters/runner.py:213`. Most callers don't invoke
`converter.convert(...)` directly - they call
`run_converters_on_folder(...)`, which:

1. Scans a folder for files matching the **source** media type's
   extensions.
2. Calls `converter.convert(source_media, params)` on each match.
3. Resolves the **target** media type's default embedder and
   embeds each converted output.
4. Assigns sequential IDs starting after the current max in `medias`.
5. Records an `origin` of `{"importer": "converter", "params": {...}}`
   so the result is replayable.

```python
def run_converters_on_folder(
    folder_path: Path,
    converter_names: list[str] | None = None,
    target_media_type: str = "",
    medias: dict[int, dict] | None = None,
    thin: bool = False,
    on_progress: ProgressCallback | None = None,
    base_origin: dict | None = None,
    recursive: bool = True,
    converter_specs: list | None = None,
) -> None: ...
```

Two entry shapes:

- **`converter_names`** - legacy, names only. Each converter runs
  with its declared field defaults.
- **`converter_specs`** - multi-media path. A list of
  `SourceSpec(converter, params, ...)` (or equivalent dicts) so the
  caller can pass per-converter params resolved from a UI form. This
  is what multi-media importers (`server_folder`, `server_files`,
  `local_folder`, `local_files`) use after the multi-media-import
  refactor.

```python
from pathlib import Path
from vtscore.converters.runner import run_converters_on_folder

medias: dict[int, dict] = {}
run_converters_on_folder(
    folder_path=Path("/data/recordings"),
    converter_names=["audio2image"],
    target_media_type="image",
    medias=medias,
    base_origin={"importer": "server_folder", "params": {"path": "/data/recordings"}},
)
# `medias` is now populated with spectrograms embedded via the default image embedder.
```

The runner also exposes `apply_converter_to_demo` (`runner.py:395`)
for the demo-dataset case: convert every existing media in a dict
in-place (replacing it with the converted outputs).

---

## How `convert()` outputs flow into media dicts

The runner builds each output media dict via
`_build_converted_media_dict` (`vtscore/converters/runner.py:123`):

```python
{
    "id": <assigned by runner>,
    "media_type": <converter.target_type>,
    "embedder": <name of target media type's default embedder>,
    "file_size": <len of media_bytes or media_string.encode()>,
    "md5": <hashlib.md5 of bytes/string>,
    "embedding": <vector from target_emb.embed_media(...)>,
    "filename": <converter output's filename>,
    "category": "custom",
    "origin": {"importer": "converter", "params": {...}},
    "origin_name": f"{source_rel}→{output_filename}",
    "media_bytes": <if produced>,
    "media_string": <if produced>,
    "media_path": <source path>,
    "duration": <output.get("duration", 0)>,
    # plus any optional fields the target type expects:
    #   "width", "height", "word_count", "character_count"
}
```

`origin` is the canonical persisted form (CLAUDE.md "No Persisted
Vectors"). The recorded `params` include the converter name, the
source filename (relative to the import root), every
user-supplied param (prefixed `converter_param_`), and the parent
importer + parent path/url so the full provenance chain is
recoverable.

---

## Implementing a new converter

Sketch - the walk-through is in
[../../docs/EXTENDING-plugins.md#adding-a-media-converter](../../../docs/EXTENDING-plugins.md#adding-a-media-converter).

1. Create `vtscore/converters/<source>2<target>.py`.
2. Subclass `MediaConverter`. Implement `source_type`,
   `target_type`, and `convert(media, params)`.
3. Declare `display_name`, `description`, and any
   `fields` you need.
4. At the bottom of the module, expose `CONVERTER = MyConverter()`.
5. Restart the process - discovery picks it up on next import.

A minimal example:

```python
from typing import Any
from vtscore.converters.base import MediaConverter
from vtscore.plugins import PluginField


class Text2EmojiMediaConverter(MediaConverter):
    display_name = "Text → Emoji"
    description = "Replace common words with their emoji equivalents."
    fields = [
        PluginField(
            key="lang",
            label="Language",
            field_type="select",
            options=["en", "es", "fr"],
            default="en",
        ),
    ]

    @property
    def source_type(self) -> str:
        return "text"

    @property
    def target_type(self) -> str:
        return "text"   # same target type is allowed

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        lang = self.get_param(params, "lang")
        text = media.get("media_string", "")
        if not text:
            return []
        emojified = _emojify(text, lang)
        return [{
            "filename": (media.get("filename") or "output") + ".emoji.txt",
            "media_string": emojified,
            "duration": 0,
        }]


CONVERTER = Text2EmojiMediaConverter()
```

---

## Gotchas

- **Converter output is embedded by the target's *default* embedder.**
  `run_converters_on_folder` calls `embedders_for_type(target_type)[0]`
  (`vtscore/converters/runner.py:82`). To embed with a non-default
  target embedder, set it as default in the registry or call
  `_emit_converted_outputs` yourself with a hand-resolved embedder.
- **Converters don't produce vectors.** They produce media dicts.
  The runner does the embedding pass. If you call `convert()`
  directly without the runner, you're responsible for embedding the
  outputs.
- **Heavy deps are imported lazily inside `convert()`.** Most
  converters depend on third-party packages
  (`librosa`+`matplotlib` for `audio2image`, `cv2` for
  `video2image`, `pytesseract` for `image2text`, `pymupdf` for
  `document2*`, `openai-whisper` for `audio2text`). Each does the
  import inside `convert` and returns `[]` on `ImportError` rather
  than crashing - install the relevant extras before relying on a
  converter.
- **Temporary files.** Converters that need a file path (e.g.
  `audio2image` decoding via librosa) write `media_bytes` to a
  `tempfile.NamedTemporaryFile`, run the operation, and `unlink` the
  file in a `finally`. The runner's `_embed_converted_output`
  (`vtscore/converters/runner.py:325`) does the same thing for the
  embedding pass. No persisted intermediates.
- **`get_param` treats empty strings as unset.** A UI that submits
  empty inputs gets the field's `default`, not `""`. This is
  intentional - fall through to the declared default when the user
  doesn't touch the field.
- **`name` defaults to `f"{source}2{target}"`.** If two converters
  share the same source/target pair, override `name` on one of them
  or discovery will silently shadow the duplicate.
- **`apply_converter_to_demo` mutates `medias` in place** -
  `medias.clear()` followed by `medias.update(converted)`
  (`runner.py:462`). Callers that need the original around must
  snapshot first.

---

## Cross-references

- [media](media.md) - the `MediaType` / `MediaEmbedder` / `MediaClipper`
  ABCs and registry; converters write into media dicts produced by
  those types.
- [embedding](embedding.md) - the embedder façade the runner uses
  to vectorise converter outputs.
- [plugins](plugins.md) - the `PluginField` / `PluginBase` /
  `PluginRegistry` scaffolding converters share with every other
  plugin family.
- [../../docs/EXTENDING-plugins.md](../../../docs/EXTENDING-plugins.md) -
  the full walkthrough for adding a converter.
