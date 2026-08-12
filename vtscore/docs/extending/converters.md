# Writing a `MediaConverter`

A media converter transforms a media of one type into one or more
media of a **different** type: video → image (sampled frames), audio →
image (spectrogram), audio → text (ASR), image → text (OCR), document
→ image (rendered pages). Converters unlock cross-format access -
running an image embedder over audio via a spectrogram, surfacing
searchable text behind a scanned document - without requiring new
embedders. Subclass [`MediaConverter`](../../converters/base.py),
implement `source_type`, `target_type`, and
`convert(media, params=None)`, optionally
declare user-configurable parameters via `fields`, and expose a
module-level `CONVERTER` sentinel.

**App-side counterpart:** [`docs/EXTENDING-media.md § Adding a Media
Converter`](../../../docs/EXTENDING-media.md#adding-a-media-converter)
for how converters compose with importers. This guide focuses on the
library API and third-party packaging.

## Contents

- [When to write a converter](#when-to-write-a-converter)
- [The contract](#the-contract)
- [Where the file goes](#where-the-file-goes)
- [Parameters](#parameters)
- [Entry-point registration](#entry-point-registration)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## When to write a converter

Write a converter when an existing media type already has a good
embedder for the **target** representation and you want to feed
something else through it. Examples that ship in the box:

| Built-in | Source → Target | Why it's useful |
|----------|----------------|-----------------|
| `audio2image` | audio → image | Apply SigLIP / DINOv3 over spectrograms; visualise audio patterns |
| `audio2text` | audio → text | ASR (Whisper) so audio is searchable by text content |
| `image2text` | image → text | OCR so scanned text is searchable |
| `video2image` | video → image | Sample frames as separate images — a fixed count per video, or one frame every N seconds |
| `video2audio` | video → audio | Extract the audio track for CLAP embedding |
| `document2image` | document → image | Render PDF pages as searchable images |
| `document2text` | document → text | Extract embedded text from PDFs |

If you instead want to add a new content kind (point clouds, 3D
meshes), write a [media type](media-types.md). If you want a different
encoder for the same kind, write an [embedder](embedders.md).

## The contract

`MediaConverter` is an ABC. Required overrides:

| Member | Type | Purpose |
|--------|------|---------|
| `source_type` (property) | `str` | The `type_id` of the input media type |
| `target_type` (property) | `str` | The `type_id` of the output media type |
| `convert(media, params=None)` | `(dict, dict\|None) -> list[dict]` | Convert one media into one or more target-type dicts |

**Both parameters are part of the signature.** The abstract method is
`convert(self, media, params=None)`; a subclass that declares
`convert(self, media)` raises `TypeError` on the first conversion,
because every framework call site passes a populated `params` dict.

`name` is auto-derived as `f"{source_type}2{target_type}"` - don't
override it. Two converters with the same source / target would clash;
keep the pairing unique.

Optional overrides:

| Member | Default | Purpose |
|--------|---------|---------|
| `display_name` | derived | Human-readable label for the chooser UI |
| `description` | `""` | One-line description |
| `summary_template` | `""` | One-line import-row preview with `{key}` placeholders for parameter values (falls back to `description`) |
| `fields` | `[]` | User-configurable parameters |

**Don't override `convert_normalized()`** - it's the framework entry
point that wraps `convert()` (see below).

### What `convert()` returns

A list of dicts (empty list if conversion fails or produces no
output). Each dict must contain at minimum:

- `"filename"` - a descriptive name (e.g. `f"{source_stem}_page1.png"`);
- the data fields expected by the target media type (`media_bytes` +
  `duration` for image/audio/video; `media_string` for text);
- any extras the target media type's `pickle_extra_fields` declares
  (e.g. `width`, `height` for images).

`convert()` does **not** populate `id`, `embeddings`, or `md5` - those
are filled in by the caller. Don't compute embeddings inside
`convert()`; the loader pipeline embeds the produced media afterwards.

### Reading the source bytes

Read the input through `resolve_media_bytes(media)`
([`vtscore/converters/base.py`](../../converters/base.py)), not
`media["media_bytes"]`. In full-import mode the source dict carries
inline bytes, but a reference (*thin*) import hands the converter only
`{filename, media_path}` - so a converter that reads `media_bytes`
alone silently produces nothing for every thin import. The helper falls
back to reading `media_path` and returns `None` when neither yields
bytes.

### Reading parameters

User-supplied parameters arrive in the `params` dict, and by the time
`convert()` runs the framework has already validated and default-filled
it. Every in-tree call site (importer multi-media ingestion, the
converter-folder runner, the clipper-chain runner) goes through
`convert_normalized(media, params)`, which loads `params` through the
per-plugin marshmallow schema built from your `fields` - enforcing
declared `min` / `max` ranges and `select` whitelists, raising
`ValueError` on a bad value - and then fills every missing or
empty-string key with that field's `default`.

So inside `convert()` you can index `params[key]` directly: it is a
non-`None` dict in which every declared field key is present.

`self.get_param(params, key)` remains as a thin shim for third-party
call sites that invoke `convert()` by hand rather than
`convert_normalized()`; it does the same default fallback for a
`params` that may be `None` or partial. New code should prefer
`convert_normalized()` at the call site and plain indexing in the body.

Note that the converter `params` path is **not** the `field_values`
normalization pass described in the [plugin
README](README.md#framework-side-normalization): converter params are
shape-checked against the schema, but nothing strips whitespace,
substitutes `template_vars`, or runs the URL / server-path guards. A
converter that accepts a URL or a server path must validate it itself.

Values from web requests still arrive as strings for `text` fields, so
coerce explicitly (`int(value)`, `float(value)`) where the schema
doesn't already do it for you (`number` fields load as `int` / `float`).

## Where the file goes

In-tree, converters are **flat modules** under `vtscore.converters/`:

```
vtscore/converters/<source>2<target>.py
```

The registry ([`vtscore/converters/__init__.py`](../../converters/__init__.py))
scans this directory for `.py` files (excluding `__init__.py` and
`base.py`) and registers any module-level `CONVERTER` sentinel. No
`__init__.py` edits are needed for the discovery itself - it's
filesystem-driven - but the package's existing `__init__.py` imports
all built-in converter classes for legacy `from vtscore.converters
import Audio2ImageMediaConverter` users.

Out-of-tree, ship a Python distribution with a `vtscore.converters`
entry point (see below).

## Parameters

Converters declare user-configurable knobs the same way every other
plugin family does - a list of `PluginField`s
([`vtscore/plugins/__init__.py`](../../plugins/__init__.py)) on the
class. The audio→image spectrogram converter is a good reference for
the spread of field types (select, number with min/max/step) it can
accept; see [`vtscore/converters/audio2image.py`](../../converters/audio2image.py).

Field values flow into `convert(media, params)` as a flat dict keyed by
`PluginField.key`. The multi-media import flow stores per-row
`params` alongside each `SourceSpec`, so the same converter can run
twice in one dataset with different settings (e.g. a `video2image`
that samples 30 frames per video on one row and 5 frames on another).

## Entry-point registration

Out-of-tree converters declare a `vtscore.converters` entry point that
resolves to the instantiated `CONVERTER` sentinel:

```toml
[project]
name = "vtsearch-my-converter"
version = "0.1.0"
dependencies = ["vtsearch"]

[project.entry-points."vtscore.converters"]
audio2text_whisperx = "my_pkg.audio2text_whisperx:CONVERTER"
```

Notice the entry-point **name** can differ from the converter's
auto-derived `name`. The registry uses the converter's
`source_type`/`target_type` derived name as the registry key, so a
third-party `audio2text` converter will clash with the built-in
`audio2text`. Pick a unique source/target pair or contribute the
improvement upstream.

## Worked example

A minimal `audio2text` alternative using OpenAI Whisper for ASR (the
in-tree `audio2text` uses faster-whisper; this third-party one wraps
the official Whisper Python package). It declares two parameters:
model size and language.

```python
# my_pkg/audio2text_whisper.py
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from vtscore.converters.base import MediaConverter, resolve_media_bytes
from vtscore.plugins import PluginField


class Audio2TextWhisperConverter(MediaConverter):
    """Transcribe audio to text using OpenAI Whisper."""

    display_name = "Audio → Text (Whisper)"
    description = "Run OpenAI Whisper ASR on audio files."
    fields = [
        PluginField(
            key="model_size",
            label="Whisper model size",
            field_type="select",
            options=["tiny", "base", "small", "medium", "large"],
            default="base",
            description="Larger models are slower but more accurate.",
            required=False,
        ),
        PluginField(
            key="language",
            label="Language code",
            field_type="text",
            default="",
            placeholder="auto-detect",
            description="ISO 639-1 code (e.g. 'en', 'es'); blank = auto-detect.",
            required=False,
        ),
    ]

    @property
    def source_type(self) -> str:
        return "audio"

    @property
    def target_type(self) -> str:
        return "text"

    def convert(self, media: dict[str, Any], params: dict[str, Any] | None = None) -> list[dict]:
        # convert_normalized() default-filled every declared key, so both
        # of these are present. "language" defaults to "" (auto-detect).
        model_size = params["model_size"]
        language = params["language"] or None

        # Works for thin imports too, where only media_path is set.
        media_bytes = resolve_media_bytes(media)
        if not media_bytes:
            return []

        try:
            import whisper  # noqa: PLC0415
        except ImportError:
            return []

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(media_bytes)
            tmp_path = tmp.name
        try:
            model = whisper.load_model(model_size)
            result = model.transcribe(tmp_path, language=language)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        text = (result.get("text") or "").strip()
        if not text:
            return []

        stem = Path(media.get("filename", "audio.wav")).stem
        return [{
            "filename": f"{stem}_transcript.txt",
            "media_string": text,
            "duration": 0,
        }]


CONVERTER = Audio2TextWhisperConverter()
```

The `pyproject.toml`:

```toml
[project.entry-points."vtscore.converters"]
audio2text_whisper = "my_pkg.audio2text_whisper:CONVERTER"
```

This name collides with the built-in `audio2text` - to ship both, give
the third-party one a distinct source/target pair (e.g. add an
`audio2text_whisper` source-type alias) or contribute the model-size
parameter upstream.

## Testing pattern

Converter tests live in `tests/converters/` (app-tier) and
`tests_lib/` doesn't currently have a converters folder - most
converter tests are app-tier because they exercise the
post-load-conversion pipeline. For library-only smoke tests, drop a
file in `tests_lib/core/` or `tests_lib/datasets/`:

```python
# tests_lib/core/test_audio2text_whisper.py
import io
import wave
import pytest

from vtscore.converters import get_converter, list_converters_for_source


def _make_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\x00\x00" * 16_000)
    return buf.getvalue()


class TestAudio2TextWhisperRegistration:
    def test_is_discoverable(self):
        names = [c.name for c in list_converters_for_source("audio")]
        # Built-in or our entry-point; assert ours under the chosen name
        assert "audio2text" in names or "audio2text_whisper" in names

    def test_fields_have_defaults(self):
        conv = get_converter("audio2text")  # or your unique name
        keys = {f.key for f in conv.fields}
        assert "model_size" in keys


class TestAudio2TextWhisperConvert:
    @pytest.mark.skipif(
        not _has_module("whisper"),
        reason="openai-whisper not installed",
    )
    def test_produces_text_or_empty_list(self):
        from my_pkg.audio2text_whisper import Audio2TextWhisperConverter

        conv = Audio2TextWhisperConverter()
        media = {"filename": "a.wav", "media_bytes": _make_wav()}
        out = conv.convert(media, {"model_size": "tiny"})
        # Silent WAV produces no transcript; non-silent would produce one dict
        assert isinstance(out, list)


def _has_module(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None
```

The built-in [`tests/converters/`](../../../tests/converters/) tests
are the long-form reference - they wire converters into a full dataset
load and assert the resulting media types end up correctly in
`medias`.
