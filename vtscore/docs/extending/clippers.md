# Writing a `MediaClipper`

A media clipper splits a single media into one or more media of the
**same** type. The audio tiling clipper turns a 30-second recording
into fifteen 2-second segments; the text-sentence clipper splits a
paragraph into per-sentence clips; the scene clipper splits a video at
detected scene changes. Default clippers return the media unchanged
(one-element list) - they're the no-op that lets the framework
uniformly route everything through a clipper. Subclass
[`MediaClipper`](../../media/clipper.py)
([`vtscore/media/clipper.py`](../../media/clipper.py)), declare
which type it operates on, implement `clip()`, and add the instance to
the media-type sub-package's `CLIPPERS` list.

**App-side counterpart:** [`docs/EXTENDING-media.md § Adding a Media
Clipper`](../../../docs/EXTENDING-media.md#adding-a-media-clipper).
This guide focuses on the library API and the default-vs-tiling
convention.

## Contents

- [The contract](#the-contract)
- [Default clippers vs. tiling clippers](#default-clippers-vs-tiling-clippers)
- [Shared helpers](#shared-helpers)
- [The `CLIPPERS` list sentinel](#the-clippers-list-sentinel)
- [Parameters and re-parameterisation](#parameters-and-re-parameterisation)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## The contract

`MediaClipper` is an ABC. Required overrides:

| Member | Type | Purpose |
|--------|------|---------|
| `name` (property) | `str` | Unique registry key, with no parameter suffix (`"sound_tiling"`, `"video_scene"`) |
| `media_type` (property) | `str` | The `MediaType.type_id` this clipper operates on |
| `clip(media)` | `(dict) -> list[dict]` | Split one media into one or more media dicts of the same type |

Optional overrides:

| Member | Default | Purpose |
|--------|---------|---------|
| `display_name` | derived from `name` | Friendlier label for UI tabs and dropdowns |
| `description` | `""` | One-line tooltip surfaced on hover in the clipper chooser |
| `summary_template` | `description` | One-line preview with `{key}` placeholders filled from the current parameter values |
| `parameters` (property) | `[]` | List of `{key, label, type, default, min, max, step, description}` dicts describing user-configurable knobs |
| `creation_questions` (property) | `parameters` | Questions shown when the user first picks this clipper (defaults to `parameters`) |
| `with_params(params)` | returns `self` | Return a new clipper instance with overridden parameters |
| `resolve_for_durations(durations)` | returns `self` | **Reserved - never called.** Kept for compatibility; an override here is inert, so put the logic in `resolve_for_media` |
| `resolve_for_media(media)` | returns `self` | Per-media hook called per item - used by auto-selecting clippers (e.g. tile when long, pass-through when short) |

Each dict returned by `clip()` must preserve the original's structure
(`id`, `media_type`, `category`, `origin`, `origin_name`, …) and update only
the content fields (`media_bytes` / `media_string`, `duration`,
`file_size`, and type-specific extras like `width`, `height`,
`clip_index`, `clip_start`, `clip_end`). Default
clippers return `[media]` unchanged. Don't compute embeddings inside
`clip()`; the loader pipeline embeds the produced clips afterwards.

## Default clippers vs. tiling clippers

Every media type ships at least one **default** clipper whose `clip()`
returns `[media]` unchanged. The convention is
`<type>_default` for the name and "Import each X as-is, without
splitting." for the description. Existing default clippers:

| Default clipper | Media type |
|-----------------|------------|
| `SoundDefaultClipper` (`sound_default`) | audio |
| `ImageDefaultClipper` (`image_default`) | image |
| `TextDefaultClipper` (`text_default`) | text |
| `VideoDefaultClipper` (`video_default`) | video |
| `DocumentDefaultClipper` (`document_default`) | document |
| `FaceDefaultClipper` (`face_default`) | face |

Each of those is a four-line subclass of the concrete
`DefaultClipper` base (`vtscore/media/clipper.py`), which fixes the three
things every default clipper agrees on - the `"None"` display label, the
pass-through `clip()`, and the name/type/description trio passed to
`__init__`:

```python
from vtscore.media.clipper import DefaultClipper


class SoundDefaultClipper(DefaultClipper):
    """Returns the audio media unchanged."""

    def __init__(self) -> None:
        super().__init__("sound_default", "audio", "Import each audio file as-is, without splitting.")
```

Subclass it (rather than instantiating `DefaultClipper` directly) so the
media type keeps a named class the registry, the docs and out-of-tree
code can refer to.

**Tiling** clippers actually split. The convention is `<type>_tiling` -
`sound_tiling` for audio tiles, `video_tiling` for video tiles.

**The name carries no parameter suffix.** A parameterised clipper is
registered once, under a stable name; its window size / overlap / stride
live in the `parameters` descriptors and are re-bound per import through
`with_params()`. Two instances of the same class would collide in the
registry (it is keyed on `name`), so each media type's `CLIPPERS` list
holds exactly one instance per class, and the constructor arguments are
just that instance's *defaults*.

Scene-based or content-aware clippers get their own names -
`VideoSceneClipper` is `video_scene`, `SoundSilenceClipper` is
`sound_silence`. There's no "tiling" suffix when the split isn't
grid-based.

## Shared helpers

`vtscore/media/clipper.py` carries the pieces that would otherwise be
copied between clippers. Reach for them before hand-rolling an
equivalent - a second copy of the tiling arithmetic is exactly the kind
of drift that makes two media types tile differently.

| Helper | Purpose |
|--------|---------|
| `DefaultClipper` | Concrete base for the per-type no-op clipper (above) |
| `clip_with_bounds(media, index, start, end)` | Shallow-copy *media* and stamp `duration` / `clip_index` / `clip_start` / `clip_end`, rounded consistently. Add your own fields (`media_bytes`, `file_size`, `scene_index`, …) to the returned dict |
| `tile_starts(total, duration, min_overlap=0.0)` | Start times of equally-spaced tiles: first at 0, last ending at *total*, overlapping by at least *min_overlap* |
| `validate_tiling_params(duration, min_overlap)` | Constructor validation for a fixed-length tiling clipper (positive tile, non-negative overlap, overlap < tile) |
| `tiling_parameters(duration, min_overlap, *, item_label)` | The standard `duration` / `min_overlap` parameter descriptors, with *item_label* naming the unit in the help text |

Per-type emission helpers live next to their clippers rather than here,
because they touch type-specific bytes: `_emit_wav_segments` in
`vtscore/media/audio/clipper.py` slices WAV bytes for every audio clipper,
and `_emit_text_pieces` in `vtscore/media/text/clipper.py` rebuilds the
word/character counts for the paragraph and sentence clippers.

## The `CLIPPERS` list sentinel

Unlike the single-instance `MEDIA_TYPE` and `EMBEDDER` sentinels, the
clipper sentinel is a **list**. A single media-type sub-package
exports every clipper for that type in one list:

```python
# vtscore/media/audio/__init__.py
from vtscore.media.audio.clipper import (
    SoundDefaultClipper,
    SoundSilenceClipper,
    SoundSpeechActivityClipper,
    SoundTilingClipper,
)
from vtscore.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
CLIPPERS = [
    SoundTilingClipper(10.0, 1.0),   # constructor args are this instance's defaults
    SoundDefaultClipper(),
    SoundSilenceClipper(),
    SoundSpeechActivityClipper(),
]
```

The framework's discovery scan calls `register_clipper()` for every
entry, keyed by `clipper.name`. A clipper file can define multiple
`MediaClipper` classes; only those listed in `CLIPPERS` get registered.
List **one instance per class** - a second `SoundTilingClipper(5.0)`
would register under the same `sound_tiling` name and silently replace
the first. A user who wants 5-second tiles re-parameterises the single
registered instance through the clipper chooser, which calls
`with_params({"duration": 5.0})`.

The same module scan reads a sibling `CLEANERS` list into the separate
cleaner registry (`MediaCleaner` subclasses `MediaClipper` but never
appears in a clipper chooser).

## Parameters and re-parameterisation

Clippers exposing knobs declare them via the `parameters` property:

```python
@property
def parameters(self) -> list[dict[str, Any]]:
    return [
        {
            "key": "duration",
            "label": "Window (seconds)",
            "type": "number",
            "default": self._duration,
            "min": 0.5,
            "max": 60,
            "step": 0.5,
            "description": "Length of each tile in seconds.",
        },
    ]
```

When the user changes a value in the clipper chooser, the framework
calls `with_params({"duration": 3.0})` - return a **new instance**
with the updated value; never mutate `self`. The default implementation
returns `self` unchanged, which is correct for parameter-less
clippers.

The resolved clipper's `name` **and** the param dict the user chose are
both recorded in each clip's origin, as a
`{"kind": "clipper", "name": ..., "params": {...}}` chain step. The name
alone would not be enough, precisely because it doesn't encode the
parameter values; storing the two together is what makes cross-dataset
replay deterministic regardless of which parameterisation the user
picked.

Note also `summary_template` - a one-line preview with `{key}`
placeholders that the frontend fills from the current parameter values
when rendering the import row (e.g.
`"Cut each audio file into {duration}s tiles (min overlap
{min_overlap}s)."`). It defaults to `description`, so static clippers
don't need to override it.

## Worked example

A 50%-overlap audio tiling clipper. Drop it into
`vtscore/media/audio/clipper.py` (alongside the existing default and
tiling clippers) and add an instance to the audio package's
`CLIPPERS` list.

```python
# vtscore/media/audio/clipper.py  (new class added to the existing file)
from __future__ import annotations

import io
import wave
from typing import Any

from vtscore.media.clipper import MediaClipper


class SoundOverlapClipper(MediaClipper):
    """Tile audio with 50% overlap between consecutive segments."""

    def __init__(self, duration: float) -> None:
        if duration <= 0:
            raise ValueError("duration must be positive")
        self._duration = duration

    @property
    def name(self) -> str:
        # Stable across re-parameterisation: no "{duration}s" suffix.
        return "sound_overlap"

    @property
    def media_type(self) -> str:
        return "audio"

    @property
    def description(self) -> str:
        return "Tile audio with 50% overlap between consecutive segments."

    @property
    def parameters(self) -> list[dict[str, Any]]:
        return [{
            "key": "duration", "label": "Window (seconds)",
            "type": "number", "default": self._duration,
            "min": 0.5, "max": 60, "step": 0.5,
            "description": "Length of each overlapping tile in seconds.",
        }]

    def with_params(self, params: dict[str, Any]) -> "SoundOverlapClipper":
        return SoundOverlapClipper(float(params.get("duration", self._duration)))

    def clip(self, media: dict[str, Any]) -> list[dict[str, Any]]:
        wav_bytes = media.get("media_bytes")
        if wav_bytes is None:
            return [media]

        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            total_frames = wf.getnframes()
            n_ch = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            full_pcm = wf.readframes(total_frames)

        window = self._duration
        step = window / 2  # 50% overlap
        total_s = total_frames / sr

        # Short clips pass through unchanged.
        if total_s <= window:
            return [media]

        clips = []
        t = 0.0
        while t + window <= total_s + 1e-9:
            start_f = int(t * sr)
            end_f = int((t + window) * sr)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as out:
                out.setnchannels(n_ch)
                out.setsampwidth(sampwidth)
                out.setframerate(sr)
                bps = n_ch * sampwidth
                out.writeframes(full_pcm[start_f * bps:end_f * bps])
            sliced = buf.getvalue()
            clips.append({
                # preserves id / media_type / category / origin / origin_name
                **media,
                "media_bytes": sliced,
                "duration": window,
                "file_size": len(sliced),
                # The audio media type reads these back to serve a window of
                # the parent file; "start"/"end" would be ignored.
                "clip_index": len(clips),
                "clip_start": round(t, 6),
                "clip_end": round(t + window, 6),
            })
            t += step
        return clips
```

Register it by appending to the package's `CLIPPERS` list:

```python
# vtscore/media/audio/__init__.py
CLIPPERS = [
    SoundTilingClipper(10.0, 1.0),
    SoundDefaultClipper(),
    SoundSilenceClipper(),
    SoundSpeechActivityClipper(),
    SoundOverlapClipper(2.0),  # new
]
```

The next import of `vtscore.media.audio` registers the new clipper
under the name `sound_overlap`; the dataset-loader pipeline routes
audio files through it when the user picks that clipper in the
chooser, and each produced clip carries an `origin` plus a 50%-overlap
slice of the original WAV.

## Testing pattern

Clipper tests live in `tests_lib/core/` (registration + behaviour) or
`tests_lib/detectors/test_clipper_chain.py`. The autouse
`reset_contexts` fixture resets all contexts between tests.

```python
# tests_lib/core/test_sound_overlap_clipper.py
import io
import wave

from vtscore.media import clippers_for_type, get_clipper


def _make_wav(duration_s: float, sr: int = 16_000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(duration_s * sr))
    return buf.getvalue()


class TestSoundOverlapClipper:
    def test_is_registered(self):
        names = [c.name for c in clippers_for_type("audio")]
        assert "sound_overlap" in names

    def test_clip_returns_overlapping_windows(self):
        cl = get_clipper("sound_overlap")
        media = {
            "id": 1, "media_type": "audio", "category": "",
            "filename": "x.wav", "origin": None, "origin_name": "x.wav",
            "media_bytes": _make_wav(5.0),
        }
        clips = cl.clip(media)
        # 5s with 2s window, 1s step → starts at 0, 1, 2, 3 → 4 clips
        assert len(clips) == 4
        assert clips[0]["clip_start"] == 0.0
        assert clips[1]["clip_start"] == 1.0

    def test_short_clip_passes_through(self):
        cl = get_clipper("sound_overlap")
        media = {
            "id": 1, "media_type": "audio", "category": "",
            "filename": "x.wav", "origin": None, "origin_name": "x.wav",
            "media_bytes": _make_wav(1.0),
        }
        assert cl.clip(media) == [media]

    def test_with_params_returns_new_instance(self):
        cl = get_clipper("sound_overlap")
        re = cl.with_params({"duration": 4.0})
        assert re is not cl                    # a new instance, not a mutation
        assert re.name == cl.name              # name is stable; params are not in it
        assert re.parameters[0]["default"] == 4.0
        assert cl.parameters[0]["default"] == 2.0  # original unchanged
```

See [`tests_lib/detectors/test_clipper_chain.py`](../../../tests_lib/detectors/test_clipper_chain.py)
for end-to-end load-pipeline tests that route media through a clipper
and verify the produced clips end up in `medias` with the right
origins.
