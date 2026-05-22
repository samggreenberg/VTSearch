# Writing a `MediaClipper`

A media clipper splits a single media into one or more media of the
**same** type. The audio tiling clipper turns a 30-second recording
into fifteen 2-second segments; the text-sentence clipper splits a
paragraph into per-sentence clips; the scene clipper splits a video at
detected scene changes. Default clippers return the media unchanged
(one-element list) — they're the no-op that lets the framework
uniformly route everything through a clipper. Subclass
[`MediaClipper`](../../media/clipper.py)
([`vtscore/media/clipper.py:9`](../../media/clipper.py)), declare
which type it operates on, implement `clip()`, and add the instance to
the media-type sub-package's `CLIPPERS` list.

**App-side counterpart:** [`docs/EXTENDING-media.md § Adding a Media
Clipper`](../../../docs/EXTENDING-media.md#adding-a-media-clipper).
This guide focuses on the library API and the default-vs-tiling
convention.

## Contents

- [The contract](#the-contract)
- [Default clippers vs. tiling clippers](#default-clippers-vs-tiling-clippers)
- [The `CLIPPERS` list sentinel](#the-clippers-list-sentinel)
- [Parameters and re-parameterisation](#parameters-and-re-parameterisation)
- [Worked example](#worked-example)
- [Testing pattern](#testing-pattern)

## The contract

`MediaClipper` is an ABC. Required overrides:

| Member | Type | Purpose |
|--------|------|---------|
| `name` (property) | `str` | Unique registry key (`"sound_tiling_2.0s"`, `"video_scene"`) |
| `media_type` (property) | `str` | The `MediaType.type_id` this clipper operates on |
| `clip(media)` | `(dict) -> list[dict]` | Split one media into one or more media dicts of the same type |

Optional overrides:

| Member | Default | Purpose |
|--------|---------|---------|
| `display_name` | derived from `name` | Friendlier label for UI tabs and dropdowns |
| `description` | `""` | One-line tooltip surfaced on hover in the clipper chooser |
| `parameters` (property) | `[]` | List of `{key, label, type, default, min, max, step, description}` dicts describing user-configurable knobs |
| `creation_questions` (property) | `parameters` | Questions shown when the user first picks this clipper (defaults to `parameters`) |
| `with_params(params)` | returns `self` | Return a new clipper instance with overridden parameters |
| `resolve_for_durations(durations)` | returns `self` | Per-dataset hook called once at load time |
| `resolve_for_media(media)` | returns `self` | Per-media hook called per item — used by auto-selecting clippers (e.g. tile when long, pass-through when short) |

Each dict returned by `clip()` must preserve the original's structure
(`id`, `type`, `category`, `origin`, `origin_name`, …) and update only
the content fields (`media_bytes` / `media_string`, `duration`, and
type-specific extras like `width`, `height`, `start`, `end`). Default
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

**Tiling** clippers actually split. The convention is
`<type>_tiling[_<param>]` — e.g. `sound_tiling_2.0s` for 2-second
audio tiles, `video_tiling_2.0s` for 2-second video tiles. They are
usually parameterised (window size, overlap, stride) so the same class
ships multiple registered instances with different param values.

Scene-based or content-aware clippers get their own names —
`VideoSceneClipper` is `video_scene`. There's no "tiling" suffix when
the split isn't grid-based.

## The `CLIPPERS` list sentinel

Unlike the single-instance `MEDIA_TYPE` and `EMBEDDER` sentinels, the
clipper sentinel is a **list**. A single media-type sub-package
exports every clipper for that type in one list:

```python
# vtscore/media/audio/__init__.py
from vtscore.media.audio.clipper import (
    SoundDefaultClipper,
    SoundTilingClipper,
)
from vtscore.media.audio.media_type import AudioMediaType

MEDIA_TYPE = AudioMediaType()
CLIPPERS = [
    SoundDefaultClipper(),
    SoundTilingClipper(2.0),
    SoundTilingClipper(5.0),
]
```

The framework's discovery scan calls `register_clipper()` for every
entry, keyed by `clipper.name`. A clipper file can define multiple
`MediaClipper` classes; only those listed in `CLIPPERS` get registered.

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
calls `with_params({"duration": 3.0})` — return a **new instance**
with the updated value; never mutate `self`. The default implementation
returns `self` unchanged, which is correct for parameter-less
clippers.

The resolved clipper's `name` (which usually includes the param value,
e.g. `sound_tiling_3.0s`) is what gets recorded in each clip's origin,
so cross-dataset replay is deterministic regardless of which
parameterisation the user picked.

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
        return f"sound_overlap_{self._duration}s"

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
            clips.append({
                **media,  # preserves id/type/category/origin/origin_name
                "media_bytes": buf.getvalue(),
                "duration": window,
                "start": t,
                "end": t + window,
            })
            t += step
        return clips
```

Register it by appending to the package's `CLIPPERS` list:

```python
# vtscore/media/audio/__init__.py
CLIPPERS = [
    SoundDefaultClipper(),
    SoundTilingClipper(2.0),
    SoundOverlapClipper(2.0),  # new
]
```

The next import of `vtscore.media.audio` registers the new clipper
under the name `sound_overlap_2.0s`; the dataset-loader pipeline routes
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
        assert "sound_overlap_2.0s" in names

    def test_clip_returns_overlapping_windows(self):
        cl = get_clipper("sound_overlap_2.0s")
        media = {
            "id": 1, "media_type": "audio", "category": "",
            "filename": "x.wav", "origin": None, "origin_name": "x.wav",
            "media_bytes": _make_wav(5.0),
        }
        clips = cl.clip(media)
        # 5s with 2s window, 1s step → starts at 0, 1, 2, 3 → 4 clips
        assert len(clips) == 4
        assert clips[0]["start"] == 0.0
        assert clips[1]["start"] == 1.0

    def test_short_clip_passes_through(self):
        cl = get_clipper("sound_overlap_2.0s")
        media = {
            "id": 1, "media_type": "audio", "category": "",
            "filename": "x.wav", "origin": None, "origin_name": "x.wav",
            "media_bytes": _make_wav(1.0),
        }
        assert cl.clip(media) == [media]

    def test_with_params_returns_new_instance(self):
        cl = get_clipper("sound_overlap_2.0s")
        re = cl.with_params({"duration": 4.0})
        assert re.name == "sound_overlap_4.0s"
        assert cl.name == "sound_overlap_2.0s"  # original unchanged
```

See [`tests_lib/detectors/test_clipper_chain.py`](../../../tests_lib/detectors/test_clipper_chain.py)
for end-to-end load-pipeline tests that route media through a clipper
and verify the produced clips end up in `medias` with the right
origins.
