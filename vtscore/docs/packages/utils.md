# `vtscore.utils`

The leftover-helpers package. Most of what used to live here has
moved to topical homes - state, plugins, sync, concurrency, security,
and audio helpers all live in their own top-level packages now (see
`vtscore.state`, `vtscore.plugins`, `vtscore.sync`,
`vtscore.concurrency`, `vtscore.security`, and
`vtscore.media.audio`). What's left is genuinely homeless: a single
hit-dict builder shared by the CLI and the API, plus the synthetic
media generators used by the offline demo importer. Two modules,
both small, both stable.

**Source:** `vtscore/utils/__init__.py`, `vtscore/utils/hits.py`,
`vtscore/utils/synthetic/`.

## `vtscore.utils.hits.build_media_hit`

The single source of truth for the scored-media hit-dict shape.
Used by `vtscore.cli` when assembling autodetect results and by the
app's `/api/labels/fill-from-sort` route when materialising scored
rows for the UI. If you score a media against a detector and want to
return the result, build the row through this function rather than
constructing the dict by hand - that's how new optional fields
(clip boundaries, MD5, origin) get picked up automatically.

```python
def build_media_hit(
    cid: int,
    media: dict[str, Any],
    score: float,
    **extra: Any,
) -> dict[str, Any]:
```

Defined at `vtscore/utils/hits.py:8`.

### Shape

The returned dict always has these keys:

| Key        | Type    | Source                                                       |
|------------|---------|--------------------------------------------------------------|
| `id`       | `int`   | the *cid* argument                                           |
| `filename` | `str`   | `media["filename"]`, falling back to `f"media_{cid}"`        |
| `category` | `str`   | `media["category"]`, falling back to `"unknown"`             |
| `score`    | `float` | `round(score, 4)`                                            |

Optional keys are added only when the corresponding field is present
on *media*:

| Key            | Added when…                              |
|----------------|------------------------------------------|
| `origin`       | `media["origin"] is not None`            |
| `origin_name`  | `media["origin_name"]` is truthy         |
| `md5`          | `media["md5"]` is truthy                 |
| `clip_start`   | `media["clip_start"] is not None`        |
| `clip_end`     | `media["clip_end"] is not None`          |
| `clip_box`     | `media["clip_box"] is not None`          |
| `clip_index`   | `media["clip_index"] is not None`        |

The `clip_*` keys are how sub-medias produced by a clipper round-trip
through scoring. They are passed through unchanged so the consumer
can reconstruct the time-range / spatial-region the score applies
to.

### Example

```python
from vtscore.utils.hits import build_media_hit

media = {
    "filename": "talk_01.wav",
    "category": "podcast",
    "origin": {"importer": "server_folder", "params": {"folder": "/srv/sounds"}},
    "origin_name": "/srv/sounds/talk_01.wav",
    "md5": "abc123",
    "clip_start": 12.5,
    "clip_end": 17.5,
    "clip_index": 2,
}

hit = build_media_hit(cid=42, media=media, score=0.873, label="good")

# hit == {
#     "id": 42,
#     "filename": "talk_01.wav",
#     "category": "podcast",
#     "score": 0.873,
#     "origin": {"importer": "server_folder", "params": {"folder": "/srv/sounds"}},
#     "origin_name": "/srv/sounds/talk_01.wav",
#     "md5": "abc123",
#     "clip_start": 12.5,
#     "clip_end": 17.5,
#     "clip_index": 2,
#     "label": "good",
# }
```

Extra keyword arguments are merged in last, so callers can attach
detector-specific or call-site-specific fields (`label="good"`,
`detector="my-detector"`) without needing to extend this function.

## `vtscore.utils.synthetic` - offline media generators

Deterministic, internet-free media synthesis for the
`SyntheticDatasetImporter`. Each generator writes a small,
semantically-clustered dataset to a folder so users can exercise
VTSearch without an internet connection or any public-dataset
downloads. Three media types, three functions, identical shape:

```python
from pathlib import Path
from vtscore.utils.synthetic import (
    generate_audio_dataset,
    generate_image_dataset,
    generate_video_dataset,
)

audio_paths: list[Path] = generate_audio_dataset(Path("synth/audio"), count=24)
image_paths: list[Path] = generate_image_dataset(Path("synth/img"),   count=24)
video_paths: list[Path] = generate_video_dataset(Path("synth/video"), count=24)
```

Each function takes `output_dir`, `count`, an optional `seed`
(default `42`), and an optional `on_progress` callback compatible
with `vtscore.media.base.ProgressCallback`. Each is **deterministic**
given the same `(count, seed)` - re-running produces the same files,
and existing files matching the naming scheme are kept on subsequent
calls (the importer relies on this cache across reloads).

### `generate_audio_dataset`

`vtscore/utils/synthetic/audio.py:172`. Cycles through six "ideas"
producing 16-bit mono PCM WAV files at 48 kHz:

| Idea    | What it sounds like                                       |
|---------|-----------------------------------------------------------|
| `tone`  | Single sine, random 180–1200 Hz, AR envelope.             |
| `chord` | 3-note chord (major / minor / dim / sus4) over 1–2 s.     |
| `drum`  | Kick / snare / hihat pattern at a random BPM (80–160).    |
| `rain`  | Low-passed noise with occasional droplet clicks.          |
| `wind`  | AM-modulated brown-ish noise.                             |
| `bird`  | FM-swept chirps with silence between.                     |

Output filenames: `<idea>_<index>.wav`, zero-padded to a stable
width. 48 kHz matches `CLAP_SAMPLE_RATE` in `vtscore.config`, so
CLAP-family embedders consume the files without resampling.

### `generate_image_dataset`

`vtscore/utils/synthetic/images.py:140`. Cycles through two ideas at
256x256 PNG:

- `smiley` - a face on a coloured background, with one of four
  emotions (happy / sad / neutral / angry), random face / skin
  colour, size, and position.
- `shapes` - 1–5 coloured circles, squares, and rotated triangles
  on a plain background.

Requires `PIL` (Pillow), imported lazily. Output filenames:
`<idea>_<index>.png`.

### `generate_video_dataset`

`vtscore/utils/synthetic/video.py:190`. Cycles through four ideas
encoded as 2-second 12 fps mp4 at 224x224:

- `ball` - bouncing-ball animation.
- `walker` - walking smiley with vertical bob.
- `rotator` - rotating regular polygon (3–7 sides).
- `marquee` - scrolling text rendered as coloured per-character
  rectangles (no font file dependency).

Encodes via `imageio_ffmpeg` (the bundled static ffmpeg already
required by the video media type). Output filenames:
`<idea>_<index>.mp4`.

### `ProgressCallback` shape

All three generators take an `on_progress` callable of type:

```python
ProgressCallback = Callable[[str, str, int, int], None]
# (status, message, current, total)
```

The `status` field is always `"downloading"` so synthetic generation
maps to the dataset pipeline's "fetching files" step - the
importer's consumer doesn't need to special-case synthetic origins.
The callback is fired at the start, once per file (with
`current = i`), and once at the end (with `current = total`).

```python
def on_progress(status: str, message: str, current: int, total: int) -> None:
    print(f"[{current}/{total}] {message}")

generate_image_dataset(Path("synth"), count=8, on_progress=on_progress)
```

## What's *not* here

Things you might expect to find under `vtscore.utils` based on the
name alone, and where they actually live:

| If you want…              | Look in…                                      |
|---------------------------|-----------------------------------------------|
| state contexts, locks     | `vtscore.state`                               |
| plugin base / registry    | `vtscore.plugins`                             |
| sync sources              | `vtscore.sync`                                |
| job manager, progress     | `vtscore.concurrency`                         |
| path / URL / pickle safety | `vtscore.security`                           |
| audio helpers (wav, resample) | `vtscore.media.audio`                     |
| settings accessors        | `vtsearch.settings_factory` (app-tier only)   |

The "leftover" framing is honest: this package is small on purpose.
New helpers should land in the topical package they actually belong
to, not here.
