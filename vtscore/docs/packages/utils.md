# `vtscore.utils`

Small cross-cutting helpers with no better topical home. The package
`__init__` re-exports nothing; import from the submodules.

There is a theme to what stays: each module is a **single chokepoint**
for something that would otherwise be re-implemented at dozens of call
sites and drift. One place that declares MD5 as non-security, and one
wording for the "AGPL extra not installed" error.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/utils/hashing.py` | Content fingerprints (`content_md5`, `content_sha1`, `new_md5`, `file_md5`) |
| `vtscore/utils/optional_deps.py` | Actionable errors when an opt-out AGPL dependency isn't installed |
| `vtscore/utils/synthetic/` | Offline synthetic media generators (`audio.py`, `images.py`, `video.py`) |
| `vtscore/utils/hits.py`, `scores.py`, `import_metadata.py` | Deprecated aliases (#5399) for the modules their helpers were folded into; see the table below |

## `vtscore.utils.hashing` - content fingerprints

```python
def content_md5(data: bytes | str) -> str: ...
def content_sha1(data: bytes | str) -> str: ...
def new_md5() -> hashlib._Hash: ...          # incremental, for .update() streaming
def file_md5(file_path: Path | str) -> str:  # constant-memory, chunked
```

Every hash VTSearch computes over media bytes answers "are these the
same bytes?" - dedup, cache keys, ETags, origin tracking. **None of them
is a security primitive.** Nothing here authenticates, signs, or protects
anything.

That distinction has a runtime consequence, which is the whole reason
this module exists. On a FIPS-enabled host OpenSSL refuses to hand out
MD5 at all, and `hashlib.md5(...)` raises `ValueError: [digital envelope
routines] unsupported`. CPython normally falls back to its built-in
`_md5`, but the Python builds shipped by FIPS-oriented distributions
(RHEL, Fedora) strip that fallback so the policy cannot be bypassed.
`usedforsecurity=False` is the supported escape hatch: it declares the
non-security use and those builds then allow the digest.

Call these instead of `hashlib` directly for anything that fingerprints
content, so the declaration is made in one place and the next
environment quirk is a one-file change.

---

## `vtscore.utils.optional_deps` - AGPL opt-out errors

```python
def agpl_unavailable_message(package: str, feature: str) -> str: ...
def agpl_import_error(package: str, feature: str) -> ImportError: ...
```

Two runtime dependencies are AGPL-3.0-or-later: `ultralytics` (YOLO) and
`PyMuPDF` (PDF rendering / document text extraction). They live in the
`agpl` extra rather than in `[project.dependencies]`, but every
documented install path requests that extra, so a normal install has
them exactly as it always did. The split exists so a deployment that
cannot take copyleft code can decline:
`VTSEARCH_NO_AGPL=1 bash scripts/install.sh`, or the
`requirements/*-no-agpl.txt` mirrors.

On such an install the backed features are simply unavailable, and a
bare `ImportError` says nothing about why. These helpers name the
package, the feature, and both ways back.

Keep the plain `import` inside a `try` at the call site rather than
routing it through `importlib.import_module`, so static analysis still
sees the real module and its stubs:

```python
try:
    import fitz
except ImportError as exc:
    raise agpl_import_error("PyMuPDF", "Rendering PDF pages") from exc
```

Raise it `from` the original so a genuinely broken install stays
distinguishable from a deliberately absent one.

---

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

`vtscore/utils/synthetic/audio.py`. Cycles through six "ideas"
producing 16-bit mono PCM WAV files at 48 kHz, each 1–2.5 s long:

| Idea    | What it sounds like                                       |
|---------|-----------------------------------------------------------|
| `tone`  | Single sine, random 180–1200 Hz, AR envelope.             |
| `chord` | 3-note chord (major / minor / dim / sus4).                |
| `drum`  | Kick / snare / hihat pattern at a random BPM (80–160).    |
| `rain`  | Low-passed noise with occasional droplet clicks.          |
| `wind`  | AM-modulated brown-ish noise.                             |
| `bird`  | FM-swept chirps with silence between.                     |

Output filenames: `<idea>_<index>.wav`, zero-padded to a stable
width. 48 kHz matches `CLAP_SAMPLE_RATE` in `vtscore.config`, so
CLAP-family embedders consume the files without resampling.

### `generate_image_dataset`

`vtscore/utils/synthetic/images.py`. A small world of cartoon smiley
faces at 512x512 PNG, drawn 2x supersampled, in three kinds (six faces,
two shapes and two scenes in every ten pictures):

- `face` - one round face, in one of seven colours (yellow the
  commonest, orange the nearest miss) with one of seven expressions:
  smile, grin or wink (the `SMILING_EXPRESSIONS`), or frown, flat,
  surprised or angry.
- `shapes` - 1–5 circles, squares, triangles and stars in the same
  colours.
- `scene` - 3–6 small faces and shapes scattered over one background.

Backgrounds are plain, polka-dotted, striped, checked or a gradient, in
any of the colours. The mix is built so that "the yellow smiley faces"
is worth training a detector for: a text query finds the yellow faces
and is less sure which are smiling.

Requires `PIL` (Pillow), imported lazily. Output filenames:
`<kind>_<index>.png`. Each picture is seeded on `(seed, index)`, so two
seeds make two sets that share no picture.

`describe_image_dataset(count, seed=42)` returns what the same call
draws, without drawing it - one dict per picture:

```python
from vtscore.utils.synthetic import describe_image_dataset

for picture in describe_image_dataset(10, seed=1):
    face = picture["objects"][0]
    if picture["kind"] == "face" and face["color"] == "yellow" and face["smiling"]:
        print(picture["filename"], face["expression"], face["box"])
```

Its stable keys are `filename`, `kind`, `background` (`style` and colour
names) and `objects` (each a `shape`, a `color` name and a `box` as
fractions of the picture; a face also has `expression` and `smiling`).
The rest are drawing parameters and may change.

### `generate_video_dataset`

`vtscore/utils/synthetic/video.py`. Cycles through four ideas
encoded as 2-second 12 fps mp4 at 224x224:

- `ball` - bouncing-ball animation.
- `walker` - walking smiley with vertical bob.
- `rotator` - rotating regular polygon (3–7 sides).
- `marquee` - scrolling text rendered as coloured per-character
  rectangles (no font file dependency).

Draws frames with Pillow and encodes via `imageio_ffmpeg` (the bundled
static ffmpeg already required by the video media type). Output filenames:
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
| runtime config            | `vtscore.config`                              |
| hit dicts (`build_media_hit`, `hit_custom_metadata`) | `vtscore.datasets.labelset` ([`datasets.md`](datasets.md#hits)); `vtscore.utils.hits` is a deprecated alias |
| non-finite score sentinel, `scored_only` / `scored_mask` | `vtscore.training.thresholds.knobs` ([`training.md`](training.md#unscorable-media-non_finite_score_sentinel-scored_only)); `vtscore.utils.scores` is a deprecated alias |
| the stat-free `packages_distributions` seed | `vtscore.media.torch_setup` ([`media.md`](media.md#stat-free-package-metadata)); `vtscore.utils.import_metadata` is a deprecated alias |

This package is small on purpose: new helpers should land in the topical
package they belong to.

---

## Cross-references

- [`datasets.md`](datasets.md#what-you-load-into) - the `md5` every loader stamps with `content_md5` / `file_md5`; the built-in [`synthetic` importer](datasets.md#built-in-importers) wraps `vtscore.utils.synthetic`.
- [`docs/DEPLOYMENT.md`](../../../docs/DEPLOYMENT.md#installing-without-the-agpl-dependencies) - the install that makes the `optional_deps` errors appear: what is unavailable without the AGPL extra.
