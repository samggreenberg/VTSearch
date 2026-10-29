# `vtscore.utils`

Small cross-cutting helpers with no better topical home. The package
`__init__` re-exports nothing; import from the submodules.

There is a theme to what stays: each module is a **single chokepoint**
for something that would otherwise be re-implemented at dozens of call
sites and drift. One place that declares MD5 as non-security, one
wording for the "AGPL extra not installed" error, and one atomic-write
ritual.

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/utils/io.py` | Server-side file I/O for plugins: JSON reads, atomic and resumable writes, CSV-cell escaping, a cross-process file lock |
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

## `vtscore.utils.io` - server-side file I/O

Shared file I/O for plugins that read or write server-side files: one
place for the read-and-report-errors dance and for the tmp-file + `fsync`
+ `os.replace` ritual, so a crash mid-write never leaves a half-written
file. It was the top-level `vtscore/io.py` until #5398; that name is now a
deprecated alias of this module.

```python
def read_server_bytes(path, *, missing_ok: bool = False) -> bytes | None: ...
def read_server_json(path, *, missing_ok: bool = False) -> Any: ...
def atomic_write_text(path, text: str) -> None: ...
def atomic_write_bytes(path, data: bytes) -> None: ...
def atomic_write_json(path, obj, *, indent: int = 2) -> None: ...
def sanitize_csv_cell(value: str) -> str: ...
def desanitize_csv_cell(value: str) -> str: ...

@contextmanager
def atomic_write_stream(path, *, encoding="utf-8", newline="") -> Iterator[IO[str]]: ...

@contextmanager
def resumable_write_stream(path, position: int | None, *, encoding="utf-8", newline="") -> Iterator[IO[str]]: ...
def commit_write_stream(f: IO[str]) -> int: ...

@contextmanager
def file_lock(path) -> Iterator[None]: ...
```

**Reads.** Every failure a caller-supplied path can produce - missing,
not a file, unreadable, (for JSON) unparseable - is raised as
`ValueError` with standard text, so a host can map it to a client error.
`missing_ok=True` returns `None` for a missing file instead. These are
not confinement checks; path confinement happens earlier, in
[field normalization](plugins.md#field-value-normalization).
`read_server_bytes` is public but not in `__all__`.

**Writes.** The `atomic_write_*` helpers write a sibling tmp file,
`fsync` it, and `os.replace` it over the destination.

**Streaming writes.** `atomic_write_stream` is the same ritual for a writer
that has no finished string to hand `atomic_write_text` - a `csv` writer fed
from a generator, an NDJSON export written row by row. It yields the open
handle, `fsync`s and renames on a clean exit, and unlinks the tmp file if the
block raises, so an export that dies halfway through its record iterator
leaves the destination untouched rather than half-written.

**Resumable writes.** `resumable_write_stream` is the opposite trade, for an
export a later run must be able to continue (`--stream-results --resume`):
it writes *in place*, so what was written outlives the process.
`commit_write_stream(f)` flushes and `fsync`s the handle and returns its byte
length, which the writer records somewhere durable. Passing that offset back
as `position` cuts the file back to it - dropping anything a killed run wrote
after its last commit - and appends; `position=None` starts the file afresh.
A file that is missing or shorter than `position` raises `ValueError`, since
what was committed is then gone. A reader can see the file mid-run.

**CSV cells.** `sanitize_csv_cell` prefixes a value starting with `=`,
`+`, `-`, `@`, tab or carriage return with a single quote, so a label
someone typed cannot execute as a formula when the export is opened in a
spreadsheet. `desanitize_csv_cell` reverses it on re-import, stripping a leading
apostrophe only when the next character is one the sanitiser would have
escaped. One ambiguity is inherent to the scheme and unresolvable by the
reader: a value that genuinely begins `'-foo` is indistinguishable from
the sanitised form of `-foo`, and loses its apostrophe.

**`file_lock` locks a sibling, not the file.** The lock is taken on
`<path>.lock` rather than on the data file, because an atomic write
replaces the data file's inode via `os.replace` - any fd held against
the old inode would be locking a file nobody can reach any more. The
sibling's inode is stable.

It takes an in-process `threading.Lock` first and *then* the POSIX
`flock`, so threads within one process serialise even where `flock` is
unavailable. `flock` releases when the process exits, so a crash never
leaves a stale lock. On Windows (no `fcntl`) only the in-process lock
applies and cross-process protection degrades silently.

---

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
| the stat-free `packages_distributions` seed | `vtscore.media.torch_utils.torch_setup` ([`media.md`](media.md#stat-free-package-metadata)); `vtscore.utils.import_metadata` is a deprecated alias |

This package is small on purpose: new helpers should land in the topical
package they belong to.

---

## Cross-references

- [`datasets.md`](datasets.md#what-you-load-into) - the `md5` every loader stamps with `content_md5` / `file_md5`; the built-in [`synthetic` importer](datasets.md#built-in-importers) wraps `vtscore.utils.synthetic`.
- [`exporters.md`](exporters.md#writing-a-custom-exporter) and [`labels.md`](labels.md#built-in-label-importers) - the file exporters, label importers and JSON source that read and write through `vtscore.utils.io`.
- [`detectors.md`](detectors.md#registry) and [`datasets.md`](datasets.md#on-disk-dataset-registry) - the two registries whose index files are written with `atomic_write_json` under `file_lock`.
- [`docs/DEPLOYMENT.md`](../../../docs/DEPLOYMENT.md#installing-without-the-agpl-dependencies) - the install that makes the `optional_deps` errors appear: what is unavailable without the AGPL extra.
