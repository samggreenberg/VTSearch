# `vtscore.cli`, `vtscore.cli_pipeline`, `vtscore.cli_progress`

The Flask-free command-line entry points for VTSearch's autodetect
workflow: load a dataset (from pickle or via an importer), train each
Auto-Find detector against it, score every media, and hand the results
to an exporter. Three modules cooperate - `vtscore.cli` is the
imperative pipeline (four entry-point functions plus helpers),
`vtscore.cli_pipeline` parses YAML pipeline files into the same call
shape, and `vtscore.cli_progress` is a tiny format-aware emitter that
both modules use for status output (human prose or NDJSON).

**Source:** `vtscore/cli.py` (~745 lines), `vtscore/cli_pipeline.py`
(~278 lines), `vtscore/cli_progress.py` (~128 lines).
**See also:** [`/home/user/VTSearch/docs/CLI.md`](../../../docs/CLI.md)
for the user-facing `python app.py --autodetect` reference; the
functions documented here are the underlying primitives.

## When to use what

- `vtscore.cli` - the supported library entry points. Call these
  directly from a Python script or a custom wrapper CLI.
- `vtscore.cli_pipeline` - same flow, but driven by a YAML file. Use
  this when you want a reusable, file-shaped artefact instead of a
  long argv invocation.
- `vtscore.cli_progress` - wire your script's `stdout`/`stderr` for
  human vs. machine consumption. Both other modules emit through
  this; you call `set_format("json")` once at startup to flip the
  whole CLI into NDJSON mode.

All three depend on `vtscore.config.CoreConfig.from_settings()`, so
the app-side builder must be registered before calling them in an
app context. Library-only callers should construct a `CoreConfig`
directly and skip these entry points entirely if they don't want the
autodetect *workflow* - the same loaders, trainers, and exporters
are accessible piece-by-piece from `vtscore.datasets`,
`vtscore.detectors`, and `vtscore.exporters`.

## `vtscore.cli` - autodetect entry points

Four public entry points, all sharing the same internal `_run_pipeline`
helper. Each variant differs only in how it produces media chunks:

| Function                              | Source         | Streaming?      |
|---------------------------------------|----------------|-----------------|
| `autodetect_main`                     | Pickle file    | Whole at once   |
| `autodetect_main_chunked`             | Pickle file    | Chunked         |
| `autodetect_importer_main`            | Named importer | Whole at once   |
| `autodetect_importer_main_chunked`    | Named importer | Chunked         |

All four take optional `settings_path`, `exporter_name`,
`exporter_field_values`, and the keyword-only `dry_run=False`. They
print errors via `cli_progress.emit_error()` and `sys.exit(1)` on
failure - i.e. they're meant to be called from a `__main__`-style
wrapper, not as well-behaved library functions. If you want the
library function shape, call `_run_pipeline` (private but stable).

### Signatures

```python
def autodetect_main(
    dataset_path: str,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *, dry_run: bool = False,
) -> None: ...                                                # vtscore/cli.py:637

def autodetect_main_chunked(
    dataset_path: str,
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *, dry_run: bool = False,
) -> None: ...                                                # vtscore/cli.py:691

def autodetect_importer_main(
    importer_name: str,
    field_values: dict[str, Any],
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *, dry_run: bool = False,
) -> None: ...                                                # vtscore/cli.py:661

def autodetect_importer_main_chunked(
    importer_name: str,
    field_values: dict[str, Any],
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *, dry_run: bool = False,
) -> None: ...                                                # vtscore/cli.py:716
```

### Behaviour

- **Pickle variants** load via `vtscore.datasets.loader.load_dataset_from_pickle`
  (thin load). **Importer variants** resolve *importer_name* in
  `vtscore.datasets.importers`, validate *field_values* against the
  importer's declared `fields`, then call `run_cli(...)` or
  `run_chunked_cli(...)`.
- **Chunked variants** stream the source in `chunk_size`-sized
  batches; peak RAM stays at roughly `chunk_size` medias regardless
  of total length. Detectors are trained **once** against the first
  non-empty chunk and reused for every subsequent chunk; the
  exporter sees a single merged results dict at the end.
- **Default exporter** is `"gui"` (prints to stdout) when
  `exporter_name` is `None`.

```python
from vtscore.cli import autodetect_main, autodetect_importer_main

autodetect_main(
    "data/saved_datasets/my-audio.pkl",
    settings_path="data/settings.json",
    exporter_name="server_json_file",
    exporter_field_values={"filepath": "results.json"},
)

autodetect_importer_main(
    importer_name="server_folder",
    field_values={"folder": "/srv/sounds", "media_type": "audio"},
    settings_path="data/settings.json",
    exporter_name="server_csv_file",
    exporter_field_values={"filepath": "out.csv"},
)
```

### Dry-run mode

Every entry point accepts `dry_run=True`. In that mode the function
prints (or emits, in JSON format) the *plan*: which source it would
read, which settings file it would use, which detectors would be
trained, and which exporter would run - without loading any media or
embedding anything. Validation still happens (importer name lookup,
exporter name lookup, exporter field validation), so a typo fails
fast.

In `text` format the plan is printed as labelled sections:

```
DRY RUN - no media will be loaded, embedded, scored, or exported.

Source:
  Importer: server_folder
  Params:
    folder: /srv/sounds
    media_type: audio
  Chunk size: whole dataset

Settings: data/settings.json
Auto-Find detectors (1):
  - my-detector  [media_type=audio, labels=42, file=data/detectors/my-detector.json]

Exporter: server_json_file
  filepath: results.json
```

In `json` format the same information is emitted as a single
`dry_run_plan` NDJSON event.

### One-shot label-import helper

```python
def import_labels_into_detector_from_file(
    det_name: str,
    importer_name: str,
    filepath: str,
) -> tuple[int, int]:
```

Defined at `vtscore/cli.py:300`. Runs a named label importer
(`vtscore.labels.importers`) against a single file, merges the
returned `LabeledElement`s into the named detector's labelset, and
returns `(applied, skipped)`. Used by the pipeline-YAML
`import_labels:` block (see below); also callable directly when you
want to ingest a CSV/JSON of labels without running a full
autodetect pass.

### What `_run_pipeline` does

All four entry points delegate to `_run_pipeline` (defined at
`vtscore/cli.py:583`). The interesting steps:

1. Build a `CoreConfig` via `CoreConfig.from_settings(settings_path=...)`.
2. If `dry_run`, validate + emit the plan and return.
3. Otherwise, iterate the *media_source* iterator chunk by chunk.
4. On the first non-empty chunk, train each Auto-Find (or override)
   detector via `_load_and_train_detectors`. Detectors with a
   `media_type` mismatch or an `input_spec.clipper` mismatch against
   the loaded dataset are *skipped* with a warning event, not
   errored.
5. Score each chunk via `_score_medias_with_detectors`, merging hits
   into the accumulated results in place.
6. Hand the merged `{media_type, detectors_run, results}` dict to
   the exporter via `_run_exporter`.

Detectors whose label origins cannot be resolved from the CLI
environment (e.g. labels collected through the browser's `local_folder`
importer have no `resolve_file()` path) raise `ValueError` - that's a
hard error, not a skip, because it indicates the run cannot be
reproduced as the user expects.

## `vtscore.cli_pipeline` - YAML pipeline files

Source: `vtscore/cli_pipeline.py`. One public function:

```python
def load_pipeline_file(path: str | Path) -> dict[str, Any]:
```

Reads *path* (must exist), parses it as YAML, validates the shape,
and returns a normalised config dict ready for `run_pipeline_file` to
dispatch.

**Top-level keys** (any combination, but exactly one of `dataset:` or
`importer:` must be set):

| Key             | Type                  | Meaning                                                                  |
|-----------------|-----------------------|--------------------------------------------------------------------------|
| `dataset`       | `str` path            | Path to a dataset pickle.                                                |
| `importer`      | `{name, fields?}`     | Importer name + per-field values. Mutually exclusive with `dataset`.     |
| `settings`      | `str` path            | Override settings file path.                                             |
| `detectors`     | `list[str]`           | Override `autofind_detectors` for this run only.                          |
| `chunk_size`    | positive `int`        | Stream the source in chunks of this size.                                |
| `import_labels` | `{detector, file, importer?}` | Run a label importer + merge into a detector before scoring.   |
| `exporter`      | `{name, fields?}`     | Exporter name + per-field values.                                        |

Field-key validation happens against the live plugin registry - a
typo in `importer.name` or any `fields.*` key fails at parse time
before media touches RAM.

```yaml
# pipeline.yaml
importer:
  name: server_folder
  fields:
    folder: /srv/sounds
    media_type: audio
settings: data/settings.json
detectors:
  - my-detector
chunk_size: 256
exporter:
  name: server_csv_file
  fields:
    filepath: out.csv
```

```python
from vtscore.cli_pipeline import load_pipeline_file, run_pipeline_file

config = load_pipeline_file("pipeline.yaml")
# config is a fully-validated dict; inspect or mutate before running.

run_pipeline_file("pipeline.yaml")
# Loads + dispatches against vtscore.cli._run_pipeline.
```

`run_pipeline_file` (defined at `vtscore/cli_pipeline.py:209`) is the
"do everything" wrapper: it calls `load_pipeline_file`, runs the
optional `import_labels` block, then dispatches to `_run_pipeline`
with `override_detectors=config["detectors"]` so the YAML file can
declare a detector list inline without mutating `settings.json`.

## `vtscore.cli_progress` - format-aware output

Source: `vtscore/cli_progress.py`. The whole module is thread-safe by
construction (writes go straight to `sys.stdout`/`sys.stderr` with
`flush()`); state is a single module-global `_format` flag.

### API

```python
FORMATS = ("text", "json")

def set_format(fmt: str) -> None: ...
def get_format() -> str: ...

def emit(
    event: str,
    *,
    text: str | None = None,
    stream: TextIO | None = None,
    **fields: Any,
) -> None: ...

def emit_error(message: str) -> None: ...

def progress_callback(status: str, message: str = "", current: int = 0, total: int = 0) -> None: ...
```

### Behaviour

- `set_format(fmt)` - call once at startup; flag is module-global.
  `"text"` (default) sends prose to stdout, errors to stderr, tqdm
  bars to stderr. `"json"` sends NDJSON to stdout and errors *also*
  to stdout so a single pipe captures the whole stream. Any other
  value raises `ValueError`.
- `emit(event, *, text=None, stream=None, **fields)` - in `text`
  mode, writes *text* (when given) to *stream* with a newline +
  flush. In `json` mode, writes one NDJSON line
  `{"event": event, "ts": <iso8601-z>, **fields}` to *stream*; *text*
  is ignored. *stream* defaults to `sys.stdout`.
- `emit_error(message)` - `Error: <message>\n` to stderr in text
  mode, `{"event":"error",...}` to stdout in JSON mode. Caller is
  responsible for `sys.exit(1)` afterwards.
- `progress_callback` - a drop-in `ProgressCallback`
  (`vtscore.media.base.ProgressCallback`) that emits `progress`
  events in JSON mode and is a no-op in text mode. Pass it to any
  loader / embedder API that accepts a `progress_callback`.

```python
from vtscore import cli_progress

cli_progress.set_format("json")
cli_progress.emit(
    "chunk_start",
    text=f"Processing chunk {n} ({len(chunk)} medias)...",
    chunk_num=n,
    chunk_size=len(chunk),
)
```

### Event reference

Events the CLI emits today. Every event includes `event` and `ts`;
each row lists the extra fields.

| Event              | Fields                                                            | Source                                |
|--------------------|-------------------------------------------------------------------|---------------------------------------|
| `chunk_start`      | `chunk_num: int`, `chunk_size: int`                               | `_score_chunk` in `cli.py`            |
| `chunks_done`      | `total_medias: int`, `chunks: int`                                | `_run_live_pipeline` in `cli.py`      |
| `detector_skipped` | `detector: str`, plus reason-specific fields                      | `_load_and_train_detectors`           |
| `export_complete`  | `message: str`                                                    | `_run_exporter` in `cli.py`           |
| `dry_run_plan`     | `source`, `settings_path`, `autofind_detectors`, `exporter`, `exporter_field_values` | `_emit_dry_run_plan`        |
| `progress`         | `status: str`, optional `message`, `current`, `total`, `pct`      | `progress_callback`                   |
| `error`            | `message: str`                                                    | `emit_error` in JSON mode             |

Progress ticks with no `message` and `total <= 0` are dropped, so
consumers never see empty `{"status":"idle"}` records.

### Consuming the NDJSON stream

```bash
python app.py --autodetect --dataset my.pkl --progress-format json 2>/dev/null \
  | jq -c 'select(.event == "progress" or .event == "error")'
```

Stderr is reserved for unstructured noise (tqdm bars, library
warnings); discard it with `2>/dev/null` and you still see error
events on stdout.

## Cross-references

- The user-facing CLI flags (`--autodetect`, `--pipeline`,
  `--progress-format`, etc.) are documented in
  [`/home/user/VTSearch/docs/CLI.md`](../../../docs/CLI.md). The
  functions documented here are the underlying primitives those
  flags call.
- [`config.md`](config.md) explains `CoreConfig.from_settings()`,
  which the pipeline builds once per invocation.
- [`utils.md`](utils.md) documents `build_media_hit`, the helper
  every detector-score row goes through before the exporter sees
  it.
