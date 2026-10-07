# `vtscore.cli`, `vtscore.cli_pipeline`, `vtscore.cli_progress`

The Flask-free command-line entry points for VTSearch's autodetect
workflow: load a dataset (from pickle or via an importer), train each
AutoFind detector against it, score every media, and hand the results
to an exporter. Three modules cooperate - `vtscore.cli` is the
imperative pipeline (four entry-point functions plus helpers),
`vtscore.cli_pipeline` parses YAML pipeline files into the same call
shape, and `vtscore.cli_progress` is a tiny format-aware emitter that
both modules use for status output (human prose or NDJSON).

## Contents

| Module | Concern |
|--------|---------|
| `vtscore/cli.py` | The imperative autodetect pipeline - four entry points plus helpers |
| `vtscore/cli_pipeline.py` | Parse a YAML pipeline file into the same call shape |
| `vtscore/cli_progress.py` | Format-aware status emitter (human prose or NDJSON) |

**See also:** [`docs/CLI.md`](../../../docs/CLI.md) for the user-facing
`python app.py --autodetect` reference; the functions documented here
are the underlying primitives.

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

`vtscore.cli` and `vtscore.cli_pipeline` depend on
`vtscore.config.CoreConfig.from_settings()`, so
the app-side builder must be registered before calling them in an
app context. Library-only callers should construct a `CoreConfig`
directly and skip these entry points entirely if they don't want the
autodetect *workflow* - the same loaders, trainers, and exporters
are accessible piece-by-piece from `vtscore.datasets`,
`vtscore.detectors`, and `vtscore.exporters`.

## `vtscore.cli` - autodetect entry points

Four public entry points, all thin shims over one private
`_autodetect(spec, ...)` (which in turn drives the shared
`_run_pipeline`). Each variant differs only in the `_SourceSpec` it
builds - i.e. in how it produces media chunks:

| Function                              | Source         | Streaming?      |
|---------------------------------------|----------------|-----------------|
| `autodetect_main`                     | Pickle file    | Whole at once   |
| `autodetect_main_chunked`             | Pickle file    | Chunked         |
| `autodetect_importer_main`            | Named importer | Whole at once   |
| `autodetect_importer_main_chunked`    | Named importer | Chunked         |

All four take optional `settings_path`, `exporter_name`,
`exporter_field_values`, and the keyword-only `dry_run=False`,
`stream_results=False`, `keep_negatives=False`, `save_dataset=False` and
`override_detectors=None`. They catch every
exception, report it via `cli_progress.emit_error()`, and
`sys.exit(1)` - i.e. they're meant to be called from a
`__main__`-style wrapper, not as well-behaved library functions.
The raising equivalent is the private `_run_pipeline` (see below).

### Signatures

```python
def autodetect_main(
    dataset_path: str,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
) -> None: ...                                                # vtscore/cli.py

def autodetect_main_chunked(
    dataset_path: str,
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
) -> None: ...                                                # vtscore/cli.py

def autodetect_importer_main(
    importer_name: str,
    field_values: dict[str, Any],
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
) -> None: ...                                                # vtscore/cli.py

def autodetect_importer_main_chunked(
    importer_name: str,
    field_values: dict[str, Any],
    chunk_size: int,
    settings_path: str | None = None,
    exporter_name: str | None = None,
    exporter_field_values: dict[str, Any] | None = None,
    *,
    dry_run: bool = False,
    stream_results: bool = False,
    keep_negatives: bool = False,
    save_dataset: bool = False,
    override_detectors: list[str] | None = None,
) -> None: ...                                                # vtscore/cli.py
```

### Behaviour

- **Pickle variants** load the pickle via `vtscore.datasets.loader`.
  **Importer variants** resolve *importer_name* in
  `vtscore.datasets.importers`, validate *field_values* with the
  importer's `validate_cli_field_values`, then call `run_cli(...)` or
  `run_chunked_cli(...)` (thin when the importer's `reference_files`
  field asks for it) and embed whatever the load left unembedded.
- **Chunked variants** stream the source in `chunk_size`-sized
  batches; peak RAM stays at roughly `chunk_size` medias regardless
  of total length. Detectors are trained **once** against the first
  non-empty chunk and reused for every subsequent chunk; the
  exporter sees a single merged results dict at the end.
- **`override_detectors`**: a list of detector names to train and score
  in place of the settings file's `autofind_detectors` (the file itself is
  never modified). The app's `--import-labels-into NAME` passes `[NAME]`,
  and the pipeline YAML passes its `detectors:` list (or, without one,
  `[import_labels.detector]`). The dry-run plan lists these instead of the
  AutoFind list.
- **Default exporter**: when `exporter_name` is `None`, the settings
  file's `autofind_exporter` (with its saved field values) is used;
  if that is unset too, `"gui"` (prints to stdout).
- **`stream_results=True`** hands the (streaming-capable) exporter a
  lazy record iterator instead of accumulating a merged results dict,
  so nothing proportional to the hit count is held in RAM;
  `keep_negatives=True` additionally streams below-threshold hits
  (labelled `"bad"`). Both are accepted by all four entry points -
  they pair naturally with the chunked variants, but a whole-dataset
  run also benefits, since the hits need not accumulate even when the
  medias already have. The `--stream-results` **CLI flag** and the
  pipeline-YAML `stream_results:` key are narrower: both require a
  chunk size. The exporter must declare `supports_streaming`, or the
  run fails with the list of exporters that do.

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
    field_values={"path": "/srv/sounds", "media_type": "audio"},
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
    path: /srv/sounds
    media_type: audio
  Chunk size: whole dataset

Settings: data/settings.json
AutoFind detectors (1):
  - my-detector  [media_type=audio, labels=42, file=data/detectors/my-detector.json]

Exporter: server_json_file
  filepath: results.json
```

In `json` format the same information is emitted as a single
`dry_run_plan` NDJSON event.

### One-shot label-import helpers

```python
def import_labels_into_detector(
    det_name: str,
    importer_name: str,
    field_values: dict[str, Any],
    *,
    create_media_type: str = "",
) -> tuple[int, int]:

def import_labels_into_detector_from_file(
    det_name: str,
    importer_name: str,
    filepath: str,
) -> tuple[int, int]:
```

Defined in `vtscore/cli.py`. `import_labels_into_detector` runs a
named label importer (`vtscore.labels.importers`) with the given field
values, merges the returned `LabeledElement`s into the named
detector's labelset, and returns `(applied, skipped)`. Required fields
are checked and values normalized as for the other CLI plugin paths.
`import_labels_into_detector_from_file` is the file-path shorthand
(`{"filepath": filepath}`). Used by the pipeline-YAML `import_labels:`
block and `--import-labels-into` (see below); also callable directly
when you want to ingest labels without running a full autodetect pass.

A detector with no file in the detectors dir raises
`DetectorNotFoundError` (a `ValueError` subclass carrying `det_name`)
unless `create_media_type` names a registered media type. In that case the
detector is **created** from the imported labels: the JSON the Dashboard's
New Detector writes (no seed examples, `embedder_type` left for the first
train to resolve), plus a `vtscore.detectors.registry` entry owned by
`vtscore.state.current_user.get_current_user()`, so it appears in that
user's Drafts. No entry is added when one already owns the name. Nothing
is written when the import yields no `good`/`bad` label, and an existing
detector ignores `create_media_type`. The app's `--create-detector` and the
YAML's `import_labels.create: true` drive it, taking the media type from
`--detector-media-type` / `import_labels.media_type`, else from the source:
a dataset pickle's `meta.json`, or an importer's `media_type` field.

### What `_run_pipeline` does

All four entry points delegate to `vtscore/cli.py::_run_pipeline`
(via `_autodetect`). The interesting steps:

1. Build a `CoreConfig` via `CoreConfig.from_settings(settings_path=...)`
   and resolve the exporter (explicit name, else `autofind_exporter`).
2. If `dry_run`, validate + emit the plan and return without consuming
   the source.
3. If `stream_results`, hand off to the streaming path
   (`_run_streaming_pipeline`): train on the first chunk, then pass the
   exporter a header plus a lazy `(detector_name, hit)` iterator via
   `export_cli_streaming`.
4. Otherwise iterate the source chunk by chunk. On the first non-empty
   chunk, train each AutoFind (or `override_detectors`) detector via
   `_load_and_train_detectors`. A detector whose `media_type` has no
   direct or one-hop converter route from the dataset's types is
   *skipped* with a `detector_skipped` event; one whose
   `input_spec.clipper` doesn't match the dataset is *re-clipped* at
   scoring time (a `detector_reclip` event), not skipped. Nobody can
   vote in a headless run, so a detector's line is never spot-checked:
   it keeps the balance's unchecked set (the mixture's F-beta argmax
   under the balance's cap) and is announced with a
   `detector_unchecked` event; every result it produces carries the
   same state under `balance`.
5. Score each chunk via `_score_medias_with_detectors`, merging hits
   into the accumulated results in place.
6. Hand the merged `{media_type, detectors_run, results}` dict to
   the exporter via `_run_exporter`. An exporter whose
   `supported_payloads` include `"detector_bundles"` (the portable
   detector) gets the trained detectors via `export_cli_detectors`
   instead.

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
| `detectors`     | `list[str]`           | Override `autofind_detectors` for this run only. Defaults to `[import_labels.detector]` when `import_labels` is set. |
| `chunk_size`    | positive `int`        | Stream the source in chunks of this size.                                |
| `stream_results` | `bool`               | Stream hits to the exporter (see above). Requires `chunk_size`.          |
| `keep_negatives` | `bool`               | Also stream below-threshold hits. Requires `stream_results`.             |
| `import_labels` | `{detector, file, importer?, create?, media_type?}` | Run a label importer + merge into a detector before scoring; `create: true` makes the detector if it is missing (`media_type` overrides the source's). |
| `exporter`      | `{name, fields?}`     | Exporter name + per-field values.                                        |

Unknown top-level keys raise `ValueError`; a missing file raises
`FileNotFoundError`. Field-key validation happens against the live
plugin registry - a typo in `importer.name` or any `fields.*` key
fails at parse time before media touches RAM.

```yaml
# pipeline.yaml
importer:
  name: server_folder
  fields:
    path: /srv/sounds
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

`run_pipeline_file(path)` is the "do everything" wrapper: it calls
`load_pipeline_file`, runs the optional `import_labels` block, then
dispatches to `_run_pipeline` with
`override_detectors=config["detectors"]` so the YAML file can declare
a detector list inline without mutating `settings.json`. A
`FileNotFoundError` / `ValueError` becomes `Error: ...` on stderr and
`sys.exit(1)`.

## `vtscore.cli_progress` - format-aware output

Source: `vtscore/cli_progress.py`. Every write goes straight to
`sys.stdout`/`sys.stderr` with a `flush()`; the only state is a
single module-global `_format` flag.

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

def notification_subscriber(notification: Notification) -> None: ...
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
- `notification_subscriber(notification)` - subscribe it to
  `vtscore.concurrency.notifications.notifications` for the life of a
  run so plugin notifications (GUI toasts) aren't dropped headless.
  Text mode: one `Note:` / `Done:` / `Warning:` / `Error:` line on
  **stderr**, followed by one indented `- item` line per entry in the
  notification's `items`. JSON mode: a `notification` event on stdout.
  Never ends the run, even at `level="error"`.

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

Events emitted by `vtscore.cli` and `vtscore.cli_progress`. Every
event includes `event` and `ts`; each row lists the extra fields.

| Event              | Fields                                                            | Source                                |
|--------------------|-------------------------------------------------------------------|---------------------------------------|
| `chunk_start`      | `chunk_num: int`, `chunk_size: int`                               | `_score_chunk` in `cli.py`            |
| `chunks_done`      | `total_medias: int`, `chunks: int`                                | `_run_live_pipeline` in `cli.py`      |
| `detector_skipped` | `detector: str`, plus reason-specific fields                      | `_load_and_train_detectors`           |
| `detector_reclip`  | `detector`, `detector_input_spec`, `dataset_input_spec`           | `_load_and_train_detectors`           |
| `detector_unchecked` | `detector`, `beta`, `status` (`unchecked`), `count` (the exported set's size) | `_record_line_state` in `cli.py` |
| `medias_skipped`   | `skipped: int`, `skipped_ids` (first 100), `embedder`             | `_emit_skipped_medias` in `cli.py`    |
| `medias_unembedded`| `unembedded: int`, `unembedded_ids` (first 100)                   | `_embed_loaded_medias` in `cli.py`    |
| `export_complete`  | `message: str`, optional `open_url` (validated `http(s)` URL)     | `_run_exporter` in `cli.py`           |
| `dry_run_plan`     | `source`, `settings_path`, `autofind_detectors` (the detectors the run would score), `detectors_source` (`"autofind"` or `"override"`), `exporter`, `exporter_field_values` | `_emit_dry_run_plan`        |
| `progress`         | `status: str`, optional `message`, `current`, `total`, `pct`      | `progress_callback`                   |
| `notification`     | `level`, `message`, `detail`, `source`, `items`                   | `notification_subscriber`             |
| `error`            | `message: str`                                                    | `emit_error` in JSON mode             |

Progress ticks with no `message` and `total <= 0` are dropped, so
consumers never see empty `{"status":"idle"}` records. The app's
`--import-labels-into` flag also emits `labels_imported`
(`detector`, `applied`, `skipped`, `created`), but from the app tier, not
from these modules.

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
  [`docs/CLI.md`](../../../docs/CLI.md). The
  functions documented here are the underlying primitives those
  flags call.
- [`config.md`](config.md) explains `CoreConfig.from_settings()`,
  which the pipeline builds once per invocation.
- [`utils.md`](utils.md) documents `build_media_hit`, the helper
  every detector-score row goes through before the exporter sees
  it.
