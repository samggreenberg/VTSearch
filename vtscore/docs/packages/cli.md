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

`vtscore.cli` and `vtscore.cli_pipeline` read their settings (the
AutoFind list and exporter, the detectors directory, …) with
`vtscore.config.get_setting()`, and apply `--settings` with
`redirect_settings_path()`, so the app's setting getters - or a
library consumer's `CoreConfig` builder - must be registered before
calling them. Library-only callers can skip these entry points entirely
if they don't want the autodetect *workflow* - the same loaders, trainers, and exporters
are accessible piece-by-piece from `vtscore.datasets`,
`vtscore.detectors`, and `vtscore.exporters`.

## `vtscore.cli` - autodetect entry points

Four public entry points, all thin shims over one private
`_autodetect(spec, ...)` (which drives `_run_source`, and through it the
shared `_run_pipeline`). Each variant differs only in the `_SourceSpec` it
builds - i.e. in how it produces media chunks:

| Function                              | Source         | Streaming?      |
|---------------------------------------|----------------|-----------------|
| `autodetect_main`                     | Pickle file    | Whole at once   |
| `autodetect_main_chunked`             | Pickle file    | Chunked         |
| `autodetect_importer_main`            | Named importer | Whole at once   |
| `autodetect_importer_main_chunked`    | Named importer | Chunked         |

All four take optional `settings_path`, `exporter_name`,
`exporter_field_values`, and the keyword-only `dry_run=False`,
`stream_results=False`, `keep_negatives=False`, `save_dataset=False`,
`override_detectors=None` and `override_clipper=None` (an `input_spec`,
`{"clipper": name, "clipper_params": {...}}`, that every detector is scored
at in place of its own; see [Behaviour](#behaviour)); the two importer
variants also take `outputs=None` (several datasets from one import).
They catch every
exception, report it via `cli_progress.emit_error()`, and
`sys.exit(1)` - i.e. they're meant to be called from a
`__main__`-style wrapper, not as well-behaved library functions.
The raising equivalent is the private `_run_source` (see below).

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
    override_clipper: dict[str, Any] | None = None,
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
    override_clipper: dict[str, Any] | None = None,
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
    outputs: list[OutputSpec] | None = None,
    override_clipper: dict[str, Any] | None = None,
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
    outputs: list[OutputSpec] | None = None,
    override_clipper: dict[str, Any] | None = None,
) -> None: ...                                                # vtscore/cli.py
```

### Behaviour

- **Pickle variants** load the pickle via `vtscore.datasets.loader`.
  **Importer variants** resolve *importer_name* in
  `vtscore.datasets.importers`, validate *field_values* with the
  importer's `validate_cli_field_values`, then call `run_cli(...)` or
  `run_chunked_cli(...)` (thin when the importer's `reference_files`
  field asks for it) and embed whatever the load left unembedded.
- **`save_dataset=True`** imports the source through the GUI's load
  pipeline, registers it, then scores the saved pickle (a `detection_skipped`
  event when no detector applies; a `dataset_deleted` event when the
  `autofind_cli_delete_dataset` setting removes it afterwards).
  **`outputs=[OutputSpec, ...]`** (`vtscore.datasets.importers.base`) makes
  one importer run produce one dataset per entry, each its own
  `_run_pipeline` pass. The user-facing rules for both - what is kept, what
  is deleted, how several datasets' exports are told apart, the exit code -
  are in [`docs/CLI.md`](../../../docs/CLI.md#saving-the-dataset-to-the-dashboard---tempimport)
  and its [Sources](../../../docs/CLI.md#sources) section.
- **Chunked variants** stream the source in `chunk_size`-sized
  batches; peak RAM stays at roughly `chunk_size` medias regardless
  of total length. Detectors are trained **once** against the first
  non-empty chunk and reused for every subsequent chunk; the
  exporter sees a single merged results dict at the end. With
  `save_dataset=True` the import itself still holds the whole dataset, as a
  GUI import does; `chunk_size` then bounds only the scoring pass over the
  saved copy.
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
  medias already have. Streaming never holds the whole dataset, so it
  cannot save one: with `save_dataset=True` the run fails. The
  `--stream-results` **CLI flag** and the pipeline-YAML `stream_results:`
  key are narrower still: both also require a chunk size. The exporter must
  declare `supports_streaming`, or the run fails with the list of exporters
  that do.
- **`as_labelset=True`** (with `stream_results`; refused without it) streams
  the hits as label entries instead, through the exporter's
  `export_labelset_streaming`: every detector's entries in one export, each
  naming its `detector`, with no score, threshold or balance. The exporter must
  report `supports_labelset_streaming`. The CLI flag is `--as-labelset`; the
  pipeline-YAML key is `as_labelset:`.

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
embedding anything. Validation still happens (importer name lookup and
field validation, the pickle's existence, exporter name lookup and field
validation), so a typo fails fast. The `Save to dashboard:` line follows
`save_dataset`, which defaults to `False` here; `python app.py` passes `True`
unless `--tempimport` is set.

In `text` format the plan is printed as labelled sections (an example is
in [`docs/CLI.md`](../../../docs/CLI.md#dry-run-mode)); in `json` format
it is a single `dry_run_plan` NDJSON event.

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
(via `_autodetect` and `_run_source`, which first saves the dataset when
`save_dataset` is set; a multi-dataset source runs it once per dataset). The
interesting steps:

1. Apply `settings_path` with `redirect_settings_path()`, read the
   AutoFind list with `get_setting()`, and resolve the exporter (explicit
   name, else `autofind_exporter`).
2. If `dry_run`, validate + emit the plan and return without consuming
   the source.
3. If `stream_results`, hand off to the streaming path
   (`_run_streaming_pipeline`): train on the first chunk, then pass the
   exporter a header plus a lazy `(detector_name, hit)` iterator via
   `export_cli_streaming`, or, with `as_labelset`, a lazy iterator of label
   entries via `export_labelset_streaming`.
4. Otherwise iterate the source chunk by chunk. On the first non-empty
   chunk, train each AutoFind (or `override_detectors`) detector via
   `_load_and_train_detectors`. A detector whose `media_type` has no
   direct or one-hop converter route from the dataset's types is
   *skipped* with a `detector_skipped` event; one whose
   `input_spec.clipper` doesn't match the dataset is *re-clipped* at
   scoring time (a `detector_reclip` event), not skipped. An
   `override_clipper` takes the place of every detector's `input_spec`
   (#5282): it is validated with
   `vtscore.detectors.input_spec.resolve_clipper_spec` before anything is
   loaded, re-clips unless the dataset already matches it (a `*_default`
   clipper never does), and raises `ValueError` for a detector whose media
   type it does not clip; its `detector_reclip` event adds
   `override_input_spec`. Nobody can
   vote in a headless run, so a detector's line is never spot-checked:
   it exports the set its line keeps, unchecked, and is announced with a
   `detector_unchecked` event; every result it produces carries the
   same state under `balance`. Under the label quota (3 Goods and 4
   Bads, or a Good and 16 Bads) there is no trained head: a detector
   made from a typed query scores as that query's own set, in the
   dataset's text embedder, from before its first label (a
   `detector_query` event, #4758), and on a single-vector space past
   the quota until 30 labels (#4736), and one with no typed query as the
   Goods' centroid from its first Good (a `detector_centroid` event,
   #4643). The event's fields are the quota's
   (`vtscore.detectors.label_quota.served_quota`).
5. Score each chunk via `_score_medias_with_detectors`, merging hits
   into the accumulated results in place.
6. Hand the merged `{media_type, detectors_run, results}` dict (plus a
   `dataset` block in a multi-dataset run) to
   the exporter via `_run_exporter`. An exporter whose
   `supported_payloads` include `"detector_bundles"` (the portable
   detector) gets the trained detectors via `export_cli_detectors`
   instead.

Detectors whose label origins cannot be resolved from the CLI
environment (e.g. labels collected through the browser's `local_folder`
importer have no `resolve_file()` path) raise `ValueError` - that's a
hard error, not a skip, because it indicates the run cannot be
reproduced as the user expects. A detector made from a typed query
scores as its query only while its labels are under the quota both as
written and as resolved (#4758): one whose labels meet the quota but
fail to resolve still raises.

## `vtscore.cli_pipeline` - YAML pipeline files

Source: `vtscore/cli_pipeline.py`. Two public functions:

```python
def load_pipeline_file(path: str | Path) -> dict[str, Any]:
def run_pipeline_file(path: str | Path) -> None:
```

`load_pipeline_file` reads *path* (must exist), parses it as YAML,
validates the shape, and returns a normalised config dict ready for
`run_pipeline_file` to dispatch.

The file's schema - exactly one of `dataset:` / `importer:`, plus
`settings`, `detectors`, `chunk_size`, `tempimport`, `stream_results`,
`keep_negatives`, `import_labels` and `exporter` - is documented with an
annotated example in [`docs/CLI.md`](../../../docs/CLI.md#pipeline-file).
An `importer.outputs:` list takes the keys an `OutputSpec` takes (each
entry needs `media_type`) and is mutually exclusive with both
`importer.fields.media_type` and `importer.fields.source_specs`.

Unknown top-level keys raise `ValueError`; a missing file raises
`FileNotFoundError`. Field-key validation happens against the live
plugin registry - a typo in `importer.name` or any `fields.*` key
fails at parse time before media touches RAM.

```python
from vtscore.cli_pipeline import load_pipeline_file, run_pipeline_file

config = load_pipeline_file("pipeline.yaml")
# config is a fully-validated dict; inspect or mutate before running.

run_pipeline_file("pipeline.yaml")
# Loads + dispatches against vtscore.cli._run_source.
```

`run_pipeline_file(path)` is the "do everything" wrapper: it calls
`load_pipeline_file`, runs the optional `import_labels` block, then
dispatches to `_run_source` - the same path the flag entry points take, so
the dataset is saved unless `tempimport: true` - with
`override_detectors` set to the `detectors:` list (else
`[import_labels.detector]`), so the YAML file can declare
a detector list inline without mutating `settings.json`. A
`FileNotFoundError` / `ValueError` goes through `cli_progress.emit_error`
(`Error: ...` on stderr, or an `error` event in JSON mode) and
`sys.exit(1)`. Its output follows the `cli_progress` format, so call
`cli_progress.set_format("json")` first for an NDJSON stream; the label
import is a `labels_imported` event, as it is for `--import-labels-into`.

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

Events emitted by `vtscore.cli`, `vtscore.cli_pipeline` and
`vtscore.cli_progress`. Every event includes `event` and `ts`; each row
lists the extra fields.

| Event              | Fields                                                            | Source                                |
|--------------------|-------------------------------------------------------------------|---------------------------------------|
| `chunk_start`      | `chunk_num: int`, `chunk_size: int`                               | `_score_chunk` in `cli.py`            |
| `chunks_done`      | `total_medias: int`, `chunks: int`                                | `_run_live_pipeline` in `cli.py`      |
| `detector_skipped` | `detector: str`, plus reason-specific fields                      | `_load_and_train_detectors`           |
| `detector_reclip`  | `detector`, `detector_input_spec`, `dataset_input_spec`, plus `override_input_spec` when the clipper override did it | `_input_spec_reclip` / `_override_reclip` in `cli.py` |
| `detector_unchecked` | `detector`, `beta`, `status` (`unchecked`), `count` (the exported set's size) | `_record_line_state` in `cli.py` |
| `detector_centroid` | `detector`, `tier` (`centroid`), `n_good`, `n_bad`, `goods_owed`, `bads_owed`, `good_quota`, `bad_quota`, `dry_bad_quota` (#4643), `hold_labels`, `hold_owed` (`0`) | `_report_centroid` in `cli.py` |
| `detector_query`   | `detector`, `query` (the typed query), `tier` (`query`), `n_good`, `n_bad`, `goods_owed`, `bads_owed`, `good_quota`, `bad_quota`, `dry_bad_quota` (#4758), `hold_labels`, `hold_owed` (past the quota, held on the query, #4889) | `_report_centroid` in `cli.py` |
| `medias_skipped`   | `skipped: int`, `skipped_ids` (first 100), `embedder`             | `_emit_skipped_medias` in `cli.py`    |
| `medias_unembedded`| `unembedded: int`, `unembedded_ids` (first 100)                   | `_embed_loaded_medias` in `cli.py`    |
| `detection_skipped` | `reason`, plus `dataset` (its name) in a multi-dataset run       | `_emit_detection_skipped` in `cli.py` |
| `dataset_saved`    | `dataset_id`, `name`, `num_items`, `pkl_path`, `already_saved`    | `_save_source_dataset` / `_emit_dataset_saved` in `cli.py` |
| `dataset_start`    | `name`, `media_type`, `category`, `dataset_id` (`null` unless saved), `index` (0-based), `count` | `_RunDataset.announce` in `cli.py`, per dataset of a multi-dataset run |
| `dataset_failed`   | `name`, `error`                                                   | `_save_multi_source_datasets` / `_temporary_output_runs` in `cli.py` |
| `dataset_deleted`  | `dataset_id`, `name`                                              | `_delete_after_detection` in `cli.py` |
| `labels_imported`  | `detector`, `applied`, `skipped`, `created`                       | `_import_labels` in `cli_pipeline.py` |
| `export_complete`  | `message: str`, optional `open_url` (validated `http(s)` URL)     | `_run_exporter` in `cli.py`           |
| `dry_run_plan`     | `source`, `settings_path`, `autofind_detectors` (the detectors the run would score), `detectors_source` (`"autofind"` or `"override"`), `override_clipper` (the clipper override's `input_spec`, or `null`), `exporter`, `exporter_field_values` | `_emit_dry_run_plan`        |
| `progress`         | `status: str`, optional `message`, `current`, `total`, `pct`; `dataset` while a multi-dataset run imports | `progress_callback`, `_relay_import_progress` |
| `notification`     | `level`, `message`, `detail`, `source`, `items`                   | `notification_subscriber`             |
| `error`            | `message: str`                                                    | `emit_error` in JSON mode             |

Progress ticks with no `message` and `total <= 0` are dropped, so
consumers never see empty `{"status":"idle"}` records. In text mode a
saving run's import is narrated as one `import_progress` line per phase
instead; that event has no JSON form. The app tier (`vtsearch.cli_main`)
adds three more: `labels_imported` (same fields) for `--import-labels-into`,
`labels_import_dry_run` for it under `--dry-run`, and `authenticated_user`
(`user`) for `--user`.

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
- [`config.md`](config.md) explains `get_setting()` and
  `redirect_settings_path()`, which the pipeline reads its knobs through.
- [`utils.md`](utils.md) documents `build_media_hit`, the helper
  every detector-score row goes through before the exporter sees
  it.
