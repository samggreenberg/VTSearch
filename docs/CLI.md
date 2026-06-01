# Command-line interface

VTSearch provides a CLI workflow for running detectors on datasets and exporting results, all without starting the web server.

## Auto-detect (run detectors on a dataset)

Score every item in a dataset with the detectors flagged for
autorun and output the items each model predicts as "Good."

Models are specified via a **settings file** (`--settings`) whose
`autorun_detectors` list names registered models.  Each name
maps to `data/detectors/<name>.json`; the CLI re-resolves the
labelset's origins, embeds them with the dataset's embedder, trains an
MLP, and applies it to the dataset.  See below for the exact format.

**From a pickle file:**

```bash
python app.py --autodetect --dataset path/to/dataset.pkl --settings settings.json
```

**From any supported data source** (folder, HTTP archive):

```bash
python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --settings settings.json
python app.py --autodetect --importer http_archive --url https://example.com/data.zip --settings settings.json
```

Use `python app.py --list-importers` to see all available importers. The full set includes: `server_folder`, `server_files`, `local_folder`, `local_files`, `pickle`, `http_archive`, `combine_datasets`, `demo`, `synthetic`. Each importer adds its own flags; run `python app.py --autodetect --importer <name> --help` to see them.

**Chunked loading**: for large datasets, use `--chunk-size N` to process in batches to limit memory:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --chunk-size 1000
python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --settings settings.json --chunk-size 500
```

`--chunk-size` bounds the *loading and embedding* working set, but the default
flow still accumulates every hit in memory and buffers the whole result set
before the exporter writes it. For a media source with more items (and more
hits) than fit in RAM — e.g. a folder tree of billions of images — add
`--stream-results` (requires `--chunk-size` and a streaming-capable exporter:
`server_json_file`, `server_csv_file`, or `gui`):

```bash
python app.py --autodetect --importer server_folder --path /data/images \
  --media-type image --settings settings.json --chunk-size 500 \
  --stream-results --exporter server_json_file --filepath hits.ndjson
```

With `--stream-results` the folder is enumerated lazily (the full file list is
never held in memory), each chunk's hits are written straight to the exporter,
and nothing accumulates across chunks. `server_json_file` switches to
newline-delimited JSON (NDJSON): a metadata header line followed by one hit per
line. The tradeoff: streamed hits are ordered by chunk, **not** globally sorted
by score (sort the NDJSON afterwards if you need a global ranking). Only
above-threshold (predicted-good) hits are written; add `--keep-negatives` to
also stream the below-threshold items (tagged `label=bad`).

**Exporting results**: by default results are printed to the console. Add `--exporter <name>` to send them elsewhere:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter server_json_file --filepath results.json
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter server_csv_file --filepath results.csv
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter webhook --url https://example.com/hook
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter email_smtp --to recipient@example.com
```

Available exporters: `server_json_file` (JSON to server path), `server_csv_file` (CSV to server path), `webhook` (HTTP POST, optional `--auth-header`), `email_smtp` (SMTP email, requires `--to`), `gui` (default: print to console).

**How to get the files:**

- **Dataset file**: Export from the web UI via the dataset menu ("Export dataset"), or use a cached `.pkl` file from the `data/embeddings/` directory after loading a demo dataset.
- **Settings file**: A JSON file listing the detector names that should run during `--autodetect`. Each name maps to a JSON labelset under `data/detectors/<name>.json`; the CLI re-resolves the labelset's origins, embeds them with the dataset's embedder, trains a fresh MLP, and scores the dataset.

```json
{
  "autorun_detectors": ["Dog Barks", "Cat Meows"],
  "detectors_dir": "data/detectors"
}
```

- **Detector file**: Created from the dashboard by labeling items in the right pane. The file stores origin info plus labels (no weights); the MLP is rebuilt from origins at scoring time.

**Example output:**

```
Predicted Good (5 items):

  1-34094-A-6.wav
  1-30226-A-0.wav
  1-17150-B-2.wav
  1-22694-A-4.wav
  1-77445-A-1.wav
```

Items with origin information include the origin display string before the filename.

### Dry-run mode

Add `--dry-run` to any `--autodetect` invocation to print the plan
without loading media, training detectors, scoring, or exporting:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --dry-run
python app.py --autodetect --importer server_folder --path /data/sounds \
    --media-type audio --settings settings.json --exporter server_json_file \
    --filepath out.json --dry-run
```

The output names the source (pickle file or importer + params), the
settings file, every detector listed under `autorun_detectors` (with its
media type and label count), and the exporter + its field values:

```
DRY RUN: no media will be loaded, embedded, scored, or exported.

Source:
  Importer: server_folder
  Params:
    path: /data/sounds
    media_type: audio
  Chunk size: whole dataset

Settings: settings.json
Autorun detectors (2):
  - Dog Barks  [media_type=audio, labels=12, file=data/detectors/Dog Barks.json]
  - Cat Meows  [media_type=audio, labels=8, file=data/detectors/Cat Meows.json]

Exporter: server_json_file
  filepath: out.json
```

`--dry-run` validates importer and exporter names, checks that the
dataset pickle (if given) exists, verifies required CLI fields are
populated, and reports any detector JSON files that are missing; so
typos in a cron-style invocation fail immediately instead of after a
multi-minute embedding pass. `--import-labels-into ... --label-importer-file ...`
is announced as part of the plan but skipped (no detector JSON is
modified).

## Pipeline file

For repeatable runs (cron, CI), put the whole autodetect invocation in a YAML
file and pass it via `--pipeline`:

```bash
python app.py --pipeline pipeline.yaml
```

The YAML supports every knob the `--autodetect` flag set does. It cannot be
combined with the other autodetect flags; declare everything inline.

```yaml
# Pick exactly one source.
dataset: data/sounds.pkl
# --- or ---
importer:
  name: server_folder              # see `python app.py --list-importers`
  fields:                          # importer-specific PluginField values
    path: /data/sounds
    media_type: audio
    recursive: true

# Optional. Path to the same settings JSON the --settings flag accepts.
# Defaults to data/settings.json.
settings: settings.json

# Optional. When set, overrides settings.json's `autorun_detectors` list
# for this run only. The file on disk is NOT modified.
detectors:
  - Dog Barks
  - Cat Meows

# Optional. Process medias in batches of N. Same as --chunk-size.
chunk_size: 1000

# Optional. Stream each chunk's hits straight to the exporter instead of
# accumulating them (same as --stream-results). Requires chunk_size and a
# streaming-capable exporter. Output is chunk-ordered, not globally sorted.
stream_results: false

# Optional. With stream_results, also emit below-threshold hits (label=bad).
# Same as --keep-negatives. Off by default.
keep_negatives: false

# Optional. One-shot merge of an external label file into a detector
# before scoring (same as --import-labels-into / --label-importer /
# --label-importer-file).
import_labels:
  detector: dog-barks
  importer: server_json_file       # default: server_json_file
  file: new_labels.json

# Optional. Where results go. Defaults to the `gui` exporter (console).
exporter:
  name: server_json_file
  fields:
    filepath: results.json
```

Plugin names (`importer.name`, `exporter.name`, `import_labels.importer`) are
validated against the registered plugins at load time, so a typo fails fast
before any media is loaded.

## Web server modes

**Development (Flask dev server)**: bind to `0.0.0.0:5000`:

```bash
python app.py
```

A `--local` flag is accepted for historical reasons and only changes the
banner text (`LOCAL` vs. `PRODUCTION`); the bind address is the same either
way. This entry point uses Flask's built-in dev server and is not
recommended for production.

**Production (gunicorn)**: run the WSGI app under the bundled config:

```bash
VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app
```

`VTSEARCH_SERVER_INIT=1` runs the same startup sequence (model
initialization, autoload preloading, settings-source sync) that `python
app.py` runs; gunicorn imports `app.py` rather than executing its
`__main__` block, so the env var is what triggers initialization. The
bundled Docker images already run gunicorn this way. See
[DEPLOYMENT.md](DEPLOYMENT.md#tuning) for tuning.

**Authentication mode** (`--login`): select the login provider (dev
server only; set up the provider in code when running under gunicorn):

```bash
python app.py --login trivial    # multi-user mode with simple username auth
```

Without `--login`, the app uses `DefaultLoginProvider` (single-user, always authenticated).

**Solo mediaType** (`--solo-media-type`): streamline the UI for users
who only ever look at one media type (e.g. images, optionally pulled
in via converters from videos/documents). When set, the dataset
importer and new-detector flows hide their mediaType pickers and lock
to this type, the converter list filters to converters whose output is
this type, and the type's default embedder is warmed at startup:

```bash
python app.py --solo-media-type image
```

Valid values are the registered media-type ids (`audio`, `image`,
`video`, `text`, `document`). The flag is a per-process **fallback**:
any user who explicitly sets their own solo mediaType (or explicitly
opts back into "show everything") via the Settings dialog overrides
the CLI value for themselves, and the choice persists across restarts.
A user who has never touched the setting sees the CLI value.

**Solo mediaEmbedder** (`--solo-embedder`): lock the embedding model
for one or more mediaTypes so the dataset-importer modal hides its
embedder picker for those types and silently uses the named embedder.
Repeatable, one `--solo-embedder` per mediaType; the format is
`TYPE=EMBEDDER`:

```bash
python app.py --solo-embedder image=siglip --solo-embedder audio=clap
```

Other mediaTypes still show the normal embedder picker. The flag warms
each locked embedder at startup even when no datasets or detectors are
registered yet. Same fallback semantics as `--solo-media-type`: any
user can override per-mediaType via the Settings dialog ("Ask each
time" is the opt-out), and their choice persists across restarts.

**Hidden plugins** (`--hide-plugin family:name`, repeatable): drop a
plugin from picker / listing API responses for this deployment without
editing plugin code. The format is `family:name` where `family` is one
of the keys printed by `--list-plugins` (`importers`, `exporters`,
`label_importers`, `labelset_sources`, `converters`, `media_sources`,
`media_types`, `embedders`, `clippers`, `settings_importers`,
`settings_exporters`, `settings_sources`) and `name` is the plugin's
registered name:

```bash
python app.py --hide-plugin converters:audio2image \
              --hide-plugin embedders:e5 \
              --hide-plugin importers:synthetic
```

Hidden plugins remain importable and callable by name via execution
endpoints (autodetect, label import, etc.); this is a UI declutter,
not a security boundary. The CLI flag merges with the persisted
`hidden_plugins` key in the server settings file (`data/settings.json`
or whatever path `--settings` points at), where a deployment can set
`{"hidden_plugins": {"converters": ["audio2image"]}}` and pick it up on
every restart. Use `--list-plugins --format names` to discover the
available `family:name` pairs.

## Inspecting plugins and the API schema

`python app.py --list-plugins` enumerates every auto-discovered plugin;
dataset importers, exporters, label importers/sources, settings I/O,
media converters/types/embedders/clippers, and media sources; and
exits without starting the server. Three output formats:

```bash
python app.py --list-plugins                          # human-readable
python app.py --list-plugins --format json            # machine-readable
python app.py --list-plugins --format names           # one "family:name" per line
python app.py --list-plugins --plugin-family importers --format names
                                                      # one bare name per line (completion-friendly)
```

Per-family shortcuts are available for every plugin family; they're
equivalent to `--list-plugins --plugin-family <family>` and accept the
same `--format` flag:

```bash
python app.py --list-importers                        # dataset importers
python app.py --list-exporters --format names         # results exporters, bare names
python app.py --list-embedders --format json          # embedders as JSON
# Also: --list-converters, --list-clippers, --list-media-types,
# --list-media-sources, --list-label-importers, --list-labelset-sources,
# --list-settings-importers, --list-settings-exporters, --list-settings-sources.
```

Use `--format names --plugin-family <family>` (or any `--list-<family>
--format names` shortcut) from a shell-completion script to suggest
valid values for `--importer`, `--exporter`, etc.

The HTTP API's machine-readable OpenAPI 3.0 spec is served at
`GET /api/openapi.json` (and browsable via Swagger UI at `GET /api/docs`)
on the running server. See
[API.md § Machine-readable schema](API.md#machine-readable-schema).

---

*Readme Reader code phrase:* `command palette unlocked`
