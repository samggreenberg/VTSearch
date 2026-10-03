<!-- This file is served raw at GET /api/achievements/docs/cli/raw and its
     footer phrase is hash-matched against _DOCS_RAW in
     vtsearch/achievements_catalog.py. Don't remove or reword the "Readme
     Reader code phrase" line without updating achievements_catalog.py to
     match. See CLAUDE.md. -->

# Command-line interface

Everything here goes through one entry point, `python app.py`. With no workflow flag it starts the web server; with `--autodetect` or `--pipeline` it imports a dataset, saves it to the dashboard, runs detectors over it, and exports the hits without starting the server (`--tempimport` discards the dataset instead of saving it); with `--list-plugins` it prints what is installed and exits. `python app.py --help` lists every flag.

- [Auto-detect (run detectors on a dataset)](#auto-detect-run-detectors-on-a-dataset) — the headless scoring run: sources, [saving the dataset or `--tempimport`](#saving-the-dataset-to-the-dashboard---tempimport), exporters, streaming, [dry runs](#dry-run-mode), [label import](#importing-labels-into-a-detector), [progress output](#progress-output-format)
- [Pipeline file](#pipeline-file) — the same run declared in YAML, for cron and CI
- [Web server modes](#web-server-modes) — port, logging, gunicorn, login providers, and the admin-set restrictions (solo media type, hidden plugins, retention, …)
- [Inspecting plugins and the API schema](#inspecting-plugins-and-the-api-schema)

## Auto-detect (run detectors on a dataset)

Import a dataset, save it to the dashboard, score every item with the
detectors flagged for Auto-Find, and output the items each model predicts as
"Good." Add `--tempimport` to score it without keeping it (see [Saving the
dataset to the dashboard](#saving-the-dataset-to-the-dashboard---tempimport)).

Models are specified via a **settings file** (`--settings`) whose
`autofind_detectors` list names registered models.  Each name
maps to a JSON file under `data/detectors/` named after a **slug** of the
detector name, not the name itself (see [Detector file
names](#detector-file-names) below); the CLI re-resolves the
labelset's origins, embeds them with the dataset's embedder, trains a
head, and applies it to the dataset.  See below for the exact format.
`--import-labels-into NAME` replaces that list for the run: NAME is then the
only detector scored (see [Importing labels into a
detector](#importing-labels-into-a-detector)).

### Which user's Auto-Find list runs

`autofind_detectors` (and the Auto-Find results exporter) are **per-user**:
each user curates their own list on the Dashboard's **AutoRun** detector tab
(move a detector between **Drafts** and **AutoRun** with its ⋯ menu). By default the
CLI runs as the built-in **`default`** user. Its list is the one the web app's
**AutoRun** tab edits on a server without logins; only when that user has no
list of its own does the run fall back to the `--settings` file's
`autofind_detectors` (so the flat-file workflow above still works on a fresh
install).

To run *another* user's Auto-Find list (e.g. a nightly cron of their favorite
detectors), authenticate with `--user` + `--api-key`, mirroring the server's
`api_key` login:

```bash
python app.py --autodetect --dataset data.pkl --user alice --api-key "$ALICE_KEY"
```

The key is checked against `data/api_keys.json` (the same file the server's
`--login api_key` uses); on success the run reads `alice`'s Auto-Find list and
results exporter, and the imported dataset is saved to `alice`'s dashboard.
Without `--user`, the `default` user (and the `--settings` file) applies.

### Saving the dataset to the dashboard (`--tempimport`)

By default the CLI **keeps what it imports**. The source (`--dataset` or
`--importer`) is imported through the same load pipeline the dashboard's
**Add dataset** uses — clipping, embedding, duplicate collapse, the saved
`.pkl` under `data/saved_datasets/` and the registry entry — so the next time
the UI is opened (or right away, if a server is running against the same data
directory) the dataset is on the dashboard, owned by the user the run ran as.
Detection then runs over that saved copy, so its hits are the ones Find would
give on that dashboard row.

```bash
# Import a folder, save it to the dashboard, and run the Auto-Find detectors on it.
python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --settings settings.json
```

With a saving run the import is the point and detection is the extra: when no
Auto-Find detector is configured, or none applies to the dataset's media type,
the dataset is still saved and the run ends with a `Detection skipped: …` note
and exit status 0. That also makes `--autodetect` with an empty Auto-Find list
a plain headless import.

A `--dataset` pickle is **copied** in (the dashboard deletes a dataset's pickle
when the dataset is deleted, so it never adopts your file). Pointing
`--dataset` at a pickle the dashboard already holds (one under
`data/saved_datasets/`) imports nothing new; the run just scores it.

**`--tempimport`** makes the run temporary instead — the behaviour
`--autodetect` had before datasets were saved: the source is scored straight
from the importer and nothing is kept. Having no applicable detector is then an
error, since the run would do nothing. `--tempimport` implies `--autodetect`,
though spelling out both reads better in scripts:

```bash
# Score and discard: nothing is added to the dashboard.
python app.py --autodetect --tempimport --dataset data.pkl --settings settings.json
```

A saving run holds the whole dataset in memory while it imports, exactly as a
GUI import does; `--chunk-size` then bounds only the scoring pass over the saved
copy. `--stream-results` exists for sources too large to hold, so it **requires
`--tempimport`** — the run is refused otherwise.

**From a pickle file:**

```bash
python app.py --autodetect --dataset path/to/dataset.pkl --settings settings.json
```

**From any supported data source** (folder, archive file, HTTP archive):

```bash
python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --settings settings.json
# --path may point at a single archive file, which is extracted and imported:
python app.py --autodetect --importer server_folder --path /data/sounds.zip --media-type audio --settings settings.json
# --dig-archives also extracts any archives found inside the scanned folder:
python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --dig-archives --settings settings.json
# http_archive accepts a web URL or a local server path to an archive:
python app.py --autodetect --importer http_archive --url https://example.com/data.zip --settings settings.json
python app.py --autodetect --importer http_archive --url /data/sounds.tar.gz --media-type audio --settings settings.json
```

Use `python app.py --list-importers` to see all available importers. The full set includes: `server_folder`, `server_files`, `local_folder`, `local_files`, `local_archive_member`, `pickle`, `http_archive`, `combine_datasets`, `demo`, `synthetic`. Each importer adds its own flags; run `python app.py --autodetect --importer <name> --help` to see them. `--help` resolves the named plugin first, so its flags are listed at the end of the usual help output (the same works for `--exporter <name> --help`).

**Reference mode**: importers that offer a "Reference files in place" checkbox
in the GUI (`server_folder`, `server_files`) expose it here as
`--reference-files` / `--no-reference-files`. Enabled, the dataset stores a path
reference to each original file instead of its bytes, which saves memory —
and, as in the GUI, makes the run depend on those files staying put. It is
**off by default**, so a CLI import ingests a source exactly the way the same
importer does in the GUI.

Leave it off unless every item really is re-readable from its original
location. Reference mode swaps a media's bytes for a path, so an item that has
no file to point back at (a remote source with no local copy) keeps neither —
it cannot be embedded, and is then skipped at scoring. That silently shortens
the hit list *and* moves the detector's threshold, because the threshold is
calibrated against the population actually being scored.

**Chunked loading**: for large datasets, use `--chunk-size N` to process in batches to limit memory:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --chunk-size 1000
python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio --settings settings.json --chunk-size 500
```

`--chunk-size` bounds the *loading and embedding* working set of a
`--tempimport` run, but the default flow still accumulates every hit in memory
and buffers the whole result set before the exporter writes it. For a media
source with more items (and more hits) than fit in RAM — e.g. a folder tree of
billions of images — add `--stream-results` (requires `--tempimport`,
`--chunk-size` and a streaming-capable exporter: `server_json_file`,
`server_csv_file`, `gui`, `webhook`, or `email_smtp`):

```bash
python app.py --autodetect --tempimport --importer server_folder --path /data/images \
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

The delivery exporters stream too, but batch rather than flush per hit:
`webhook` POSTs the hits in `--batch-size` groups (each request body carries the
run metadata, a zero-based `batch_index`, and a `hits` array), and `email_smtp`
sends one email per `--batch-size` hits. Both default to 500 hits per batch and
always deliver at least once (even for a zero-hit run), so the receiver learns
the run happened.

**Exporting results**: by default results are printed to the console. Add `--exporter <name>` to send them elsewhere:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter server_json_file --filepath results.json
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter server_csv_file --filepath results.csv
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter webhook --url https://example.com/hook
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter email_smtp --to recipient@example.com
```

Available exporters: `server_json_file` (JSON to server path), `server_csv_file` (CSV to server path), `webhook` (HTTP POST, optional `--auth-header`), `email_smtp` (SMTP email, requires `--to`), `portable_detector` (standalone ONNX scoring bundles; see below), `gui` (default: print to console), `open_url` (open a scheme-validated URL per hit, useful for hand-off to another tool). Run `python app.py --list-exporters` for the current set.

**Exporting the detectors themselves** (`portable_detector`): instead of the
scored hits, write one standalone, portable scoring bundle per detector the run
trained — the ONNX head (sigmoid baked in) plus a `manifest.json` and `README.md`,
carrying **no embeddings and no raw media**. It lets CI/automation produce a
shareable scoring model; the request-scoped equivalent is
`POST /api/detectors/{detector_id}/portable-bundle`
(see [`docs/api/detectors.md`](api/detectors.md#export-portable-bundle)). There is
deliberately no GUI affordance for either — the bundle is an expert artifact, and
as a dashboard menu item it read as a confusing second "export" beside **Export
labels**. The `--dataset`/`--importer` still supplies the embedder space the
detector trains in; the media is embedded but the hits are discarded.

```bash
# One bundle per Auto-Find detector, named after the detector.
python app.py --autodetect --dataset data.pkl --settings settings.json \
  --exporter portable_detector --filepath 'data/{detector_name}-detector.zip'
```

The `--filepath` accepts `{detector_name}` (and the date variables
`{YYYYMMDD-HHMMSS}`, `{YYYYMMDD}`, `{YYYY}`, `{MM}`, `{DD}`, `{username}`). When
the path omits `{detector_name}` and the run trained more than one detector, the
detector name is inserted before the extension so bundles don't overwrite each
other. Detectors whose scoring isn't a plain forward pass over one whole-item
vector (patch DINOv2/v3, structural SIFT/VLAD) are skipped with a note rather
than failing the run.

**Scoring across source types (converter routing).** A detector declares the
embedding space it needs (its `media_type`); it does not store a converter. When
the dataset's media are a different type, the CLI routes them to the detector's
type through a one-hop converter from the registry, so **one image detector
scores native images, videos (via `video2image`), and documents (via
`document2image`) in the same run** — a dataset can even mix all three. Media of
a type with no route to the detector's type are skipped for that detector, and a
detector whose type is unreachable from every source type in the dataset is
skipped entirely (with a note). When a converter fans one media out into several
(a video into frames), the per-clip scores are aggregated back to the source
media by **max**: the video is a positive hit when *any* of its frames clears the
threshold, and it surfaces as a single hit on the video, not one per frame.

**The CLI scores a media exactly as the GUI does.** Both go through one row
builder (`scoring_rows_for_snap`), so on a patch dataset a media's score is the
max over its score rows — image-level vector plus every raw patch — not the
image-level vector alone. That is also the geometry the detector's threshold was
calibrated on, so a CLI run and a GUI Find agree on which media clear it (issue
#3180).

**Matching the detector's clipper granularity.** A detector trained on a
specific clipper (its `input_spec.clipper` — e.g. 2-second audio tiles, or an
image grid) is **re-clipped at scoring time** when the loaded dataset wasn't
already clipped to match: each routed, target-typed media is split with the
detector's clipper and the clips are re-embedded, so a raw dataset is scored at
the granularity the detector expects. The per-clip scores fold back to the source
media by the same **max** rule as converter fan-out. A dataset already loaded
with the matching clipper is scored as-is (no redundant re-clip), and a detector
with no `input_spec.clipper` scores whole media.

**The threshold is calibrated on whatever the run ends up scoring.** Converting
and re-clipping change the population, not just the item count: the max over a
media's clips is never below the media's own whole-item score, so a cut fitted
on the loaded medias and applied to the routed ones sits systematically lower in
the distribution it decides — more hits than the algorithm chose, in a run whose
numbers all look reasonable. So the routing pass happens **before** calibration,
and each detector's cut is realized on the converted, re-clipped, re-embedded
snapshot its own scoring pass will read (issue #3647). On a natively-typed
dataset needing no re-clip the two are the same set and nothing changes; on a
converter-routed or re-clipped one the threshold moves, and moving it is the
fix. The first chunk is prepared once and handed to both passes, so the
correction costs no extra conversion or embedding work.

**The exported set is unchecked, and the run says so.** Each detector's line
keeps the top of its ranking at its balance (the `beta` setting, F-beta's
beta: 1 unless you change it; 2 leans toward recall, 0.5 toward precision):
the count at which the vote-anchored mixture's F-beta peaks, capped at the
top 32 unvoted items for beta 1 and 0.5 and the top 128 for beta 2. In the
app a spot check measures how much of that set is right and how much it
found; nobody can vote in a headless run, so the run exports the unchecked
set as it is. The run prints a line naming the detector and the size of the
set - `Detector 'det' exports its top 32 unchecked (at F1); nobody is here to
check it.` - which is a `detector_unchecked` event under
`--progress-format json` carrying `beta`, `status` and `count`, and every
result the detector produces carries a `balance` object beside its
`threshold` (`status` `unchecked`, with `count`), which the JSON exporters
write out with the hits. See [the line state](api/labeling.md#the-line-state).

**How to get the files:**

- **Dataset file** (only for `--dataset`; `--importer` reads the source media directly): a demo dataset loaded once in the web UI leaves a cached `.pkl` under `data/embeddings/`. The currently active dataset can also be downloaded as a pickle through the API, `GET /api/dataset/export` (see [`docs/api/datasets.md`](api/datasets.md#export-dataset)); the dashboard has no menu item for it.
- **Settings file**: A JSON file listing the detector names that should run during `--autodetect`. Each name maps to a JSON labelset under `data/detectors/` (see [Detector file names](#detector-file-names) below); the CLI re-resolves the labelset's origins, embeds them with the dataset's embedder, trains a fresh head, and scores the dataset.

```json
{
  "autofind_detectors": ["Dog Barks", "Cat Meows"],
  "detectors_dir": "data/detectors"
}
```

- **Detector file**: Created from the dashboard by labeling items in the right pane. The file stores origin info plus labels (no weights); the head is rebuilt from origins at scoring time.

#### Detector file names

Everything that names a detector — `autofind_detectors`, `--import-labels-into`,
the pipeline file's `detectors:` list — uses the **human-readable name**, exactly
as it appears in the UI (`Dog Barks`). The *filename* is a slug derived from that
name, so the two don't match literally: the name is lowercased and every run of
characters outside `[a-z0-9_-]` collapses to a single `_`, with leading and
trailing `_` stripped. So `Dog Barks` lives at `data/detectors/dog_barks.json`,
and `Bird calls (v2)` at `data/detectors/bird_calls_v2.json`. A name that slugs
to nothing falls back to `detector.json`, and a name long enough to overrun the
filesystem's limit is truncated with a short content hash appended so two long
names sharing a prefix can't collide.

Don't hand-construct these paths when you can avoid it: `--dry-run` (below)
prints the resolved file for every detector it would run, which is the reliable
way to check that a name in your settings file points where you think it does.

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
settings file, every detector listed under `autofind_detectors` (with its
media type and label count), and the exporter + its field values:

```
DRY RUN: no media will be loaded, embedded, scored, or exported.

Source:
  Importer: server_folder
  Params:
    path: /data/sounds
    media_type: audio
  Chunk size: whole dataset
  Save to dashboard: yes (the imported dataset is kept; --tempimport discards it)

Settings: settings.json
Auto-Find detectors (2):
  - Dog Barks  [media_type=audio, labels=12, file=data/detectors/dog_barks.json]
  - Cat Meows  [media_type=audio, labels=8, file=data/detectors/cat_meows.json]

Exporter: server_json_file
  filepath: out.json
```

When `--stream-results` is set, the plan adds a `Streaming: yes (...)` line
under the source (noting whether negatives are dropped or included), so a
streaming run can be sanity-checked before it starts. The `Save to dashboard`
line says whether the run would keep the dataset; a dry run never saves it.

`--dry-run` validates importer and exporter names, checks that the
dataset pickle (if given) exists, verifies required CLI fields are
populated, and reports any detector JSON files that are missing; so
typos in a cron-style invocation fail immediately instead of after a
multi-minute embedding pass. `--import-labels-into ... --label-importer-file ...`
is announced as part of the plan but skipped (no detector JSON is
modified), and the plan lists that detector alone, under
`Detectors (1; overrides the settings' Auto-Find list)`. In
`--progress-format json` the `dry_run_plan` event's `autofind_detectors` lists
whichever detectors the run would score, and `detectors_source` says where they
came from: `autofind` (the settings file) or `override` (`--import-labels-into`).

### Importing labels into a detector

`--import-labels-into` merges an external label file into a detector's labelset
*before* the run trains and scores, so a batch of labels produced elsewhere
(another VTSearch instance, an annotation tool, a script) can be folded in
headlessly. Three flags work together:

| Flag | Meaning |
|------|---------|
| `--import-labels-into NAME` | Detector to merge into, by its human-readable name (see [Detector file names](#detector-file-names)). |
| `--label-importer-file PATH` | The label file to read. Shorthand for `--label-importer-field filepath=PATH`. |
| `--label-importer-field KEY=VALUE` | Set any one of the label importer's fields; repeat it for several. This is how you drive a label importer that reads something other than a file. |
| `--label-importer NAME` | Which label importer to run. Defaults to `server_json_file`; `python app.py --list-label-importers` shows the rest. |
| `--create-detector` | Create the detector from the imported labels if it doesn't exist yet (see [Creating the detector](#creating-the-detector)). |
| `--detector-media-type TYPE` | Media type for a detector `--create-detector` creates. Defaults to the source's. |

`--import-labels-into` needs at least one of `--label-importer-file` or
`--label-importer-field`.

**The detector you import into is the one the run scores with.** The settings
file's `autofind_detectors` list (the Dashboard's **AutoRun** tab) is not
consulted, so the detector need not be on AutoRun, and nothing else on AutoRun
runs alongside it. Labelling a detector and running it over a new dataset is
therefore a single command that never needs the UI. The settings file still
supplies everything else it normally does — `detectors_dir`, and the Auto-Find
results exporter when there is no `--exporter`.

```bash
# Merge new labels into "Dog Barks", then score the dataset with it alone.
python app.py --autodetect --dataset data.pkl --settings settings.json \
    --import-labels-into "Dog Barks" --label-importer-file new_labels.json

# Same, from a CSV.
python app.py --autodetect --dataset data.pkl --settings settings.json \
    --import-labels-into "Dog Barks" \
    --label-importer server_csv_file --label-importer-file new_labels.csv
```

The two server-side importers expect labels keyed by media MD5:
`server_json_file` reads `{"labels": [{"md5": "...", "label": "good"}, ...]}`,
and `server_csv_file` reads a header row of `md5,label`. Only `good` and `bad`
labels are accepted; entries with any other label, and `(md5, label)` pairs the
detector already holds, are skipped and reported in the count.

#### Creating the detector

Without `--create-detector` the detector must already exist; a name with no
detector ends the run with `Detector 'NAME' not found.` before any media is
loaded. With it, a missing detector is created from the imported labels, so a
labelled detector can go from a label file to hits without ever being made in
the UI:

```bash
# Create "Dog Barks" from the label file, then score the folder with it.
python app.py --autodetect --importer server_folder --path /data/sounds --media-type audio \
    --import-labels-into "Dog Barks" --create-detector --label-importer-file dog_barks.json
```

The detector is written and registered the way the Dashboard's **New Detector**
does it, owned by the user the run runs as (`--user`, or the default user), so
it appears on that user's **Drafts** tab afterwards. When it already exists,
`--create-detector` changes nothing: the labels are merged into it as usual, so
the same command works on every run of a cron job.

Its **media type** is `--detector-media-type` when given, otherwise the
source's: the `media_type` recorded in a `--dataset` pickle, or the importer's
`--media-type` field. A source that declares neither (a legacy pickle, or an
importer with no media type field, such as `local_folder`) needs
`--detector-media-type`, and the run says so before loading anything. Set it
also when the detector should differ from the source, e.g. `image` to score
video frames through a converter.

The labels have to be able to train a head: at least one `good` and one `bad`
that resolve. Labels keyed only by MD5 (the `server_json_file` / `server_csv_file`
shape) resolve against the dataset the run is scoring, so they work when those
items are in it. An import with no `good` or `bad` label creates nothing.

Under `--dry-run` the announcement names the detector it would create and its
media type, and nothing is written; the plan lists the detector as `MISSING`,
since it does not exist yet.

The import is a **one-shot mutation of the detector JSON on disk**: the merged
labelset persists after the run, which is why it happens before scoring rather
than only for the run's duration. It is part of the `--autodetect` flow (or the
pipeline file's `import_labels:` block) — there is no standalone import mode.
Under `--dry-run` the import is announced but not performed, so no detector JSON
is touched.

### Progress output format

By default, `--autodetect` prints human-readable progress to the console. For
scripted or CI callers, add `--progress-format json` to emit
**newline-delimited JSON (NDJSON)** on stdout instead — one progress event per
line, which is easy to parse from a wrapping script:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --progress-format json
```

The two choices are `text` (default, prose) and `json` (NDJSON events). The
event schema is defined in `vtscore.cli_progress`.

A plugin can also raise a **notification** — a message it wants the user to
see about something it decided not to fail on ("skipped 3 unreadable files").
In the GUI those become toasts; on the command line they print as
`Warning: [Server Folder] Skipped 3 unreadable files` on **stderr** in text
mode, and as `notification` events on stdout in JSON mode:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json \
    --progress-format json \
  | jq -r 'select(.event == "notification") | "\(.level): \(.message)"'
```

A `notification` never ends the run, at any level — including
`"level": "error"`, which reports something the code continued past. The
fatal-error record is the separate `error` event.

A saving run (no `--tempimport`) reports the dataset it kept as a
`dataset_saved` event carrying its registry `dataset_id`, `name`, `num_items`
and `pkl_path` (`already_saved` is true when the pickle was already on the
dashboard), and a run that had no detector to use as `detection_skipped` with a
`reason`. While the import runs, its progress arrives as ordinary `progress`
events:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json \
    --progress-format json \
  | jq -r 'select(.event == "dataset_saved") | .dataset_id'
```

A media the scorer cannot embed — a corrupt image, an unresolvable thin path, a
pre-computed vector of the wrong width — is **skipped**, not fatal: one bad file
must not take a long run down with it. Each skip is reported as a
`medias_skipped` event carrying the count and the first of the affected ids, so
a hit count that is short of the file count is never silent:

```bash
python app.py --autodetect --importer server_folder --path /data/images \
    --media-type image --settings settings.json --progress-format json \
  | jq -r 'select(.event == "medias_skipped") | .text'
```

An exporter can format its results into a URL for a browser to open rather than
delivering them anywhere (the built-in `open_url` exporter does exactly this).
There is no browser on the command line, so the URL is printed under the
exporter's confirmation message, and carried as an `open_url` field on the
`export_complete` event in JSON mode — enough for a wrapping script to open it:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json \
    --progress-format json --exporter open_url \
    --url-template 'https://example.com/review?ids={ids}' \
  | jq -r 'select(.event == "export_complete") | .open_url'
```

## Pipeline file

For repeatable runs (cron, CI), put the whole autodetect invocation in a YAML
file and pass it via `--pipeline`:

```bash
python app.py --pipeline pipeline.yaml
```

The YAML supports every knob the `--autodetect` flag set does. It cannot be
combined with the other autodetect flags; declare everything inline. Like the
flags, a pipeline run saves its dataset to the dashboard unless the file sets
`tempimport: true`.

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

# Optional. When set, overrides settings.json's `autofind_detectors` list
# for this run only. The file on disk is NOT modified. When absent and
# `import_labels:` is set, the run scores with `import_labels.detector` alone.
detectors:
  - Dog Barks
  - Cat Meows

# Optional. Process medias in batches of N. Same as --chunk-size.
chunk_size: 1000

# Optional. Discard the imported dataset after detection instead of saving
# it to the dashboard (same as --tempimport). Off by default.
tempimport: false

# Optional. Stream each chunk's hits straight to the exporter instead of
# accumulating them (same as --stream-results). Requires tempimport: true,
# chunk_size and a streaming-capable exporter. Output is chunk-ordered, not
# globally sorted.
stream_results: false

# Optional. With stream_results, also emit below-threshold hits (label=bad).
# Same as --keep-negatives. Off by default.
keep_negatives: false

# Optional. One-shot merge of external labels into a detector before
# scoring (same as --import-labels-into / --label-importer /
# --label-importer-field). `importer` takes the same name + fields shape
# as `importer:` and `exporter:`, so any label importer works. Unless
# `detectors:` is set, this detector is then the only one the run scores.
import_labels:
  detector: Dog Barks             # the detector's name, not its filename slug
  importer:
    name: server_json_file         # default: server_json_file
    fields:
      filepath: new_labels.json
  create: false                    # true = create the detector if missing (--create-detector)
  # media_type: audio              # with create: the new detector's type (--detector-media-type)

# Optional. Where results go. Defaults to the `gui` exporter (console).
exporter:
  name: server_json_file
  fields:
    filepath: results.json
```

Plugin names (`importer.name`, `exporter.name`, `import_labels.importer.name`)
and their `fields` keys are validated against the registered plugins at load
time, so a typo fails fast before any media is loaded.

The shorter `import_labels` form is still accepted: `importer: <name>` given as
a plain string, with `file: <path>` as shorthand for the importer's `filepath`
field.

## Web server modes

**Development (Flask dev server)**: bind to `0.0.0.0:5000`:

```bash
python app.py
```

A `--local` flag is accepted for historical reasons and only changes the
banner text (`LOCAL` vs. `PRODUCTION`); the bind address is the same either
way. This entry point uses Flask's built-in dev server and is not
recommended for production.

**Port** (`--port`): bind the dev server to a port other than the default
`5000`. Precedence is `--port` > `VTSEARCH_PORT` env var > `5000`. This lets
several instances share a host (e.g. co-located single-GPU SLURM jobs on a
multi-GPU node). Gunicorn ignores this flag; under WSGI use `VTSEARCH_BIND`
instead.

```bash
python app.py --port 8080
```

**Verbose logging** (`-v` / `--verbose`): logging defaults to `WARNING`, which
keeps the console quiet — including the per-request access log. Pass `-v` to
raise the level to `INFO`, which turns on the dev-server access log (one
`GET /api/... 200` line per request) plus VTSearch's own INFO records; `-vv`
raises it to `DEBUG`. The flag only raises verbosity, so `-v` on top of
`VTSEARCH_LOG_LEVEL=debug` stays at DEBUG. It applies to both the web server and
`--autodetect`:

```bash
python app.py -v             # INFO + access log
python app.py -vv            # DEBUG
VTSEARCH_LOG_LEVEL=info python app.py   # same as -v, via env
```

Under gunicorn there is no `-v` flag; set `VTSEARCH_LOG_LEVEL=info` (or `debug`)
to get the same access log.

**Production (gunicorn)**: run the WSGI app under the bundled config:

```bash
VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app
```

`VTSEARCH_SERVER_INIT=1` runs the same startup sequence (model
initialization, embedder preloading) that `python
app.py` runs; gunicorn imports `app.py` rather than executing its
`__main__` block, so the env var is what triggers initialization. The
bundled Docker images already run gunicorn this way. See
[DEPLOYMENT.md](DEPLOYMENT.md#tuning) for tuning.

**Authentication mode** (`--login`): select the login provider (dev
server only; set up the provider in code when running under gunicorn).
Two providers are accepted:

```bash
python app.py --login trivial    # multi-user mode with simple username auth (cookie-based, no password)
python app.py --login api_key    # Bearer-key auth against data/api_keys.json
```

- **`trivial`** shows a username prompt (no password) and tracks the user via a
  cookie; useful for low-stakes multi-user setups.
- **`api_key`** authenticates each request via an `Authorization: Bearer <key>`
  header, checking the key against `data/api_keys.json`. This is the same key
  store the CLI's `--user` + `--api-key` flags use (see [Which user's Auto-Find
  list runs](#which-users-auto-find-list-runs) above), so a key minted for the
  server also works for a per-user `--autodetect` run.

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
`video`, `text`, `document`, `face`; `python app.py --list-media-types`
prints the installed set). This is an **admin-set server
restriction**, not a user preference: it applies to every user, users
cannot change or opt out of it from the Settings dialog (the Server tab
shows it read-only), and `PUT /api/settings` refuses to touch it. The
flag is a process-level override of the server-tier `solo_media_type`
key, so an operator can also set it persistently by writing
`"solo_media_type": "image"` into `data/settings.json`; the flag wins
for the lifetime of the process. The gunicorn-launched Docker images
never parse `argv`, so the same restriction is settable there as
`VTSEARCH_SOLO_MEDIA_TYPE=image` — an explicit flag wins over the
variable, and both run the same validation.

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
registered yet. Unlike `--solo-media-type`, this one is a per-process
**fallback** over a per-user setting: any user can override it
per-mediaType via the Settings dialog ("Ask each time" is the opt-out),
and their choice persists across restarts. Under Docker, set the same
locks as a comma-separated `VTSEARCH_SOLO_EMBEDDERS=image=siglip,audio=clap`;
an explicit flag wins.

**Hidden plugins** (`--hide-plugin family:name`, repeatable): drop a
plugin from picker / listing API responses for this deployment without
editing plugin code. The format is `family:name` where `family` is one
of the keys printed by `--list-plugins` (`importers`,
`datasource_importers`, `seed_importers`, `exporters`, `label_importers`,
`labelset_sources`, `converters`, `media_sources`, `media_types`,
`embedders`, `clippers`, `cleaners`, `settings_importers`,
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
every restart. The merge is a **union**: either source can add a hide,
and neither can un-hide what the other hid. Under Docker, pass the same
pairs comma-separated as
`VTSEARCH_HIDE_PLUGINS=converters:audio2image,embedders:e5`; an explicit
flag wins. Use `--list-plugins --format names` to discover the available
`family:name` pairs.

**Dataset retention** (`--dataset-max-age-days DAYS`): stamp every
dataset created by this server process with an expiry `DAYS` days after
creation; expired datasets are aged off from the registry. The value
must be a positive integer:

```bash
python app.py --dataset-max-age-days 14
```

Unlike the solo flags, this is a **server-wide override**, not a
per-user fallback: it applies to every user, overrides the persisted
`dataset_max_age_days` in the settings file for the lifetime of the
process, and is **not** editable via the Settings dialog or the
settings API (it is exposed read-only so the dashboard can show the
Age-Off column). Omit the flag to use the persisted value (no expiry if
none is set).

Because the Docker images launch under gunicorn (which never parses
`argv`), the same override is also available as the
`VTSEARCH_DATASET_MAX_AGE_DAYS` environment variable, honored at server
init; an explicit `--dataset-max-age-days` flag wins over it. The
LabBench image pins `VTSEARCH_DATASET_MAX_AGE_DAYS=14`.

**Support email** (`--support-email ADDRESS`): set the recipient for the
Help modal's "Email us" contact link so it opens a pre-addressed compose
window:

```bash
python app.py --support-email support@example.org
```

Like `--dataset-max-age-days`, this is a **server-wide override**: it
applies to every user, overrides the persisted `support_email` in the
settings file for the lifetime of the process, and is **not** editable
via the Settings dialog or the settings API (it is exposed read-only so
the frontend can build the `mailto:` link). Omit the flag to use the
persisted value, which defaults to the built-in project address. The
same override is also available as the `VTSEARCH_SUPPORT_EMAIL`
environment variable for the gunicorn-launched Docker images; an
explicit `--support-email` flag wins over it.

**Semantic embedders only** (`--semantic-only`): lock the instance to the
**Semantic** embedder type, hiding the still-prototype **Patch Semantic**
and **Structural** types from every surface:

```bash
python app.py --semantic-only
```

With the lock on:

- `GET /api/embedders` withholds every patch / structural embedder, so
  Add Dataset ▸ Advanced shows no "Region embedder" / "Instance embedder"
  picker and the primary Embedder picker lists Semantic embedders only.
- The New-detector modal drops its "Detector Embedder Type" picker (one
  option is not a choice) and creates Semantic detectors.
- `POST /api/detectors` and the dataset-import routes reject a
  patch/structural type with **400**, so a stale client or a hand-rolled
  request can't bind one behind the UI's back.

This is a coarser tool than `--hide-plugin embedders:<name>`, which hides
one named embedder: use `--semantic-only` when you want the whole
prototype tier gone and don't want to track which embedders belong to it.

Like `--dataset-max-age-days` and `--support-email`, this is a
**server-wide override**: it applies to every user, overrides the
persisted `semantic_only` in the settings file for the lifetime of the
process, and is **not** editable via the Settings dialog or the settings
API (it is exposed read-only, and the Settings ▸ Server tab reports it).
The flag can only *enable* the lock — there is no `--no-semantic-only` —
so a deployment that sets `semantic_only: true` in its settings file
can't have the restriction loosened by a stray flag. The same override is
available as the `VTSEARCH_SEMANTIC_ONLY` environment variable (`1` /
`true` / `yes` / `on`) for the gunicorn-launched Docker images; an
explicit `--semantic-only` flag wins over it.

**Hide ingest ETAs** (`--hide-ingest-eta`): drop the remaining-time
estimate from ingest progress bars (dataset imports, staging imports,
and a labelset's missing-media fetch), for a deployment where those jobs
are too erratic for any timing profile to predict:

```bash
python app.py --hide-ingest-eta
```

The bars still fill and show their counts, so users can see an import
is moving; they just don't show a remaining-time estimate. Every other
progress bar keeps its ETA. Like `--semantic-only`, this is a
**server-wide override** that can only switch the estimates off: there
is no `--no-hide-ingest-eta`, the persisted `hide_ingest_eta` setting in
the settings file can turn it on too, and neither is editable via the
Settings dialog or the settings API (the Settings ▸ Server tab reports
it). The env-var equivalent is `VTSEARCH_HIDE_INGEST_ETA`; an explicit
flag wins over it. See
[When ingest ETAs can't be trusted](DEPLOYMENT.md#when-ingest-etas-cant-be-trusted).

## Inspecting plugins and the API schema

`python app.py --list-plugins` enumerates every auto-discovered plugin;
dataset importers, exporters, label importers/sources, settings I/O,
media converters/types/embedders/clippers/cleaners, and media sources; and
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
# Also: --list-datasource-importers, --list-seed-importers,
# --list-converters, --list-clippers, --list-cleaners,
# --list-media-types, --list-media-sources, --list-label-importers,
# --list-labelset-sources, --list-settings-importers,
# --list-settings-exporters, --list-settings-sources.
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
