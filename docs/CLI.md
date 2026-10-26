<!-- This file is served raw at GET /api/achievements/docs/cli/raw and its
     footer phrase is hash-matched against _DOCS_RAW in
     vtsearch/achievements_catalog.py. Don't remove or reword the "Readme
     Reader code phrase" line without updating achievements_catalog.py to
     match. See CLAUDE.md. -->

# Command-line interface

Everything here goes through one entry point, `python app.py`. With no workflow flag it starts the web server; with `--autodetect` or `--pipeline` it imports a dataset, saves it to the dashboard, runs detectors over it, and exports the hits without starting the server (`--tempimport` discards the dataset instead of saving it); with `--list-plugins` it prints what is installed and exits. `python app.py --help` lists every flag.

- [Auto-detect (run detectors on a dataset)](#auto-detect-run-detectors-on-a-dataset) — the headless scoring run: [whose AutoFind list](#which-users-autofind-list-runs), [saving the dataset or `--tempimport`](#saving-the-dataset-to-the-dashboard---tempimport), [sources](#sources), [chunking and streaming](#large-datasets-chunking-and-streaming), [exporters](#exporting-results), [how the run scores](#how-the-run-scores), [settings and detector files](#settings-and-detector-files), [dry runs](#dry-run-mode), [label import](#importing-labels-into-a-detector), [progress output and exit status](#progress-output-format)
- [Pipeline file](#pipeline-file) — the same run declared in YAML, for cron and CI
- [Headless runs vs. the web server](#headless-runs-vs-the-web-server) — which startup steps, settings and flags each path applies, and why
- [Web server modes](#web-server-modes) — port, logging, gunicorn, login providers, and the admin-set restrictions (solo media type, hidden plugins, retention, …)
- [Inspecting plugins and the API schema](#inspecting-plugins-and-the-api-schema)

## Auto-detect (run detectors on a dataset)

Import a dataset, save it to the dashboard, score every item with the
detectors flagged for AutoFind, and output the items each model predicts as
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

### Which user's AutoFind list runs

`autofind_detectors` (and the AutoFind results exporter) are **per-user**:
each user curates their own list on the Dashboard's **AutoFind** detector tab
(move a detector between **Drafts** and **AutoFind** with its ⋯ menu). By default the
CLI runs as the built-in **`default`** user. Its list is the one the web app's
**AutoFind** tab edits on a server without logins; only when that user has no
list of its own does the run fall back to the `--settings` file's
`autofind_detectors` (so the flat-file workflow above still works on a fresh
install).

To run *another* user's AutoFind list (e.g. a nightly cron of their favorite
detectors), authenticate with `--user` + `--api-key`, mirroring the server's
`api_key` login:

```bash
python app.py --autodetect --dataset data.pkl --user alice --api-key "$ALICE_KEY"
```

The key is checked against `data/api_keys.json` (the same file the server's
`--login api_key` uses); on success the run reads `alice`'s AutoFind list and
results exporter, and the imported dataset is saved to `alice`'s dashboard.
Without `--user`, the `default` user (and the `--settings` file) applies. A
[pipeline file](#pipeline-file) run takes the same two flags.

### Saving the dataset to the dashboard (`--tempimport`)

By default the CLI **keeps what it imports**. The source (`--dataset` or
`--importer`) is imported through the same load pipeline the dashboard's
**Add dataset** uses — clipping, embedding, duplicate collapse, the saved
`.pkl` under `data/saved_datasets/` and the registry entry — so the next time
the UI is opened (or right away, if a server is running against the same data
directory) the dataset is on the dashboard, owned by the user the run ran as.
Detection then runs over that saved copy, so its hits are the ones Find would
give on that dashboard row.

With a saving run the import is the point and detection is the extra: when no
AutoFind detector is configured, or none applies to the dataset's media type,
the dataset is still saved and the run ends with a `Detection skipped: …` note
and exit status 0. That also makes `--autodetect` with an empty AutoFind list
a plain headless import.

A `--dataset` pickle is **copied** in (the dashboard deletes a dataset's pickle
when the dataset is deleted, so it never adopts your file). Pointing
`--dataset` at a pickle the dashboard already holds (one under
`data/saved_datasets/`) imports nothing new; the run just scores it.

**Deleting the dataset once AutoFind has run.** A scheduled run that imports a
new batch every night fills the dashboard with datasets nobody opens. Turn on
**Delete the dataset after AutoFind** (Settings › **AutoFind**, under
**Command Line**), or put `"autofind_cli_delete_dataset": true` in the
`--settings` file for a run without `--user`, and a saving run deletes the
dataset it imported once its detectors have run and the results are exported,
pickle included, as the dashboard's **Delete** would. It is off by default, and
it is per user, so a `--user` run follows that user's choice. It deletes only
after real detection — the AutoFind list's detectors, or the ones
`--import-labels-into` or a pipeline file's `detectors:` names in their place —
so the dataset is kept:

- when there was nothing to detect with (the `Detection skipped` cases above:
  the import was the whole point);
- when the run fails, so the dataset can be searched again from the dashboard;
- when the run did not import it (a `--dataset` pickle the dashboard already
  held).

AutoFind started from the dashboard (a dataset's **Run AutoFind**, the one an
import starts, the **Find** button) never deletes anything. Unlike
`--tempimport`, the dataset still goes through the dashboard's import pipeline,
so the hits are the ones a dashboard import of the same source would give.

**`--tempimport`** makes the run temporary instead: the source is scored
straight from the importer and nothing is kept. Having no applicable detector is then an
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

### Sources

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

Use `python app.py --list-importers` to see all available importers. The built-in set is: `server_folder`, `server_files`, `local_folder`, `local_files`, `local_archive_member`, `pickle`, `http_archive`, `combine_datasets`, `demo`, `synthetic`. Each importer adds its own flags; run `python app.py --autodetect --importer <name> --help` to see them. `--help` resolves the named plugin first, so its flags are listed at the end of the usual help output (the same works for `--exporter <name> --help`).

**Several datasets from one importer run (`--outputs`).** The dashboard's
**Multi-Dataset** import ([`docs/api/datasets.md`](api/datasets.md#loading-datasets))
has a command-line form: `--outputs` takes a JSON list with one object per
dataset the single importer run should produce, in the shape the web API's
`outputs` entries take. Each object names its dataset's `media_type` and may
add `source_specs`, `category`, `dataset_name`, `embedder`, `embedders`,
`clipper`, `clipper_params`, `clipper_chain` and `cleaners`; it replaces the importer's `--media-type`
(setting both is refused). The importer acquires its source once (an archive
is downloaded and unpacked once) and every entry becomes its own dataset:

```bash
python app.py --autodetect --importer server_folder --path /data/field_kit \
  --outputs '[{"media_type": "audio"}, {"media_type": "image"}]' \
  --settings settings.json --exporter server_json_file --filepath hits.json
```

Each dataset is then handled on its own, in the list's order, exactly as a
single-dataset run would handle it: saved to the dashboard (unless
`--tempimport`) under `"<importer's name> – <Category>"` (`field_kit – Audio`,
`field_kit – Image`; an entry's own `dataset_name` wins), scored by the
AutoFind detectors that reach its media type (directly or through a converter
route), and **exported separately** — the exporter runs once per dataset, and
each results set carries a `dataset` block (`name`, `media_type`, `category`,
and the registry `id` when saved) saying which dataset it holds. So that the
files do not overwrite each other, `{dataset_name}` in any exporter field is
replaced by the dataset's name, and a `--filepath` that does not mention it
gets the name inserted before the extension (`hits-field_kit – Audio.json`,
`hits-field_kit – Image.json`); a list of one dataset leaves the path as given.
Each dataset's pass opens with a `=== Dataset 1 of 2: … ===` line (a
`dataset_start` event under `--progress-format json`), whichever exporter
runs. `--chunk-size` and `--stream-results` apply per dataset (a
streamed NDJSON file's `_meta` line carries the `dataset` block too).

A dataset no detector applies to is skipped with a `Detection skipped for
dataset '…'` note and the others still run; only a `--tempimport` run in which
*no* dataset could be scored fails for it (a saving run's datasets are the
point, as for a single one). A dataset whose import failed — the importer
produced nothing for it, say — is reported as `Import of dataset '…' failed:
…`, the others are still scored and exported, and the run then exits non-zero
naming it, so one bad output costs nothing but its own results. Only importers
that make several datasets per run accept the flag (the ones the dashboard
offers **Multi-Dataset** for; `demo` does not). `--create-detector` without
`--detector-media-type` takes the media type the outputs share, and needs the
flag when they differ.

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
the hit list *and* moves the detector's threshold, because the line's
prevalence is estimated on the population actually being scored.

### Large datasets: chunking and streaming

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

**Streaming as a labelset** (`--as-labelset`, with `--stream-results`): stream the
hits as label entries instead, so the export reads back in through a label
importer (`--import-labels-into`, or a label import in the app):

```bash
python app.py --autodetect --tempimport --importer server_folder --path /data/images \
  --media-type image --settings settings.json --chunk-size 500 \
  --stream-results --as-labelset --exporter server_csv_file --filepath labels.csv
```

Each entry is what a labelset export records for the item (`md5`, `label`,
`origin`, `origin_name`, `filename`, `category`, and the importer's metadata),
plus a `detector` field naming the detector that labelled it. Every detector the
run scores streams into the one export, so an item two detectors score appears
twice, once per detector. No score, threshold or balance leaves, since those
were measured by this run's head (#5131). `--keep-negatives` adds the
below-threshold items as `bad` labels. The same five exporters support it:

- `server_json_file` writes a labels JSON file (`{"labels": [...]}`, with
  `detector_meta`, `detectors`, `keep_negatives` and `format:
  "vtsearch-labelset-stream/v1"` beside it), not NDJSON. The `server_json_file`
  label importer reads it as it stands.
- `server_csv_file` writes `detector,label,md5,origin_name,filename,category,origin`
  rows, which the `server_csv_file` label importer reads (it ignores `detector`).
- `gui` prints each entry with its label and detector; `webhook` POSTs
  `labels` batches (`format: "vtsearch-labelset-batch/v1"`); `email_smtp` sends
  one email per batch, listed per detector.

Reading the file back in is not streamed: a label importer returns the whole
list, and training on it needs every label's embedding. Use it to get a huge
set of predictions out of the CLI, not to train on millions of labels.

### Exporting results

By default results are printed to the console. Add `--exporter <name>` to send them elsewhere:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter server_json_file --filepath results.json
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter server_csv_file --filepath results.csv
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter webhook --url https://example.com/hook
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter email_smtp \
    --from vtsearch@your-domain.example --to recipient@example.com
```

Available exporters: `server_json_file` (JSON to server path), `server_csv_file` (CSV to server path), `webhook` (HTTP POST, requires `--url`, optional `--auth-header`), `email_smtp` (SMTP email, requires `--from` and `--to`), `portable_detector` (standalone ONNX scoring bundles; see below), `gui` (default: print to console), `open_url` (format the hits' ids into one scheme-validated URL for another tool's viewer; see [Progress output format](#progress-output-format)). Run `python app.py --list-exporters` for the current set.

The default `gui` exporter prints:

```
Predicted Good (5 items):

  1-34094-A-6.wav
  1-30226-A-0.wav
  1-17150-B-2.wav
  1-22694-A-4.wav
  1-77445-A-1.wav
```

Items with origin information include the origin display string before the filename.

**Exporting the detectors themselves** (`portable_detector`): instead of the
scored hits, write one standalone, portable scoring bundle per detector the run
trained — the ONNX head (sigmoid baked in) plus a `manifest.json` and `README.md`,
carrying **no embeddings and no raw media**. It lets CI/automation produce a
shareable scoring model; the request-scoped equivalent is
`POST /api/detectors/{detector_id}/portable-bundle`
(see [`docs/api/detectors.md`](api/detectors.md#export-portable-bundle)); the
dashboard deliberately has no menu item for it. The `--dataset`/`--importer`
still supplies the embedder space the detector trains in; the media is embedded
but the hits are discarded.

```bash
# One bundle per AutoFind detector, named after the detector.
python app.py --autodetect --dataset data.pkl --settings settings.json \
  --exporter portable_detector --filepath 'data/{detector_name}-detector.zip'
```

The `--filepath` accepts `{detector_name}` (and the date variables
`{YYYYMMDD-HHMMSS}`, `{YYYYMMDD}`, `{YYYY}`, `{MM}`, `{DD}`, `{username}`). When
the path omits `{detector_name}` and the run trained more than one detector, the
detector name is inserted before the extension so bundles don't overwrite each
other. Structural (SIFT/VLAD) detectors, whose geometric verification is not an
ONNX forward pass, are skipped with a note rather than failing the run. Patch
(DINOv2/v3, EUPE) detectors are exported in a degraded mode that scores the
whole item as one vector, and their bundle's manifest and README say so.

### How the run scores

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
calibrated on, so a CLI run and a GUI Find agree on which media clear it.

**Matching the detector's clipper granularity.** A detector trained on a
specific clipper (its `input_spec.clipper` — e.g. 2-second audio tiles, or an
image grid) is **re-clipped at scoring time** when the loaded dataset wasn't
already clipped to match: each routed, target-typed media is split with the
detector's clipper and the clips are re-embedded, so a raw dataset is scored at
the granularity the detector expects. The per-clip scores fold back to the source
media by the same **max** rule as converter fan-out. A dataset already loaded
with the matching clipper is scored as-is (no redundant re-clip), and a detector
with no `input_spec.clipper` scores whole media.

**The line's prevalence is estimated on whatever the run ends up scoring.** The
line itself comes from the labels: the class model the calibration folds'
held-out scores of the votes imply. Only how common matches are is read off the
scored corpus, and converting and re-clipping change that population, not just
the item count: the max over a media's clips is never below the media's own
whole-item score, so an estimate taken on the loaded medias and applied to the
routed ones would sit in the wrong distribution. So calibration reads the
converted, re-clipped, re-embedded snapshot each detector's own scoring pass
will read. On a natively-typed dataset needing no re-clip the two are the same
set; on a converter-routed or re-clipped one the threshold follows the
population actually scored. The first chunk is prepared once and handed to
both passes, so this costs no extra conversion or embedding work.

**The exported set is unchecked, and the run says so.** Each detector's line
is its labels' line at its balance (F-beta's beta: the one the detector
keeps, chosen when it was created in the app; a detector that keeps
none takes the `beta` setting, 1 unless you change it; 4 leans toward
recall, 1/4 toward precision): the class
model the labels imply, cut where the expected F-beta peaks at the prevalence
estimated on the scored corpus. It keeps every item above the cut,
possibly none - there is no fixed count and no cap. In the app a spot check
measures how much of that set is right and how much it found; nobody can vote
in a headless run, so the run exports the unchecked set as it is. The run prints a line naming the detector and the size of the
set - `Detector 'det' exports its top 32 unchecked (at F1); nobody is here to
check it.` - which is a `detector_unchecked` event under
`--progress-format json` carrying `beta`, `status` and `count`, and every
result the detector produces carries a `balance` object beside its
`threshold` (`status` `unchecked`, with `count`), which the JSON exporters
write out with the hits. See [the line state](api/labeling.md#the-line-state).

**A detector with too few labels runs as its typed query, or as the Goods'
centroid, and the run says so.** Under the label quota (3 Goods and 4 Bads,
or a Good and 16 Bads) there is no trained head, here as in the app's Tune and
AutoFind (see [find-label](api/find.md#find-label-score--label-the-active-dataset)).
A detector made from a typed query (its `text_query`, else its first `text`
example) scores as that query's own set: each item's cosine to the query in
the dataset's text embedder, cut where the text sort draws its line at the
detector's balance, past the label quota too on a single-vector space
until it has 30 labels. It needs no label, so such a detector runs before
its first one. A detector with no typed query scores as the Goods' centroid
from its first Good; with no Good it stops the run (`Detector '<name>' has no
labels.` when it has no label at all). Either way the run prints
a line naming what the detector scored as and what it still owes -
`Detector 'Dogs' has 1 Good and 0 Bad labels, under the quota, so it scores as
its typed query's own set ('a dog'), not a trained detector (2 more Goods and
4 more Bads, or 16 more Bads for one).` - which is a `detector_query` or a
`detector_centroid` event under `--progress-format json`, carrying `detector`,
`tier` and the quota's counts (see the [event
reference](../vtscore/docs/packages/cli.md#event-reference)). A query's head
scores in the text embedder, so a `portable_detector` bundle of one names that
embedder.

### Settings and detector files

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
DRY RUN - no media will be loaded, embedded, scored, or exported.

Source:
  Importer: server_folder
  Params:
    media_type: audio
    path: /data/sounds
    recursive: True
    dig_archives: False
    reference_files: False
  Chunk size: whole dataset
  Save to dashboard: yes (the imported dataset is kept; --tempimport discards it)

Settings: settings.json
AutoFind detectors (2):
  - Dog Barks  [media_type=audio, labels=12, file=data/detectors/dog_barks.json]
  - Cat Meows  [media_type=audio, labels=8, file=data/detectors/cat_meows.json]

Exporter: server_json_file
  filepath: out.json
```

With `--outputs`, the source block lists the datasets the run would make, one
line per entry (its category, media type, source rows and any per-dataset
options), under `Datasets (2; one importer run, one dataset per entry):`; the
JSON `dry_run_plan` event's `source` carries them as `outputs`.

When `--stream-results` is set, the plan adds a `Streaming: yes (...)` line
under the source (noting whether negatives are dropped or included; `yes, as a
labelset (...)` with `--as-labelset`), so a streaming run can be sanity-checked
before it starts. The dry run also fails, as the run would, when the exporter
cannot stream that way. The `Save to dashboard`
line says whether the run would keep the dataset; a dry run never saves it.
With the `autofind_cli_delete_dataset` setting on, it reads `Save to dashboard:
until detection has run (…)`, and the JSON `dry_run_plan` event's `source`
carries `"delete_after_detection": true`.

`--dry-run` validates importer and exporter names, checks that the
dataset pickle (if given) exists, verifies required CLI fields are
populated, and reports any detector JSON files that are missing; so
typos in a cron-style invocation fail immediately instead of after a
multi-minute embedding pass. `--import-labels-into ... --label-importer-file ...`
is announced as part of the plan but skipped (no detector JSON is
modified), and the plan lists that detector alone, under
`Detectors (1; overrides the settings' AutoFind list)`. In
`--progress-format json` the `dry_run_plan` event's `autofind_detectors` lists
whichever detectors the run would score, and `detectors_source` says where they
came from: `autofind` (the settings file) or `override` (`--import-labels-into`).

### Importing labels into a detector

`--import-labels-into` merges an external label file into a detector's labelset
*before* the run trains and scores, so a batch of labels produced elsewhere
(another VTSearch instance, an annotation tool, a script) can be folded in
headlessly. These flags work together:

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
file's `autofind_detectors` list (the Dashboard's **AutoFind** tab) is not
consulted, so the detector need not be on AutoFind, and nothing else on AutoFind
runs alongside it. Labelling a detector and running it over a new dataset is
therefore a single command that never needs the UI. The settings file still
supplies everything else it normally does — `detectors_dir`, and the AutoFind
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

The labels have to give the run a detector: at least one `good` that resolves
(under the label quota it scores as the Goods' centroid; see [How the run
scores](#how-the-run-scores)). Labels keyed only by MD5 (the `server_json_file` / `server_csv_file`
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

The two choices are `text` (default, prose) and `json` (NDJSON events). Every
event and its fields are listed in the [event
reference](../vtscore/docs/packages/cli.md#event-reference). A [pipeline
file](#pipeline-file) run emits the same events.

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

**Exit status.** A run that completes exits 0, including a saving run that
skipped detection. A run that fails exits 1 after its `error` event (in text
mode, an `Error: …` line on stderr). A flag combination the CLI refuses
before anything runs — `--stream-results` without `--tempimport`, `--pipeline`
beside `--dataset`, `--user` without `--api-key`, … — is an argparse usage
error: exit 2, with the message on stderr rather than as an event.

A saving run (no `--tempimport`) reports the dataset it kept as a
`dataset_saved` event carrying its registry `dataset_id`, `name`, `num_items`
and `pkl_path` (`already_saved` is true when the pickle was already on the
dashboard), and a run that had no detector to use as `detection_skipped` with a
`reason`. A run that then deleted the dataset (the
[`autofind_cli_delete_dataset`](#saving-the-dataset-to-the-dashboard---tempimport)
setting) reports a `dataset_deleted` event with the same `dataset_id` and
`name`, after `export_complete`. While the import runs, its progress arrives as
ordinary `progress` events. A multi-dataset run (`--outputs`) reports each
dataset in turn: a `dataset_start` event (`name`, `media_type`, `category`,
`dataset_id`, `index`, `count`) opens each dataset's detection, `dataset_saved`
/ `detection_skipped` / `export_complete` then arrive once per dataset
(`detection_skipped` naming it in `dataset`), a dataset whose import failed is a
`dataset_failed` event (`name`, `error`), and the import's `progress` events
carry the dataset's name as `dataset`:

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
  | jq -r 'select(.event == "medias_skipped") | .skipped_ids[]'
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

The YAML covers the source, settings file, detectors, chunking, streaming,
label import and exporter. The flags for those cannot be combined with
`--pipeline`: the run is refused with an error naming the flag, so declare them
in the file. `--dry-run` has no pipeline form and is refused too. Like the
flags, a pipeline run saves its dataset to the dashboard unless the file sets
`tempimport: true`.

The flags that configure the process rather than describe the run apply to a
pipeline run as they do to `--autodetect`: `--user` / `--api-key` (see [Which
user's AutoFind list runs](#which-users-autofind-list-runs)),
`--progress-format`, `-v`, and the admin overrides under [Web server
modes](#web-server-modes) (`--solo-media-type`, `--hide-plugin`, …). Most of
those overrides change nothing in a headless run; [Headless runs vs. the web
server](#headless-runs-vs-the-web-server) lists which ones do:

```bash
python app.py --pipeline nightly.yaml --user alice --api-key "$ALICE_KEY" --progress-format json
```

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
  # Optional. Several datasets from this one importer run (same as --outputs):
  # one mapping per dataset, in the shape the web API's `outputs` entries take
  # (media_type, source_specs, category, dataset_name, embedder, embedders,
  # clipper, clipper_params, clipper_chain, cleaners). Replaces
  # `fields.media_type` and `fields.source_specs` (setting either is refused);
  # each dataset is saved, scored and exported on its own.
  # outputs:
  #   - media_type: audio
  #   - media_type: image
  #     category: document
  #     source_specs: [{source_type: document, converter: document2image}]
  #     dataset_name: scans

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

# Optional. With stream_results, stream the hits as re-importable label
# entries, each naming its detector, through the exporter's labelset
# streaming mode (same as --as-labelset). Off by default.
as_labelset: false

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
time, so a typo fails fast before any media is loaded. So is
`importer.outputs`: every entry needs a `media_type`, an unknown key is
refused, an unknown media type or converter is refused with the web API's own
message, and the list is rejected for an importer that makes one dataset per
run. The run then behaves as [`--outputs`](#sources) does: one dataset per
entry, each exported to its own file (`{dataset_name}` in `exporter.fields`
names it; a `filepath` without it gains the name before the extension).

The shorter `import_labels` form is still accepted: `importer: <name>` given as
a plain string, with `file: <path>` as shorthand for the importer's `filepath`
field.

## Headless runs vs. the web server

`python app.py` takes one of two paths. With `--autodetect` (or `--tempimport`)
or `--pipeline` it runs **headless**: it imports, scores and exports, then
exits. Otherwise it starts the **web server**. Gunicorn's
`VTSEARCH_SERVER_INIT=1` import counts as the server path. Everything in
`app.py` that runs before `argv` is parsed is shared by both paths, but the
two startups differ after that. This table lists every difference and why it
is there; each one is intended.

| Setting or step | Web server | Headless run | Why |
|---|---|---|---|
| Startup banners (mode, admin overrides in force, import hooks, Help-modal doc links) | Printed to stdout | Not printed | A headless run's stdout carries its results, and is pure NDJSON under `--progress-format json`. |
| Embedder preload (each registered dataset's and detector's embedder, plus the `--solo-media-type` / `--solo-embedder` ones) | Warmed at startup | None. The run loads the embedder its dataset needs the first time it embeds | Warming embedders the run never uses costs minutes and memory. |
| GC freeze after the preload (`VTSEARCH_GC_FREEZE`) | Yes | No | It keeps garbage-collection pauses out of interactive requests, and a run serves no requests. |
| Stall diagnostics (GC-pause log, heartbeat watchdog, `VTSEARCH_STALL_*`, the diagnostics-bars log line) | Started | Not started | The bars measure request and phase latency on an interactive server (#3853); a run has neither. |
| Web-UI admin overrides: `--solo-media-type`, `--solo-embedder`, `--hide-plugin`, `--support-email`, `--semantic-only`, `--autopilot-only` (flag, env var or settings key) | Shape the UI, the listing routes and, for `--semantic-only`, the create and import routes | Validated, so a bad value still exits 2, then inert: nothing headless reads them. `--hide-plugin` hides nothing from `--list-plugins`, and a hidden plugin can still be named with `--importer` / `--exporter`; `--semantic-only` does not refuse a patch or structural embedder; `--solo-embedder` does not choose the run's embedder | They control what a web user is offered. The person running the CLI names plugins and embedders explicitly, and hidden plugins stay callable by name on the server too. |
| `autofind_export_enabled` (a user's **Export results when AutoFind finishes**) | Off pauses the AutoFind exporter for that user's web runs | Not read: without `--exporter`, a run still falls back to `autofind_exporter` | On the command line the exporter is where the run's results go, not an alert sent alongside them. |
| `--on-dataset-imported` / `VTSEARCH_ON_DATASET_IMPORTED` | Called after every import a user starts from the web app | The flag's module is imported (a typo exits 2), but the hook never fires. The variable is not read | The hooks are for imports a user waits on in the web app ([EXTENDING.md § Dataset-Import Hooks](EXTENDING.md#dataset-import-hooks)). |
| Login: `--login` on the server; `--user` + `--api-key` headless | `--login` activates the provider; `--user` and `--api-key` are ignored | `--user` + `--api-key` authenticate against the same `data/api_keys.json`; `--login` is ignored | The server authenticates each request; a run authenticates once, as the user it runs as. |
| `--port`, `VTSEARCH_PORT`, the single-instance lock, the port preflight, the dev server's SSE cap | Applied | None | A run binds no port and takes no lock, so it can run beside a server on the same data directory. |
| `--local` | Changes the mode banner | Ignored | Banner text only. |
| `--progress-format`, plugin notifications | Progress goes to the SSE stream; notifications become toasts | `text` or `json` on stdout; notifications are printed | No browser is watching a headless run. |

These are the same on both paths:

- **Native thread count, OpenBLAS kernels, logging** (`VTSEARCH_TORCH_THREADS`,
  `OPENBLAS_CORETYPE`, `VTSEARCH_LOG_LEVEL`, `VTSEARCH_LOG_FORMAT`,
  `VTSEARCH_DIAGNOSE`, `-v`). `app.py` sets these at import, before it parses
  `argv`, so a headless run gets the server's thread count too: the CPU
  allocation, unless the variable says otherwise. Only code that imports
  `vtscore` without `app.py` defaults to 1 thread.
- **The `VTSEARCH_*` admin-override variables.** Both paths apply them, and an
  explicit flag still wins. The only one that changes a headless run is
  `VTSEARCH_DATASET_MAX_AGE_DAYS`, which sets the expiry stamp on the dataset a
  saving run saves. A malformed value is warned about on stderr, never on the
  NDJSON stdout.
- **Dataset retention.** `--dataset-max-age-days`, its variable and the
  persisted `dataset_max_age_days` key all stamp a dataset that either path
  saves.
- **`--settings FILE`.** Both paths read and write the server-tier settings from
  `FILE`. A headless run switches to it before it imports, so the save uses
  that file's saved-datasets directory, retention and concurrency gates. A
  pipeline file names it as `settings:`.
- **Runtime preparation** (`initialize_models`). Both warm sklearn's threadpool
  controller before any worker thread starts, which avoids a loader-lock
  deadlock, and seed the fast `packages_distributions` before transformers is
  imported (#3715). A headless `--dry-run` loads nothing and skips this step.
- **`VTSEARCH_SERVER_INIT=1`.** It applies only when gunicorn imports `app.py`.
  Under `python app.py` it is ignored, so a headless run in a Docker image
  (which sets the variable for every process) gets none of the server's
  startup.
- **The default user, per-user settings and settings sources.** Without a login
  or `--user`, both paths run as `default`. Each user's `settings_source` syncs
  lazily the first time that user's settings are read.

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
`VTSEARCH_LOG_LEVEL=debug` stays at DEBUG. It applies to the web server,
`--autodetect` and `--pipeline`:

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
bundled Docker images already run gunicorn this way. The variable only
applies on import: under `python app.py` it is ignored, so a headless run in
a container (where the image sets it for every process) stays headless. See
[DEPLOYMENT.md](DEPLOYMENT.md#tuning) for tuning.

**Server settings file** (`--settings FILE`): read and write the server-tier
settings from `FILE` instead of `data/settings.json`. This is the same flag
that points an `--autodetect` run at its settings, so it means the same thing
on both paths:

```bash
python app.py --settings /srv/vtsearch/settings.json
```

Startup prints the file in force. Per-user settings files are unaffected.
Gunicorn never parses `argv`, so to move the file there, relocate the whole
data directory with `VTSEARCH_DATA_DIR`.

**Authentication mode** (`--login`): select the login provider (dev
server only; set up the provider in code when running under gunicorn).
Two providers are accepted:

```bash
python app.py --login trivial    # multi-user mode with simple username auth (cookie-based, no password)
python app.py --login api_key    # Bearer-key auth against data/api_keys.json
```

- **`trivial`** shows a username prompt (no password) and tracks the user via a
  cookie; useful for low-stakes multi-user setups. Without `VTSEARCH_SECRET_KEY`
  the cookie is signed with a random per-process key, so users sign in again
  after a restart (see [DEPLOYMENT.md § Security](DEPLOYMENT.md#security)).
- **`api_key`** authenticates each request via an `Authorization: Bearer <key>`
  header, checking the key against `data/api_keys.json`. This is the same key
  store the CLI's `--user` + `--api-key` flags use (see [Which user's AutoFind
  list runs](#which-users-autofind-list-runs) above), so a key minted for the
  server also works for a per-user `--autodetect` run.

Without `--login`, the app uses `DefaultLoginProvider` (single-user, always authenticated).

**Solo mediaType** (`--solo-media-type`): streamline the UI for users
who only ever look at one media type (e.g. images, optionally pulled
in via converters from videos/documents). When set, the dataset
importer and new-detector flows hide their mediaType pickers and lock
to this type, the converter list filters to converters whose output is
this type, the read-only fields that could only name this type (Find
Results, the metadata panes, the Dashboard's Type columns, …) drop out,
and the type's default embedder is warmed at startup:

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
`hidden_plugins` key in the server settings file (`data/settings.json`),
where a deployment can set
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
init; an explicit `--dataset-max-age-days` flag wins over it. Every
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

**Autopilot only** (`--autopilot-only`): keep Train and Tune on their
Autopilot tabs, for a deployment whose users should only see the guided
flow:

```bash
python app.py --autopilot-only
```

Train loses its Manual tab and Tune its Review tab, and neither view
renders a tab bar. Like `--semantic-only`, this is a **server-wide
override** that can only turn the lock on: there is no
`--no-autopilot-only`, the persisted `autopilot_only` setting can turn it
on too, and neither is editable via the Settings dialog or the settings
API (the Settings ▸ Server tab reports it). The env-var equivalent is
`VTSEARCH_AUTOPILOT_ONLY`; an explicit flag wins over it. See
[`autopilot_only`](SETTINGS.md#server-tier--datasettingsjson) for what
it means for Tune's Review actions, and for a detector Autopilot can't
start.

**Run a function when an import finishes** (`--on-dataset-imported
MODULE:FUNCTION`): call your own function each time a user's dataset
import from the web app succeeds or fails, for example to email them:

```bash
python app.py --on-dataset-imported mailer:notify
```

`MODULE` must be importable on the server (installed, or on
`PYTHONPATH`); it is imported at startup, so a typo stops the server
before it serves anything. The flag is repeatable, and the functions run
in the order given. Each receives one `DatasetImported` event; what it
carries, and which imports fire it, is in
[EXTENDING.md § Dataset-Import Hooks](EXTENDING.md#dataset-import-hooks).
Unlike the overrides above, this has no settings-file counterpart and
never will: a settings file that could name code to run would let anyone
who can import settings run code on the server. The env-var equivalent
is `VTSEARCH_ON_DATASET_IMPORTED` (comma-separated); an explicit flag
wins over it.

## Inspecting plugins and the API schema

`python app.py --list-plugins` enumerates every auto-discovered plugin
(dataset, datasource and seed importers, exporters, label importers/sources,
settings I/O, media converters/types/embedders/clippers/cleaners, and media
sources) and exits without starting the server. Three output formats:

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
