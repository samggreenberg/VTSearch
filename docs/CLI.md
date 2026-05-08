# Command-line interface

VTSearch provides a CLI workflow for running detectors on datasets and exporting results — all without starting the web server.

## Auto-detect (run trainable models on a dataset)

Score every item in a dataset with the trainable models flagged for
autorun and output the items each model predicts as "Good."

Models are specified via a **settings file** (`--settings`) whose
`autorun_trainable_models` list names registered models.  Each name
maps to `data/trainable_models/<name>.json`; the CLI re-resolves the
labelset's origins, embeds them with the dataset's embedder, trains an
MLP, and applies it to the dataset.  See below for the exact format.

**From a pickle file:**

```bash
python app.py --autodetect --dataset path/to/dataset.pkl --settings settings.json
```

**From any supported data source** (folder, HTTP archive):

```bash
python app.py --autodetect --importer folder --path /data/sounds --media-type audio --settings settings.json
python app.py --autodetect --importer http_archive --url https://example.com/data.zip --settings settings.json
```

Available importers: `folder`, `pickle`, `http_archive`, `combine_datasets`, `demo`. Each importer adds its own flags — run `python app.py --autodetect --importer <name> --help` to see them.

**Chunked loading** — for large datasets, use `--chunk-size N` to process in batches to limit memory:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --chunk-size 1000
python app.py --autodetect --importer folder --path /data/sounds --media-type audio --settings settings.json --chunk-size 500
```

**Exporting results** — by default results are printed to the console. Add `--exporter <name>` to send them elsewhere:

```bash
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter server_json_file --filepath results.json
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter server_csv_file --filepath results.csv
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter webhook --url https://example.com/hook
python app.py --autodetect --dataset data.pkl --settings settings.json --exporter email_smtp --to recipient@example.com
```

Available exporters: `server_json_file` (JSON to server path), `server_csv_file` (CSV to server path), `webhook` (HTTP POST, optional `--auth-header`), `email_smtp` (SMTP email, requires `--to`), `gui` (default — print to console).

**How to get the files:**

- **Dataset file** — Export from the web UI via the dataset menu ("Export dataset"), or use a cached `.pkl` file from the `data/embeddings/` directory after loading a demo dataset.
- **Settings file** — A JSON file listing the trainable-model names that should run during `--autodetect`. Each name maps to a JSON labelset under `data/trainable_models/<name>.json`; the CLI re-resolves the labelset's origins, embeds them with the dataset's embedder, trains a fresh MLP, and scores the dataset.

```json
{
  "autorun_trainable_models": ["Dog Barks", "Cat Meows"],
  "trainable_models_dir": "data/trainable_models"
}
```

- **Trainable-model file** — Created from the dashboard by labeling items in the right pane. The file stores origin info plus labels (no weights); the MLP is rebuilt from origins at scoring time.

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

## Web server modes

**Development (Flask dev server)** — bind to `0.0.0.0:5000`:

```bash
python app.py
```

A `--local` flag is accepted for historical reasons and only changes the
banner text (`LOCAL` vs. `PRODUCTION`); the bind address is the same either
way. This entry point uses Flask's built-in dev server and is not
recommended for production.

**Production (gunicorn)** — run the WSGI app under the bundled config:

```bash
VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app
```

`VTSEARCH_SERVER_INIT=1` runs the same startup sequence (model
initialization, autoload preloading, settings-source sync) that `python
app.py` runs — gunicorn imports `app.py` rather than executing its
`__main__` block, so the env var is what triggers initialization. The
bundled Docker images already run gunicorn this way. See
[DEPLOYMENT.md](DEPLOYMENT.md#gunicorn-tuning) for tuning.

**Authentication mode** (`--login`) — select the login provider (dev
server only; set up the provider in code when running under gunicorn):

```bash
python app.py --login trivial    # multi-user mode with simple username auth
```

Without `--login`, the app uses `DefaultLoginProvider` (single-user, always authenticated).
