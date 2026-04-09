# Command-line interface

VTSearch provides a CLI workflow for running detectors on datasets and exporting results — all without starting the web server.

## Auto-detect (run detectors on a dataset)

Score every item in a dataset with your autorun processors (detectors) and output the items predicted as "Good."

Detectors are specified via a **settings file** (`--settings`) that lists autorun processors. Each processor is a recipe referencing a processor importer (e.g. `server_detector_file` for a pre-trained detector JSON on disk). See below for how to create one.

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
- **Settings file** — A JSON file listing autorun processors. Each processor references a processor importer and its field values. Example:

```json
{
  "autorun_processors": [
    {
      "processor_name": "my detector",
      "processor_importer": "server_detector_file",
      "field_values": { "filepath": "/path/to/detector.json" }
    }
  ]
}
```

- **Detector file** — In the web UI, vote on some items, then export a detector to the server via the sorting panel. Detector files store origin information (not weights); weights are re-derived at load time.

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

**Default (production) mode** — binds to `127.0.0.1:5000` (localhost only):

```bash
python app.py
```

**Local mode** (`--local`) — binds to `0.0.0.0:5000` (accessible from other
devices on the network):

```bash
python app.py --local
```

Both modes run the same startup sequence (model initialization, autoload
preloading, settings source sync). The only difference is the network bind
address.

**Authentication mode** (`--login`) — select the login provider:

```bash
python app.py --login trivial    # multi-user mode with simple username auth
```

Without `--login`, the app uses `DefaultLoginProvider` (single-user, always authenticated).
