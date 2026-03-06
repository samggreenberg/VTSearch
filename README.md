# VTSearch

A media explorer web app. Browse collections of audio clips, images, text paragraphs, videos, or documents — listen/view them in the browser and vote items as "good" or "bad." Supports text-based semantic sorting (via LAION-CLAP, CLIP, X-CLIP, or E5-base-v2 embeddings depending on media type) and learned sorting (via a small neural network trained on your votes). Several demo datasets can be loaded directly from the UI. Built with Flask (Python), Angular (TypeScript), and PyTorch.

## Setup and running tests

See [docs/SETUP.md](docs/SETUP.md) for prerequisites, getting the code, virtual environment setup, installing dependencies, and running the test suite.

## Running the app

```bash
python app.py
```

You should see output like:

```
 * Running on http://127.0.0.1:5000
```

Open that URL in your browser. The app starts with no clips loaded — use the menu to load a demo dataset (see below).

Press **Ctrl+C** in the terminal to stop the server.

## Command-line interface

VTSearch provides several CLI workflows for running detectors, importing labels, and importing processors — all without starting the web server. See [docs/CLI.md](docs/CLI.md) for the full CLI reference.

## Loading a demo dataset

When the app is running, click the hamburger menu in the top-left corner to open the dataset panel. From there you can browse the available demo datasets and load one. Each demo is downloaded and embedded on first use, then cached for instant loading afterward.

See [docs/demos.md](docs/demos.md) for the full list of available demo datasets.

You can also load your own data from pickle files or folders via the same menu.

## Project structure

```
├── app.py                          # Flask entry point, registers blueprints, CLI arg parsing
├── vtsearch/                       # Main application package
│   ├── config.py                   # Constants (SAMPLE_RATE, paths, model IDs)
│   ├── medias.py                   # Test media generation & embedding cache
│   ├── cli.py                      # CLI utilities: autodetect workflow
│   ├── settings.py                 # Persistent settings & autorun processors
│   ├── routes/                     # Flask blueprints
│   ├── models/                     # ML models (embeddings, training, progress)
│   ├── media/                      # Media type plugins (audio, image, text, video, document)
│   ├── converters/                 # Media converters (document→image, video→audio, etc.)
│   ├── datasets/                   # Dataset loading, downloading, importers
│   ├── eval/                       # Evaluation framework (metrics, runner, visualisation)
│   ├── exporters/                  # Results exporter plugins
│   ├── labels/importers/           # Label importer plugins
│   ├── processors/importers/       # Processor importer plugins
│   ├── audio/                      # Audio generation utility
│   └── utils/                      # Global state (medias, votes) & progress helpers
├── static/                         # Angular build output (HTML, JS, CSS, assets)
├── tests/                          # Test suite (pytest)
├── docs/                           # Extended documentation
│   ├── HANDOFF.md                  # Project handoff & orientation guide
│   ├── DEPLOYMENT.md               # Deployment, offline mode, operations
│   ├── ARCHITECTURE.md             # Architecture deep-dive
│   ├── API.md                      # HTTP API reference (all REST endpoints)
│   ├── EXTENDING.md                # Plugin authoring guide
│   ├── EVAL.md                     # Evaluation framework guide
│   ├── CLI.md                      # CLI reference
│   ├── ML.md                       # ML model details
│   ├── SETUP.md                    # Setup instructions
│   ├── demos.md                    # Demo dataset listing
│   └── old_io.md                   # Retired IO module reference implementations
└── requirements*.txt               # Dependency files (cpu, gpu, dev, importers, exporters)
```

## HTTP API

VTSearch exposes a REST-style JSON API. See [docs/API.md](docs/API.md) for the full endpoint reference, including media listing, sorting, voting, dataset management, detector/exporter/importer operations, settings, and trainable models.

## Deployment

For production deployment, offline/air-gapped operation, Docker hardening, environment variables, network dependency details, and data directory management, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

New to the project? Start with [docs/HANDOFF.md](docs/HANDOFF.md) for a full orientation including a documentation map, key concepts, deployment checklist, and common workflows.

## Machine learning

VTSearch trains a small MLP neural network on user votes to learn a binary classifier over pretrained embeddings. See [docs/ML.md](docs/ML.md) for full details on the model architecture, training configuration, PyTorch settings, and embedding models.

## Evaluation

VTSearch includes an evaluation framework that measures sorting quality on demo datasets. Run it with:

```bash
python -m vtsearch.eval --plot-dir eval_output
```

This runs text-sort and learned-sort evaluations across all demo datasets, prints a summary, and saves visualisation charts as PNGs. See [docs/EVAL.md](docs/EVAL.md) for the full guide, including:

- **[CLI reference](docs/EVAL.md#cli-reference)** — All flags and options for the eval runner.
- **[Understanding the metrics](docs/EVAL.md#understanding-the-metrics)** — What mAP, P@k, R@k, F1, and other metrics mean.
- **[Visualisations](docs/EVAL.md#visualisations)** — Charts generated by the eval framework.
- **[Writing a custom evaluation script](docs/EVAL.md#writing-a-custom-evaluation-script)** — How to sweep over parameters, run voting-iteration simulations, and use the Python API directly.

## Extending with plugins

VTSearch has a plugin architecture for media types, data importers, results exporters, label importers, and processor importers. See [docs/EXTENDING.md](docs/EXTENDING.md) for full documentation, including:

- **[Adding a Data Importer](docs/EXTENDING.md#adding-a-data-importer)** — Auto-discovered plugins that load datasets from new sources (S3, databases, APIs, etc.). Subclass `DatasetImporter`, expose an `IMPORTER` instance, and the system wires up API routes and UI forms automatically.
- **[Adding a Results Exporter](docs/EXTENDING.md#adding-a-results-exporter)** — Auto-discovered plugins that export autodetect results to new destinations (files, webhooks, email, etc.). Subclass `LabelsetExporter` and expose an `EXPORTER` instance.
- **[Adding a Label Importer](docs/EXTENDING.md#adding-a-label-importer)** — Auto-discovered plugins that import pre-existing labels from external sources (JSON, CSV, databases). Subclass `LabelImporter` and expose a `LABEL_IMPORTER` instance.
- **[Adding a Processor Importer](docs/EXTENDING.md#adding-a-processor-importer)** — Auto-discovered plugins that import processors (detectors/extractors) from external sources. Subclass `ProcessorImporter` and expose a `PROCESSOR_IMPORTER` instance.
- **[Adding a Media Type](docs/EXTENDING.md#adding-a-media-type)** — Support new content types (code, 3D models, etc.) by subclassing `MediaType` with embedding, serving, and clip-loading methods.
- **[Dependency Management](docs/EXTENDING.md#dependency-management)** — How the layered requirements file structure works and where to add new dependencies.
