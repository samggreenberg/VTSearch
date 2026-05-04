# VTSearch

A media explorer web app. Browse collections of audio clips, images, text paragraphs, videos, or documents — listen/view them in the browser and vote items as "good" or "bad." Supports text-based semantic sorting (via LAION-CLAP, SigLIP, X-CLIP, or E5-base-v2 embeddings depending on media type) and learned sorting (via a small neural network trained on your votes). Several demo datasets can be loaded directly from the UI. Built with Flask (Python), Angular (TypeScript), and PyTorch.

## Setup and running tests

See [docs/SETUP.md](docs/SETUP.md) for prerequisites, getting the code, virtual environment setup, installing dependencies, and running the test suite.

## Running the app

For development, start the Flask dev server:

```bash
python app.py
```

You should see output like:

```
 * Running on http://0.0.0.0:5000
```

Open `http://localhost:5000` in your browser. The app starts with no clips loaded — use the menu to load a demo dataset (see below).

Press **Ctrl+C** in the terminal to stop the server.

For production, run under gunicorn (the Docker images do this automatically):

```bash
VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app
```

See [docs/SETUP.md](docs/SETUP.md#running-the-app) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for details.

> **New to VTSearch?** Read **[docs/USER_GUIDE.md](docs/USER_GUIDE.md)** — a walkthrough of loading a dataset, using Autopilot to label, and exporting results. Most users never need anything else.

## Command-line interface

VTSearch provides several CLI workflows for running detectors, importing labels, and importing processors — all without starting the web server. See [docs/CLI.md](docs/CLI.md) for the full CLI reference.

## Loading a demo dataset

When the app is running, click the hamburger menu in the top-left corner to open the dataset panel. From there you can browse the available demo datasets and load one. Each demo is downloaded and embedded on first use, then cached for instant loading afterward.

See [docs/demos.md](docs/demos.md) for the full list of available demo datasets.

You can also load your own data from pickle files or folders via the same menu.

## Project structure

```
├── app.py                          # Flask entry point, registers blueprints, CLI arg parsing
├── gunicorn.conf.py                # Gunicorn WSGI config (single worker + threads)
├── vtsearch/                       # Main application package
│   ├── config.py                   # Constants (CLAP_SAMPLE_RATE, paths, model IDs)
│   ├── medias.py                   # Test media generation & embedding cache
│   ├── cli.py                      # CLI utilities: autodetect workflow
│   ├── settings.py                 # Persistent settings & autorun processors
│   ├── auth/                       # Authentication (LoginProvider ABC, DefaultLoginProvider)
│   ├── routes/                     # Flask blueprints
│   ├── models/                     # ML models (embeddings, training, progress)
│   ├── media/                      # Media type plugins (audio, image, text, video, document)
│   ├── converters/                 # Media converters (document→image, video→audio, etc.)
│   ├── datasets/                   # Dataset loading, downloading, importers
│   ├── eval/                       # Evaluation framework (metrics, runner, visualisation)
│   ├── exporters/                  # Results exporter plugins
│   ├── labels/                     # Label importers & labelset sync sources
│   ├── processors/importers/       # Processor importer plugins
│   ├── settings_io/                # Settings importers, exporters & sync sources
│   ├── audio/                      # Audio generation utility
│   └── utils/                      # State proxies (per-dataset/per-detector contexts) & progress helpers
├── static/                         # Angular build output (HTML, JS, CSS, assets)
├── tests/                          # Test suite (pytest)
├── docs/                           # Extended documentation
│   ├── HANDOFF.md                  # Project handoff & orientation guide
│   ├── DEPLOYMENT.md               # Deployment, offline mode, operations
│   ├── ARCHITECTURE.md             # Architecture deep-dive
│   ├── API.md                      # HTTP API reference (all REST endpoints)
│   ├── EXTENDING.md                # Plugin authoring index (auth, deps, checklists)
│   ├── EXTENDING-plugins.md        # Data/results/label/processor/settings importers & sources
│   ├── EXTENDING-media.md          # Media types, embedders, clippers, converters, sources
│   ├── EXTENDING-processors.md     # Detectors, localizers, extractors
│   ├── EVAL.md                     # Evaluation framework guide
│   ├── CLI.md                      # CLI reference
│   ├── ML.md                       # ML model details
│   ├── SETUP.md                    # Setup instructions
│   ├── USER_GUIDE.md               # End-user walkthrough (Autopilot, voting, sorting)
│   ├── demos.md                    # Demo dataset listing
│   ├── plan-sync-sources.md        # Sync sources design document (implemented)
│   └── design/                     # Architecture design documents
└── pyproject.toml                  # Project metadata and dependencies
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

VTSearch has a plugin architecture for media types, data importers, results exporters, label importers, processor importers, and sync sources. The extending guide is split into three topic-specific docs plus an index:

- **[docs/EXTENDING.md](docs/EXTENDING.md)** — index, authentication providers, dependency management, and a one-stop checklist for every extension type.
- **[docs/EXTENDING-plugins.md](docs/EXTENDING-plugins.md)** — data importers, results exporters, label importers, processor importers, settings importers/exporters, settings sources, labelset sources (eight auto-discovered plugin families).
- **[docs/EXTENDING-media.md](docs/EXTENDING-media.md)** — media types, embedders, clippers, converters, media sources.
- **[docs/EXTENDING-processors.md](docs/EXTENDING-processors.md)** — detectors, localizers, extractors.
