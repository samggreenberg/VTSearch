# VTSearch

A trainable media search tool. VTSearch searches collections of audio clips, images, text paragraphs, videos, and documents using a **detector** (a small trained ranker that scores every item in the collection by how well it matches what you're looking for). You search either by **training a new detector** (vote a handful of items "good" or "bad" and a small neural net learns from your votes to rank the rest of the collection) or by **using an existing detector** (one you saved earlier, exported from another VTSearch instance, or imported from disk). Trained detectors are reusable: apply the same one to any future dataset of the same media type. A natural-language query ("dog barking", "red car in snow") seeds either flow via pretrained embeddings (LAION-CLAP for audio, SigLIP for images, X-CLIP for video, E5-base-v2 for text), and also works as a quick stand-alone search when you don't need a trained detector. Several demo datasets are available directly from the UI. Built with Flask (Python), Angular (TypeScript), and PyTorch.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/user/assets/dashboard-loaded.dark.png" />
  <img src="docs/user/assets/dashboard-loaded.light.png" alt="The VTSearch dashboard with a synthetic dataset loaded and a trained detector listed in the sidebar" width="720" />
</picture>

> **New to VTSearch?** Read **[docs/user/USER_GUIDE.md](docs/user/USER_GUIDE.md)** for a walkthrough of loading a dataset, training a detector with Autopilot (or applying an existing one), and exporting the matches. Most users never need anything else.

## Quick start

```bash
bash scripts/install.sh                            # install Python deps (auto-detects CPU vs GPU)
cd frontend && npm install && npm run build:prod   # build the Angular frontend into static/
cd .. && python app.py --local                     # start the app at http://localhost:5000
```

See [docs/SETUP.md](docs/SETUP.md) for prerequisites, virtual environment setup, and the full walkthrough.

## Setup and running tests

See [docs/SETUP.md](docs/SETUP.md) for prerequisites, getting the code, virtual environment setup, installing dependencies, and running the test suite.

## Running the app

For development, start the Flask dev server:

```bash
python app.py
```

Use `--local` to run in local development mode:

```bash
python app.py --local
```

You should see output like:

```
 * Running on http://0.0.0.0:5000
```

Open `http://localhost:5000` in your browser. The app starts with no clips loaded. Use the menu to load a demo dataset (see below).

Press **Ctrl+C** in the terminal to stop the server.

For production, run under gunicorn (the Docker images do this automatically):

```bash
VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app
```

See [docs/SETUP.md](docs/SETUP.md#running-the-app) and [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for details.

## Loading a demo dataset

When the app is running, click the **+** button on the **Datasets** card to open the **Add Dataset** dialog, then pick the **Demo** tab. From there you can browse the available demo datasets and load one. Each demo is downloaded and embedded on first use, then cached for instant loading afterward.

See [docs/demos.md](docs/demos.md) for the full list of available demo datasets.

You can also load your own data from pickle files or folders via the same dialog.

---

## Command-line interface

VTSearch provides several CLI workflows for applying detectors to datasets, importing labels, and importing processors, all without starting the web server. See [docs/CLI.md](docs/CLI.md) for the full CLI reference.

## Project structure

VTSearch is split into two Python packages along an **app tier / library tier** line:

- **`vtsearch/`** — the app tier: Flask routes, authentication, settings, the achievements state machine, and the CLI entry point (`vtsearch/cli_main.py`). Anything that depends on Flask/Werkzeug lives here.
- **`vtscore/`** — the library tier: the ML (training, MLP, thresholds), embedding runtime, media-type plugins (audio, image, text, video, document), converters, datasets/importers, exporters, labels, evaluation, projection (VTSBrowse), concurrency, security, and the plugin/sync machinery. It is import-clean of Flask so it can be reused as a standalone library. The CLI orchestration (`vtscore/cli.py`, `cli_pipeline.py`, `cli_progress.py`) lives here too.

The remaining top level:

```
├── app.py            # Flask entry point: builds the app, registers blueprints, parses CLI args
├── gunicorn.conf.py  # Gunicorn WSGI config (single worker + threads)
├── vtsearch/         # App tier (Flask routes, auth, settings, CLI entry point)
├── vtscore/          # Library tier (ML, embedding, media, datasets, plugins, projection)
├── frontend/         # Angular SPA source (TypeScript, SCSS); builds into static/
├── static/           # Angular build output (HTML, JS, CSS, assets)
├── tests/            # App-tier test suite (pytest); grouped by folder (core, api, sorting, …)
├── tests_lib/        # Library-tier test suite (mirrors tests/, import-clean of Flask)
├── docs/             # Extended documentation (see docs/ARCHITECTURE.md for the full map)
└── pyproject.toml    # Project metadata and dependencies
```

For the complete directory map, dependency graph, and the app-tier/library-tier rules, see **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

## HTTP API

VTSearch exposes a REST-style JSON API. See [docs/API.md](docs/API.md) for the full endpoint reference, including media listing, sorting, voting, dataset management, detector CRUD and scoring, exporter and importer operations, and settings.

## Deployment

For production deployment, offline/air-gapped operation, Docker hardening, environment variables, network dependency details, and data directory management, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

New to the project? Start with [docs/HANDOFF.md](docs/HANDOFF.md) for a full orientation including a documentation map, key concepts, deployment checklist, and common workflows.

## Machine learning

VTSearch trains a small MLP neural network on user votes to learn a binary classifier over pretrained embeddings. See [docs/ML.md](docs/ML.md) for full details on the model architecture, training configuration, PyTorch settings, embedding models, and the Coverage Atlas that drives diversity sampling and domain-shift detection.

## Evaluation

VTSearch includes an evaluation framework that measures sorting quality on demo datasets. Run it with:

```bash
python -m vtscore.eval --plot-dir eval_output
```

This runs text-sort and learned-sort evaluations across all demo datasets, prints a summary, and saves visualisation charts as PNGs. See [docs/EVAL.md](docs/EVAL.md) for the full guide, including:

- **[CLI reference](docs/EVAL.md#cli-reference)**: All flags and options for the eval runner.
- **[Understanding the metrics](docs/EVAL.md#understanding-the-metrics)**: What mAP, P@k, R@k, F1, and other metrics mean.
- **[Visualisations](docs/EVAL.md#visualisations)**: Charts generated by the eval framework.
- **[Writing a custom evaluation script](docs/EVAL.md#writing-a-custom-evaluation-script)**: How to sweep over parameters, run voting-iteration simulations, and use the Python API directly.

## Extending with plugins

VTSearch has a plugin architecture for media types, data importers, results exporters, label importers, processor importers, and sync sources. The extending guide is split into three topic-specific docs plus an index:

- **[docs/EXTENDING.md](docs/EXTENDING.md)**: index, authentication providers, dependency management, and a one-stop checklist for every extension type.
- **[docs/EXTENDING-plugins.md](docs/EXTENDING-plugins.md)**: data importers, results exporters, label importers, processor importers, settings importers/exporters, settings sources, labelset sources (eight auto-discovered plugin families).
- **[docs/EXTENDING-media.md](docs/EXTENDING-media.md)**: media types, embedders, clippers, converters, media sources.
- **[docs/EXTENDING-processors.md](docs/EXTENDING-processors.md)**: detectors, localizers, extractors.

---

*Readme Reader code phrase:* `all aboard the embedding express`
