# Setup Guide

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting the code](#getting-the-code)
  - [Setting up an SSH key](#setting-up-an-ssh-key)
  - [Clone the repository](#clone-the-repository)
- [Setting up a virtual environment](#setting-up-a-virtual-environment)
- [Installing dependencies](#installing-dependencies)
- [Building the frontend](#building-the-frontend)
- [Running the app](#running-the-app)
- [Docker](#docker)
  - [Prerequisites](#prerequisites-1)
  - [CPU (default)](#cpu-default)
  - [GPU](#gpu)
  - [Data persistence](#data-persistence)
  - [Rebuilding](#rebuilding)
- [Running the tests](#running-the-tests)
- [Environment variables](#environment-variables)
- [Next steps](#next-steps)

## Prerequisites

You need **Python 3.10+** installed. Check by running:

```bash
python3 --version
```

If you see something like `Python 3.11.4`, you're good. If the command isn't found, install Python from [python.org/downloads](https://www.python.org/downloads/) or with your system package manager.

<details><summary>Ubuntu / Debian</summary>

```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv
```

</details>

<details><summary>RHEL / Fedora / Rocky / Alma</summary>

```bash
sudo dnf install python3 python3-pip
```

</details>

<details><summary>macOS (Homebrew)</summary>

```bash
brew install python
```

</details>

You also need **Git** to download the code:

```bash
git --version
```

If it's not installed:

<details><summary>Ubuntu / Debian</summary>

```bash
sudo apt install git
```

</details>

<details><summary>RHEL / Fedora / Rocky / Alma</summary>

```bash
sudo dnf install git
```

</details>

<details><summary>macOS (Homebrew)</summary>

```bash
brew install git
```

</details>

## Getting the code

We recommend cloning over SSH so you don't have to enter your password on every push/pull.

### Setting up an SSH key

1. **Generate a key** (skip this if you already have one at `~/.ssh/id_ed25519`):

   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   ```

   Press Enter to accept the default file location, then choose a passphrase (or leave it empty).

2. **Start the SSH agent and add your key**:

   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. **Copy the public key** to your clipboard:

   Linux:

   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```

   macOS:

   ```bash
   pbcopy < ~/.ssh/id_ed25519.pub
   ```

4. **Add the key to GitHub**: Go to [github.com/settings/ssh/new](https://github.com/settings/ssh/new), paste the public key, give it a title, and click **Add SSH key**.

5. **Verify the connection**:

   ```bash
   ssh -T git@github.com
   ```

   You should see a message like *"Hi username! You've successfully authenticated…"*.

### Clone the repository

```bash
git clone git@github.com:samggreenberg/VTSearch.git
cd VTSearch
```

## Setting up a virtual environment

A virtual environment keeps this project's dependencies separate from the rest of your system. This is optional but recommended.

```bash
python3 -m venv venv
```

Then activate it:

Linux / macOS:

```bash
source venv/bin/activate
```

Windows (Command Prompt):

```bat
venv\Scripts\activate.bat
```

Windows (PowerShell):

```powershell
venv\Scripts\Activate.ps1
```

When activated, you'll see `(venv)` at the start of your terminal prompt.

## Installing dependencies

Runtime + dev dependencies are declared in `pyproject.toml` (under
`[project.dependencies]` and `[project.optional-dependencies].dev`).
`requirements/base.txt` and `requirements/gpu.txt` just forward to it via
`-e .[dev]`, so pyproject is the single source of truth and deptry
catches any drift. The labbench / image-embedders requirements files
under `requirements/` are deliberately standalone; they pin a minimal
subset for size-constrained Docker images.

**For CPU only** (recommended if you don't have a compatible GPU):

```bash
bash scripts/install-cpu.sh
```

**For GPU** (NVIDIA CUDA-compatible systems):

```bash
bash scripts/install-gpu.sh          # defaults to CUDA 11.8 (cu118)
bash scripts/install-gpu.sh cu121    # for CUDA 12.1
bash scripts/install-gpu.sh cu124    # for CUDA 12.4
```

Both scripts run `pip install -r requirements/{base,gpu}.txt`, which
installs every runtime + dev dep and editable-installs the `vtsearch`
package itself.

The CPU `requirements/base.txt` includes `--extra-index-url` for the
smaller CPU-only PyTorch wheel (~200 MB) instead of the default CUDA
build (~2 GB).

## Building the frontend

The Angular frontend must be built after checking out the code; the compiled files are not committed to Git. You'll need **Node.js 18+** and **npm**.

Check if they're installed:

```bash
node --version
npm --version
```

If not installed:

<details><summary>Ubuntu / Debian</summary>

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

</details>

<details><summary>RHEL / Fedora / Rocky / Alma</summary>

```bash
curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo -E bash -
sudo dnf install -y nodejs
```

</details>

<details><summary>macOS (Homebrew)</summary>

```bash
brew install node
```

</details>

Then install dependencies and build:

```bash
cd frontend; npm install; npm run build:prod; cd ..
```

This compiles the Angular app into `static/` (index.html, main.js, polyfills.js, styles.css). You must run `npm install` before the first build; it installs the Angular CLI and other tools locally.

For development with live reload (proxies API calls to Flask at localhost:5000):

```bash
cd frontend
npm start
```

## Running the app

For local development, start the Flask dev server:

```bash
python app.py
```

You should see output like:

```
 * Running on http://0.0.0.0:5000
```

Open `http://localhost:5000` in your browser. The server binds to `0.0.0.0:5000`, so it is also reachable from other devices on the network.

`python app.py` uses Flask's built-in dev server (fine for development, but **not recommended for production**). For production, run under gunicorn using the bundled config:

```bash
VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app
```

`VTSEARCH_SERVER_INIT=1` triggers the same startup sequence (model init, autoload preloading, settings-source sync) that `python app.py` runs, since gunicorn imports `app.py` rather than executing its `__main__` block. `gunicorn.conf.py` pins a single worker with 8 threads; VTSearch keeps all dataset/model state in-process, so multiple workers would each hold their own copy. See [DEPLOYMENT.md](DEPLOYMENT.md#gunicorn-tuning) for tuning.

The Docker images already use this configuration (see [Docker](#docker) below).

## Docker

If you prefer containers over a local Python install, VTSearch ships with ready-made Docker support.

### Prerequisites

Install [Docker](https://docs.docker.com/get-docker/) (includes Docker Compose). For GPU images you also need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

### CPU (default)

Using Docker Compose (recommended; run from the repo root):

```bash
docker compose -f docker/compose/docker-compose.yml up            # build & run (foreground)
docker compose -f docker/compose/docker-compose.yml up -d         # build & run (detached)
docker compose -f docker/compose/docker-compose.yml down          # stop & remove
```

Or with plain Docker (from the repo root):

```bash
docker build -f docker/Dockerfile -t vtsearch .
docker run -p 5000:5000 -v vtsearch-data:/app/data vtsearch
```

> **Note:** Every VTSearch Dockerfile (`Dockerfile`, `Dockerfile.gpu`,
> `Dockerfile.labbench`, `Dockerfile.image-embedders`,
> `Dockerfile.image-embedders.gpu`) includes a Node.js `frontend` build
> stage that runs `npm ci` and `npm run build:prod` inside the image, so
> you do **not** need to build the Angular app on the host first; any
> stale checked-in `static/` is overwritten with the freshly built bundle.

### GPU

Using Docker Compose:

```bash
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.gpu.yml up
```

Or with plain Docker:

```bash
docker build -f docker/Dockerfile.gpu -t vtsearch:gpu .
docker run --gpus all -p 5000:5000 -v vtsearch-data:/app/data vtsearch:gpu
```

### LabBench (SigLIP-only image search)

For the LabBench deployment (image search with the SigLIP
embedder), use the streamlined `docker/Dockerfile.labbench` variant. It skips
audio, video, document, text, and extractor plugin dependencies, and **bakes
the SigLIP model weights into the image at build time** so the container is
ready to serve immediately on first run (no Hugging Face download).

```bash
docker compose -f docker/compose/docker-compose.labbench.yml up
```

Or with plain Docker:

```bash
docker build -f docker/Dockerfile.labbench -t vtsearch:labbench .
docker run -p 5000:5000 -v vtsearch-data:/app/data vtsearch:labbench
```

The model cache lives in `/opt/vtsearch/models` (set via `VTSEARCH_MODELS_DIR`)
so the baked weights are not masked when `/app/data` is mounted as a volume.

### Data persistence

The `data/` directory inside the container (models, embeddings, settings, media files) is declared as a Docker volume. The commands above mount it as a named volume called `vtsearch-data` so everything persists across container restarts. To use a host directory instead:

```bash
docker run -p 5000:5000 -v /path/on/host:/app/data vtsearch
```

### Rebuilding

After pulling new code, rebuild the image:

```bash
docker compose -f docker/compose/docker-compose.yml build           # CPU
docker compose \
  -f docker/compose/docker-compose.yml \
  -f docker/compose/docker-compose.gpu.yml build                    # GPU
docker compose -f docker/compose/docker-compose.labbench.yml build  # LabBench (SigLIP-only)
```

Add `--no-cache` to force a full rebuild (e.g. after dependency changes).

## Running the tests

Install dependencies (pytest and ruff are included in `requirements/base.txt`):

```bash
pip install -r requirements/base.txt
```

The recommended way to run tests uses the helper script, which installs
dependencies automatically and supports grouped test subsets:

```bash
./run-tests.sh              # full fast CPU suite
./run-tests.sh core         # basic app functionality only
./run-tests.sh sorting api  # multiple groups
```

Available groups: `core`, `api`, `sorting`, `datasets`, `io`, `models`,
`downloads`, `integration`, `cli`, `converters`. See [`CLAUDE.md`](../CLAUDE.md) for the
full group-to-file mapping.

You can also run pytest directly:

```bash
python -m pytest tests/ -v
```

This runs fast CPU tests only. Additional test modes:

**Full CPU tests** (includes slow CLI subprocess tests):

```bash
python -m pytest tests/ -v -m 'not gpu'
```

**GPU tests** (requires CUDA):

```bash
python -m pytest tests/test_gpu.py -v -m gpu
```

**All tests**:

```bash
python -m pytest tests/ -v -m ''
```

## Environment variables

VTSearch reads several optional environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_SECRET_KEY` | `vtsearch-dev-key-change-in-production` | Flask session secret key (set this in production) |
| `VTSEARCH_LOG_LEVEL` | `WARNING` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `VTSEARCH_MODELS_DIR` | `data/models` | Directory for HuggingFace model cache |
| `VTSEARCH_SERVER_INIT` | unset | Set to `1` when running under gunicorn; triggers model init / settings sync at import time |
| `VTSEARCH_BIND` | `0.0.0.0:5000` | Gunicorn bind address (`host:port`) |
| `VTSEARCH_THREADS` | `8` | Threads per gunicorn worker |
| `VTSEARCH_TIMEOUT` | `0` | Gunicorn worker timeout in seconds (`0` = disabled; long imports / training would otherwise SIGKILL the worker) |

See [DEPLOYMENT.md](DEPLOYMENT.md) for additional deployment-specific configuration, including the full env-var reference and gunicorn tuning.

## Next steps

- **Use the app**: See [user/USER_GUIDE.md](user/USER_GUIDE.md) for a walkthrough
  of loading a dataset, labeling with Autopilot, and exporting results.
- **Run tests**: See [Running the tests](#running-the-tests) above.
- **CLI workflows**: See [CLI.md](CLI.md) for running detectors and
  exporters from the command line.
- **Extend**: See [EXTENDING.md](EXTENDING.md) for adding new media
  types, importers, or exporters.
