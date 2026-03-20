# Setup Guide

## Prerequisites

You need **Python 3.10+** installed. Check by running:

```bash
python3 --version
```

If you see something like `Python 3.11.4`, you're good. If the command isn't found, install Python from [python.org/downloads](https://www.python.org/downloads/) or with your system package manager.

Ubuntu / Debian:

```bash
sudo apt update && sudo apt install python3 python3-pip python3-venv
```

macOS (with Homebrew):

```bash
brew install python
```

You also need **Git** to download the code:

```bash
git --version
```

If it's not installed:

Ubuntu / Debian:

```bash
sudo apt install git
```

macOS (with Homebrew):

```bash
brew install git
```

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

Dependencies are declared in `pyproject.toml`. All media types (audio, image, text, video, document) are included in the base install — you just pick CPU or GPU for PyTorch.

**For CPU only** (recommended if you don't have a compatible GPU):

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[cpu,dev]"
```

**For GPU** (NVIDIA CUDA-compatible systems):

```bash
bash install-gpu.sh          # defaults to CUDA 11.8
bash install-gpu.sh cu121    # for CUDA 12.1
bash install-gpu.sh cu124    # for CUDA 12.4
```

The `--extra-index-url` flag pulls the smaller CPU-only PyTorch wheel (~200 MB) instead of the default CUDA build (~2 GB). The `install-gpu.sh` script handles selecting the right CUDA version.

## Building the frontend (optional)

The Angular frontend is pre-built and committed to `static/`, so you can skip this step if you're only working on the backend. If you need to modify the frontend, you'll need **Node.js 18+** and **npm**.

Check if they're installed:

```bash
node --version
npm --version
```

If not installed:

Ubuntu / Debian:

```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs
```

macOS (with Homebrew):

```bash
brew install node
```

Then install dependencies and build:

```bash
cd frontend; npm install; npm run build:prod; cd ..
```

This compiles the Angular app into `static/` (index.html, main.js, polyfills.js, styles.css). You must run `npm install` before the first build — it installs the Angular CLI and other tools locally.

For development with live reload (proxies API calls to Flask at localhost:5000):

```bash
cd frontend
npm start
```

## Running the app

Start the server:

```bash
python app.py
```

You should see output like:

```
 * Running on http://127.0.0.1:5000
```

Open that URL in your browser. For faster startup during development, use `--local` mode (loads embedding models lazily instead of eagerly):

```bash
python app.py --local
```

## Docker

If you prefer containers over a local Python install, VTSearch ships with ready-made Docker support.

### Prerequisites

Install [Docker](https://docs.docker.com/get-docker/) (includes Docker Compose). For GPU images you also need the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/).

### CPU (default)

Using Docker Compose (recommended):

```bash
docker compose up            # build & run (foreground)
docker compose up -d         # build & run (detached)
docker compose down          # stop & remove
```

Or with plain Docker:

```bash
docker build -t vtsearch .
docker run -p 5000:5000 -v vtsearch-data:/app/data vtsearch
```

### GPU

Using Docker Compose:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

Or with plain Docker:

```bash
docker build -f Dockerfile.gpu -t vtsearch:gpu .
docker run --gpus all -p 5000:5000 -v vtsearch-data:/app/data vtsearch:gpu
```

### Data persistence

The `data/` directory inside the container (models, embeddings, settings, media files) is declared as a Docker volume. The commands above mount it as a named volume called `vtsearch-data` so everything persists across container restarts. To use a host directory instead:

```bash
docker run -p 5000:5000 -v /path/on/host:/app/data vtsearch
```

### Rebuilding

After pulling new code, rebuild the image:

```bash
docker compose build           # CPU
docker compose -f docker-compose.yml -f docker-compose.gpu.yml build  # GPU
```

Add `--no-cache` to force a full rebuild (e.g. after dependency changes).

## Running the tests

Install dev dependencies (includes pytest):

```bash
pip install -e ".[dev]"
```

Then run:

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

## Next steps

- **Load a dataset**: Click the hamburger menu in the top-left corner to browse demo datasets or import your own data.
- **Run tests**: See [Running the tests](#running-the-tests) above.
- **CLI workflows**: See [CLI.md](CLI.md) for running detectors and exporters from the command line.
- **Extend**: See [EXTENDING.md](EXTENDING.md) for adding new media types, importers, or exporters.
