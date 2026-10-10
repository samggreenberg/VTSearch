# Setup Guide

**First install?** Work through the sections from [Prerequisites](#prerequisites)
to [Running the app](#running-the-app) in order, then open the
[User Guide](user/USER_GUIDE.md#step-by-step-your-first-search) to load a dataset
and train your first detector. [Docker](#docker) replaces the Python and Node
steps if you would rather not install them, and the [SLURM](#running-on-a-slurm-gpu-cluster)
section is for shared GPU clusters. You will need Python 3.11+, Git, and (for the
frontend build) Node.js 20.19+.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Getting the code](#getting-the-code)
  - [Setting up an SSH key](#setting-up-an-ssh-key)
  - [Clone the repository](#clone-the-repository)
- [Setting up a virtual environment](#setting-up-a-virtual-environment)
- [Installing dependencies](#installing-dependencies)
  - [How auto-detection decides](#how-auto-detection-decides)
  - [Picking the CUDA tag](#picking-the-cuda-tag)
  - [What gets installed](#what-gets-installed)
- [Building the frontend](#building-the-frontend)
- [Running the app](#running-the-app)
- [Docker](#docker)
  - [Prerequisites](#prerequisites-1)
  - [CPU (default)](#cpu-default)
  - [GPU](#gpu)
  - [LabBench (SigLIP-only image search)](#labbench-siglip-only-image-search)
  - [All image embedders](#all-image-embedders)
  - [Data persistence](#data-persistence)
  - [Rebuilding](#rebuilding)
- [Running on a SLURM GPU cluster](#running-on-a-slurm-gpu-cluster)
  - [One-time setup on the cluster](#one-time-setup-on-the-cluster)
  - [One-time setup on your local machine](#one-time-setup-on-your-local-machine)
  - [Daily workflow](#daily-workflow)
  - [Tuning the allocation](#tuning-the-allocation)
- [Running the tests](#running-the-tests)
- [Environment variables](#environment-variables)
- [Next steps](#next-steps)

## Prerequisites

You need **Python 3.11+** installed. Check by running:

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

Activate your virtual environment first (if you made one): the script installs
into whichever `pip` is on your `PATH`. Then run the one installer, which
handles both CPU and GPU machines. With no argument it **auto-detects** whether
this host has an NVIDIA GPU and installs the matching dependency set, so you
don't have to know — or tell it — what hardware you have:

```bash
bash scripts/install.sh              # auto-detect CPU vs GPU (recommended)
```

To override the auto-detection, pass an argument:

```bash
bash scripts/install.sh cpu          # force the CPU-only install
bash scripts/install.sh gpu          # force the GPU install (auto-detect the CUDA tag)
bash scripts/install.sh cu118        # force the GPU install with an explicit tag (CUDA 11.8, older drivers)
bash scripts/install.sh cu128        # ... CUDA 12.8 (Blackwell; drops Volta)
```

Expect several minutes either way; pip goes quiet for tens of seconds at a time
while it resolves versions, which is normal.

### How auto-detection decides

Auto mode makes one of three calls:

- **`nvidia-smi` lists a GPU** → the GPU install, with the CUDA wheel tag
  picked from the GPU's compute capability (see below).
- **No NVIDIA device in the machine at all** (no NVIDIA PCI display
  controller) → the CPU install, with the smaller CPU-only torch wheel
  (~200 MB vs ~2 GB).
- **An NVIDIA card is present but `nvidia-smi` can't see it** — the usual state
  of a fresh cloud GPU instance whose driver isn't installed yet → the script
  explains the situation and **asks** whether to install the NVIDIA driver
  (needs `sudo`, may need a reboot), fall back to a CPU-only install, or stop.

That last case matters for **unattended installs** (a provisioning script, a
Dockerfile, CI, a shell with no terminal): with no one to answer the prompt,
the script stops with exit code 1 rather than run `sudo` or silently land on
CPU. Choose ahead of time: `VTSEARCH_AUTO_DRIVER=1` installs the driver,
`VTSEARCH_ASSUME_CPU=1` goes CPU-only, or pass `cpu` / `gpu` / `cuXYZ`
explicitly, which skips the check entirely. The installer's other switches
(DKMS conversion, a pinned driver `.run` file, verbose output) are listed under
[DEPLOYMENT.md § Install-time](DEPLOYMENT.md#install-time-scriptsinstallsh),
and install problems on GPU boxes are covered in
[DEPLOYMENT.md § Troubleshooting](DEPLOYMENT.md#troubleshooting).

### Picking the CUDA tag

When a GPU is visible, `scripts/detect_cuda_tag.py` picks the torch wheel's
CUDA tag from the GPU's compute capability and the driver's CUDA version; pass
an explicit `cuXYZ` tag only to override it. If the tag can't be determined it
falls back to `cu124`, the widest wheel. You can preview the choice without
installing anything: `python scripts/detect_cuda_tag.py`.

Behind that detection: the CUDA tag picks a torch wheel that only ships kernels
for certain GPU architectures, so it has to match your hardware. There's a
**floor** — newer GPUs need newer tags: Ampere/Ada on `cu118`+, Hopper (H100)
on `cu121`+, Blackwell on `cu128`+ — and a **ceiling**: the newest wheels
*drop* the oldest architectures, so "just use the latest tag" is wrong for old
hardware. For example, `cu128` dropped Volta (`sm_70`), so a **Tesla V100 needs
`cu124`** (or `cu121`/`cu118`), not `cu128` or `cu129`. Rule of thumb: pick the
newest tag your driver supports that still covers your GPU. `cu129` is what
the auto-detect picks for Turing through Blackwell on a driver at CUDA 12.9 or
later, and it is the **only tag that gets cuML** (GPU UMAP / k-means): its
torch pins the CUDA 12.9 libraries that RAPIDS ≥ 26.8 is built on, and RAPIDS
≥ 26.8 is the first whose cudf takes the pandas 3 that `pyproject.toml` pins
(#4390). A Volta card, or an older driver, steps down to `cu124` (or older),
where the installer skips cuML and the app runs UMAP / k-means on the CPU.
A mismatched wheel imports fine and then raises
`cudaErrorNoKernelImageForDevice` on the first GPU op; VTSearch detects this at
runtime and falls back to CPU (with a warning) rather than crashing, but you
only get GPU acceleration with a matching wheel.

### What gets installed

Both paths install every runtime and dev dependency from `pyproject.toml`
(through `requirements/base.txt` or `requirements/gpu.txt`, which forward to
`-e .[dev,agpl]`), editable-install the `vtsearch` package itself, and add a few
packages that need special handling: `toponymy` (names regions on the Browse
map) and `facenet-pytorch` (the face embedder), both installed without their
over-strict dependency pins. In a git checkout they also install the
`pre-commit` git hook.

The **GPU** path additionally:

- installs **cuML / RAPIDS** from NVIDIA's package index for GPU-accelerated
  UMAP and k-means, on the `cu129` tag only (see
  [Picking the CUDA tag](#picking-the-cuda-tag); on any other tag the step
  skips itself with a message). This is a **multi-GB** download; it is
  best-effort (a failure leaves the CPU fallback in place) and
  `VTSEARCH_SKIP_CUML=1` skips it, e.g. on a host that can't reach
  `pypi.nvidia.com`.
- runs a **smoke test** at the end (a CUDA op through torch, then a cuML import)
  and warns if either fails.
- on a driver that isn't DKMS-managed, offers to convert it so the next kernel
  update doesn't break the GPU.

The installer does **not** touch Node.js or npm; the frontend is a separate
step, below.

**Skipping the AGPL dependencies.** Two dependencies are AGPL-3.0-or-later —
`ultralytics` (YOLO) and `PyMuPDF` — and a default install includes them. To
install without them, set `VTSEARCH_NO_AGPL=1` on the install command (or
install `requirements/base-no-agpl.txt` directly); the YOLO extractor/clipper,
PDF import, and the document converters then report themselves as unavailable
and everything else works unchanged. See
[DEPLOYMENT.md](DEPLOYMENT.md#installing-without-the-agpl-dependencies).

## Building the frontend

The Angular frontend must be built after checking out the code; the compiled files are not committed to Git. You'll need **Node.js 20.19+** (or 22.12+ / 24+) and **npm**. Angular 21's engine range accepts all three; the shipped Dockerfiles use `node:20-slim`. The install commands below use Node 22 as the LTS default, but a Node 20.19+ install already on your machine works without upgrading.

Check if they're installed:

```bash
node --version
npm --version
```

If not installed:

<details><summary>Ubuntu / Debian</summary>

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
```

</details>

<details><summary>RHEL / Fedora / Rocky / Alma</summary>

```bash
curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo -E bash -
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

This compiles the Angular app into `static/`, which is what `python app.py` serves. You must run `npm install` before the first build; it installs the Angular CLI and other tools locally under `frontend/node_modules/`. Re-run `npm run build:prod` after pulling new code: the server does not rebuild the frontend itself, and an old bundle against a new server shows a version-mismatch warning.

For development with live reload (proxies API calls to Flask at localhost:5000):

```bash
cd frontend
npm start
```

## Running the app

For local use, start the Flask dev server:

```bash
python app.py
```

Startup takes a minute or so while it loads the ML libraries. When it is ready
it prints:

```
🌐 Open http://localhost:5000 in your browser
```

Open `http://localhost:5000` in your browser. The server binds to
`0.0.0.0:5000`, so it is also reachable from other devices on the network. To
use another port, pass `--port 8080` (or set `VTSEARCH_PORT`). The rest of the
server flags — login providers, admin restrictions, logging verbosity — are in
[CLI.md § Web server modes](CLI.md#web-server-modes). Press **Ctrl+C** to stop.

If the first model download or dataset import fails behind a proxy or on an
offline host, see [DEPLOYMENT.md § Network dependencies](DEPLOYMENT.md#network-dependencies).

`python app.py` uses Flask's built-in dev server, which is fine for development
but **not for production**. For production, run under gunicorn
(`VTSEARCH_SERVER_INIT=1 gunicorn -c gunicorn.conf.py app:app`); why the
variable is needed, the single-worker config and tuning are in
[DEPLOYMENT.md § Running under gunicorn](DEPLOYMENT.md#running-under-gunicorn).
The Docker images below already run that way.

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
> stale build already in the host's `static/` is overwritten with the
> freshly built bundle.
>
> Which image to pick, and what each one is for, is tabled in
> [DEPLOYMENT.md § Choosing an image](DEPLOYMENT.md#choosing-an-image).

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

The GPU images ship torch's `cu129` build, so the host driver must accept
CUDA 12.9 and the GPU must be Turing or newer: a V100 (Volta) host needs a
local `bash scripts/install.sh cu124` instead. See
[DEPLOYMENT.md § Choosing an image](DEPLOYMENT.md#choosing-an-image) for the
accepted driver branches.

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

### All image embedders

`docker/Dockerfile.image-embedders` (CPU) and
`docker/Dockerfile.image-embedders.gpu` (CUDA) are image-only builds that install
every image embedder and bake in the weights of SigLIP (the default), SigLIP 2,
CLIP, DINOv2, DINOv3 and EUPE. The two larger SO400M embedders (SigLIP-L and
SigLIP2-L) are installed but not baked: they download into the `/app/data`
volume the first time someone picks one. DINOv3 is gated on Hugging Face,
so to bake it, run the cache script once on the host with your own token before
building; no token enters the build. Without it the build still succeeds and
DINOv3 simply stays unavailable. (EUPE's license forbids commercial use; see the
README's License section.)

```bash
HF_TOKEN=hf_xxx ./scripts/cache_gated_models.sh      # optional, one-time: populates ./model_cache/

docker build -f docker/Dockerfile.image-embedders -t vtsearch:image-embedders .
docker run -p 5000:5000 -v vtsearch-data:/app/data vtsearch:image-embedders

docker compose -f docker/compose/docker-compose.image-embedders.gpu.yml up   # GPU variant
```

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
docker compose -f docker/compose/docker-compose.image-embedders.gpu.yml build  # All image embedders (GPU)
```

Add `--no-cache` to force a full rebuild (e.g. after dependency changes).

## Running on a SLURM GPU cluster

VTSearch is happiest with a GPU (for embedding and detector training). On a
shared SLURM cluster — like the JHU HLTCOE "Grid" — you don't run heavy work on
the login nodes; you ask SLURM for a GPU compute node and run the app there,
then forward its port back to your local machine so you can use the browser UI.

Two helper scripts in [`scripts/slurm/`](../scripts/slurm/) automate the loop:

- **`vtsearch-slurm.sh`** runs *on the cluster*. It allocates a GPU node with
  `srun`, activates the virtualenv, and runs `app.py` on the node — printing
  which node and port it landed on. It holds the allocation until you quit.
- **`vtsearch-tunnel.sh`** runs *on your local machine*. It finds your running
  VTSearch job, SSH-forwards your local port to that compute node, and drops
  you into the project directory for git/edits.

Both scripts are parameterized entirely by environment variables (no hard-coded
usernames, hostnames, or paths), so they should adapt to most SLURM clusters
with a shared filesystem.

> **Why a per-user port?** GPU nodes usually hold several GPUs, so SLURM can
> pack multiple users' single-GPU jobs onto one physical node — where a single
> shared `:5000` would collide (the second app can't bind it, and a naive
> tunnel would forward you into someone else's session). So the scripts derive
> a port from your UID (`10000 + UID % 20000`); both compute the same value, so
> the tunnel finds your app with no extra coordination. `app.py` honors this
> via `VTSEARCH_PORT` (or `--port`). Override with `VTS_PORT` on *both* scripts
> if you ever need to.

### One-time setup on the cluster

1. **SSH in** to a login node and clone VTSearch onto the cluster's shared
   filesystem. Many clusters give each user a large scratch/experiment area
   (the HLTCOE Grid uses `/exp/$USER`); the helper scripts default to
   `/exp/$USER/projects/VTSearch`, but you can put it anywhere and set
   `VTS_DIR` (see [Tuning](#tuning-the-allocation)).

   ```bash
   mkdir -p /exp/$USER/projects && cd /exp/$USER/projects
   git clone git@github.com:samggreenberg/VTSearch.git
   cd VTSearch
   ```

2. **Create the virtualenv and install the GPU dependencies.** Match the CUDA
   wheel to your cluster's drivers and cards (a node whose driver reports CUDA
   12.4 takes `cu124`; a Turing-or-newer card on a driver at CUDA 12.9 or
   later takes `cu129`, the one tag that also gets cuML; a V100 takes `cu124`
   whatever the driver):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   bash scripts/install.sh cu129         # or cu118 / cu121 / cu124 / cu128 to match your node's GPU and driver
   ```

   The scripts default to a venv named `.venv` in the project dir; override with
   `VTS_VENV` if yours differs.

   > **Module-based Python (e.g. the HLTCOE Grid).** Many clusters ship Python
   > only via environment modules, and the system `python3` may be too old
   > (VTSearch needs 3.11+). Load a recent one first, and build the venv with
   > the **versioned** interpreter name so a `pyenv` shim on your `PATH` can't
   > shadow it:
   >
   > ```bash
   > module avail python                 # find an available 3.11+ module
   > module load python/3.12.3
   > which python3.12                     # should be the module's, not a pyenv shim
   > python3.12 -m venv .venv
   > source .venv/bin/activate
   > python --version                     # confirm 3.12.x before installing
   > bash scripts/install.sh cu124
   > ```
   >
   > A venv built this way is **not** self-contained — its `python` needs the
   > module's `libpython` at runtime — so the launcher must load the module
   > before activating the venv. Set `VTS_MODULE` (see
   > [Tuning](#tuning-the-allocation)) so `vtsearch` does this for you:
   > `VTS_MODULE="python/3.12.3" vtsearch` (or `export` it in `~/.bashrc`).

3. **Build the frontend** (needs Node.js 20.19+; see [Building the
   frontend](#building-the-frontend)):

   ```bash
   cd frontend && npm install && npm run build:prod && cd ..
   ```

4. **Install the launcher on your PATH** so you can just type `vtsearch`:

   ```bash
   mkdir -p ~/.local/bin
   cp scripts/slurm/vtsearch-slurm.sh ~/.local/bin/vtsearch
   chmod +x ~/.local/bin/vtsearch
   ```

   > **Tip:** caches (HuggingFace, pip, etc.) can be large; on clusters where
   > `/home` is small, redirect them onto your scratch area in `~/.bashrc`
   > (e.g. `export HF_HOME=/exp/$USER/.cache/huggingface`). Setting `HF_TOKEN`
   > there too avoids anonymous Hugging Face rate limits on a shared egress IP.

### One-time setup on your local machine

1. **Add an SSH host entry** for the cluster login node so the tunnel script
   can reach it by a short name. In `~/.ssh/config`:

   ```sshconfig
   Host cluster
       HostName login.your-cluster.edu     # e.g. login1.hltcoe.jhu.edu
       User your-cluster-username
       IdentityFile ~/.ssh/id_ed25519
   ```

   Verify it works: `ssh cluster true` should connect without prompting. If your
   cluster is only reachable through a VPN or campus network, connect to that
   first.

2. **Install the tunnel script** on your PATH:

   ```bash
   mkdir -p ~/.local/bin
   cp scripts/slurm/vtsearch-tunnel.sh ~/.local/bin/vtsearch-tunnel
   chmod +x ~/.local/bin/vtsearch-tunnel
   ```

   (Clone VTSearch on your local machine too, or just copy the one script — it only
   needs SSH access to the cluster.) If you named your SSH host something other
   than `cluster`, point the script at it with `CLUSTER_HOST=mycluster vtsearch-tunnel`.

### Daily workflow

1. **On the cluster**, start VTSearch and leave the terminal running:

   ```bash
   ssh cluster
   vtsearch
   ```

   This queues a GPU allocation; once it lands, the app starts and the terminal
   prints the compute node it got. Keep this terminal open — closing it releases
   the node.

2. **On your local machine**, in a second terminal, open the tunnel:

   ```bash
   vtsearch-tunnel
   ```

   It finds the running job automatically (no need to know the node name or
   port), forwards your local port to it, and drops you into the project
   directory on the login node for git pulls / edits. It prints the URL to
   browse — **http://localhost:&lt;port&gt;** (the per-user port, see above).

3. **To pick up code changes**: pull on the cluster, then in the `vtsearch`
   terminal press **Ctrl+C** (this stops `app.py` but *keeps* the GPU node) and
   press **Enter** to restart the app. Press **q** then Enter to release the
   node and quit.

> Interactive SLURM jobs don't persist — you re-allocate each session. Only run
> **one** `app.py` per allocation; VTSearch keeps all model/dataset state in one
> process, and a second copy would double the memory and can OOM the job.

### Tuning the allocation

`vtsearch-slurm.sh` reads these environment variables (defaults shown). Set them
inline, e.g. `VTS_MEM=64G VTS_GPU=a100 vtsearch`:

| Variable | Default | Meaning |
|----------|---------|---------|
| `VTS_DIR` | `/exp/$USER/projects/VTSearch` | Path to the VTSearch checkout on the cluster |
| `VTS_VENV` | `.venv` | Virtualenv to activate (relative to `VTS_DIR`, or absolute) |
| `VTS_MODULE` | (none) | Environment module(s) to `module load` before activating the venv (space-separated). Needed when your venv is built from a module-provided Python, e.g. `VTS_MODULE="python/3.12.3"` |
| `VTS_PART` | `gpu` | SLURM partition |
| `VTS_GPU` | (auto-picked) | GPU type requested via `--gres=gpu:<type>:1`. Unset, the type is chosen from what is free (see below); set it to pin one |
| `VTS_GPU_TYPES` | `a100 l40s v100` | Candidate types for the auto-pick, **fastest first** |
| `VTS_GPU_FALLBACK` | `l40s` | Type to request when the scheduler can't be queried |
| `VTS_CPUS` | `8` | CPU cores |
| `VTS_MEM` | `48G` | Memory (headroom for two model loads in one process) |
| `VTS_TIME` | `8:00:00` | Walltime |
| `VTS_PORT` | `10000 + UID % 20000` | Port the app binds (passed as `VTSEARCH_PORT`); per-user by default so co-located jobs don't collide |

`vtsearch-tunnel.sh` reads:

| Variable | Default | Meaning |
|----------|---------|---------|
| `CLUSTER_HOST` | `cluster` | SSH host alias for the cluster login node |
| `VTS_DIR` | `/exp/$USER/projects/VTSearch` | Project dir to drop into on the login node |
| `VTS_PORT` | (cluster-computed) | Forwarded port; defaults to the same per-user value the launcher binds. Set it only if you overrode `VTS_PORT` on the cluster too |
| `VTS_BIND` | (unset — loopback only) | Address the forwarded port is bound to *locally*. Unset, only this machine can reach it. Set it to serve the app to another device through this one — see below |

It also takes one flag, `--no-shell`, which holds the forward open and nothing
else (`ssh -N`, no TTY, no login shell) instead of dropping you into the login
node. That is the mode a service unit wants; the interactive default is the mode
a human wants.

#### Relaying the app to another device

If the machine running the tunnel is a always-on box that holds the VPN — and
the device you actually browse from is a different one — the two knobs above are
what bridge them. `VTS_BIND` binds the forwarded port to an address that other
device can reach:

```bash
VTS_BIND=$(tailscale ip -4) vtsearch-tunnel
# → ssh -L 100.x.y.z:PORT:NODE:PORT cluster
# → browse http://100.x.y.z:PORT from any device on the same private mesh
```

**Bind to that specific address, not `0.0.0.0`.** The wildcard also serves the
app to everything else on the local network, which on a home or campus Wi-Fi is
a larger audience than intended.

To keep the tunnel up without a terminal, install
[`scripts/slurm/vtsearch-tunnel.service`](../scripts/slurm/vtsearch-tunnel.service)
as a systemd **user** unit — it runs the script with `--no-shell` under
`Restart=always`:

```bash
mkdir -p ~/.config/systemd/user
cp scripts/slurm/vtsearch-tunnel.service ~/.config/systemd/user/
# set Environment=VTS_BIND=... in the copy, then:
systemctl --user daemon-reload
systemctl --user enable --now vtsearch-tunnel
loginctl enable-linger "$USER"   # keeps it running with nobody logged in
```

That last line is the one that gets missed: without lingering, a user unit stops
when your last session ends — precisely when an unattended box needed it to keep
going.

The unit supervises the **script**, not a bare `ssh`, and that distinction
matters. SLURM re-picks the GPU node on every allocation, so a reconnect aimed
at the previous node forwards to nothing while still looking healthy. Re-running
the script re-queries `squeue` and rediscovers node and port, which is also why
no `autossh` is needed here.

#### Which GPU type gets requested

Many clusters (HLTCOE's included) reject an untyped `--gres=gpu:1`, so a type
has to be named — and any single name is a pin that goes stale. Pinning the slow
type costs a multiple on every embed (a V100 embeds `siglip2_l` **2.3×** slower
than an L40S on identical fp32 code); pinning a scarce fast type has meant
multi-day queue waits. So the launcher doesn't pin: it runs
[`scripts/slurm/pick_gpu.py`](../scripts/slurm/pick_gpu.py), which reads
`scontrol show node` and requests the **fastest type in `VTS_GPU_TYPES` that has
a free GPU right now**, falling back to whatever is most available, then to the
largest pool, then to `VTS_GPU_FALLBACK`. Run it by hand to see the reasoning:

```bash
python3 scripts/slurm/pick_gpu.py --explain
```

Set `VTS_GPU` to override it outright (nothing is queried then). Set
`VTS_GPU_TYPES` to match your own hardware and QOS — the default omits
`h100`/`h200` because the HLTCOE `4gpu_tier` QOS caps them at 0, and a job
requesting a type your QOS forbids pends forever.

Adjust `VTS_GPU_TYPES`, `VTS_PART`, and the CUDA wheel passed to `install.sh` to
match your cluster's hardware. To find the exact GPU type strings, check a node's
gres: `scontrol show node <node> | grep -i Gres` (e.g. `Gres=gpu:v100:8` → use
`v100`) or list partition gres with `sinfo -o '%P %G'`. Note these variables take
just the type (`v100`), not the full `gpu:v100:8` spec — the launcher adds the
`gpu:` prefix and `:1` count itself.

## Running the tests

The Python test and lint tools (pytest, ruff, pyright, …) are already
installed if you ran `bash scripts/install.sh` above, and the frontend's
build and unit-test tools come from `npm install` in `frontend/`. Whatever is
missing, `./run-tests.sh` installs on first run.

The recommended way to run tests uses the helper script, which installs
dependencies automatically and supports grouped test subsets:

```bash
./run-tests.sh              # full fast CPU suite
./run-tests.sh core         # basic app functionality only
./run-tests.sh sorting api  # multiple groups
```

The group list, what each covers, and which gates a group run skips are in
[TESTING.md § Test Groups](TESTING.md#test-groups). A full `./run-tests.sh` is
the merge gate; a bare run on a branch that changes only markdown narrows itself
to the `docs` gate automatically.

You can also run pytest directly:

```bash
python -m pytest tests/ tests_lib/ -v
```

This runs fast CPU tests only. Additional test modes:

**Full CPU tests** (adds the `slow`-marked tests: a CLI subprocess run and the toponymy fit tests):

```bash
python -m pytest tests/ tests_lib/ -v -m 'not gpu'
```

**GPU tests** (requires CUDA):

```bash
python -m pytest tests_lib/gpu/test_gpu.py -v -m gpu
```

**All tests**:

```bash
python -m pytest tests/ tests_lib/ -v -m ''
```

## Environment variables

VTSearch reads several optional environment variables. The ones below are the
ones a first install is most likely to want; the installer's own switches
(`VTSEARCH_AUTO_DRIVER`, `VTSEARCH_SKIP_CUML`, …) are covered under
[Installing dependencies](#installing-dependencies).

| Variable | Default | Description |
|----------|---------|-------------|
| `VTSEARCH_DATA_DIR` | `<repo root>/data` | Where all runtime state lives (settings, datasets, detectors, model cache, demo downloads). Point it outside the checkout to keep state across re-clones. |
| `VTSEARCH_SECRET_KEY` | `vtsearch-dev-key-change-in-production` | Flask session secret key (set this in production) |
| `VTSEARCH_LOG_LEVEL` | `WARNING` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`); `INFO`/`DEBUG` also enable the per-request access log. `python app.py -v`/`-vv` is the CLI shortcut. |
| `VTSEARCH_MODELS_DIR` | `$VTSEARCH_DATA_DIR/models` | Directory for HuggingFace model cache |
| `VTSEARCH_PORT` | `5000` | Port for the `python app.py` dev server (also `--port`). Lets several instances share a host, e.g. co-located SLURM jobs. Gunicorn uses `VTSEARCH_BIND` instead. |

The full reference — gunicorn (`VTSEARCH_BIND`, `VTSEARCH_THREADS`, `VTSEARCH_TIMEOUT`), log files, threading and offline variables — is [DEPLOYMENT.md § Environment variables](DEPLOYMENT.md#environment-variables).

## Next steps

- **Use the app**: See [user/USER_GUIDE.md](user/USER_GUIDE.md) for a walkthrough
  of loading a dataset, labeling with Autopilot, and exporting results.
- **Run tests**: See [Running the tests](#running-the-tests) above.
- **Learn the codebase**: See [ARCHITECTURE.md](ARCHITECTURE.md#key-concepts)
  for the core vocabulary (media items, votes, media types, processors,
  origins) and the module-by-module map, and
  [FRONTEND.md](FRONTEND.md) for the Angular SPA.
- **CLI workflows**: See [CLI.md](CLI.md) for running detectors and
  exporters from the command line.
- **Evaluate sorting quality**: `python -m vtscore.eval --plot-dir eval_output`
  runs the evaluation suite over the demo datasets; see [EVAL.md](EVAL.md).
- **Deploy**: See [DEPLOYMENT.md](DEPLOYMENT.md) for production, offline /
  air-gapped, and GPU deployments.
- **Extend**: See [EXTENDING.md](EXTENDING.md) for adding new media
  types, importers, or exporters.
