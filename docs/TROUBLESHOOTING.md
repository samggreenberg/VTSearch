# Troubleshooting

Symptoms and fixes, in two groups: the app, its install and its container
first, then GPU installs (the NVIDIA driver, the CUDA wheel and cuML). For
installing VTSearch see [SETUP.md](SETUP.md); for running it as a service, and
for every environment variable named below, see [DEPLOYMENT.md](DEPLOYMENT.md).

- [General](#general)
  - [The app freezes for seconds during labeling (diagnosing a stall)](#the-app-freezes-for-seconds-during-labeling-diagnosing-a-stall)
  - [Models fail to download](#models-fail-to-download)
  - [Out of memory](#out-of-memory)
  - [Docker build fails on pip install](#docker-build-fails-on-pip-install)
  - [Settings not persisting across container restarts](#settings-not-persisting-across-container-restarts)
  - [Demo dataset download fails](#demo-dataset-download-fails)
  - [Video import crashes with `FATAL FIPS SELFTEST FAILURE`](#video-import-crashes-with-fatal-fips-selftest-failure)
  - [A feature you just pulled isn't in the UI ("out-of-date build")](#a-feature-you-just-pulled-isnt-in-the-ui-out-of-date-build)
  - ["No module named" errors](#no-module-named-errors)
  - [Browse maps render unlettered ("VTSBrowse signposts are disabled")](#browse-maps-render-unlettered-vtsbrowse-signposts-are-disabled)
- [GPU installs: driver, CUDA and cuML](#gpu-installs-driver-cuda-and-cuml)
  - [`install.sh` installs CPU torch on a machine that has a GPU](#installsh-installs-cpu-torch-on-a-machine-that-has-a-gpu)
  - [Making the GPU driver survive reboots and kernel updates](#making-the-gpu-driver-survive-reboots-and-kernel-updates)
  - [GPU install's cuML step (skipped on `cu118` and `cu128`)](#gpu-installs-cuml-step-skipped-on-cu118-and-cu128)
  - [cuML crashes compiling a kernel (`cuda_fp8.hpp` / nvrtc errors)](#cuml-crashes-compiling-a-kernel-cuda_fp8hpp--nvrtc-errors)

## General

### The app freezes for seconds during labeling (diagnosing a stall)

A rare multi-second pause in which *every* in-flight request finishes at the
same instant is not one slow endpoint. It is either a thread
holding the GIL, a lock convoy, or the process not being scheduled at all,
and the per-request timer cannot tell those apart. The app carries three
instruments that can, all on at the default log level:

- **The stall watchdog** (`VTSEARCH_STALL_WATCHDOG_MS`). A heartbeat thread
  measures how late it wakes. When it misses by more than the threshold it
  logs `stall: heartbeat late by …ms` with the process's CPU time over the
  gap, the threads that consumed it, major page faults, RSS, the cgroup memory
  counters and GC activity. Until the server is ready the threshold is
  `VTSEARCH_STALL_STARTUP_MS` (5 s) instead: the ML imports and the embedder
  preload hold the GIL across cold-cache disk reads that have exactly a
  stall's signature, and a `stall:` line from that window says `bar 5000ms`.
  Read it like this:
  - *process cpu ≈ wall, one thread on top* → that thread held the GIL.
    Just above the `stall:` line, the watchdog writes every thread's Python
    stack (`Stall snapshot …`, to `VTSEARCH_STALL_DUMP_FILE`, which defaults
    to the log file), with the thread that burned the most CPU first. Read
    that thread's top frames. They are taken the moment the heartbeat wakes,
    and a thread holding the GIL in C code can only give it up at the call
    that stalled, so that is where it still is. `VTSEARCH_STALL_LIVE_DUMP=1`
    adds `faulthandler`'s dump from *during* the stall, which can crash the
    app (#4345).
  - *no thread consumed cpu* → the process was not running: look at `majflt`
    and the cgroup `limit hits` (memory pressure), or at the node.
  - *cpu spread across threads* → contention rather than one holder; the lock
    and phase lines below say where.
- **GC pauses** (`VTSEARCH_GC_WARN_MS`): `gc pause: generation 2 took …ms`. A
  full collection holds the GIL and shows in a thread dump only as an
  arbitrary allocation site, so it is named separately. Left unset this bar
  **tracks `VTSEARCH_SLOW_PHASE_MS`**, so lowering the phase bar for a
  diagnostic session cannot leave collections below it invisible — they would
  otherwise still inflate every phase they land in, with nothing saying why.
- **Phase and lock timers** (`VTSEARCH_SLOW_PHASE_MS`): `slow phase: <name>
  total …ms cpu=…ms gc=…ms (phase=…ms, …)` for the learned-sort retrain, the
  per-vote labelset rewrite, the labeling-status replay and a vote rehydrate;
  and `lock wait: <lock> waited …ms` when a vote, the sort thread or the status
  poll queued behind another holder.

**Read the `cpu=` column first.** Every request and phase line carries the
thread's own CPU time beside its wall time, and the two together say what
kind of slow it was:

| | reading |
|---|---|
| `cpu ≈ wall` | it did the work — a real cost that grows with the data |
| `cpu ≪ wall`, `gc` small | it blocked (a lock, an `fsync`, a slow filesystem) or was descheduled — an 8-thread torch process in an 8-CPU cgroup on a shared node is descheduled routinely |
| `gc ≈ wall` | a collection froze every thread; the phase is innocent |

`gc` is not additive with `cpu` — a collection running on that thread burned
its CPU too, so `total 1137ms cpu=1133ms gc=139ms` reads "on the CPU
throughout, 139 ms of it collecting".

To capture one: launch with **`VTSEARCH_DIAGNOSE=1`** (one switch for every
bar — see [the env table](DEPLOYMENT.md#application); under the SLURM launcher it reaches the app through
the launcher's environment, so no launcher change is needed), label until a stall is felt,
then read the log around the `stall:` line.
`scripts/experiments/stall_3853/analyze_app_log.py` prints that window for
every stall in a log.

**Every run records the bars it is using**, as a `diagnostics config:` line at
startup. Without it a log cannot be read honestly after the fact: "no slow
requests" means *nothing was slow* and *the bar was a second* equally well. The
line goes out at WARNING, so a stock log level keeps it, whenever a bar is off
its shipped default: `VTSEARCH_DIAGNOSE=1`, or any threshold variable set by
hand. With every bar stock it drops to INFO, so a normal start stays quiet; a
WARNING-level log without the line was written at the shipped defaults of its
build. The analyzer prints that line first, and says so when a log lacks it.

For what the app cannot see — NFS latency on the mount every vote writes to,
and major faults on the app process — run
`scripts/experiments/stall_3853/sample_host.py` alongside the session; it
writes JSONL that lines up with the app log by timestamp, and summarises
itself with `--summarize`.

**A felt pause is often a sum, not an outlier**: a vote cycle whose chain of
requests, retrain and collection each cost less than any threshold can still
add up to the half-second a reviewer notices, which a per-request bar cannot
see by construction. At `VTSEARCH_LOG_LEVEL=INFO`
every request is logged as `request trace: …` with the same wall/CPU/GC
figures, and `analyze_app_log.py` adds them up into a **per-vote budget**:
each cycle's span from keypress to keypress, how much of it the server had a
request in flight (`busy`), and how much it did not (`gap` — client work,
browser queueing and think time). A 2 s cycle with 200 ms of `busy` was not
the server.

### Models fail to download

**Symptom**: Error during startup or dataset loading mentioning
`ConnectionError` or `OSError: We couldn't connect to`.

**Fix**: Ensure the machine has internet access to `huggingface.co`, or
pre-download models with `./scripts/install/download_models.sh` and set
`HF_HUB_OFFLINE=1` (see [Offline deployment](DEPLOYMENT.md#offline-deployment)).

### Out of memory

**Symptom**: Process killed or `torch.cuda.OutOfMemoryError`.

**Fix**: Load fewer media types simultaneously. The smart-preload pass
at startup warms every embedder referenced by the dataset and detector
registries; unregister datasets you no longer use so their embedders
aren't preloaded. For GPU, ensure adequate VRAM (4+ GB recommended).

### Docker build fails on pip install

**Symptom**: Network timeout downloading PyTorch wheels.

**Fix**: Retry the build (transient network issue), or use a local PyTorch
wheel mirror. For air-gapped environments, pre-download all wheels and
`COPY` them into the build context.

### Settings not persisting across container restarts

**Symptom**: Settings reset to defaults after `docker compose down && up`.

**Fix**: Ensure you're mounting a persistent volume at `/app/data`. Check
with `docker volume ls` and verify the `vtsearch-data` volume exists.

### Demo dataset download fails

**Symptom**: Error in web UI when selecting a demo dataset.

**Fix**: Check internet connectivity. For offline use, import data locally
via the folder or pickle importer instead.

### Video import crashes with `FATAL FIPS SELFTEST FAILURE`

**Symptom**: on a FIPS-enabled host, importing a video kills the process
outright — no traceback, just a core dump and:

```
crypto/fips/fips.c:154 OpenSSL internal error: FATAL FIPS SELFTEST FAILURE
```

**Cause**: the `opencv-python-headless` wheel vendors its own OpenSSL.
`cv2.abi3.so` needs `libavformat`, which needs the `libssl`/`libcrypto`
1.1.1 pair shipped inside `opencv_python_headless.libs`, so merely
importing `cv2` maps a second, non-FIPS OpenSSL into a process that
already holds the system's FIPS-validated one. The duplicate trips the
FIPS self-test and OpenSSL responds with `abort()`. Downgrading does not
help — every wheel from 4.9 through 5.0 vendors the same OpenSSL 1.1.1w.

**Fix**: none needed for video. All video decoding goes through ffmpeg
(`vtscore.media.video.decode`), which runs out-of-process and so never
loads OpenSSL into the interpreter. Confirm the host resolved to it:

```bash
python -c "from vtscore.media.video import decode; print(decode.backend())"
```

`ffmpeg` is the expected answer. If it prints `opencv`, this install has
no ffmpeg at all and video decoding will fall back to `cv2` and crash —
install one (`dnf install ffmpeg`, or `pip install imageio-ffmpeg` for
the bundled static binary).

**Still applies to image features.** The structural image embedder and
the YOLO-based image processors import `cv2` directly, so those remain
unusable on a FIPS host. To use them, replace the wheel with a build
that links the system OpenSSL instead of vendoring one — a distro
package (`dnf install python3-opencv`), or `opencv-python` built from
source. Everything else, including the whole video workflow, works with
the stock wheel installed.

### A feature you just pulled isn't in the UI ("out-of-date build")

**Symptom**: a toast reading *"This page is running an out-of-date build"*, or a
`⚠ bundle v …` chip beside the version in the Settings footer. Alternatively no
warning at all, but a control that the docs describe is missing, does nothing,
or behaves like an older release.

**Cause**: `static/` is a build artifact and is not tracked by git, so pulling
new code and restarting the server upgrades the *Python* side only. The browser
keeps loading whichever bundle was last built. The version shown in Settings is
the server's, so it looks current while the JavaScript is not.

**Fix**: rebuild the frontend and hard-reload the page (Ctrl/Cmd-Shift-R — a
normal reload can serve the old bundle from cache):
```bash
cd frontend && npm install && npm run build:prod
```

### "No module named" errors

**Symptom**: `ModuleNotFoundError` for a media type or importer package.

**Fix**: Reinstall to ensure all dependencies are present:
```bash
bash scripts/install.sh
```

### Browse maps render unlettered ("VTSBrowse signposts are disabled")

**Symptom**: the Browse canvas draws the map but never names any region, and
the server log carries a one-time

```
VTSBrowse signposts are disabled: the required 'toponymy' library is not importable.
```

**Cause**: `toponymy` is installed in a dedicated `--no-deps` step rather than
declared as an ordinary dependency (its `transformers<5.0.0` pin would
downgrade the transformers stack every embedder sits on). Any environment that
skipped that step has no labeler.

**Fix**: on a host install, re-run the installer, which performs the step (and
installs `apricot-select` alongside it):

```bash
bash scripts/install.sh
```

In a container, rebuild the image from a current checkout: every Dockerfile
under `docker/` carries the step. `pip install toponymy` inside a running
container is a workaround rather than a repair -- pass `--no-deps` if you try
it, or pip will downgrade transformers and break the embedders.

## GPU installs: driver, CUDA and cuML

### `install.sh` installs CPU torch on a machine that has a GPU

**Symptom**: On a GPU host (e.g. an AWS `g4dn` with a Tesla T4), `scripts/install.sh`
detects the card has no driver and offers to fix it:

```
NOTICE: An NVIDIA GPU is physically present, but no usable driver was found ...

What would you like to do?
  [i] Install the NVIDIA driver now (needs sudo; may require a reboot) -- recommended
  [c] Install CPU-only torch instead (no GPU acceleration)
  [s] Stop and fix it yourself
Choice [I/c/s]:
```

**Cause**: The script's CPU-vs-GPU decision asks `nvidia-smi` whether a GPU is
usable. On a fresh cloud GPU instance booted from a **base AMI** (anything but
the AWS Deep Learning AMI), the card is attached but the **NVIDIA kernel driver
isn't installed**, so `nvidia-smi` is absent and CUDA can't run. `pip` cannot
fix this — the driver is a system package, not a Python wheel.

**Fix**: Pick `[i]` (the default) and the installer installs the driver for you:
a distro-aware, best-effort `sudo` install (`ubuntu-drivers` / `apt` on
Debian-family, `cuda-drivers` via `dnf`/`yum` on RHEL-family), then it re-checks
`nvidia-smi` and proceeds straight into the GPU install if the GPU came online.
If the kernel module needs a **reboot** to load (common), it tells you to reboot
and re-run `bash scripts/install.sh`.

On the RHEL family (RHEL 8/9, Rocky, Alma, CentOS Stream) a bare cloud box
fails a plain `dnf install cuda-drivers` in several ways, and the installer
works through them in order:

- **`No match for argument: cuda-drivers`**: NVIDIA's CUDA repo is not
  enabled. The installer drops `cuda-<slug>.repo` (distro + major version,
  e.g. `rhel9`, plus the CPU arch) into `/etc/yum.repos.d`, enables **EPEL**
  best-effort (where `dkms` lives), and retries. A reboot does not fix this
  one: until the repo is enabled, nothing was installed.
- **`All matches were filtered out by modular filtering`**: the CUDA repo
  packages the driver as a **DNF module**. The installer falls back to
  `dnf module install nvidia-driver:latest-dkms` (proprietary; covers
  Turing/Ampere/Ada such as the g4dn's T4), then `nvidia-driver:open-dkms`
  (the open kernel module that Hopper and Blackwell require).
- **`nothing provides dkms >= 3.1.8 needed by kmod-nvidia-latest-dkms`**: an
  unregistered box finds no `epel-release` package, so `dkms` never installs.
  The installer bootstraps EPEL from its canonical URL
  (`epel-release-latest-$(rpm -E %rhel).noarch.rpm`); if `dkms` still can't
  be had, it installs the **precompiled, kABI-tracking** streams
  `nvidia-driver:latest` / `nvidia-driver:open` instead, which need no `dkms`.
- **Every dnf path dead-ends** (e.g. `missing groups or modules:
  nvidia-driver:open`): the installer falls back to **NVIDIA's
  self-contained `.run` installer**, the route [AWS
  documents](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html)
  for EC2. It needs only a C toolchain and kernel headers (installed
  best-effort first), and compiles the module against the running kernel. By
  default it fetches the latest driver from AWS's public, credential-free
  `ec2-linux-nvidia-drivers` S3 bucket; `VTSEARCH_NVIDIA_RUNFILE_URL` pins a
  version or points at a mirror. A reboot is usually required afterwards so
  the freshly built `nvidia` module loads.

You can also do it by hand:

```bash
# Ubuntu / Debian:
sudo apt-get update && sudo apt-get install -y nvidia-driver-535   # or newer

# RHEL 9 family (Rocky / Alma / CentOS Stream): enable the CUDA repo first.
# `dkms` lives in EPEL. On an unregistered RHEL box `dnf install epel-release`
# finds nothing, so bootstrap EPEL from its URL instead:
sudo dnf install -y \
  "https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm"
sudo dnf install -y dkms                                           # for the DKMS streams
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
# The driver is a DNF module on RHEL 8/9, so `dnf install cuda-drivers` is
# rejected with "filtered out by modular filtering" -- install the module:
sudo dnf module install -y nvidia-driver:latest-dkms              # builds via DKMS
# ...or, if you can't get dkms, the precompiled (no-DKMS) stream instead:
#   sudo dnf module install -y nvidia-driver:latest
# ...or, if EVERY dnf path dead-ends (bare unregistered RHEL), use NVIDIA's
# self-contained .run installer (needs only gcc/make + kernel-devel):
#   sudo dnf install -y "kernel-devel-$(uname -r)" kernel-headers gcc make
#   curl -fSL -o nvidia.run \
#     https://us.download.nvidia.com/tesla/<ver>/NVIDIA-Linux-x86_64-<ver>.run
#   sudo sh nvidia.run --silent --disable-nouveau

sudo reboot                                                        # if needed

nvidia-smi              # should list the GPU and a CUDA version
bash scripts/install.sh # now auto-detects the GPU
```

See also [AWS's GPU-driver guide](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html),
or use the **AWS Deep Learning AMI**, which ships the driver preinstalled.

The installer detects the physical card via its PCI vendor ID (so it knows the
GPU is there even without a driver). The other prompt choices: `[c]` installs
CPU-only torch (same as `bash scripts/install.sh cpu`); `[s]` stops. On a
**non-interactive** shell (CI, `curl … | bash`, Docker build) it can't prompt,
so it stops — unless you set `VTSEARCH_AUTO_DRIVER=1` (auto-install the driver)
or `VTSEARCH_ASSUME_CPU=1` (proceed CPU-only) to choose unattended.

**About the output.** On a bare cloud GPU box the cascade's early attempts
are *expected* to fail. Each attempt runs under a live `Installing …`
heartbeat with its output captured to a log, and ends in a green `✓` or a dim
*"not available here; trying another approach…"*; the captured output is shown
only if a step genuinely fails. The heartbeat also keeps long, silent steps
(the kernel-module compile, the smoke test's torch import) visibly alive. To
stream every command's raw output instead, re-run with
`VTSEARCH_VERBOSE=1 bash scripts/install.sh`.

### Making the GPU driver survive reboots and kernel updates

**Symptom**: The GPU worked yesterday, but after a stop/start (or overnight) you
have to re-run `scripts/install.sh` to get GPU support back. `nvidia-smi` is gone
or `torch.cuda.is_available()` is `False`, even though the CUDA `torch` wheel is
still installed.

**Cause**: This is almost never the Python side. The CUDA `torch`/`torchvision`/
`torchaudio` wheels live in your environment and **persist** across reboots — a
`git pull` or a frontend build never touches them. What resets is the **NVIDIA
kernel driver**, which is a *kernel module*, not a Python package, so a virtualenv
fundamentally cannot hold it. On a cloud GPU box the driver **files** survive a
stop/start on the EBS root volume, but if the OS pulled an **automatic kernel
upgrade** while it was running (Ubuntu `unattended-upgrades`, Amazon Linux/`dnf`
automatic updates), you boot into a *new kernel* and a **pre-built** `nvidia`
module no longer matches it — so `nvidia-smi` goes dark until the module is
rebuilt. Re-running the installer rebuilds it against the new kernel; that's why
it "comes back every time."

**Which half is resetting** — run these to confirm:

```bash
nvidia-smi                                                   # driver visible now?
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
#   2.x+cu126  True   -> both fine
#   2.x+cu126  False  -> CUDA torch OK, the kernel MODULE broke (kernel update)
#   2.x+cpu           -> the env got a CPU torch (env-recreation, a separate issue)
dkms status | grep -i nvidia || echo "NOT dkms-managed -> kernel updates WILL break it"
uname -r; ls /lib/modules/                                   # >1 kernel = an update happened
```

**What the installer does for you**: `scripts/install.sh` makes the driver
set-and-forget wherever it can — it registers the module with **DKMS** when `dkms`
is available (including on the self-contained `.run` fallback path, via `--dkms`),
so the module **auto-rebuilds for each new kernel**; it enables NVIDIA's
**persistence daemon** (`nvidia-persistenced`) so the driver initializes at every
boot; and every GPU install prints whether the result is DKMS-managed
(kernel-update-proof) or still kernel-pinned. On the Debian/Ubuntu path the
distro's `nvidia-driver-*` packages are already DKMS builds, so they're covered
too.

**If the driver is already up but reported NON-DKMS**, the installer *offers to
convert it in place* — it reinstalls the driver via NVIDIA's `.run` installer
with `--dkms` (bootstrapping EPEL + `dkms` first), then recommends a reboot so the
freshly built module loads. Answer `Y` at the prompt, or drive it unattended with
`VTSEARCH_AUTO_DKMS=1 bash scripts/install.sh` (`VTSEARCH_SKIP_DKMS=1` skips the
offer). Your GPU keeps working either way; declining just leaves it kernel-pinned.

**To make a NON-DKMS driver durable by hand** — e.g. on a bare, unregistered RHEL
box where `dkms` can't be reached and only the precompiled kABI-stream or a plain
`.run` module could be installed — use one of:

- **Bake a custom AMI** (or use the **AWS Deep Learning AMI**, which ships the
  driver preinstalled and maintained). A stop/start then starts from a known-good
  driver. This is the most robust option and the recommendation for a fleet.
- **Install `dkms` first, then re-run the installer** so the module becomes
  DKMS-managed. On RHEL, `dkms` lives in EPEL — and on a **hardened or unregistered**
  box `dnf install <epel-url.rpm>` fails the GPG check (`Public key ... is not
  installed`) unless you import EPEL's key first:
  ```bash
  sudo rpm --import "https://dl.fedoraproject.org/pub/epel/RPM-GPG-KEY-EPEL-$(rpm -E %rhel)"
  sudo dnf install -y \
    "https://dl.fedoraproject.org/pub/epel/epel-release-latest-$(rpm -E %rhel).noarch.rpm"
  sudo dnf install -y dkms
  bash scripts/install.sh
  ```
  If the driver is **already loaded** (nvidia-smi works), the `.run` reinstall aborts
  with `nvidia-modeset appears to be already loaded` — stop the driver and unload the
  modules first, then reinstall with `--dkms`:
  ```bash
  sudo systemctl stop nvidia-persistenced
  sudo rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia   # dependents before the base module
  lsmod | grep nvidia || echo "modules unloaded"           # confirm none remain
  bash scripts/install.sh                                   # or: sudo sh nvidia.run --silent --dkms
  sudo reboot
  ```
  (The installer's built-in DKMS conversion, above, does this stop/unload for you.)
- **Pin the kernel** so it stops changing under the module (blocks kernel security
  updates until you unpin — a tradeoff):
  ```bash
  # Ubuntu/Debian:
  sudo apt-mark hold "linux-image-$(uname -r)" "linux-headers-$(uname -r)"
  # RHEL/Amazon Linux:
  sudo dnf install -y 'dnf-command(versionlock)'
  sudo dnf versionlock add kernel kernel-core kernel-modules
  ```

### GPU install's cuML step (skipped on `cu118` and `cu128`)

**What you see**: on a `cu118` or `cu128` install the step
prints `(skipped: cuML needs RAPIDS >= 26.8 …)` and the smoke test at the end
reports `cuML not importable -> GPU UMAP/k-means will use the CPU fallback`.

**Cause**: pandas is held at `>=3` everywhere (`pyproject.toml`), and cudf,
which cuML depends on, is the first to take pandas 3 at RAPIDS **26.8**; an
older RAPIDS would downgrade pandas to 2.x mid-install. RAPIDS 26.x in turn
requires `nvidia-nvjitlink-cu12 >= 12.9`. The installer pairs it only with the
two torch tags where that was checked on real GPUs (#4390): `cu129`, whose
torch pins the whole 12.9 set, and `cu126`, whose torch pins the other CUDA
libraries at 12.6.x but leaves nvjitlink open, so pip lifts it to 12.9 with no
conflict. `cu118` (CUDA 11 libraries) and `cu128` are untested and skipped.

**What to do**: re-run as `bash scripts/install.sh cu129` if the GPU is Turing
or newer (`sm_75+`; every card since the RTX 20 series and the T4) and
`nvidia-smi` reports a driver CUDA version of 12.9 or later, or as
`bash scripts/install.sh cu126` if the driver reports 12.6 or later (this is
also the tag for a Volta card, which has no `cu129` torch). The auto-detect
already picks one of the two on such hosts, so an explicit older tag is
usually why you got here. RAPIDS 25 can still be assembled by hand on the
skipped tags (`VTS_CUML_CU12_SPEC='cuml-cu12<26'`), at the cost of pandas
dropping to 2.x in that venv.

**On `cu126`** the step installs torch 2.14.1+cu126 beside cuML 26.10 and
pandas 3. On the GRID that venv ran UMAP and k-means fits on a V100S and an
A100 with no CPU fallback (#4390). Leave cuDNN at the version torch pins:
cuDNN 9.20, the `cu129` set's, refuses GPUs below `sm_75`, and torch and
onnxruntime then fail on a V100.

**On `cu129`** the step runs, and torch and RAPIDS share one set of
`nvidia-*-cu12` wheels (torch pins them at 12.9.x, RAPIDS 26.8 is built on
12.9), so pip should install it without a "dependency conflicts" report. If
one appears, the venv holds a torch from another tag (one built before the
`cu129` move, or a `VTS_CUML_CU12_SPEC` override); the installer captures the
report to a log and the **GPU smoke test** at the end (a tiny torch CUDA matmul
plus a cuML import) is what decides whether the stack works. Re-run with
`VTSEARCH_VERBOSE=1` to stream the raw output.

### cuML crashes compiling a kernel (`cuda_fp8.hpp` / nvrtc errors)

**Symptom**: A VTSBrowse projection or coverage-atlas build dies with an
nvrtc compile error like:

```
.../nvidia/cu13/include/cuda_fp8.hpp(...): error: this declaration has no
storage class or type specifier  __NV_SILENCE_DEPRECATION_BEGIN
... N errors detected in the compilation of ".../<hash>.cubin.cu".
```

**Cause**: a **RAPIDS built on a newer CUDA than the libraries your torch
pinned**. cuML compiles its cuVS/raft kernels with nvrtc lazily, on the first
UMAP/k-means `fit`. RAPIDS **26.x** requires `nvidia-nvjitlink-cu12 >= 12.9`
and is built on CUDA 12.9; a torch from an older tag pins the other CUDA
libraries lower (`cu126` = 12.6.x, `cu128` = 12.8.x), so pip reports the
conflict:

```
cuml-cu12 26.8.0 requires nvidia-nvjitlink-cu12<13,>=12.9, but you have
nvidia-nvjitlink-cu12 12.4.127 which is incompatible.
```

...and, if mismatched libraries land anyway, makes cupy's nvrtc compile the
wrong-version fp8/fp6/fp4 headers (the `__NV_SILENCE_DEPRECATION_BEGIN` macro
the resident nvrtc doesn't define), producing the `cuda_fp8.hpp` crash above.

`docker/Dockerfile.gpu` installs RAPIDS **only alongside the `cu129` torch**,
whose pins are the 12.9.x libraries RAPIDS is built on, and `scripts/install.sh`
does the same on `cu129` and `cu126` (where pip lifts only nvjitlink, and the
fits were checked on a V100S and an A100, #4390), so fresh installs are
unaffected. A venv built before that move (torch `cu124` plus `cuml-cu12<26`
and pandas 2) is consistent with itself and keeps working; the crash comes
from mixing eras in one venv, e.g. by upgrading cuML there by hand.

VTSearch also **degrades to the CPU UMAP/k-means path** whenever a cuML fit
fails for any reason (logging a one-time warning) instead of crashing, so the
run still completes — just slower, without GPU acceleration. To restore the
GPU path, rebuild the venv on one CUDA minor:

```bash
# 1. Inspect the installed CUDA stack (look for a torch tag other than cu129
#    next to cuml-cu12 26.x, and any stray *-cu13 wheels from an out-of-band install).
pip list | grep -iE '^torch |cu13|cupy|cuml|cuvs|libraft|pylibraft|nvidia-nvjitlink'

# 2. If step 1 showed stray CUDA-13 wheels, remove them first:
pip uninstall -y $(pip list --format=freeze | grep -iE '(-cu13|cupy-cuda13x)' | cut -d= -f1)

# 3. Reinstall on the cu129 tag (Turing or newer, driver CUDA >= 12.9), which
#    pins torch and RAPIDS to the same 12.9.x libraries, or on cu126 (Volta):
bash scripts/install.sh cu129
```

If you don't need GPU UMAP/k-means at all, set `VTSEARCH_SKIP_CUML=1`
(skips the cuML install) and the CPU `umap-learn`/`scikit-learn` paths run
without any toolchain risk.
