# Running VTSearch on a SLURM GPU cluster

VTSearch needs a GPU to be comfortable (embedding + training), and on a shared
SLURM cluster (like the JHU HLTCOE "Grid") you don't run anything heavy on the
login nodes. Instead you ask SLURM for a GPU compute node, run the app there,
and forward its port back to your local machine for the browser UI. The
launcher and the tunnel make that day-to-day loop a two-command affair:

| Script | Runs on | What it does |
|--------|---------|--------------|
| [`vtsearch-slurm.sh`](vtsearch-slurm.sh) | the **cluster** (a login node) | Allocates a GPU compute node with `srun`, activates the venv, and runs `app.py` on it. Prints the node it landed on. Holds the allocation until you quit. |
| [`vtsearch-tunnel.sh`](vtsearch-tunnel.sh) | your **local machine** | Finds your running VTSearch job, SSH-forwards your local port to that compute node (auto-discovering the node + per-user port), and drops you into the project dir. `VTS_BIND=<addr>` serves the port to another device; `--no-shell` holds the tunnel open without a login shell. |
| [`pick_gpu.py`](pick_gpu.py) | the **cluster** | Picks the GPU type to request (via `scontrol`): the fastest type in `VTS_GPU_TYPES` with a free GPU now, else the type with the most free, else the largest pool, else `VTS_GPU_FALLBACK`. The launcher calls it, and `VTS_GPU` overrides it. `--explain` prints the per-type free/total table. |
| [`job_records.py`](job_records.py) | the **cluster** | Refuses an array that would take the cluster past 50% of Slurm's `MaxJobCount`, or you past 25%. Every array task is a job record, pending ones included. Called by `scripts/experiments/calibration/launch_cells.sh` before it submits, and by preflight's `--array-tasks`. |
| [`vtsearch-tunnel.service`](vtsearch-tunnel.service) | your **local machine** | systemd *user* unit that keeps the tunnel up with no terminal attached, for an always-on box that holds the VPN on other devices' behalf. |

The launcher and tunnel are parameterized by environment variables (no hard-coded usernames,
hostnames, or paths) so they should work on most SLURM clusters with a shared
filesystem. See the comment block at the top of each script for the knobs, and
[Tuning the allocation](#tuning-the-allocation) below for the ones you are most
likely to set.

First time on a cluster? Work through the two one-time setups below, on the
[cluster](#one-time-setup-on-the-cluster) and on [your local
machine](#one-time-setup-on-your-local-machine). After that, the [daily
workflow](#daily-workflow) is one command on each side.

> **Why a per-user port?** GPU nodes usually hold several GPUs, so SLURM can
> pack multiple users' single-GPU jobs onto one physical node — where a single
> shared `:5000` would collide (the second app can't bind it, and a naive
> tunnel would forward you into someone else's session). So the scripts derive
> a port from your UID (`10000 + UID % 20000`); both compute the same value, so
> the tunnel finds your app with no extra coordination. `app.py` honors this
> via `VTSEARCH_PORT` (or `--port`). Override with `VTS_PORT` on *both* scripts
> if you ever need to.

## One-time setup on the cluster

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
   12.6 takes `cu126`; a Turing-or-newer card on a driver at CUDA 12.9 or
   later takes `cu129`, the one tag that also gets cuML; a V100 takes `cu126`
   on any driver from CUDA 12.6 up):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   bash scripts/install.sh cu129         # or cu118 / cu126 / cu128 to match your node's GPU and driver
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
   > bash scripts/install.sh cu126
   > ```
   >
   > A venv built this way is **not** self-contained — its `python` needs the
   > module's `libpython` at runtime — so the launcher must load the module
   > before activating the venv. Set `VTS_MODULE` (see
   > [Tuning](#tuning-the-allocation)) so `vtsearch` does this for you:
   > `VTS_MODULE="python/3.12.3" vtsearch` (or `export` it in `~/.bashrc`).

3. **Build the frontend** (needs Node.js 20.19+, 22.12+ or 24+; see [Building the
   frontend](../../docs/SETUP.md#building-the-frontend)):

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

## One-time setup on your local machine

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

## Daily workflow

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
   press **Enter** to restart the app. Type **i** then Enter first to rerun
   `scripts/install.sh` when dependencies changed; **q** then Enter releases
   the node and quits.

> Interactive SLURM jobs don't persist — you re-allocate each session. Only run
> **one** `app.py` per allocation; VTSearch keeps all model/dataset state in one
> process, and a second copy would double the memory and can OOM the job.

## Tuning the allocation

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

### Relaying the app to another device

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
[`vtsearch-tunnel.service`](vtsearch-tunnel.service)
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

### Which GPU type gets requested

Many clusters (HLTCOE's included) reject an untyped `--gres=gpu:1`, so a type
has to be named — and any single name is a pin that goes stale. Pinning the slow
type costs a multiple on every embed (a V100 embeds `siglip2_l` **2.3×** slower
than an L40S on identical fp32 code); pinning a scarce fast type has meant
multi-day queue waits. So the launcher doesn't pin: it runs
[`pick_gpu.py`](pick_gpu.py), which reads
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

## Sharing demo datasets on a cluster

Clusters usually have a communal large-dataset area on a big shared volume.
Rather than every user downloading the multi-GB demo datasets into their own
data dir (often on a small per-user quota), keep one shared cache there and
symlink it into each data dir with
[`scripts/data/link-demo-cache.sh`](../data/link-demo-cache.sh) — see
[`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md#sharing-demo-downloads-between-data-dirs-multi-user-servers).
Demo sources may also already exist elsewhere on the cluster (other groups'
dataset folders); anything matching the extraction layout can be copied
straight into the cache.

## Running the test suite on a cluster cpu node

[`suite.sbatch`](suite.sbatch) runs `./run-tests.sh` for one branch on a cpu
node. It takes the worktree to test and the ref as **arguments**, checks the ref
out (detached) in that worktree, refuses to run when the local branch and
`origin/<ref>` disagree (behind, ahead or diverged — it names which and exits 2
rather than guessing), and puts the worktree back on its original checkout when
the job ends.

Submit the copy in your **`dev` checkout**, never the one inside the worktree
under test, so a branch cannot edit the gate that judges it:

```bash
sbatch --job-name=suite-<n> --output=<logdir>/tests-%j.out \
    /exp/$USER/projects/VTSearch/scripts/slurm/suite.sbatch <tests-worktree> <branch>
```

`sbatch` copies the script into slurmd's spool at submit time, so the running
job is not affected by the checkout it performs, or by a later `git pull` of
the `dev` checkout. The job then fetches and, if the submitted copy is not
byte-for-byte `origin/dev`'s, runs `origin/dev`'s instead (#5484: the `dev`
checkout once sat 791 commits behind, and its copy refused every current
worktree). The log's `=== suite.sbatch` line says which copy ran. Give the suite a worktree of its own (`git worktree add
--detach`): it moves that worktree's HEAD for the length of the job. Read the
log's `=== HEAD` and `=== ref` lines to see which commit actually ran.
The job runs on a fresh `VTSEARCH_DATA_DIR` under `$TMPDIR` and removes it at
the end, so the data dir your shell exports is never touched. On the HLTCOE
GRID, `~/.bashrc` names the live app's data there (#4706). The `=== data` line
names both dirs. When `gh` is on `PATH`, the job also posts its verdict as an
informational `suite-grid` commit status on the tested SHA (see
[`docs/branch-protection.md`](../../docs/branch-protection.md#suite-grid-is-informational-not-required));
`SUITE_POST_STATUS=0` turns that off.
`tests_lib/meta/test_suite_sbatch.py` runs the real script against a throwaway
origin to pin the guard's behaviour.
