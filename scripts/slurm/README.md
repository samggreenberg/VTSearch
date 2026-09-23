# Running VTSearch on a SLURM GPU cluster

VTSearch needs a GPU to be comfortable (embedding + training), and on a shared
SLURM cluster you don't run anything heavy on the login nodes. These two helper
scripts make the day-to-day loop a two-command affair:

| Script | Runs on | What it does |
|--------|---------|--------------|
| [`vtsearch-slurm.sh`](vtsearch-slurm.sh) | the **cluster** (a login node) | Allocates a GPU compute node with `srun`, activates the venv, and runs `app.py` on it. Prints the node it landed on. Holds the allocation until you quit. |
| [`vtsearch-tunnel.sh`](vtsearch-tunnel.sh) | your **local machine** | Finds your running VTSearch job, SSH-forwards your local port to that compute node (auto-discovering the node + per-user port), and drops you into the project dir. `VTS_BIND=<addr>` serves the port to another device; `--no-shell` holds the tunnel open without a login shell. |
| [`vtsearch-tunnel.service`](vtsearch-tunnel.service) | your **local machine** | systemd *user* unit that keeps the tunnel up with no terminal attached, for an always-on box that holds the VPN on other devices' behalf. |

Both are parameterized by environment variables (no hard-coded usernames,
hostnames, or paths) so they should work on most SLURM clusters with a shared
filesystem. See the comment block at the top of each script for the knobs, and
[`docs/SETUP.md`](../../docs/SETUP.md#running-on-a-slurm-gpu-cluster) for a full
first-time walkthrough.

## Quick start

On the cluster (after cloning VTSearch and setting up the venv + frontend):

```bash
cp scripts/slurm/vtsearch-slurm.sh ~/.local/bin/vtsearch && chmod +x ~/.local/bin/vtsearch
vtsearch          # allocates a GPU node and starts the app; leave it running
```

> On clusters whose Python comes from environment modules (e.g. the HLTCOE
> Grid), set `VTS_MODULE` so the launcher loads it before activating the venv,
> e.g. `VTS_MODULE="python/3.12.3" vtsearch`. See
> [`docs/SETUP.md`](../../docs/SETUP.md#running-on-a-slurm-gpu-cluster) for the
> full module-based setup. Pick the CUDA wheel to match your GPU — older cards
> like the V100 need `cu124`, *not* the newest `cu128` (which drops Volta).

On your local machine (after adding a `cluster` host to `~/.ssh/config`):

```bash
cp scripts/slurm/vtsearch-tunnel.sh ~/.local/bin/vtsearch-tunnel && chmod +x ~/.local/bin/vtsearch-tunnel
vtsearch-tunnel   # forwards your local port to the GPU node; prints the URL to browse
```

## Sharing demo datasets on a cluster

Clusters usually have a communal large-dataset area on a big shared volume.
Rather than every user downloading the multi-GB demo datasets into their own
data dir (often on a small per-user quota), keep one shared cache there and
symlink it into each data dir with
[`scripts/link-demo-cache.sh`](../link-demo-cache.sh) — see
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
the `dev` checkout. Give the suite a worktree of its own (`git worktree add
--detach`): it moves that worktree's HEAD for the length of the job. Read the
log's `=== HEAD` and `=== ref` lines to see which commit actually ran.
`tests_lib/meta/test_suite_sbatch.py` runs the real script against a throwaway
origin to pin the guard's behaviour.
