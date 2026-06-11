# Running VTSearch on a SLURM GPU cluster

VTSearch needs a GPU to be comfortable (embedding + training), and on a shared
SLURM cluster you don't run anything heavy on the login nodes. These two helper
scripts make the day-to-day loop a two-command affair:

| Script | Runs on | What it does |
|--------|---------|--------------|
| [`vtsearch-slurm.sh`](vtsearch-slurm.sh) | the **cluster** (a login node) | Allocates a GPU compute node with `srun`, activates the venv, and runs `app.py` on it. Prints the node it landed on. Holds the allocation until you quit. |
| [`vtsearch-tunnel.sh`](vtsearch-tunnel.sh) | your **local machine** | Finds your running VTSearch job, SSH-forwards your local port to that compute node (auto-discovering the node + per-user port), and drops you into the project dir. |

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

On your local machine (after adding a `cluster` host to `~/.ssh/config`):

```bash
cp scripts/slurm/vtsearch-tunnel.sh ~/.local/bin/vtsearch-tunnel && chmod +x ~/.local/bin/vtsearch-tunnel
vtsearch-tunnel   # forwards your local port to the GPU node; prints the URL to browse
```
