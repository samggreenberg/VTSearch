#!/bin/bash
# Allocate a GPU node on a SLURM cluster and run VTSearch on it.
#
# Run this in a terminal ON THE CLUSTER (a login node) and LEAVE IT RUNNING --
# it holds the SLURM allocation and the app. When the allocation lands, it
# prints the compute node it got and a hint for the tunnel script you run on
# your local machine (see scripts/slurm/vtsearch-tunnel.sh).
#
# Ctrl+C stops the APP but KEEPS the node, then offers to restart it (handy
# after a git pull / code edit). Type q at that prompt to release the node and
# quit.
#
# Install: copy this somewhere on your PATH on the cluster and make it
# executable, e.g.
#     cp scripts/slurm/vtsearch-slurm.sh ~/.local/bin/vtsearch && chmod +x ~/.local/bin/vtsearch
#
# Override any default with an env var, e.g.:  VTS_MEM=64G VTS_GPU=a100 vtsearch
set -u

# --- Allocation knobs (override via env vars) -------------------------------
# Where the VTSearch checkout lives on the cluster's shared filesystem. Default
# follows the common "/exp/$USER" convention; set VTS_DIR if yours differs.
DIR=${VTS_DIR:-/exp/$USER/projects/VTSearch}
# Path (relative to $DIR, or absolute) to the Python virtualenv to activate.
VENV=${VTS_VENV:-.venv}
PART=${VTS_PART:-gpu}       # SLURM partition
GPU=${VTS_GPU:-l40s}        # GPU type for --gres=gpu:<type>:1
CPUS=${VTS_CPUS:-8}         # CPU cores
MEM=${VTS_MEM:-48G}         # memory (one app.py needs headroom for two model loads)
TIME=${VTS_TIME:-8:00:00}   # walltime
# Per-user port: GPU nodes hold several GPUs, so SLURM can pack multiple users'
# jobs onto one physical node where a single shared :5000 would collide. Derive
# the port from your UID so it's stable across sessions and the tunnel script
# computes the same value. Range 10000-29999 stays below the OS ephemeral range.
PORT=${VTS_PORT:-$((10000 + $(id -u) % 20000))}

echo ">>> Requesting 1x $GPU, $CPUS cores, $MEM, ${TIME} walltime on partition '$PART'..."
echo ">>> (this may queue; once it lands, VTSearch starts and prints its node + tunnel hint)"

# Export the resolved dir/venv/port so they survive into the srun child shell.
# app.py reads VTSEARCH_PORT for the dev server's bind port.
export VTS_DIR="$DIR" VTS_VENV="$VENV" VTSEARCH_PORT="$PORT"

exec srun --job-name=vtsearch --pty \
    -p "$PART" --gres=gpu:${GPU}:1 -c "$CPUS" --mem "$MEM" -t "$TIME" \
    bash -lc '
        # A no-op INT handler keeps THIS wrapper alive on Ctrl+C while still
        # letting child processes (python) take the default SIGINT and die.
        trap ":" INT
        cd "$VTS_DIR" || { echo "project dir missing: $VTS_DIR (set VTS_DIR)"; exit 1; }
        source "$VTS_VENV/bin/activate" || { echo "venv missing: $VTS_VENV (set VTS_VENV)"; exit 1; }
        node=$(hostname)
        echo
        echo "========================================================="
        echo "  VTSearch node: $node   port: $VTSEARCH_PORT   (branch: $(git branch --show-current 2>/dev/null))"
        echo "  On your LOCAL MACHINE, in a new terminal, run:"
        echo "      vtsearch-tunnel        (auto-discovers this node + port)"
        echo "========================================================="
        while true; do
            echo
            echo ">>> Starting app.py  (Ctrl+C stops it but keeps the node)..."
            echo
            # Embedders run on CPU; give torch all the cores SLURM gave us
            # (default is 1 thread, which is painfully slow for SigLIP etc.).
            VTSEARCH_TORCH_THREADS=${SLURM_CPUS_ON_NODE:-8} python app.py
            echo
            echo ">>> app.py stopped. GPU node ($node) is still yours."
            printf ">>> [Enter] = restart (picks up code changes)   |   q + [Enter] = quit & release node: "
            read -r reply || break
            [ "$reply" = "q" ] && break
        done
        echo ">>> Releasing the allocation. Bye."
    '
