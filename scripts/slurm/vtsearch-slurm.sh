#!/bin/bash
# Allocate a GPU node on a SLURM cluster and run VTSearch on it.
#
# Run this in a terminal ON THE CLUSTER (a login node) and LEAVE IT RUNNING --
# it holds the SLURM allocation and the app. When the allocation lands, it
# prints the compute node it got and a hint for the tunnel script you run on
# your local machine (see scripts/slurm/vtsearch-tunnel.sh).
#
# Ctrl+C stops the APP but KEEPS the node, then offers to restart it (handy
# after a git pull / code edit). At that prompt, type i to reinstall deps
# (runs scripts/install.sh in the venv -- handy after a dependency change,
# without re-sshing and re-activating by hand), or q to release the node and
# quit.
#
# Install: copy this somewhere on your PATH on the cluster and make it
# executable, e.g.
#     cp scripts/slurm/vtsearch-slurm.sh ~/.local/bin/vtsearch && chmod +x ~/.local/bin/vtsearch
#
# The GPU type is auto-picked from what is free (see pick_gpu.py, read out of
# the checkout at $VTS_DIR when this script has been copied away from it).
#
# Override any default with an env var, e.g.:  VTS_MEM=64G VTS_GPU=a100 vtsearch
set -u

# --- Allocation knobs (override via env vars) -------------------------------
# Where the VTSearch checkout lives on the cluster's shared filesystem. Default
# follows the common "/exp/$USER" convention; set VTS_DIR if yours differs.
DIR=${VTS_DIR:-/exp/$USER/projects/VTSearch}
# Path (relative to $DIR, or absolute) to the Python virtualenv to activate.
VENV=${VTS_VENV:-.venv}
# Optional environment module(s) to `module load` before activating the venv.
# Needed on clusters whose Python comes from `module load python/X.Y` -- a venv
# built from a module interpreter is NOT self-contained (its python needs the
# module's libpython/LD_LIBRARY_PATH at runtime), so the module must be loaded
# in the job shell too. Space-separated; empty = don't touch modules.
# e.g.  VTS_MODULE="python/3.12.3" vtsearch
MODULE=${VTS_MODULE:-}
PART=${VTS_PART:-gpu}       # SLURM partition
CPUS=${VTS_CPUS:-8}         # CPU cores
MEM=${VTS_MEM:-48G}         # memory (one app.py needs headroom for two model loads)
TIME=${VTS_TIME:-8:00:00}   # walltime
# Per-user port: GPU nodes hold several GPUs, so SLURM can pack multiple users'
# jobs onto one physical node where a single shared :5000 would collide. Derive
# the port from your UID so it's stable across sessions and the tunnel script
# computes the same value. Range 10000-29999 stays below the OS ephemeral range.
PORT=${VTS_PORT:-$((10000 + $(id -u) % 20000))}

# --- GPU type ---------------------------------------------------------------
# Ask the scheduler which type to request instead of pinning one: a pin is wrong
# in one direction or the other. Pinning the slow type wastes 2.3x on every
# embed; pinning a scarce fast type once meant ~5-day queue waits (issue #3144).
# VTS_GPU still wins outright -- pick_gpu.py returns it without querying.
#
# This script is meant to be COPIED to ~/.local/bin/vtsearch, so it cannot
# assume the picker sits beside it. Look next to the script first (running it
# from the checkout), then inside $DIR, then give up quietly and pin l40s.
PICK_GPU=""
for candidate in "$(dirname "$0")/pick_gpu.py" "$DIR/scripts/slurm/pick_gpu.py"; do
    [ -f "$candidate" ] && { PICK_GPU="$candidate"; break; }
done
GPU=""
if [ -n "$PICK_GPU" ] && command -v python3 >/dev/null 2>&1; then
    GPU=$(VTS_PART="$PART" python3 "$PICK_GPU")
fi
GPU=${GPU:-${VTS_GPU:-l40s}}

echo ">>> Requesting 1x $GPU, $CPUS cores, $MEM, ${TIME} walltime on partition '$PART'..."
echo ">>> (this may queue; once it lands, VTSearch starts and prints its node + tunnel hint)"

# Export the resolved dir/venv/port so they survive into the srun child shell.
# app.py reads VTSEARCH_PORT for the dev server's bind port.
export VTS_DIR="$DIR" VTS_VENV="$VENV" VTS_MODULE="$MODULE" VTSEARCH_PORT="$PORT"

# The per-node script below is too long to pass as an inline `bash -lc '...'`
# argument: this cluster's srun caps that single argument at 2000 chars and
# rejects anything longer ("script argument list exceeds 2000 characters").
# So write the body to a temp file on a SHARED filesystem -- the compute node
# has to read it, so node-local /tmp won't do -- and have srun run that file.
# This keeps the srun argv tiny no matter how long the body grows.
RUNNER_DIR="/exp/$USER"; [ -d "$RUNNER_DIR" ] || RUNNER_DIR="$HOME"
RUNNER=$(mktemp "$RUNNER_DIR/.vtsearch-run.XXXXXX") \
    || { echo "could not create runner temp file in $RUNNER_DIR"; exit 1; }
# Clean up the temp file on exit (normal quit, Ctrl+C, or kill).
trap 'rm -f "$RUNNER"' EXIT INT TERM
# Quoted heredoc: $VTS_DIR etc. stay literal here and expand on the compute
# node, exactly as they did inside the old single-quoted `bash -lc '...'`.
cat > "$RUNNER" <<'REMOTE'
        # A no-op INT handler keeps THIS wrapper alive on Ctrl+C while still
        # letting child processes (python) take the default SIGINT and die.
        trap ":" INT
        cd "$VTS_DIR" || { echo "project dir missing: $VTS_DIR (set VTS_DIR)"; exit 1; }
        # Load environment modules first if requested: a venv built from a
        # module-provided python needs that module loaded at runtime too.
        [ -n "$VTS_MODULE" ] && { module load $VTS_MODULE || { echo "module load failed: $VTS_MODULE (set VTS_MODULE)"; exit 1; }; }
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
            # Inner prompt loop so "i" (reinstall) can re-prompt instead of
            # auto-restarting -- you get to read the install output and then
            # decide to restart or quit. Enter falls through to restart app.py.
            while true; do
                printf ">>> [Enter] = restart (picks up code changes)   |   i + [Enter] = reinstall deps   |   q + [Enter] = quit & release node: "
                read -r reply || break 2          # EOF (e.g. Ctrl+D): release node
                case "$reply" in
                    q) break 2 ;;                 # quit both loops -> release node
                    i)
                        echo
                        echo ">>> Reinstalling deps via scripts/install.sh (auto-detects CPU/GPU)..."
                        echo ">>> (runs in the active venv: $VTS_VENV)"
                        echo
                        bash scripts/install.sh \
                            || echo ">>> install.sh FAILED -- see output above; node kept, deps unchanged."
                        echo
                        ;;                        # loop back to the prompt
                    *) break ;;                   # Enter/anything else -> restart app.py
                esac
            done
        done
        echo ">>> Releasing the allocation. Bye."
REMOTE

srun --job-name=vtsearch --pty \
    -p "$PART" --gres=gpu:${GPU}:1 -c "$CPUS" --mem "$MEM" -t "$TIME" \
    bash -l "$RUNNER"
