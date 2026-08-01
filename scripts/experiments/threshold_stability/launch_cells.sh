#!/usr/bin/env bash
# Threshold-stability study (#2790) SLURM array: one CPU task per (class, seed),
# all arms inside, then the Stage-A replay. No GPU — the SigLIP 2 / whole
# embeddings are read from the reused #2790 cache (THRSTAB_CACHE_DIR), so a fully
# cached run never touches a model. If the cache is incomplete the sweep will try
# to embed (needs a GPU); pre-populate the cache first.
#
# Usage: bash launch_cells.sh   (after exporting THRSTAB_* + VTS_REPO)
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-evalfw}"
HERE="$WT/scripts/experiments/threshold_stability"
export THRSTAB_EXP="${THRSTAB_EXP:-/exp/$USER/threshold-stability}"
export THRSTAB_RESULTS="${THRSTAB_RESULTS:-$THRSTAB_EXP/results}"
export THRSTAB_CACHE_DIR="${THRSTAB_CACHE_DIR:-$THRSTAB_EXP/cache}"
LOGS="$THRSTAB_EXP/logs"
mkdir -p "$LOGS" "$THRSTAB_RESULTS/cells"

MEM="${THRSTAB_MEM:-16G}"
CPUS="${THRSTAB_CPUS:-4}"
TIME="${THRSTAB_TIME:-4:00:00}"
CONC="${THRSTAB_CONC:-16}"
PART="${THRSTAB_PARTITION:-cpu}"

ENVX="export THRSTAB_EXP=$THRSTAB_EXP THRSTAB_RESULTS=$THRSTAB_RESULTS THRSTAB_CACHE_DIR=$THRSTAB_CACHE_DIR VTS_REPO=$WT"

N=$(cd "$HERE" && eval "$ENVX"; python run_cells.py --print-cells 2>/dev/null | tail -1)
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
  echo "ERROR: could not determine cell count (got '$N')" >&2
  exit 1
fi
echo "cells to run: $N (array 0-$((N-1))%$CONC, partition=$PART)"

B=$(sbatch --parsable --job-name=thrstab-cells --array=0-$((N-1))%$CONC \
  --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition="$PART" \
  --export=ALL --output="$LOGS/cells-%A_%a.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py")
echo "cells array: $B"

A=$(sbatch --parsable --dependency=afterany:$B --job-name=thrstab-analyze --mem=8G \
  --cpus-per-task=2 --time=0:30:00 --partition="$PART" \
  --export=ALL --output="$LOGS/analyze-%j.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python analyze.py")
echo "analyze: $A"
echo "Report -> $THRSTAB_RESULTS/REPORT.md   Logs -> $LOGS"
