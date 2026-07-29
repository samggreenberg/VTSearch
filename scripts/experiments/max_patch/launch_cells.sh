#!/usr/bin/env bash
# Launch the Max-Patch cells array + summarize on /exp (shared), reading the
# pickles prepare wrote.  One SLURM-array task per (dataset, embedder, category,
# seed) cell; all styles for a cell run inside it.  v100 is used because the
# cluster has many idle v100s and the per-cell GPU work (tiny MLP retrains +
# a <=2 GB fp16 patch score-matrix) fits 16 GB comfortably.
#
# Usage: bash launch_cells.sh
set -uo pipefail
WT="${VTS_REPO:-/exp/$USER/projects/vts-maxpatch}"
HERE=$WT/scripts/experiments/max_patch
export MAXPATCH_EXP="${MAXPATCH_EXP:-/exp/$USER/max-patch}"
LOGS="$MAXPATCH_EXP/logs"
mkdir -p "$LOGS"
export MAXPATCH_EMBEDDERS="${MAXPATCH_EMBEDDERS:-dinov3_patch,siglip}"
export MAXPATCH_N_CATEGORIES="${MAXPATCH_N_CATEGORIES:-6}"
export MAXPATCH_N_SEEDS="${MAXPATCH_N_SEEDS:-4}"
export MAXPATCH_MAX_STEPS="${MAXPATCH_MAX_STEPS:-150}"

GRES="${MAXPATCH_GRES:-gpu:v100:1}"
MEM="${MAXPATCH_MEM:-48G}"
CPUS="${MAXPATCH_CPUS:-6}"
TIME="${MAXPATCH_TIME:-4:00:00}"
CONC="${MAXPATCH_CONC:-8}"

# Count cells from the prepared datasets.
N=$(cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1; \
    MAXPATCH_EXP=$MAXPATCH_EXP python run_cells.py --print-cells 2>/dev/null | tail -1)
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
  echo "ERROR: could not determine cell count (got '$N'); is prepare done?" >&2
  exit 1
fi
echo "cells to run: $N (array 0-$((N-1))%$CONC, $GRES)"

WRAP="source $WT/gridenv.sh && export MAXPATCH_EXP=$MAXPATCH_EXP MAXPATCH_EMBEDDERS=$MAXPATCH_EMBEDDERS MAXPATCH_N_CATEGORIES=$MAXPATCH_N_CATEGORIES MAXPATCH_N_SEEDS=$MAXPATCH_N_SEEDS MAXPATCH_MAX_STEPS=$MAXPATCH_MAX_STEPS && cd $HERE"

B=$(sbatch --parsable --job-name=mp-cells --array=0-$((N-1))%$CONC \
  --gres="$GRES" --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition=gpu \
  --export=ALL --output="$LOGS/cells-%A_%a.out" \
  --wrap="$WRAP && python run_cells.py")
echo "cells array: $B"

S=$(sbatch --parsable --dependency=afterany:$B --job-name=mp-sum --mem=16G \
  --cpus-per-task=4 --time=0:30:00 --partition=gpu --gres=gpu:v100:1 \
  --export=ALL --output="$LOGS/summarize-%j.out" \
  --wrap="source $WT/gridenv.sh && export MAXPATCH_EXP=$MAXPATCH_EXP && cd $HERE && python analyze.py")
echo "summarize: $S"
echo "$B" > "$LOGS/.cells_jobid"
echo "Report -> $MAXPATCH_EXP/results/REPORT.md"
