#!/usr/bin/env bash
# Submit the calibration cells array + analyze step.  Run after prepare has
# written prepare_info.json (launch_all.sh submits this as an afterok dependency,
# or run it by hand once prepare is done).
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-calib}"
HERE="$WT/scripts/experiments/calibration"
export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"
LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS"

GRES="${CALIB_GRES:-gpu:v100:1}"
MEM="${CALIB_MEM:-48G}"
CPUS="${CALIB_CPUS:-6}"
TIME="${CALIB_TIME:-4:00:00}"
CONC="${CALIB_CONC:-8}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=${VTSEARCH_DATA_DIR:-} VTSEARCH_MODELS_DIR=${VTSEARCH_MODELS_DIR:-} HF_HOME=${HF_HOME:-}"

N=$(cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1; eval "$ENVX"; python run_cells.py --print-cells 2>/dev/null | tail -1)
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
  echo "ERROR: could not determine cell count (got '$N'); is prepare done?" >&2
  exit 1
fi
echo "cells to run: $N (array 0-$((N-1))%$CONC, $GRES)"

B=$(sbatch --parsable --job-name=cal-cells --array=0-$((N-1))%$CONC \
  --gres="$GRES" --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition=gpu \
  --export=ALL --output="$LOGS/cells-%A_%a.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py")
echo "cells array: $B"

A=$(sbatch --parsable --dependency=afterany:$B --job-name=cal-analyze --mem=16G \
  --cpus-per-task=4 --time=0:40:00 --partition=cpu \
  --export=ALL --output="$LOGS/analyze-%j.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python analyze.py")
echo "analyze: $A"
echo "$B" > "$LOGS/.cells_jobid"
echo "Report -> $CALIB_RESULTS/REPORT.md"
