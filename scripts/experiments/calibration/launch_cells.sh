#!/usr/bin/env bash
# Submit the calibration cells array + analyze step.  Run after prepare has
# written prepare_info.json (launch_all.sh submits this as an afterok dependency,
# or run it by hand once prepare is done).
set -uo pipefail

# This script's own directory, for the record check below: VTS_REPO can name an
# older frozen worktree that predates the helper.
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# Partition + optional GPU request.  Cells that train the linear head (#2799)
# are small enough to run on the cpu partition, where the array is not capped
# by the GPU QOS - set CALIB_PARTITION=cpu CALIB_GRES=none for that.
PARTITION="${CALIB_PARTITION:-gpu}"
GRES_ARG=(--gres="$GRES")
[[ "$GRES" == "none" || -z "$GRES" ]] && GRES_ARG=()

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=${VTSEARCH_DATA_DIR:-} VTSEARCH_MODELS_DIR=${VTSEARCH_MODELS_DIR:-} HF_HOME=${HF_HOME:-}"

N=$(cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1; eval "$ENVX"; python run_cells.py --print-cells 2>/dev/null | tail -1)
if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
  echo "ERROR: could not determine cell count (got '$N'); is prepare done?" >&2
  exit 1
fi
echo "cells to run: $N (array 0-$((N-1))%$CONC, partition=$PARTITION gres=$GRES)"

# A study that submits SEVERAL arrays needs one name per arm: the completion
# waiter in the `grid-experiments` skill counts jobs BY NAME (`squeue -n`), and
# preflight's duplicate-name check reads the same field, so arms sharing a name
# make "has this arm drained?" unanswerable.  Defaults to the historic name, so
# every single-array launcher is unaffected.
JOB_NAME="${CALIB_JOB_NAME:-cal-cells}"

# `sbatch --parsable` prints nothing when the submission is refused, and the
# launch used to carry on with an empty id: #4668's last four arrays went that
# way, and the dependent analyze step with them (#4701).
require_jobid() {
  if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    echo "       'Slurm temporarily unable to accept job' means the cluster's job-record table" >&2
    echo "       (MaxJobCount) is full; see scripts/slurm/job_records.py" >&2
    exit 1
  fi
}

# Every array task is a job record from the moment it is queued, and the
# cluster's MaxJobCount caps records for every user at once; the %CONC throttle
# caps running tasks, not records.  So check the array against the live queue
# before sbatch sees it: a launch loop then stops at the array that would push
# the cluster past half its records, not at the one sbatch refuses (#4701).
# CALIB_SKIP_RECORD_CHECK=1 skips it.
if [[ "${CALIB_SKIP_RECORD_CHECK:-0}" != "1" ]]; then
  if ! python3 "$SELF/../../slurm/job_records.py" --tasks "$N"; then
    echo "ERROR: not submitting $JOB_NAME's $N-task array (see above)." >&2
    echo "       Chunk it or pack cells per task, or set CALIB_SKIP_RECORD_CHECK=1 if you mean it." >&2
    exit 1
  fi
fi

B=$(sbatch --parsable --job-name="$JOB_NAME" --array=0-$((N-1))%$CONC \
  "${GRES_ARG[@]}" --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition="$PARTITION" \
  --export=ALL --output="$LOGS/cells-%A_%a.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py")
require_jobid "$B" "$JOB_NAME's cells array"
echo "cells array: $B"
# Recorded now, so a refused analyze step below still leaves the array findable.
echo "$B" > "$LOGS/.cells_jobid"

# Which analyzer runs after the cells: the #2781 study's analyze.py (default)
# or the #2799 safe-threshold study's analyze_safe.py (set by launch_safe.sh).
ANALYZE="${CALIB_ANALYZE:-analyze.py}"
# Size the analysis step from the frame it will read, not from a default: a
# study that emits a per-(step, arm, k) side frame hands its analyzer tens of
# millions of rows, and the 16G/40min default is where such a run dies - after
# the cells have already been paid for.
AMEM="${CALIB_ANALYZE_MEM:-16G}"
ATIME="${CALIB_ANALYZE_TIME:-0:40:00}"
A=$(sbatch --parsable --dependency=afterany:$B --job-name="$JOB_NAME-analyze" --mem="$AMEM" \
  --cpus-per-task=4 --time="$ATIME" --partition=cpu \
  --export=ALL --output="$LOGS/analyze-%j.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python $ANALYZE")
require_jobid "$A" "$JOB_NAME's analyze step"
echo "analyze: $A"
echo "Report -> $CALIB_RESULTS/REPORT.md"
