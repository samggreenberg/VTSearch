#!/usr/bin/env bash
# Apptitude (#4356): does the app work on born-digital images? Every v0 class
# of app screens, run the way the State of the App runs coco_better's Binary
# Photo path: the typed query's text sort, shipped defaults, 150 clicks, the
# full-label ceiling, at the balance presets.
#
#   bash launch.sh prepare             # select cells (one CPU job)
#   bash launch.sh size <idx,...>      # time cells before sizing the array
#   bash launch.sh cells               # the array: every class x seed
#   bash launch.sh redo <idx,...>
#   bash launch.sh status
#
# What is pinned, and why:
#   * the text opening: each class types the text its labels were reviewed
#     under (EXPERIMENT_QUERIES["apptitude"]), and a cell that cannot open on it
#     is refused rather than run on known-goods;
#   * the split is by app (the cell's `split_group`, #4356): an app's screens
#     are all voted on or all withheld. This is the dataset's, not a knob;
#   * no test bands: a screen genre has no size axis;
#   * every class in classes_v0.txt (CALIB_CATEGORY_MODE=all + the file);
#   * shipped defaults for everything else, the full-label ceiling on, and the
#     rank frames and pick log the State of the App reads.
#
# The balance: APPTITUDE_BETA=0.25|1|4 runs every session at that preset and
# suffixes the run dir (-b025, -b1, -b4); unset is the app's default.
#
# The prevalence: APPTITUDE_PREVALENCE=0.01 thins every class's positives to that
# share of its cell before the split (CALIB_TARGET_PREVALENCE) and prefixes the
# run dir's suffix (-p001-b1). The pool is only the twelve classes' agreed
# screens, so a class's natural share runs from 1% (cart) to 36% (settings);
# a common rate is what lets classes, and app screens and photos, be compared.
#
# The typed query's hold (#4736): APPTITUDE_QUERY_HOLD=0 shows the trained head from
# the label quota instead of holding the query's set until 30 labels
# (CALIB_QUERY_HOLD), and adds -hold<N> to the run dir. Unset is the app. The
# hold also changes what the weak-separation check reads, so only a run prices
# it past the first check; the analyzer's offline price is exact before it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
CALIB="$WT/scripts/experiments/calibration"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
# The pile's datadir, never the live app's (the bashrc exports that one).
if [[ "$VTSEARCH_DATA_DIR" == "/expscratch/$USER/vtsearch-data" ]]; then
  echo "VTSEARCH_DATA_DIR is the live app's data; pile_env.sh should have replaced it" >&2; exit 2
fi
export VTS_REPO="$WT"

if [[ -n "${APPTITUDE_BETA:-}" ]]; then
  export CALIB_BETA="$APPTITUDE_BETA"
  _BTAG="-b$(python3 -c "import sys; b=float(sys.argv[1]); print(f'{b:g}'.replace('.', '') if b < 1 else f'{b:g}')" "$APPTITUDE_BETA")"
fi
if [[ -n "${APPTITUDE_PREVALENCE:-}" ]]; then
  export CALIB_TARGET_PREVALENCE="$APPTITUDE_PREVALENCE"
  _PTAG="-p$(python3 -c "import sys; print(f'{float(sys.argv[1]):g}'.replace('.', ''))" "$APPTITUDE_PREVALENCE")"
fi
if [[ -n "${APPTITUDE_QUERY_HOLD:-}" ]]; then
  export CALIB_QUERY_HOLD="$APPTITUDE_QUERY_HOLD"
  _PTAG="${_PTAG:-}-hold$APPTITUDE_QUERY_HOLD"
fi
export APPTITUDE_RUN="${APPTITUDE_RUN:-$(date +%Y-%m-%d)}"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/apptitude/runs/$APPTITUDE_RUN${_PTAG:-}${_BTAG:-}}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

export CALIB_DATASETS=apptitude
export CALIB_APPTITUDE_EMBEDDERS="${CALIB_APPTITUDE_EMBEDDERS:-siglip}"
export CALIB_CATEGORY_MODE=all
export CALIB_CATEGORY_FILE="${APPTITUDE_CLASSES:-$HERE/classes_v0.txt}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_N_SEEDS="${APPTITUDE_SEEDS:-10}"
export CALIB_MAX_STEPS="${APPTITUDE_MAX_STEPS:-150}"
export CALIB_CELL_ORDER=seed
export CALIB_SKYLINE_ARMS="${CALIB_SKYLINE_ARMS-skyline_train_full}"
export CALIB_SAFE_THRESHOLDS=1
export CALIB_EMIT_PICKS=1
export CALIB_RANK_FRAME_STEPS="${CALIB_RANK_FRAME_STEPS:-10,25,50,100,150}"
export CALIB_REPOOL_VARIANTS="" CALIB_SCHEDULE_VARIANTS="" CALIB_FOLD_COUNTS=""

MEM="${CALIB_MEM:-3G}"
CPUS="${CALIB_CPUS:-2}"
TIME="${CALIB_TIME:-1:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-60}"
JOB_NAME="${CALIB_JOB_NAME:-apptitude${_PTAG:-}${_BTAG:-}}"
LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO CALIB_DATASETS=$CALIB_DATASETS CALIB_APPTITUDE_EMBEDDERS=$CALIB_APPTITUDE_EMBEDDERS"
ENVX="$ENVX CALIB_CATEGORY_MODE=$CALIB_CATEGORY_MODE CALIB_CATEGORY_FILE=$CALIB_CATEGORY_FILE"
ENVX="$ENVX CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING CALIB_REQUIRE_SEED_QUERY=$CALIB_REQUIRE_SEED_QUERY"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_CELL_ORDER=$CALIB_CELL_ORDER CALIB_SKYLINE_ARMS=$CALIB_SKYLINE_ARMS"
ENVX="$ENVX CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS CALIB_EMIT_PICKS=$CALIB_EMIT_PICKS"
ENVX="$ENVX CALIB_RANK_FRAME_STEPS=$CALIB_RANK_FRAME_STEPS CALIB_BETA=${CALIB_BETA:-}"
ENVX="$ENVX CALIB_TARGET_PREVALENCE=${CALIB_TARGET_PREVALENCE:-} CALIB_QUERY_HOLD=${CALIB_QUERY_HOLD:-}"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS= CALIB_TEST_BANDS="

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job (#2897 lost both arms exactly this way).
submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$J" > "$LOGS/.jobid_$name"
    echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) — NOT LAUNCHED" >&2
    return 1
  fi
}

case "${1:-status}" in
prepare)
  echo "CALIB_EXP=$CALIB_EXP classes=$(grep -vc '^#' "$CALIB_CATEGORY_FILE") seeds=$CALIB_N_SEEDS beta=${CALIB_BETA:-default}"
  submit prepare --job-name=apptitude-prep --mem=8G --cpus-per-task=2 --time=1:00:00 \
    --partition="$PARTITION" --export=ALL --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $CALIB && python prepare_data.py"
  ;;

size)
  IDXS="${2:?usage: launch.sh size <comma-separated cell indices>}"
  SIZE_RESULTS="$CALIB_EXP/sizing"
  mkdir -p "$SIZE_RESULTS/cells"
  ln -sfn "$CALIB_RESULTS/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  [[ -e "$CALIB_RESULTS/crops" ]] && ln -sfn "$CALIB_RESULTS/crops" "$SIZE_RESULTS/crops"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="apptitude-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time=2:00:00 --partition="$PARTITION" --export=ALL --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$SIZE_RESULTS && cd $CALIB && time python run_cells.py --index $idx"
  done
  echo "read Elapsed and MaxRSS off 'sacct -j <id> --format=Elapsed,MaxRSS,State' before sizing the array"
  ;;

cells)
  [[ -s "$CALIB_RESULTS/prepare_info.json" ]] || { echo "no prepared grid at $CALIB_RESULTS (run prepare first)" >&2; exit 3; }
  N=$(cd "$CALIB" && python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
  fi
  echo "cells: $N (array 0-$((N-1))%$CONC on $PARTITION)"
  python - "$CALIB_RESULTS/grid_shape.json" "$N" <<PYSHAPE
import json, sys

json.dump(
    {
        "n_cells": int(sys.argv[2]),
        "datasets": ["apptitude"],
        "embedders": "$CALIB_APPTITUDE_EMBEDDERS".split(","),
        "classes": "$CALIB_CATEGORY_FILE",
        "n_seeds": int("$CALIB_N_SEEDS"),
        "max_steps": int("$CALIB_MAX_STEPS"),
        "cell_order": "$CALIB_CELL_ORDER",
        "split": "split_group (app)",
        "beta": "${CALIB_BETA:-}",
        "target_prevalence": "${CALIB_TARGET_PREVALENCE:-}",
        "query_hold": "${CALIB_QUERY_HOLD:-}",
        "job_name": "$JOB_NAME",
    },
    open(sys.argv[1], "w"),
    indent=2,
)
PYSHAPE
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" --diverges "${CALIB_BETA:+beta}" || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  submit cells --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL --output="$LOGS/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $CALIB && python run_cells.py"
  ;;

redo)
  # A failed task leaves its previous output in place: delete the stale cell
  # files first, or a re-run mixes two runs' cells.
  IDXS="${2:?usage: launch.sh redo <comma-separated indices>}"
  submit redo --job-name="$JOB_NAME-redo" --array="$IDXS%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL --output="$LOGS/redo-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $CALIB && python run_cells.py"
  ;;

status)
  echo "CALIB_EXP=$CALIB_EXP"
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R" | grep -E "apptitude|JOBID" || true
  echo "cells written: $(ls "$CALIB_RESULTS/cells" 2>/dev/null | grep -c '^task_[0-9]*\.csv' || true)"
  ;;
*)
  echo "usage: launch.sh {prepare|size <idx>|cells|redo <idx-list>|status}" >&2; exit 1 ;;
esac
