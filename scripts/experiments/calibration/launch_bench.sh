#!/usr/bin/env bash
# Broad VTSearch overview benchmark: production defaults across the pile.
#
#   bash launch_bench.sh prepare      # category selection + exemplar crops
#   bash launch_bench.sh size         # time one cheap + one expensive cell
#   bash launch_bench.sh cells        # the full array
#   bash launch_bench.sh status
#
# Everything here rides the SHIPPED defaults on purpose: this study asks "what
# does a current VTSearch user get?", so the only knobs set are sizing ones
# (which datasets/embedders/categories/seeds), never behaviour ones.  A knob
# that changes the decision rule would make these numbers unreadable as a
# baseline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/bench-overview}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

# --- the grid (sizing only) ------------------------------------------------
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,caltech101_m,coco_val}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip2_l,dinov3_patch}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip,siglip2_l,dinov3_patch}"
export CALIB_COCO_EMBEDDERS="${CALIB_COCO_EMBEDDERS:-siglip,siglip2_l,dinov3_patch}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-3}"
export CALIB_N_PER_BAND="${CALIB_N_PER_BAND:-2}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-6}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

# Counterfactual extras off: this is a baseline, not a contrast.
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""

# PRODUCTION GEOMETRY ONLY.  The calibration study's default is
# "max_patch,max_patch_pca_hac" because that study wanted the contrast -- but a
# *study* default is not a *shipped* default.  Per vtscore/eval/patch_styles.py,
# `max_patch` IS the production patch pipeline, while the HAC hybrids lost the
# Max-Patch study at the operating point (PR #2749) and production no longer
# carries the tree they delegate to; they survive only for an open calibration
# follow-up.  Running them here would put non-production rows in a benchmark
# whose whole purpose is "what does a current user get", and would double the
# cost of every patch cell.
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells"

MEM="${CALIB_MEM:-64G}"
CPUS="${CALIB_CPUS:-6}"
TIME="${CALIB_TIME:-6:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-24}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO"
ENVX="$ENVX CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS CALIB_CALTECH_EMBEDDERS=$CALIB_CALTECH_EMBEDDERS"
ENVX="$ENVX CALIB_COCO_EMBEDDERS=$CALIB_COCO_EMBEDDERS"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_N_PER_BAND=$CALIB_N_PER_BAND"
ENVX="$ENVX CALIB_N_CATEGORIES=$CALIB_N_CATEGORIES CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
# Explicit, not via --export=ALL.  This one decides whether the run measures the
# production geometry or silently adds a retired arm, so it must not depend on
# the submitting shell's environment surviving into the job.
ENVX="$ENVX CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"
# Interaction axis: does the simulated user drag boxes, or only answer Good/Bad?
# Explicit for the same reason CALIB_PATCH_STYLES is: it decides what was measured.
ENVX="$ENVX CALIB_INTERACTION=${CALIB_INTERACTION:-boxes}"
ENVX="$ENVX CALIB_CATEGORY_MODE=${CALIB_CATEGORY_MODE:-}"

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job (#2897 lost both arms this way).
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
  echo "CALIB_EXP=$CALIB_EXP"
  echo "datasets=$CALIB_DATASETS seeds=$CALIB_N_SEEDS per_band=$CALIB_N_PER_BAND"
  submit prepare --job-name=bench-prep --mem="${CALIB_PREP_MEM:-96G}" --cpus-per-task=8 \
    --time=3:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py"
  ;;

size)
  # Size from a REAL cell, not a guess (and one per cost class: a patch cell can
  # be 10x a whole-image one, which changes the array budget entirely).
  # Writes into a throwaway results dir so the timing cells are not mistaken for
  # study cells by the resume logic.
  IDXS="${2:?usage: launch_bench.sh size <comma-separated cell indices>}"
  SIZE_RESULTS="$CALIB_EXP/sizing"
  mkdir -p "$SIZE_RESULTS/cells"
  ln -sfn "$CALIB_RESULTS/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_RESULTS/crops" "$SIZE_RESULTS/crops"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="bench-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time=2:00:00 --partition="$PARTITION" --export=ALL \
      --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$SIZE_RESULTS && cd $HERE && time python run_cells.py --index $idx"
  done
  ;;

cells)
  N=$(cd "$HERE" && python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
  fi
  echo "cells: $N (array 0-$((N-1))%$CONC on $PARTITION)"
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  submit cells --job-name=bench-cells --array="0-$((N-1))%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R"
  echo "=== cells written ==="
  ls "$CALIB_RESULTS/cells" 2>/dev/null | grep -c 'task_.*\.csv$' || echo 0
  echo "=== zero-byte cells (resume would SKIP these) ==="
  find "$CALIB_RESULTS/cells" -name 'task_*.csv' -size 0 2>/dev/null | wc -l
  ;;

*)
  echo "usage: $0 {prepare|size|cells|status}" >&2; exit 1
  ;;
esac
