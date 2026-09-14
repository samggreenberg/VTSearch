#!/usr/bin/env bash
# Where should the ANCHORED refit stop? (#3825)
#
#   bash launch_3825.sh gate       # replay the #3585 fold corpus through every stopping rule
#   bash launch_3825.sh bench      # per-call cost, min-of-k, on real fold shapes
#   bash launch_3825.sh ab         # the trajectory A/B, one grid per arm in AB_ARMS
#   bash launch_3825.sh abanalyze  # pair the A/B grids
#   bash launch_3825.sh analyse    # the gate tables and the report's figures
#   bash launch_3825.sh status
#
# THE QUESTION.  #3585 replaced the unanchored fit and, measuring the loop next
# to it, found that `_anchored_em` had become ~90% of a calibration fold's fit:
# 97-200 iterations of a parameter-delta criterion at 1e-8, against the init's
# ~15, and on a large minority of real folds no convergence at all - the loop
# exits on `max_iter`.  The fix #3585 found transfers (stop on the likelihood,
# not on the parameters) and `_anchored_em` already takes an opt-in `loglik_tol`
# for it.  What does NOT transfer is the cushion: there, the thing being changed
# was the initialiser of a fit that then re-converged, with the anchored refit
# still downstream to absorb the difference.  Here the anchored refit IS the
# shipped threshold - between this loop and the green/red line there is a
# quantile and a snap, and nothing else.
#
# WHY THERE IS NO CAPTURE STAGE.  There is nothing to capture: #3585's corpus is
# 84 cells of real `fit_fold_anchored_cut` inputs - every array the shipped fold
# path saw, on a stride - and the anchored refit is exactly what consumes them.
# Capturing arrays instead of verdicts is what lets a question nobody had asked
# yet be answered against inputs nobody re-ran.  CORPUS points at it.
#
# THE ARMS are in `arms_3825.py`, on two axes: the tolerance, and WHICH
# LIKELIHOOD is watched - the weighted semi-supervised objective this EM
# ascends, or the free sample's alone (the rule the sort path uses, which for an
# anchored fit is not the estimator's own objective and has no theorem making it
# monotone).  Two arms perturb the incumbent instead of replacing it, because
# "the candidate moves N% of admitted sets" is unreadable without them.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
CALIB="$WT/scripts/experiments/calibration"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/anchem-3825}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"
REUSE_PREPARE="${REUSE_PREPARE:-/expscratch/$USER/bench-overview/results}"
# The #3585 corpus, reused verbatim.  Same grid, same shipped path, same arrays.
CORPUS="${CORPUS:-/expscratch/$USER/gmm-3585/corpus}"
ANALYSIS="${ANALYSIS:-$CALIB_EXP/analysis}"

# BLAS pinned at top level so the gate and the bench measure the same thing.
# The anchored loop deliberately never dispatches to BLAS (#3166), so an
# unpinned comparison would price the thread pool rather than the fit.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# The #3585 grid, unchanged: it is the one the corpus was captured from, and the
# A/B has to be paired against cells of the same shape.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,caltech101_m,coco_val}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_COCO_EMBEDDERS="${CALIB_COCO_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-2}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-100}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_SAFE_THRESHOLDS=1
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_CUT_INCL_KS=""
export CALIB_FIT_QUALITY=0
export CALIB_REQUIRE_OPENING="${CALIB_REQUIRE_OPENING:-mixed}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells" "$ANALYSIS"

if [[ ! -e "$CALIB_RESULTS/prepare_info.json" ]]; then
  [[ -f "$REUSE_PREPARE/prepare_info.json" ]] || {
    echo "REUSE_PREPARE=$REUSE_PREPARE has no prepare_info.json" >&2; exit 3; }
  ln -sfn "$REUSE_PREPARE/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
  ln -sfn "$REUSE_PREPARE/crops" "$CALIB_RESULTS/crops"
  echo "linked prepare from $REUSE_PREPARE"
fi

MEM="${CALIB_MEM:-12G}"
CPUS="${CALIB_CPUS:-2}"
TIME="${CALIB_TIME:-2:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-40}"
JOB_NAME="${CALIB_JOB_NAME:-anchem-3825}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS CALIB_CALTECH_EMBEDDERS=$CALIB_CALTECH_EMBEDDERS"
ENVX="$ENVX CALIB_COCO_EMBEDDERS=$CALIB_COCO_EMBEDDERS CALIB_CATEGORY_MODE=$CALIB_CATEGORY_MODE"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_CELL_ORDER=$CALIB_CELL_ORDER CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"
ENVX="$ENVX CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS CALIB_FIT_QUALITY=0"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS= CALIB_CUT_INCL_KS="
ENVX="$ENVX CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING"
ENVX="$ENVX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

# Chain a stage behind another job GRID-side (DEP=afterany:<id>): a waiter on
# the laptop dies with the VPN, and every stage here outlives an ssh session.
DEP="${DEP:-}"
DEP_ARG=()
[[ -n "$DEP" ]] && DEP_ARG=(--dependency="$DEP")

submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$J" > "$LOGS/.jobid_$name"
    echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) -- NOT LAUNCHED" >&2
    return 1
  fi
}

n_cells() { ( cd "$CALIB" && python run_cells.py --print-cells 2>/dev/null | tail -1 ); }

AB_ARMS="${AB_ARMS:-baseline ll1e-3 ll1e-6}"

case "${1:-status}" in
gate)
  [[ -d "$CORPUS" ]] || { echo "no corpus at $CORPUS" >&2; exit 3; }
  submit gate --job-name="$JOB_NAME-gate" "${DEP_ARG[@]}" --mem="${GATE_MEM:-32G}" --cpus-per-task=2 \
    --time="${GATE_TIME:-8:00:00}" --partition="$PARTITION" --export=ALL \
    --output="$LOGS/gate-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python gate_3825.py --corpus $CORPUS --out $ANALYSIS"
  ;;

bench)
  # One fold per sampled case, min-of-5.  The cost of this stage is
  # (folds x arms x sizes x reps) EM runs and the 50k resample is the expensive
  # one - the shipped rule runs 200 iterations there - so the sample size is a
  # knob and the time limit is generous rather than fitted.
  submit bench --job-name="$JOB_NAME-bench" "${DEP_ARG[@]}" --mem=16G --cpus-per-task=2 \
    --time="${BENCH_TIME:-6:00:00}" --partition="$PARTITION" --export=ALL \
    --output="$LOGS/bench-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python bench_3825.py --corpus $CORPUS --out $ANALYSIS/bench3825.csv --per-cell ${BENCH_PER_CELL:-1} --reps ${BENCH_REPS:-5}"
  ;;

ab)
  # The trajectory A/B the gate cannot do.  One grid per arm, paired by
  # (category, seed) at analysis time.  Three arms rather than two because the
  # gate cannot pick the tolerance on its own: the cheap end and the
  # conservative end bracket it, and `baseline` is what both are read against.
  N=$(n_cells)
  [[ "$N" =~ ^[0-9]+$ ]] || { echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1; }
  echo "cells: $N per arm; arms: $AB_ARMS"
  for arm in $AB_ARMS; do
    AB_RESULTS="$CALIB_EXP/ab_$arm/results"
    mkdir -p "$AB_RESULTS/cells"
    ln -sfn "$REUSE_PREPARE/prepare_info.json" "$AB_RESULTS/prepare_info.json"
    ln -sfn "$REUSE_PREPARE/crops" "$AB_RESULTS/crops"
    submit "ab_$arm" --job-name="$JOB_NAME-ab-$arm" "${DEP_ARG[@]}" --array="0-$((N-1))%$CONC" \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
      --partition="$PARTITION" --export=ALL \
      --output="$LOGS/ab-$arm-%A_%a.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$AB_RESULTS ANCHORED_STOP_ARM=$arm && cd $HERE && python run_cells_arm_3825.py"
  done
  ;;

abanalyze)
  # One pairing per candidate against the shipped rule.  `analyze_ab.py` takes
  # exactly two grids, so a third arm is a second invocation, not a third column.
  for arm in $AB_ARMS; do
    [[ "$arm" == "baseline" ]] && continue
    submit "abanalyze_$arm" --job-name="$JOB_NAME-abanalyze-$arm" "${DEP_ARG[@]}" --mem=32G --cpus-per-task=2 \
      --time=2:00:00 --partition="$PARTITION" --export=ALL \
      --output="$LOGS/abanalyze-$arm-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_AB_ON=$CALIB_EXP/ab_$arm/results CALIB_AB_OFF=$CALIB_EXP/ab_baseline/results CALIB_AB_OUT=$ANALYSIS/ab_$arm && mkdir -p $ANALYSIS/ab_$arm && cd $CALIB && python analyze_ab.py"
  done
  ;;

analyse)
  submit analyse --job-name="$JOB_NAME-analyse" "${DEP_ARG[@]}" --mem=32G --cpus-per-task=2 \
    --time=1:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/analyse-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python analyze_3825.py --analysis $ANALYSIS && python figures_3825.py --analysis $ANALYSIS"
  ;;

status)
  echo "exp:     $CALIB_EXP"
  echo "corpus:  $(find "$CORPUS" -name 'cell_*.npz' 2>/dev/null | wc -l) captures (reused from #3585)"
  echo "queue:   $(squeue -u "$USER" -h -n "$JOB_NAME-gate,$JOB_NAME-bench,$JOB_NAME-analyse" -o %i | wc -l) named tasks"
  echo "gate:    $(ls -1 "$ANALYSIS"/gate3825_*.csv 2>/dev/null | wc -l) frames"
  echo "bench:   $([[ -f "$ANALYSIS/bench3825.csv" ]] && echo yes || echo no)"
  for arm in $AB_ARMS; do
    echo "ab/$arm: $(find "$CALIB_EXP/ab_$arm/results/cells" -name 'task_*.csv' ! -name '*__*' 2>/dev/null | wc -l) cells, $(squeue -u "$USER" -h -n "$JOB_NAME-ab-$arm" -o %i | wc -l) queued"
  done
  ;;

*)
  echo "usage: $0 {gate|bench|ab|abanalyze|analyse|status}" >&2
  exit 1
  ;;
esac
