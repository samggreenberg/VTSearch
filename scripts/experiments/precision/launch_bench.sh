#!/usr/bin/env bash
# Precision study (#3143), part 3: run the SAME benchmark arm against each
# precision's gallery and check the decision metrics move less than the margin
# the calibration studies resolve (0.005).
#
#   bash launch_bench.sh prepare                 # per-arm category selection + exemplars
#   bash launch_bench.sh size fp32_l40s 0        # time ONE real cell first
#   bash launch_bench.sh cells                   # both arms' arrays
#   bash launch_bench.sh status
#   bash launch_bench.sh analyze
#
# Everything rides the SHIPPED defaults except the gallery precision.  A knob
# that changed the decision rule would make these numbers unreadable as a
# precision contrast, which is the same reason launch_bench.sh in
# ../calibration/ pins production geometry.
#
# The two arms are paired by (category, seed): identical categories, identical
# sim/test split, identical exemplar id — only the vectors differ.  That is what
# makes a *paired* SE small enough to resolve 0.005 at this cell count; an
# unpaired comparison at the same cost could not.
set -euo pipefail

# `set -e` can abort anywhere, including mid-preflight -- which is how this
# launcher once exited 1 with no output at all and no job submitted (see
# lessons/2026-08-17-a-launcher-that-exits-1-with-no-output.md).  An ERR trap
# makes an abort name itself, so "no output" can never again read as a quiet pass.
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
CALIB="$WT/scripts/experiments/calibration"

source "$WT/gridenv.sh"

export VTS_REPO="$WT"
export VTS_PRECISION_STUDY="${VTS_PRECISION_STUDY:-/expscratch/$USER/precision-3143}"
export VTS_PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
BENCH_ROOT="$VTS_PRECISION_STUDY/bench"

# The candidate against the reference.  autocast is off the bench by default: it
# is a *slower* implementation of the same idea, so it only earns a bench run if
# the drift analysis says the weight cast moved something.
ARMS="${VTS_BENCH_ARMS:-fp32_l40s,fp16_l40s}"

# --- the grid (sizing only) ------------------------------------------------
# Six seeds rather than the overview benchmark's three: this study has to support
# a claim that a difference is BELOW 0.005, and an underpowered null is worthless
# — "we could not resolve it" is not "it is not there".  Pairing does most of the
# work; the seeds buy the rest.
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip2_l}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-6}"
export CALIB_N_PER_BAND="${CALIB_N_PER_BAND:-2}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-6}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

# Counterfactual extras off: a baseline contrast, not a rule study.
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"

MEM="${CALIB_MEM:-16G}"
CPUS="${CALIB_CPUS:-6}"
TIME="${CALIB_TIME:-6:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
# 12 per arm, not 24: two arrays run at once and memory is a PER-USER quota.
# 2 arms x 12 x 16G = 384G, comfortably inside the 1100G cap with room for the
# analysis jobs behind it (in #3129 a fat array parked five diagnostics for 25
# minutes behind its own reservation).
CONC="${CALIB_CONC:-12}"

arm_env() {
  # Everything the job needs, explicit rather than via --export=ALL: the data dir
  # is what decides which precision's gallery this arm reads, so it must not
  # depend on the submitting shell's environment surviving into the job.
  local arm="$1"
  local pile="$VTS_PRECISION_STUDY/piles/$arm"
  local exp="$BENCH_ROOT/$arm"
  echo "export CALIB_EXP=$exp CALIB_RESULTS=$exp/results" \
       "VTSEARCH_DATA_DIR=$pile/datadir" \
       "VTSEARCH_MODELS_DIR=$VTS_PILE/models" \
       "HF_HOME=$VTS_PILE/models" \
       "VTS_REPO=$WT" \
       "CALIB_DATASETS=$CALIB_DATASETS CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS" \
       "CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_N_PER_BAND=$CALIB_N_PER_BAND" \
       "CALIB_N_CATEGORIES=$CALIB_N_CATEGORIES CALIB_MAX_STEPS=$CALIB_MAX_STEPS" \
       "CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS=" \
       "CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"
}

submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) — NOT LAUNCHED" >&2
    return 1
  fi
}

case "${1:-status}" in
prepare)
  for arm in ${ARMS//,/ }; do
    exp="$BENCH_ROOT/$arm"
    mkdir -p "$exp/logs" "$exp/results/cells"
    if [[ ! -f "$VTS_PRECISION_STUDY/piles/$arm/provenance.json" ]]; then
      echo "FAIL: arm $arm has no provenance.json — build its cells first" >&2; exit 2
    fi
    submit "prep-$arm" --job-name="precbench-prep-$arm" --mem=32G --cpus-per-task=8 \
      --time=2:00:00 --partition="$PARTITION" \
      --output="$exp/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $(arm_env "$arm") && cd $CALIB && python prepare_data.py"
  done
  echo
  echo "when both finish: bash $0 verify-pairing   (categories+exemplars MUST match)"
  ;;

verify-pairing)
  # The pairing is the whole design.  Category selection reads only boxes and
  # counts, and exemplar candidates come from a per-category RNG seed, so the two
  # arms SHOULD agree exactly — but "should" is what #2877 shipped on.  If they
  # disagree, the arms are different studies and the paired SE is a fiction.
  cd "$HERE" && python verify_pairing.py --arms "$ARMS" --root "$BENCH_ROOT"
  ;;

size)
  ARM="${2:?usage: launch_bench.sh size <arm> <cell index>}"
  IDX="${3:?usage: launch_bench.sh size <arm> <cell index>}"
  exp="$BENCH_ROOT/$ARM"
  SIZE_RESULTS="$exp/sizing"
  mkdir -p "$SIZE_RESULTS/cells" "$exp/logs"
  ln -sfn "$exp/results/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  ln -sfn "$exp/results/crops" "$SIZE_RESULTS/crops"
  submit "size-$ARM-$IDX" --job-name="precbench-size" --mem="$MEM" --cpus-per-task="$CPUS" \
    --time=2:00:00 --partition="$PARTITION" \
    --output="$exp/logs/size-$IDX-%j.out" \
    --wrap="source $WT/gridenv.sh && $(arm_env "$ARM") && export CALIB_RESULTS=$SIZE_RESULTS && cd $CALIB && time python run_cells.py --index $IDX"
  ;;

cells)
  for arm in ${ARMS//,/ }; do
    exp="$BENCH_ROOT/$arm"
    mkdir -p "$exp/logs" "$exp/results/cells"
    N=$(cd "$CALIB" && env $(arm_env "$arm" | sed 's/^export //') python run_cells.py --print-cells 2>/dev/null | tail -1)
    if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
      echo "ERROR: could not determine cell count for $arm (got '$N')" >&2; exit 1
    fi
    JOB_NAME="precbench-$arm"
    echo "$arm: $N cells (array 0-$((N-1))%$CONC on $PARTITION)"
    bash "$WT/scripts/experiments/preflight.sh" --exp "$exp" --arms prod \
      --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" || {
      echo "PREFLIGHT FAILED for $arm" >&2; exit 2; }
    submit "cells-$arm" --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
      --partition="$PARTITION" \
      --output="$exp/logs/cells-%A_%a.out" \
      --wrap="source $WT/gridenv.sh && $(arm_env "$arm") && cd $CALIB && python run_cells.py"
  done
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.20j %.9T %.11M %.6D %R" | grep -E 'precbench|JOBID' || echo "(no precbench jobs)"
  for arm in ${ARMS//,/ }; do
    d="$BENCH_ROOT/$arm/results/cells"
    n=$(ls "$d" 2>/dev/null | grep -c 'task_[0-9]*\.csv$' || echo 0)
    z=$(find "$d" -name 'task_*.csv' -size 0 2>/dev/null | wc -l || true)
    echo "  $arm: $n cells written, $z zero-byte (resume would SKIP those)"
  done
  ;;

analyze)
  cd "$HERE" && python analyze_bench_precision.py --arms "$ARMS" --root "$BENCH_ROOT"
  ;;

*)
  echo "usage: $0 {prepare|verify-pairing|size <arm> <idx>|cells|status|analyze}" >&2; exit 1
  ;;
esac
