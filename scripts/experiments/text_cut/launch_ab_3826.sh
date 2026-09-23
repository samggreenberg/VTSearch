#!/usr/bin/env bash
# #3826 trajectory A/B: does the guarded text-sort line change what Autopilot's
# opening buys?  Two grids identical but for VTSEARCH_TEXT_SORT_CUT, paired per
# (category, seed) by calibration/analyze_ab.py.
#
#   bash launch_ab_3826.sh prepare      # text-seeded category selection (CALIB_REQUIRE_SEED_QUERY=1)
#   bash launch_ab_3826.sh size [IDX]   # time one cell before the arrays
#   bash launch_ab_3826.sh ab           # both arms
#   bash launch_ab_3826.sh abanalyze    # the paired analysis (ON = guarded_tail, OFF = gmm_midpoint)
#   bash launch_ab_3826.sh status
#
# Sizing, by the #3840 design rule: per-cell SD of dcost sigma ~ 0.04, and
# resolving delta = 0.01 at 2 SE needs (2 * 0.04 / 0.01)^2 = 64 paired cells.
# The grid is 3 datasets x siglip x whole_image x every text-queried selected
# category x CALIB_N_SEEDS seeds.  `status` prints the count; it must be >= 64.
#
# The ship rule is pre-registered on issue #3826 before `ab` is submitted.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
CALIB="$WT/scripts/experiments/calibration"
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
export VTS_REPO="$WT"

BASE="${TEXTCUT_AB_BASE:-/expscratch/$USER/textcut-ab-3826}"
PREP="$BASE/prepare/results"
LOGS="$BASE/logs"
mkdir -p "$LOGS" "$PREP/cells" "$PREP/crops"

export CALIB_DATASETS="caltech101_m,coco_val,visual_genome_m"
export CALIB_VG_EMBEDDERS=siglip CALIB_CALTECH_EMBEDDERS=siglip CALIB_COCO_EMBEDDERS=siglip
export CALIB_PATCH_STYLES=whole_image
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_REQUIRE_OPENING=text
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-2}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-100}"
export CALIB_CELL_ORDER=seed
export CALIB_SAFE_THRESHOLDS=1
export CALIB_REPOOL_VARIANTS="" CALIB_SCHEDULE_VARIANTS="" CALIB_FOLD_COUNTS="" CALIB_CUT_INCL_KS="" CALIB_FIT_QUALITY=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

MEM="${CALIB_MEM:-12G}"
CPUS="${CALIB_CPUS:-2}"
TIME="${CALIB_TIME:-3:00:00}"
CONC="${CALIB_CONC:-12}"
DEPARG=()
[[ -n "${DEP:-}" ]] && DEPARG=(--dependency="$DEP")

envx() {  # $1 = results dir
  echo "source $WT/gridenv.sh && source $WT/scripts/experiments/pile/pile_env.sh && export VTS_REPO=$WT" \
       "CALIB_EXP=$(dirname "$1") CALIB_RESULTS=$1 CALIB_DATASETS=$CALIB_DATASETS" \
       "CALIB_VG_EMBEDDERS=siglip CALIB_CALTECH_EMBEDDERS=siglip CALIB_COCO_EMBEDDERS=siglip" \
       "CALIB_PATCH_STYLES=whole_image CALIB_REQUIRE_SEED_QUERY=1 CALIB_REQUIRE_OPENING=text" \
       "CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS CALIB_CELL_ORDER=seed CALIB_SAFE_THRESHOLDS=1" \
       "CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS= CALIB_CUT_INCL_KS= CALIB_FIT_QUALITY=0" \
       "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"
}

n_cells() {
  (cd "$CALIB" && export CALIB_EXP="$BASE/prepare" CALIB_RESULTS="$PREP" && python run_cells.py --print-cells 2>/dev/null | tail -1)
}

case "${1:-}" in
prepare)
  sbatch --parsable "${DEPARG[@]}" --job-name=tcab-prep --mem=32G --cpus-per-task=2 --time=2:00:00 \
    --partition=cpu --output="$LOGS/prepare-%j.out" \
    --wrap="$(envx "$PREP") && cd $CALIB && python prepare_data.py"
  ;;
size)
  IDX="${2:-0}"
  SZ="$BASE/sizing/results"
  mkdir -p "$SZ/cells"
  ln -sfn "$PREP/prepare_info.json" "$SZ/prepare_info.json"
  ln -sfn "$PREP/crops" "$SZ/crops"
  sbatch --parsable "${DEPARG[@]}" --job-name=tcab-size --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition=cpu --output="$LOGS/size-%j.out" \
    --wrap="$(envx "$SZ") && export TEXTCUT_ARM=guarded_tail && cd $HERE && /usr/bin/time -v python run_cells_ab_3826.py --index $IDX"
  ;;
ab)
  N=$(n_cells)
  [[ "$N" =~ ^[0-9]+$ ]] || { echo "could not count cells (got '$N'); did prepare finish?" >&2; exit 1; }
  (( N >= 64 )) || { echo "only $N cells; the #3840 rule needs >= 64 for delta = 0.01" >&2; exit 1; }
  for arm in gmm_midpoint guarded_tail; do
    R="$BASE/ab_$arm/results"
    mkdir -p "$R/cells"
    ln -sfn "$PREP/prepare_info.json" "$R/prepare_info.json"
    ln -sfn "$PREP/crops" "$R/crops"
    sbatch --parsable "${DEPARG[@]}" --job-name="tcab-$arm" --array="0-$((N-1))%$CONC" \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition=cpu \
      --output="$LOGS/ab-$arm-%A_%a.out" \
      --wrap="$(envx "$R") && export TEXTCUT_ARM=$arm && cd $HERE && python run_cells_ab_3826.py"
  done
  ;;
abanalyze)
  ON="$BASE/ab_guarded_tail/results"; OFF="$BASE/ab_gmm_midpoint/results"
  sbatch --parsable "${DEPARG[@]}" --job-name=tcab-an --mem=16G --cpus-per-task=2 --time=1:00:00 \
    --partition=cpu --output="$LOGS/analyze-ab-%j.out" \
    --wrap="$(envx "$ON") && export CALIB_AB_ON=$ON CALIB_AB_OFF=$OFF CALIB_AB_OUT=$BASE/analysis && mkdir -p $BASE/analysis && cd $CALIB && python analyze_ab.py && cd $HERE && python analyze_ab_3826.py --on $ON --off $OFF --out $BASE/analysis"
  ;;
status)
  echo "cells:   $(n_cells)"
  for arm in gmm_midpoint guarded_tail; do
    echo "ab/$arm: $(ls "$BASE/ab_$arm/results/cells" 2>/dev/null | grep -c '^task_[0-9]*\.csv$') cells written"
  done
  squeue -u "$USER" -o "%.14i %.14j %.8T %.8M" | grep tcab || true
  ;;
*)
  echo "usage: $0 {prepare|size [IDX]|ab|abanalyze|status}" >&2
  exit 2
  ;;
esac
