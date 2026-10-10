#!/usr/bin/env bash
# #4746: should Autopilot's acquisition target precision move with the preset?  The high end, priced on the
# objective against today's app.
#
#   bash launch_acqtarget_4746.sh plan             # print each arm's knobs; submits nothing
#   bash launch_acqtarget_4746.sh size [idx] ARM   # time ONE cell
#   bash launch_acqtarget_4746.sh baseline         # copy the typed query's set at today's lines from PREP_BASE
#   bash launch_acqtarget_4746.sh preflight [ARM..]
#   bash launch_acqtarget_4746.sh status           # cells written per arm
#
# The arms are submitted in chunks by drive_chunks.sh (MaxJobCount, #4701):
#
#   LAUNCHER=$PWD/launch_acqtarget_4746.sh BASE=/expscratch/$USER/acqtarget-4746 JOB_PREFIX=acqt4746 \
#     RUNGS="$(bash launch_acqtarget_4746.sh arms)" PER_RUNG=30 DRAIN_BY="..." bash drive_chunks.sh
#
# WHY.  #3546 shipped target pick precision 0.5 at every preset (ACQUISITION_TARGET_PRECISION): Boundary picks and
# the atlas probe sample where the labels line's corpus posterior falls below 1/2.  Its grid priced only 0.5 and
# 0.25, each the same at all three presets.  Re-scoring its cells (step 1 of #4746) settles the low end at beta 4;
# nothing above ~0.55 (the line itself at beta 1/4) was ever run.  This prices 0.75 where a precision-leaning
# preset might want it.
#
# THE ARMS, each preset on its own sessions (`_b025`, `_b1`):
#
#   ctl_*     today's app (target 0.5)
#   tp75_*    target pick precision 0.75 (CALIB_ACQ_TARGET_P=0.75)
#
# Also in the table, not in the default list: ctl_b4 and tp25_b4, for a beta-4 read on today's app if one is wanted.
#
# Everything else is today's production.  The bench is #4668's / #4671's: COCO Better, binary SigLIP, 144 cells
# x 5 seeds, 150 votes, the user's pool at 1% (the withheld half at 0.44%).  The prepare and the typed query's
# baseline are #4671's (PREP_BASE): prepare_data.py, text_baseline.py and vtscore/training/thresholds are
# unchanged between #4671's run commit (44e4423c5) and this one, and the analyzer re-checks the baseline against
# every arm's own rows.  Every arm runs on one frozen commit, so the control pairs with the test arm on
# (category, seed).
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4746-run}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${ACQTARGET_BASE:-/expscratch/$USER/acqtarget-4746}"
PREP_BASE="${PREP_BASE:-/expscratch/$USER/autopilot-4671}"
export CALIB_HAYSTACK_PREVALENCE=0.01

ALL_ARMS="${ACQTARGET_ARMS:-ctl_b025 tp75_b025 ctl_b1 tp75_b1}"
TABLE_ARMS="ctl_b025 tp75_b025 ctl_b1 tp75_b1 ctl_b4 tp25_b4"

# --- science knobs: production everywhere the arm table does not say otherwise ---------------------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT CALIB_OPENING_DIVERSITY CALIB_LIVE_THRESHOLD
unset CALIB_ACQ_RANK_PERCENTILE CALIB_ACQ_P_CROSSING CALIB_ACQ_ORIGIN CALIB_ACQ_TARGET_P CALIB_ACQ_INCLUSION_OFFSET
unset CALIB_WALK_SHAPE CALIB_BAND_SHARE CALIB_SPOT_CHECK CALIB_SIGMA_FLOOR CALIB_MORE_WALK
unset CALIB_STARTUP_SCHEDULE CALIB_NEW_WALK CALIB_GOOD_DRY_RUN CALIB_QUOTA_DRY_BADS
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""

# --- environment (#4668's bench) ---------------------------------------------------------------------------
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all
export CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1

# --- sizing: 720 paired runs an arm resolve ~0.010 / 0.006 of the objective at beta 1/4 / 1 at 2 SE (#4584) ----
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"

# --- ops ---------------------------------------------------------------------------------------------------
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
export CALIB_MEM="${CALIB_MEM:-4G}"
export CALIB_TIME="${CALIB_TIME:-1:30:00}"
export CALIB_ANALYZE=noop.py
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"

# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  if ! [[ "$1" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  fi
}

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="acqt4746-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

link_prepare() {
  local src="$PREP_BASE/prepare/results"
  [[ -f "$src/prepare_info.json" ]] || { echo "ERROR: no prepare at $src" >&2; exit 1; }
  cp -n "$src/prepare_info.json" "$CALIB_RESULTS/"
  [[ -d "$src/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$src/crops" "$CALIB_RESULTS/crops"
  return 0
}

# The arm table: each arm names only what differs from today's app.
rung_env() {
  unset CALIB_BETA CALIB_ACQ_TARGET_P
  RUNG_DIVERGES=""
  local arm="${1%_*}" preset="${1##*_}"
  case "$preset" in
    b025) export CALIB_BETA=0.25 ;;
    b1) export CALIB_BETA=1 ;;
    b4) export CALIB_BETA=4 ;;
    *) echo "unknown preset in '$1'; expected one of: $TABLE_ARMS" >&2; exit 2 ;;
  esac
  case "$arm" in
    ctl) ;;
    tp75) export CALIB_ACQ_TARGET_P=0.75 ;;
    tp25) export CALIB_ACQ_TARGET_P=0.25 ;;
    *) echo "unknown arm '$1'; expected one of: $TABLE_ARMS" >&2; exit 2 ;;
  esac
  [[ "$CALIB_BETA" != "1" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,beta"
  [[ -n "${CALIB_ACQ_TARGET_P:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,acq_target_p"
  RUNG_DIVERGES="${RUNG_DIVERGES#,}"
  return 0
}

run_preflight() {
  local div=()
  [[ -n "$RUNG_DIVERGES" ]] && div=(--diverges "$RUNG_DIVERGES")
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 5 \
    "${div[@]}" --job-name "$CALIB_JOB_NAME" --mem "$CALIB_MEM" --conc 30
}

case "$MODE" in
  arms) echo "$ALL_ARMS" ;;

  size)
    IDX="${2:-0}"
    ARM="${3:?size needs an arm}"
    rung_env "$ARM"
    set_exp "sizing-$ARM-$IDX"
    link_prepare
    S=$(sbatch --parsable --job-name="acqt4746-size-$ARM-$IDX" --mem="$CALIB_MEM" --cpus-per-task=1 \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index $IDX --outdir $CALIB_RESULTS/cells")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX, $ARM) -> $CALIB_EXP/logs/size-$S.out"
    ;;

  baseline)
    # The typed query's own set at today's per-preset line (#4603): what every arm shows before a detector.
    src="$PREP_BASE/text_baseline.csv"
    [[ -f "$src" ]] || { echo "ERROR: no baseline at $src; run launch_autopilot_4671.sh baseline" >&2; exit 1; }
    mkdir -p "$BASE"
    cp -n "$src" "$BASE/text_baseline.csv"
    echo "baseline: $src -> $BASE/text_baseline.csv"
    ;;

  preflight)
    shift
    for arm in ${*:-$ALL_ARMS}; do
      (
        rung_env "$arm"
        set_exp "$arm"
        link_prepare
        echo "=== $arm: diverges=${RUNG_DIVERGES:-none}"
        run_preflight
      ) || exit 1
    done
    ;;

  plan)
    for arm in $TABLE_ARMS; do
      rung_env "$arm"
      printf '%-10s beta=%-5s target_p=%-5s diverges=%s\n' "$arm" "$CALIB_BETA" \
        "${CALIB_ACQ_TARGET_P:-app}" "${RUNG_DIVERGES:-none}"
    done
    echo "default arms: $ALL_ARMS"
    ;;

  status)
    for arm in $ALL_ARMS; do
      set_exp "$arm"
      n="$(find "$CALIB_RESULTS/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -n "$CALIB_JOB_NAME" -r -o %i 2>/dev/null | wc -l)"
      printf '%-10s %5s cells written  %4s tasks queued/running\n' "$arm" "$n" "$q"
    done
    ;;

  *)
    echo "usage: $0 {arms|size [idx] ARM|baseline|preflight [ARM..]|plan|status}" >&2
    exit 2
    ;;
esac
