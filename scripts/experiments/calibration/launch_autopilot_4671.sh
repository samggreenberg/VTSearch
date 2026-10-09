#!/usr/bin/env bash
# #4671: price two Autopilot pieces that shipped without an F-beta read, against today's app, at every preset.
#
#   bash launch_autopilot_4671.sh prepare          # stage 0, ONCE (cpu, reads the pile in place)
#   bash launch_autopilot_4671.sh size [idx] ARM   # time ONE cell
#   bash launch_autopilot_4671.sh plan             # print each arm's knobs; submits nothing
#   bash launch_autopilot_4671.sh baseline         # the typed query's set at today's lines, per cell
#   bash launch_autopilot_4671.sh status           # cells written per arm, against the grid size
#
# The arms are submitted in chunks by drive_chunks.sh (MaxJobCount, #4701), not by a `rungs` mode here:
#
#   LAUNCHER=$PWD/launch_autopilot_4671.sh BASE=/expscratch/$USER/autopilot-4671 JOB_PREFIX=ap4671 \
#     RUNGS="$(bash launch_autopilot_4671.sh arms)" bash drive_chunks.sh
#
# THE ARMS, each at beta 1/4, 1 and 4 (`_b025`, `_b1`, `_b4`), each preset on its own sessions:
#
#   ctl_*     today's app
#   open_*    the opening before #4288: `g3@top,b4@mid` (3 Goods off the top, 4 Bads at the line), no More
#             walk; against today's dry-stop opening `g3@top,b4@mid,g20+dry1/16@top` (#4222), shipped on AP
#   nowalk_*  New takes the Hard pick instead of the Coverage Atlas's walk (`CALIB_NEW_WALK=hard`), the phase
#             machine unchanged: what the atlas walk's picks buy, which nothing has measured
#
# Everything else is today's production. The bench is #4668's: COCO Better, binary SigLIP, 144 cells x 5
# seeds, 150 votes, the user's pool at 1% (the withheld half at 0.44%). Every arm runs on one frozen commit, so
# the control pairs with both test arms on (category, seed).
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4671-run}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${AUTOPILOT_BASE:-/expscratch/$USER/autopilot-4671}"
export CALIB_HAYSTACK_PREVALENCE=0.01

PRESETS="b025 b1 b4"
ALL_ARMS=""
for arm in ctl open nowalk; do
  for p in $PRESETS; do ALL_ARMS="$ALL_ARMS ${arm}_$p"; done
done
ALL_ARMS="${ALL_ARMS# }"

# --- science knobs: production everywhere the arm table does not say otherwise ---------------------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT CALIB_OPENING_DIVERSITY CALIB_LIVE_THRESHOLD
unset CALIB_ACQ_RANK_PERCENTILE CALIB_ACQ_P_CROSSING CALIB_ACQ_ORIGIN CALIB_ACQ_TARGET_P CALIB_ACQ_INCLUSION_OFFSET
unset CALIB_WALK_SHAPE CALIB_BAND_SHARE CALIB_SPOT_CHECK CALIB_SIGMA_FLOOR CALIB_MORE_WALK
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

# --- sizing: 720 paired runs an arm resolve ~0.010 / 0.006 / 0.0075 of the objective at 2 SE (#4584) ----------
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
  export CALIB_JOB_NAME="ap4671-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

link_prepare() {
  local src="$BASE/prepare/results"
  [[ -f "$src/prepare_info.json" ]] || { echo "ERROR: run '$0 prepare' first" >&2; exit 1; }
  cp -n "$src/prepare_info.json" "$CALIB_RESULTS/"
  [[ -d "$src/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$src/crops" "$CALIB_RESULTS/crops"
  return 0
}

# The arm table: each arm names only what differs from today's app.
rung_env() {
  unset CALIB_BETA CALIB_STARTUP_SCHEDULE CALIB_NEW_WALK
  RUNG_DIVERGES=""
  local arm="${1%_*}" preset="${1##*_}"
  case "$preset" in
    b025) export CALIB_BETA=0.25 ;;
    b1) export CALIB_BETA=1 ;;
    b4) export CALIB_BETA=4 ;;
    *) echo "unknown preset in '$1'; expected one of: $ALL_ARMS" >&2; exit 2 ;;
  esac
  case "$arm" in
    ctl) ;;
    open) export CALIB_STARTUP_SCHEDULE="g3@top,b4@mid" ;;
    nowalk) export CALIB_NEW_WALK=hard ;;
    *) echo "unknown arm '$1'; expected one of: $ALL_ARMS" >&2; exit 2 ;;
  esac
  [[ "$CALIB_BETA" != "1" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,beta"
  [[ -n "${CALIB_STARTUP_SCHEDULE:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,startup_schedule"
  [[ -n "${CALIB_NEW_WALK:-}" ]] && RUNG_DIVERGES="$RUNG_DIVERGES,new_walk"
  RUNG_DIVERGES="${RUNG_DIVERGES#,}"
  return 0
}

run_preflight() {
  local div=()
  [[ -n "$RUNG_DIVERGES" ]] && div=(--diverges "$RUNG_DIVERGES")
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 5 \
    "${div[@]}" --job-name "$CALIB_JOB_NAME" --mem "$CALIB_MEM" --conc 10
}

case "$MODE" in
  arms) echo "$ALL_ARMS" ;;

  prepare)
    set_exp prepare
    P=$(sbatch --parsable --job-name=ap4671-prep --mem=32G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P -> $CALIB_EXP/logs/prepare-$P.out"
    ;;

  size)
    IDX="${2:-0}"
    ARM="${3:?size needs an arm}"
    rung_env "$ARM"
    set_exp "sizing-$ARM-$IDX"
    link_prepare
    S=$(sbatch --parsable --job-name="ap4671-size-$ARM-$IDX" --mem="$CALIB_MEM" --cpus-per-task=1 \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index $IDX --outdir $CALIB_RESULTS/cells")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX, $ARM) -> $CALIB_EXP/logs/size-$S.out"
    ;;

  baseline)
    # The typed query's own set at today's per-preset line (#4603), what every arm shows before a detector.
    set_exp prepare
    T=$(sbatch --parsable --job-name=ap4671-baseline --mem=32G --cpus-per-task=2 \
      --time=1:00:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/baseline-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $CALIB_RESULTS --out $BASE/text_baseline.csv")
    require_jobid "$T" "text baseline"
    echo "baseline job: $T -> $BASE/text_baseline.csv"
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
    for arm in $ALL_ARMS; do
      rung_env "$arm"
      printf '%-12s beta=%-5s opening=%-16s new_walk=%-6s diverges=%s\n' "$arm" "$CALIB_BETA" \
        "${CALIB_STARTUP_SCHEDULE:-app}" "${CALIB_NEW_WALK:-atlas}" "${RUNG_DIVERGES:-none}"
    done
    ;;

  status)
    for arm in $ALL_ARMS; do
      set_exp "$arm"
      n="$(find "$CALIB_RESULTS/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -n "$CALIB_JOB_NAME" -r -o %i 2>/dev/null | wc -l)"
      printf '%-12s %5s cells written  %4s tasks queued/running\n' "$arm" "$n" "$q"
    done
    ;;

  *)
    echo "usage: $0 {arms|prepare|size [idx] ARM|baseline|preflight [ARM..]|plan|status}" >&2
    exit 2
    ;;
esac
