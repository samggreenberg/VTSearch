#!/usr/bin/env bash
# #4864: the calibration split's rounding, dithered (the app) against nearest.
#
#   bash launch_split_rounding_4864.sh plan            # every arm's knobs; submits nothing
#   bash launch_split_rounding_4864.sh size [ARM]      # time ONE bundle of cells (default nearest)
#   bash launch_split_rounding_4864.sh arms [ARM..]    # submit binary arm arrays (default: both)
#   bash launch_split_rounding_4864.sh status          # cells written per binary arm
#   bash launch_split_rounding_4864.sh region-size     # time one cell of r_nearest
#   bash launch_split_rounding_4864.sh region          # both region arms
#   bash launch_split_rounding_4864.sh region-status
#
# WHY.  The folds hold out a share of each class, and how many Goods that is
# at a given count of Goods is a coin (`_dithered_count`, #3288), drawn again
# at every vote.  Over a band of Goods counts the labels line is that coin's,
# so it can switch on any vote.  `CALIB_SPLIT_ROUNDING=nearest` sizes each
# class's holdout to the nearest whole vote, with no coin: the switch then
# happens once.  What it may cost is #3286's ripple, sizes that step together
# across the eval's runs.  The issue's comment pre-registers the decision rule.
#
# ARMS.  Today's app, binary SigLIP, coco_better, 144 cells x 5 seeds, 150
# votes, the user's pool at 1%, beta 1 (as #4583's shipped arm):
#   dither  - CALIB_SPLIT_ROUNDING=dither (the app when the study ran, e61fedcb7)
#   nearest - the app since #4864 shipped (CALIB_SPLIT_ROUNDING unset; the arm
#             CALIB_SPLIT_ROUNDING=nearest when the study ran)
# REGION.  siglip+dinov3_patch, max_patch, beta 1, #4219's 72 classes x 2
# seeds, the patch space's shipped 0.5 split (as #4583's region arms):
#   r_dither, r_nearest.  Its prepare is #4583's (same embedders, same classes).
#
# Both arms of a pair run on one commit: launch from a frozen run worktree
# (VTS_REPO), never the one being edited.
#
# BUNDLED TASKS.  Each binary array task runs BUNDLE cells: task j runs cells
# j, j + T, j + 2T, ... with T = 720 / BUNDLE, so a lost task loses cells
# spread over every seed and class.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4864-run}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${SPLITROUND_BASE:-/expscratch/$USER/splitround-4864}"
# The #4519 1% grid's prepare serves this pool as-is (prepare never reads the
# split; thinning happens inside a cell), as it did for #4583.
LADDER="${LADDER_BASE:-/expscratch/$USER/progression-4184-h0.01}"
REGION_PREPARE="${REGION_PREPARE:-/expscratch/$USER/calsplit-4583/region/prepare}"

ALL_ARMS="dither nearest"
REGION_ARMS="r_dither r_nearest"

# --- science knobs: production except the one swept here ------------------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_LIVE_THRESHOLD CALIB_ACQ_INCLUSION_OFFSET CALIB_SPOT_CHECK CALIB_MIN_PRECISION
unset CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT CALIB_BETA CALIB_SPLIT_ROUNDING
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""
export CALIB_HAYSTACK_PREVALENCE=0.01
# Recorded only: the class model's held-out Goods at every click (`model_n_pos`), so
# each arm's switches between the count line and the labels line can be counted.
export CALIB_LINE_COUNTS=1

# --- environment (as #4583) -------------------------------------------------------
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all
export CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"

# --- ops -----------------------------------------------------------------------
# #4583's shipped arm (890647): a bundle of 8 took 26-47 min, < 1.2 GB.
BUNDLE="${BUNDLE:-8}"
MEM="${CALIB_MEM:-3G}"
TIME="${CALIB_TIME:-4:00:00}"
CONC="${CALIB_CONC:-45}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Pinned BLAS kernels so a run does not depend on its node's CPU (#4481).
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"
# A line-only knob's sigma at beta 1: #4583's paired split/count contrasts (0.030-0.035).
SIGMA="${SPLITROUND_SIGMA:-0.035}"
RESOLVE="${SPLITROUND_RESOLVE:-0.003}"

# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  [[ "$1" =~ ^[0-9]+$ ]] || {
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  }
}

arm_env() { # sets the knob and the declared divergence for arm $1
  # Since #4864 shipped, nearest is the app (unset) and the dither is the arm.
  unset CALIB_SPLIT_ROUNDING
  ARM_DIVERGES=""
  case "$1" in
    dither | r_dither) export CALIB_SPLIT_ROUNDING=dither && ARM_DIVERGES=split_rounding ;;
    nearest | r_nearest) ;;
    *)
      echo "unknown arm '$1'; expected one of: $ALL_ARMS $REGION_ARMS" >&2
      exit 2
      ;;
  esac
}

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="splitround4864-$1"
  mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
  ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

link_prepare() {
  local src="$LADDER/prepare/results"
  [[ -f "$src/prepare_info.json" ]] || {
    echo "ERROR: no prepare at $src" >&2
    exit 1
  }
  cp -n "$src/prepare_info.json" "$CALIB_RESULTS/"
  [[ -d "$src/crops" && ! -e "$CALIB_RESULTS/crops" ]] && cp -r "$src/crops" "$CALIB_RESULTS/crops"
  return 0
}

grid_size() {
  (cd "$HERE" && source "$WT/scripts/grid/gridenv.sh" >/dev/null 2>&1 && eval "$ENVX" && python run_cells.py --print-cells 2>/dev/null | tail -1)
}

run_preflight() { # n_array_tasks
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  local div=()
  [[ -n "$ARM_DIVERGES" ]] && div=(--diverges "$ARM_DIVERGES")
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 3 "${div[@]}" \
    --job-name "$CALIB_JOB_NAME" --mem "$MEM" --conc "$CONC" --array-tasks "$1" \
    --resolve-delta "$RESOLVE" --sigma "$SIGMA" --reuse-prepare "$LADDER/prepare/results" || {
    echo "preflight FAILED ($CALIB_JOB_NAME)" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
}

submit_bundled() { # n_cells n_tasks array_spec
  local n="$1" T="$2" spec="$3"
  local body="for k in \$(seq 0 $((BUNDLE - 1))); do i=\$((SLURM_ARRAY_TASK_ID + k * $T)); [ \$i -lt $n ] || continue; python run_cells.py --index \$i --outdir $CALIB_RESULTS/cells || echo \"cell \$i FAILED\"; done"
  sbatch --parsable --job-name="$CALIB_JOB_NAME" --array="$spec" --mem="$MEM" --cpus-per-task=1 \
    --time="$TIME" --partition=cpu --export=ALL --output="$CALIB_EXP/logs/cells-%A_%a.out" \
    --wrap="source $WT/scripts/grid/gridenv.sh && $ENVX && cd $HERE && $body"
}

region_env() {
  export CALIB_COCO_BETTER_EMBEDDERS=siglip+dinov3_patch
  export CALIB_CATEGORY_FILE="$WT/docs/experiments/2026-09-28-head-switch-4219/region_categories.txt"
  export CALIB_N_SEEDS="${REGION_SEEDS:-2}"
  arm_env "$1"
  BASE="$BASE/region"
  LADDER="${REGION_PREPARE%/prepare}"
  # #4583's region arm (891256): 25 min mean, 66 min max, MaxRSS 21 GB.
  MEM="${REGION_MEM:-24G}"
  TIME="${REGION_TIME:-3:00:00}"
  CONC="${REGION_CONC:-16}"
  # #4583's region pair: sd of the paired objective 0.026 over votes 1-150 at 143 runs.
  SIGMA="${REGION_SIGMA:-0.026}"
  RESOLVE="${REGION_RESOLVE:-0.0045}"
  BUNDLE=1
}

case "$MODE" in
  plan)
    for a in $ALL_ARMS $REGION_ARMS; do
      arm_env "$a"
      printf '%-10s split_rounding=%-8s diverges=%s\n' "$a" "${CALIB_SPLIT_ROUNDING:-dither}" "${ARM_DIVERGES:-none}"
    done
    echo "binary: bundle=$BUNDLE cells/task, conc=%$CONC per arm, mem=$MEM, time=$TIME"
    echo "repo=$WT ($(git -C "$WT" rev-parse --short HEAD))"
    ;;

  size)
    ARM="${2:-nearest}"
    arm_env "$ARM"
    set_exp "sizing-$ARM"
    link_prepare
    N=$(grid_size)
    [[ "$N" =~ ^[0-9]+$ ]] || {
      echo "ERROR: cell count '$N'" >&2
      exit 1
    }
    T=$(((N + BUNDLE - 1) / BUNDLE))
    J=$(submit_bundled "$N" "$T" "0-0")
    require_jobid "$J" "size"
    echo "size job $J: $BUNDLE cells of $ARM -> $CALIB_EXP/logs; then sacct -j $J --format=Elapsed,MaxRSS,State"
    ;;

  arms)
    shift
    ARMS="${*:-$ALL_ARMS}"
    for a in $ARMS; do
      (
        arm_env "$a"
        set_exp "$a"
        link_prepare
        N=$(grid_size)
        [[ "$N" =~ ^[0-9]+$ && "$N" -gt 0 ]] || {
          echo "ERROR: cell count '$N' for $a" >&2
          exit 1
        }
        T=$(((N + BUNDLE - 1) / BUNDLE))
        run_preflight "$T"
        J=$(submit_bundled "$N" "$T" "0-$((T - 1))%$CONC")
        require_jobid "$J" "$a"
        echo "$a: $N cells in $T tasks of $BUNDLE -> job $J"
      ) || exit 1
    done
    ;;

  status)
    for a in $ALL_ARMS; do
      d="$BASE/$a/results/cells"
      n="$(find "$d" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -r -n "splitround4864-$a" -o %i 2>/dev/null | wc -l)"
      printf '%-9s %5s cells written  %3s tasks queued/running\n' "$a" "$n" "$q"
    done
    ;;

  region-size)
    region_env r_nearest
    set_exp sizing
    link_prepare
    J=$(sbatch --parsable --job-name=splitround4864-region-size --mem="$MEM" --cpus-per-task=1 --time="$TIME" \
      --partition=cpu --export=ALL --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/scripts/grid/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index 0 --outdir $CALIB_RESULTS/cells")
    require_jobid "$J" "region size"
    echo "region size job $J; then sacct -j $J --format=Elapsed,MaxRSS,State"
    ;;

  region)
    for a in $REGION_ARMS; do
      (
        region_env "$a"
        set_exp "$a"
        link_prepare
        N=$(grid_size)
        [[ "$N" =~ ^[0-9]+$ && "$N" -gt 0 ]] || {
          echo "ERROR: cell count '$N' for $a" >&2
          exit 1
        }
        run_preflight "$N"
        J=$(submit_bundled "$N" "$N" "0-$((N - 1))%$CONC")
        require_jobid "$J" "$a"
        echo "$a: $N cells, one per task -> job $J"
      ) || exit 1
    done
    ;;

  region-status)
    for a in $REGION_ARMS; do
      d="$BASE/region/$a/results/cells"
      n="$(find "$d" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -r -n "splitround4864-$a" -o %i 2>/dev/null | wc -l)"
      printf '%-9s %4s cells written  %3s tasks queued/running\n' "$a" "$n" "$q"
    done
    ;;

  *)
    echo "usage: $0 {plan|size [ARM]|arms [ARM..]|status|region-size|region|region-status}" >&2
    exit 2
    ;;
esac
