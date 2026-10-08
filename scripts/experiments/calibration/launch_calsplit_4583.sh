#!/usr/bin/env bash
# #4583: the calibration split and fold count, re-measured under the labels line
# at the three balance presets.
#
#   bash launch_calsplit_4583.sh plan             # every arm's knobs; submits nothing
#   bash launch_calsplit_4583.sh size [ARM]       # time ONE bundle of cells (default f03k2_b1)
#   bash launch_calsplit_4583.sh arms [ARM..]     # submit arm arrays (default: all twelve)
#   bash launch_calsplit_4583.sh status           # cells written per arm
#
# WHY.  `PRODUCTION_SPLIT_BY_SPACE` (0.3 single-vector, 0.5 patch; #3287) and
# `DEFAULT_CALIBRATE_COUNT` (2; #2897, #3314) were chosen on paired Δcost of the
# mixture line.  Since #4452 the line is the labels line, whose class model IS
# the folds' held-out scores.  #4582 re-scored the old runs at F-beta and found
# the split flips on `dinov3_patch` at beta <= 1 and that one fold beats two at
# beta 1/4 - above the mixture line.  This measures both knobs above today's.
#
# ARMS.  Today's app (the balance, the labels line, the weak check), binary
# SigLIP, coco_better, 144 cells x 5 seeds, 150 votes, the user's pool at 1%
# (as #4519 / #4548), the withheld half at the bench's own 0.44%:
#   calibration fraction {0.3 = shipped for SigLIP, 0.5} x calibrate count
#   {2 = shipped, 4} x beta {0.25, 1, 4}.  Arm names: f03k2_b025 ... f05k4_b4.
# Every arm is its own trajectory (the knobs move the line, the line moves the
# questions); arms pair on (category, seed), same commit for all twelve.
#
# REGION (phase 2).  The issue's region contrast: siglip+dinov3_patch, max_patch,
# beta 1, the shipped 0.5 (the patch space's default) against 0.3, which #4582's
# re-score found better at beta <= 1 above the old line.  #4219's 72 classes
# (the hardest and easiest quartiles) x 2 seeds, because a region cell costs
# ~2 h and ~17 GB (#4219).  Its own prepare; one cell per task.
#   bash launch_calsplit_4583.sh region-prepare   # ONCE
#   bash launch_calsplit_4583.sh region-size      # time one cell of r_f05_b1
#   bash launch_calsplit_4583.sh region           # both arms
#   bash launch_calsplit_4583.sh region-status
#
# BUNDLED TASKS.  12 x 720 = 8,640 cells is over the per-user 2,000-task submit
# cap (MaxSubmitPU counts array tasks).  Each array task runs BUNDLE cells
# sequentially: task j of an arm runs cells j, j + T, j + 2T, ... with T = 720 /
# BUNDLE, so a lost task loses cells spread over every seed and class, never a
# block of one class.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4583}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${CALSPLIT_BASE:-/expscratch/$USER/calsplit-4583}"
# The #4519 1% grid: its prepare and its click-0 baseline serve this pool as-is
# (prepare never reads the knobs swept here; thinning happens inside a cell).
LADDER="${LADDER_BASE:-/expscratch/$USER/progression-4184-h0.01}"

FRACTIONS="0.3 0.5"
COUNTS="2 4"
BETAS="0.25 1 4"

arm_name() { # fraction count beta -> f03k2_b025
  local f="${1/./}" b="${3/./}"
  echo "f${f}k${2}_b${b}"
}
ALL_ARMS=""
for f in $FRACTIONS; do for k in $COUNTS; do for b in $BETAS; do
  ALL_ARMS="$ALL_ARMS $(arm_name "$f" "$k" "$b")"
done; done; done
ALL_ARMS="${ALL_ARMS# }"

# --- science knobs: production except the three swept here ---------------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_LIVE_THRESHOLD CALIB_ACQ_INCLUSION_OFFSET CALIB_SPOT_CHECK CALIB_MIN_PRECISION
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""
export CALIB_HAYSTACK_PREVALENCE=0.01

# --- environment (as #4184 / #4519) ----------------------------------------------
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
# A 1% cell took 4-9 min and < 1 GB on #4548's r8 arms (887706), so a bundle of
# 8 is ~35-70 min; K=4 fits twice the folds, and 8 x 18 min still fits the limit.
BUNDLE="${BUNDLE:-8}"
MEM="${CALIB_MEM:-3G}"
TIME="${CALIB_TIME:-4:00:00}"
# 12 arms x %6 = 72 tasks, 144 charged CPUs of the 240 cap: room for a peer study.
CONC="${CALIB_CONC:-6}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# Pinned BLAS kernels so a run does not depend on its node's CPU (#4481).
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"

# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  [[ "$1" =~ ^[0-9]+$ ]] || {
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  }
}

arm_env() { # sets the three knobs and the declared divergences for arm $1
  local a="$1" f k b
  [[ "$a" =~ ^f(0[0-9])k([0-9]+)_b([0-9]+)$ ]] || {
    echo "unknown arm '$a'; expected one of: $ALL_ARMS" >&2
    exit 2
  }
  f="0.${BASH_REMATCH[1]#0}"
  k="${BASH_REMATCH[2]}"
  b="${BASH_REMATCH[3]}"
  [[ "$b" == "025" ]] && b=0.25
  # A shipped value is left UNSET, so it resolves exactly as production does
  # (0.3 is SigLIP's per-space default, 2 the app's count, 1 the default
  # balance) and preflight sees only the knobs an arm moves.
  unset CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT CALIB_BETA
  ARM_F="$f" ARM_K="$k" ARM_B="$b"
  ARM_DIVERGES=""
  [[ "$f" != "0.3" ]] && export CALIB_CALIBRATION_FRACTION="$f" && ARM_DIVERGES="$ARM_DIVERGES,calibration_fraction"
  [[ "$k" != "2" ]] && export CALIB_CALIBRATE_COUNT="$k" && ARM_DIVERGES="$ARM_DIVERGES,calibrate_count"
  [[ "$b" != "1" ]] && export CALIB_BETA="$b" && ARM_DIVERGES="$ARM_DIVERGES,beta"
  ARM_DIVERGES="${ARM_DIVERGES#,}"
  return 0
}

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="calsplit4583-$1"
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
  (cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1 && eval "$ENVX" && python run_cells.py --print-cells 2>/dev/null | tail -1)
}

run_preflight() {
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  local div=()
  [[ -n "$ARM_DIVERGES" ]] && div=(--diverges "$ARM_DIVERGES")
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 3 "${div[@]}" \
    --job-name "$CALIB_JOB_NAME" --mem "$MEM" --conc "$CONC" || {
    echo "preflight FAILED ($CALIB_JOB_NAME)" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
}

# One task = BUNDLE cells, strided by the task count so every seed and class is
# spread over every task.
submit_bundled() { # arm n_tasks array_spec
  local n="$1" T="$2" spec="$3"
  local body="for k in \$(seq 0 $((BUNDLE - 1))); do i=\$((SLURM_ARRAY_TASK_ID + k * $T)); [ \$i -lt $n ] || continue; python run_cells.py --index \$i --outdir $CALIB_RESULTS/cells || echo \"cell \$i FAILED\"; done"
  sbatch --parsable --job-name="$CALIB_JOB_NAME" --array="$spec" --mem="$MEM" --cpus-per-task=1 \
    --time="$TIME" --partition=cpu --export=ALL --output="$CALIB_EXP/logs/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && $body"
}

# --- region (phase 2) -------------------------------------------------------------
REGION_ARMS="r_f05_b1 r_f03_b1"
region_env() {
  export CALIB_COCO_BETTER_EMBEDDERS=siglip+dinov3_patch
  export CALIB_CATEGORY_FILE="$WT/docs/experiments/2026-09-28-head-switch-4219/region_categories.txt"
  export CALIB_N_SEEDS="${REGION_SEEDS:-2}"
  unset CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT CALIB_BETA
  ARM_DIVERGES=""
  case "${1:-}" in
    r_f05_b1 | prepare | sizing) ;;
    r_f03_b1) export CALIB_CALIBRATION_FRACTION=0.3 && ARM_DIVERGES=calibration_fraction ;;
    *) echo "unknown region arm '$1'; expected one of: $REGION_ARMS" >&2 && exit 2 ;;
  esac
  BASE="$BASE/region"
  LADDER="$BASE"
  MEM="${REGION_MEM:-20G}"
  TIME="${REGION_TIME:-5:00:00}"
  CONC="${REGION_CONC:-25}"
  BUNDLE=1
}

case "$MODE" in
  region-prepare)
    region_env prepare
    set_exp prepare
    P=$(sbatch --parsable --job-name=calsplit4583-region-prep --mem=32G --cpus-per-task=2 \
      --time=3:00:00 --partition=cpu --export=ALL --output="$CALIB_EXP/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "region prepare"
    echo "region prepare job: $P -> $CALIB_EXP/logs/prepare-$P.out"
    ;;

  region-size)
    region_env sizing
    set_exp sizing
    link_prepare
    J=$(sbatch --parsable --job-name=calsplit4583-region-size --mem="$MEM" --cpus-per-task=1 --time="$TIME" \
      --partition=cpu --export=ALL --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index 0 --outdir $CALIB_RESULTS/cells")
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
        run_preflight
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
      printf '%-9s %4s cells written\n' "$a" "$n"
    done
    ;;

  plan)
    for a in $ALL_ARMS; do
      arm_env "$a"
      printf '%-11s fraction=%-4s count=%s beta=%-4s diverges=%s\n' "$a" "$ARM_F" "$ARM_K" "$ARM_B" "${ARM_DIVERGES:-none}"
    done
    echo "bundle=$BUNDLE cells/task, conc=%$CONC per arm, mem=$MEM, time=$TIME"
    ;;

  size)
    ARM="${2:-f03k2_b1}"
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
        run_preflight
        J=$(submit_bundled "$N" "$T" "0-$((T - 1))%$CONC")
        require_jobid "$J" "$a"
        echo "$a: $N cells in $T tasks of $BUNDLE -> job $J"
      ) || exit 1
    done
    ;;

  status)
    for a in $ALL_ARMS; do
      arm_env "$a"
      d="$BASE/$a/results/cells"
      n="$(find "$d" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -r -n "calsplit4583-$a" -o %i 2>/dev/null | wc -l)"
      printf '%-11s %5s cells written  %3s tasks queued/running\n' "$a" "$n" "$q"
    done
    ;;

  *)
    echo "usage: $0 {plan|size [ARM]|arms [ARM..]|status|region-prepare|region-size|region|region-status}" >&2
    exit 2
    ;;
esac
