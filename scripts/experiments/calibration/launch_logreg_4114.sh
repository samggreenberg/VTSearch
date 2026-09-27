#!/usr/bin/env bash
# #4114: does a converged logistic head beat the shipped SVM head IN THE LOOP?
#
#   bash launch_logreg_4114.sh prepare          # ONCE (cpu, reads the pile in place)
#   bash launch_logreg_4114.sh baseline         # the click-0 text anchor, shared by every arm
#   bash launch_logreg_4114.sh size [idx] [ARM] # time ONE cell (default: cell 0, lrconv)
#   bash launch_logreg_4114.sh arms [ARM..]     # submit arm arrays (default: all three)
#   bash launch_logreg_4114.sh status           # cells written per arm, against the grid size
#
# #3197 found that refitted on the SAME Autopilot vote sets, balanced L2 logistic
# regression fitted to convergence at C = 1 ranks as well as the shipped SVM or
# slightly better, and is more robust to mis-votes.  What lost to the SVM was the
# old logistic head's FIT (Adam, early-stopped), not its loss.  In the loop the
# head also picks the next vote and the shipped cut was tuned on other heads'
# score distributions, so the replay result does not carry by itself.
#
# Arms (each is its own trajectory, paired with the others on the CELL):
#
#   svm      the SHIPPED head (CALIB_HEAD unset -> PRODUCTION_HEAD, LinearSVC, C = 1)
#   lrconv   LogisticRegression(C = 1, balanced, L2) to convergence, lifted into
#            Linear(D, 1) (`LINEAR_LOGREG_HEAD`, vtscore/training/logreg.py)
#   linear   the early-stopped logistic head the SVM replaced: the fidelity arm,
#            so #3197's svm - linear gap can be checked on this bench
#
# Environment: COCO Better (owner, 2026-09-27: current data, not vg_scale or the
# #3197 pile), the #4184 bench - every class@band cell, SigLIP, binary voting,
# today's production for everything but the head.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4114}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${LOGREG_BASE:-/expscratch/$USER/logreg-4114}"

ALL_ARMS="svm lrconv linear"

# --- science knobs -------------------------------------------------------------
# The shipped threshold path.  Everything but the head is unset and resolves to
# production; CALIB_HEAD is set per arm below and nowhere else.
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_LIVE_THRESHOLD CALIB_CALIBRATION_FRACTION CALIB_ACQ_INCLUSION_OFFSET
unset VTSEARCH_SVM_HEAD_C VTSEARCH_TRAIN_EPOCHS VTSEARCH_TRAIN_PATIENCE
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""

# --- environment ---------------------------------------------------------------
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all
export CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1

# --- sizing --------------------------------------------------------------------
# 144 cells x 5 seeds = 720 paired cells per arm: paired SE ~0.04/sqrt(720) ~
# 0.0015, so #3197's replay gap (-0.005 to -0.009 oracle cost) resolves at 2 SE.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

# --- ops -----------------------------------------------------------------------
# #4184 measured r7_acq4 (this svm arm) at ~6 min and 1.6 GB per cell.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_CPUS=1
export CALIB_MEM="${CALIB_MEM:-3G}"
export CALIB_TIME="${CALIB_TIME:-1:00:00}"
export CALIB_CONC="${CALIB_CONC:-30}"
export CALIB_ANALYZE=noop.py
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

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
  export CALIB_JOB_NAME="logreg4114-$1"
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

arm_env() {
  unset CALIB_HEAD
  ARM_DIVERGES=""
  case "$1" in
    svm) ;;  # CALIB_HEAD left UNSET on purpose: resolves to PRODUCTION_HEAD
    lrconv) export CALIB_HEAD=linear_logreg; ARM_DIVERGES="head" ;;
    linear) export CALIB_HEAD=linear; ARM_DIVERGES="head" ;;
    *) echo "unknown arm '$1'; expected one of: $ALL_ARMS" >&2; exit 2 ;;
  esac
  return 0
}

activate_venv() {
  # preflight imports vtscore; the system python cannot.
  # shellcheck disable=SC1091
  source "$WT/gridenv.sh" >/dev/null 2>&1 || { echo "ERROR: no venv via $WT/gridenv.sh" >&2; exit 1; }
}

run_preflight() {
  activate_venv
  [[ -x "$WT/scripts/experiments/preflight.sh" ]] || return 0
  local div=()
  [[ -n "$ARM_DIVERGES" ]] && div=(--diverges "$ARM_DIVERGES")
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 10 \
    "${div[@]}" --job-name "$CALIB_JOB_NAME" --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
    echo "preflight FAILED ($CALIB_JOB_NAME)" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
}

case "$MODE" in
  prepare)
    set_exp prepare
    P=$(sbatch --parsable --job-name=logreg4114-prep --mem=32G --cpus-per-task=2 \
      --time=1:30:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P -> $CALIB_EXP/logs/prepare-$P.out"
    ;;

  baseline)
    set_exp prepare
    T=$(sbatch --parsable --job-name=logreg4114-baseline --mem=32G --cpus-per-task=2 \
      --time=1:00:00 --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/baseline-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $CALIB_RESULTS --out $BASE/text_baseline.csv")
    require_jobid "$T" "text baseline"
    echo "baseline job: $T -> $BASE/text_baseline.csv"
    ;;

  size)
    IDX="${2:-0}"
    ARM="${3:-lrconv}"
    arm_env "$ARM"
    set_exp "sizing-$ARM"
    link_prepare
    S=$(sbatch --parsable --job-name="logreg4114-size-$ARM-$IDX" --mem="$CALIB_MEM" --cpus-per-task=1 \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$CALIB_EXP/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && /usr/bin/time -v python run_cells.py --index $IDX --outdir $CALIB_RESULTS/cells")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX, $ARM) -> $CALIB_EXP/logs/size-$S.out"
    ;;

  arms)
    shift
    ARMS="${*:-$ALL_ARMS}"
    for arm in $ARMS; do
      (
        arm_env "$arm"
        set_exp "$arm"
        link_prepare
        echo "=== $arm: head=${CALIB_HEAD:-<production>} -> $CALIB_EXP"
        run_preflight
        bash "$HERE/launch_cells.sh"
      ) || exit 1
    done
    ;;

  status)
    for arm in $ALL_ARMS; do
      arm_env "$arm"
      set_exp "$arm"
      n="$(find "$CALIB_RESULTS/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv' -size +0 2>/dev/null | wc -l)"
      z="$(find "$CALIB_RESULTS/cells" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv' -size 0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -n "$CALIB_JOB_NAME" -o %i 2>/dev/null | wc -l)"
      printf '%-8s %5s cells written  %3s zero-byte  %3s queued/running jobs\n' "$arm" "$n" "$z" "$q"
    done
    echo "expected per arm: 144 cells x $CALIB_N_SEEDS seeds = $((144 * CALIB_N_SEEDS))"
    ;;

  *)
    sed -n 2,27p "$0"; exit 2 ;;
esac
