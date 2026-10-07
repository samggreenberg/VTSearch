#!/usr/bin/env bash
# #3546: what Autopilot's acquisition cut should be, priced on the objective at
# the three balance presets.
#
#   bash launch_acqcut_3546.sh plan             # every arm's knobs; submits nothing
#   bash launch_acqcut_3546.sh arms [ARM..]     # submit arm arrays (default: all)
#   bash launch_acqcut_3546.sh status           # cells written per arm
#
# WHY.  The shipped cut, threshold_at(inclusion_for_threshold(line) - 4), is a
# rank pin nobody chose: under the balance it sits near the 98.5th pool
# percentile at every preset, flat from vote 50 to 150 (measured 2026-10-07 on
# #4583's shipped arms, posted on #3546).  This prices the alternatives.
#
# ARMS, each at beta 1/4, 1 and 4 (arm names <rule>_b025 / _b1 / _b4):
#   ctl     today's cut: line - 4 Inclusion steps (the control)
#   line    sampling at the line itself (offset 0)
#   off2    line - 2 steps
#   off6    line - 6 steps        (#4581: the offset's own value, -4 chosen on a cost plateau)
#   pin985  a fixed rank pin at the 98.5th percentile: is today's cut "just a pin"?
#   inc0    Inclusion 0 - 4, the old origin (#4333's floor-off arm)
#   tp50    target pick precision 0.5: sample where the labels line's posterior crosses 1/2
#   tp25    target pick precision 0.25 (today's Hard picks are ~22% Good)
#
# Today's app otherwise (the balance, the labels line, the weak check), binary
# SigLIP on coco_better, 144 cells, 150 votes, the user's pool at 1% as #4519 /
# #4583, the withheld half at the bench's 0.44%.  THREE seeds: a screen.  The
# objective's sigma for an acquisition change is 0.08 at beta 1 (#4584), so 432
# runs resolve ~0.008 at beta 1 and ~0.012 at beta 1/4; the #4428 arms differed
# by 0.05-0.15.  Seeds are added to the contenders if the screen is close.
#
# Eight cells per array task (the 2,000-task submit cap): task j runs cells j,
# j+T, j+2T, ... so a lost task loses cells spread over every class and seed.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-3546}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${ACQCUT_BASE:-/expscratch/$USER/acqcut-3546}"
# The #4519 1% grid's prepare serves this pool as-is.
LADDER="${LADDER_BASE:-/expscratch/$USER/progression-4184-h0.01}"

RULES="ctl line off2 off6 pin985 inc0 tp50 tp25"
BETAS="b025 b1 b4"
ALL_ARMS=""
for r in $RULES; do for b in $BETAS; do ALL_ARMS="$ALL_ARMS ${r}_$b"; done; done
ALL_ARMS="${ALL_ARMS# }"

# --- science knobs: production except the acquisition cut and beta ----------------
export CALIB_SAFE_THRESHOLDS=1
unset CALIB_HEAD CALIB_BLEND_SCHEDULE CALIB_EXCLUDE_VOTED CALIB_CALIBRATION_SEEDS CALIB_LIVE_CUT_RULE
unset CALIB_LIVE_THRESHOLD CALIB_SPOT_CHECK CALIB_MIN_PRECISION CALIB_CALIBRATION_FRACTION CALIB_CALIBRATE_COUNT
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_REPOOL_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_ANCHORED=0
export CALIB_CUT_INCL_KS=""
export CALIB_HAYSTACK_PREVALENCE=0.01

# --- environment (as #4583) --------------------------------------------------------
export CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all
export CALIB_PATCH_STYLES=max_patch
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-3}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"

# --- ops -----------------------------------------------------------------------
BUNDLE="${BUNDLE:-8}"
MEM="${CALIB_MEM:-3G}"
TIME="${CALIB_TIME:-4:00:00}"
# 24 arms x %5 = 120 tasks, 240 charged CPUs: the whole cap.  Lower it if a peer runs.
CONC="${CALIB_CONC:-5}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export OPENBLAS_CORETYPE="${OPENBLAS_CORETYPE:-Haswell}"

# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  [[ "$1" =~ ^[0-9]+$ ]] || {
    echo "ERROR: $2 was REFUSED by sbatch (no job id came back)." >&2
    exit 1
  }
}

arm_env() { # sets the arm's knobs and the divergences preflight must see declared
  local a="$1" rule b
  rule="${a%_*}"
  b="${a##*_}"
  unset CALIB_ACQ_INCLUSION_OFFSET CALIB_ACQ_RANK_PERCENTILE CALIB_ACQ_ORIGIN CALIB_ACQ_TARGET_P CALIB_BETA
  ARM_DIVERGES=""
  case "$b" in
    b025) export CALIB_BETA=0.25 && ARM_DIVERGES="beta" ;;
    b1) ;;
    b4) export CALIB_BETA=4 && ARM_DIVERGES="beta" ;;
    *) echo "unknown arm '$a'; expected one of: $ALL_ARMS" >&2 && exit 2 ;;
  esac
  case "$rule" in
    ctl) ;;
    line) export CALIB_ACQ_INCLUSION_OFFSET=0 && ARM_DIVERGES="$ARM_DIVERGES,acq_offset" ;;
    off2) export CALIB_ACQ_INCLUSION_OFFSET=-2 && ARM_DIVERGES="$ARM_DIVERGES,acq_offset" ;;
    off6) export CALIB_ACQ_INCLUSION_OFFSET=-6 && ARM_DIVERGES="$ARM_DIVERGES,acq_offset" ;;
    pin985)
      export CALIB_ACQ_INCLUSION_OFFSET=0 CALIB_ACQ_RANK_PERCENTILE=0.985
      ARM_DIVERGES="$ARM_DIVERGES,acq_offset,acq_rank_percentile"
      ;;
    inc0) export CALIB_ACQ_ORIGIN=inclusion && ARM_DIVERGES="$ARM_DIVERGES,acq_origin" ;;
    tp50 | tp25)
      export CALIB_ACQ_INCLUSION_OFFSET=0 CALIB_ACQ_TARGET_P="0.${rule#tp}"
      ARM_DIVERGES="$ARM_DIVERGES,acq_offset,acq_target_p"
      ;;
    *) echo "unknown arm '$a'; expected one of: $ALL_ARMS" >&2 && exit 2 ;;
  esac
  ARM_DIVERGES="${ARM_DIVERGES#,}"
  return 0
}

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="acqcut3546-$1"
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
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 2 "${div[@]}" \
    --job-name "$CALIB_JOB_NAME" --mem "$MEM" --conc "$CONC" || {
    echo "preflight FAILED ($CALIB_JOB_NAME)" >&2
    [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
}

submit_bundled() { # n_cells n_tasks array_spec
  local n="$1" T="$2" spec="$3"
  local body="for k in \$(seq 0 $((BUNDLE - 1))); do i=\$((SLURM_ARRAY_TASK_ID + k * $T)); [ \$i -lt $n ] || continue; python run_cells.py --index \$i --outdir $CALIB_RESULTS/cells || echo \"cell \$i FAILED\"; done"
  sbatch --parsable --job-name="$CALIB_JOB_NAME" --array="$spec" --mem="$MEM" --cpus-per-task=1 \
    --time="$TIME" --partition=cpu --export=ALL --output="$CALIB_EXP/logs/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && $body"
}

case "$MODE" in
  plan)
    for a in $ALL_ARMS; do
      arm_env "$a"
      printf '%-12s beta=%-4s offset=%-4s pin=%-5s origin=%-9s target_p=%-4s diverges=%s\n' "$a" \
        "${CALIB_BETA:-1}" "${CALIB_ACQ_INCLUSION_OFFSET:--4}" "${CALIB_ACQ_RANK_PERCENTILE:--}" \
        "${CALIB_ACQ_ORIGIN:-line}" "${CALIB_ACQ_TARGET_P:--}" "${ARM_DIVERGES:-none}"
    done
    echo "seeds=$CALIB_N_SEEDS bundle=$BUNDLE cells/task conc=%$CONC per arm mem=$MEM time=$TIME"
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
      d="$BASE/$a/results/cells"
      n="$(find "$d" -maxdepth 1 -name 'task_[0-9][0-9][0-9][0-9].csv*' -size +0 2>/dev/null | wc -l)"
      q="$(squeue -u "$USER" -h -r -n "acqcut3546-$a" -o %i 2>/dev/null | wc -l)"
      printf '%-12s %5s cells written  %3s tasks queued/running\n' "$a" "$n" "$q"
    done
    ;;

  *)
    echo "usage: $0 {plan|arms [ARM..]|status}" >&2
    exit 2
    ;;
esac
