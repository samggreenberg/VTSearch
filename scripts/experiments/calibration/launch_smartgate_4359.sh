#!/usr/bin/env bash
# #4359: how much objective could a later Smart light buy?  The bound.
#
#   bash launch_smartgate_4359.sh plan | arms [ARM..] | status
#
# WHY.  Smart (FPR + FNR at each model's Inclusion 0 cut) ignores the preset and
# goes green at a median vote 42 on today's app, after which runs still gain
# 0.15-0.17 of the objective (measured 2026-10-07 on #3546's tp50 arms, posted
# on #4359).  Smart and Stable both green end Autopilot's Hard phase (Hard ->
# New, then Done with Span).  Before re-keying Smart, this measures the most a
# later gate could buy: today's app against a session that never leaves Hard.
#
# ARMS at beta 1/4, 1, 4: app_<beta> (today's app) and never_<beta>
# (CALIB_SMART_GATE=never: Smart held yellow for the phase decision).  Today's
# app otherwise (the target-precision acquisition cut, #3546), binary SigLIP,
# coco_better, 144 cells x 3 seeds, 150 votes, 1% pool, as #3546's screen.
#
# Eight cells per array task (the 2,000-task submit cap).
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- check what was submitted" >&2' ERR

MODE="${1:-}"

export VTS_REPO="${VTS_REPO:-/expscratch/$USER/worktrees/vts-4359}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"
BASE="${SMARTGATE_BASE:-/expscratch/$USER/smartgate-4359}"
# The #4519 1% grid's prepare serves this pool as-is.
LADDER="${LADDER_BASE:-/expscratch/$USER/progression-4184-h0.01}"

RULES="app never"
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
MEM="${CALIB_MEM:-2G}"  # a 1% cell peaks under 0.8 GB (#4548); 3G cost a third of the memory-capped slots
TIME="${CALIB_TIME:-4:00:00}"
# 6 arms x %18 = 108 tasks, 216 charged CPUs.  Lower it if a peer runs.
CONC="${CALIB_CONC:-18}"
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
  unset CALIB_SMART_GATE CALIB_BETA CALIB_ACQ_INCLUSION_OFFSET CALIB_ACQ_RANK_PERCENTILE CALIB_ACQ_ORIGIN CALIB_ACQ_TARGET_P
  ARM_DIVERGES=""
  case "$b" in
    b025) export CALIB_BETA=0.25 && ARM_DIVERGES="beta" ;;
    b1) ;;
    b4) export CALIB_BETA=4 && ARM_DIVERGES="beta" ;;
    *) echo "unknown arm '$a'; expected one of: $ALL_ARMS" >&2 && exit 2 ;;
  esac
  case "$rule" in
    app) ;;
    never) export CALIB_SMART_GATE=never && ARM_DIVERGES="$ARM_DIVERGES,smart_gate" ;;
    *) echo "unknown arm '$a'; expected one of: $ALL_ARMS" >&2 && exit 2 ;;
  esac
  ARM_DIVERGES="${ARM_DIVERGES#,}"
  return 0
}

set_exp() {
  export CALIB_EXP="$BASE/$1"
  export CALIB_RESULTS="$CALIB_EXP/results"
  export CALIB_JOB_NAME="smartgate4359-$1"
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
      printf '%-10s beta=%-4s smart_gate=%-5s diverges=%s\n' "$a" "${CALIB_BETA:-1}" "${CALIB_SMART_GATE:-app}" "${ARM_DIVERGES:-none}"
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
      q="$(squeue -u "$USER" -h -r -n "smartgate4359-$a" -o %i 2>/dev/null | wc -l)"
      printf '%-12s %5s cells written  %3s tasks queued/running\n' "$a" "$n" "$q"
    done
    ;;

  *)
    echo "usage: $0 {plan|arms [ARM..]|status}" >&2
    exit 2
    ;;
esac
