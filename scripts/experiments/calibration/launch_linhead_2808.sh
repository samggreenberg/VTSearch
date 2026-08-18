#!/usr/bin/env bash
# #2808: is the shipped linear head's spike reduction limited by EARLY STOPPING?
#
#   bash launch_linhead_2808.sh prepare   # category selection + exemplar crops (CPU, pile-backed)
#   bash launch_linhead_2808.sh chain     # prepare -> cells -> analysis, all GRID-side
#   bash launch_linhead_2808.sh cells     # the arm arrays (after prepare)
#   bash launch_linhead_2808.sh status
#
# WHY THIS RUN EXISTS.  #2790 credited the linear head with a ~55% (COCO) /
# ~73% (VG) deep-spike reduction, measured with an **sklearn** linear arm.
# #2847's within-codebase 2x2 credited the *torch* head with only 0.79x
# (58.5% -> 46.3%), with the fold-anchored threshold doing the rest.  The
# standing suspect for that gap is that the shipped head is EARLY-STOPPED:
# TRAIN_EPOCHS=200 / TRAIN_PATIENCE=10 leaves it ~0.77 rank-correlated with the
# converged logistic fit (tests_lib/detectors/test_linear_head_fidelity.py has
# to raise epochs to 2000 and disable early-stop to reach that fixed point).
#
# THE ARMS.  All three ride today's production threshold (fold-anchored GMM
# cut); only the head and its training budget move.
#
#   C_mlp         mlp    200/10    reference; #2847's B_mlp_fused (26.5% deep)
#   A_shipped     linear 200/10    TODAY'S PRODUCTION; #2847's D_lin_fused (12.2%)
#   B_converged   linear 2000/0    the fidelity test's convergence conditions
#
# C_mlp is the POSITIVE CONTROL and is deliberately listed first: if the mlp arm
# does not spike here, this harness never showed the phenomenon and no contrast
# below it means anything.  A_shipped is simultaneously the FIDELITY CHECK - it
# should land near #2847's 12.2%, and if it does not, the two studies are not
# measuring the same stack and nothing may be quoted across them.
#
# WHAT THE OUTCOME MEANS.  If B_converged recovers #2790's reduction and
# A_shipped does not, this is a **config** decision (raise TRAIN_EPOCHS / relax
# TRAIN_PATIENCE) with a retrain-latency price to pay, NOT a head decision.  If
# A and B land together, the head's contribution is simply smaller than #2790
# reported and #2847's conclusion stands; say so and close #2808.
#
# THE COMPARISON THIS RUN MUST NOT MAKE.  Do not score these numbers against
# #2790's directly.  That study's `whole` path ran the superseded min-cost
# argmin threshold rule - its own S1 suspect - so a cross-harness delta
# confounds the head with the threshold rule.  Only within-run arm contrasts,
# paired at (dataset, embedder, category, seed), are licensed here.
#
# EMBEDDER CAVEAT (read before quoting #2847's absolute numbers).  #2847 ran
# `siglip2`, a MIDDLE RUNG that the pile deliberately dropped (pile README).
# The pile carries `siglip` (the shipped default) and `siglip2_l`.  This study
# therefore runs BOTH of those and CANNOT reproduce #2847's cell exactly; the
# fidelity check above is an approximate one across an embedder change, and the
# report must say so rather than treating a mismatch as a finding.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/linhead-2808}"

# --- the grid (sizing only) ------------------------------------------------
# VG coverage is issue item 3: #2847 ran COCO only, while #2790's claim spanned
# VG on SigLIP1 AND SigLIP2.  Both live in the pile already, so this is a
# CPU-only array with no GPU prepare.
export CALIB_DATASETS="${CALIB_DATASETS:-coco_val,visual_genome_m}"
export CALIB_COCO_EMBEDDERS="${CALIB_COCO_EMBEDDERS:-siglip,siglip2_l}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip2_l}"
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-100}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-6}"
export CALIB_N_PER_BAND="${CALIB_N_PER_BAND:-2}"

# --- the stack under test --------------------------------------------------
# SAFE_THRESHOLDS=1 is the fold-anchored cut = TODAY'S PRODUCTION.  The harness
# default is "0"; a STUDY default is not a SHIPPED default (#3129), so this is
# set explicitly and baked into ENVX rather than left to the submitting shell.
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=0
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_PATCH_STYLES="max_patch"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_EXP/results"

# Whole-image cells peak ~1.1 GB (GRID-PLAYBOOK measured table) -> 8G, 1 cpu.
# Memory is a per-user QOS quota and it binds against my own jobs; #3146's
# precbench array is live on the cpu partition, so CONC stays modest on purpose
# rather than claiming the allowance and starving it.
MEM="${CALIB_MEM:-8G}"
CPUS="${CALIB_CPUS:-1}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-12}"
# The converged arm trains 10x the epochs with early-stop disabled, so it gets
# its own wall-clock; one budget for both would either kill it or over-reserve
# the cheap arms and backfill slower.
TIME_FAST="${CALIB_TIME_FAST:-4:00:00}"
TIME_SLOW="${CALIB_TIME_SLOW:-12:00:00}"

ENVX="export CALIB_EXP=$CALIB_EXP"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO"
ENVX="$ENVX CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_COCO_EMBEDDERS=$CALIB_COCO_EMBEDDERS CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_N_CATEGORIES=$CALIB_N_CATEGORIES CALIB_N_PER_BAND=$CALIB_N_PER_BAND"
ENVX="$ENVX CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS CALIB_ANCHORED=$CALIB_ANCHORED"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
ENVX="$ENVX CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"

# arm -> "HEAD TRAIN_EPOCHS TRAIN_PATIENCE TIME"
# Control first: the analyzer treats SPIKE_ARMS[0] as the positive control.
ARM_ORDER=(C_mlp A_shipped B_converged)
declare -A ARMS=(
  [C_mlp]="mlp 200 10 $TIME_FAST"
  [A_shipped]="linear 200 10 $TIME_FAST"
  [B_converged]="linear 2000 0 $TIME_SLOW"
)

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job (#2897 lost both arms that way and never noticed).
submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$J" > "$LOGS/.jobid_$name"
    echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) - NOT LAUNCHED" >&2
    return 1
  fi
}

case "${1:-status}" in
prepare)
  echo "CALIB_EXP=$CALIB_EXP"
  echo "datasets=$CALIB_DATASETS seeds=$CALIB_N_SEEDS steps=$CALIB_MAX_STEPS"
  export CALIB_RESULTS="$CALIB_EXP/results"
  submit prepare --job-name=lin2808-prep --mem=96G --cpus-per-task=8 \
    --time=3:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$CALIB_EXP/results && cd $HERE && python prepare_data.py"
  ;;

cells)
  [[ -f "$CALIB_EXP/results/prepare_info.json" ]] || {
    echo "ERROR: no prepare_info.json at $CALIB_EXP/results - prepare has not finished" >&2; exit 1; }

  if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
    bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 4 \
      --job-name lin2808 --mem "$MEM" --conc "$CONC" || {
      echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1; }
  fi

  DEPS=""
  for arm in "${ARM_ORDER[@]}"; do
    read -r head epochs patience atime <<<"${ARMS[$arm]}"
    RES="$CALIB_EXP/results/$arm"
    mkdir -p "$RES/cells"
    # Each arm's runner reads prepare_info + crops from its own results dir.
    ln -sfn "$CALIB_EXP/results/prepare_info.json" "$RES/prepare_info.json"
    ln -sfn "$CALIB_EXP/results/crops" "$RES/crops"

    # Cell count is per-arm identical, but compute it inside the arm's own env
    # so a mis-scoped grid shows up here rather than as a short array.
    N=$(cd "$HERE" && source "$WT/gridenv.sh" >/dev/null 2>&1; \
        eval "$ENVX"; export CALIB_RESULTS="$RES" CALIB_HEAD="$head"; \
        python run_cells.py --print-cells 2>/dev/null | tail -1)
    if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
      echo "ERROR: arm $arm - could not determine cell count (got '$N')" >&2; exit 1
    fi
    echo "=== arm $arm (head=$head epochs=$epochs patience=$patience): $N cells, time=$atime"

    # The epoch knobs are baked into the wrap, NOT inherited via --export=ALL:
    # they are the whole independent variable, and vtscore/config.py reads them
    # once at import, so a var that fails to survive into the job would produce
    # a silently identical arm rather than an error.
    submit "cells_$arm" --job-name="lin2808-$arm" --array=0-$((N-1))%"$CONC" \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$atime" --partition="$PARTITION" \
      --export=ALL --output="$LOGS/cells-$arm-%A_%a.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$RES CALIB_HEAD=$head VTSEARCH_TRAIN_EPOCHS=$epochs VTSEARCH_TRAIN_PATIENCE=$patience && cd $HERE && python run_cells.py"
    J=$(cat "$LOGS/.jobid_cells_$arm")
    DEPS="${DEPS:+$DEPS:}$J"
  done

  # Analysis runs GRID-side on afterany so it fires whether or not every cell
  # succeeded - and so it survives the laptop going away.  It must report what
  # it dropped, which is why it runs even on partial arms.
  submit analyze --dependency=afterany:"$DEPS" --job-name=lin2808-analyze \
    --mem=16G --cpus-per-task=4 --time=1:00:00 --partition=cpu --export=ALL \
    --output="$LOGS/analyze-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && export SPIKE_ARMS=${ARM_ORDER[*]} SPIKE_OUT=$CALIB_EXP/analysis && cd $HERE && python analyze_spikes.py"
  ;;

chain)
  # prepare -> (cells + analysis).  The cell count cannot be known until prepare
  # writes prepare_info.json, so the array submission itself is a dependent job.
  bash "$0" prepare
  P=$(cat "$LOGS/.jobid_prepare")
  submit chain --dependency=afterok:"$P" --job-name=lin2808-chain \
    --mem=4G --cpus-per-task=1 --time=0:30:00 --partition=cpu --export=ALL \
    --output="$LOGS/chain-%j.out" \
    --wrap="cd $HERE && bash launch_linhead_2808.sh cells"
  ;;

status)
  echo "CALIB_EXP=$CALIB_EXP"
  squeue -u "$USER" -o "%.12i %.20j %.9P %.8T %.11M %R" | grep -E "lin2808|JOBID" || echo "(no lin2808 jobs queued)"
  for arm in "${ARM_ORDER[@]}"; do
    n=$(find "$CALIB_EXP/results/$arm/cells" -name '*.csv' -size +0 2>/dev/null | wc -l)
    z=$(find "$CALIB_EXP/results/$arm/cells" -name '*.csv' -size 0 2>/dev/null | wc -l)
    echo "  $arm: $n non-empty cells, $z ZERO-BYTE (resume counts those as done - delete before resuming)"
  done
  ;;
*)
  echo "usage: $0 {prepare|chain|cells|status}" >&2; exit 2;;
esac
