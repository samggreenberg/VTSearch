#!/usr/bin/env bash
# Acquisition/reporting threshold decoupling — docs/experiments/2026-08-07-acquisition-inclusion/REPORT.md
#
# #2847 (PR #2873) found today's fused threshold finds HALF as many positives as
# the conformal path it replaced (median 9 -> 4 per 100 votes, p=1e-20) while
# final cost is unchanged (p=0.09).  The cause is that one number does two jobs:
# it is the reported decision line AND the rank position Autopilot's `hard` pick
# samples around.  Production's cut sits at pool percentile 0.885 where the
# conformal cut sat at 0.959, so the selector samples ~11.5% deep instead of
# ~4.1% deep and brings back fewer positives.
#
# This run moves ONLY the selector's cut, holding reporting at inclusion 0 in
# every arm so cost stays comparable:
#
#   prod      k_acq =  0   control - today's production, both jobs on one number
#   acq_m1    k_acq = -1
#   acq_m2    k_acq = -2   the proposed operating point
#   acq_m3    k_acq = -3
#   acq_m4    k_acq = -4   far end; expected to be too greedy
#   acq_p2    k_acq = +2   FALSIFICATION arm - must make positives WORSE
#   rank_pin  cut pinned at the conformal path's own pool percentile (0.959)
#
# Negative k_acq RAISES the cut (a false alarm is priced higher), moving it UP
# the ranking -> the pick lands nearer the top -> more positives.  The +2 arm is
# load-bearing: if it does not move positives the wrong way, the mechanism is
# wrong and nothing else in the run is interpretable.
#
# Grid identical to #2847's so `prod` is directly comparable to that run's arm D.
# Prepare is REUSED (no GPU stage).
set -uo pipefail

export VTS_REPO=${VTS_REPO:-/exp/$USER/projects/vts-acq-incl}
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

export CALIB_EXP="/exp/$USER/acq-incl"

# --- environment: the #2847 grid, unchanged ---
export CALIB_DATASETS=coco_val
export CALIB_COCO_EMBEDDERS=siglip2
export CALIB_MAX_STEPS=100
export CALIB_N_SEEDS=${CALIB_N_SEEDS:-8}
export CALIB_HEAD=linear
export CALIB_SAFE_THRESHOLDS=1
export CALIB_ANCHORED=0
export CALIB_SCHEDULE_VARIANTS=
export CALIB_REPOOL_VARIANTS=

# --- ops: cpu partition, single-threaded cells; the real cap is the cpu_limit
# QOS (cpu=240/user, 2 charged per task) => 120 concurrent whatever %N says.
export CALIB_PARTITION=cpu
export CALIB_GRES=none
export CALIB_MEM=${CALIB_MEM:-8G}
export CALIB_CPUS=1
export CALIB_TIME=${CALIB_TIME:-2:00:00}
export CALIB_CONC=${CALIB_CONC:-18}

export VTSEARCH_DATA_DIR="$CALIB_EXP/datadir"
export VTSEARCH_MODELS_DIR="/exp/$USER/max-patch/models"
export HF_HOME="/exp/$USER/.cache/huggingface"

# Analysis is cross-arm and runs once, by hand, after all seven drain.
export CALIB_ANALYZE=${CALIB_ANALYZE:-noop.py}

mkdir -p "$CALIB_EXP/logs"

if [[ ! -f "$CALIB_EXP/results/prepare_info.json" ]]; then
  echo "ERROR: no prepare_info.json at $CALIB_EXP/results" >&2; exit 1
fi

if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 4 || {
    echo "preflight FAILED" >&2; [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
  }
fi

# arm -> "ACQ_INCLUSION_OFFSET ACQ_RANK_PERCENTILE"  ("-" = unset)
#
# The offset is relative to CALIB_INCLUSION, which this grid holds at 0, so every
# arm below is the same cut it was when the study ran.  `prod` now has to name
# its 0 explicitly: the default is -3, the shipped acquisition cut.
declare -A ARMS=(
  [prod]="0 -"
  [acq_m1]="-1 -"
  [acq_m2]="-2 -"
  [acq_m3]="-3 -"
  [acq_m4]="-4 -"
  [acq_p2]="2 -"
  [rank_pin]="0 0.959"
)

for arm in prod acq_m1 acq_m2 acq_m3 acq_m4 acq_p2 rank_pin; do
  read -r inc pct <<<"${ARMS[$arm]}"
  export CALIB_ACQ_INCLUSION_OFFSET=""
  export CALIB_ACQ_RANK_PERCENTILE=""
  [[ "$inc" != "-" ]] && export CALIB_ACQ_INCLUSION_OFFSET="$inc"
  [[ "$pct" != "-" ]] && export CALIB_ACQ_RANK_PERCENTILE="$pct"
  export CALIB_RESULTS="$CALIB_EXP/results/$arm"
  mkdir -p "$CALIB_RESULTS/cells"
  ln -sfn "$CALIB_EXP/results/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_EXP/results/crops" "$CALIB_RESULTS/crops"
  echo "=== arm $arm (acq_inclusion_offset='${CALIB_ACQ_INCLUSION_OFFSET}' acq_rank_percentile='${CALIB_ACQ_RANK_PERCENTILE}')"
  bash "$HERE/launch_cells.sh" || echo "ARM $arm SUBMIT FAILED" >&2
done
