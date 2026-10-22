#!/usr/bin/env bash
# #4800: on the Region path, once the typed query has missed (the Bads quota met before 3 Goods), is the region
# centroid built from a Good's own box the better set for Find to give? Since #4758 a typed-query detector's Find
# gives the typed query's own set until the label quota. #4758 read the centroid as the better set on those votes
# off #4785's control run, at #4732's line over the withheld half, beta 1 only. This prices it at today's line at
# beta 1/4, 1 and 4.
#
# One run at the app's balance prices every balance: neither set picks and Autopilot's opening reads no balance, so
# the designs share every session and differ only in what Find returns. Each vote under the quota carries tagged
# rows (CALIB_CENTROID_LINE_VARIANTS): the typed query's set at each balance (query@<beta>), and the centroid at the
# app's rule over the withheld half (a Find on another collection) and over the unvoted media (a Find on the
# labelled collection, #4779). The run is cut at 20 votes: a vote with a Good and under the quota holds at most
# 2 Goods and 15 Bads.
#
#   missed_query_4800.sh dirs [BETA]                  run dir <root>/<date>-missedq4800<tag>-region-b<beta>
#   missed_query_4800.sh idx [FIRST [LAST]]           the grid indices of Region's cells for seeds FIRST..LAST
#   missed_query_4800.sh variants                     the run's CALIB_CENTROID_LINE_VARIANTS
#   missed_query_4800.sh preflight BETA [IDX]         the preflight a launch of the same cells would pass (launch
#                                                     runs launch_bands.sh redo, which runs none)
#   missed_query_4800.sh launch BETA [IDX]            a cpu array (default: every seed's Region cells)
#
# The run at beta 1 is the study. A few cells at beta 1/4 and 4 check that the votes under the quota are the same
# session at every balance (the analyzer's --same-session).
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: MQ4800_DATE (default 2026-10-19), MQ4800_SEEDS (4), MQ4800_TAG (e.g. smoke), MQ4800_STEPS (20), SOTA_ROOT,
# and anything launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${MQ4800_DATE:-2026-10-19}"
SEEDS="${MQ4800_SEEDS:-4}"
STEPS="${MQ4800_STEPS:-20}"
GRID_SRC="${MQ4800_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since (a docstring only)
RULE="count_gap_floor_skip25_b1"                       # CENTROID_LINE_RULE, the app's centroid line

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "missedq4800${MQ4800_TAG:-}-region-b$(btag "$1")"; }

variants() {
  local out=() b
  for b in 0.25 1 4; do out+=("query@$b" "$RULE@$b" "$RULE~unvoted@$b"); done
  local IFS=,
  echo "${out[*]}"
}

region_idx() {  # FIRST LAST: the second 144 cells of each seed's 288 (seed-major: 144 Binary, then 144 Region)
  python3 - "$1" "$2" <<'PY'
import sys
first, last = int(sys.argv[1]), int(sys.argv[2])
print(",".join(f"{s * 288 + 144}-{s * 288 + 287}" for s in range(first, last + 1)))
PY
}

case "${1:-}" in
dirs)
  for b in ${2:-0.25 1 4}; do
    r="$S/$D-$(run_of "$b")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$b")",
 "note": "#4800 the region centroid against the typed query under the quota, priced as tagged rows; region path, beta $b, $STEPS clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
idx)
  region_idx "${2:-0}" "${3:-$((SEEDS - 1))}"
  ;;
variants)
  variants
  ;;
preflight)
  # launch_bands.sh's redo runs no preflight: run it on the exact env a launch would submit with, by standing a
  # function in for sbatch (nothing is queued; launch_bands records job id 0 for the redo, the launch rewrites it).
  sbatch() {
    local a name="" mem="" arr="" spec n=0 r
    for a in "$@"; do
      case "$a" in --job-name=*) name="${a#*=}" ;; --mem=*) mem="${a#*=}" ;; --array=*) arr="${a#*=}" ;; esac
    done
    spec="${arr%%\%*}"
    for r in ${spec//,/ }; do
      if [[ "$r" == *-* ]]; then n=$((n + ${r#*-} - ${r%-*} + 1)); else n=$((n + 1)); fi
    done
    bash "$VTS_REPO/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod --job-name "$name" \
      --mem "$mem" --conc "${arr##*%}" --array-tasks "$n" --patch --require-text-seed \
      --require-region-voting "coco_better:siglip+dinov3_patch" >&2 && echo 0
  }
  export -f sbatch
  exec bash "$0" launch "${@:2}"
  ;;
launch)
  b="${2:?beta (0.25, 1 or 4)}"
  idx="${3:-$(region_idx 0 $((SEEDS - 1)))}"
  run="$(run_of "$b")"
  export CALIB_CENTROID_LINE_VARIANTS="$(variants)"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-26G}" CALIB_TIME="${CALIB_TIME:-2:00:00}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10}"
  export CALIB_JOB_NAME="$run"
  # Today's app: no arm knob of an earlier study rides along from the caller's shell.
  unset CALIB_HEAD CALIB_LABELS_SCALE CALIB_LABELS_SCALE_SHADOW CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_EARLY_CHECK
  unset CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN
  unset CALIB_SAVE_TEST_SCORES CALIB_SAVE_TEST_SCORES_AT CALIB_SAVE_FOLD_SCALES CALIB_SPOT_CHECK CALIB_HEAD_COUNT_BADS
  unset CALIB_CENTROID_LINE CALIB_QUERY_TIER CALIB_QUERY_HOLD CALIB_LABEL_QUOTA VTSEARCH_SVM_HEAD_C
  export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}"
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: missed_query_4800.sh {dirs [BETA] | idx [FIRST [LAST]] | variants | {preflight|launch} BETA [IDX]}" >&2
  exit 1 ;;
esac
