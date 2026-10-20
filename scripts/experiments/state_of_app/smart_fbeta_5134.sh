#!/usr/bin/env bash
# #5134: Smart in the objective's units. Today's Smart prices each cached retrain at FPR + FNR and goes green when
# that stops falling; the line, the objective and the other lights all moved to F-beta. This runs today's app with
# every candidate recorded (the reading pass, which is also the A/B's control) and the app with Smart read off one
# candidate, green once it stops rising (the arms).
#
#   read         the app, CALIB_SMART_READINGS=1: every candidate's trend on every row, the light unchanged
#   <signal>     CALIB_SMART_SIGNAL=<signal>, one of ap fbeta fbeta_best heldout_ap heldout_fbeta line
#                (vtscore.eval.autopilot_flow.SMART_SIGNALS; readings on)
#
# The Binary Photo path at beta 1/4, 1 and 4, 5 seeds x 144 cells on the grid #4743 priced (2026-10-08-b1), paired by
# cell and seed, 200 clicks and the closing check, as #5039's closed loop ran. An arm's run is the control's until the
# first click the two lights differ. Trajectory pass (no ceiling); cells gzipped.
#
#   smart_fbeta_5134.sh dirs [ARM]                 run dirs <root>/<date>-smart5134<tag>-<arm>-b{025,1,4}
#   smart_fbeta_5134.sh cpack ARM BETA [IDX]       one arm at one preset as cpu-partition packs (CALIB_PACK_JOBS jobs
#                                                  of CALIB_PACK_PAR cells at CALIB_CPUS thread each)
#   smart_fbeta_5134.sh gpack ARM BETA [IDX]       the same as one V100 pack (the GPU hidden; gpu partition)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: SMART5134_DATE (default 2026-10-20), SMART5134_SEEDS (5), SMART5134_TAG (e.g. smoke), SOTA_ROOT, and anything
# launch.sh reads (CALIB_MEM, CALIB_PACK_JOBS, CALIB_PACK_PAR, CALIB_PACK_TIME, CALIB_PACK_MEM).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${SMART5134_DATE:-2026-10-20}"
SEEDS="${SMART5134_SEEDS:-5}"
ARMS=(read ap fbeta fbeta_best heldout_ap heldout_fbeta line)
GRID_SRC="${SMART5134_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since

knobs() { if [[ "$1" == read ]]; then echo "CALIB_SMART_READINGS=1"; else echo "CALIB_SMART_SIGNAL=$1"; fi; }
btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "smart5134${SMART5134_TAG:-}-$1-b$(btag "$2")"; }
check_arm() { [[ " ${ARMS[*]} " == *" $1 "* ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }; }

all_idx() {
  python3 - "$SEEDS" <<'PY'
import sys
print(",".join(str(s * 288 + i) for s in range(int(sys.argv[1])) for i in range(144)))
PY
}

setup() {  # ARM BETA: the environment one arm at one preset runs under
  local arm="$1" b="$2" run
  run="$(run_of "$arm" "$b")"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=200
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150,200}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_SMART_SIGNAL CALIB_SMART_READINGS CALIB_SMART_GATE CALIB_STOP_RULE CALIB_STOP_LEVEL CALIB_STOP_CAP
  unset CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN
  unset CALIB_SAVE_TEST_SCORES CALIB_SAVE_FOLD_SCALES CALIB_SAVE_TEST_SCORES_AT CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T
  unset CALIB_WEAK_D CALIB_SPOT_CHECK CALIB_EARLY_CHECK CALIB_LABELS_SCALE CALIB_HEAD_COUNT_BADS
  local kv
  for kv in $(knobs "$arm"); do export "${kv?}"; done
  [[ "$arm" == read ]] || export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}smart_signal"
  export CALIB_CPUS="${CALIB_CPUS:-1}" CALIB_MEM="${CALIB_MEM:-2G}"
  export CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
}

case "${1:-}" in
dirs)
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    for b in 0.25 1 4; do
      r="$S/$D-$(run_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 200, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$b")",
 "note": "#5134 arm $arm ($(knobs "$arm")), beta $b, 200 clicks, binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
cpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (0.25, 1 or 4)}"
  check_arm "$arm"
  idx="${4:-$(all_idx)}"
  setup "$arm" "$b"
  export CALIB_PARTITION=cpu
  export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}"
  unset CALIB_PACK_GRES
  exec bash "$HERE/launch.sh" pack "$idx"
  ;;
gpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (0.25, 1 or 4)}"
  check_arm "$arm"
  idx="${4:-$(all_idx)}"
  setup "$arm" "$b"
  export CALIB_PARTITION=gpu CALIB_PACK_GRES=gpu:v100:1
  export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-40}"
  export CALIB_PACK_MEM="${CALIB_PACK_MEM:-$((CALIB_PACK_PAR * 3 / 2))G}"
  exec bash "$HERE/launch.sh" pack "$idx"
  ;;
*)
  echo "usage: smart_fbeta_5134.sh {dirs [ARM] | cpack ARM BETA [IDX] | gpack ARM BETA [IDX]}" >&2; exit 1 ;;
esac
