#!/usr/bin/env bash
# #4736: the typed query's hold, as the app runs it, on COCO Better's Binary opening. The hold changes what
# Test gives, never a pick, and it ends at QUERY_HOLD_LABELS (30) votes, so a session cut at 30 votes carries
# every click it touches: its untagged rows are the arm, its `hold:head` rows are today's head at the same
# clicks, and #4786's full sessions (2026-10-11-line4786all-b025/-b1, -line4786thin-b4) are today's from
# vote 30 on. handover_4736.py `arm` checks the picks against those, click for click.
#
#   hold_4736.sh dirs                      run dirs <root>/<date>-hold4736-b{025,1,4}
#   hold_4736.sh launch BETA               the preset's 720 Binary sessions as a cpu array
#   hold_4736.sh pack BETA [JOBS]          the same as V100 pack jobs (the cpu cap full of other arrays)
#   hold_4736.sh region BETA IDX[,IDX...]  Region cells as V100 packs: the guard that the hold never fires there
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: HOLD4736_DATE (default 2026-10-12), HOLD4736_STEPS (30), SOTA_ROOT, HOLD4736_SEEDS (5), and anything
# launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_PACK_PAR).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${HOLD4736_DATE:-2026-10-12}"
SEEDS="${HOLD4736_SEEDS:-5}"
STEPS="${HOLD4736_STEPS:-30}"
GRID_SRC="${HOLD4736_GRID:-$S/2026-10-08-b1/results}"  # the grid #4786's sessions ran on

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }

binary_idx() {
  python3 - "$SEEDS" <<'PY'
import sys
print(",".join(str(s * 288 + i) for s in range(int(sys.argv[1])) for i in range(144)))
PY
}

setup() {
  local b="$1" run="$2"
  export CALIB_MEM="${CALIB_MEM:-3G}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run-b$(btag "$b")" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export CALIB_JOB_NAME="$run-b$(btag "$b")"
}

case "${1:-}" in
dirs)
  for run in hold4736 hold4736region; do
    for b in 0.25 1 4; do
      r="$S/$D-$run-b$(btag "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$run-b$(btag "$b")",
 "note": "#4736 the typed query's hold (the app), beta $b, $STEPS votes, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
launch)
  setup "${2:?beta (0.25, 1 or 4)}" hold4736
  exec bash "$HERE/launch.sh" redo "$(binary_idx)"
  ;;
pack)
  setup "${2:?beta (0.25, 1 or 4)}" hold4736
  export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_PACK_JOBS="${3:-1}"
  export CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}" CALIB_TIME="${CALIB_TIME:-8:00:00}"
  exec bash "$HERE/launch.sh" pack "$(binary_idx)"
  ;;
region)
  setup "${2:?beta (0.25, 1 or 4)}" hold4736region
  export CALIB_MEM="${HOLD4736_REGION_MEM:-24G}"
  export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_PACK_JOBS=1
  export CALIB_PACK_PAR="${CALIB_PACK_PAR:-4}" CALIB_TIME="${CALIB_TIME:-8:00:00}"
  exec bash "$HERE/launch.sh" pack "${3:?comma-separated Region cell indices (144-287 in seed 0)}"
  ;;
*)
  echo "usage: hold_4736.sh {dirs | launch BETA | pack BETA [JOBS] | region BETA IDX[,IDX...]}" >&2; exit 1 ;;
esac
