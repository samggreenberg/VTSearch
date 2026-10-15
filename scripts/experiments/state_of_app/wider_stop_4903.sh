#!/usr/bin/env bash
# #4903: today's app (stop rule `lights|level`, #4878) with its sessions run past the eval's 150-click horizon,
# to 400, so the stop rules can be read where the eval never looked: how many sessions no rule has stopped by
# 150, 200, 300 and 400 clicks, what those sessions still gain, and what a click cap ("the rule, or N clicks")
# costs at each N (wider_stop_4903.py). Recording only: the sessions are the app's, with the readings #4878
# records at every click (phase, the lights, line_efbeta, line_head_count).
#
# The Binary Photo opening, 400 votes, no ceiling (SOTA_PASS=trajectory), on the grid #4743 priced
# (2026-10-08-b1), 144 classes a seed and a preset. Cells are gzipped.
#
#   wider_stop_4903.sh dirs                 run dirs <root>/<date>-stop4903<tag>-b{025,1,4}
#   wider_stop_4903.sh launch BETA [IDX]    that preset's sessions as a cpu array (default: every seed's)
#   wider_stop_4903.sh pack BETA [IDX]      the same as V100 packs, the GPU hidden (the cpu cap full)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: STOP4903_STEPS (400), STOP4903_DATE (default 2026-10-15), STOP4903_TAG (e.g. smoke), STOP4903_SEEDS (5),
# SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_CONC).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${STOP4903_DATE:-2026-10-15}"
SEEDS="${STOP4903_SEEDS:-5}"
STEPS="${STOP4903_STEPS:-400}"
RUN="stop4903${STOP4903_TAG:-}"
GRID_SRC="${STOP4903_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }

all_idx() {
  python3 - "$SEEDS" <<'PY'
import sys
print(",".join(str(s * 288 + i) for s in range(int(sys.argv[1])) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  for b in 0.25 1 4; do
    r="$S/$D-$RUN-b$(btag "$b")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$RUN-b$(btag "$b")",
 "note": "#4903 the app's stop rule (lights|level) past the horizon, to $STEPS clicks, beta $b, binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
launch|pack)
  b="${2:?beta (0.25, 1 or 4)}"
  idx="${3:-$(all_idx)}"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-3G}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$RUN-b$(btag "$b")" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export CALIB_JOB_NAME="$RUN-b$(btag "$b")"
  if [[ "$1" == pack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}"
    export CALIB_PACK_PAR="${CALIB_PACK_PAR:-24}" CALIB_TIME="${CALIB_TIME:-12:00:00}"
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: wider_stop_4903.sh {dirs | launch BETA [IDX] | pack BETA [IDX]}" >&2; exit 1 ;;
esac
