#!/usr/bin/env bash
# #4664: price a check in every session, early, against Autopilot as shipped. #4482 found the check where the user
# stops worth ~5x the same clicks of Autopilot at click 50, but its check arms stopped at the check. Here the
# session goes on after it, to 200 clicks, and closes with the end-of-run check as every arm does.
#
#   a0          today's app (its own checks: weak separation, and before Done), 200 clicks
#   e30/e50/e75 the same, with every session checked at 30 / 50 / 75 votes (CALIB_EARLY_CHECK): a prompted check
#               running then counts, otherwise Autopilot prompts one at the first step from N votes past the
#               opening, however recently another ended (the cooldown variant is read offline)
#   c50         today's app stopping at click 50, so its closing check is the check where the user stops: #4482's
#               c50 arm at today's head, to read the check at a stop beside the check inside a session
#
# The Binary Photo path, beta 1 and 4, 5 seeds x 144 cells, on the grid #4743 priced (2026-10-08-b1), paired by
# cell and seed. Trajectory pass (no ceiling); cells gzipped.
#
#   early_check_4664.sh dirs [ARM]                 run dirs <root>/<date>-early4664-<arm>-b{1,4} (every arm, or ARM)
#   early_check_4664.sh launch ARM BETA [IDX]      one arm at one preset as a cpu array (default: every seed's)
#   early_check_4664.sh pack ARM BETA [IDX]        the same as V100 packs, the GPU hidden (the cpu cap full)
#   early_check_4664.sh cpack ARM BETA [IDX]       the same as cpu-partition packs: one job runs CALIB_PACK_PAR
#                                                  cells at CALIB_CPUS (default 1) thread each, for when the
#                                                  2,000-job submit cap is full; a cell is charged one CPU, not
#                                                  an array task's whole core
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: EARLY4664_DATE (default 2026-10-14), EARLY4664_SEEDS (5), EARLY4664_TAG (e.g. smoke), SOTA_ROOT, and anything
# launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_TIME, CALIB_PACK_JOBS, CALIB_PACK_PAR, CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${EARLY4664_DATE:-2026-10-14}"
SEEDS="${EARLY4664_SEEDS:-5}"
ARMS=(a0 e30 e50 e75 c50)
declare -A AT=([a0]="" [e30]=30 [e50]=50 [e75]=75 [c50]="")
declare -A STEPS_OF=([c50]=50)
GRID_SRC="${EARLY4664_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

btag() { case "$1" in 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "early4664${EARLY4664_TAG:-}-$1-b$(btag "$2")"; }

all_idx() {
  python3 - "$SEEDS" <<'PY'
import sys
print(",".join(str(s * 288 + i) for s in range(int(sys.argv[1])) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    STEPS="${STEPS_OF[$arm]:-200}"
    for b in 1 4; do
      r="$S/$D-$(run_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$b")",
 "note": "#4664 arm $arm (CALIB_EARLY_CHECK=${AT[$arm]:-unset}), beta $b, $STEPS clicks, binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
launch|pack|cpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (1 or 4)}"
  [[ -v "AT[$arm]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  idx="${4:-$(all_idx)}"
  run="$(run_of "$arm" "$b")"
  STEPS="${STEPS_OF[$arm]:-200}"
  mem_in="${CALIB_MEM:-}"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-3G}" CALIB_TIME="${CALIB_TIME:-2:00:00}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150,200}"
  export CALIB_JOB_NAME="$run"
  if [[ -n "${AT[$arm]}" ]]; then
    export CALIB_EARLY_CHECK="${AT[$arm]}"
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}early_check"
  else
    unset CALIB_EARLY_CHECK
  fi
  if [[ "$1" == pack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}"
    export CALIB_PACK_PAR="${CALIB_PACK_PAR:-24}" CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  if [[ "$1" == cpack ]]; then
    export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}" CALIB_MEM="${mem_in:-2G}"
    export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}"
    export CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
    unset CALIB_PACK_GRES
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: early_check_4664.sh {dirs | launch ARM BETA [IDX] | pack ARM BETA [IDX] | cpack ARM BETA [IDX]}" >&2; exit 1 ;;
esac
