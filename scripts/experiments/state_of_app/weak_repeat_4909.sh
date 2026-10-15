#!/usr/bin/env bash
# #4909: re-price the weak check's repeat at today's head. #4496 chose "again after 25 votes while the labels separate
# weakly" on the 10-05 app, where a check's picks were worth far more than Autopilot's; #4664 found them worth about
# the same now, while checks take 25-29% of a session's clicks. Every arm is today's app to 200 clicks, closing with
# the end-of-run check, and differs only in the weak-separation check (the check before Done is the app's in all):
#
#   r25   the app before #4909: the weak check again 25 votes after a check ends, while the labels separate weakly
#         (CALIB_WEAK_REPEAT=25)
#   r0    the weak check once (CALIB_WEAK_REPEAT=0), the app since #4909
#   r50   again after 50 votes (CALIB_WEAK_REPEAT=50)
#   nw    no weak check (CALIB_WEAK_MIN_T past the budget): the bracket, what the first prompt itself is worth
#
# The Binary Photo path, beta 1 and 4, 5 seeds x 144 cells, on the grid #4743 priced (2026-10-08-b1), paired by
# cell and seed: an arm's run is byte-identical to r25's until the first weak prompt one of them runs and the other
# does not. Trajectory pass (no ceiling); cells gzipped.
#
#   weak_repeat_4909.sh dirs [ARM]               run dirs <root>/<date>-repeat4909<tag>-<arm>-b{1,4} (every arm, or ARM)
#   weak_repeat_4909.sh launch ARM BETA [IDX]    one arm at one preset as a cpu array (default: every seed's cells)
#   weak_repeat_4909.sh cpack ARM BETA [IDX]     the same as cpu-partition packs: one job runs CALIB_PACK_PAR cells at
#                                                CALIB_CPUS (default 1) thread each, for when the 2,000-job submit cap
#                                                is full; a cell is charged one CPU, not an array task's whole core
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it). Pin the
# CPU vendor (SBATCH_CONSTRAINT=amd) so paired arms do not drift apart on BLAS (#4481).
# Env: REPEAT4909_DATE (default 2026-10-15), REPEAT4909_SEEDS (5), REPEAT4909_TAG (e.g. smoke), SOTA_ROOT, and
# anything launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_TIME, CALIB_PACK_JOBS, CALIB_PACK_PAR, CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${REPEAT4909_DATE:-2026-10-15}"
SEEDS="${REPEAT4909_SEEDS:-5}"
ARMS=(r25 r0 r50 nw)
# Each arm's knobs, as env assignments. The study ran at b5cdd719a, where r25 was the app and set none; #4909
# shipped r0, so r25 is pinned and declared now, and r0 is the app's (its pin is the app's value).
declare -A KNOBS=([r25]="CALIB_WEAK_REPEAT=25" [r0]="CALIB_WEAK_REPEAT=0" [r50]="CALIB_WEAK_REPEAT=50" [nw]="CALIB_WEAK_MIN_T=100000")
declare -A DIVERGES=([r25]="weak_repeat" [r0]="" [r50]="weak_repeat" [nw]="weak_min_t")
GRID_SRC="${REPEAT4909_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

btag() { case "$1" in 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "repeat4909${REPEAT4909_TAG:-}-$1-b$(btag "$2")"; }

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
    knobs="${KNOBS[$arm]:-the app weak check}"
    for b in 1 4; do
      r="$S/$D-$(run_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 200, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$b")",
 "note": "#4909 arm $arm ($knobs), beta $b, 200 clicks, binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
launch|cpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (1 or 4)}"
  [[ -v "KNOBS[$arm]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  idx="${4:-$(all_idx)}"
  run="$(run_of "$arm" "$b")"
  mem_in="${CALIB_MEM:-}"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-3G}" CALIB_TIME="${CALIB_TIME:-2:00:00}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=200
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150,200}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_WEAK_D CALIB_SPOT_CHECK CALIB_EARLY_CHECK
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  if [[ -n "${DIVERGES[$arm]}" ]]; then
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}${DIVERGES[$arm]}"
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
  echo "usage: weak_repeat_4909.sh {dirs [ARM] | launch ARM BETA [IDX] | cpack ARM BETA [IDX]}" >&2; exit 1 ;;
esac
