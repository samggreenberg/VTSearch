#!/usr/bin/env bash
# #4482: what a uniform band pick is worth against an Autopilot pick, as curves over the click a user has reached.
# Binary Photo at beta 1, paired by cell and seed, every arm today's app except where named:
#   a0      Autopilot as shipped, 200 clicks
#   a8/a4/a2  one in 8 / 4 / 2 of the picks past the opening is a band pick (CALIB_BAND_SHARE), 200 clicks
#   c50/c100/c150  as shipped, stopping at 50 / 100 / 150 clicks: the check where the user stops
# Every run ends with the app's check (spot_check="weak" is "end" plus the prompted one), so each arm also reads
# "after the check" at its last click.
#
#   band_4482.sh dirs SEEDS                       run dirs <root>/<date>-band4482-<arm>-b1 on the 2026-10-06 grid
#   band_4482.sh launch ARM SEEDS [FIRST [LAST]]  queue one arm's binary cells, seeds FIRST..LAST of a SEEDS-seed grid
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${BAND_DATE:-2026-10-08}"
GRID_SRC="${BAND_GRID:-$S/2026-10-06-b1/results}"  # prepare_data.py unchanged since this grid was prepared
ARMS=(a0 a8 a4 a2 c50 c100 c150)
declare -A SHARE=([a0]="" [a8]=8 [a4]=4 [a2]=2 [c50]="" [c100]="" [c150]="")
declare -A STEPS=([a0]=200 [a8]=200 [a4]=200 [a2]=200 [c50]=50 [c100]=100 [c150]=150)

dir_of() { echo "$S/$D-band4482-$1-b1"; }

case "${1:-}" in
dirs)
  SEEDS="${2:?seeds}"
  for arm in "${ARMS[@]}"; do
    r="$(dir_of "$arm")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<EOF
{"n_cells": $((288 * SEEDS)), "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": ${STEPS[$arm]}, "cell_order": "seed", "test_bands": "all", "job_name": "band4482-$arm-b1",
 "note": "#4482 arm $arm (CALIB_BAND_SHARE=${SHARE[$arm]:-unset}, ${STEPS[$arm]} clicks), beta 1, binary only, trajectory pass; grid symlinked from $GRID_SRC"}
EOF
  done
  ls -d "$S/$D"-band4482-*
  ;;
launch)
  ARM="${2:?arm}"; SEEDS="${3:?seeds}"; FIRST="${4:-0}"; LAST="${5:-$((SEEDS - 1))}"
  [[ -n "${STEPS[$ARM]:-}" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  (( FIRST >= 0 && FIRST <= LAST && LAST < SEEDS )) || { echo "seeds must satisfy 0 <= FIRST <= LAST < SEEDS" >&2; exit 2; }
  export SOTA_BETA=1 SOTA_DATE="$D-band4482-$ARM-b1" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="${STEPS[$ARM]}"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150,200}"
  export CALIB_MEM="${CALIB_MEM:-4G}" CALIB_TIME="${CALIB_TIME:-1:30:00}" CALIB_JOB_NAME="band4482-$ARM-b1"
  if [[ -n "${SHARE[$ARM]}" ]]; then
    export CALIB_BAND_SHARE="${SHARE[$ARM]}"
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}band_share"
  else
    unset CALIB_BAND_SHARE
  fi
  idx="$(python3 -c "import sys; a, b = int(sys.argv[1]), int(sys.argv[2]); print(','.join(f'{s * 288}-{s * 288 + 143}' for s in range(a, b + 1)))" "$FIRST" "$LAST")"
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: band_4482.sh {dirs SEEDS | launch ARM SEEDS [FIRST [LAST]]}" >&2; exit 1 ;;
esac
