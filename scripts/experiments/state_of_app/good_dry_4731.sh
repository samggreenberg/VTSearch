#!/usr/bin/env bash
# #4731: the COCO guard for a Good phase that runs dry. State of the App arms at one commit, paired run for run
# and identical until the Good walk's 16th miss in a row with a Good in hand:
#   C  the app
#   Q  CALIB_GOOD_DRY_RUN=16 CALIB_QUOTA_DRY_BADS=16: the Good phase ends on that run of misses, and a Good with
#      16 Bads gets the trained head instead of the Goods' centroid
#   D  CALIB_GOOD_DRY_RUN=16 alone: the phase change without the quota tier
# The rule fires in ~19% of Binary sessions at beta 1 (2026-10-08-b1's picks), so FHIBE alone cannot clear it.
#
#   good_dry_4731.sh dirs SEEDS                  run dirs <root>/<date>-gooddry4731{C,Q,D}-b1 on the 2026-10-08 grid
#   good_dry_4731.sh launch ARM SEEDS [FIRST [LAST]]
#                                                queue one arm's Binary sessions, seeds FIRST..LAST, at beta 1
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: GOODDRY_DATE (default 2026-10-09), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PARTITION).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${GOODDRY_DATE:-2026-10-09}"
GRID_SRC="${GOODDRY_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

dir_of() { echo "$S/$D-gooddry4731$1-b1"; }

indices() {  # FIRST LAST: the Binary block of each seed (seed-major blocks of 144 binary then 144 region cells)
  python3 - "$1" "$2" <<'PY'
import sys
print(",".join(f"{s * 288}-{s * 288 + 143}" for s in range(int(sys.argv[1]), int(sys.argv[2]) + 1)))
PY
}

case "${1:-}" in
dirs)
  SEEDS="${2:?seeds}"
  for arm in C Q D; do
    r="$(dir_of "$arm")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<EOF
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "gooddry4731$arm-b1",
 "note": "#4731 arm $arm, beta 1, Binary only, trajectory pass; grid symlinked from $GRID_SRC"}
EOF
  done
  ls -d "$S/$D"-gooddry4731*
  ;;
launch)
  ARM="${2:?C|Q|D}"; SEEDS="${3:?seeds}"; FIRST="${4:-0}"; LAST="${5:-$((SEEDS - 1))}"
  (( FIRST >= 0 && FIRST <= LAST && LAST < SEEDS )) || { echo "seeds must satisfy 0 <= FIRST <= LAST < SEEDS" >&2; exit 2; }
  unset CALIB_GOOD_DRY_RUN CALIB_QUOTA_DRY_BADS
  case $ARM in
    C) ;;
    Q) export CALIB_GOOD_DRY_RUN=16 CALIB_QUOTA_DRY_BADS=16
       export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}good_dry_run,quota_dry_bads" ;;
    D) export CALIB_GOOD_DRY_RUN=16
       export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}good_dry_run" ;;
    *) echo "ARM must be C, Q or D" >&2; exit 2 ;;
  esac
  export CALIB_MEM="${CALIB_MEM:-4G}"
  export SOTA_BETA=1 SOTA_DATE="$D-gooddry4731$ARM-b1" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS"
  export CALIB_JOB_NAME="gooddry4731$ARM-binary-b1"
  exec bash "$HERE/launch.sh" redo "$(indices "$FIRST" "$LAST")"
  ;;
*)
  echo "usage: good_dry_4731.sh {dirs SEEDS | launch ARM SEEDS [FIRST [LAST]]}" >&2; exit 1 ;;
esac
