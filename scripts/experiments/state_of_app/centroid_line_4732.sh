#!/usr/bin/env bash
# #4732: price the Goods' centroid's line on the State of the App's Binary Photo opening. From the
# first Good until the label quota, Test gives the centroid, cut at its two-Gaussian midpoint; the
# session shows the typed query's sort meanwhile, so this is what a Find in the opening returns.
# One run at the app's balance, every candidate line priced as tagged rows on the same sessions
# (CALIB_CENTROID_LINE_VARIANTS, see fhibe/centroid_line_4732.sh), the run cut at SOTA_MAX_STEPS=40,
# well past the quota.
#
#   centroid_line_4732.sh dirs SEEDS                 run dir <root>/<date>-centroid4732-b1 on the 2026-10-08 grid
#   centroid_line_4732.sh launch SEEDS [FIRST [LAST]]
#                                                    queue the Binary sessions of seeds FIRST..LAST
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: CL4732_DATE (default 2026-10-10), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PARTITION).
# CL4732_PACK=1 packs the cells into one job on a V100 node instead of an array (CALIB_PACK_PAR at a time).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${CL4732_DATE:-2026-10-10}"
GRID_SRC="${CL4732_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared
STEPS="${SOTA_MAX_STEPS:-40}"
DIR="$S/$D-centroid4732-b1"

indices() {  # FIRST LAST: the Binary block of each seed (seed-major blocks of 144 binary then 144 region cells)
  python3 - "$1" "$2" <<'PY'
import sys
print(",".join(f"{s * 288}-{s * 288 + 143}" for s in range(int(sys.argv[1]), int(sys.argv[2]) + 1)))
PY
}

case "${1:-}" in
dirs)
  SEEDS="${2:?seeds}"
  r="$DIR/results"
  mkdir -p "$r/cells"
  ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
  ln -sfn "$GRID_SRC/crops" "$r/crops"
  cat >"$r/grid_shape.json" <<EOF
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "cl4732-binary-b1",
 "note": "#4732 centroid line variants, beta 1, Binary only, trajectory pass to $STEPS; grid symlinked from $GRID_SRC"}
EOF
  ls -la "$r"
  ;;
launch)
  SEEDS="${2:?seeds}"; FIRST="${3:-0}"; LAST="${4:-$((SEEDS - 1))}"
  (( FIRST >= 0 && FIRST <= LAST && LAST < SEEDS )) || { echo "seeds must satisfy 0 <= FIRST <= LAST < SEEDS" >&2; exit 2; }
  export CALIB_CENTROID_LINE_VARIANTS="${CALIB_CENTROID_LINE_VARIANTS:-midpoint@0.25,midpoint@1,midpoint@4,guarded@0.25,guarded@1,guarded@4,count@0.25,count@1,count@4}"
  export CALIB_MEM="${CALIB_MEM:-4G}"
  export SOTA_BETA=1 SOTA_DATE="$D-centroid4732-b1" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export CALIB_JOB_NAME="cl4732-binary-b1"
  idx="$(indices "$FIRST" "$LAST")"
  if [[ "${CL4732_PACK:-0}" == 1 ]]; then
    # The cpu cap is full: one multi-CPU job on a V100 node (launch_bands.sh pack, #4490).
    export CALIB_PARTITION="${CALIB_PARTITION:-gpu}" CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}"
    export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-30}" CALIB_TIME="${CALIB_TIME:-4:00:00}"
    exec bash "$HERE/launch.sh" pack "$(python3 -c "import sys; print(','.join(str(i) for r in sys.argv[1].split(',') for i in range(int(r.split('-')[0]), int(r.split('-')[1]) + 1)))" "$idx")"
  fi
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: centroid_line_4732.sh {dirs SEEDS | launch SEEDS [FIRST [LAST]]}" >&2; exit 1 ;;
esac
