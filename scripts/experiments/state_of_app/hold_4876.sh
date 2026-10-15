#!/usr/bin/env bash
# #4876: re-price #4736's typed-query hold on one build. The price itself is offline (hold_4876.py reads the
# shipped-hold sessions: #4874's SigLIP arm on photos, #4356's 1%-prevalence Apptitude runs on screens). These
# runs are its check: the same sessions on today's dev at other holds must pick the same images and show
# exactly what the offline price says.
#
#   hold_4876.sh dirs                     photo run dirs <root>/2026-10-14-hold4876-h{0,15}-b{025,1,4},
#                                         symlinked to #4874's grid so an index is the same cell
#   hold_4876.sh photos HOLD BETA         the check subset of the photo grid at CALIB_QUERY_HOLD=HOLD
#   hold_4876.sh screens HOLD BETA        every Apptitude session at 1% prevalence, at hold HOLD; runs only
#                                         from the report's tree (564ba13de), see below
#
# Launch from a FROZEN worktree: every task imports the worktree this script sits in.
# Env: HOLD_SUBSET (photo indices; default seeds 1 and 3, every fourth SigLIP cell), SOTA_ROOT, CALIB_CONC.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
GRID="$S/2026-10-14-s2l-4874-grid"

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }

# #4874's grid: 288 cells a seed (0-143 SigLIP, 144-287 SigLIP 2 L), 5 seeds. The check takes SigLIP cells
# from two seeds, a quarter of the classes x bands each, offset so the two seeds cover different cells.
subset() {
  python3 - <<'PY'
print(",".join(str(s * 288 + i) for s in (1, 3) for i in range(s % 4, 144, 4)))
PY
}

case "${1:-}" in
dirs)
  for h in 0 15; do
    for b in 0.25 1 4; do
      r="$S/2026-10-14-hold4876-h$h-b$(btag "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID/results/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID/results/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip2_l"], "n_seeds": 5,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "hold4876-h$h-b$(btag "$b")",
 "query_hold": "$h", "note": "#4876 check of the offline hold price, beta $b; grid symlinked from $GRID"}
JSON
      echo "$r"
    done
  done
  ;;
photos)
  h="${2:?hold}" b="${3:?beta (0.25, 1 or 4)}"
  export SOTA_EMBEDDERS="siglip,siglip2_l" SOTA_SEEDS=5 SOTA_BETA="$b"
  export SOTA_DATE="2026-10-14-hold4876-h$h-b$(btag "$b")" CALIB_QUERY_HOLD="$h"
  export CALIB_MEM="${CALIB_MEM:-5G}" CALIB_CPUS="${CALIB_CPUS:-1}" CALIB_CONC="${CALIB_CONC:-20}"
  export CALIB_JOB_NAME="hold4876-h$h-b$(btag "$b")"
  exec bash "$HERE/launch.sh" redo "${HOLD_SUBSET:-$(subset)}"
  ;;
screens)
  # The screens check re-runs the 2026-10-14 grids under /expscratch/sgreenberg/rico-suave/runs, which were
  # prepared when Apptitude was still named Rico Suave: their prepare_info.json names the dataset `rico_suave`,
  # which this tree no longer knows. Run it from the tree the report ran from, where the name still resolves.
  echo "hold_4876.sh screens: the 2026-10-14 grids were prepared as rico_suave; run this from 564ba13de" >&2
  exit 2
  ;;
*)
  echo "usage: hold_4876.sh {dirs | photos HOLD BETA | screens HOLD BETA}" >&2
  exit 1
  ;;
esac
