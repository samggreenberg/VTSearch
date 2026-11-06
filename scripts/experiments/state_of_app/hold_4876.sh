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
#   hold_4876.sh review-dirs HOLD         #5556: run dirs <root>/<REVIEW>-hold5556-hHOLD-b{025,1,4}, symlinked to
#                                         the 2026-11-03 Binary Photo review's grids (#5525)
#   hold_4876.sh review HOLD BETA         the check subset of that review at CALIB_QUERY_HOLD=HOLD, as its driver
#                                         ran it (trajectory pass, its rank frames); set RUN_TREE to a frozen
#                                         worktree at the review's commit (605274e2e)
#
# Launch from a FROZEN worktree: every task imports the worktree this script sits in (or RUN_TREE's, for review).
# Env: HOLD_SUBSET (photo indices; default seeds 1 and 3, every fourth SigLIP cell), SOTA_ROOT, CALIB_CONC,
# REVIEW (default 2026-11-03), RUN_TREE.
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

REVIEW="${REVIEW:-2026-11-03}"
# The review's driver (sota-1103/drive.sh): 100 seeds prepared, these rank frames, 2 CPUs and 2G a task.
REVIEW_FRAMES=1,2,3,4,5,6,7,8,9,10,12,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125,130,135,140,145,150

case "${1:-}" in
review-dirs)
  h="${2:?hold}"
  for b in 0.25 1 4; do
    src="$S/$REVIEW-b$(btag "$b")/results"
    r="$S/$REVIEW-hold5556-h$h-b$(btag "$b")/results"
    mkdir -p "$r/cells"
    ln -sfn "$src/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$src/crops" "$r/crops"
    python3 - "$src/grid_shape.json" "$r/grid_shape.json" "$h" "$b" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))
g.update(job_name=f"hold5556-h{sys.argv[3]}", query_hold=sys.argv[3],
         note=f"#5556 check of the offline hold price at {sys.argv[3]} labels, beta {sys.argv[4]}; grid symlinked from {sys.argv[1]}")
json.dump(g, open(sys.argv[2], "w"), indent=1)
PY
    echo "$r"
  done
  ;;
review)
  h="${2:?hold}" b="${3:?beta (0.25, 1 or 4)}"
  tree="${RUN_TREE:?set RUN_TREE to a frozen worktree at the review commit}"
  export SOTA_SEEDS=100 SOTA_BETA="$b" SOTA_PASS=trajectory SOTA_RANK_FRAME_STEPS="$REVIEW_FRAMES"
  export SOTA_DATE="$REVIEW-hold5556-h$h-b$(btag "$b")" CALIB_QUERY_HOLD="$h"
  export CALIB_MEM="${CALIB_MEM:-2G}" CALIB_CPUS="${CALIB_CPUS:-2}" CALIB_TIME="${CALIB_TIME:-1:00:00}"
  export CALIB_CONC="${CALIB_CONC:-72}" CALIB_JOB_NAME="hold5556-h$h-b$(btag "$b")"
  cd "$tree/scripts/experiments/state_of_app"
  exec bash launch.sh redo "${HOLD_SUBSET:-$(subset)}"
  ;;
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
  echo "usage: hold_4876.sh {dirs | photos HOLD BETA | screens HOLD BETA | review-dirs HOLD | review HOLD BETA}" >&2
  exit 1
  ;;
esac
