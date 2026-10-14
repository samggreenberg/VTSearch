#!/usr/bin/env bash
# #4816: sessions to price a hand-over judge for the typed query's hold that reads the votes where the query's
# set and the head's disagree. What is shown changes no pick (#4736), so one session set per preset prices every
# rule: CALIB_SET_OVERLAP=1 records, at every click a trained head scored the pool, how the head's set and the
# typed query's split the unvoted media (task_NNNN__overlap.csv) and, in the pick log, where each pick sat in
# them at the previous click. CALIB_LINE_COUNTS=1 rides along, so #4736's judges (d', the class model's Goods)
# read the same sessions. Today's app otherwise (#4736's hold, #4798's gate): the Binary Photo opening, 150 votes,
# no ceiling (SOTA_PASS=trajectory), on the grid #4743 priced (2026-10-08-b1), 5 seeds x 144 classes = 720 runs
# a preset. Cells are gzipped (CALIB_CELLS_GZIP=1).
#
#   overlap_4816.sh dirs                 run dirs <root>/<date>-overlap4816<tag>-b{025,1,4}
#   overlap_4816.sh launch BETA [IDX]    that preset's Binary sessions as a cpu array (IDX: a subset, e.g. a smoke)
#   overlap_4816.sh pack BETA [IDX]      the same as V100 packs, the GPU hidden (the cpu cap full)
#
# OV4816_PATH=region runs the Region path instead (DINOv3 patches opened on SigLIP's text sort, max_patch),
# 2 seeds x 144 classes, as V100 packs (cells peak near 18 GB and take ~50 min), into
# <date>-overlap4816region<tag>-b*. The hold never fires there (patch spaces hand over at the quota).
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: OV4816_DATE (default 2026-10-13), OV4816_TAG (e.g. smoke), SOTA_ROOT, OV4816_SEEDS (5 Binary, 2 Region),
# and anything launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_PACK_JOBS, CALIB_PACK_PAR).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${OV4816_DATE:-2026-10-13}"
PATH_KIND="${OV4816_PATH:-binary}"
case "$PATH_KIND" in
  binary) SEEDS="${OV4816_SEEDS:-5}"; FIRST=0; RUN="overlap4816${OV4816_TAG:-}" ;;
  region) SEEDS="${OV4816_SEEDS:-2}"; FIRST=144; RUN="overlap4816region${OV4816_TAG:-}" ;;  # each seed's Region block
  *) echo "OV4816_PATH must be binary or region" >&2; exit 2 ;;
esac
GRID_SRC="${OV4816_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }

all_idx() {
  python3 - "$SEEDS" "$FIRST" <<'PY'
import sys
n, first = int(sys.argv[1]), int(sys.argv[2])
print(",".join(str(s * 288 + first + i) for s in range(n) for i in range(144)))
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
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "$RUN-b$(btag "$b")",
 "note": "#4816 set overlap + line counts, today's app, beta $b, $PATH_KIND path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
launch|pack)
  b="${2:?beta (0.25, 1 or 4)}"
  idx="${3:-$(all_idx)}"
  export CALIB_SET_OVERLAP=1 CALIB_LINE_COUNTS=1 CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  if [[ "$PATH_KIND" == region ]]; then export CALIB_MEM="${CALIB_MEM:-24G}"; else export CALIB_MEM="${CALIB_MEM:-3G}"; fi
  export SOTA_BETA="$b" SOTA_DATE="$D-$RUN-b$(btag "$b")" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=150
  export CALIB_JOB_NAME="$RUN-b$(btag "$b")"
  if [[ "$1" == pack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-2}"
    export CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}" CALIB_TIME="${CALIB_TIME:-12:00:00}"
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: overlap_4816.sh {dirs | launch BETA [IDX] | pack BETA [IDX]}" >&2; exit 1 ;;
esac
