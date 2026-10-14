#!/usr/bin/env bash
# #4384: the early head started from the typed query. Two session sets a preset, from one frozen worktree:
#  - the control: today's app with CALIB_QUERY_PRIOR_SHADOW=1, which records at every click a trained head is
#    drawn where the withheld positives sit under candidate heads started from the typed query
#    (task_NNNN__queryprior.csv). Recorded only, so the sessions are the app's;
#  - the arm (QP4384_ARM=<scale>): every head fit also takes the typed query's direction as a cheap feature at
#    that scale, beside the Goods' centroid (CALIB_QUERY_PRIOR). The shadow rides along (its cq<scale> is the
#    arm's own head).
# Both on the Binary Photo opening, 150 votes, no ceiling (SOTA_PASS=trajectory), on the grid #4743 priced
# (2026-10-08-b1), 5 seeds x 144 classes = 720 runs a preset. #4736's hold stays on in both (its head rows are
# tagged hold:head), so handover_4736.py re-prices the hold on the arm's sessions. Cells are gzipped.
#
#   queryprior_4384.sh dirs                 run dirs <root>/<date>-qp4384{ctl,arm<scale>}<tag>-b{025,1,4}
#   queryprior_4384.sh launch BETA [IDX]    that preset's sessions as a cpu array (IDX: a subset, e.g. a smoke)
#   queryprior_4384.sh pack BETA [IDX]      the same as V100 packs, the GPU hidden (the cpu cap full)
#
# Since #4859 the typed-query prior at scale 10 IS the app, so an unset QP4384_ARM runs it (the run dirs
# keep the name `ctl`) and QP4384_ARM=0 is the head before #4859 (the Goods' centroid alone).
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: QP4384_ARM (unset = the control), QP4384_SHADOW (default 1), QP4384_DATE (default 2026-10-13),
# QP4384_TAG (e.g. smoke), SOTA_ROOT, QP4384_SEEDS (5), and anything launch.sh reads (CALIB_MEM, CALIB_CONC,
# CALIB_PACK_JOBS, CALIB_PACK_PAR).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${QP4384_DATE:-2026-10-13}"
SEEDS="${QP4384_SEEDS:-5}"
ARM="${QP4384_ARM:-}"
SHADOW="${QP4384_SHADOW:-1}"
if [[ -n "$ARM" ]]; then RUN="qp4384arm${ARM}${QP4384_TAG:-}"; else RUN="qp4384ctl${QP4384_TAG:-}"; fi
[[ "$SHADOW" == 0 ]] && RUN="${RUN}noshadow"
GRID_SRC="${QP4384_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

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
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "$RUN-b$(btag "$b")",
 "note": "#4384 typed-query prior ($([[ -n "$ARM" ]] && echo "arm, scale $ARM" || echo control)$([[ "$SHADOW" == 0 ]] || echo ", shadow on")), today's app otherwise, beta $b, binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
launch|pack)
  b="${2:?beta (0.25, 1 or 4)}"
  idx="${3:-$(all_idx)}"
  [[ "$SHADOW" == 0 ]] || export CALIB_QUERY_PRIOR_SHADOW=1
  if [[ -n "$ARM" ]]; then
    export CALIB_QUERY_PRIOR="$ARM" PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}query_prior"
  fi
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-3G}"
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
  echo "usage: queryprior_4384.sh {dirs | launch BETA [IDX] | pack BETA [IDX]}" >&2; exit 1 ;;
esac
