#!/usr/bin/env bash
# #4936: measure the calibration fold models' scale against the final model's, per class, before correcting the
# labels line for it. Every session is today's app to 150 clicks (nothing in the session changes); at the listed
# clicks the test-score snapshot also carries every live fold model's scores of the withheld half and of the sim
# set, the sim set's truth and votes, and the vote behind each held-out score (CALIB_SAVE_FOLD_SCALES). The
# quantile line is fitted beside today's on the same head and draws nothing (CALIB_LABELS_SCALE_SHADOW=quantile):
# its xfer_* columns check that the offline replay of the haystacks is the run's own.
#
# Paths of the grid #4743 priced (2026-10-08-b1): Binary Photo (siglip, the first 144 cells of each seed's 288)
# and Region Photo (siglip+dinov3_patch max_patch, the other 144). Trajectory pass (no ceiling); cells gzipped.
#
#   fold_scale_4936.sh dirs                       run dirs <root>/<date>-fscale4936<tag>-<path>-b{025,1,4}
#   fold_scale_4936.sh cpack PATH BETA [IDX]      cpu-partition packs: one job runs CALIB_PACK_PAR cells at
#                                                 CALIB_CPUS (default 1) thread each
#   fold_scale_4936.sh gpack PATH BETA [IDX]      one pack on a V100 node, the GPU hidden (outside the cpu cap)
#   fold_scale_4936.sh idx PATH [SEEDS]           the grid indices of a path's cells, seed-major
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# run_cells.py pins the CPU kernels (#4110), so a cell does not depend on the node it lands on.
# Env: FSCALE4936_DATE (default 2026-10-15), FSCALE4936_SEEDS (5), FSCALE4936_TAG (e.g. smoke),
# FSCALE4936_AT (the snapshot clicks), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PACK_JOBS,
# CALIB_PACK_PAR, CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${FSCALE4936_DATE:-2026-10-15}"
SEEDS="${FSCALE4936_SEEDS:-5}"
AT="${FSCALE4936_AT:-10,25,50,75,100,125,150}"
PATHS=(binary region)
GRID_SRC="${FSCALE4936_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
offset() { case "$1" in binary) echo 0 ;; region) echo 144 ;; *) echo "path must be binary or region" >&2; exit 2 ;; esac; }
run_of() { echo "fscale4936${FSCALE4936_TAG:-}-$1-b$(btag "$2")"; }

path_idx() {  # PATH [SEEDS]
  python3 - "$(offset "$1")" "${2:-$SEEDS}" <<'PY'
import sys
off, n = int(sys.argv[1]), int(sys.argv[2])
print(",".join(str(s * 288 + off + i) for s in range(n) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  for p in "${PATHS[@]}"; do
    for b in 0.25 1 4; do
      r="$S/$D-$(run_of "$p" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$p" "$b")",
 "note": "#4936 fold-scale instrument (snapshots at $AT), $p path, beta $b, 150 clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
idx)
  path_idx "${2:?path (${PATHS[*]})}" "${3:-$SEEDS}"
  ;;
cpack|gpack)
  p="${2:?path (${PATHS[*]})}" b="${3:?beta (0.25, 1 or 4)}"
  idx="${4:-$(path_idx "$p")}"
  run="$(run_of "$p" "$b")"
  mem_in="${CALIB_MEM:-}"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=150
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_LABELS_SCALE CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_EARLY_CHECK
  export CALIB_LABELS_SCALE_SHADOW=quantile
  export CALIB_SAVE_TEST_SCORES=1 CALIB_SAVE_TEST_SCORES_AT="$AT" CALIB_SAVE_FOLD_SCALES=1
  if [[ "$1" == gpack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_CPUS=1 CALIB_MEM="${mem_in:-2G}"
    export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-32}"
    export CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}" CALIB_MEM="${mem_in:-2G}"
  export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}"
  export CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
  unset CALIB_PACK_GRES
  exec bash "$HERE/launch.sh" pack "$idx"
  ;;
*)
  echo "usage: fold_scale_4936.sh {dirs | idx PATH [SEEDS] | {cpack|gpack} PATH BETA [IDX]}" >&2
  exit 1 ;;
esac
