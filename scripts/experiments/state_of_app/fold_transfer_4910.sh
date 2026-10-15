#!/usr/bin/env bash
# #4910: price the quantile transfer inside the labels line. Today's line pools the calibration folds' held-out
# scores as each fold model scored them, then cuts on the final model's corpus scores; the arm carries fold k's
# scores onto the final model's scale through fold k's haystack first (CALIB_LABELS_SCALE). Every arm is today's
# app to 150 clicks and differs only in the scale the labels line's class model reads:
#
#   c    as shipped (raw), with the quantile line fitted beside it on the same head (CALIB_LABELS_SCALE_SHADOW=quantile):
#        the gate's same-head reading, and the closed loop's control
#   q    the quantile transfer (CALIB_LABELS_SCALE=quantile), the raw line beside it
#   qt   the transfer with its tails kept past the haystack's ends (quantile_tails), the raw line beside it
#
# Two paths of the grid #4743 priced (2026-10-08-b1): Binary Photo (siglip, the first 144 cells of each seed's 288)
# and Region Photo (siglip+dinov3_patch max_patch, the other 144), at the three presets, paired by cell and seed.
# Trajectory pass (no ceiling); cells gzipped.
#
#   fold_transfer_4910.sh dirs [ARM]                  run dirs <root>/<date>-xfer4910<tag>-<arm>-<path>-b{025,1,4}
#   fold_transfer_4910.sh launch ARM PATH BETA [IDX]  one arm, path and preset as a cpu array (default: every seed's)
#   fold_transfer_4910.sh cpack ARM PATH BETA [IDX]   the same as cpu-partition packs: one job runs CALIB_PACK_PAR
#                                                     cells at CALIB_CPUS (default 1) thread each
#   fold_transfer_4910.sh gpack ARM PATH BETA [IDX]   the same as one pack on a V100 node, the GPU hidden: the gpu
#                                                     partition is outside the cpu cap (4 GPUs per user)
#   fold_transfer_4910.sh idx PATH [SEEDS]            the grid indices of a path's cells, seed-major
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it). Pin the
# CPU vendor so paired arms do not drift apart on BLAS (#4481): every V100 node is Intel, so this study runs on
# Intel throughout (SBATCH_CONSTRAINT=intel on the cpu partition too).
# Env: XFER4910_DATE (default 2026-10-15), XFER4910_SEEDS (5), XFER4910_TAG (e.g. smoke), SOTA_ROOT, and anything
# launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_TIME, CALIB_PACK_JOBS, CALIB_PACK_PAR, CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${XFER4910_DATE:-2026-10-15}"
SEEDS="${XFER4910_SEEDS:-5}"
ARMS=(c q qt)
PATHS=(binary region)
declare -A KNOBS=(
  [c]="CALIB_LABELS_SCALE_SHADOW=quantile"
  [q]="CALIB_LABELS_SCALE=quantile CALIB_LABELS_SCALE_SHADOW=raw"
  [qt]="CALIB_LABELS_SCALE=quantile_tails CALIB_LABELS_SCALE_SHADOW=raw"
)
declare -A DIVERGES=([c]="" [q]="labels_scale" [qt]="labels_scale")
GRID_SRC="${XFER4910_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
offset() { case "$1" in binary) echo 0 ;; region) echo 144 ;; *) echo "path must be binary or region" >&2; exit 2 ;; esac; }
run_of() { echo "xfer4910${XFER4910_TAG:-}-$1-$2-b$(btag "$3")"; }

path_idx() {  # PATH [SEEDS]
  python3 - "$(offset "$1")" "${2:-$SEEDS}" <<'PY'
import sys
off, n = int(sys.argv[1]), int(sys.argv[2])
print(",".join(str(s * 288 + off + i) for s in range(n) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    for p in "${PATHS[@]}"; do
      for b in 0.25 1 4; do
        r="$S/$D-$(run_of "$arm" "$p" "$b")/results"
        mkdir -p "$r/cells"
        ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
        ln -sfn "$GRID_SRC/crops" "$r/crops"
        cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$p" "$b")",
 "note": "#4910 arm $arm (${KNOBS[$arm]}), $p path, beta $b, 150 clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
        echo "$r"
      done
    done
  done
  ;;
idx)
  path_idx "${2:?path (${PATHS[*]})}" "${3:-$SEEDS}"
  ;;
launch|cpack|gpack)
  arm="${2:?arm (${ARMS[*]})}" p="${3:?path (${PATHS[*]})}" b="${4:?beta (0.25, 1 or 4)}"
  [[ -v "KNOBS[$arm]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  idx="${5:-$(path_idx "$p")}"
  run="$(run_of "$arm" "$p" "$b")"
  mem_in="${CALIB_MEM:-}"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-3G}" CALIB_TIME="${CALIB_TIME:-2:00:00}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=150
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_LABELS_SCALE CALIB_LABELS_SCALE_SHADOW CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_EARLY_CHECK
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  if [[ -n "${DIVERGES[$arm]}" ]]; then
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}${DIVERGES[$arm]}"
  fi
  if [[ "$1" == gpack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_CPUS=1 CALIB_MEM="${mem_in:-2G}"
    export CALIB_PACK_JOBS=1 CALIB_PACK_PAR="${CALIB_PACK_PAR:-32}" CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
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
  echo "usage: fold_transfer_4910.sh {dirs [ARM] | idx PATH [SEEDS] | {launch|cpack|gpack} ARM PATH BETA [IDX]}" >&2
  exit 1 ;;
esac
