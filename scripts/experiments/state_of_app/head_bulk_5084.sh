#!/usr/bin/env bash
# #5084: the head's count line at a balance of 1 and below, closed loop. While the class model holds fewer than 6
# held-out Goods the labels line is the head's count (#4786, #4798): beta ** 0.708 times what stands out in the head's
# scores, #4603's count of the logits above median + 4 spreads less a Gaussian bulk's share. A weak head's bulk is
# heavier than a Gaussian above 3 spreads, so the count reads the bulk's tail as positives: offline
# (head_bulk_5084.py) Region Photo's line keeps 2.6x its positives at beta 1, and a smaller multiplier gains on Region
# Photo at every prevalence while Binary Photo and FHIBE faces hold. The head's count line is Train's line too while
# it serves (Autopilot's acquisition reads it), so the sessions themselves change: this prices them end to end. The
# arms read beta <= 1 only, so beta 4 is today's session bit for bit and does not run.
#
#   ctl      the app: CALIB_HEAD_COUNT_PRECISION_SCALE=off CALIB_HEAD_COUNT_PRECISION_EXPONENT=off
#   s71e1    0.71 * beta times what stands out (CALIB_HEAD_COUNT_PRECISION_SCALE=0.71, _EXPONENT=1)
#   s50e1    0.5 * beta
#
# The grid #4711 prepared after #5191 (2026-10-23-regstyles4711-grid): Binary Photo (siglip, the first 144 cells of
# each seed's 288) and Region Photo (siglip+dinov3_patch max_patch, the other 144), paired by cell and seed, as #5039's
# closed loop ran: Binary Photo to 200 clicks and the closing check, Region Photo (a cell is ~1 h and ~24 GB) to 150.
# Trajectory pass (no ceiling); cells gzipped.
#
#   head_bulk_5084.sh dirs                          run dirs <root>/<date>-headbulk5084<tag>-<arm>-<path>-b{025,1}
#   head_bulk_5084.sh {cpack|gpack} ARM PATH BETA [IDX]
#                                                   cpu-partition packs (CALIB_PACK_JOBS jobs of CALIB_PACK_PAR cells
#                                                   at CALIB_CPUS thread each), or packs on V100 nodes (GPU hidden)
#   head_bulk_5084.sh idx PATH [SEEDS]              the grid indices of a path's cells, seed-major
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: HEADBULK5084_DATE (default 2026-10-23), HEADBULK5084_SEEDS (5), HEADBULK5084_TAG (e.g. smoke),
# HEADBULK5084_GRID, HEADBULK5084_STEPS_BINARY (200), HEADBULK5084_STEPS_REGION (150), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PACK_JOBS, CALIB_PACK_PAR,
# CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${HEADBULK5084_DATE:-2026-10-23}"
SEEDS="${HEADBULK5084_SEEDS:-5}"
PATHS=(binary region)
ARMS=(ctl s71e1 s50e1)
declare -A KNOBS=([ctl]="CALIB_HEAD_COUNT_PRECISION_SCALE=off CALIB_HEAD_COUNT_PRECISION_EXPONENT=off"
  [s71e1]="CALIB_HEAD_COUNT_PRECISION_SCALE=0.71 CALIB_HEAD_COUNT_PRECISION_EXPONENT=1"
  [s50e1]="CALIB_HEAD_COUNT_PRECISION_SCALE=0.5 CALIB_HEAD_COUNT_PRECISION_EXPONENT=1")
GRID_SRC="${HEADBULK5084_GRID:-$S/2026-10-23-regstyles4711-grid/results}"

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; *) echo "beta must be 0.25 or 1" >&2; exit 2 ;; esac; }
offset() { case "$1" in binary) echo 0 ;; region) echo 144 ;; *) echo "path must be binary or region" >&2; exit 2 ;; esac; }
steps_of() { case "$1" in binary) echo "${HEADBULK5084_STEPS_BINARY:-200}" ;; region) echo "${HEADBULK5084_STEPS_REGION:-150}" ;; esac; }
run_of() { echo "headbulk5084${HEADBULK5084_TAG:-}-$1-$2-b$(btag "$3")"; }

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
    for p in "${PATHS[@]}"; do
      for b in 0.25 1; do
        r="$S/$D-$(run_of "$arm" "$p" "$b")/results"
        mkdir -p "$r/cells"
        ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
        ln -sfn "$GRID_SRC/crops" "$r/crops"
        cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $(steps_of "$p"), "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$p" "$b")",
 "note": "#5084 arm $arm (${KNOBS[$arm]}), $p path, beta $b, $(steps_of "$p") clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
        echo "$r"
      done
    done
  done
  ;;
idx)
  path_idx "${2:?path (${PATHS[*]})}" "${3:-$SEEDS}"
  ;;
cpack|gpack)
  arm="${2:?arm (${ARMS[*]})}" p="${3:?path (${PATHS[*]})}" b="${4:?beta (0.25 or 1)}"
  [[ -v "KNOBS[$arm]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  idx="${5:-$(path_idx "$p")}"
  run="$(run_of "$arm" "$p" "$b")"
  mem_in="${CALIB_MEM:-}"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$(steps_of "$p")"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150,200}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_HEAD_COUNT_PRECISION_SCALE CALIB_HEAD_COUNT_PRECISION_EXPONENT CALIB_HEAD_COUNT_RECALL_EXPONENT
  unset CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN CALIB_SAVE_TEST_SCORES CALIB_SAVE_FOLD_SCALES
  unset CALIB_SAVE_TEST_SCORES_AT CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_WEAK_D CALIB_SPOT_CHECK CALIB_EARLY_CHECK
  unset CALIB_LABELS_SCALE CALIB_HEAD_COUNT_BADS
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}head_count_precision_scale,head_count_precision_exponent"
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
  echo "usage: head_bulk_5084.sh {dirs | idx PATH [SEEDS] | {cpack|gpack} ARM PATH BETA [IDX]}" >&2; exit 1 ;;
esac
