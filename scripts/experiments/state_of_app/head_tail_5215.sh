#!/usr/bin/env bash
# #5215: the head's count line at a balance of 1 and below, gated on how far the head's top stands out, closed loop.
# #5084 shipped 0.71 beta times what stands out in place of #4603's beta ** 0.708. Offline (head_tail_5215.py) that
# costs Binary Photo at 1% prevalence, on heads whose positives sit under the bulk's heavy tail; keeping #4603's count
# where the top logit stands under 6.1 bulk spreads recovers most of it there. The head's count line is Train's line
# too while it serves, so the sessions themselves change: this prices them end to end, on Binary Photo (the bench the
# gate moves; Region's tops stand above 6.1 on 85% of heads). Beta 4 is untouched and does not run.
#
#   ctl      the app: 0.71 beta on every head (CALIB_HEAD_COUNT_PRECISION_TOP_Z=off)
#   tz61     #4603's count where the top stands under 6.1 bulk spreads (CALIB_HEAD_COUNT_PRECISION_TOP_Z=6.1)
#
# The grid #4711 prepared after #5191 (2026-10-23-regstyles4711-grid), Binary Photo's cells (siglip, the first 144 of
# each seed's 288), paired by cell and seed as #5084's closed loop ran: 200 clicks and the closing check, trajectory
# pass (no ceiling), cells gzipped.
#
#   head_tail_5215.sh dirs                  run dirs <root>/<date>-headtail5215<tag>-<arm>-binary-b{025,1}
#   head_tail_5215.sh cpack ARM BETA [IDX]  cpu-partition packs (CALIB_PACK_JOBS jobs of CALIB_PACK_PAR cells)
#   head_tail_5215.sh idx [SEEDS]           the grid indices of Binary Photo's cells, seed-major
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: HEADTAIL5215_DATE (default 2026-10-25), HEADTAIL5215_SEEDS (5), HEADTAIL5215_TAG (e.g. smoke),
# HEADTAIL5215_GRID, HEADTAIL5215_STEPS (200), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PACK_JOBS,
# CALIB_PACK_PAR, CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${HEADTAIL5215_DATE:-2026-10-25}"
SEEDS="${HEADTAIL5215_SEEDS:-5}"
STEPS="${HEADTAIL5215_STEPS:-200}"
ARMS=(ctl tz61)
declare -A KNOBS=([ctl]="CALIB_HEAD_COUNT_PRECISION_TOP_Z=off" [tz61]="CALIB_HEAD_COUNT_PRECISION_TOP_Z=6.1")
GRID_SRC="${HEADTAIL5215_GRID:-$S/2026-10-23-regstyles4711-grid/results}"

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; *) echo "beta must be 0.25 or 1" >&2; exit 2 ;; esac; }
run_of() { echo "headtail5215${HEADTAIL5215_TAG:-}-$1-binary-b$(btag "$2")"; }

path_idx() {  # [SEEDS]
  python3 - "${1:-$SEEDS}" <<'PY'
import sys
n = int(sys.argv[1])
print(",".join(str(s * 288 + i) for s in range(n) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  for arm in "${ARMS[@]}"; do
    for b in 0.25 1; do
      r="$S/$D-$(run_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$b")",
 "note": "#5215 arm $arm (${KNOBS[$arm]}), binary path, beta $b, $STEPS clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
idx)
  path_idx "${2:-$SEEDS}"
  ;;
cpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (0.25 or 1)}"
  [[ -v "KNOBS[$arm]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  idx="${4:-$(path_idx)}"
  run="$(run_of "$arm" "$b")"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150,200}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_HEAD_COUNT_PRECISION_SCALE CALIB_HEAD_COUNT_PRECISION_EXPONENT CALIB_HEAD_COUNT_PRECISION_TOP_Z
  unset CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN
  unset CALIB_SAVE_TEST_SCORES CALIB_SAVE_FOLD_SCALES CALIB_SAVE_TEST_SCORES_AT CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T
  unset CALIB_WEAK_D CALIB_SPOT_CHECK CALIB_EARLY_CHECK CALIB_LABELS_SCALE CALIB_HEAD_COUNT_BADS
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}head_count_precision_top_z"
  export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}" CALIB_MEM="${CALIB_MEM:-2G}"
  export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}"
  export CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
  unset CALIB_PACK_GRES
  exec bash "$HERE/launch.sh" pack "$idx"
  ;;
*)
  echo "usage: head_tail_5215.sh {dirs | idx [SEEDS] | cpack ARM BETA [IDX]}" >&2; exit 1 ;;
esac
