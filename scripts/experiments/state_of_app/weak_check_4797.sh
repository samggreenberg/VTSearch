#!/usr/bin/env bash
# #4797: the weak-separation check at today's head, on COCO Better's Binary and Region paths. #4778 found it costs face
# sessions, which hold all their Goods by the first check, while #4496's Binary pairs paid most where few Goods were in
# hand; Region prompts in most sessions with few Goods, and the check has never been priced there. Paired sessions:
#
#   app      the app as shipped (the weak check once a session, the Done check, the closing check)
#   end      CALIB_SPOT_CHECK=end: no weak check; the Done check and the closing check stay
#
# The grid #4711 prepared after #5191 (2026-10-23-regstyles4711-grid), each seed's 288 cells seed-major: Binary Photo's
# 144 (siglip) then Region Photo's 144 (siglip+dinov3_patch). 150 clicks and the closing check, trajectory pass (no
# ceiling), cells gzipped.
#
#   weak_check_4797.sh dirs                       run dirs <root>/<date>-weak4797<tag>-<arm>-<path>-b{025,1,4}
#   weak_check_4797.sh idx PATH [SEEDS]           the grid indices of one path's cells, seed-major, comma-separated
#   weak_check_4797.sh cpack ARM PATH BETA [IDX]  cpu-partition packs (CALIB_PACK_JOBS jobs of CALIB_PACK_PAR cells)
#   weak_check_4797.sh gpack ARM PATH BETA [IDX]  one pack on a V100 node, the GPU hidden (outside the cpu cap)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: WEAK4797_DATE (default 2026-10-28), WEAK4797_SEEDS_BINARY (5), WEAK4797_SEEDS_REGION (1), WEAK4797_TAG (e.g.
# smoke), WEAK4797_GRID, WEAK4797_STEPS (150), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PACK_JOBS,
# CALIB_PACK_PAR, CALIB_PACK_TIME, CALIB_PACK_CPUS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${WEAK4797_DATE:-2026-10-28}"
STEPS="${WEAK4797_STEPS:-150}"
ARMS=(app end)
PATHS=(binary region)
BETAS=(0.25 1 4)
declare -A KNOBS=([app]="" [end]="CALIB_SPOT_CHECK=end")
declare -A OFFSET=([binary]=0 [region]=144)
declare -A SEEDS=([binary]="${WEAK4797_SEEDS_BINARY:-5}" [region]="${WEAK4797_SEEDS_REGION:-1}")
# A Region cell holds every image's patch rows (#4711's smoke: ~24 GB); a Binary cell fits in 2 GB.
declare -A MEM=([binary]=2G [region]=26G)
GRID_SRC="${WEAK4797_GRID:-$S/2026-10-23-regstyles4711-grid/results}"

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "weak4797${WEAK4797_TAG:-}-$1-$2-b$(btag "$3")"; }
check() {  # ARM PATH
  [[ -v "KNOBS[$1]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  [[ -v "OFFSET[$2]" ]] || { echo "path must be one of ${PATHS[*]}" >&2; exit 2; }
}

path_idx() {  # PATH [SEEDS]
  python3 - "${OFFSET[$1]}" "${2:-${SEEDS[$1]}}" <<'PY'
import sys
off, n = int(sys.argv[1]), int(sys.argv[2])
print(",".join(str(s * 288 + off + i) for s in range(n) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  [[ -s "$GRID_SRC/prepare_info.json" ]] || { echo "no prepared grid at $GRID_SRC" >&2; exit 2; }
  for arm in "${ARMS[@]}"; do
    for p in "${PATHS[@]}"; do
      for b in "${BETAS[@]}"; do
        r="$S/$D-$(run_of "$arm" "$p" "$b")/results"
        mkdir -p "$r/cells"
        ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
        ln -sfn "$GRID_SRC/crops" "$r/crops"
        cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": ${SEEDS[$p]},
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$p" "$b")",
 "note": "#4797 arm $arm (${KNOBS[$arm]:-as shipped}), $p path, beta $b, $STEPS clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
        echo "$r"
      done
    done
  done
  ;;
idx)
  p="${2:?path (${PATHS[*]})}"
  check app "$p"
  path_idx "$p" "${3:-}"
  ;;
cpack | gpack)
  arm="${2:?arm (${ARMS[*]})}" p="${3:?path (${PATHS[*]})}" b="${4:?beta (${BETAS[*]})}"
  check "$arm" "$p"
  idx="${5:-$(path_idx "$p")}"
  run="$(run_of "$arm" "$p" "$b")"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="${SEEDS[$p]}" SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
  export CALIB_JOB_NAME="$run"
  # Today's app: no arm knob of an earlier study rides along from the caller's shell.
  unset CALIB_HEAD CALIB_LABELS_SCALE CALIB_LABELS_SCALE_SHADOW CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_WEAK_D
  unset CALIB_SPOT_CHECK CALIB_EARLY_CHECK CALIB_HEAD_COUNT_PRECISION_SCALE CALIB_HEAD_COUNT_PRECISION_EXPONENT
  unset CALIB_HEAD_COUNT_PRECISION_TOP_Z CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD
  unset CALIB_LINE_BOUND_MARGIN CALIB_SAVE_TEST_SCORES CALIB_SAVE_TEST_SCORES_AT CALIB_SAVE_FOLD_SCALES
  unset CALIB_HEAD_COUNT_BADS CALIB_CENTROID_LINE CALIB_CENTROID_LINE_VARIANTS CALIB_QUERY_TIER CALIB_QUERY_HOLD
  unset CALIB_LABEL_QUOTA CALIB_QUERY_MISS VTSEARCH_SVM_HEAD_C SOTA_EMBEDDERS SOTA_DATASET SOTA_POOL_PREVALENCE
  unset SOTA_CELL_PREVALENCE SOTA_PATCH_STYLES
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  if [[ -n "${KNOBS[$arm]}" ]]; then
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}spot_check"
  fi
  export CALIB_CPUS=1 CALIB_MEM="${CALIB_MEM:-${MEM[$p]}}"
  if [[ "$1" == gpack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}"
    export CALIB_PACK_JOBS=1 CALIB_PACK_PAR="${CALIB_PACK_PAR:-12}" CALIB_TIME="${CALIB_PACK_TIME:-24:00:00}"
    # A core per cell (the V100 nodes are hyperthreaded) and memory over both NUMA nodes (#4711).
    export CALIB_PACK_CPUS="${CALIB_PACK_CPUS:-$((2 * CALIB_PACK_PAR))}"
    export CALIB_PACK_INTERLEAVE="${CALIB_PACK_INTERLEAVE:-1}"
  else
    export CALIB_PARTITION=cpu CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}"
    export CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
    export CALIB_PACK_INTERLEAVE="${CALIB_PACK_INTERLEAVE:-1}"
    unset CALIB_PACK_GRES
  fi
  exec bash "$HERE/launch.sh" pack "$idx"
  ;;
*)
  echo "usage: weak_check_4797.sh {dirs | idx PATH [SEEDS] | {cpack|gpack} ARM PATH BETA [IDX]}" >&2; exit 1 ;;
esac
