#!/usr/bin/env bash
# #5473: does Autopilot need its Diversity step? Today Done waits on Span too (green at 40 evidence-bearing Coverage
# Atlas nodes) or on the labels line's level (#4878). Skip the step and Done waits on Smart and Stable alone, so a
# session stops sooner; this prices what it loses. Paired sessions:
#
#   app      the app as shipped (autopilot_goal_diversity 40)
#   nodiv    CALIB_SPAN_GREEN=0: Span green from the first vote, so no New (Diversity) phase
#
# COCO Better's Binary path (siglip), the grid #4711 prepared after #5191 (2026-10-23-regstyles4711-grid), 144 cells
# x 5 seeds, seed-major. 150 clicks and the closing check, trajectory pass (no ceiling), cells gzipped. The face side
# is fhibe/sota_face.sh with CALIB_SPAN_GREEN=0.
#
#   skip_diversity_5473.sh dirs                     run dirs <root>/<date>-nodiv5473<tag>-<arm>-b{025,1,4}
#   skip_diversity_5473.sh idx [SEEDS]              the grid indices of the Binary cells, seed-major, comma-separated
#   skip_diversity_5473.sh cpack ARM BETA [IDX]     cpu-partition packs (CALIB_PACK_JOBS jobs of CALIB_PACK_PAR cells)
#   skip_diversity_5473.sh gpack ARM BETA [IDX]     one pack on a V100 node, the GPU hidden (outside the cpu cap)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: NODIV5473_DATE (default 2026-10-30), NODIV5473_SEEDS (5), NODIV5473_TAG (e.g. smoke), NODIV5473_GRID,
# NODIV5473_STEPS (150), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PACK_JOBS, CALIB_PACK_PAR,
# CALIB_PACK_TIME, CALIB_PACK_CPUS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${NODIV5473_DATE:-2026-10-30}"
STEPS="${NODIV5473_STEPS:-150}"
ARMS=(app nodiv)
BETAS=(0.25 1 4)
declare -A KNOBS=([app]="" [nodiv]="CALIB_SPAN_GREEN=0")
SEEDS="${NODIV5473_SEEDS:-5}"
GRID_SRC="${NODIV5473_GRID:-$S/2026-10-23-regstyles4711-grid/results}"

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "nodiv5473${NODIV5473_TAG:-}-$1-b$(btag "$2")"; }
check() { [[ -v "KNOBS[$1]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }; }

# The Binary path's cells: the first 144 of each seed's 288 (Region's follow, and this study does not run them).
binary_idx() {  # [SEEDS]
  python3 - "${1:-$SEEDS}" <<'PY'
import sys
print(",".join(str(s * 288 + i) for s in range(int(sys.argv[1])) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  [[ -s "$GRID_SRC/prepare_info.json" ]] || { echo "no prepared grid at $GRID_SRC" >&2; exit 2; }
  for arm in "${ARMS[@]}"; do
    for b in "${BETAS[@]}"; do
      r="$S/$D-$(run_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$b")",
 "note": "#5473 arm $arm (${KNOBS[$arm]:-as shipped}), binary path, beta $b, $STEPS clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
idx)
  binary_idx "${2:-}"
  ;;
cpack | gpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (${BETAS[*]})}"
  check "$arm"
  idx="${4:-$(binary_idx)}"
  run="$(run_of "$arm" "$b")"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
  export CALIB_JOB_NAME="$run"
  # Today's app: no arm knob of an earlier study rides along from the caller's shell.
  unset CALIB_HEAD CALIB_LABELS_SCALE CALIB_LABELS_SCALE_SHADOW CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_WEAK_D
  unset CALIB_SPOT_CHECK CALIB_EARLY_CHECK CALIB_HEAD_COUNT_PRECISION_SCALE CALIB_HEAD_COUNT_PRECISION_EXPONENT
  unset CALIB_HEAD_COUNT_PRECISION_TOP_Z CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD
  unset CALIB_LINE_BOUND_MARGIN CALIB_SAVE_TEST_SCORES CALIB_SAVE_TEST_SCORES_AT CALIB_SAVE_FOLD_SCALES
  unset CALIB_HEAD_COUNT_BADS CALIB_CENTROID_LINE CALIB_CENTROID_LINE_VARIANTS CALIB_QUERY_TIER CALIB_QUERY_HOLD
  unset CALIB_LABEL_QUOTA CALIB_QUERY_MISS VTSEARCH_SVM_HEAD_C SOTA_EMBEDDERS SOTA_DATASET SOTA_POOL_PREVALENCE
  unset SOTA_CELL_PREVALENCE SOTA_PATCH_STYLES CALIB_SPAN_GREEN CALIB_NEW_WALK
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  if [[ -n "${KNOBS[$arm]}" ]]; then
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}span_green"
  fi
  export CALIB_CPUS=1 CALIB_MEM="${CALIB_MEM:-2G}"
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
  echo "usage: skip_diversity_5473.sh {dirs | idx [SEEDS] | {cpack|gpack} ARM BETA [IDX]}" >&2; exit 1 ;;
esac
