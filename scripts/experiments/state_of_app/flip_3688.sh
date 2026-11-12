#!/usr/bin/env bash
# #3688: how much does ONE wrong vote change a run's outcome, at the position the loop serves it? Paired sessions,
# identical up to the flipped click (the vote draws no randomness):
#
#   ctl   the honest simulated user (the app's eval as shipped)
#   g1 g2 g5 g20   CALIB_FLIP_VOTE=good:K: the K-th true positive served is voted Bad (a hidden positive, the
#                  contamination the issue calls adversarial: the loop serves the most class-looking items first)
#   b1 b5          CALIB_FLIP_VOTE=bad:K: the K-th true negative served is voted Good (a wrong positive; on Region
#                  Photo a Good with no box is the whole-image vector, `region_box_for_category` returns None)
#
# Only the vote is wrong; the withheld half is scored on the truth. COCO Better at the app's balance (beta 1), the
# grid the 2026-11-03 review prepared (post-#5191), 144 cells x 5 seeds, seed-major, 150 clicks and the closing
# check, trajectory pass, cells gzipped. Binary Photo (siglip) is the first 144 of each seed's 288, Region Photo
# (siglip+dinov3_patch max_patch) the other 144.
#
#   flip_3688.sh dirs [PATH]                     run dirs <root>/<date>-flip3688<tag>-<arm>-<path>-b1
#   flip_3688.sh idx PATH [SEEDS]                the grid indices of PATH's cells, seed-major, comma-separated
#   flip_3688.sh cpack ARM PATH [IDX]            cpu-partition packs (CALIB_PACK_JOBS jobs of CALIB_PACK_PAR cells)
#   flip_3688.sh gpack ARM PATH [IDX]            one pack on a V100 node, the GPU hidden (outside the cpu cap)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: FLIP3688_DATE (default 2026-11-12), FLIP3688_SEEDS (5), FLIP3688_TAG (e.g. smoke), FLIP3688_GRID,
# FLIP3688_STEPS (150), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PACK_JOBS, CALIB_PACK_PAR,
# CALIB_PACK_TIME, CALIB_PACK_CPUS).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${FLIP3688_DATE:-2026-11-12}"
STEPS="${FLIP3688_STEPS:-150}"
ARMS=(ctl g1 g2 g5 g20 b1 b5)
PATHS=(binary region)
declare -A KNOBS=([ctl]="" [g1]="CALIB_FLIP_VOTE=good:1" [g2]="CALIB_FLIP_VOTE=good:2" [g5]="CALIB_FLIP_VOTE=good:5"
  [g20]="CALIB_FLIP_VOTE=good:20" [b1]="CALIB_FLIP_VOTE=bad:1" [b5]="CALIB_FLIP_VOTE=bad:5")
SEEDS="${FLIP3688_SEEDS:-5}"
GRID_SRC="${FLIP3688_GRID:-$S/2026-11-03-b1/results}"

run_of() { echo "flip3688${FLIP3688_TAG:-}-$1-$2-b1"; }
check() { [[ -v "KNOBS[$1]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }; }
offset() { case "$1" in binary) echo 0 ;; region) echo 144 ;; *) echo "path must be binary or region" >&2; exit 2 ;; esac; }

path_idx() {  # PATH [SEEDS]
  python3 - "$(offset "$1")" "${2:-$SEEDS}" <<'PY'
import sys
off, n = int(sys.argv[1]), int(sys.argv[2])
print(",".join(str(s * 288 + off + i) for s in range(n) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  [[ -s "$GRID_SRC/prepare_info.json" ]] || { echo "no prepared grid at $GRID_SRC" >&2; exit 2; }
  for p in ${2:-${PATHS[*]}}; do
    offset "$p" >/dev/null
    for arm in "${ARMS[@]}"; do
      r="$S/$D-$(run_of "$arm" "$p")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$p")",
 "note": "#3688 arm $arm (${KNOBS[$arm]:-honest user}), $p path, beta 1, $STEPS clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
idx)
  path_idx "${2:?path (${PATHS[*]})}" "${3:-}"
  ;;
cpack | gpack)
  arm="${2:?arm (${ARMS[*]})}" p="${3:?path (${PATHS[*]})}"
  check "$arm"
  idx="${4:-$(path_idx "$p")}"
  run="$(run_of "$arm" "$p")"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA=1 SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
  export CALIB_JOB_NAME="$run"
  # Today's app: no arm knob of an earlier study rides along from the caller's shell.
  unset CALIB_HEAD CALIB_LABELS_SCALE CALIB_LABELS_SCALE_SHADOW CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_WEAK_D
  unset CALIB_SPOT_CHECK CALIB_EARLY_CHECK CALIB_HEAD_COUNT_PRECISION_SCALE CALIB_HEAD_COUNT_PRECISION_EXPONENT
  unset CALIB_HEAD_COUNT_PRECISION_TOP_Z CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD
  unset CALIB_LINE_BOUND_MARGIN CALIB_SAVE_TEST_SCORES CALIB_SAVE_TEST_SCORES_AT CALIB_SAVE_FOLD_SCALES
  unset CALIB_HEAD_COUNT_BADS CALIB_CENTROID_LINE CALIB_CENTROID_LINE_VARIANTS CALIB_QUERY_TIER CALIB_QUERY_HOLD
  unset CALIB_LABEL_QUOTA CALIB_QUERY_MISS VTSEARCH_SVM_HEAD_C SOTA_EMBEDDERS SOTA_DATASET SOTA_POOL_PREVALENCE
  unset SOTA_CELL_PREVALENCE SOTA_PATCH_STYLES CALIB_SPAN_GREEN CALIB_NEW_WALK CALIB_MIN_GOOD CALIB_HARD_DRY_RUN
  unset CALIB_FLIP_VOTE
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  if [[ -n "${KNOBS[$arm]}" ]]; then
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}flip_vote"
  fi
  if [[ "$p" == region ]]; then
    export CALIB_MEM="${CALIB_MEM:-24G}"
  else
    export CALIB_MEM="${CALIB_MEM:-2G}"
  fi
  export CALIB_CPUS=1
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
  echo "usage: flip_3688.sh {dirs [PATH] | idx PATH [SEEDS] | {cpack|gpack} ARM PATH [IDX]}" >&2; exit 1 ;;
esac
