#!/usr/bin/env bash
# #4278: does the converged logistic head still starve Region voting on easy classes? #4213 (2026-09-28) found its
# acquisition cut drifting down as Bad images flooded ~200 region rows each into the fit (score 0.47 -> 0.34 against
# the SVM's 0.50 -> 0.42), so its Hard picks landed among negatives (Good 2.4% against 7.1% by clicks 101-150) and
# easy classes found 11.1 positives by click 150 against 18.8. Acquisition has moved twice since: the line stopped
# falling back (#4295), then the cut became the labels line's 0.5 posterior (#3546), a class model fitted to the
# head's own scores. The shipped head moved too: the SVM started from the Goods' centroid (#4785). This re-measures
# the two heads on today's app before anything is built for the old cut. Every arm is today's app to 150 clicks and
# differs only in the head:
#
#   ctl  the shipped head (CALIB_HEAD unset -> PRODUCTION_HEAD, linear_svm_prior)
#   lr   LogisticRegression(C = 1, balanced, L2) fitted to convergence (CALIB_HEAD=linear_logreg, #4114)
#
# Region Photo of the grid #4743 priced (2026-10-08-b1): siglip+dinov3_patch max_patch, the second 144 cells of each
# seed's 288, paired by cell and seed. Trajectory pass (no ceiling); cells gzipped.
#
#   logreg_region_4278.sh dirs [ARM]                run dirs <root>/<date>-logreg4278<tag>-<arm>-region-b{025,1,4}
#   logreg_region_4278.sh idx [SEEDS]               the grid indices of Region's cells, seed-major
#   logreg_region_4278.sh launch ARM BETA [IDX]     one arm and preset as a cpu array (default: every seed's cells)
#   logreg_region_4278.sh gpack ARM BETA [IDX]      the same as one pack on a V100/DGX node, the GPU hidden: the gpu
#                                                   partition is outside the cpu cap (4 GPUs per user)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it). Give each
# cell's two arms the same CPU vendor (SBATCH_CONSTRAINT), so BLAS drift (#4481) is not part of the paired gap.
# Env: LOGREG4278_DATE (default 2026-10-18), LOGREG4278_SEEDS (3), LOGREG4278_TAG (e.g. smoke), LOGREG4278_STEPS
# (150), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_TIME, CALIB_PACK_PAR, CALIB_PACK_MEM,
# CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${LOGREG4278_DATE:-2026-10-18}"
SEEDS="${LOGREG4278_SEEDS:-3}"
STEPS="${LOGREG4278_STEPS:-150}"
ARMS=(ctl lr)
declare -A KNOBS=([ctl]="" [lr]="CALIB_HEAD=linear_logreg")
declare -A DIVERGES=([ctl]="" [lr]="head")
GRID_SRC="${LOGREG4278_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since (a docstring only)

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "logreg4278${LOGREG4278_TAG:-}-$1-region-b$(btag "$2")"; }

region_idx() {  # [SEEDS]
  python3 - "${1:-$SEEDS}" <<'PY'
import sys
print(",".join(str(s * 288 + 144 + i) for s in range(int(sys.argv[1])) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    for b in 0.25 1 4; do
      r="$S/$D-$(run_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$b")",
 "note": "#4278 arm $arm (${KNOBS[$arm]:-the shipped head}), region path, beta $b, $STEPS clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
idx)
  region_idx "${2:-$SEEDS}"
  ;;
launch|gpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (0.25, 1 or 4)}"
  [[ -v "KNOBS[$arm]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  idx="${4:-$(region_idx)}"
  run="$(run_of "$arm" "$b")"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-26G}" CALIB_TIME="${CALIB_TIME:-5:00:00}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_HEAD CALIB_LABELS_SCALE CALIB_LABELS_SCALE_SHADOW CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_EARLY_CHECK
  unset CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN
  unset CALIB_SAVE_TEST_SCORES CALIB_SAVE_TEST_SCORES_AT CALIB_SAVE_FOLD_SCALES CALIB_SPOT_CHECK CALIB_HEAD_COUNT_BADS
  unset VTSEARCH_SVM_HEAD_C
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  if [[ -n "${DIVERGES[$arm]}" ]]; then
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}${DIVERGES[$arm]}"
  fi
  if [[ "$1" == gpack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_CPUS=1
    export CALIB_PACK_JOBS=1 CALIB_PACK_PAR="${CALIB_PACK_PAR:-28}" CALIB_TIME="${CALIB_PACK_TIME:-16:00:00}"
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}"
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: logreg_region_4278.sh {dirs [ARM] | idx [SEEDS] | {launch|gpack} ARM BETA [IDX]}" >&2
  exit 1 ;;
esac
