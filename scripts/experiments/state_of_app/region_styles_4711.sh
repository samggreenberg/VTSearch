#!/usr/bin/env bash
# #4711: on the Region path, does a raw-patch HAC tree beat today's tree-free MaxPatch? #2886 shipped MaxPatch and
# deleted the production HAC tree (MaxHAC) in August, on a study whose tree arms lost only at the operating point
# (docs/experiments/2026-07-29-max-patch/REPORT.md), under a threshold stack that has since been rebuilt (#2895,
# closed into this). The raw-patch-leaf trees survive as eval styles (vtscore/eval/patch_styles.py) and run through
# the same train / calibrate / score path as max_patch, so every arm is today's app to 150 clicks and differs only
# in the region geometry:
#
#   mp   the shipped geometry (max_patch): nearest raw patch for a Good, every raw patch flooded for a Bad
#   hac  max_patch_hac: a Good snaps to the best-matching node of a raw-patch-leaf HAC tree; every node flooded
#   pca  max_patch_pca_hac: the same tree, merge order decided in a 32-dim per-image PCA space
#
# MaxHAC itself is not an arm (owner, 2026-10-23: not re-implemented). SigLIP binary is the reference: the mp arm
# also runs each seed's Binary cells, on the same grid and seeds.
#
# Region Photo on coco_better as rebuilt by #5191 (the #4954 corrections): a grid prepared for this study, 144
# Region cells a seed, paired by cell and seed. Beta 1, 3 seeds, trajectory pass (no ceiling); cells gzipped.
#
#   region_styles_4711.sh grid                        prepare the grid (one cpu job) into <root>/<date>-regstyles4711-grid
#   region_styles_4711.sh dirs [ARM]                  run dirs <root>/<date>-regstyles4711<tag>-<arm>-region-b1
#   region_styles_4711.sh idx [SEEDS]                 the grid indices of Region's cells, seed-major
#   region_styles_4711.sh binidx [SEEDS]              the grid indices of Binary's cells (the mp arm's reference)
#   region_styles_4711.sh preflight ARM [IDX]         the preflight a launch of the same cells would pass (launch
#                                                     runs launch.sh redo, which runs none)
#   region_styles_4711.sh launch ARM [IDX]            one arm as a cpu array (default: every seed's Region cells)
#   region_styles_4711.sh gpack ARM [IDX]             the same as one pack on a V100 node, the GPU hidden: the gpu
#                                                     partition is outside the cpu cap (4 GPUs per user)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it). Give each
# cell's arms the same CPU vendor (SBATCH_CONSTRAINT), so BLAS drift (#4481) is not part of the paired gap.
# Env: RS4711_DATE (default 2026-10-23), RS4711_SEEDS (3), RS4711_TAG (e.g. smoke), RS4711_STEPS (150), RS4711_GRID,
# SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_TIME, CALIB_PACK_PAR, CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${RS4711_DATE:-2026-10-23}"
SEEDS="${RS4711_SEEDS:-3}"
STEPS="${RS4711_STEPS:-150}"
ARMS=(mp hac pca)
declare -A STYLE=([mp]=max_patch [hac]=max_patch_hac [pca]=max_patch_pca_hac)
# A tree carries 2N-1 rows an image against max_patch's N+1, so its cells are sized up (measured on the smoke run).
declare -A MEM=([mp]=26G [hac]=48G [pca]=48G)
GRID_NAME="$D-regstyles4711-grid"
GRID_SRC="${RS4711_GRID:-$S/$GRID_NAME/results}"

run_of() { echo "regstyles4711${RS4711_TAG:-}-$1-region-b1"; }

half_idx() {  # OFFSET SEEDS: one path's 144 cells of each seed's 288 (seed-major: 144 Binary, then 144 Region)
  python3 - "$1" "$2" <<'PY'
import sys
off, n = int(sys.argv[1]), int(sys.argv[2])
print(",".join(f"{s * 288 + off}-{s * 288 + off + 143}" for s in range(n)))
PY
}

common_env() {  # ARM: today's app, the arm's style, beta 1, trajectory pass
  local arm="$1"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-${MEM[$arm]}}" CALIB_TIME="${CALIB_TIME:-5:00:00}"
  export SOTA_BETA=1 SOTA_DATE="$D-$(run_of "$arm")" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
  export SOTA_PATCH_STYLES="${STYLE[$arm]}"
  export CALIB_JOB_NAME="$(run_of "$arm")"
  # Today's app: no arm knob of an earlier study rides along from the caller's shell.
  unset CALIB_HEAD CALIB_LABELS_SCALE CALIB_LABELS_SCALE_SHADOW CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_EARLY_CHECK
  unset CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN
  unset CALIB_SAVE_TEST_SCORES CALIB_SAVE_TEST_SCORES_AT CALIB_SAVE_FOLD_SCALES CALIB_SPOT_CHECK CALIB_HEAD_COUNT_BADS
  unset CALIB_CENTROID_LINE CALIB_CENTROID_LINE_VARIANTS CALIB_QUERY_TIER CALIB_QUERY_HOLD CALIB_LABEL_QUOTA
  unset CALIB_QUERY_MISS VTSEARCH_SVM_HEAD_C SOTA_EMBEDDERS SOTA_DATASET SOTA_POOL_PREVALENCE SOTA_CELL_PREVALENCE
  if [[ "${STYLE[$arm]}" != max_patch ]]; then
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}patch_style"
  fi
}

check_arm() { [[ -v "STYLE[$1]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }; }

case "${1:-}" in
grid)
  # The grid is selection + crops only; every run reads the live cells. One per study, prepared after #5191's rebuild.
  unset SOTA_BETA SOTA_PATCH_STYLES SOTA_EMBEDDERS SOTA_DATASET
  export SOTA_DATE="$GRID_NAME" SOTA_SEEDS="$SEEDS"
  exec bash "$HERE/launch.sh" prepare
  ;;
dirs)
  [[ -s "$GRID_SRC/prepare_info.json" ]] || { echo "no prepared grid at $GRID_SRC (run: $0 grid)" >&2; exit 2; }
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    r="$S/$D-$(run_of "$arm")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm")",
 "note": "#4711 arm $arm (${STYLE[$arm]}), region path, beta 1, $STEPS clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
idx)
  half_idx 144 "${2:-$SEEDS}"
  ;;
binidx)
  half_idx 0 "${2:-$SEEDS}"
  ;;
preflight)
  # launch.sh's redo runs no preflight: run it on the exact env a launch would submit with, by standing a function in
  # for sbatch (nothing is queued).
  sbatch() {
    local a name="" mem="" arr="" spec n=0 r
    for a in "$@"; do
      case "$a" in --job-name=*) name="${a#*=}" ;; --mem=*) mem="${a#*=}" ;; --array=*) arr="${a#*=}" ;; esac
    done
    spec="${arr%%\%*}"
    for r in ${spec//,/ }; do
      if [[ "$r" == *-* ]]; then n=$((n + ${r#*-} - ${r%-*} + 1)); else n=$((n + 1)); fi
    done
    bash "$VTS_REPO/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod --job-name "$name" \
      --mem "$mem" --conc "${arr##*%}" --array-tasks "$n" --patch --require-text-seed \
      --require-region-voting "coco_better:siglip+dinov3_patch" >&2 && echo 0
  }
  export -f sbatch
  exec bash "$0" launch "${@:2}"
  ;;
launch|gpack)
  arm="${2:?arm (${ARMS[*]})}"
  check_arm "$arm"
  idx="${3:-$(half_idx 144 "$SEEDS")}"
  common_env "$arm"
  if [[ "$1" == gpack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_CPUS=1
    export CALIB_PACK_JOBS=1 CALIB_PACK_PAR="${CALIB_PACK_PAR:-12}" CALIB_TIME="${CALIB_PACK_TIME:-16:00:00}"
    # A core per cell: the V100 nodes are hyperthreaded, and two cells on one core each ran ~3x slower than the smoke.
    export CALIB_PACK_CPUS="${CALIB_PACK_CPUS:-$((2 * CALIB_PACK_PAR))}"
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}"
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: region_styles_4711.sh {grid | dirs [ARM] | idx [SEEDS] | binidx [SEEDS] | {preflight|launch|gpack} ARM [IDX]}" >&2
  exit 1 ;;
esac
