#!/usr/bin/env bash
# #5241: SAM3's zero-shot sort as the opening, an opt-in path beside SigLIP's (SigLIP stays the default: it is fast).
# #5213 found SAM3's set at its own 0.5 cut has F1 0.57 on Region Photo, against SigLIP's typed set 0.37. Two arms on
# #4711's grid, seed 0, Binary and Region cells (288), beta 1, 150 clicks, trajectory pass:
#
#   sig    today's app: SigLIP's typed sort opens, its set until the quota / hold / miss, then the head
#   sam3   SAM3's typed-prompt scores open (they order the opening's picks), its set at the 0.5 cut is what Test
#          gives through click 150 on both paths (CALIB_QUERY_HOLD past any session + CALIB_QUERY_HOLD_PATCH, no
#          miss), and the head trains and is scored all the same, its rows tagged hold:head. Picks never depend on
#          what Test shows, so every takeover rule is priced afterwards on these sessions.
#   sam3gmm  SAM3's other line, the GMM midpoint of its scores: one click on the Binary cells. The set is the same
#          at every click and on both paths (the withheld half is the cell's), so its one row prices the line.
#
#   sam3_opening_5241.sh dirs [ARM]            run dirs <root>/<date>-sam3open5241<tag>-<arm>-b1 on #4711's grid
#   sam3_opening_5241.sh preflight ARM [IDX]   the preflight a launch of the same cells would pass
#   sam3_opening_5241.sh launch ARM [IDX]      one arm as a cpu array (default: seed 0's 288 cells)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it), with the
# same SBATCH_CONSTRAINT for both arms (#4481). Env: S5241_DATE (2026-10-26), S5241_TAG, S5241_STEPS (150),
# S5241_GRID, SAM3_SCORES (#5213's shards), SOTA_ROOT, and anything launch.sh reads.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${S5241_DATE:-2026-10-26}"
STEPS="${S5241_STEPS:-150}"
ARMS=(sig sam3 sam3gmm)
GRID_SRC="${S5241_GRID:-$S/2026-10-23-regstyles4711-grid/results}"
SCORES="${SAM3_SCORES:-/expscratch/sgreenberg/sam3-5213/scores}"
run_of() { echo "sam3open5241${S5241_TAG:-}-$1-b1"; }
common_env() {  # ARM: today's app at beta 1, trajectory pass, plus the arm's opening
  local arm="$1"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-26G}" CALIB_TIME="${CALIB_TIME:-5:00:00}"
  export SOTA_BETA=1 SOTA_DATE="$D-$(run_of "$arm")" SOTA_PASS=trajectory SOTA_SEEDS=1 SOTA_MAX_STEPS="$STEPS"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150}"
  export SOTA_PATCH_STYLES=max_patch
  export CALIB_JOB_NAME="$(run_of "$arm")"
  # Today's app: no arm knob of an earlier study rides along from the caller's shell.
  unset CALIB_HEAD CALIB_LABELS_SCALE CALIB_LABELS_SCALE_SHADOW CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_EARLY_CHECK
  unset CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN
  unset CALIB_SAVE_TEST_SCORES CALIB_SAVE_TEST_SCORES_AT CALIB_SAVE_FOLD_SCALES CALIB_SPOT_CHECK CALIB_HEAD_COUNT_BADS
  unset CALIB_CENTROID_LINE CALIB_CENTROID_LINE_VARIANTS CALIB_QUERY_TIER CALIB_QUERY_HOLD CALIB_LABEL_QUOTA
  unset CALIB_QUERY_MISS VTSEARCH_SVM_HEAD_C SOTA_EMBEDDERS SOTA_DATASET SOTA_POOL_PREVALENCE SOTA_CELL_PREVALENCE
  unset CALIB_SEED_SAM3 CALIB_SEED_SAM3_PROMPTS CALIB_QUERY_LINE CALIB_QUERY_HOLD_PATCH CALIB_HANDOVER_SHADOW
  if [[ "$arm" == sam3* ]]; then
    export CALIB_SEED_SAM3="$SCORES" CALIB_SEED_SAM3_PROMPTS=typed CALIB_QUERY_LINE=fixed:0.5
    # The hold counts labels, check votes included, so it is set past any session's.
    export CALIB_QUERY_HOLD=100000 CALIB_QUERY_HOLD_PATCH=1 CALIB_QUERY_MISS=0
    # #4776's evidence (held-out Goods' ranks under the head), for a takeover rule; single-vector cells only.
    export CALIB_HANDOVER_SHADOW=1
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}seed_sam3,query_line,query_hold_patch"
  fi
  if [[ "$arm" == sam3gmm ]]; then
    export CALIB_QUERY_LINE=gmm SOTA_MAX_STEPS=1
  fi
}
check_arm() { [[ " ${ARMS[*]} " == *" $1 "* ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }; }
case "${1:-}" in
dirs)
  [[ -s "$GRID_SRC/prepare_info.json" ]] || { echo "no prepared grid at $GRID_SRC" >&2; exit 2; }
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    r="$S/$D-$(run_of "$arm")/results"
    st="$STEPS" n=288
    [[ "$arm" == sam3gmm ]] && st=1 n=144
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": $n, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": 1,
 "max_steps": $st, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm")",
 "note": "#5241 arm $arm, Binary + Region (max_patch), beta 1, $st clicks, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
preflight)
  # launch.sh's redo runs no preflight: run it on the exact env a launch would submit with (nothing is queued).
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
launch)
  arm="${2:?arm (${ARMS[*]})}"
  check_arm "$arm"
  common_env "$arm"
  export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}"
  default=0-287
  [[ "$arm" == sam3gmm ]] && default=0-143
  exec bash "$HERE/launch.sh" redo "${3:-$default}"
  ;;
*)
  sed -n '2,19p' "$0"
  exit 2
  ;;
esac
