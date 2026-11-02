#!/usr/bin/env bash
# #5490: a class no model knows. Someone wants a 2028 Ford Bar, not a 2027 Ford Foo; no text tower or detector has
# seen either, so they type the general class (or a name nothing knows) and vote. FGVC-Aircraft's sibling variants
# (`fgvc_aircraft`: 46 variants in 16 families, plus 5,400 other aircraft and 10,000 airplane-free COCO images) are
# the stand-in. Does SAM3 suffer more there than SigLIP?
#
#   unknown_class_5490.sh prepare               select the 46 cells and cache crops, into the grid dir (one CPU job)
#   unknown_class_5490.sh score MODEL [NSHARDS] zero-shot scores of every level's prompt over the pool: sam3 (#5213's
#                                               scorer) or gdino (#5395's, GroundingDINO-tiny), as a GPU array
#   unknown_class_5490.sh zeroshot              grade them, with SigLIP's text sort, at every level -> <root>/zeroshot
#   unknown_class_5490.sh dirs [ARM]            run dirs <sota>/<date>-uc5490-<arm>-b1 on the grid
#   unknown_class_5490.sh preflight ARM [IDX]   the preflight a launch of the same cells would pass
#   unknown_class_5490.sh launch ARM [IDX]      one arm as a cpu array (default: seed 0's 92 cells, Binary + Region)
#
# ARM is OPENER-LEVEL or ex4. OPENER is sig (SigLIP's typed sort, the app's default) or sam3 (#5241's opt-in
# opening: SAM3's set at its 0.5 cut held through the session, the head trained and scored all the same as
# hold:head rows). LEVEL is what the user types (CALIB_FGVC_LEVEL): family ("Boeing 737", the general class),
# variant (the real name, an upper bound), fake (an invented model name, "Boeing Veltra"). ex4 opens on 4 photos of
# the target, the "supply examples" route. Beta 1, 150 clicks, trajectory pass.
#
# Launch from a FROZEN worktree (VTS_REPO is this script's tree), every arm with the same SBATCH_CONSTRAINT (#4481).
# Env: U5490_ROOT (/expscratch/$USER/fgvc-5490), U5490_DATE (2026-11-02), U5490_STEPS (150), SOTA_ROOT, U5490_GPU.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CAL="$REPO/scripts/experiments/calibration"
ROOT="${U5490_ROOT:-/expscratch/$USER/fgvc-5490}"
S="${SOTA_ROOT:-/expscratch/$USER/state-of-the-app}"
D="${U5490_DATE:-2026-11-02}"
STEPS="${U5490_STEPS:-150}"
GRID="$S/$D-uc5490-grid"
ARMS=(sig-family sam3-family sig-variant sam3-variant sig-fake sam3-fake ex4)
mkdir -p "$ROOT/logs"
run_of() { echo "uc5490-$1-b1"; }

env_lines() {  # the environment every scoring / grading task sources
  cat <<EOF
source $REPO/scripts/grid/gridenv.sh
source $REPO/scripts/experiments/pile/pile_env.sh
export VTSEARCH_DATA_DIR=/expscratch/$USER/vts-cache/datadir
export VTS_REPO=$REPO CALIB_EXP=$GRID CALIB_RESULTS=$GRID/results CALIB_DATASETS=fgvc_aircraft
export CALIB_FGVC_AIRCRAFT_EMBEDDERS="siglip,siglip+dinov3_patch" CALIB_CATEGORY_MODE=all
export PYTHONPATH=$CAL:\${PYTHONPATH:-}
cd $CAL
EOF
}

level_of() {  # ARM -> CALIB_FGVC_LEVEL
  case "${1#*-}" in family) echo family ;; variant) echo variant ;; fake) echo fake_name ;; *) echo family ;; esac
}

common_env() {  # ARM: today's app at beta 1, trajectory pass, on fgvc_aircraft, plus the arm's opening
  local arm="$1"
  export SOTA_DATASET=fgvc_aircraft CALIB_FGVC_LEVEL="$(level_of "$arm")"
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
  unset CALIB_QUERY_MISS VTSEARCH_SVM_HEAD_C SOTA_EMBEDDERS SOTA_POOL_PREVALENCE SOTA_CELL_PREVALENCE
  unset CALIB_SEED_SAM3 CALIB_SEED_SAM3_PROMPTS CALIB_QUERY_LINE CALIB_QUERY_HOLD_PATCH CALIB_HANDOVER_SHADOW
  unset CALIB_SEED_EXAMPLES CALIB_STRATIFY_TARGET SOTA_OPENING
  case "$arm" in
  sam3-*)
    # #5241's sam3 arm, the prompt set the level names.
    export CALIB_SEED_SAM3="$ROOT/scores-sam3" CALIB_SEED_SAM3_PROMPTS="$CALIB_FGVC_LEVEL" CALIB_QUERY_LINE=fixed:0.5
    export CALIB_QUERY_HOLD=100000 CALIB_QUERY_HOLD_PATCH=1 CALIB_QUERY_MISS=0
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}seed_sam3,query_line,query_hold_patch"
    ;;
  ex4)
    # FHIBE's example opening (#4699): the first 4 votes are photos of the target, its centroid sorts the pool.
    export CALIB_SEED_EXAMPLES=4 CALIB_STRATIFY_TARGET=1 SOTA_OPENING=example
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}seed_examples,stratify_target"
    ;;
  esac
}
check_arm() { [[ " ${ARMS[*]} " == *" $1 "* ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }; }

case "${1:-}" in
prepare)
  export SOTA_DATASET=fgvc_aircraft SOTA_DATE="$D-uc5490-grid" SOTA_PASS=trajectory SOTA_SEEDS=1
  exec bash "$HERE/launch.sh" prepare
  ;;
score)
  m="${2:?model (sam3 or gdino)}"; ns="${3:-8}"; gpu="${U5490_GPU:-l40s}"
  dt=bf16; [[ $gpu == v100 ]] && dt=fp16
  case "$m" in
  sam3) cmd="python sam3_score_5213.py --model facebook/sam3 --dtype $dt" ;;
  gdino) cmd="python $HERE/gdino_score_5395.py --model IDEA-Research/grounding-dino-tiny --dtype $dt" ;;
  *) echo "model must be sam3 or gdino" >&2; exit 2 ;;
  esac
  {
    echo '#!/bin/bash'
    echo 'set -euo pipefail'
    env_lines
    echo "$cmd --results $GRID/results --dataset fgvc_aircraft --out $ROOT/scores-$m --shard \$SLURM_ARRAY_TASK_ID --nshards $ns"
  } >"$ROOT/score-$m.sh"
  sbatch --parsable -p gpu --gres="gpu:$gpu:1" -c 8 --mem 48G -t 12:00:00 -J "uc5490-$m" --array="0-$((ns - 1))%4" \
    -o "$ROOT/logs/score-$m-%A_%a.out" "$ROOT/score-$m.sh"
  ;;
zeroshot)
  {
    echo '#!/bin/bash'
    echo 'set -euo pipefail'
    env_lines
    echo "python $HERE/unknown_class_5490.py zeroshot --results $GRID/results --scores sam3=$ROOT/scores-sam3 gdino=$ROOT/scores-gdino --out $ROOT/zeroshot"
  } >"$ROOT/zeroshot.sh"
  sbatch --parsable -p cpu -c 4 --mem 32G -t 2:00:00 -J uc5490-zeroshot -o "$ROOT/logs/zeroshot-%j.out" "$ROOT/zeroshot.sh"
  ;;
dirs)
  [[ -s "$GRID/results/prepare_info.json" ]] || { echo "no prepared grid at $GRID" >&2; exit 2; }
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    r="$S/$D-$(run_of "$arm")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID/results/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID/results/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": 92, "datasets": ["fgvc_aircraft"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": 1,
 "max_steps": $STEPS, "cell_order": "seed", "test_bands": "", "job_name": "$(run_of "$arm")",
 "note": "#5490 arm $arm, Binary + Region (max_patch), beta 1, $STEPS clicks, trajectory pass; grid symlinked from $GRID"}
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
    local seed=--require-text-seed
    [[ "${SOTA_OPENING:-}" == example ]] && seed=""
    bash "$VTS_REPO/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod --job-name "$name" \
      --mem "$mem" --conc "${arr##*%}" --array-tasks "$n" --patch $seed \
      --require-region-voting "fgvc_aircraft:siglip+dinov3_patch" >&2 && echo 0
  }
  export -f sbatch
  exec bash "$0" launch "${@:2}"
  ;;
launch)
  arm="${2:?arm (${ARMS[*]})}"
  check_arm "$arm"
  common_env "$arm"
  export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}"
  exec bash "$HERE/launch.sh" redo "${3:-0-91}"
  ;;
*)
  sed -n '2,25p' "$0"
  exit 2
  ;;
esac
