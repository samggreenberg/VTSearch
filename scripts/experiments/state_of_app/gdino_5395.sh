#!/usr/bin/env bash
# #5395: GroundingDINO zero-shot text prompts on #4711's Region grid, the second zero-shot detector beside #5213's
# SAM3. The same two halves and the same prompts, so a GroundingDINO row reads against a SAM3 and a SigLIP row cell
# for cell:
#
#   gdino_5395.sh smoke GPU [N]          time N images (default 200) of shard 0 for every model on one GPU type, plus
#                                        the checks: fp32 vs bf16, and the backbone cache vs none (50 images)
#   gdino_5395.sh score MODEL [NSHARDS]  every image of the grid's pool, as an array of NSHARDS (default 12) shards
#   gdino_5395.sh baseline MODEL         text_baseline.py --sam3 on its scores, both prompt sets -> <root>/gdino_MODEL_*.csv
#   gdino_5395.sh overlap MODEL          #5240's split: both sets graded on in-VG / not-in-VG images -> <root>/overlap
#
# MODEL is a tag: tiny (IDEA-Research/grounding-dino-tiny, the original), mmt (MM-GDINO-T o365v1_goldg_v3det), mml
# (MM-GDINO-L o365v2_oiv6_goldg). All three train without COCO; the pool is COCO train2017.
# Env: GDINO_ROOT (default /expscratch/$USER/gdino-5395), GDINO_GRID (#4711's grid), GDINO_GPU (score: l40s),
# GDINO_DTYPE (fp32; v100 has no bf16 -> fp16).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CAL="$REPO/scripts/experiments/calibration"
ROOT="${GDINO_ROOT:-/expscratch/$USER/gdino-5395}"
GRID="${GDINO_GRID:-/expscratch/$USER/state-of-the-app/2026-10-23-regstyles4711-grid}"
mkdir -p "$ROOT/logs"

model_of() {
  case "$1" in
  tiny) echo IDEA-Research/grounding-dino-tiny ;;
  mmt) echo openmmlab-community/mm_grounding_dino_tiny_o365v1_goldg_v3det ;;
  mml) echo openmmlab-community/mm_grounding_dino_large_o365v2_oiv6_goldg ;;
  *) echo "unknown model tag $1" >&2; exit 2 ;;
  esac
}

env_lines() {  # the environment every task sources
  cat <<EOF
source $REPO/scripts/grid/gridenv.sh
source $REPO/scripts/experiments/pile/pile_env.sh
export VTSEARCH_DATA_DIR=/expscratch/$USER/vts-cache/datadir
export VTS_REPO=$REPO CALIB_EXP=$GRID CALIB_RESULTS=$GRID/results CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS="siglip,siglip+dinov3_patch" CALIB_CATEGORY_MODE=all CALIB_N_SEEDS=3
export PYTHONPATH=$CAL:\${PYTHONPATH:-}
cd $CAL
EOF
}

score_cmd() {  # MODEL OUT SHARD NSHARDS DTYPE [LIMIT [EXTRA]]
  echo "python gdino_score_5395.py --results $GRID/results --out $2 --model $(model_of "$1") --shard $3 --nshards $4 --dtype $5${6:+ --limit $6}${7:+ $7}"
}

case "${1:-}" in
smoke)
  gpu="${2:?gpu type}"; n="${3:-200}"
  lo=bf16; [[ $gpu == v100 ]] && lo=fp16
  {
    echo '#!/bin/bash'
    echo 'set -euo pipefail'
    env_lines
    for m in tiny mmt mml; do
      score_cmd "$m" "$ROOT/smoke-$gpu/$m-fp32" 0 1 fp32 "$n"
      score_cmd "$m" "$ROOT/smoke-$gpu/$m-$lo" 0 1 "$lo" "$n"
    done
    # The cache check: the backbone on every prompt, which must give the cached run's scores.
    score_cmd tiny "$ROOT/smoke-$gpu/tiny-fp32-nocache" 0 1 fp32 50 --no-cache
  } >"$ROOT/smoke-$gpu.sh"
  sbatch --parsable -p gpu --gres="gpu:$gpu:1" -c 8 --mem 48G -t 4:00:00 -J "gdino-smoke-$gpu" \
    -o "$ROOT/logs/smoke-$gpu-%j.out" "$ROOT/smoke-$gpu.sh"
  ;;
score)
  m="${2:?model tag}"; ns="${3:-12}"; gpu="${GDINO_GPU:-l40s}"
  dt="${GDINO_DTYPE:-fp32}"; [[ $gpu == v100 && $dt == bf16 ]] && dt=fp16
  {
    echo '#!/bin/bash'
    env_lines
    score_cmd "$m" "$ROOT/scores-$m" '$SLURM_ARRAY_TASK_ID' "$ns" "$dt"
  } >"$ROOT/score-$m.sh"
  sbatch --parsable -p gpu --gres="gpu:$gpu:1" -c 8 --mem 48G -t "${GDINO_TIME:-12:00:00}" -J "gdino-score-$m" \
    --array="0-$((ns - 1))%4" -o "$ROOT/logs/score-$m-%A_%a.out" "$ROOT/score-$m.sh"
  ;;
baseline)
  m="${2:?model tag}"
  {
    echo '#!/bin/bash'
    echo 'set -euo pipefail'
    env_lines
    for p in typed noun both; do
      echo "python text_baseline.py --results $GRID/results --out $ROOT/gdino_${m}_$p.csv --sam3 $ROOT/scores-$m --sam3-prompts $p --zs-name gdino_$m --native-cut 0.35"
    done
  } >"$ROOT/baseline-$m.sh"
  sbatch --parsable -p cpu -c 4 --mem 32G -t 2:00:00 -J "gdino-baseline-$m" -o "$ROOT/logs/baseline-$m-%j.out" \
    "$ROOT/baseline-$m.sh"
  ;;
overlap)
  m="${2:?model tag}"; ids="${GDINO_VG_IDS:-/expscratch/$USER/sam3-5240}"; mkdir -p "$ROOT/overlap"
  {
    echo '#!/bin/bash'
    echo 'set -euo pipefail'
    env_lines
    for half in vg novg; do
      for p in typed noun; do
        echo "python text_baseline.py --results $GRID/results --out $ROOT/overlap/gdino_${m}_${p}_$half.csv --sam3 $ROOT/scores-$m --sam3-prompts $p --zs-name gdino_$m --native-cut 0.35 --keep-ids $ids/${half}_ids.txt"
      done
    done
  } >"$ROOT/overlap-$m.sh"
  sbatch --parsable -p cpu -c 4 --mem 32G -t 3:00:00 -J "gdino-overlap-$m" -o "$ROOT/logs/overlap-$m-%j.out" \
    "$ROOT/overlap-$m.sh"
  ;;
*)
  sed -n '2,17p' "$0"
  exit 2
  ;;
esac
