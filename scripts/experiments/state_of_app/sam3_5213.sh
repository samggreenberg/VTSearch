#!/usr/bin/env bash
# #5213: SAM3 zero-shot text prompts on #4711's Region grid. Two halves:
#
#   sam3_5213.sh smoke GPU [N]     time N images (default 200) of shard 0 on one GPU type (l40s, v100, a100),
#                                  into <root>/smoke-GPU; also the dtype check (bf16 vs fp32 on l40s/a100)
#   sam3_5213.sh score [NSHARDS]   every image of the grid's pool, as an array of NSHARDS (default 8) shards
#   sam3_5213.sh baseline          text_baseline.py --sam3 on the scores, both prompt sets -> <root>/sam3_*.csv
#   sam3_5213.sh overlap DIR       #5240: SigLIP + both SAM3 sets graded on all / in-VG / not-in-VG images -> DIR,
#                                  then sam3_overlap_5240.py analyze -> DIR/analysis
#
# The GPU half is calibration/sam3_score_5213.py; the CPU half is text_baseline.py's --sam3 mode, which scores the
# cells' own pools and splits, so a SAM3 row reads against SigLIP's text_baseline.csv row cell for cell.
# Env: SAM3_ROOT (default /expscratch/$USER/sam3-5213), SAM3_GRID (#4711's grid), SAM3_GPU (score: l40s),
# SAM3_MODEL (facebook/sam3), SAM3_DTYPE (bf16; v100 has no bf16 -> fp16). HF_TOKEN must hold the gated access.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CAL="$REPO/scripts/experiments/calibration"
ROOT="${SAM3_ROOT:-/expscratch/$USER/sam3-5213}"
GRID="${SAM3_GRID:-/expscratch/$USER/state-of-the-app/2026-10-23-regstyles4711-grid}"
MODEL="${SAM3_MODEL:-facebook/sam3}"
mkdir -p "$ROOT/logs"

env_lines() {  # the environment every task sources
  cat <<EOF
source $REPO/gridenv.sh
source $REPO/scripts/experiments/pile/pile_env.sh
export VTSEARCH_DATA_DIR=/expscratch/$USER/vts-cache/datadir
export VTS_REPO=$REPO CALIB_EXP=$GRID CALIB_RESULTS=$GRID/results CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS="siglip,siglip+dinov3_patch" CALIB_CATEGORY_MODE=all CALIB_N_SEEDS=3
export PYTHONPATH=$CAL:\${PYTHONPATH:-}
cd $CAL
EOF
}

score_cmd() {  # OUT SHARD NSHARDS DTYPE [LIMIT]
  echo "python sam3_score_5213.py --results $GRID/results --out $1 --model $MODEL --shard $2 --nshards $3 --dtype $4${5:+ --limit $5}"
}

case "${1:-}" in
smoke)
  gpu="${2:?gpu type}"; n="${3:-200}"
  dt="${SAM3_DTYPE:-bf16}"; [[ $gpu == v100 ]] && dt=fp16
  out="$ROOT/smoke-$gpu"
  {
    echo '#!/bin/bash'
    env_lines
    score_cmd "$out" 0 1 "$dt" "$n"
    # The dtype check: the same images at fp32, compared by the reader (scores must rank alike).
    [[ $gpu != v100 ]] && score_cmd "$out-fp32" 0 1 fp32 "$n"
    true
  } >"$ROOT/smoke-$gpu.sh"
  sbatch --parsable -p gpu --gres="gpu:$gpu:1" -c 8 --mem 48G -t 2:00:00 -J "sam3-smoke-$gpu" \
    -o "$ROOT/logs/smoke-$gpu-%j.out" "$ROOT/smoke-$gpu.sh"
  ;;
score)
  ns="${2:-8}"; gpu="${SAM3_GPU:-l40s}"
  dt="${SAM3_DTYPE:-bf16}"; [[ $gpu == v100 ]] && dt=fp16
  {
    echo '#!/bin/bash'
    env_lines
    score_cmd "$ROOT/scores" '$SLURM_ARRAY_TASK_ID' "$ns" "$dt"
  } >"$ROOT/score.sh"
  sbatch --parsable -p gpu --gres="gpu:$gpu:1" -c 8 --mem 48G -t "${SAM3_TIME:-12:00:00}" -J sam3-score \
    --array="0-$((ns - 1))%4" -o "$ROOT/logs/score-%A_%a.out" "$ROOT/score.sh"
  ;;
baseline)
  {
    echo '#!/bin/bash'
    env_lines
    for p in typed noun both; do
      echo "python text_baseline.py --results $GRID/results --out $ROOT/sam3_$p.csv --sam3 $ROOT/scores --sam3-prompts $p"
    done
  } >"$ROOT/baseline.sh"
  sbatch --parsable -p cpu -c 4 --mem 32G -t 2:00:00 -J sam3-baseline -o "$ROOT/logs/baseline-%j.out" "$ROOT/baseline.sh"
  ;;
overlap)
  out="${2:?out dir}"; mkdir -p "$out/logs"
  {
    echo '#!/bin/bash'
    echo 'set -euo pipefail'
    env_lines
    echo "python $HERE/sam3_overlap_5240.py ids --out $out"
    for half in all vg novg; do
      k=""; [[ $half != all ]] && k=" --keep-ids $out/${half}_ids.txt"
      echo "python text_baseline.py --results $GRID/results --out $out/siglip_$half.csv$k"
      for p in typed noun; do
        echo "python text_baseline.py --results $GRID/results --out $out/sam3_${p}_$half.csv --sam3 $ROOT/scores --sam3-prompts $p$k"
      done
    done
    echo "python $HERE/sam3_overlap_5240.py analyze --root $out --out $out/analysis"
  } >"$out/overlap.sh"
  sbatch --parsable -p cpu -c 4 --mem 32G -t 3:00:00 -J sam3-overlap -o "$out/logs/overlap-%j.out" "$out/overlap.sh"
  ;;
*)
  sed -n '2,15p' "$0"
  exit 2
  ;;
esac
