#!/usr/bin/env bash
# #5244: SAM3 cross-image exemplars from a Region session's votes, on #4711's grid.
#
#   sam3_exemplars_5244.sh plan                 the cells' withheld halves + first Goods/Bad -> <root>/plan (CPU)
#   sam3_exemplars_5244.sh sample               importance-sample each withheld half -> <root>/plan_sampled (CPU): every
#                                               positive, the top SAM3X_TOP (800) negatives by native SAM3 score, and
#                                               SAM3X_FRAC (0.08) of the rest at weight 1/frac (the 4-GPU cap, #5244)
#   sam3_exemplars_5244.sh smoke GPU [N] [DT]   N images (every positive + negatives, default 300) of 2 cells, with
#                                               canvases dumped -> <root>/smoke-GPU-DT
#   sam3_exemplars_5244.sh score [NSHARDS]      every cell's sampled half, an array of NSHARDS (default 144, one cell each)
#   sam3_exemplars_5244.sh analyze              sam3_exemplars_5244.py on the scores -> <root>/analysis (CPU)
#
# The GPU half is calibration/sam3_exemplars_5244.py; the native text-only scores it is compared with are #5213's
# (SAM3_NATIVE, default /expscratch/$USER/sam3-5213/scores). Env: SAM3X_ROOT (default /expscratch/$USER/sam3-5244),
# SAM3X_GPU (score: v100), SAM3X_DTYPE (fp16 on v100, else bf16), SAM3X_CONC (array concurrency, 4: the user GPU cap).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
CAL="$REPO/scripts/experiments/calibration"
ROOT="${SAM3X_ROOT:-/expscratch/$USER/sam3-5244}"
GRID="${SAM3_GRID:-/expscratch/$USER/state-of-the-app/2026-10-23-regstyles4711-grid}"
NATIVE="${SAM3_NATIVE:-/expscratch/$USER/sam3-5213/scores}"
REGION="${SAM3X_REGION:-/expscratch/$USER/state-of-the-app/2026-10-23-regstyles4711}"
mkdir -p "$ROOT/logs"

env_lines() {  # the environment every task sources
  cat <<EOF
set -euo pipefail
source $REPO/scripts/grid/gridenv.sh
source $REPO/scripts/experiments/pile/pile_env.sh
export VTSEARCH_DATA_DIR=/expscratch/$USER/vts-cache/datadir
export VTS_REPO=$REPO CALIB_EXP=$GRID CALIB_RESULTS=$GRID/results CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS="siglip,siglip+dinov3_patch" CALIB_CATEGORY_MODE=all CALIB_N_SEEDS=3
export PYTHONPATH=$CAL:\${PYTHONPATH:-}
cd $CAL
EOF
}

dtype_for() { [[ $1 == v100 ]] && echo fp16 || echo "${SAM3X_DTYPE:-bf16}"; }

case "${1:-}" in
plan)
  { echo '#!/bin/bash'; env_lines
    echo "python sam3_exemplars_5244.py plan --results $GRID/results --region $REGION-mp-region-b1/results/cells --out $ROOT"
  } >"$ROOT/plan.sh"
  sbatch --parsable -p cpu -c 2 --mem 32G -t 1:00:00 -J sam3x-plan -o "$ROOT/logs/plan-%j.out" "$ROOT/plan.sh"
  ;;
sample)
  { echo '#!/bin/bash'; env_lines
    echo "python sam3_exemplars_5244.py sample --out $ROOT --native $NATIVE --top ${SAM3X_TOP:-800} --frac ${SAM3X_FRAC:-0.08}"
  } >"$ROOT/sample.sh"
  sbatch --parsable -p cpu -c 2 --mem 16G -t 1:00:00 -J sam3x-sample -o "$ROOT/logs/sample-%j.out" "$ROOT/sample.sh"
  ;;
smoke)
  gpu="${2:?gpu type}"; n="${3:-300}"; dt="${4:-$(dtype_for "$gpu")}"
  out="$ROOT/smoke-$gpu-$dt"; mkdir -p "$out"
  # Two cells: a small-band one (the canvas halves its pixels) and a large one.
  mkdir -p "$out/plan"
  for c in bus__small dog__large; do ln -sf "$ROOT/plan/$c.json" "$out/plan/$c.json"; done
  { echo '#!/bin/bash'; env_lines
    echo "python sam3_exemplars_5244.py score --out $out --dtype $dt --limit $n --dump-canvases"
  } >"$out/smoke.sh"
  sbatch --parsable -p gpu --gres="gpu:$gpu:1" -c 8 --mem 48G -t 2:00:00 -J "sam3x-smoke-$gpu" \
    -o "$ROOT/logs/smoke-$gpu-$dt-%j.out" "$out/smoke.sh"
  ;;
score)
  ns="${2:-144}"; gpu="${SAM3X_GPU:-v100}"; dt="$(dtype_for "$gpu")"
  { echo '#!/bin/bash'; env_lines
    echo "python sam3_exemplars_5244.py score --out $ROOT --plan plan_sampled --dtype $dt --shard \$SLURM_ARRAY_TASK_ID --nshards $ns"
  } >"$ROOT/score.sh"
  sbatch --parsable -p gpu --gres="gpu:$gpu:1" -c 8 --mem 32G -t "${SAM3X_TIME:-3:00:00}" -J sam3x-score \
    --array="0-$((ns - 1))%${SAM3X_CONC:-4}" -o "$ROOT/logs/score-%A_%a.out" "$ROOT/score.sh"
  ;;
analyze)
  { echo '#!/bin/bash'; env_lines
    echo "python $HERE/sam3_exemplars_5244.py --root $ROOT --native $NATIVE --region $REGION-mp-region-b1/results/cells --out $ROOT/analysis"
  } >"$ROOT/analyze.sh"
  sbatch --parsable -p cpu -c 4 --mem 32G -t 1:00:00 -J sam3x-analyze -o "$ROOT/logs/analyze-%j.out" "$ROOT/analyze.sh"
  ;;
*)
  sed -n '2,13p' "$0"
  exit 2
  ;;
esac
