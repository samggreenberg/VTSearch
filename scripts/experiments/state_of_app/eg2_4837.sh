#!/usr/bin/env bash
# #4837: EmbeddingGemma 2 against SigLIP on COCO Better's Binary path, session for session.
#
# Both arms run in ONE grid (SOTA_EMBEDDERS=siglip,embeddinggemma2), so they share the cells, the splits,
# the seeds, the commit and the environment; the embedder is the only difference. Per seed the grid is 288
# runs (seed-major, then embedder, then category): 0-143 SigLIP, 144-287 EmbeddingGemma 2. Each arm opens on
# its OWN typed query (its text tower ranks the corpus) and learns in its own space, as the app would run it.
#
#   eg2_4837.sh prepare                 the grid: one CPU job (needs coco_better__embeddinggemma2.pkl in the pile)
#   eg2_4837.sh dirs                    run dirs <root>/<date>-eg2-4837-b{025,1,4}, symlinked to the grid
#   eg2_4837.sh launch BETA [IDX]       the preset's sessions as a cpu array (default: every index)
#   eg2_4837.sh pack BETA JOBS [IDX]    the same as V100 pack jobs
#   eg2_4837.sh analyze BETA            text baseline (both embedders), analysis-binary and analysis-eg2
#
# EmbeddingGemma 2 needs transformers >= 5.18, which the experiment venv predates: EG2_PYTHONPATH (a
# --target install of it) goes ahead of the venv for BOTH arms, so the library is not a difference either.
# Launch from a FROZEN worktree: VTS_REPO is the worktree this script sits in, and every task imports it.
# Env: EG2_DATE (default 2026-10-13), EG2_SEEDS (5), SOTA_ROOT, EG2_PYTHONPATH, and anything launch.sh reads.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${EG2_DATE:-2026-10-13}"
SEEDS="${EG2_SEEDS:-5}"
GRID="$S/$D-eg2-4837-grid"
EMBEDDERS="siglip,embeddinggemma2"
OVERLAY="${EG2_PYTHONPATH:-/expscratch/sgreenberg/eg2-4837/overlay-exp}"
export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export SOTA_EMBEDDERS="$EMBEDDERS"

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }

all_idx() {
  python3 - "$SEEDS" <<'PY'
import sys
print(",".join(str(i) for i in range(288 * int(sys.argv[1]))))
PY
}

setup() {
  local b="$1"
  export CALIB_MEM="${CALIB_MEM:-5G}" CALIB_CPUS="${CALIB_CPUS:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-eg2-4837-b$(btag "$b")" SOTA_SEEDS="$SEEDS"
  export CALIB_JOB_NAME="eg2-4837-b$(btag "$b")"
}

case "${1:-}" in
prepare)
  export SOTA_DATE="$D-eg2-4837-grid" SOTA_SEEDS="$SEEDS"
  exec bash "$HERE/launch.sh" prepare
  ;;
dirs)
  for b in 0.25 1 4; do
    r="$S/$D-eg2-4837-b$(btag "$b")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID/results/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID/results/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "embeddinggemma2"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "eg2-4837-b$(btag "$b")",
 "note": "#4837 EmbeddingGemma 2 vs SigLIP, Binary path, beta $b; grid symlinked from $GRID"}
JSON
    echo "$r"
  done
  ;;
launch)
  setup "${2:?beta (0.25, 1 or 4)}"
  # A range, not the list: sbatch refuses a 1,440-index --array list as too long.
  exec bash "$HERE/launch.sh" redo "${3:-0-$((288 * SEEDS - 1))}"
  ;;
pack)
  setup "${2:?beta (0.25, 1 or 4)}"
  export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_PACK_JOBS="${3:?jobs}"
  export CALIB_PACK_PAR="${CALIB_PACK_PAR:-24}" CALIB_TIME="${CALIB_TIME:-12:00:00}"
  exec bash "$HERE/launch.sh" pack "${4:-$(all_idx)}"
  ;;
analyze)
  b="${2:?beta (0.25, 1 or 4)}"
  EXP="$S/$D-eg2-4837-b$(btag "$b")"
  WT="$(cd "$HERE/../../.." && pwd)"
  source "$WT/gridenv.sh"
  source "$WT/scripts/experiments/pile/pile_env.sh"
  export VTS_REPO="$WT" CALIB_EXP="$EXP" CALIB_RESULTS="$EXP/results" CALIB_DATASETS=coco_better
  export CALIB_COCO_BETTER_EMBEDDERS="$EMBEDDERS" CALIB_CATEGORY_MODE=all CALIB_N_SEEDS="$SEEDS"
  cd "$HERE/../calibration"
  # Both arms' click 0: each run opens on its own embedder's typed query.
  [[ -s "$EXP/text_baseline.csv" ]] || python text_baseline.py --results "$EXP/results" --out "$EXP/text_baseline.csv"
  for path in binary eg2; do
    python "$HERE/analyze.py" --exp "$EXP" --baseline "$EXP/text_baseline.csv" --out "$EXP/analysis-$path" \
      --path "$path" --seeds "${EG2_ANALYZE_SEEDS:-0}"
  done
  echo "done: $EXP/analysis-{binary,eg2}"
  ;;
*)
  echo "usage: eg2_4837.sh {prepare | dirs | launch BETA [IDX] | pack BETA JOBS [IDX] | analyze BETA}" >&2
  exit 1
  ;;
esac
