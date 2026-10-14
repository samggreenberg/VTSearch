#!/usr/bin/env bash
# #4874: SigLIP 2 L (siglip2_l, the premium image embedder) against SigLIP on COCO Better's Binary path,
# session for session. #4837's eg2_4837.sh with the arm swapped.
#
# Both arms run in ONE grid (SOTA_EMBEDDERS=siglip,siglip2_l), so they share the cells, the splits, the seeds,
# the commit and the environment; the embedder is the only difference. Per seed the grid is 288 runs
# (seed-major, then embedder, then category): 0-143 SigLIP, 144-287 SigLIP 2 L. Each arm opens on its OWN typed
# query (its text tower ranks the corpus) and learns in its own space, as the app would run it.
#
#   s2l_4874.sh prepare                 the grid: one CPU job (needs coco_better__siglip2_l.pkl in the pile)
#   s2l_4874.sh dirs                    run dirs <root>/<date>-s2l-4874-b{025,1,4}, symlinked to the grid
#   s2l_4874.sh launch BETA [IDX]       the preset's sessions as a cpu array (default: every index)
#   s2l_4874.sh pack BETA JOBS [IDX]    the same as V100 pack jobs
#   s2l_4874.sh analyze BETA            text baseline (both embedders), analysis-binary and analysis-s2l
#
# Launch from a FROZEN worktree: VTS_REPO is the worktree this script sits in, and every task imports it.
# Env: S2L_DATE (default 2026-10-14), S2L_SEEDS (5), SOTA_ROOT, and anything launch.sh reads.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${S2L_DATE:-2026-10-14}"
SEEDS="${S2L_SEEDS:-5}"
GRID="$S/$D-s2l-4874-grid"
EMBEDDERS="siglip,siglip2_l"
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
  export SOTA_BETA="$b" SOTA_DATE="$D-s2l-4874-b$(btag "$b")" SOTA_SEEDS="$SEEDS"
  export CALIB_JOB_NAME="s2l-4874-b$(btag "$b")"
}

case "${1:-}" in
prepare)
  export SOTA_DATE="$D-s2l-4874-grid" SOTA_SEEDS="$SEEDS"
  exec bash "$HERE/launch.sh" prepare
  ;;
dirs)
  for b in 0.25 1 4; do
    r="$S/$D-s2l-4874-b$(btag "$b")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID/results/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID/results/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip2_l"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "s2l-4874-b$(btag "$b")",
 "note": "#4874 SigLIP 2 L vs SigLIP, Binary path, beta $b; grid symlinked from $GRID"}
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
  EXP="$S/$D-s2l-4874-b$(btag "$b")"
  WT="$(cd "$HERE/../../.." && pwd)"
  source "$WT/gridenv.sh"
  source "$WT/scripts/experiments/pile/pile_env.sh"
  export VTS_REPO="$WT" CALIB_EXP="$EXP" CALIB_RESULTS="$EXP/results" CALIB_DATASETS=coco_better
  export CALIB_COCO_BETTER_EMBEDDERS="$EMBEDDERS" CALIB_CATEGORY_MODE=all CALIB_N_SEEDS="$SEEDS"
  cd "$HERE/../calibration"
  # Both arms' click 0: each run opens on its own embedder's typed query.
  [[ -s "$EXP/text_baseline.csv" ]] || python text_baseline.py --results "$EXP/results" --out "$EXP/text_baseline.csv"
  for path in binary s2l; do
    python "$HERE/analyze.py" --exp "$EXP" --baseline "$EXP/text_baseline.csv" --out "$EXP/analysis-$path" \
      --path "$path" --seeds "${S2L_ANALYZE_SEEDS:-0}"
  done
  echo "done: $EXP/analysis-{binary,s2l}"
  ;;
*)
  echo "usage: s2l_4874.sh {prepare | dirs | launch BETA [IDX] | pack BETA JOBS [IDX] | analyze BETA}" >&2
  exit 1
  ;;
esac
