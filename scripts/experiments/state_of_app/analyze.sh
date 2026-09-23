#!/usr/bin/env bash
# State of the App (#4159), after the array: the three artefacts a reader opens.
#
#   bash analyze.sh [<exp dir>]          # default: today's run
#
#   analysis/text_baseline.csv   click 0: what typing the query alone scores
#   analysis/viewer.html         cost / F1 / ... over clicks, every cell and path,
#                                with the click-0 notch and the full-label ceiling
#   analysis/cells.csv, influence.csv, images.csv, image_detector.csv, summary.md
#
# Run it on a compute node (srun): text_baseline re-reads every cell.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB="$HERE/../calibration"
WT="$(cd "$HERE/../../.." && pwd)"
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

EXP="${1:-${CALIB_EXP:-/expscratch/$USER/state-of-the-app/$(date +%Y-%m-%d)}}"
OUT="$EXP/analysis"
mkdir -p "$OUT"
export VTS_REPO="$WT" CALIB_EXP="$EXP" CALIB_RESULTS="$EXP/results" CALIB_DATASETS=coco_quarry
export CALIB_COCO_QUARRY_EMBEDDERS="siglip,siglip+dinov3_patch" CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="$(python -c "import json;print(json.load(open('$EXP/results/grid_shape.json'))['n_seeds'])")"

cd "$CALIB"
if [[ ! -s "$OUT/text_baseline.csv" ]]; then
  python text_baseline.py --results "$EXP/results" --out "$OUT/text_baseline.csv"
fi
python viewer.py --results "$EXP" --arms results=prod --baseline "$OUT/text_baseline.csv" \
  --out "$OUT/viewer.html" --title "State of the App: $(basename "$EXP")" \
  --subtitle "coco_quarry, every class at every size; SigLIP binary and DINOv3 region, shipped defaults (#4159)"
python "$HERE/analyze.py" --exp "$EXP" --baseline "$OUT/text_baseline.csv" --out "$OUT"
echo "done: $OUT"
