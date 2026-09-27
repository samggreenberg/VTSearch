#!/usr/bin/env bash
# State of the App (#4159), after the array: the three artefacts a reader opens.
#
#   bash analyze.sh [<exp dir>]          # default: today's run
#
#   analysis/text_baseline.csv   click 0: what typing the query alone scores
#   analysis/viewer.html         cost / F1 / ... over clicks, every cell and path,
#                                with the click-0 notch and the full-label ceiling
#   analysis/cells.csv, influence.csv, images.csv, image_detector.csv, summary.md
#   analysis/figures/*.png, analysis/images.md + images/ (thumbnails)
#
# Run it on a compute node (srun): text_baseline re-reads every cell.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB="$HERE/../calibration"
WT="$(cd "$HERE/../../.." && pwd)"
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

EXP="${1:-${CALIB_EXP:-/expscratch/$USER/state-of-the-app/$(date +%Y-%m-%d)}}"
# SOTA_PATH=binary|region writes that path's report inputs to analysis-<path>/.
SOTA_PATH="${SOTA_PATH:-all}"
OUT="$EXP/analysis$([[ "$SOTA_PATH" == all ]] || echo "-$SOTA_PATH")"
mkdir -p "$OUT"
export VTS_REPO="$WT" CALIB_EXP="$EXP" CALIB_RESULTS="$EXP/results" CALIB_DATASETS=coco_better
export CALIB_COCO_BETTER_EMBEDDERS="siglip,siglip+dinov3_patch" CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="$(python -c "import json;print(json.load(open('$EXP/results/grid_shape.json'))['n_seeds'])")"

cd "$CALIB"
# One text baseline per run directory, shared by every path's analysis: it is
# the same text sort whichever path the clicks then take.
BASELINE="$EXP/text_baseline.csv"
if [[ ! -s "$BASELINE" ]]; then
  # SigLIP only: the region path opens on SigLIP's text sort too, so one
  # baseline serves both, and it avoids reading the 7.5 GB patch cell.
  CALIB_COCO_BETTER_EMBEDDERS=siglip python text_baseline.py --results "$EXP/results" --out "$BASELINE"
fi
python viewer.py --results "$EXP" --arms results=prod --baseline "$BASELINE" \
  --out "$OUT/viewer.html" --title "State of the App: $(basename "$EXP")" \
  --subtitle "coco_better, every class at every size; SigLIP binary and DINOv3 region, shipped defaults (#4159)"
python "$HERE/analyze.py" --exp "$EXP" --baseline "$BASELINE" --out "$OUT" --path "$SOTA_PATH" --seeds "${SOTA_ANALYZE_SEEDS:-0}"
python "$HERE/figures.py" --analysis "$OUT" --out "$OUT/figures"
python "$HERE/thumbs.py" --analysis "$OUT" --out "$OUT/images" --n 12
echo "done: $OUT"
