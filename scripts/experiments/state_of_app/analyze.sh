#!/usr/bin/env bash
# State of the App (#4159), after the array: the three artefacts a reader opens.
#
#   bash analyze.sh [<exp dir>]          # default: today's run
#
#   text_baseline.csv            click 0: what typing the query alone scores,
#                                AP and the line at each floor (#4357)
#   analysis/viewer.html         the shared calibration viewer, for browsing
#                                cells; the report reads the tables below.
#                                It opens on AP and does not offer cost,
#                                which the review retired (#4357, #4576)
#   analysis/cells.csv, lines.csv, curves.csv, influence.csv, images.csv,
#   image_detector.csv, summary.md
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
# A baseline from before #4357 has no line columns, and one from before #4363
# has no F1 or p-tagged ones; one from before #4471 has the old presets' betas
# (0.5 / 1 / 2) and not the ones the analyzer reads. Rebuild it rather than
# leave every click-0 line blank.
_BETA_COLS=$(python -c "from _rank_metrics import BETAS, beta_tag; print(' '.join(f'text_fb_share_{beta_tag(b)}' for b in BETAS))")
_stale_baseline() {
  [[ ! -s "$BASELINE" ]] && return 0
  local header c
  header=$(head -1 "$BASELINE")
  for c in text_f1_p50 $_BETA_COLS; do [[ ",$header," == *",$c,"* ]] || return 0; done
  return 1
}
if _stale_baseline; then
  # SigLIP only: the region path opens on SigLIP's text sort too, so one
  # baseline serves both, and it avoids reading the 7.5 GB patch cell.
  CALIB_COCO_BETTER_EMBEDDERS=siglip python text_baseline.py --results "$EXP/results" --out "$BASELINE"
fi
# The page names the path it holds, and holds only that path's panels (#4654:
# a Binary page used to say "SigLIP binary and DINOv3 region"). Each panel is
# named by its path, not its embedder key (#4655): the region path's key, the
# composite `siglip+dinov3_patch`, read bare as SigLIP and DINOv3 compared. The
# names are analyze.py's ARMS; viewer_betas.sh passes the same two.
case "$SOTA_PATH" in
  binary) VIEW_EMB=siglip; VIEW_PATH="SigLIP binary" ;;
  region) VIEW_EMB=siglip+dinov3_patch; VIEW_PATH="DINOv3 region, opened on SigLIP's text sort" ;;
  *) VIEW_EMB="$CALIB_COCO_BETTER_EMBEDDERS"; VIEW_PATH="SigLIP binary and DINOv3 region" ;;
esac
python viewer.py --results "$EXP" --arms results=prod --baseline "$BASELINE" --embedders "$VIEW_EMB" \
  --out "$OUT/viewer.html" --title "State of the App: $(basename "$EXP")" \
  --subtitle "coco_better, every class at every size; $VIEW_PATH, shipped defaults (#4159)" \
  --embedder-label "siglip=SigLIP binary" --embedder-label "siglip+dinov3_patch=DINOv3 region" \
  --hide-metrics cost
python "$HERE/analyze.py" --exp "$EXP" --baseline "$BASELINE" --out "$OUT" --path "$SOTA_PATH" --seeds "${SOTA_ANALYZE_SEEDS:-0}"
python "$HERE/figures.py" --analysis "$OUT" --out "$OUT/figures"
python "$HERE/thumbs.py" --analysis "$OUT" --out "$OUT/images" --n 12
echo "done: $OUT"
