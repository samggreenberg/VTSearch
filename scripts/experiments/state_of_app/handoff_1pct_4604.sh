#!/usr/bin/env bash
# #4604 at a second prevalence: the same pricing on #4583's shipped arm (f03k2: binary SigLIP, 5 seeds, the user's
# pool thinned to 1%, the withheld half at the bench's 0.44%), with the typed query's guarded line rebuilt at 1%
# (the #4519 baseline #4583 used predates #4136, so its display line is the old midpoint).
#
# usage: handoff_1pct_4604.sh OUT     (compute node: the baseline embeds 144 queries and reads the SigLIP pickle)
# then:  handoff_report_4604.py --dir OUT ... picks up the binary1pct_* tables beside the main six.
set -euo pipefail
OUT="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
C="${CALSPLIT_BASE:-/expscratch/sgreenberg/calsplit-4583}"
source "$WT/scripts/experiments/pile/pile_env.sh"
# As #4583's launcher (the #4519 1% grid): only the pool's prevalence differs from the State of the App runs.
export CALIB_HAYSTACK_PREVALENCE=0.01 CALIB_DATASETS=coco_better CALIB_COCO_BETTER_EMBEDDERS=siglip
export CALIB_CATEGORY_MODE=all CALIB_N_SEEDS=5 CALIB_REQUIRE_OPENING=text CALIB_REQUIRE_SEED_QUERY=1 CALIB_SAFE_THRESHOLDS=1
mkdir -p "$OUT"
BASE="$OUT/text_baseline_1pct.csv"
if [[ ! -s "$BASE" ]]; then
  (cd "$WT/scripts/experiments/calibration" && python text_baseline.py --results "$C/f03k2_b1/results" --out "$BASE")
fi
declare -A BETA=([b025]=0.25 [b1]=1 [b4]=4)
for p in b025 b1 b4; do
  python "$HERE/handoff_extract_4604.py" --exp "$C/f03k2_$p" --embedder siglip --seeds 5 --tag "binary1pct_$p" --out "$OUT"
  python "$HERE/handoff_price_4604.py" --steps "$OUT/steps_binary1pct_$p.csv.gz" --picks "$OUT/picks_binary1pct_$p.csv.gz" \
    --baseline "$BASE" --beta "${BETA[$p]}" --tag "binary1pct_$p" --out "$OUT"
done
