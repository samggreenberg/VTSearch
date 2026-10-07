#!/usr/bin/env bash
# #4604: price showing a detector during Autopilot's opening, offline, from the 2026-10-05 Binary Photo and
# 2026-10-06 Region Photo State of the App runs (no new sessions). Extract, price each path x preset, report.
#
# usage: handoff_4604.sh OUT [DOCS]     (run on a compute node; ~2 min at 8 CPUs)
set -euo pipefail
OUT="$1"
DOCS="${2:-}"
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
BIN_BASE="${BIN_BASE:-/expscratch/sgreenberg/notch-4599/text_baseline-guarded-binary10.csv}"
REG_BASE="${REG_BASE:-$S/2026-10-06-b1/text_baseline-2seeds-post4136.csv}"
mkdir -p "$OUT"
declare -A BETA=([b025]=0.25 [b1]=1 [b4]=4)
for p in b025 b1 b4; do
  python "$HERE/handoff_extract_4604.py" --exp "$S/2026-10-05-$p" --embedder siglip --seeds 10 --tag "binary_$p" --out "$OUT"
  python "$HERE/handoff_extract_4604.py" --exp "$S/2026-10-06-$p" --embedder siglip+dinov3_patch --seeds 2 --tag "region_$p" --out "$OUT"
  python "$HERE/handoff_price_4604.py" --steps "$OUT/steps_binary_$p.csv.gz" --picks "$OUT/picks_binary_$p.csv.gz" \
    --cells "$S/2026-10-05-$p/analysis-binary-4605/cells.csv" --baseline "$BIN_BASE" --beta "${BETA[$p]}" --tag "binary_$p" --out "$OUT"
  python "$HERE/handoff_price_4604.py" --steps "$OUT/steps_region_$p.csv.gz" --picks "$OUT/picks_region_$p.csv.gz" \
    --cells "$S/2026-10-06-$p/analysis-region-4605/cells.csv" --baseline "$REG_BASE" --beta "${BETA[$p]}" --tag "region_$p" --out "$OUT"
done
python "$HERE/handoff_report_4604.py" --dir "$OUT" ${DOCS:+--docs "$DOCS"} --baseline "binary=$BIN_BASE" --baseline "region=$REG_BASE" \
  --ap "binary=$S/2026-10-05-b1/analysis-binary:$S/2026-10-05-b1/analysis-binary-4605" \
  --ap "region=$S/2026-10-06-b1/analysis-region:$S/2026-10-06-b1/analysis-region-4605"
