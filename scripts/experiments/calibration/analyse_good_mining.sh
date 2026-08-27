#!/usr/bin/env bash
# Assemble the #3267 Good Mining analysis: tables, figures, contact sheets.
#
#   bash analyse_good_mining.sh            # everything, into $CALIB_EXP/analysis
#   bash analyse_good_mining.sh curves     # tables + figures only, no contact sheets
#
# `curves` exists for re-running the analysis after a plotting change: the
# contact sheets are the slow step and they read the pick log, which does not
# move when the figures do.
#
# One command because the pieces have to agree: the report cites figure files by
# name and contact sheets by cell, and a hand-run step that is skipped leaves a
# report referencing something that is not there.  Reports here cite only
# analysis code that is in the tree (scripts/check-docs.py enforces it for
# docs/experiments/), which is the other reason this is a script rather than a
# paragraph of instructions.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $?" >&2' ERR

WT="${VTS_REPO:-/exp/$USER/projects/vts-goodmine-3267}"
HERE="$WT/scripts/experiments/calibration"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/good-mining-3267}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"
OUT="${GM_OUT:-$CALIB_EXP/analysis}"
MODE="${1:-all}"
case "$MODE" in
  all|curves) ;;
  *) echo "usage: $0 [all|curves]" >&2; exit 2 ;;
esac

# shellcheck disable=SC1091
source "$WT/gridenv.sh"
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"
cd "$HERE" || exit 2

echo "=== 1/5  where each inclusion lands on the real seed sorts"
python probe_startup_cuts.py --json "$OUT/startup_cuts.json" | tail -25

echo
echo "=== 2/5  the ZERO-CLICK anchor: what typing the query gets for free"
# Every quality curve is drawn against this.  Typing a query and reading the
# ranked haystack costs nothing, so it is what a clicked detector has to beat -
# without it the curves start at the first trainable click and the far right has
# nothing to be compared with.  It scores the SAME seed sort the arms opened on
# (`seed_query_text`, not the bare category name), on the same test split, under
# the same inclusion weights.
#
# Cached: it re-embeds one text query per category and re-reads every pickle, so
# it is the slow step here and its answer does not change between analysis runs.
# Delete the CSV to force a recompute.
TEXT_BASELINE="$OUT/text_baseline.csv"
if [[ -s "$TEXT_BASELINE" ]]; then
  echo "reusing $TEXT_BASELINE ($(wc -l < "$TEXT_BASELINE") rows) - delete it to recompute"
else
  python text_baseline.py --results "$CALIB_EXP/results" --out "$TEXT_BASELINE" || exit 1
fi
export GM_TEXT_BASELINE="$TEXT_BASELINE"

echo
echo "=== 3/5  tables, verdict and figures"
GM_OUT="$OUT" python analyze_startup.py || exit 1

if [[ "$MODE" == "curves" ]]; then
  echo
  echo "=== skipping the contact sheets (mode: curves)"
  echo
  echo "figures -> $OUT/figures"
  ls -1 "$OUT/figures"/*_vs_clicks*.png 2>/dev/null
  ls -lh "$OUT/viewer.html" 2>/dev/null
  echo
  echo "To land these in the repo's report, copy them back and commit:"
  echo "  rsync -av --include='*_vs_clicks*.png' --exclude='*' \\"
  echo "    grid:$OUT/figures/ docs/experiments/good-mining-3267/figures/"
  echo "  scp grid:$OUT/viewer.html docs/experiments/good-mining-3267/viewer.html"
  echo "  cp $OUT/REPORT_startup.md docs/experiments/good-mining-3267/REPORT_generated.md"
  echo "report -> $OUT/REPORT_startup.md"
  exit 0
fi

echo
echo "=== 4/5  contact sheets: what the openings actually clicked on"
# Three cells, chosen for different reasons rather than for looking good:
# the widest disagreement between arms (default, picked from the data), and one
# cell at each end of the prevalence axis - the mechanism is supposed to matter
# most at the scarce end, so a sheet from each is what lets a reader check that.
python make_startup_sheets.py --out "$OUT/figures" || exit 1
for cell in "${GM_SHEET_CELLS[@]:-coco_val:person:0}" ; do
  python make_startup_sheets.py --cell "$cell" --out "$OUT/figures" || true
done

echo
echo "=== 5/5  what is here"
ls -1 "$OUT" "$OUT/figures"
echo
echo "report -> $OUT/REPORT_startup.md"
