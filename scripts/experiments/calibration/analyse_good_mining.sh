#!/usr/bin/env bash
# Assemble the #3267 Good Mining analysis: tables, figures, contact sheets.
#
#   bash analyse_good_mining.sh            # everything, into $CALIB_EXP/analysis
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

# shellcheck disable=SC1091
source "$WT/gridenv.sh"
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"
cd "$HERE" || exit 2

echo "=== 1/4  where each inclusion lands on the real seed sorts"
python probe_startup_cuts.py --json "$OUT/startup_cuts.json" | tail -25

echo
echo "=== 2/4  tables, verdict and figures"
GM_OUT="$OUT" python analyze_startup.py || exit 1

echo
echo "=== 3/4  contact sheets: what the openings actually clicked on"
# Three cells, chosen for different reasons rather than for looking good:
# the widest disagreement between arms (default, picked from the data), and one
# cell at each end of the prevalence axis - the mechanism is supposed to matter
# most at the scarce end, so a sheet from each is what lets a reader check that.
python make_startup_sheets.py --out "$OUT/figures" || exit 1
for cell in "${GM_SHEET_CELLS[@]:-coco_val:person:0}" ; do
  python make_startup_sheets.py --cell "$cell" --out "$OUT/figures" || true
done

echo
echo "=== 4/4  what is here"
ls -1 "$OUT" "$OUT/figures"
echo
echo "report -> $OUT/REPORT_startup.md"
