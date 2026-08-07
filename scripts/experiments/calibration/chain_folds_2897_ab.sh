#!/usr/bin/env bash
# Fold-count study (#2897): the bridge from the screen to the live A/B.
#
# Submitted by the screen's launcher as an `afterany` dependency on its analyze
# job, so the second stage starts without anyone being awake to read the first.
#
# It does NOT decide anything: it reads the decision the analyzer already made
# under the rules pre-registered in docs/plans/calibration-fold-count-experiment.md
# and recorded in `results/summary.json`, then acts on it.
#
#   * some voting mode's `h3_recommended_k` != 2  ->  launch arms {2} u {K*}
#   * every mode kept production's 2              ->  launch nothing
#
# The null result is the whole point of the second branch.  H3 says plainly
# that if H1 fails the answer is 2 and the study ships nothing; burning two
# more full simulations to re-confirm a null would be pure cost.  A study that
# auto-escalates regardless of its own verdict is not running the experiment,
# it is just spending.
#
# Usage: bash chain_folds_2897_ab.sh          (reads $CALIB_RESULTS/summary.json)
set -uo pipefail

SUMMARY="${CALIB_RESULTS:?CALIB_RESULTS unset}/summary.json"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [[ ! -s "$SUMMARY" ]]; then
  echo "chain: no summary.json at $SUMMARY - the screen's analyzer did not finish." >&2
  echo "chain: NOT launching the A/B; this needs a human." >&2
  exit 1
fi

# Every distinct recommended K across the voting modes, minus the baseline.
mapfile -t KSTARS < <(python - "$SUMMARY" <<'PY'
import json, sys

summary = json.load(open(sys.argv[1]))
ks = set()
for mode, block in (summary.get("by_voting") or {}).items():
    k = block.get("h3_recommended_k")
    kept = block.get("h3_kept_production")
    print(f"# {mode}: recommended_k={k} kept_production={kept}", file=sys.stderr)
    if k is not None and not kept:
        ks.add(int(k))
for k in sorted(ks):
    print(k)
PY
)

if ((${#KSTARS[@]} == 0)); then
  echo "chain: every voting mode kept production's 2 (h3_kept_production)."
  echo "chain: that IS the result - no A/B to run.  See $CALIB_RESULTS/REPORT.md"
  exit 0
fi

echo "chain: screen recommends K* = ${KSTARS[*]}; launching live A/B arms 2 ${KSTARS[*]}"

# Keep the A/B arms off the cramped /exp quota, same as the screen.
export CALIB_AB_BASE="${CALIB_AB_BASE:-$(dirname "$CALIB_EXP")}"

exec bash "$HERE/launch_folds_2897_ab.sh" 2 "${KSTARS[@]}"
