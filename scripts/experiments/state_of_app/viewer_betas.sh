#!/usr/bin/env bash
# State of the App (#4159): the review's viewer, every session set on one page (#4636).
#
#   bash viewer_betas.sh <date> [<out>]      # e.g. 2026-10-05
#
# A review runs one set of sessions per preset (SOTA_BETA=0.25|1|4, #4413), each
# in its own run directory: <root>/<date>-b025, -b1 and -b4. This puts the three
# on one page, a chip per beta the sessions RAN at, beside the metric menu's
# F1/4, F1 and F4, which score the returned set at each preset. So a reader can
# ask how the app at beta 4 does on large targets, at F4, without a re-run.
# Each set's click 0 is the text sort's own line at that set's beta (#4603).
# This is the viewer.html a report commits; analyze.sh still writes one per run.
#
#   <out>        default: <root>/<date>-betas/viewer.html
#   SOTA_ROOT    default: /expscratch/$USER/state-of-the-app
#   SOTA_RUNS_BUDGET_MB   the per-seed payload (default 1): three sets triple the
#                page, and it has to fit the repo's 4,000 KB cap to be committed.
#                viewer.py says when it does not.
#
# Run it on a compute node (srun -p cpu --mem=16G -c 2): it reads every cell of
# three runs. Run analyze.sh on the -b1 run first, for its text baseline.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB="$HERE/../calibration"
WT="$(cd "$HERE/../../.." && pwd)"
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

DATE="${1:?usage: bash viewer_betas.sh <date> [<out>]}"
ROOT="${SOTA_ROOT:-/expscratch/$USER/state-of-the-app}"
OUT="${2:-$ROOT/$DATE-betas/viewer.html}"
missing=()
for tag in b025 b1 b4; do
  [[ -d "$ROOT/$DATE-$tag/results/cells" ]] || missing+=("$ROOT/$DATE-$tag/results/cells")
done
if ((${#missing[@]})); then
  echo "viewer_betas: no session set at ${missing[*]}" >&2
  exit 2
fi
# One text baseline serves all three: the typed query is the same at every
# beta, and the baseline records the line the app draws on it at each preset.
BASELINE="$ROOT/$DATE-b1/text_baseline.csv"
if [[ ! -s "$BASELINE" ]]; then
  echo "viewer_betas: no $BASELINE; run analyze.sh on $ROOT/$DATE-b1 first" >&2
  exit 2
fi

export VTS_REPO="$WT"
cd "$CALIB"
python viewer.py \
  --beta-run "0.25=$ROOT/$DATE-b025" --beta-run "1=$ROOT/$DATE-b1" --beta-run "4=$ROOT/$DATE-b4" \
  --baseline "$BASELINE" --out "$OUT" --runs-budget-mb "${SOTA_RUNS_BUDGET_MB:-1}" \
  --title "State of the App: $DATE, at each balance" \
  --subtitle "coco_better, every class at every size; one set of sessions per preset beta (#4413, #4636), shipped defaults (#4159)" \
  --hide-metrics cost
echo "done: $OUT"
