#!/usr/bin/env bash
# State of the App at several prevalences (#4959): one viewer page, each prevalence a dataset on its menu.
#
#   bash viewer_prevalence.sh <date> <out.html> [<beta>]      # e.g. 2026-10-16 viewer_prevalence.html
#
# A review's own sessions sit at the bench's prevalence (coco_better: 0.44%, about 100 positives a class and size).
# Its prevalence scenarios are session sets of their own (state_of_app/launch.sh's SOTA_POOL_PREVALENCE /
# SOTA_CELL_PREVALENCE), in <root>/<date>-<scenario>-b025|b1|b4. This puts them beside the review's on one page:
# the scenario is folded into the dataset name ("coco_better, pool 1%"), as the Face page folds its starting photos,
# and the chips are the presets (#4636). Each scenario's click 0 is its own text baseline.
#
#   SOTA_PREV_SCENARIOS  "<dir tag>=<label>" pairs split on "|", default "h0.01=pool at 1%|p0.001=0.1% whole cell"
#   <beta>               one preset only (0.25, 1 or 4): three prevalences x three presets are over the repo's
#                        4,000 KB cap, and a page that must split splits by preset (owner, 2026-10-08)
#
# Run it on a compute node (srun -p cpu --mem=96G -c 2): it reads every cell of every set. Run analyze.sh on each
# set's -b1 run first, for its text baseline.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB="$HERE/../calibration"
WT="$(cd "$HERE/../../.." && pwd)"
source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

DATE="${1:?usage: bash viewer_prevalence.sh <date> <out.html> [<beta>]}"
OUT="${2:?usage: bash viewer_prevalence.sh <date> <out.html> [<beta>]}"
ONLY="${3:-}"
ROOT="${SOTA_ROOT:-/expscratch/$USER/state-of-the-app}"
# The review's own sets first, under the bench's prevalence; then each scenario, "<dir tag>=<label>", split on "|".
SETS=("=0.44% (default)")
IFS='|' read -r -a SCENS <<<"${SOTA_PREV_SCENARIOS:-h0.01=pool at 1%|p0.001=0.1% whole cell}"
for sc in "${SCENS[@]}"; do SETS+=("-${sc%%=*}=${sc#*=}"); done

args=() baselines=()
for set in "${SETS[@]}"; do
  sfx="${set%%=*}" label="${set#*=}"
  base="$ROOT/$DATE$sfx-b1/text_baseline.csv"
  [[ -s "$base" ]] || { echo "viewer_prevalence: no $base; run analyze.sh on $ROOT/$DATE$sfx-b1 first" >&2; exit 2; }
  baselines+=("$base=$label")
  for pair in 0.25:b025 1:b1 4:b4; do
    b="${pair%%:*}" t="${pair##*:}"
    [[ -n "$ONLY" && "$ONLY" != "$b" ]] && continue
    d="$ROOT/$DATE$sfx-$t"
    [[ -d "$d/results/cells" ]] || { echo "viewer_prevalence: no session set at $d" >&2; exit 2; }
    args+=(--beta-run "$b=$d::{ds}, $label")
  done
done

# One baseline file, its datasets named as the page names them.
MERGED="$(mktemp -d)/text_baseline_prevalence.csv"
python - "$MERGED" "${baselines[@]}" <<'PY'
import sys
import pandas as pd

out, *pairs = sys.argv[1:]
parts = []
for pair in pairs:
    path, label = pair.split("=", 1)
    df = pd.read_csv(path, low_memory=False)
    df["dataset"] = df["dataset"].astype(str) + f", {label}"
    parts.append(df)
pd.concat(parts, ignore_index=True).to_csv(out, index=False)
PY

export VTS_REPO="$WT"
cd "$CALIB"
title="State of the App: $DATE, at three prevalences"
[[ -n "$ONLY" ]] && title="$title, beta $ONLY"
python viewer.py "${args[@]}" --baseline "$MERGED" --out "$OUT" --runs-budget-mb 0 \
  --title "$title" \
  --subtitle "coco_better, every class at every size: the review's own sessions at the bench's 0.44%, the user's pool thinned to 1% (the withheld half stays at 0.44%), and the whole cell thinned to 0.1%; one set of sessions per preset beta (#4636, #4959)" \
  --embedder-label "siglip=SigLIP binary" --embedder-label "siglip+dinov3_patch=DINOv3 region" \
  --hide-metrics cost
echo "done: $OUT"
