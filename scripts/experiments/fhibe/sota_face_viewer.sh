#!/usr/bin/env bash
# State of the App: Face (#4762): the committed viewer, every session set on one page.
#
# A chip per beta (#4636), as the photo reviews' viewer_betas.sh builds it. The starting photos are
# folded into the dataset name ("fhibe_faces_1024, 1 photo", ...), so the page's dataset menu holds
# size x starting photos under each chip. People are shown as strata, never by subject ID
# (person_strata.py's label: age bracket, pronoun, skin tone; viewer.py --category-map).
#
#   sota_face_viewer.sh OUT.html
#
# Env: SOTA_FACE_DATE (default today), SOTA_RUNS_BUDGET_MB (default 1; 0 drops the per-seed lines).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CALIB="$HERE/../calibration"
WT="$(cd "$HERE/../../.." && pwd)"
FHIBE_ROOT="${VTS_FHIBE_ROOT:-/expscratch/$USER/fhibe}"
RUNS="$FHIBE_ROOT/${VTS_FHIBE_RELEASE:-fhibe-full-resolution-674a7dcf}-derived/runs"
D="${SOTA_FACE_DATE:-$(date +%Y-%m-%d)}"
OUT="${1:?usage: sota_face_viewer.sh OUT.html}"
BASE="$RUNS/$D-sota-baseline"
STRATA="$BASE/person_strata.csv"
for f in "$BASE/example_baseline_k1.csv" "$BASE/example_baseline_k4.csv" "$STRATA"; do
  [[ -s "$f" ]] || { echo "sota_face_viewer: no $f" >&2; exit 3; }
done
source "$WT/gridenv.sh" >/dev/null 2>&1
export VTS_REPO="$WT"

label() { [[ "$1" == 1 ]] && echo "1 photo" || echo "$1 photos"; }
# The click-0 baseline in the page's dataset names; the strata map as category,label.
MERGED="$BASE/example_baseline_viewer.csv"
CMAP="$BASE/person_strata_map.csv"
python - "$BASE" "$MERGED" "$STRATA" "$CMAP" <<'PY'
import sys
import pandas as pd

base, merged, strata, cmap = sys.argv[1:]
parts = []
for k, lab in ((1, "1 photo"), (4, "4 photos")):
    df = pd.read_csv(f"{base}/example_baseline_k{k}.csv")
    df["dataset"] = df["dataset"].astype(str) + f", {lab}"
    parts.append(df)
pd.concat(parts, ignore_index=True).to_csv(merged, index=False)
pd.read_csv(strata)[["category", "label"]].to_csv(cmap, index=False)
PY
chmod 600 "$MERGED" "$CMAP"

args=()
for k in 1 4; do
  for b in 0.25 1 4; do
    case "$b" in 0.25) t=025 ;; *) t="$b" ;; esac
    args+=(--beta-run "$b=$RUNS/$D-sota-k$k-b$t::{ds}, $(label "$k")")
  done
done
cd "$CALIB"
python viewer.py "${args[@]}" --baseline "$MERGED" --category-map "$CMAP" --out "$OUT" \
  --runs-budget-mb "${SOTA_RUNS_BUDGET_MB:-1}" \
  --title "State of the App: Face — $D, at each balance" \
  --subtitle "FHIBE face crops (FaceNet), the #4699 baseline's 562 people, from 1 and from 4 starting photos; one set of sessions per preset beta (#4636); people shown as strata (age, pronoun, skin tone), shipped defaults (#4762)" \
  --embedder-label "face=FaceNet face crop" --hide-metrics cost
echo "done: $OUT"
