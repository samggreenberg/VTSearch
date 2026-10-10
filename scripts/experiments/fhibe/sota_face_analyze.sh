#!/usr/bin/env bash
# State of the App: Face (#4762): analyse the six session sets sota_face.sh ran.
#
# The State of the App's own tools, per (starting photos K, size), with the example sort as click 0
# (example_baseline.py, `sota_face.sh baseline`) where the photo reviews have the typed query:
#   * state_of_app/analyze.py per session set (K x beta x size): cells, lines, balances, curves,
#     influence, stops (-> <out>/k<K>-b<beta>-<size>/);
#   * perp.py --kind balance over the three betas, each off its own sessions (-> k<K>-<size>-perp/);
#   * by_click.py over the three betas: the objective at every click, what the session shows
#     (-> k<K>-<size>-objective_by_click.csv).
#
#   sota_face_analyze.sh OUT [K ...]     K defaults to "1 4"
#
# Run on a compute node (srun ... -c 8): analyze.py reads every cell of a session set.
# Env: SOTA_FACE_DATE (default today), and SOTA_FACE_SIZES (default "1024 640").
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOTA="$HERE/../state_of_app"
FHIBE_ROOT="${VTS_FHIBE_ROOT:-/expscratch/$USER/fhibe}"
RUNS="$FHIBE_ROOT/${VTS_FHIBE_RELEASE:-fhibe-full-resolution-674a7dcf}-derived/runs"
D="${SOTA_FACE_DATE:-$(date +%Y-%m-%d)}"
OUT="${1:?usage: sota_face_analyze.sh OUT [K ...]}"
shift
KS=("${@:-1 4}")
read -r -a KS <<<"${KS[*]}"
read -r -a SIZES <<<"${SOTA_FACE_SIZES:-1024 640}"
BETAS=(0.25 1 4)
BASELINES="$RUNS/$D-sota-baseline"
mkdir -p "$OUT" && chmod 700 "$OUT"

# The run dir's balance tag, as launch.sh writes it (0.25 -> 025, 1 -> 1, 4 -> 4). A lookup, not a
# python call: the task list asks for it 6,744 times.
btag() {
  case "$1" in
    0.25) echo 025 ;;
    1) echo 1 ;;
    4) echo 4 ;;
    *) python3 -c "import sys; b=float(sys.argv[1]); print(f'{b:g}'.replace('.', '') if b < 1 else f'{b:g}')" "$1" ;;
  esac
}

for k in "${KS[@]}"; do
  base="$BASELINES/example_baseline_k$k.csv"
  [[ -s "$base" ]] || { echo "no click-0 baseline at $base (run sota_face.sh baseline)" >&2; exit 3; }
  for size in "${SIZES[@]}"; do
    runs=()
    for b in "${BETAS[@]}"; do
      tag=$(btag "$b")
      exp="$RUNS/$D-sota-k$k-b$tag"
      a="$OUT/k$k-b$tag-$size"
      (
        # The run's own environment, so experiment_config reads its grid.
        export FHIBE_EXAMPLES="$k" FHIBE_MAX_EXAMPLES=4 FHIBE_BETA="$b" FHIBE_RUN="$D-sota"
        export FHIBE_IDENTITIES="$RUNS/2026-10-09-k4/identities.txt" FHIBE_DATASETS=fhibe_faces_1024,fhibe_faces_640
        ENVX=$(bash -c "source $HERE/launch.sh status >/dev/null 2>&1; echo \"\$ENVX\"")
        source "$HERE/../../../gridenv.sh" >/dev/null 2>&1
        eval "$ENVX"
        cd "$SOTA/../calibration"
        python "$SOTA/analyze.py" --exp "$exp" --baseline "$base" --path face --dataset "fhibe_faces_$size" --out "$a" |
          tail -1
      )
      runs+=(--run "$b=$a")
    done
    (
      source "$HERE/../../../gridenv.sh" >/dev/null 2>&1
      cd "$SOTA/../calibration"
      python "$SOTA/perp.py" --kind balance "${runs[@]}" --out "$OUT/k$k-$size-perp" | tail -1
      python "$SOTA/by_click.py" "${runs[@]}" --baseline "$base" --embedder face \
        --out "$OUT/k$k-$size-objective_by_click.csv" | tail -1
    )
  done
done
echo "done: $OUT"
