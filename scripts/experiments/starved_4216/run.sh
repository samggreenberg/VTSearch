#!/usr/bin/env bash
# #4216: re-read the starved share off finished runs. No simulated user is run here.
#
#   bash run.sh scan      # positives found per run, 74 run dirs (32 CPUs, ~3 min)
#   bash run.sh analyze   # starved.py + cost.py into $STARVED_DIR/analysis
#   bash run.sh reach     # each embedder's typed-query sort, offline (positives in its top 150)
#   bash run.sh figures   # figures + tables into the report directory
#
# Each step runs on a compute node through srun. STARVED_DIR defaults to
# /expscratch/$USER/starved-4216; RUNS_ROOT (what run_dirs.txt is relative to) to /expscratch/$USER.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
STARVED_DIR="${STARVED_DIR:-/expscratch/$USER/starved-4216}"
RUNS_ROOT="${RUNS_ROOT:-/expscratch/$USER}"
DOCS="$REPO/docs/experiments/2026-10-14-starved-small-4216"
SRUN=(srun --ntasks=1 --partition=cpu --time=01:00:00 --immediate=1200)
ENV="source $REPO/scripts/grid/gridenv.sh >/dev/null 2>&1 && cd $HERE"

case "${1:-}" in
  scan)
    "${SRUN[@]}" --cpus-per-task=32 --mem=48G bash -c "$ENV && python scan_found.py --root $RUNS_ROOT \
      --dirs run_dirs.txt --out $STARVED_DIR/scan --procs 32" ;;
  analyze)
    "${SRUN[@]}" --cpus-per-task=2 --mem=16G bash -c "$ENV && python starved.py --scan $STARVED_DIR/scan \
      --out $STARVED_DIR/analysis && python cost.py --runs $STARVED_DIR/analysis/runs.csv \
      --sota $RUNS_ROOT/state-of-the-app --priced-4743 $RUNS_ROOT/good-dry-4731/priced-4743 --out $STARVED_DIR/analysis" ;;
  reach)
    "${SRUN[@]}" --cpus-per-task=4 --mem=24G bash -c "$ENV && source $REPO/scripts/experiments/pile/pile_env.sh && \
      python reach.py --results $RUNS_ROOT/state-of-the-app/2026-10-08-b1/results \
      --embedders siglip,siglip2_l,clip_l,clip,embeddinggemma2 --seeds 24 --seeds-other 5 \
      --examples bowl@small,chair@small,knife@small,laptop@small --out $STARVED_DIR/analysis" ;;
  figures)
    "${SRUN[@]}" --cpus-per-task=2 --mem=8G bash -c "$ENV && python figures.py --analysis $STARVED_DIR/analysis --out $DOCS" ;;
  *)
    echo "usage: $0 scan|analyze|figures" >&2; exit 2 ;;
esac
