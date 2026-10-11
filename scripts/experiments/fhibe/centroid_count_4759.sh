#!/usr/bin/env bash
# #4759: price counts tuned for the Goods' centroid on the FHIBE cells. Since #4732 the centroid's line
# is the typed query's count (#4603, `count`), which keeps one of a person's ~3 withheld face crops. One
# run per K on the baseline's identities (#4699, 562 people x 4 arms), at the app's balance, with every
# candidate priced as tagged rows on the same sessions (CALIB_CENTROID_LINE_VARIANTS, as #4732 did):
# `count` is the reference, `midpoint` anchors the reader (analyze_centroid_line_4732.read_cell), and
# the candidates are the rules #4759 added to CENTROID_LINE_RULES, each at beta 1/4, 1 and 4. The
# centroid never picks and the opening reads no balance, so the rows pair exactly and differ only on
# the clicks Test gives the centroid. That tier ends at the quota, well inside FHIBE_MAX_STEPS=40.
#
#   centroid_count_4759.sh dirs K       run dir runs/<date>-4759-k<K>, grid symlinked from the baseline's
#   centroid_count_4759.sh launch K     queue the array (fhibe/launch.sh cells)
#   centroid_count_4759.sh pack K       the same cells in one V100 job (fhibe/launch.sh pack)
#   centroid_count_4759.sh size K IDX   time one cell first
#
# Launch from a FROZEN worktree: every task imports the worktree this script sits in.
# Env: CC4759_DATE (default 2026-10-10), CC4759_RULES (the candidates), and anything fhibe/launch.sh
# reads (CALIB_PARTITION, CALIB_CONC).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FHIBE_ROOT="${VTS_FHIBE_ROOT:-/expscratch/$USER/fhibe}"
RUNS="$FHIBE_ROOT/${VTS_FHIBE_RELEASE:-fhibe-full-resolution-674a7dcf}-derived/runs"
D="${CC4759_DATE:-2026-10-10}"
BASE="${CC4759_BASE:-2026-10-09}"  # the #4699 baseline: identities sampled at K=4, one prepared grid per K
RULES="${CC4759_RULES:-midpoint count count_z3 count_z3.5 mirror_z3.5 gap count_gap gap_floor}"

variants() {
  local out=() r b
  for r in $RULES; do
    for b in 0.25 1 4; do out+=("$r@$b"); done
  done
  local IFS=,
  echo "${out[*]}"
}

K="${2:?K (1 or 4)}"
export FHIBE_EXAMPLES="$K" FHIBE_MAX_EXAMPLES=4
export FHIBE_RUN="$D-4759"
export FHIBE_IDENTITIES="$RUNS/$BASE-k4/identities.txt"
export FHIBE_MAX_STEPS="${FHIBE_MAX_STEPS:-40}"
export CALIB_SKYLINE_ARMS=""  # no ceiling: the study reads the opening only
export CALIB_RANK_FRAME_STEPS="${CALIB_RANK_FRAME_STEPS:-10,25}"
export CALIB_CENTROID_LINE_VARIANTS="${CALIB_CENTROID_LINE_VARIANTS:-$(variants)}"
export CALIB_JOB_NAME="${CALIB_JOB_NAME:-cc4759-k$K}"
SRC="$RUNS/$BASE-k$K/results"
RES="$RUNS/$FHIBE_RUN-k$K/results"

case "${1:-}" in
dirs)
  [[ -s "$SRC/prepare_info.json" ]] || { echo "no prepared grid at $SRC" >&2; exit 3; }
  mkdir -p "$RES/cells"
  chmod 700 "$RUNS/$FHIBE_RUN-k$K"
  ln -sfn "$SRC/prepare_info.json" "$RES/prepare_info.json"
  ln -sfn "$SRC/crops" "$RES/crops"
  ls -la "$RES"
  ;;
variants)
  echo "$CALIB_CENTROID_LINE_VARIANTS"
  ;;
launch)
  exec bash "$HERE/launch.sh" cells
  ;;
pack)
  exec bash "$HERE/launch.sh" pack
  ;;
size)
  exec bash "$HERE/launch.sh" size "${3:?cell index}"
  ;;
*)
  echo "usage: centroid_count_4759.sh {dirs K | variants K | launch K | pack K | size K IDX}" >&2; exit 1 ;;
esac
