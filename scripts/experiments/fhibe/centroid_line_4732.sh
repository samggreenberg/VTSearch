#!/usr/bin/env bash
# #4732: price the Goods' centroid's line on the FHIBE cells. Below the label quota Test gives the
# centroid, cut at its two-Gaussian midpoint, which keeps about half the corpus on a rare target.
# One run per K on the baseline's identities (#4699, 562 people x 4 arms), at the app's balance, with
# every candidate line priced as tagged rows on the same sessions (CALIB_CENTROID_LINE_VARIANTS):
#   midpoint  the app's line until #4732 (at the pricing commit the untagged rows drew it too)
#   guarded   the typed query's guarded line, which takes no balance (#3826)
#   count     the typed query's count line at the row's balance (#4603), the app's since #4732; `text` (the typed query's own
#             display rule) is count at beta <= 1 and guarded above, so it needs no rows of its own
# each at beta 1/4, 1 and 4. The centroid never picks and the opening reads no balance, so the rows
# pair exactly and differ only on the clicks Test gives the centroid. That tier ends at the quota
# (3 Goods and 4 Bads, or a Good and 16 Bads), well inside FHIBE_MAX_STEPS=40.
#
#   centroid_line_4732.sh dirs K       run dir runs/<date>-4732-k<K>, grid symlinked from the baseline's
#   centroid_line_4732.sh launch K     queue the array (fhibe/launch.sh cells)
#   centroid_line_4732.sh pack K       the same cells in one V100 job (fhibe/launch.sh pack)
#   centroid_line_4732.sh size K IDX   time one cell first
#
# Launch from a FROZEN worktree: every task imports the worktree this script sits in.
# Env: CL4732_DATE (default 2026-10-10), and anything fhibe/launch.sh reads (CALIB_PARTITION, CALIB_CONC).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FHIBE_ROOT="${VTS_FHIBE_ROOT:-/expscratch/$USER/fhibe}"
RUNS="$FHIBE_ROOT/${VTS_FHIBE_RELEASE:-fhibe-full-resolution-674a7dcf}-derived/runs"
D="${CL4732_DATE:-2026-10-10}"
BASE="${CL4732_BASE:-2026-10-09}"  # the #4699 baseline: identities sampled at K=4, one prepared grid per K

K="${2:?K (1 or 4)}"
export FHIBE_EXAMPLES="$K" FHIBE_MAX_EXAMPLES=4
export FHIBE_RUN="$D-4732"
export FHIBE_IDENTITIES="$RUNS/$BASE-k4/identities.txt"
export FHIBE_MAX_STEPS="${FHIBE_MAX_STEPS:-40}"
export CALIB_SKYLINE_ARMS=""  # no ceiling: the study reads the opening only
export CALIB_RANK_FRAME_STEPS="${CALIB_RANK_FRAME_STEPS:-10,25}"
export CALIB_CENTROID_LINE_VARIANTS="${CALIB_CENTROID_LINE_VARIANTS:-midpoint@0.25,midpoint@1,midpoint@4,guarded@0.25,guarded@1,guarded@4,count@0.25,count@1,count@4}"
export CALIB_JOB_NAME="${CALIB_JOB_NAME:-cl4732-k$K}"
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
  echo "usage: centroid_line_4732.sh {dirs K | launch K | pack K | size K IDX}" >&2; exit 1 ;;
esac
