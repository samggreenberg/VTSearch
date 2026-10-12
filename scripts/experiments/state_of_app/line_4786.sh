#!/usr/bin/env bash
# #4786: the COCO Better Binary guard for a rule on the first trained clicks' line. A line rule
# changes what Test returns, not the ranking, so one session set per preset prices every count
# rule: CALIB_LINE_COUNTS=1 records, at every click Test gives the trained head, what each
# candidate line keeps on the withheld half and where its positives sit (task_NNNN__linecounts.csv).
# Today's app otherwise (the head started from the Goods' centroid, #4785): the Binary Photo
# opening, 150 votes, no ceiling (SOTA_PASS=trajectory), on the grid #4743 priced (2026-10-08-b1),
# 5 seeds x 144 classes = 720 runs a preset.
#
#   line_4786.sh dirs                 run dirs <root>/<date>-line4786<arm>-b{025,1,4}
#   line_4786.sh launch BETA          queue that preset's 720 Binary sessions (an array)
#
# LINE4786_ARM=<name> names a full-session arm instead (its knobs exported by the caller), into
# <date>-line4786<name>-b*; the counts run's untagged rows are its control, run for run.
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: LINE4786_DATE (default 2026-10-11), SOTA_ROOT, LINE4786_SEEDS (5), and anything launch.sh reads
# (CALIB_MEM, CALIB_CONC, CALIB_PARTITION).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${LINE4786_DATE:-2026-10-11}"
SEEDS="${LINE4786_SEEDS:-5}"
ARM="${LINE4786_ARM:-}"
RUN="line4786${ARM}"
GRID_SRC="${LINE4786_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }

case "${1:-}" in
dirs)
  for b in 0.25 1 4; do
    r="$S/$D-$RUN-b$(btag "$b")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "$RUN-binary-b$(btag "$b")",
 "note": "#4786 ${ARM:-counts} run, beta $b, Binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
launch)
  b="${2:?beta (0.25, 1 or 4)}"
  # The Binary block of each seed's 288 cells.
  idx="$(python3 - "$SEEDS" <<'PY'
import sys
print(",".join(f"{s * 288}-{s * 288 + 143}" for s in range(int(sys.argv[1]))))
PY
)"
  [[ -n "$ARM" ]] || export CALIB_LINE_COUNTS=1
  export CALIB_MEM="${CALIB_MEM:-3G}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$RUN-b$(btag "$b")" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=150
  export CALIB_JOB_NAME="$RUN-binary-b$(btag "$b")"
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: line_4786.sh {dirs | launch BETA}" >&2; exit 1 ;;
esac
