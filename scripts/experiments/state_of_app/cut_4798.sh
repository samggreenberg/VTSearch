#!/usr/bin/env bash
# #4798: the COCO Better Binary guard for a fix to the labels line's positives total. A fix to
# the class model's cut changes what Test returns, not the ranking, so one session set per preset
# prices every candidate: CALIB_LINE_COUNTS=1 records, at every click Test gives the trained head,
# what each candidate cut keeps on the withheld half and where its positives sit
# (task_NNNN__linecounts.csv.gz, k_cut_<name>). Today's app otherwise (#4785's head, #4786's count
# line while the class model is thin): the Binary Photo opening, 150 votes, no ceiling
# (SOTA_PASS=trajectory), on the grid #4743 priced (2026-10-08-b1), 5 seeds x 144 classes = 720 runs
# a preset. Cells are gzipped (CALIB_CELLS_GZIP=1).
#
#   cut_4798.sh dirs                  run dirs <root>/<date>-cut4798<arm>-b{025,1,4}
#   cut_4798.sh launch BETA           queue that preset's 720 Binary sessions (an array)
#
# CUT4798_ARM=<name> names a full-session arm instead (its knobs exported by the caller), into
# <date>-cut4798<name>-b*; the counts run's untagged rows are its control, run for run.
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: CUT4798_DATE (default 2026-10-12), SOTA_ROOT, CUT4798_SEEDS (5), and anything launch.sh reads
# (CALIB_MEM, CALIB_CONC, CALIB_PARTITION).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${CUT4798_DATE:-2026-10-12}"
SEEDS="${CUT4798_SEEDS:-5}"
ARM="${CUT4798_ARM:-}"
RUN="cut4798${ARM}"
GRID_SRC="${CUT4798_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

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
 "note": "#4798 ${ARM:-counts} run, beta $b, Binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
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
  export CALIB_MEM="${CALIB_MEM:-3G}" CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$RUN-b$(btag "$b")" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=150
  export CALIB_JOB_NAME="$RUN-binary-b$(btag "$b")"
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: cut_4798.sh {dirs | launch BETA}" >&2; exit 1 ;;
esac
