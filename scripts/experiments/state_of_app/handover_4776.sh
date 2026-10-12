#!/usr/bin/env bash
# #4776: the COCO Better Binary guard for a hand-over rule. Past the label quota Test gives the
# trained head; a rule may keep the Goods' centroid instead. Neither picks, so one session set per
# preset prices every rule: CALIB_HANDOVER_SHADOW=1 records, at every click Test gives the head, the
# centroid's Test (rows tagged handover:centroid) and a rule's evidence (task_NNNN__handover.csv).
# Today's app otherwise: the Binary Photo opening, 150 votes, no ceiling (SOTA_PASS=trajectory),
# on the grid #4743 priced (2026-10-08-b1), 5 seeds x 144 classes = 720 runs a preset.
#
#   handover_4776.sh dirs                 run dirs <root>/<date>-handover4776-b{025,1,4}
#   handover_4776.sh launch BETA          queue that preset's 720 Binary sessions (an array)
#
# HO4776_ARM=prior runs the issue's other candidate instead, full sessions with the head started from
# the Goods' centroid (CALIB_HEAD=linear_svm_prior: Test, picks, folds and line all take it), into
# <date>-prior4776-b*; the shadow arm's untagged rows are its control, run for run.
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: HO4776_DATE (default 2026-10-10), SOTA_ROOT, HO4776_SEEDS (5), and anything launch.sh reads
# (CALIB_MEM, CALIB_CONC, CALIB_PARTITION).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${HO4776_DATE:-2026-10-10}"
SEEDS="${HO4776_SEEDS:-5}"
GRID_SRC="${HO4776_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared
case "${HO4776_ARM:-shadow}" in
  shadow) RUN=handover4776 ;;
  prior) RUN=prior4776 ;;
  *) echo "HO4776_ARM must be shadow or prior" >&2; exit 2 ;;
esac

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
 "note": "#4776 $RUN arm, beta $b, Binary only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
launch)
  b="${2:?beta (0.25, 1 or 4)}"
  idx="$(python3 - "$SEEDS" <<'PY'
import sys
print(",".join(f"{s * 288}-{s * 288 + 143}" for s in range(int(sys.argv[1]))))
PY
)"
  if [[ "$RUN" == prior4776 ]]; then
    export CALIB_HEAD=linear_svm_prior PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}head"
  else
    export CALIB_HANDOVER_SHADOW=1
  fi
  export CALIB_MEM="${CALIB_MEM:-3G}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$RUN-b$(btag "$b")" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=150
  export CALIB_JOB_NAME="$RUN-binary-b$(btag "$b")"
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: handover_4776.sh {dirs | launch BETA}" >&2; exit 1 ;;
esac
