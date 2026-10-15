#!/usr/bin/env bash
# #4878: today's app, with the labels line's own reading of itself recorded at every click (line_efbeta,
# line_efbeta_corpus, line_efbeta_fresh, line_separation, line_head_count, find_efbeta), so a stop rule in the
# objective's units can be priced offline against Done (stop_signal_4878.py). Recording only: the sessions are
# the app's.
#
# STOP4878_ARM runs one of the stop rules the pricing favoured instead, as the A/B the owner asked for:
# `level` (Done on the level alone), `and` (Done once the lights AND the level hold), `or` (either), each at
# STOP4878_LEVEL (default 0.5) of the line's expected F-beta (CALIB_STOP_RULE / CALIB_STOP_LEVEL). Unset is the
# control, the app as the study ran it (lights; since #4878 the app is lights|level), whose run dirs keep the bare
# name.
#
# The Binary Photo opening, 150 votes, no ceiling (SOTA_PASS=trajectory), on the grid #4743 priced
# (2026-10-08-b1), 144 classes a seed and a preset. Cells are gzipped.
#
#   stop_signal_4878.sh dirs                 run dirs <root>/<date>-stop4878<tag>-b{025,1,4}
#   stop_signal_4878.sh launch BETA [IDX]    that preset's sessions as a cpu array (default: every seed's)
#   stop_signal_4878.sh pack BETA [IDX]      the same as V100 packs, the GPU hidden (the cpu cap full)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: STOP4878_ARM, STOP4878_LEVEL, STOP4878_DATE (default 2026-10-14), STOP4878_TAG (e.g. smoke),
# STOP4878_SEEDS (5), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_CONC).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${STOP4878_DATE:-2026-10-14}"
SEEDS="${STOP4878_SEEDS:-5}"
ARM="${STOP4878_ARM:-}"
LEVEL="${STOP4878_LEVEL:-0.5}"
case "$ARM" in
  "") RULE=lights ;;
  level) RULE=level ;;
  and) RULE="lights+level" ;;
  or) RULE="lights|level" ;;
  *) echo "STOP4878_ARM must be unset, level, and or or" >&2; exit 2 ;;
esac
RUN="stop4878${ARM}${ARM:+$LEVEL}${STOP4878_TAG:-}"
GRID_SRC="${STOP4878_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }

all_idx() {
  python3 - "$SEEDS" <<'PY'
import sys
print(",".join(str(s * 288 + i) for s in range(int(sys.argv[1])) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  for b in 0.25 1 4; do
    r="$S/$D-$RUN-b$(btag "$b")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "$RUN-b$(btag "$b")",
 "note": "#4878 stop rule $RULE (level $LEVEL; lights = today's app) with the labels line's own reading recorded, beta $b, binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
    echo "$r"
  done
  ;;
launch|pack)
  b="${2:?beta (0.25, 1 or 4)}"
  idx="${3:-$(all_idx)}"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-3G}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$RUN-b$(btag "$b")" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=150
  export CALIB_JOB_NAME="$RUN-b$(btag "$b")"
  # Named every time: the control is the app as the study ran it (lights), which since #4878 is no longer the
  # default (lights|level).
  export CALIB_STOP_RULE="$RULE" CALIB_STOP_LEVEL="$LEVEL"
  export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}stop_rule"
  if [[ "$1" == pack ]]; then
    export CALIB_PARTITION=gpu CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}" CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}"
    export CALIB_PACK_PAR="${CALIB_PACK_PAR:-24}" CALIB_TIME="${CALIB_TIME:-12:00:00}"
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: stop_signal_4878.sh {dirs | launch BETA [IDX] | pack BETA [IDX]}" >&2; exit 1 ;;
esac
