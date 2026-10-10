#!/usr/bin/env bash
# #4731: the COCO guard for a Good phase that runs dry. State of the App arms at one commit, paired run for run
# and identical until the Good walk's 16th miss in a row with a Good in hand. Since #4731 shipped, Q is the app:
#   C  CALIB_GOOD_DRY_RUN=off CALIB_QUOTA_DRY_BADS=off: the app before #4731
#   Q  the app: the Good phase ends on that run of misses, and a Good with 16 Bads gets the trained head
#      instead of the Goods' centroid
#   D  CALIB_QUOTA_DRY_BADS=off: the phase change without the quota's second tier
# The rule fires in ~19% of Binary sessions at beta 1 (2026-10-08-b1's picks), so FHIBE alone cannot clear it.
#
#   good_dry_4731.sh dirs SEEDS                  run dirs <root>/<date>-gooddry4731{C,Q,D}[-region]-b<tag> on the 2026-10-08 grid
#   good_dry_4731.sh launch ARM SEEDS [FIRST [LAST]]
#                                                queue one arm's sessions, seeds FIRST..LAST
#
# GOODDRY_BETA (0.25 | 1 | 4, default 1) is the preset and GOODDRY_PATH (binary | region, default binary) the
# production path; #4743 re-ran the guard on the opening without the More walk (#4740) at all three presets.
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: GOODDRY_DATE (default 2026-10-09), SOTA_ROOT, and anything launch.sh reads (CALIB_MEM, CALIB_PARTITION).
# GOODDRY_PACK=1 packs the arm into one job on a V100 node instead of an array (CALIB_PACK_PAR cells at a time).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${GOODDRY_DATE:-2026-10-09}"
GRID_SRC="${GOODDRY_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since this grid was prepared
BETA="${GOODDRY_BETA:-1}"
declare -A TAGB=([0.25]=b025 [1]=b1 [4]=b4)
TAG="${TAGB[$BETA]:?GOODDRY_BETA must be 0.25, 1 or 4}"
P="${GOODDRY_PATH:-binary}"
case $P in
  binary) PSUF=""; OFF=0; MEM_DEFAULT=4G ;;
  region) PSUF="-region"; OFF=144; MEM_DEFAULT=24G ;;
  *) echo "GOODDRY_PATH must be binary or region" >&2; exit 2 ;;
esac

dir_of() { echo "$S/$D-gooddry4731$1$PSUF-$TAG"; }

indices() {  # FIRST LAST: the path's block of each seed (seed-major blocks of 144 binary then 144 region cells)
  python3 - "$1" "$2" "$OFF" <<'PY'
import sys
off = int(sys.argv[3])
print(",".join(f"{s * 288 + off}-{s * 288 + off + 143}" for s in range(int(sys.argv[1]), int(sys.argv[2]) + 1)))
PY
}

case "${1:-}" in
dirs)
  SEEDS="${2:?seeds}"
  for arm in C Q D; do
    r="$(dir_of "$arm")/results"
    mkdir -p "$r/cells"
    ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
    ln -sfn "$GRID_SRC/crops" "$r/crops"
    cat >"$r/grid_shape.json" <<EOF
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "gooddry4731$arm-$P-$TAG",
 "note": "#4731 arm $arm, beta $BETA, $P path only, trajectory pass; grid symlinked from $GRID_SRC"}
EOF
  done
  ls -d "$S/$D"-gooddry4731*
  ;;
launch)
  ARM="${2:?C|Q|D}"; SEEDS="${3:?seeds}"; FIRST="${4:-0}"; LAST="${5:-$((SEEDS - 1))}"
  (( FIRST >= 0 && FIRST <= LAST && LAST < SEEDS )) || { echo "seeds must satisfy 0 <= FIRST <= LAST < SEEDS" >&2; exit 2; }
  unset CALIB_GOOD_DRY_RUN CALIB_QUOTA_DRY_BADS
  case $ARM in
    C) export CALIB_GOOD_DRY_RUN=off CALIB_QUOTA_DRY_BADS=off
       export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}good_dry_run,quota_dry_bads" ;;
    Q) ;;
    D) export CALIB_QUOTA_DRY_BADS=off
       export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}quota_dry_bads" ;;
    *) echo "ARM must be C, Q or D" >&2; exit 2 ;;
  esac
  export CALIB_MEM="${CALIB_MEM:-$MEM_DEFAULT}"
  export SOTA_BETA="$BETA" SOTA_DATE="$D-gooddry4731$ARM$PSUF-$TAG" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS"
  export CALIB_JOB_NAME="gooddry4731$ARM-$P-$TAG"
  idx="$(indices "$FIRST" "$LAST")"
  if [[ "${GOODDRY_PACK:-0}" == 1 ]]; then
    # The cpu cap is full: one multi-CPU job on a V100 node (launch_bands.sh pack, #4490).
    export CALIB_PARTITION="${CALIB_PARTITION:-gpu}" CALIB_PACK_GRES="${CALIB_PACK_GRES:-gpu:v100:1}"
    export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-30}" CALIB_TIME="${CALIB_TIME:-8:00:00}"
    exec bash "$HERE/launch.sh" pack "$(python3 -c "import sys; print(','.join(str(i) for r in sys.argv[1].split(',') for i in range(int(r.split('-')[0]), int(r.split('-')[1]) + 1)))" "$idx")"
  fi
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: good_dry_4731.sh {dirs SEEDS | launch ARM SEEDS [FIRST [LAST]]}" >&2; exit 1 ;;
esac
