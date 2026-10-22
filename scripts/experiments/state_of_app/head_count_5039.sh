#!/usr/bin/env bash
# #5039: the recall half of the head's count line, closed loop. While the class model holds fewer than 6 held-out
# Goods the labels line is the head's count (#4786, #4798): beta ** 0.708 times what stands out in the head's scores.
# Offline on #4937's snapshots (head_count_5039.py) it keeps a median 0.14 of the F-beta-best set at beta 4 on Binary
# Photo, and FHIBE faces (on this line all session) and Region Photo want fewer at beta 4. Above beta 1 the arms grow
# the count as beta ** 0.189 (1.3x at beta 4, not 2.67x) and, where a held-out Good ranks below what stands out, reach
# down the head's excess curve to its expected F-beta peak: offline that gained on all three at 0.1%, 0.44% and 1%
# prevalence. The head's count line is Train's line too while it serves (Autopilot's acquisition reads it), so the
# sessions themselves change: this prices them end to end. The rule reads beta > 1 only, so beta 1/4 and 1 are
# today's sessions bit for bit; only beta 4 runs.
#
#   ctl      the app before #5039: CALIB_HEAD_COUNT_RECALL_EXPONENT=off CALIB_HEAD_COUNT_RECALL_Z=off
#   e13z3    the count at 1.3x at beta 4, the reach from z = 3 (CALIB_HEAD_COUNT_RECALL_EXPONENT=0.18925, _Z=3)
#   e13z275  the same, the reach from z = 2.75
#
# The Binary Photo path at beta 4, 5 seeds x 144 cells, on the grid #4743 priced (2026-10-08-b1), paired by cell and
# seed, 200 clicks and the closing check, as #5040's closed loop ran: an arm's run is identical to ctl's until the first
# click whose head's count line the recall half moves. Trajectory pass (no ceiling); cells gzipped.
#
#   head_count_5039.sh dirs [ARM]               run dirs <root>/<date>-headcount5039<tag>-<arm>-b{025,1,4}
#   head_count_5039.sh cpack ARM BETA [IDX]     one arm at one preset as cpu-partition packs (CALIB_PACK_JOBS jobs of
#                                               CALIB_PACK_PAR cells at CALIB_CPUS thread each)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: HEADCOUNT5039_DATE (default 2026-10-18), HEADCOUNT5039_SEEDS (5), HEADCOUNT5039_TAG (e.g. smoke), SOTA_ROOT, and
# anything launch.sh reads (CALIB_MEM, CALIB_PACK_JOBS, CALIB_PACK_PAR, CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${HEADCOUNT5039_DATE:-2026-10-18}"
SEEDS="${HEADCOUNT5039_SEEDS:-5}"
ARMS=(ctl e13z3 e13z275)
E13=0.18925  # log(1.3) / log(4): 1.3x what stands out at beta 4
declare -A KNOBS=([ctl]="CALIB_HEAD_COUNT_RECALL_EXPONENT=off CALIB_HEAD_COUNT_RECALL_Z=off"
  [e13z3]="CALIB_HEAD_COUNT_RECALL_EXPONENT=$E13 CALIB_HEAD_COUNT_RECALL_Z=3"
  [e13z275]="CALIB_HEAD_COUNT_RECALL_EXPONENT=$E13 CALIB_HEAD_COUNT_RECALL_Z=2.75")
GRID_SRC="${HEADCOUNT5039_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "headcount5039${HEADCOUNT5039_TAG:-}-$1-b$(btag "$2")"; }

all_idx() {
  python3 - "$SEEDS" <<'PY'
import sys
print(",".join(str(s * 288 + i) for s in range(int(sys.argv[1])) for i in range(144)))
PY
}

case "${1:-}" in
dirs)
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    for b in 0.25 1 4; do
      r="$S/$D-$(run_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 200, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$b")",
 "note": "#5039 arm $arm (${KNOBS[$arm]}), beta $b, 200 clicks, binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
cpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (0.25, 1 or 4)}"
  [[ -v "KNOBS[$arm]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  idx="${4:-$(all_idx)}"
  run="$(run_of "$arm" "$b")"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=200
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150,200}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_HEAD_COUNT_RECALL_EXPONENT CALIB_HEAD_COUNT_RECALL_Z CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN CALIB_SAVE_TEST_SCORES CALIB_SAVE_FOLD_SCALES
  unset CALIB_SAVE_TEST_SCORES_AT CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_WEAK_D CALIB_SPOT_CHECK CALIB_EARLY_CHECK
  unset CALIB_LABELS_SCALE CALIB_HEAD_COUNT_BADS
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}head_count_recall_exponent,head_count_recall_z"
  export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}" CALIB_MEM="${CALIB_MEM:-2G}"
  export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}"
  export CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
  unset CALIB_PACK_GRES
  exec bash "$HERE/launch.sh" pack "$idx"
  ;;
*)
  echo "usage: head_count_5039.sh {dirs [ARM] | cpack ARM BETA [IDX]}" >&2; exit 1 ;;
esac
