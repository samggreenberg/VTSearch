#!/usr/bin/env bash
# #5040: the labels line's chances, closed loop. Offline on #4937's snapshots (chances_5040.py), fitting the 3-part
# fit's Bad component on the corpus gained +0.05 to +0.06 F-beta at beta 1/4 and +0.02 to +0.05 at beta 1 on the same
# heads, and with the counted bound loosened to 4x it also gained at beta 4. The chances also steer Autopilot (its
# acquisition reads them, its stop reads their expected F-beta), so the sessions themselves change: this prices them
# end to end. Every arm is today's app to 200 clicks, closing with the end-of-run check, and differs only in the chances:
#
#   ctl    the app before #5040: the labels' Bad component, the counted bound at 2x (CALIB_LINE_BAD=labels
#          CALIB_LINE_BOUND_MARGIN=2; the study ran at 4783a51eb, where that was the app and set nothing)
#   bssl   the Bad component fitted on the corpus, the held-out Bads counted as labelled points of it (CALIB_LINE_BAD=corpus)
#   bssl4  the same, the counted bound at 4x (CALIB_LINE_BOUND_MARGIN=4)
#   bmean4 the Bad component's mean alone fitted on the corpus, the bound at 4x (CALIB_LINE_BAD=corpus_mean), added
#          after the offline Region reading: the full fit overrates the positives near the line there, the mean does
#          not. The app since #5040: its knobs are the app's values, so the arm sets none.
#
# The Binary Photo path at beta 1/4, 1 and 4, 5 seeds x 144 cells, on the grid #4743 priced (2026-10-08-b1), paired by
# cell and seed: an arm's run is identical to ctl's until the first click whose labels line the class model draws.
# Trajectory pass (no ceiling); cells gzipped.
#
#   chances_5040.sh dirs [ARM]               run dirs <root>/<date>-chances5040<tag>-<arm>-b{025,1,4} (every arm, or ARM)
#   chances_5040.sh launch ARM BETA [IDX]    one arm at one preset as a cpu array (default: every seed's cells)
#   chances_5040.sh cpack ARM BETA [IDX]     the same as cpu-partition packs: one job runs CALIB_PACK_PAR cells at
#                                            CALIB_CPUS (default 1) thread each, for when the 2,000-job submit cap is
#                                            full; a cell is charged one CPU, not an array task's whole core
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: CHANCES5040_DATE (default 2026-10-17), CHANCES5040_SEEDS (5), CHANCES5040_TAG (e.g. smoke), SOTA_ROOT,
# CHANCES5040_DATASET (coco_better; o365_better re-reads the contrast on the confirmation bench, #4817) with
# CHANCES5040_GRID (that dataset's prepared grid: its results dir, holding prepare_info.json and crops/), and
# anything launch.sh reads (CALIB_MEM, CALIB_CONC, CALIB_TIME, CALIB_PACK_JOBS, CALIB_PACK_PAR, CALIB_PACK_TIME).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${CHANCES5040_DATE:-2026-10-17}"
SEEDS="${CHANCES5040_SEEDS:-5}"
ARMS=(ctl bssl bssl4 bmean4)
declare -A KNOBS=([ctl]="CALIB_LINE_BAD=labels CALIB_LINE_BOUND_MARGIN=2" [bssl]="CALIB_LINE_BAD=corpus CALIB_LINE_BOUND_MARGIN=2"
  [bssl4]="CALIB_LINE_BAD=corpus" [bmean4]="")
declare -A DIVERGES=([ctl]="line_bad,line_bound_margin" [bssl]="line_bad,line_bound_margin" [bssl4]="line_bad" [bmean4]="")
GRID_SRC="${CHANCES5040_GRID:-$S/2026-10-08-b1/results}"  # prepare_data.py unchanged since (one docstring edit)
DATASET="${CHANCES5040_DATASET:-coco_better}"

btag() { case "$1" in 0.25) echo 025 ;; 1) echo 1 ;; 4) echo 4 ;; *) echo "beta must be 0.25, 1 or 4" >&2; exit 2 ;; esac; }
run_of() { echo "chances5040${CHANCES5040_TAG:-}-$1-b$(btag "$2")"; }

# The grid's own block sizes: seed-major, then embedder (siglip, the Binary path, first), then cell.
# coco_better: 288 a seed, the first 144 Binary.
grid_sizes() {
  python3 - "$GRID_SRC/prepare_info.json" "$DATASET" <<'PY'
import json, sys
info = json.load(open(sys.argv[1]))["datasets"][sys.argv[2]]
sizes = [len(info[e]["selected_categories"]) for e in ("siglip", "siglip+dinov3_patch")]
print(sum(sizes), sizes[0])
PY
}
all_idx() {
  read -r per_seed n_binary < <(grid_sizes)
  python3 - "$SEEDS" "$per_seed" "$n_binary" <<'PY'
import sys
seeds, per_seed, n_binary = (int(a) for a in sys.argv[1:])
print(",".join(str(s * per_seed + i) for s in range(seeds) for i in range(n_binary)))
PY
}

case "${1:-}" in
dirs)
  for arm in "${ARMS[@]}"; do
    [[ -z "${2:-}" || "$2" == "$arm" ]] || continue
    knobs="${KNOBS[$arm]:-the shipped chances}"
    for b in 0.25 1 4; do
      r="$S/$D-$(run_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      read -r per_seed _ < <(grid_sizes)
      cat >"$r/grid_shape.json" <<JSON
{"n_cells": $per_seed, "datasets": ["$DATASET"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 200, "cell_order": "seed", "test_bands": "all", "job_name": "$(run_of "$arm" "$b")",
 "note": "#5040 arm $arm ($knobs), beta $b, 200 clicks, binary path only, trajectory pass; grid symlinked from $GRID_SRC"}
JSON
      echo "$r"
    done
  done
  ;;
launch|cpack)
  arm="${2:?arm (${ARMS[*]})}" b="${3:?beta (0.25, 1 or 4)}"
  [[ -v "KNOBS[$arm]" ]] || { echo "arm must be one of ${ARMS[*]}" >&2; exit 2; }
  idx="${4:-$(all_idx)}"
  run="$(run_of "$arm" "$b")"
  mem_in="${CALIB_MEM:-}"
  export CALIB_CELLS_GZIP="${CALIB_CELLS_GZIP:-1}" CALIB_MEM="${CALIB_MEM:-3G}" CALIB_TIME="${CALIB_TIME:-2:00:00}"
  export SOTA_BETA="$b" SOTA_DATE="$D-$run" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS" SOTA_MAX_STEPS=200
  export SOTA_DATASET="$DATASET"
  export SOTA_RANK_FRAME_STEPS="${SOTA_RANK_FRAME_STEPS:-10,25,50,100,150,200}"
  export CALIB_JOB_NAME="$run"
  unset CALIB_LINE_BAD CALIB_LINE_BOUND_MARGIN CALIB_SAVE_TEST_SCORES CALIB_SAVE_FOLD_SCALES CALIB_SAVE_TEST_SCORES_AT
  unset CALIB_WEAK_REPEAT CALIB_WEAK_MIN_T CALIB_WEAK_D CALIB_SPOT_CHECK CALIB_EARLY_CHECK CALIB_LABELS_SCALE
  for kv in ${KNOBS[$arm]}; do export "${kv?}"; done
  if [[ -n "${DIVERGES[$arm]}" ]]; then
    export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}${DIVERGES[$arm]}"
  fi
  if [[ "$1" == cpack ]]; then
    export CALIB_PARTITION=cpu CALIB_CPUS="${CALIB_CPUS:-1}" CALIB_MEM="${mem_in:-2G}"
    export CALIB_PACK_JOBS="${CALIB_PACK_JOBS:-1}" CALIB_PACK_PAR="${CALIB_PACK_PAR:-16}"
    export CALIB_TIME="${CALIB_PACK_TIME:-12:00:00}"
    unset CALIB_PACK_GRES
    exec bash "$HERE/launch.sh" pack "$idx"
  fi
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
*)
  echo "usage: chances_5040.sh {dirs [ARM] | launch ARM BETA [IDX] | cpack ARM BETA [IDX]}" >&2; exit 1 ;;
esac
