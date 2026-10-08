#!/usr/bin/env bash
# #4637: where Autopilot's More walk picks once the opening's detector is shown. Two State of the App arms at one
# commit, paired run for run and identical through the Bad phase:
#   T  the app: the walk on the typed query's top
#   D  CALIB_MORE_WALK=detector: the walk on the detector's top, shown from the end of the Bad phase
# on both production paths and the app's three presets, trajectory pass only (no ceiling).
#
#   walk_4637.sh dirs SEEDS                  run dirs <root>/<date>-walk4637{T,D}-b{025,1,4}, on the 2026-10-06 grid
#   walk_4637.sh launch ARM BETA PATH SEEDS [FIRST [LAST]]
#                                            queue one arm x preset x path (ARM T|D, BETA 0.25|1|4, PATH binary|region)
#                                            of a SEEDS-seed grid, seeds FIRST (default 0) to LAST (default SEEDS-1)
#   walk_4637.sh price OUT SEEDS_BIN SEEDS_REG [DOCS]
#                                            extract, price both arms (handoff_price_4604.py), compare (walk_compare_4637.py)
#
# Launch from a FROZEN worktree (VTS_REPO is the worktree this script sits in, and every task imports it).
# Env: WALK_DATE (default 2026-10-07), SOTA_ROOT, and for `launch` anything launch.sh/launch_bands.sh reads
# (CALIB_MEM, CALIB_PARTITION, CALIB_PACK_*). `launch` uses `redo` (an array), or `pack` when WALK_PACK=1.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
S="${SOTA_ROOT:-/expscratch/sgreenberg/state-of-the-app}"
D="${WALK_DATE:-2026-10-07}"
GRID_SRC="${WALK_GRID:-$S/2026-10-06-b1/results}"  # prepare_data.py unchanged since this grid was prepared
BIN_BASE="${BIN_BASE:-/expscratch/sgreenberg/notch-4599/text_baseline-count-binary10.csv}"
REG_BASE="${REG_BASE:-/expscratch/sgreenberg/notch-4599/text_baseline-count-region2.csv}"
declare -A TAGB=([0.25]=b025 [1]=b1 [4]=b4)
declare -A BETA=([b025]=0.25 [b1]=1 [b4]=4)

dir_of() { echo "$S/$D-walk4637$1-$2"; }

indices() {  # PATH FIRST LAST: the grid's own index ranges (seed-major blocks of 144 binary then 144 region cells)
  python3 - "$1" "$2" "$3" <<'PY'
import sys
off = {"binary": 0, "region": 144}[sys.argv[1]]
print(",".join(f"{s * 288 + off}-{s * 288 + off + 143}" for s in range(int(sys.argv[2]), int(sys.argv[3]) + 1)))
PY
}

case "${1:-}" in
dirs)
  SEEDS="${2:?seeds}"
  for arm in T D; do
    for b in b025 b1 b4; do
      r="$(dir_of "$arm" "$b")/results"
      mkdir -p "$r/cells"
      ln -sfn "$GRID_SRC/prepare_info.json" "$r/prepare_info.json"
      ln -sfn "$GRID_SRC/crops" "$r/crops"
      walk=seed; [[ $arm == D ]] && walk=detector
      cat >"$r/grid_shape.json" <<EOF
{"n_cells": 288, "datasets": ["coco_better"], "embedders": ["siglip", "siglip+dinov3_patch"], "n_seeds": $SEEDS,
 "max_steps": 150, "cell_order": "seed", "test_bands": "all", "job_name": "walk4637$arm-$b",
 "note": "#4637 arm $arm (CALIB_MORE_WALK=$walk), beta ${BETA[$b]}, trajectory pass; grid symlinked from $GRID_SRC"}
EOF
    done
  done
  ls -d "$S/$D"-walk4637*
  ;;
launch)
  ARM="${2:?T|D}"; B="${3:?0.25|1|4}"; P="${4:?binary|region}"; SEEDS="${5:?seeds}"; FIRST="${6:-0}"; LAST="${7:-$((SEEDS - 1))}"
  (( FIRST >= 0 && FIRST <= LAST && LAST < SEEDS )) || { echo "seeds must satisfy 0 <= FIRST <= LAST < SEEDS" >&2; exit 2; }
  tag="${TAGB[$B]:?beta must be 0.25, 1 or 4}"
  walk=seed; [[ $ARM == D ]] && walk=detector
  [[ $P == binary ]] && export CALIB_MEM="${CALIB_MEM:-4G}"
  [[ $P == region ]] && export CALIB_MEM="${CALIB_MEM:-24G}"
  export SOTA_BETA="$B" SOTA_DATE="$D-walk4637$ARM-$tag" SOTA_PASS=trajectory SOTA_SEEDS="$SEEDS"
  export CALIB_MORE_WALK="$walk" CALIB_JOB_NAME="walk4637$ARM-$P-$tag"
  [[ $ARM == D ]] && export PREFLIGHT_DIVERGES="${PREFLIGHT_DIVERGES:+$PREFLIGHT_DIVERGES,}more_walk"
  idx="$(indices "$P" "$FIRST" "$LAST")"
  if [[ "${WALK_PACK:-0}" == 1 ]]; then
    exec bash "$HERE/launch.sh" pack "$(python3 -c "import sys; print(','.join(str(i) for r in sys.argv[1].split(',') for i in range(int(r.split('-')[0]), int(r.split('-')[1]) + 1)))" "$idx")"
  fi
  exec bash "$HERE/launch.sh" redo "$idx"
  ;;
price)
  OUT="${2:?out}"; SB="${3:?binary seeds}"; SR="${4:?region seeds}"; DOCS="${5:-}"
  mkdir -p "$OUT"
  declare -A EMB=([binary]=siglip [region]=siglip+dinov3_patch) NSEED=([binary]=$SB [region]=$SR)
  for arm in T D; do
    for b in b025 b1 b4; do
      e="$(dir_of "$arm" "$b")"
      for p in binary region; do
        [[ ${NSEED[$p]} -gt 0 ]] || continue  # a path left out (#4639: Region is its own issue)
        python "$HERE/handoff_extract_4604.py" --exp "$e" --embedder "${EMB[$p]}" --seeds "${NSEED[$p]}" --tag "${arm}_${p}_$b" --out "$OUT"
        base="$BIN_BASE"; [[ $p == region ]] && base="$REG_BASE"
        python "$HERE/handoff_price_4604.py" --steps "$OUT/steps_${arm}_${p}_$b.csv.gz" --picks "$OUT/picks_${arm}_${p}_$b.csv.gz" \
          --baseline "$base" --beta "${BETA[$b]}" --tag "${arm}_${p}_$b" --out "$OUT" >"$OUT/price_${arm}_${p}_$b.log"
      done
    done
  done
  python "$HERE/walk_compare_4637.py" --dir "$OUT" ${DOCS:+--docs "$DOCS"}
  ;;
*)
  echo "usage: walk_4637.sh {dirs SEEDS | launch ARM BETA PATH SEEDS | price OUT SEEDS_BIN SEEDS_REG [DOCS]}" >&2; exit 1 ;;
esac
