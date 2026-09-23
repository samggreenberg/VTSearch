#!/usr/bin/env bash
# The size-generalisation matrix (#4051): train at one band, test at all three.
#
#   bash launch_bands.sh prepare      # exemplar crops for every cell
#   bash launch_bands.sh size 0,75    # time one whole-image and one patch cell
#   bash launch_bands.sh cells        # the full array
#   bash launch_bands.sh status
#
# #3156 asked whether cost rises as the target shrinks and answered it on the
# DIAGONAL: each arm trained at a band and was scored at the same band, because
# `scale_core._evaluable` excludes an image holding the class at another size
# from the cell entirely. #4044 asked for the other six squares and #4047 built
# them. This run fills the table.
#
# Per class: three training bands x three test bands, with `CALIB_TEST_BANDS=all`
# adding `fnr_small` / `fnr_medium` / `fnr_large` to every row. The diagonal
# reproduces what the shipped benchmark already reports -- `fnr_<own band>` IS
# the row's `fnr`, asserted per run -- so the new squares arrive beside a
# quantity whose value is already known, which is the cheapest available check
# that the cohorts were built off the right pool.
#
# ONE FPR, THREE FNRs. The three bands of a class share one negative pool by
# construction (the `3 *` in `SCALE_PREVALENCE`), so a negative holds no
# instance of the class and has no size for it. There is no per-band FPR to
# report and none is emitted; `quarry_export.py --check-bands` asserts the
# premise rather than assuming it.
#
# Shipped defaults only, as in launch_scale.sh: the contrast under test is the
# TEST BAND, so any other knob left non-default would be a second, uncontrolled
# one. `CALIB_TEST_BANDS` is not such a knob -- it ADDS columns and moves none,
# which `test_the_headline_columns_do_not_move` pins row by row.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/bands-4051}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

# `coco_quarry`, not `vg_scale`. The matrix is a claim about object size, and
# VG's boxes sit on a smaller instance than the frame's main one 8.3% of the
# time (#3924) -- an error that moves an image's BAND, which is the axis this
# study reads. COCO annotates all eighty classes exhaustively, so the band is
# derived from adjudicated boxes throughout.
export CALIB_DATASETS="${CALIB_DATASETS:-coco_quarry}"
# The region arm is the PAIR `siglip+dinov3_patch`, never bare `dinov3_patch`:
# DINOv3 has no text tower, so alone it opens on three random known-goods while
# the whole-image arm opens on a text sort, putting a seeding difference inside
# the voting-mode axis (#3276, #3278). Both halves are built for this dataset.
#
# Two columns rather than five. The encoder is a BLOCKING factor here -- the
# question is whether the size-transfer pattern survives the representation, so
# a second column REPLICATES the finding rather than competing with it -- and
# the region arm was 89% of `launch_scale.sh`'s wall clock, so each extra
# whole-image column is cheap and each extra patch column is not. Widen this
# only after `size` has been read on this grid.
export CALIB_COCO_QUARRY_EMBEDDERS="${CALIB_COCO_QUARRY_EMBEDDERS:-siglip,siglip+dinov3_patch}"
# All 75 cells are designated; selecting a subset would discard the design, and
# the matrix needs every band of every class to have both a diagonal and two
# off-diagonals.
export CALIB_CATEGORY_MODE=all

# THE STUDY. Every other export on this page exists to hold something still.
export CALIB_TEST_BANDS="${CALIB_TEST_BANDS:-all}"

# Sized to make a PER-CELL rate readable, because the off-diagonal is where the
# interesting variance is and it is measured on a held-out fraction of a band.
# `analyze_overview.py` refuses to print a per-cell rate under `--min-seeds`
# (10); 20 clears it with margin, and the band contrasts are pooled over 75
# categories, so 20 seeds puts 1,500 paired runs behind each one.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-20}"
# Walk every environment at seed 0, then seed 1, and so on. A SLURM array
# dispatches roughly in index order, so a run cut short loses SEEDS uniformly
# rather than whole classes: the matrix stays complete and only its standard
# errors widen, which a report can simply state. Losing whole classes would
# leave a table with holes in it, which it cannot.
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"
# The SHIPPED threshold path. A harness default is not a shipped default
# (`lessons/2026-08-12-a-study-default-is-not-a-shipped-default.md`), and this
# one defaulted to the #2781-era unfused control until #3400.
export CALIB_SAFE_THRESHOLDS="${CALIB_SAFE_THRESHOLDS:-1}"

# Declare the opening rather than letting it be decided per cell by whether a
# query happens to exist (#3278). `EXPERIMENT_QUERIES["coco_quarry"]` covers all
# 75 cells -- pinned by `test_every_cell_has_a_typed_query` -- so
# `REQUIRE_SEED_QUERY` drops nothing here and is a guard, not a selection knob.
export CALIB_REQUIRE_OPENING="${CALIB_REQUIRE_OPENING:-text}"
export CALIB_REQUIRE_SEED_QUERY="${CALIB_REQUIRE_SEED_QUERY:-1}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells"

# NOT SIZED YET. `launch_scale.sh` carries measured numbers for `vg_scale`; this
# grid is a different dataset, a different pickle size and one extra scoring
# pass per step, so those are a starting point and not a measurement.
#
# The marginal cost of `CALIB_TEST_BANDS` is EXPECTED to be small -- a cohort is
# positives-only (tens of images) against a test set whose ~9,900 negatives
# dominate every pass -- but expected is not measured. Run `size 0,75` and read
# the seconds and MaxRSS off `sacct` before quoting any wall clock or raising
# CONC. The defaults below are `launch_scale.sh`'s, inherited deliberately so
# that a first `size` run is safe rather than tuned.
MEM="${CALIB_MEM:-12G}"
CPUS="${CALIB_CPUS:-2}"
TIME="${CALIB_TIME:-6:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-70}"
JOB_NAME="${CALIB_JOB_NAME:-bands-$(basename "$CALIB_EXP")}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_COCO_QUARRY_EMBEDDERS=$CALIB_COCO_QUARRY_EMBEDDERS CALIB_CATEGORY_MODE=$CALIB_CATEGORY_MODE"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
# Named explicitly rather than left to --export=ALL: the launcher computes the
# cell COUNT and each task computes the cell LIST, and the two must enumerate
# the same grid. A knob reaching one but not the other maps an index to a
# different cell.
ENVX="$ENVX CALIB_CELL_ORDER=$CALIB_CELL_ORDER"
# The study's own knob travels with the rest for the same reason every other one
# does. It does not reach the cell LIST -- it adds columns to a row -- but a
# grid that silently ran without it is a grid that answered the old question,
# and nothing in the output would say so except three columns of NaN.
ENVX="$ENVX CALIB_TEST_BANDS=$CALIB_TEST_BANDS"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
ENVX="$ENVX CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS"
ENVX="$ENVX CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING CALIB_REQUIRE_SEED_QUERY=$CALIB_REQUIRE_SEED_QUERY"
ENVX="$ENVX CALIB_SKYLINE_ARMS=${CALIB_SKYLINE_ARMS:-}"

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job (#2897 lost both arms exactly this way).
submit() {
  local name="$1"; shift
  local J
  J=$(sbatch --parsable "$@") || { echo "SUBMIT FAILED for $name" >&2; return 1; }
  if [[ "$J" =~ ^[0-9]+$ ]]; then
    echo "$J" > "$LOGS/.jobid_$name"
    echo "$name -> job $J"
  else
    echo "$name SUBMIT FAILED (empty job id) — NOT LAUNCHED" >&2
    return 1
  fi
}

case "${1:-status}" in
prepare)
  # Assert the arm the study is FOR actually exists. A patch embedder on a
  # dataset the config forgot to mark boxed runs whole_image without complaint,
  # which is indistinguishable from success in every output.
  ( cd "$HERE" && python - <<'PYCHK' ) || { echo "REGION VOTING IS OFF for the patch arm — refusing to launch" >&2; exit 4; }
import os
import sys
sys.path.insert(0, os.getcwd())
import experiment_config as cfg
ds = cfg.DATASETS[0]
bad = [e for e in cfg.embedders_for_dataset(ds)
       if cfg.is_patch_embedder(e) and not cfg.region_voting_for(ds, e)]
if bad:
    print(f"  {ds}: patch embedders {bad} would run whole_image", file=sys.stderr)
    raise SystemExit(1)
for e in cfg.embedders_for_dataset(ds):
    print(f"  {ds} x {e}: styles={cfg.styles_for(ds, e)} region_voting={cfg.region_voting_for(ds, e)} "
          f"learn={cfg.learn_embedder(e)} text={cfg.text_embedder(e)}")
PYCHK
  # The premise the whole matrix rests on, asserted before anything is embedded
  # or queued: three FNRs are readable against ONE FPR only if the images behind
  # that FPR are the same for each band. True by construction -- and #3667 and
  # #3986 both changed what a negative IS, after the guarantee was written down.
  # One class per LINE: since #4056 a class name can hold spaces
  # (`enclosed road vehicle`), and word-splitting a space-joined list checked
  # three "classes" that do not exist and died on the first.
  ( cd "$WT/scripts/experiments/pile" && python - <<'PYC' | while IFS= read -r c; do python quarry_export.py --check-bands "$c" || exit 1; done ) || {
import sys; sys.path.insert(0, ".")
import pile_config as pc
print("\n".join(pc.SCALE_CLASSES))
PYC
    echo "SHARED NEGATIVE POOL CHECK FAILED — the matrix would not be paired" >&2; exit 5; }
  echo "CALIB_EXP=$CALIB_EXP  datasets=$CALIB_DATASETS  embedders=$CALIB_COCO_QUARRY_EMBEDDERS  seeds=$CALIB_N_SEEDS  test_bands=$CALIB_TEST_BANDS"
  submit prepare --job-name=bands-prep --mem=96G --cpus-per-task=8 \
    --time=3:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py"
  ;;

size)
  IDXS="${2:?usage: launch_bands.sh size <comma-separated cell indices>}"
  SIZE_RESULTS="$CALIB_EXP/sizing"
  mkdir -p "$SIZE_RESULTS/cells"
  ln -sfn "$CALIB_RESULTS/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_RESULTS/crops" "$SIZE_RESULTS/crops"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="bands-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time=2:00:00 --partition="$PARTITION" --export=ALL \
      --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$SIZE_RESULTS && cd $HERE && time python run_cells.py --index $idx"
  done
  echo "read the ACTUAL seconds and MaxRSS off 'sacct -j <id> --format=Elapsed,MaxRSS,State'"
  echo "before quoting a wall clock or raising CONC — the defaults here are inherited, not measured"
  ;;

cells)
  N=$(cd "$HERE" && python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
  fi
  echo "cells: $N (array 0-$((N-1))%$CONC on $PARTITION)"
  # Record the shape beside the results, before the array exists, so the
  # analysis chain has an honest denominator rather than a literal belonging to
  # whichever grid was current when someone typed it.
  python - "$CALIB_RESULTS/grid_shape.json" "$N" <<PYSHAPE
import json, sys

json.dump(
    {
        "n_cells": int(sys.argv[2]),
        "datasets": "$CALIB_DATASETS".split(","),
        "embedders": "$CALIB_COCO_QUARRY_EMBEDDERS".split(","),
        "n_seeds": int("$CALIB_N_SEEDS"),
        "max_steps": int("$CALIB_MAX_STEPS"),
        "cell_order": "$CALIB_CELL_ORDER",
        "test_bands": "$CALIB_TEST_BANDS",
        "job_name": "$JOB_NAME",
    },
    open(sys.argv[1], "w"),
    indent=2,
)
PYSHAPE
  PATCH_FLAG=""
  case "$CALIB_COCO_QUARRY_EMBEDDERS" in *_patch*) PATCH_FLAG="--patch" ;; esac
  REGION_FLAG=""
  case "$CALIB_COCO_QUARRY_EMBEDDERS" in
    *_patch*) REGION_FLAG="--require-region-voting ${CALIB_DATASETS%%,*}:$(
      tr ',' '\n' <<<"$CALIB_COCO_QUARRY_EMBEDDERS" | grep -- '_patch' | head -1)" ;;
  esac
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" $PATCH_FLAG \
    --require-text-seed $REGION_FLAG || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  submit cells --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

redo)
  # A failed task leaves its PREVIOUS output in place, so a partial re-run
  # silently mixes two runs' cells unless the stale files are deleted first.
  # Delete them, then pass the indices here. `--mem` must be sized from a cell
  # of the SAME KIND that failed: a max_patch cell peaks far above a whole_image
  # one, and sizing from the wrong kind is what produces OOMs.
  IDXS="${2:?usage: launch_bands.sh redo <comma-separated indices>}"
  echo "re-running indices: $IDXS (mem=$MEM)"
  submit redo --job-name="$JOB_NAME-redo" --array="$IDXS%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/redo-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R" | grep -E "bands|JOBID" || true
  echo "=== cells written ==="
  ls "$CALIB_RESULTS/cells" 2>/dev/null | wc -l
  ;;
*)
  echo "usage: launch_bands.sh {prepare|size <idx>|cells|redo <idx-list>|status}" >&2; exit 1 ;;
esac
