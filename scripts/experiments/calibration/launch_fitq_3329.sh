#!/usr/bin/env bash
# Is the 2-component Gaussian mixture a GOOD fit? (#3329, first bit)
#
#   bash launch_fitq_3329.sh prepare     # exemplar crops / category selection
#   bash launch_fitq_3329.sh list        # index -> cell map, for `size`
#   bash launch_fitq_3329.sh size 0,12   # time one binary and one region cell
#   bash launch_fitq_3329.sh baseline    # the click-0 text-sort anchor
#   bash launch_fitq_3329.sh cells       # the array
#   bash launch_fitq_3329.sh status
#
# THE QUESTION.  Every fit diagnostic this repo emits is *relative*:
# `evt_loglik_gain` prices a Gumbel+Normal mixture against a 2-Gaussian one,
# and the #2836 `tau_*` chain prices one cut against another.  None of them can
# say whether either model is any good, because a misspecification the two
# families share cancels in the comparison.  #3329 asks the absolute question -
# "we have ways to get the BEST fit, but I never look to see if it's a GOOD
# fit" - and this run is the first measurement of it.
#
# WHY THIS FIT, out of the ~15 the app performs.  It is the only one whose
# output is a user-visible decision on every item (the green/red line): the
# shipped fold-anchored mixture for trained detectors, and the unanchored
# `fit_score_gmm` for text sorts and the small-labelset fallback.  It is also
# where the surviving error is: with #2883 having shown `transfer` to be a
# reference artefact rather than a cost, `misspecification` (+0.0129) is the
# largest real term left in the #2836 decomposition, and #2881 closed the EVT
# *cut* line with "beating production needs a better FIT, not a better cut".
# Nobody then tested the fit.
#
# WHY vg_scale_any RATHER THAN vg_scale.  Every statistic here is
# prevalence-sensitive - a tail ratio, a balanced accuracy, a class-conditional
# moment - and `vg_scale_any` holds prevalence at 7.1% in EVERY cell by
# construction (300 positives against a shared 3900-negative pool).  The band
# axis `vg_scale` adds would vary prevalence and target size together with the
# thing under test.  Bands are the obvious follow-up once the instrument reads.
#
# THE GEOMETRY CORNER is the design.  Three arms fall out of two embedder
# columns, and they isolate the two axes independently:
#
#   siglip/whole_image            binary control
#   dinov3_patch/whole_image      same voting mode, different embedder
#   dinov3_patch/max_patch        same embedder, different voting mode
#
# The max-pooling hypothesis (H2) needs the third against the second, or a skew
# difference is just "DINOv3's scores are shaped differently".  On top of that
# each max_patch run emits BOTH poolings of the same media under the same model
# (`sim:pooled` and `sim:image`), which is the exactly-paired form of the same
# contrast - no cross-run matching at all.
#
# Shipped defaults everywhere else.  The contrast under test is the FIT, so any
# other knob left non-default would be a second, uncontrolled one.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/fitq-3329}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

# BLAS pinning at top level so `size` and the array measure the same thing.
# A login-node timing taken without this nearly cut an arm once; see
# scripts/experiments/lessons/2026-08-24-a-login-node-timing-nearly-cut-an-arm.md
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale_any}"
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-8}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-100}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"

# THE SHIPPED PATH, NAMED RATHER THAN INHERITED.  This study's whole fold scope
# reads `FoldAnchoredCut.fits`, which only exists on the fused path: without it
# the run silently emits no fold rows at all and measures the unanchored fit
# twice.  That is the premise of the study, so it is stated here rather than
# inherited.  (`CALIB_SAFE_THRESHOLDS` defaulted to 0 - pure cross-calibration,
# a threshold the app can no longer produce - until #3400; the pin is what kept
# this study off it, and is also what keeps it right when submitted into an
# older worktree.)
export CALIB_SAFE_THRESHOLDS=1

# The #3329 frame itself.
export CALIB_FIT_QUALITY=1
export CALIB_FIT_QUALITY_STRIDE="${CALIB_FIT_QUALITY_STRIDE:-5}"

# Every other side frame off.  They cost time and volume and answer questions
# this study is not asking; `__fitq` is the frame under test.
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_CUT_INCL_KS=""

# Declare the opening (#3278) rather than letting each cell decide it by
# whether a query happens to exist.  EXPERIMENT_QUERIES covers every
# vg_scale_any category, so this filter drops nothing - it is a guard.
export CALIB_REQUIRE_OPENING="${CALIB_REQUIRE_OPENING:-text}"
export CALIB_REQUIRE_SEED_QUERY="${CALIB_REQUIRE_SEED_QUERY:-1}"

LOGS="$CALIB_EXP/logs"
# `analysis/` is created here rather than by the job that writes into it:
# `text_baseline.py` does its whole pass and hands the path to pandas only at
# the end, so a missing directory costs the entire run and surfaces as a write
# error 47s in, with nothing to show for it.
mkdir -p "$LOGS" "$CALIB_RESULTS/cells" "$CALIB_EXP/analysis"

# MEASURED on this configuration by `size 0,12` (2026-08-30, jobs 601323/601324
# on rack2n09 -- backpack/seed 0 of each embedder, off a clean worktree at
# dev f8ad3253):
#
#   siglip / whole_image                  2m56s    607 MiB
#   siglip+dinov3_patch (BOTH styles)    21m16s    6.7 GiB
#
# The region cell sizes the array, and it is 7x the binary one because it runs
# whole_image AND max_patch in the one task.
#
# TIME drops from the inherited 6h to 2h -- 5.6x the measured region cell, which
# is headroom for the slower categories without parking a 6h reservation the
# backfill scheduler has to plan around.
#
# MEM stays at 12G even though the peak was 6.7 GiB: preflight check 7b floors
# patch arrays there because measured max_patch peaks run 9-14 GB across
# categories and under-sizing killed 74 of 108 cells in #3156. So 12G is now a
# measured 1.8x headroom on this cell rather than an inherited number.
#
# CPUS stays at 2: the region cell burned 20m49s of CPU over 21m14s wall (~1.0
# core) with BLAS pinned to one thread.
MEM="${CALIB_MEM:-12G}"
CPUS="${CALIB_CPUS:-2}"
TIME="${CALIB_TIME:-2:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
GRES="${CALIB_GRES:-none}"
CONC="${CALIB_CONC:-70}"
JOB_NAME="${CALIB_JOB_NAME:-fitq-$(basename "$CALIB_EXP")}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_VGSCALE_EMBEDDERS=$CALIB_VGSCALE_EMBEDDERS CALIB_CATEGORY_MODE=$CALIB_CATEGORY_MODE"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_CELL_ORDER=$CALIB_CELL_ORDER CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"
ENVX="$ENVX CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS"
ENVX="$ENVX CALIB_FIT_QUALITY=$CALIB_FIT_QUALITY CALIB_FIT_QUALITY_STRIDE=$CALIB_FIT_QUALITY_STRIDE"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS= CALIB_CUT_INCL_KS="
ENVX="$ENVX CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING CALIB_REQUIRE_SEED_QUERY=$CALIB_REQUIRE_SEED_QUERY"
ENVX="$ENVX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

# `--gres=none` is REWRITTEN AND REJECTED by the submit filter; the flag has to
# be dropped entirely for a CPU study.
GRES_ARG=(--gres="$GRES")
[[ "$GRES" == "none" || -z "$GRES" ]] && GRES_ARG=()

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
  # The dataset's own review coverage is a precondition, not a detail: a cell
  # whose reviewed images have been rebuilt away is measuring labels nobody
  # checked.
  ( cd "$WT/scripts/experiments/pile" && python check_review_coverage.py ) || {
    echo "REVIEW COVERAGE FAILED — refusing to launch a study on it" >&2; exit 3; }
  # Assert the region arm really region-votes. A patch embedder on a dataset the
  # config forgot to mark boxed runs whole_image without complaint, which is
  # indistinguishable from success in every output -- and would silently delete
  # this study's entire H2 contrast.
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
    print(f"  {ds} x {e}: styles={cfg.styles_for(ds, e)} region_voting={cfg.region_voting_for(ds, e)}")
PYCHK
  echo "CALIB_EXP=$CALIB_EXP  datasets=$CALIB_DATASETS  embedders=$CALIB_VGSCALE_EMBEDDERS  seeds=$CALIB_N_SEEDS"
  submit prepare --job-name=fitq-prep --mem=96G --cpus-per-task=8 \
    --time=3:00:00 --partition="$PARTITION" "${GRES_ARG[@]}" --export=ALL \
    --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py"
  ;;

list)
  ( cd "$HERE" && python - <<'PYLIST' )
import json
import os
import sys
sys.path.insert(0, os.getcwd())
import common
import experiment_config as cfg
from run_cells import _categories_by_dataset

info = json.loads((common.RESULTS / "prepare_info.json").read_text())
cells = cfg.array_cells(_categories_by_dataset(info))
print(f"{len(cells)} cells (order={cfg.CELL_ORDER})")
for i, c in enumerate(cells):
    styles = cfg.styles_for(c["dataset"], c["embedder"])
    print(f"  {i:4d}  {c['dataset']}/{c['embedder']}  styles={','.join(styles)}  {c['category']}  seed={c['seed']}")
PYLIST
  ;;

size)
  IDXS="${2:?usage: launch_fitq_3329.sh size <comma-separated cell indices>}"
  SIZE_RESULTS="$CALIB_EXP/sizing"
  mkdir -p "$SIZE_RESULTS/cells"
  ln -sfn "$CALIB_RESULTS/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_RESULTS/crops" "$SIZE_RESULTS/crops"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="fitq-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time=2:00:00 --partition="$PARTITION" "${GRES_ARG[@]}" --export=ALL \
      --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$SIZE_RESULTS && cd $HERE && time python run_cells.py --index $idx"
  done
  echo "when these finish: sacct -j <id> --format=Elapsed,MaxRSS,State  ->  set CALIB_MEM/CALIB_TIME from the REGION cell"
  ;;

baseline)
  # Click 0 is the free text sort, and `curves.py` refuses to draw without it.
  submit baseline --job-name=fitq-baseline --mem=16G --cpus-per-task=2 \
    --time=2:00:00 --partition="$PARTITION" "${GRES_ARG[@]}" --export=ALL \
    --output="$LOGS/baseline-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $CALIB_RESULTS --out $CALIB_EXP/analysis/text_baseline.csv"
  ;;

cells)
  N=$(cd "$HERE" && python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N'); is prepare done?" >&2; exit 1
  fi
  echo "cells: $N (array 0-$((N-1))%$CONC on $PARTITION)"
  # Record the shape beside the results BEFORE the array exists, so the analysis
  # can say "192 of 192" honestly rather than off a literal that belonged to
  # whichever grid was current when someone typed it.
  python - "$CALIB_RESULTS/grid_shape.json" "$N" <<PYSHAPE
import json, sys

json.dump(
    {
        "n_cells": int(sys.argv[2]),
        "datasets": "$CALIB_DATASETS".split(","),
        "embedders": "$CALIB_VGSCALE_EMBEDDERS".split(","),
        "styles": "$CALIB_PATCH_STYLES".split(","),
        "n_seeds": int("$CALIB_N_SEEDS"),
        "max_steps": int("$CALIB_MAX_STEPS"),
        "fit_quality_stride": int("$CALIB_FIT_QUALITY_STRIDE"),
        "cell_order": "$CALIB_CELL_ORDER",
        "job_name": "$JOB_NAME",
    },
    open(sys.argv[1], "w"),
    indent=2,
)
PYSHAPE
  REGION_ARM=$(tr ',' '\n' <<<"$CALIB_VGSCALE_EMBEDDERS" | grep -- '_patch' | head -1)
  REGION_FLAG=""
  [[ -n "$REGION_ARM" ]] && REGION_FLAG="--require-region-voting ${CALIB_DATASETS%%,*}:$REGION_ARM"
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" --patch \
    --require-text-seed --contrasts-voting-modes $REGION_FLAG || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  submit cells --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" "${GRES_ARG[@]}" --export=ALL \
    --output="$LOGS/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

redo)
  # A failed task leaves its PREVIOUS output in place, so a partial re-run
  # silently mixes two runs' cells unless the stale files are deleted first.
  IDXS="${2:?usage: launch_fitq_3329.sh redo <comma-separated indices>}"
  echo "re-running indices: $IDXS (mem=$MEM)"
  submit redo --job-name="$JOB_NAME-redo" --array="$IDXS%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" "${GRES_ARG[@]}" --export=ALL \
    --output="$LOGS/redo-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

status)
  echo "exp:     $CALIB_EXP"
  echo "queue:   $(squeue -u "$USER" -h -n "$JOB_NAME" -o %i | wc -l) tasks named $JOB_NAME"
  echo "cells:   $(find "$CALIB_RESULTS/cells" -name 'task_*.csv' ! -name '*__*' 2>/dev/null | wc -l) main frames"
  echo "fitq:    $(find "$CALIB_RESULTS/cells" -name 'task_*__fitq.csv' 2>/dev/null | wc -l) fit-quality frames"
  echo "empty:   $(find "$CALIB_RESULTS/cells" -name 'task_*.csv' -size 0 2>/dev/null | wc -l) zero-byte (delete before any resume)"
  ;;

*)
  echo "usage: $0 {prepare|list|size IDXS|baseline|cells|redo IDXS|status}" >&2
  exit 1
  ;;
esac
