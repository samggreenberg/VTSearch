#!/usr/bin/env bash
# #3796: how wide is the distribution the pinned Train/Calibrate split draws from?
#
#   bash launch_calseed_3796.sh prepare      # stage 0, ONCE
#   bash launch_calseed_3796.sh size [cell]  # time ONE cell before committing
#   bash launch_calseed_3796.sh baseline     # the click-0 text anchor
#   bash launch_calseed_3796.sh run          # the grid
#   bash launch_calseed_3796.sh analyze      # re-analyse finished cells
#
# WHY THIS GRID IS INVERTED.  Every other sweep here varies a knob to find a
# better setting.  This one varies a knob that is *already decided* - production
# pins the fold split to `CALIBRATION_SPLIT_SEED` (42) and #2934 pinned it on
# purpose - to find out what a single sample from that pin was worth believing.
# So the cell seed is HELD and the split is REDRAWN: same medias voted, same
# order, same held-out test set, one thing moving.  The spread across the draws
# is the error bar on every single-seed number this repo quotes, and a contrast
# smaller than it is not a finding.
#
# The draws are cells rather than arms (the #3287 shape) because #3794 gave the
# harness a `calibration_seed` COLUMN: one results dir, one array, one frame,
# and the column says which draw each row came from.  An arm per draw would be
# twenty arrays whose only difference is already written on every row.
#
# WHAT IT IS NOT.  It is not an argument for unpinning the split, and a "best"
# draw is not a result - 42 is not better than 7, it is one sample.  The one
# thing 42 gets is a place in the distribution, which is why it is drawn FIRST.
#
# Design + pre-registered decision rules:
#   docs/experiments/2026-09-11-calibration-seed-3796/PLAN.md
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

MODE="${1:-run}"

export VTS_REPO="${VTS_REPO:-/exp/$USER/projects/vts-calseed-3796}"
WT="$VTS_REPO"
HERE="$WT/scripts/experiments/calibration"

BASE="${CALSEED_BASE:-/expscratch/$USER/calseed-3796}"
PREP="$BASE/prepare/results"
RUN="$BASE/run"

# --- the swept axis ----------------------------------------------------------
# 42 FIRST, and it is not decoration.  It is production's own draw, so it is
# both the value every other study in this repo reports and the only draw whose
# loss to a truncated array would cost the study its reference point.  Under
# CALIB_CELL_ORDER=calibration_seed the array walks draw-major, so the first
# block of indices is the whole grid at 42 and a cut takes the LAST draws.
#
# The other nineteen are 0..18 rather than 43..61: they are arbitrary by
# construction and consecutive-from-zero says so, where a block adjacent to 42
# invites a reader to look for a pattern that cannot exist.
DRAWS="${CALSEED_DRAWS:-42,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18}"

# --- science knobs -----------------------------------------------------------
# The SHIPPED threshold path (docs/ML.md: fusion is not a setting).  Named, not
# left unset, for the reason #3287 names it: the split this study redraws lives
# INSIDE the fused path, so the path is the premise of the sweep.
export CALIB_SAFE_THRESHOLDS=1
# CALIB_HEAD unset -> the production linear SVM.  CALIB_BLEND_SCHEDULE unset ->
# the app's per-mode default.  CALIB_CALIBRATION_FRACTION unset -> the app's
# per-space default (#3290), which is the whole point: this study measures the
# noise around what a user actually gets, so every knob but the draw is the
# app's own.
export CALIB_CALIBRATION_SEEDS="$DRAWS"
export CALIB_ANALYZE=noop.py

# --- environment -------------------------------------------------------------
# `vg_scale_any`, not the band-suffixed `vg_scale`: identical prevalence in
# every cell.  A threshold IS a quantile of the calibration set, so a grid whose
# cells' calibration sets differ in size would confound the spread this study
# measures with the thing that drives it.  (The issue says "vg_scale"; this is
# the member of that family built for calibration studies, and #3287 and #3115
# both ran on it.)
export CALIB_DATASETS="${CALIB_DATASETS:-vg_scale_any}"
# Both voting modes.  The issue asks about the production region column, and one
# environment is where a spread becomes "a law" that is really two cells
# (#3115, #3287).  The region half is the PAIR: DINOv3 has no text tower, so a
# bare `dinov3_patch` arm would open on three random known-goods while the
# SigLIP arm opens on a typed query, putting a seeding contrast inside the mode
# contrast (#3278).
export CALIB_VGSCALE_EMBEDDERS="${CALIB_VGSCALE_EMBEDDERS:-siglip,siglip+dinov3_patch}"
export CALIB_REQUIRE_OPENING=text
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_CATEGORY_MODE="${CALIB_CATEGORY_MODE:-prevalence}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-5}"
# BOTH styles on the patch embedder, so voting mode is separable from embedder
# (#3115's confound-breaking corner): siglip/whole vs dinov3/whole is the
# EMBEDDER at fixed mode, dinov3/whole vs dinov3/max_patch is the MODE at fixed
# embedder.  If the noise floor differs between the modes, this is what says
# whether the mode or the representation did it.
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_REPOOL_VARIANTS=""

# --- sizing ------------------------------------------------------------------
# 150 steps because one of the three deliverables is whether the spread SHRINKS
# with vote count -- the mechanics say a bigger labelset makes any one split
# less pivotal -- and that is a curve over the horizon, not a number at its end.
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"
# 5 cell seeds: the comparison the issue asks for is spread-across-DRAWS against
# spread-across-CELL-SEEDS, and the second is an sd taken across seeds at the
# pinned draw, so it needs seeds to be taken across.
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-5}"
# Draw-major.  See CELL_ORDER in experiment_config.py: it is the #3287 argument
# one level out.  A truncated run here loses whole DRAWS uniformly -- every
# (class, cell seed) block keeps the same, smaller set of draws, so the design
# survives and only the sd's own precision falls.  Under `seed` or `category`
# the draws are innermost and a cut deletes whole blocks instead, leaving the
# draw-spread and the seed-spread estimated on different environments -- which
# is precisely the comparison this study exists to make.
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-calibration_seed}"

# No GPU work: the cells train a small head on cached pile embeddings.  BOTH
# must be named -- the flag is dropped rather than passed as `--gres=none`,
# which this cluster's submit filter rewrites and then rejects (#2897 lost both
# A/B arms to exactly that, with an empty job id as the only symptom).
export CALIB_PARTITION=cpu
export CALIB_GRES=none

# --- resources ---------------------------------------------------------------
# MEASURED on THIS pile by `size`, not carried from #3287.  #3287's 21.8-min /
# 7.7-GB region cell ran `vg_scale_any` at 4,200 medias; the 2026-09-08 rebuild
# put 18,049 in it, and #3679 measured the same change as 4.4 -> 19.1 min per
# cell.  Never carry a per-cell cost across a pile rebuild.
export CALIB_MEM="${CALIB_MEM:-32G}"
export CALIB_CONC="${CALIB_CONC:-24}"
export CALIB_CPUS=1
export CALIB_TIME="${CALIB_TIME:-12:00:00}"
# The analyzer reads every step of every cell.  #3679 needed 96G for 8.7M rows
# off 1,875 cells and was OOM-killed at 16G.
export CALIB_ANALYZE_MEM="${CALIB_ANALYZE_MEM:-96G}"
export CALIB_ANALYZE_TIME="${CALIB_ANALYZE_TIME:-3:00:00}"

# Pin the BLAS pools to one thread each, in `size` and `run` alike: a cell timed
# with different threading than the array will use is a guess with a unit
# attached.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Read the pre-embedded pile in place: no re-embed, no GPU, no model download.
# shellcheck disable=SC1091
source "$WT/scripts/experiments/pile/pile_env.sh"

require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    echo "       Nothing downstream can run; fix the submission and re-launch." >&2
    exit 1
  fi
}

ENV_COMMON() {
  echo "export CALIB_EXP=$1 CALIB_RESULTS=$2 CALIB_CALIBRATION_SEEDS=$CALIB_CALIBRATION_SEEDS" \
       "VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
}

case "$MODE" in
  prepare)
    export CALIB_EXP="$BASE/prepare"
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/prepare/logs" "$PREP/cells" "$PREP/crops"
    ENVX="$(ENV_COMMON "$CALIB_EXP" "$CALIB_RESULTS")"
    P=$(sbatch --parsable --job-name=cs3796-prep --mem=48G --cpus-per-task=2 \
      --time=2:00:00 --partition=cpu --export=ALL \
      --output="$BASE/prepare/logs/prepare-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
    require_jobid "$P" "prepare"
    echo "prepare job: $P  ->  $BASE/prepare/logs/prepare-$P.out"
    echo "cells enumerate from $PREP/prepare_info.json"
    ;;

  size)
    # Time ONE cell before committing to the array.  Under draw-major ordering
    # the first block is the whole grid at draw 42, ordered (seed, env): with 5
    # classes and two embedders, cell 0 is `siglip` (binary) and cell 5 the
    # `siglip+dinov3_patch` pair (region).  SIZE BOTH.  The region cell sets the
    # memory limit and the critical path, and a limit sized off the binary one
    # is a limit the region arm cannot run in.
    IDX="${2:-0}"
    export CALIB_EXP="$BASE/sizing"
    export CALIB_RESULTS="$PREP"
    SIZING="$BASE/sizing/cells"
    mkdir -p "$BASE/sizing/logs" "$SIZING"
    ENVX="$(ENV_COMMON "$CALIB_EXP" "$CALIB_RESULTS")"
    S=$(sbatch --parsable --job-name=cs3796-size --mem="${CALSEED_SIZE_MEM:-64G}" --cpus-per-task="$CALIB_CPUS" \
      --time="$CALIB_TIME" --partition=cpu --export=ALL \
      --output="$BASE/sizing/logs/size-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python run_cells.py --index $IDX --outdir $SIZING")
    require_jobid "$S" "size"
    echo "size job: $S (cell $IDX)  ->  $BASE/sizing/logs/size-$S.out"
    echo "read it back with: sacct -j $S --format=Elapsed,MaxRSS,State"
    ;;

  baseline)
    # The click-0 anchor: what typing the query was worth before any clicking.
    # `curves` refuses to draw a quality-over-clicks figure without it, and the
    # distance between the far left and the far right is what the spread has to
    # be read against -- a 0.02 sd means one thing against a 0.4 improvement
    # and another against a 0.05 one.
    export CALIB_RESULTS="$PREP"
    mkdir -p "$BASE/analysis" "$BASE/logs"
    ENVX="$(ENV_COMMON "$BASE" "$PREP")"
    T=$(sbatch --parsable --job-name=cs3796-baseline --mem=48G --cpus-per-task=2 \
      --time=2:00:00 --partition=cpu --export=ALL \
      --output="$BASE/logs/baseline-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python text_baseline.py --results $PREP --out $BASE/analysis/text_baseline.csv")
    require_jobid "$T" "text baseline"
    echo "baseline job: $T  ->  $BASE/analysis/text_baseline.csv"
    ;;

  run)
    if [[ ! -f "$PREP/prepare_info.json" ]]; then
      echo "ERROR: no prepare_info.json at $PREP - run '$0 prepare' first." >&2
      exit 1
    fi
    # Preflight's checks are mostly PYTHON (it imports vtscore to compare every
    # pinned knob against its shipped constant), and a non-interactive login
    # shell has no venv.  Activate it first, or those checks come back FAIL for
    # a reason unrelated to the run.
    # shellcheck disable=SC1091
    source "$WT/gridenv.sh" >/dev/null 2>&1 || {
      echo "ERROR: could not activate the venv at $WT/gridenv.sh" >&2; exit 1
    }

    export CALIB_EXP="$RUN"
    export CALIB_RESULTS="$RUN/results"
    mkdir -p "$CALIB_EXP/logs" "$CALIB_RESULTS/cells"
    [[ -e "$CALIB_RESULTS/prepare_info.json" ]] || ln -s "$PREP/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
    [[ -e "$CALIB_RESULTS/crops" ]] || ln -s "$PREP/crops" "$CALIB_RESULTS/crops"

    if [[ -x "$WT/scripts/experiments/preflight.sh" ]]; then
      bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --need-gb 40 \
        --require-min-positives 100 \
        --reuse-prepare "$PREP" \
        --require-region-voting vg_scale_any:siglip+dinov3_patch --contrasts-voting-modes --patch \
        --diverges "calibration_seed" \
        --job-name cs3796-cells --mem "$CALIB_MEM" --conc "$CALIB_CONC" || {
        echo "preflight FAILED" >&2
        [[ "${PREFLIGHT_SKIP:-0}" == "1" ]] || exit 1
      }
    fi

    CALIB_JOB_NAME=cs3796-cells bash "$HERE/launch_cells.sh" || {
      echo "cells FAILED to submit" >&2; exit 1
    }
    CELLS_ID="$(cat "$CALIB_EXP/logs/.cells_jobid" 2>/dev/null || true)"
    require_jobid "$CELLS_ID" "the cells array"

    # The authoritative shape, written where `run_cells.cell_progress` and the
    # analyzer both read it (#3736).  A task log re-enumerates from its own
    # environment and so keeps quoting the LAUNCHED count after a cut; this file
    # is what a cut rewrites.
    N_CELLS=$(cd "$HERE" && python run_cells.py --print-cells 2>/dev/null | tail -1)
    python3 - "$CALIB_RESULTS/grid_shape.json" "$N_CELLS" "$CELLS_ID" "$DRAWS" <<'PYSHAPE'
import json, os, sys
path, n_cells, job, draws = sys.argv[1:5]
json.dump(
    {
        "n_cells": int(n_cells),
        "job": job,
        "datasets": os.environ["CALIB_DATASETS"],
        "embedders": os.environ["CALIB_VGSCALE_EMBEDDERS"],
        "styles": os.environ["CALIB_PATCH_STYLES"],
        "n_categories": int(os.environ["CALIB_N_CATEGORIES"]),
        "n_seeds": int(os.environ["CALIB_N_SEEDS"]),
        "calibration_seeds": [int(d) for d in draws.split(",")],
        "max_steps": int(os.environ["CALIB_MAX_STEPS"]),
        "cell_order": os.environ["CALIB_CELL_ORDER"],
        "note": "#3796: cell seed HELD, Train/Calibrate split REDRAWN. Draw 42 is production's pin and is first.",
    },
    open(path, "w"),
    indent=2,
)
PYSHAPE
    echo "grid shape -> $CALIB_RESULTS/grid_shape.json ($N_CELLS cells)"

    ALOGS="$BASE/logs"; mkdir -p "$ALOGS" "$BASE/analysis"
    AENVX="$(ENV_COMMON "$RUN" "$CALIB_RESULTS")"
    A=$(sbatch --parsable --dependency="afterany:$CELLS_ID" --job-name=cs3796-analyze \
      --mem="$CALIB_ANALYZE_MEM" --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" \
      --partition=cpu --export=ALL --output="$ALOGS/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_calseed_3796.py --results $CALIB_RESULTS --out $BASE/analysis --baseline $BASE/analysis/text_baseline.csv")
    require_jobid "$A" "the analyze step"

    echo
    echo "cells array: $CELLS_ID   analyze: $A (afterany)"
    echo "report -> $BASE/analysis/REPORT_calseed.md"
    echo
    echo "A submission is not a launch: confirm the ids above are numeric and that"
    echo "cells appear under $CALIB_RESULTS/cells before quoting an ETA."
    ;;

  analyze)
    export CALIB_EXP="$RUN"
    export CALIB_RESULTS="$RUN/results"
    ALOGS="$BASE/logs"; mkdir -p "$ALOGS" "$BASE/analysis"
    AENVX="$(ENV_COMMON "$RUN" "$CALIB_RESULTS")"
    A=$(sbatch --parsable --job-name=cs3796-analyze --mem="$CALIB_ANALYZE_MEM" \
      --cpus-per-task=4 --time="$CALIB_ANALYZE_TIME" --partition=cpu --export=ALL \
      --output="$ALOGS/analyze-%j.out" \
      --wrap="source $WT/gridenv.sh && $AENVX && cd $HERE && python analyze_calseed_3796.py --results $CALIB_RESULTS --out $BASE/analysis --baseline $BASE/analysis/text_baseline.csv")
    require_jobid "$A" "the analyze step"
    echo "analyze job: $A  ->  $ALOGS/analyze-$A.out"
    ;;

  *)
    echo "usage: $0 {prepare|size [cell]|baseline|run|analyze}" >&2
    exit 2
    ;;
esac
