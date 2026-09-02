#!/usr/bin/env bash
# Broad VTSearch overview benchmark: production defaults across the pile.
#
#   bash launch_bench.sh prepare      # category selection + exemplar crops
#   bash launch_bench.sh size         # time one cheap + one expensive cell
#   bash launch_bench.sh cells        # the full array
#   bash launch_bench.sh status
#
# Everything here rides the SHIPPED defaults on purpose: this study asks "what
# does a current VTSearch user get?", so the only knobs set are sizing ones
# (which datasets/embedders/categories/seeds), never behaviour ones.  A knob
# that changes the decision rule would make these numbers unreadable as a
# baseline.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/bench-overview}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"

# --- the grid (sizing only) ------------------------------------------------
export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,caltech101_m,coco_val}"
# The region arm is the PAIR `siglip+dinov3_patch`, not bare `dinov3_patch`
# (#3278).  DINOv3 has no text tower, so on its own it opens on three random
# known-goods while both SigLIP arms open on a typed query -- and this benchmark
# reads its three arms SIDE BY SIDE per dataset, which is precisely where an
# arm-dependent difference in the opening stops cancelling.  The pair holds the
# opening fixed (SigLIP ranks the query for every arm) and lets the arms differ
# only in what the detector learns in, which is the axis being reported.
#
# The honest caveat, because this study's charter is "what does a current
# VTSearch user get?": the app cannot pair two spaces today (#3276 "Not in
# scope"), so a real user with a DINOv3 detector *does* get the known-good
# start.  The paired arm therefore prices REGION VOTING rather than today's
# DINOv3 flow end to end.  That is the right trade for a table whose rows are
# compared against each other -- the alternative measures voting mode and
# opening at once and can separate neither -- but it is a divergence, not a
# default, which is why it is written down here.
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,siglip2_l,siglip+dinov3_patch}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip,siglip2_l,siglip+dinov3_patch}"
export CALIB_COCO_EMBEDDERS="${CALIB_COCO_EMBEDDERS:-siglip,siglip2_l,siglip+dinov3_patch}"
# One declaration, two enforcement points: preflight check 14 refuses the array
# if any selected cell would take the other start, and run_cells.py raises on a
# cell that did.  Either alone leaves a hole -- preflight cannot see a cell that
# resumes months later, and a per-cell raise costs a queue slot per cell.
export CALIB_REQUIRE_OPENING=text
# ...and the other half of that declaration, which pairing makes mandatory rather
# than tidy: a paired arm has no known-good fallback (falling back would run bare
# `dinov3_patch` under the pair's name, so `run_cells.py` raises instead), and
# category selection here runs over the dataset's FULL vocabulary while the query
# tables cover part of it.  `CALIB_REQUIRE_SEED_QUERY=1` filters the ineligible
# categories out BEFORE selection, so each is replaced by the next eligible one
# rather than shrinking the grid.  It does move the selected set: that is the
# price of every cell opening the way a user would, and the alternative is cells
# that die one at a time deep inside the array.
export CALIB_REQUIRE_SEED_QUERY=1
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-3}"
export CALIB_N_PER_BAND="${CALIB_N_PER_BAND:-2}"
export CALIB_N_CATEGORIES="${CALIB_N_CATEGORIES:-6}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-150}"

# Counterfactual extras off: this is a baseline, not a contrast.
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""

# PRODUCTION GEOMETRY ONLY.  Per vtscore/eval/patch_styles.py, `max_patch` IS
# the production patch pipeline, while the HAC hybrids lost the Max-Patch study
# at the operating point (PR #2749) and production no longer carries the tree
# they delegate to.  Running them here would put non-production rows in a
# benchmark whose whole purpose is "what does a current user get", and would
# double the cost of every patch cell.  (The calibration study's default WAS
# "max_patch,max_patch_pca_hac" -- a *study* default, not a *shipped* one -- and
# this pin is why that never reached the benchmark.  #3400 fixed the default
# itself; the pin stays, because it is also what holds when this launcher
# submits into an older worktree.)
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-max_patch}"

# THE SHIPPED THRESHOLD PATH, for the same reason.  `docs/ML.md`: "Every trained
# threshold fuses the haystack into the cut.  There is no setting for this."
# The harness default was the #2781-era unfused control until #3400, so this
# baseline WAS measuring a threshold no user can get -- the exact hazard the
# geometry pin above was added for, one knob over.
export CALIB_SAFE_THRESHOLDS="${CALIB_SAFE_THRESHOLDS:-1}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells"

MEM="${CALIB_MEM:-64G}"
CPUS="${CALIB_CPUS:-6}"
TIME="${CALIB_TIME:-6:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-24}"
# Distinct per experiment dir: two arrays sharing a --job-name break every
# per-name query, including the completion waiter in the grid-experiments skill.
JOB_NAME="${CALIB_JOB_NAME:-bench-$(basename "$CALIB_EXP")}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO"
ENVX="$ENVX CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS CALIB_CALTECH_EMBEDDERS=$CALIB_CALTECH_EMBEDDERS"
ENVX="$ENVX CALIB_COCO_EMBEDDERS=$CALIB_COCO_EMBEDDERS"
# Named explicitly for the same reason CALIB_PATCH_STYLES is below: it decides
# what the run is allowed to have measured, so it must not depend on the
# submitting shell's environment surviving into the job.
ENVX="$ENVX CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING CALIB_REQUIRE_SEED_QUERY=$CALIB_REQUIRE_SEED_QUERY"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_N_PER_BAND=$CALIB_N_PER_BAND"
ENVX="$ENVX CALIB_N_CATEGORIES=$CALIB_N_CATEGORIES CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
# Explicit, not via --export=ALL.  This one decides whether the run measures the
# production geometry or silently adds a retired arm, so it must not depend on
# the submitting shell's environment surviving into the job.
ENVX="$ENVX CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS"

# A submission is not a launch: --parsable returns an EMPTY id when the submit
# filter refuses the job (#2897 lost both arms this way).
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
  echo "CALIB_EXP=$CALIB_EXP"
  echo "datasets=$CALIB_DATASETS seeds=$CALIB_N_SEEDS per_band=$CALIB_N_PER_BAND"
  submit prepare --job-name=bench-prep --mem=96G --cpus-per-task=8 \
    --time=3:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py"
  ;;

size)
  # Size from a REAL cell, not a guess (and one per cost class: a patch cell can
  # be 10x a whole-image one, which changes the array budget entirely).
  # Writes into a throwaway results dir so the timing cells are not mistaken for
  # study cells by the resume logic.
  IDXS="${2:?usage: launch_bench.sh size <comma-separated cell indices>}"
  SIZE_RESULTS="$CALIB_EXP/sizing"
  mkdir -p "$SIZE_RESULTS/cells"
  ln -sfn "$CALIB_RESULTS/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_RESULTS/crops" "$SIZE_RESULTS/crops"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="bench-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time=2:00:00 --partition="$PARTITION" --export=ALL \
      --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$SIZE_RESULTS && cd $HERE && time python run_cells.py --index $idx"
  done
  ;;

cells)
  N=$(cd "$HERE" && python run_cells.py --print-cells 2>/dev/null | tail -1)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
  fi
  echo "cells: $N (array 0-$((N-1))%$CONC on $PARTITION)"
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  submit cells --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/cells-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python run_cells.py"
  ;;

status)
  echo "=== queue ==="
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R"
  echo "=== cells written ==="
  ls "$CALIB_RESULTS/cells" 2>/dev/null | grep -c 'task_.*\.csv$' || echo 0
  echo "=== zero-byte cells (resume would SKIP these) ==="
  find "$CALIB_RESULTS/cells" -name 'task_*.csv' -size 0 2>/dev/null | wc -l
  ;;

*)
  echo "usage: $0 {prepare|size|cells|status}" >&2; exit 1
  ;;
esac
