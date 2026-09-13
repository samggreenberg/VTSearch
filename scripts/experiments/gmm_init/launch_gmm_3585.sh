#!/usr/bin/env bash
# Is replacing sklearn's GaussianMixture an OPTIMISATION or a calibration change? (#3585)
#
#   bash launch_gmm_3585.sh list        # index -> cell map
#   bash launch_gmm_3585.sh size 0,20   # time one binary and one region cell
#   bash launch_gmm_3585.sh capture     # the array: real fit inputs -> corpus/*.npz
#   bash launch_gmm_3585.sh sorts       # real cosine/text sort haystacks -> corpus/sorts*.npz
#   bash launch_gmm_3585.sh ab          # the trajectory A/B, one grid per fit arm
#   bash launch_gmm_3585.sh baseline    # the click-0 text-sort anchor the curves need
#   bash launch_gmm_3585.sh abfigures   # quality-over-clicks + the interactive viewer
#   bash launch_gmm_3585.sh gate        # replay the corpus through every arm
#   bash launch_gmm_3585.sh bench       # per-call cost, min-of-k, on real shapes
#   bash launch_gmm_3585.sh status
#
# THE QUESTION.  `fit_score_gmm` is 91-95% of a cosine/text sort and the larger
# half of every calibration fold, and it is sklearn's `GaussianMixture` fitting
# two Gaussians over ONE dimension - the same estimation problem the anchored
# loop next door does ~5x cheaper per iteration.  Replacing it is three hours of
# engineering.  What it is NOT is behaviour-preserving: a different init lands EM
# in a different place inside its tolerance and sometimes in a different BASIN,
# and that propagates through the per-fold quantile, the combine, `np.quantile`
# on the final haystack and `snap_cut_to_sample`.  Most tolerance-level moves die
# in the snap.  "Most" is the problem, so the deliverable is a measurement.
#
# WHY REAL CELLS AND NOT A FIXTURE CORPUS.  A synthetic sweep over (n,
# prevalence, separation) was already run on this issue's comment thread and it
# is not the gate: what decides whether a re-initialised EM lands elsewhere is
# the *shape* of the sample, and the shapes that matter are the ones a trained
# detector makes mid-run - saturated at click 80, barely bimodal at click 5,
# max-pooled and right-skewed under region voting.  So the corpus is captured
# from the shipped path itself (`capture_folds_3585.py`), one array per fit the
# app would perform, and the arms are replayed against it offline.
#
# THE GRID.  Three datasets x two embedders, which is the cheapest set that
# spans the three shapes the fit sees:
#
#   caltech101_m / siglip          SATURATED - the #3166 regime, where the snap
#                                  is load-bearing and a quantile wobble was
#                                  worth 0.026 of threshold
#   visual_genome_m, coco_val      ordinary bimodal, two environments
#   dinov3_patch / max_patch       MAX-POOLED - the heavy right-skewed Bad mode
#                                  region voting produces
#
# Shipped defaults everywhere else: the contrast under test is the FIT.
#
# PREPARE IS REUSED from the 2026-08-12 overview benchmark, whose grid is
# exactly these three datasets x these embedders.  Re-running prepare would
# re-select categories and re-cut exemplar crops for no gain; `preflight.sh
# --reuse-prepare` is what checks the crops still resolve.
set -uo pipefail
trap 'echo "ABORTED: $0 line $LINENO exited $? -- NOTHING WAS SUBMITTED" >&2' ERR

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
CALIB="$WT/scripts/experiments/calibration"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"

export VTS_REPO="$WT"
export CALIB_EXP="${CALIB_EXP:-/expscratch/$USER/gmm-3585}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"
REUSE_PREPARE="${REUSE_PREPARE:-/expscratch/$USER/bench-overview/results}"
CORPUS="${CORPUS:-$CALIB_EXP/corpus}"
ANALYSIS="${ANALYSIS:-$CALIB_EXP/analysis}"

# BLAS pinned at top level so `size`, the array and the bench all measure the
# same thing -- and because the arms differ in exactly how much BLAS they do:
# sklearn's covariance path dispatches to it and the native loop deliberately
# never does. An unpinned comparison would price the thread pool, not the fit.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

export CALIB_DATASETS="${CALIB_DATASETS:-visual_genome_m,caltech101_m,coco_val}"
export CALIB_VG_EMBEDDERS="${CALIB_VG_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_CALTECH_EMBEDDERS="${CALIB_CALTECH_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_COCO_EMBEDDERS="${CALIB_COCO_EMBEDDERS:-siglip,dinov3_patch}"
export CALIB_PATCH_STYLES="${CALIB_PATCH_STYLES:-whole_image,max_patch}"
export CALIB_CATEGORY_MODE=all
export CALIB_N_SEEDS="${CALIB_N_SEEDS:-2}"
export CALIB_MAX_STEPS="${CALIB_MAX_STEPS:-100}"
export CALIB_CELL_ORDER="${CALIB_CELL_ORDER:-seed}"
export CAPTURE_STRIDE="${CAPTURE_STRIDE:-5}"

# The shipped path, NAMED rather than inherited: the whole corpus is the
# fold-anchored estimator's inputs, and without this the run emits none at all.
export CALIB_SAFE_THRESHOLDS=1

# Every side frame off. They cost time and answer questions this study is not
# asking; the frame under test is a directory of .npz captures.
export CALIB_REPOOL_VARIANTS=""
export CALIB_SCHEDULE_VARIANTS=""
export CALIB_FOLD_COUNTS=""
export CALIB_CUT_INCL_KS=""
export CALIB_FIT_QUALITY=0

# Both openings exist on this grid by design: the overview categories carry
# typed queries where COCO/VG have them and fall back to known-goods elsewhere,
# so the declaration is `mixed` rather than a filter that would silently drop
# cells (#3278).
export CALIB_REQUIRE_OPENING="${CALIB_REQUIRE_OPENING:-mixed}"

LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells" "$CORPUS" "$ANALYSIS"

# Reuse the overview benchmark's prepare: same three datasets, same embedders.
if [[ ! -e "$CALIB_RESULTS/prepare_info.json" ]]; then
  [[ -f "$REUSE_PREPARE/prepare_info.json" ]] || {
    echo "REUSE_PREPARE=$REUSE_PREPARE has no prepare_info.json" >&2; exit 3; }
  ln -sfn "$REUSE_PREPARE/prepare_info.json" "$CALIB_RESULTS/prepare_info.json"
  ln -sfn "$REUSE_PREPARE/crops" "$CALIB_RESULTS/crops"
  echo "linked prepare from $REUSE_PREPARE"
fi

MEM="${CALIB_MEM:-12G}"
CPUS="${CALIB_CPUS:-2}"
TIME="${CALIB_TIME:-2:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-40}"
JOB_NAME="${CALIB_JOB_NAME:-gmm-$(basename "$CALIB_EXP")}"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_VG_EMBEDDERS=$CALIB_VG_EMBEDDERS CALIB_CALTECH_EMBEDDERS=$CALIB_CALTECH_EMBEDDERS"
ENVX="$ENVX CALIB_COCO_EMBEDDERS=$CALIB_COCO_EMBEDDERS CALIB_CATEGORY_MODE=$CALIB_CATEGORY_MODE"
ENVX="$ENVX CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_CELL_ORDER=$CALIB_CELL_ORDER CALIB_PATCH_STYLES=$CALIB_PATCH_STYLES"
ENVX="$ENVX CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS CALIB_FIT_QUALITY=0"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS= CALIB_CUT_INCL_KS="
ENVX="$ENVX CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING"
ENVX="$ENVX OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1"

# Chain a stage behind another job GRID-side (DEP=afterany:<id>): a waiter on
# the laptop dies with the VPN, and the gate is a 45-minute job whose input is
# the array that is still running.
DEP="${DEP:-}"
DEP_ARG=()
[[ -n "$DEP" ]] && DEP_ARG=(--dependency="$DEP")

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

n_cells() { ( cd "$CALIB" && python run_cells.py --print-cells 2>/dev/null | tail -1 ); }

case "${1:-status}" in
list)
  ( cd "$CALIB" && python - <<'PYLIST' )
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
  IDXS="${2:?usage: launch_gmm_3585.sh size <comma-separated cell indices>}"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="gmm-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time="$TIME" --partition="$PARTITION" --export=ALL \
      --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && time python capture_folds_3585.py --index $idx --stride $CAPTURE_STRIDE --out $CALIB_EXP/sizing/cell_$(printf '%04d' "$idx").npz"
  done
  echo "when these finish: sacct -j <id> --format=Elapsed,MaxRSS,State"
  ;;

capture)
  N=$(n_cells)
  if ! [[ "$N" =~ ^[0-9]+$ ]] || [[ "$N" -eq 0 ]]; then
    echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1
  fi
  echo "cells: $N (array 0-$((N-1))%$CONC on $PARTITION)"
  python - "$CALIB_RESULTS/grid_shape.json" "$N" <<PYSHAPE
import json, sys

json.dump(
    {
        "n_cells": int(sys.argv[2]),
        "datasets": "$CALIB_DATASETS".split(","),
        "embedders": sorted({*"$CALIB_VG_EMBEDDERS".split(","), *"$CALIB_COCO_EMBEDDERS".split(",")}),
        "styles": "$CALIB_PATCH_STYLES".split(","),
        "n_seeds": int("$CALIB_N_SEEDS"),
        "max_steps": int("$CALIB_MAX_STEPS"),
        "capture_stride": int("$CAPTURE_STRIDE"),
        "cell_order": "$CALIB_CELL_ORDER",
        "reuse_prepare": "$REUSE_PREPARE",
        "job_name": "$JOB_NAME",
    },
    open(sys.argv[1], "w"),
    indent=2,
)
PYSHAPE
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms capture \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" --patch \
    --reuse-prepare "$REUSE_PREPARE" --contrasts-voting-modes || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  submit capture --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL \
    --output="$LOGS/capture-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python capture_folds_3585.py --index \$SLURM_ARRAY_TASK_ID --stride $CAPTURE_STRIDE --out $CORPUS/cell_\$(printf '%04d' \$SLURM_ARRAY_TASK_ID).npz"
  ;;

sorts)
  # Two prepares, because the shapes that matter are at two SIZES: the overview
  # grid's three datasets are 838-4952 medias, and vg_scale's cells are 18,050 -
  # the closest thing in the pile to the ~250k a GUI Find subsamples from.
  submit sorts --job-name="$JOB_NAME-sorts" --mem=32G --cpus-per-task=4 \
    --time=3:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/sorts-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python capture_sorts_3585.py --results $REUSE_PREPARE --out $CORPUS/sorts_overview.npz"
  submit sortscale --job-name="$JOB_NAME-sortscale" --mem=32G --cpus-per-task=4 \
    --time=3:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/sortscale-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python capture_sorts_3585.py --results ${SCALE_PREPARE:-/expscratch/$USER/scale-3679/results} --embedders siglip,clip --max-categories 10 --out $CORPUS/sorts_vgscale.npz"
  ;;

ab)
  # The trajectory A/B the gate cannot do: the threshold feeds Autopilot's Hard
  # pick, so two arms diverge in WHICH ITEMS get voted on and only two whole
  # runs can price that.  One grid per arm, paired by (category, seed) at
  # analysis time by `analyze_ab.py`.
  N=$(n_cells)
  [[ "$N" =~ ^[0-9]+$ ]] || { echo "ERROR: could not determine cell count (got '$N')" >&2; exit 1; }
  for arm in ${AB_ARMS:-native baseline}; do
    AB_RESULTS="$CALIB_EXP/ab_$arm/results"
    mkdir -p "$AB_RESULTS/cells"
    ln -sfn "$REUSE_PREPARE/prepare_info.json" "$AB_RESULTS/prepare_info.json"
    ln -sfn "$REUSE_PREPARE/crops" "$AB_RESULTS/crops"
    submit "ab_$arm" --job-name="$JOB_NAME-ab-$arm" "${DEP_ARG[@]}" --array="0-$((N-1))%$CONC" \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
      --partition="$PARTITION" --export=ALL \
      --output="$LOGS/ab-$arm-%A_%a.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$AB_RESULTS GMM_FIT_ARM=$arm && cd $HERE && python run_cells_arm_3585.py"
  done
  ;;

baseline)
  # Click 0 is the free text sort, and `curves.py` refuses to draw without it.
  submit baseline --job-name="$JOB_NAME-baseline" "${DEP_ARG[@]}" --mem=16G --cpus-per-task=2 \
    --time=2:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/baseline-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $CALIB && python text_baseline.py --results $CALIB_RESULTS --out $ANALYSIS/text_baseline.csv"
  ;;

abfigures)
  # `curves.py` and `viewer.py` want one directory per arm under one root, each
  # holding a `cells/`.  The A/B writes `<exp>/ab_<arm>/results/cells`, so the
  # arm root is a pair of symlinks rather than a copy - the cells are 100s of MB
  # and duplicating them to satisfy a path convention is how a scratch quota
  # goes.
  ARMS_ROOT="$CALIB_EXP/arms"
  mkdir -p "$ARMS_ROOT"
  ln -sfn "$CALIB_EXP/ab_native/results" "$ARMS_ROOT/native"
  ln -sfn "$CALIB_EXP/ab_baseline/results" "$ARMS_ROOT/sklearn"
  BASE_ARG=""
  [[ -f "$ANALYSIS/text_baseline.csv" ]] && BASE_ARG="--baseline $ANALYSIS/text_baseline.csv"
  submit abfigures --job-name="$JOB_NAME-abfigures" "${DEP_ARG[@]}" --mem=32G --cpus-per-task=2 \
    --time=2:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/abfigures-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $CALIB && python curves.py --results $ARMS_ROOT --arms sklearn,native --out $ANALYSIS/figures $BASE_ARG && python viewer.py --results $ARMS_ROOT --arms sklearn=sklearn,native=native --out $ANALYSIS/viewer.html --title 'The mixture fit: sklearn vs ours (#3585)' $BASE_ARG"
  ;;

abanalyze)
  submit abanalyze --job-name="$JOB_NAME-abanalyze" "${DEP_ARG[@]}" --mem=32G --cpus-per-task=2 \
    --time=2:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/abanalyze-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_AB_ON=$CALIB_EXP/ab_native/results CALIB_AB_OFF=$CALIB_EXP/ab_baseline/results CALIB_AB_OUT=$ANALYSIS/ab && mkdir -p $ANALYSIS/ab && cd $CALIB && python analyze_ab.py"
  ;;

gate)
  submit gate --job-name="$JOB_NAME-gate" "${DEP_ARG[@]}" --mem="${GATE_MEM:-32G}" --cpus-per-task=2 \
    --time="${GATE_TIME:-8:00:00}" --partition="$PARTITION" --export=ALL \
    --output="$LOGS/gate-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python gate_3585.py --corpus $CORPUS --out $ANALYSIS"
  ;;

bench)
  submit bench --job-name="$JOB_NAME-bench" "${DEP_ARG[@]}" --mem=16G --cpus-per-task=2 \
    --time=2:00:00 --partition="$PARTITION" --export=ALL \
    --output="$LOGS/bench-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python bench_3585.py --corpus $CORPUS --out $ANALYSIS/bench.csv"
  ;;

status)
  echo "exp:     $CALIB_EXP"
  echo "queue:   $(squeue -u "$USER" -h -n "$JOB_NAME" -o %i | wc -l) tasks named $JOB_NAME"
  echo "cells:   $(find "$CALIB_RESULTS/cells" -name 'task_*.csv' ! -name '*__*' 2>/dev/null | wc -l) main frames"
  echo "corpus:  $(find "$CORPUS" -name 'cell_*.npz' 2>/dev/null | wc -l) captures"
  echo "empty:   $(find "$CORPUS" -name 'cell_*.npz' -size 0 2>/dev/null | wc -l) zero-byte (delete before any resume)"
  echo "sorts:   $(ls -1 "$CORPUS"/sorts*.npz 2>/dev/null | wc -l) sort captures"
  echo "gate:    $(ls -1 "$ANALYSIS"/gate_*.csv 2>/dev/null | wc -l) frames"
  for arm in native baseline; do
    echo "ab/$arm: $(find "$CALIB_EXP/ab_$arm/results/cells" -name 'task_*.csv' ! -name '*__*' 2>/dev/null | wc -l) cells"
  done
  ;;

*)
  echo "usage: $0 {list|size IDXS|capture|sorts|gate|bench|ab|baseline|abfigures|abanalyze|status}" >&2
  exit 1
  ;;
esac
