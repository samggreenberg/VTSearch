#!/usr/bin/env bash
# FHIBE face benchmark (#4699): one identity per cell, the session opening on K
# photos of the person, scored on the withheld half.
#
#   bash launch.sh identities          # sample identities into $CALIB_EXP/identities.txt (one CPU job)
#   bash launch.sh prepare             # select cells from the identity file (one CPU job)
#   bash launch.sh size 0,<N>          # time a photo cell and a face cell before sizing the array
#   bash launch.sh cells               # the array: every identity x dataset x seed
#   bash launch.sh pack                # the same cells packed into one job on a V100 node (cpu cap full)
#   bash launch.sh redo <idx,...>
#   bash launch.sh status
#
# The arms (FHIBE_DATASETS, all four by default) pair cell for cell:
#   fhibe_1024 / fhibe_640              the one-person photo, SigLIP, whole image
#   fhibe_faces_1024 / fhibe_faces_640  image2face crops of those copies, FaceNet
# A subject crop carries its photo's id, and identities.py admits only people
# every arm can run, so a photo cell and a face cell differ by the representation.
#
# What is pinned, and why:
#   * the EXAMPLE opening (CALIB_SEED_EXAMPLES=$FHIBE_EXAMPLES): FHIBE has no names
#     to type, and the app's no-text flow is the example sort. The K photos are
#     the run's first votes (phase "example"); clicks after them are t - K.
#   * the STRATIFIED split: an identity's ~6 photos are split on their own, so
#     every admitted identity has K in the voting half and one withheld. The
#     negatives keep the random 50/50.
#   * every identity in the file (CALIB_CATEGORY_MODE=all + CALIB_CATEGORY_FILE).
#     The default minimum of 20 photos per category would drop them all.
#   * shipped defaults for everything else, the full-label ceiling on, and the
#     rank frames and pick log the State of the App reads.
#
# #4731's two rules are the app's (the Good phase also ends after 16 misses in a
# row with a Good in hand; a Good and 16 Bads get the trained head).
# CALIB_GOOD_DRY_RUN / CALIB_QUOTA_DRY_BADS set them to another count, or to
# `off` for the app before #4731. Set either and preflight declares it.
# CALIB_CENTROID_LINE draws the Goods' centroid's line by another rule (#4732);
# CALIB_CENTROID_LINE_VARIANTS="<rule>@<beta>,..." adds tagged rows pricing other
# lines on the same sessions.
# FHIBE_STRATA=2-4,5-6,7- makes `identities` draw FHIBE_N_IDENTITIES from each
# band of photo counts and write <identities>.strata.tsv beside the file.
#
# One run directory per (date, K, balance), owner-only under the release's
# derived directory: FHIBE may not be redistributed, prepare stores image
# vectors as exemplar crops, and everything derived goes when a consent
# re-release replaces this one. An example-count sweep is one run per K, all on
# ONE identity file sampled at the largest K (FHIBE_MAX_EXAMPLES), so they pair.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="$(cd "$HERE/../../.." && pwd)"
CALIB="$WT/scripts/experiments/calibration"

source "$WT/gridenv.sh"
source "$WT/scripts/experiments/pile/pile_env.sh"
# The pile's datadir, never the live app's (the bashrc exports that one).
if [[ "$VTSEARCH_DATA_DIR" == "/expscratch/$USER/vtsearch-data" ]]; then
  echo "VTSEARCH_DATA_DIR is the live app's data; pile_env.sh should have replaced it" >&2; exit 2
fi
export VTS_REPO="$WT"

export FHIBE_EXAMPLES="${FHIBE_EXAMPLES:-1}"
export FHIBE_MAX_EXAMPLES="${FHIBE_MAX_EXAMPLES:-$FHIBE_EXAMPLES}"
if (( FHIBE_EXAMPLES > FHIBE_MAX_EXAMPLES )); then
  echo "FHIBE_EXAMPLES=$FHIBE_EXAMPLES is above FHIBE_MAX_EXAMPLES=$FHIBE_MAX_EXAMPLES, the count the identities admit" >&2
  exit 2
fi
# The balance, as the State of the App runs it: unset is the app's default.
if [[ -n "${FHIBE_BETA:-}" ]]; then
  export CALIB_BETA="$FHIBE_BETA"
  _BTAG="-b$(python3 -c "import sys; b=float(sys.argv[1]); print(f'{b:g}'.replace('.', '') if b < 1 else f'{b:g}')" "$FHIBE_BETA")"
fi
FHIBE_ROOT="${VTS_FHIBE_ROOT:-/expscratch/$USER/fhibe}"
FHIBE_RELEASE="${VTS_FHIBE_RELEASE:-fhibe-full-resolution-674a7dcf}"
export FHIBE_RUN="${FHIBE_RUN:-$(date +%Y-%m-%d)}"
export CALIB_EXP="${CALIB_EXP:-$FHIBE_ROOT/$FHIBE_RELEASE-derived/runs/$FHIBE_RUN-k$FHIBE_EXAMPLES${_BTAG:-}}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"
case "$CALIB_EXP" in
  "$FHIBE_ROOT"/*) ;;
  *) echo "CALIB_EXP=$CALIB_EXP is outside $FHIBE_ROOT, which is the only owner-only place for FHIBE results" >&2; exit 2 ;;
esac
# The identity file defaults to the run's own, written by `identities`; a K
# sweep points every run at the first one's.
export FHIBE_IDENTITIES="${FHIBE_IDENTITIES:-$CALIB_EXP/identities.txt}"

export CALIB_DATASETS="${FHIBE_DATASETS:-fhibe_1024,fhibe_640,fhibe_faces_1024,fhibe_faces_640}"
export CALIB_FHIBE_EMBEDDERS="${CALIB_FHIBE_EMBEDDERS:-siglip}"
export CALIB_FHIBE_FACE_EMBEDDERS="${CALIB_FHIBE_FACE_EMBEDDERS:-face}"
export CALIB_CATEGORY_MODE=all
export CALIB_CATEGORY_FILE="$FHIBE_IDENTITIES"
export CALIB_SEED_EXAMPLES="$FHIBE_EXAMPLES"
export CALIB_STRATIFY_TARGET=1
export CALIB_REQUIRE_OPENING=example
export CALIB_N_SEEDS="${FHIBE_SEEDS:-1}"
export CALIB_MAX_STEPS="${FHIBE_MAX_STEPS:-150}"
export CALIB_CELL_ORDER=seed
export CALIB_SKYLINE_ARMS="${CALIB_SKYLINE_ARMS-skyline_train_full}"
export CALIB_SAFE_THRESHOLDS=1
export CALIB_EMIT_PICKS=1
export CALIB_RANK_FRAME_STEPS="${CALIB_RANK_FRAME_STEPS:-10,25,50,100,150}"
export CALIB_REPOOL_VARIANTS="" CALIB_SCHEDULE_VARIANTS="" CALIB_FOLD_COUNTS=""

# Sized 2026-10-09 on the smoke run (1 example, 150 clicks, ceiling on, 2 CPUs):
# a 1024 photo cell took 3:04 at 635 MB peak (job 939415), a 1024 face cell 4:56
# at 561 MB (939416). 3G and an hour leave room for a cell whose Good phase
# never ends, which walks the example sort for all 150 clicks.
MEM="${CALIB_MEM:-3G}"
CPUS="${CALIB_CPUS:-2}"
TIME="${CALIB_TIME:-1:00:00}"
PARTITION="${CALIB_PARTITION:-cpu}"
CONC="${CALIB_CONC:-60}"
JOB_NAME="${CALIB_JOB_NAME:-fhibe-k$FHIBE_EXAMPLES${_BTAG:-}}"
LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/cells"
chmod 700 "$CALIB_EXP"

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS"
ENVX="$ENVX VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
ENVX="$ENVX VTS_REPO=$VTS_REPO CALIB_DATASETS=$CALIB_DATASETS"
ENVX="$ENVX CALIB_FHIBE_EMBEDDERS=$CALIB_FHIBE_EMBEDDERS CALIB_FHIBE_FACE_EMBEDDERS=$CALIB_FHIBE_FACE_EMBEDDERS"
ENVX="$ENVX CALIB_CATEGORY_MODE=$CALIB_CATEGORY_MODE CALIB_CATEGORY_FILE=$CALIB_CATEGORY_FILE"
ENVX="$ENVX CALIB_SEED_EXAMPLES=$CALIB_SEED_EXAMPLES CALIB_STRATIFY_TARGET=$CALIB_STRATIFY_TARGET"
ENVX="$ENVX CALIB_REQUIRE_OPENING=$CALIB_REQUIRE_OPENING CALIB_N_SEEDS=$CALIB_N_SEEDS CALIB_MAX_STEPS=$CALIB_MAX_STEPS"
ENVX="$ENVX CALIB_CELL_ORDER=$CALIB_CELL_ORDER CALIB_SKYLINE_ARMS=$CALIB_SKYLINE_ARMS"
ENVX="$ENVX CALIB_SAFE_THRESHOLDS=$CALIB_SAFE_THRESHOLDS CALIB_EMIT_PICKS=$CALIB_EMIT_PICKS"
ENVX="$ENVX CALIB_RANK_FRAME_STEPS=$CALIB_RANK_FRAME_STEPS CALIB_BETA=${CALIB_BETA:-}"
ENVX="$ENVX CALIB_REPOOL_VARIANTS= CALIB_SCHEDULE_VARIANTS= CALIB_FOLD_COUNTS="
ENVX="$ENVX CALIB_GOOD_DRY_RUN=${CALIB_GOOD_DRY_RUN:-} CALIB_QUOTA_DRY_BADS=${CALIB_QUOTA_DRY_BADS:-}"
ENVX="$ENVX CALIB_CENTROID_LINE=${CALIB_CENTROID_LINE:-} CALIB_CENTROID_LINE_VARIANTS=${CALIB_CENTROID_LINE_VARIANTS:-}"

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
identities)
  # Every listed cell, not just this run's datasets: a later run on fewer arms
  # can reuse the file and still pair with the full one.
  echo "identities: admitted at $FHIBE_MAX_EXAMPLES example(s), n=${FHIBE_N_IDENTITIES:-all} -> $FHIBE_IDENTITIES"
  submit identities --job-name=fhibe-ids --mem=8G --cpus-per-task=2 --time=1:00:00 \
    --partition="$PARTITION" --export=ALL --output="$LOGS/identities-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python identities.py --examples $FHIBE_MAX_EXAMPLES \
${FHIBE_N_IDENTITIES:+--n $FHIBE_N_IDENTITIES} ${FHIBE_STRATA:+--strata $FHIBE_STRATA} --out $FHIBE_IDENTITIES \
&& chmod 600 $FHIBE_IDENTITIES ${FHIBE_STRATA:+$FHIBE_IDENTITIES.strata.tsv}"
  ;;

prepare)
  [[ -s "$FHIBE_IDENTITIES" ]] || { echo "no identity file at $FHIBE_IDENTITIES (run identities first)" >&2; exit 3; }
  echo "CALIB_EXP=$CALIB_EXP datasets=$CALIB_DATASETS examples=$FHIBE_EXAMPLES seeds=$CALIB_N_SEEDS" \
    "identities=$(grep -vc '^#' "$FHIBE_IDENTITIES")"
  submit prepare --job-name=fhibe-prep --mem=32G --cpus-per-task=4 --time=2:00:00 \
    --partition="$PARTITION" --export=ALL --output="$LOGS/prepare-%j.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $CALIB && python prepare_data.py"
  ;;

size)
  IDXS="${2:?usage: launch.sh size <comma-separated cell indices>}"
  SIZE_RESULTS="$CALIB_EXP/sizing"
  mkdir -p "$SIZE_RESULTS/cells"
  ln -sfn "$CALIB_RESULTS/prepare_info.json" "$SIZE_RESULTS/prepare_info.json"
  ln -sfn "$CALIB_RESULTS/crops" "$SIZE_RESULTS/crops"
  for idx in ${IDXS//,/ }; do
    submit "size$idx" --job-name="fhibe-size$idx" --mem="$MEM" --cpus-per-task="$CPUS" \
      --time=2:00:00 --partition="$PARTITION" --export=ALL --output="$LOGS/size-$idx-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CALIB_RESULTS=$SIZE_RESULTS && cd $CALIB && time python run_cells.py --index $idx"
  done
  echo "read Elapsed and MaxRSS off 'sacct -j <id> --format=Elapsed,MaxRSS,State' before sizing the array"
  ;;

cells|pack)
  [[ -s "$CALIB_RESULTS/prepare_info.json" ]] || { echo "no prepared grid at $CALIB_RESULTS (run prepare first)" >&2; exit 3; }
  N=$(cd "$CALIB" && python run_cells.py --print-cells 2>/dev/null | tail -1)
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
        "embedders": {"photo": "$CALIB_FHIBE_EMBEDDERS".split(","), "face": "$CALIB_FHIBE_FACE_EMBEDDERS".split(",")},
        "n_seeds": int("$CALIB_N_SEEDS"),
        "max_steps": int("$CALIB_MAX_STEPS"),
        "cell_order": "$CALIB_CELL_ORDER",
        "seed_examples": int("$FHIBE_EXAMPLES"),
        "stratify_target": True,
        "identities": "$FHIBE_IDENTITIES",
        "beta": "${CALIB_BETA:-}",
        "good_dry_run": "${CALIB_GOOD_DRY_RUN:-}",
        "quota_dry_bads": "${CALIB_QUOTA_DRY_BADS:-}",
        "centroid_line": "${CALIB_CENTROID_LINE:-}",
        "centroid_line_variants": "${CALIB_CENTROID_LINE_VARIANTS:-}",
        "job_name": "$JOB_NAME",
    },
    open(sys.argv[1], "w"),
    indent=2,
)
PYSHAPE
  DIVERGES="seed_examples,stratify_target${CALIB_BETA:+,beta}"
  DIVERGES="$DIVERGES${CALIB_GOOD_DRY_RUN:+,good_dry_run}${CALIB_QUOTA_DRY_BADS:+,quota_dry_bads}"
  DIVERGES="$DIVERGES${CALIB_CENTROID_LINE:+,centroid_line}"
  bash "$WT/scripts/experiments/preflight.sh" --exp "$CALIB_EXP" --arms prod \
    --job-name "$JOB_NAME" --mem "$MEM" --conc "$CONC" --diverges "$DIVERGES" || {
    echo "PREFLIGHT FAILED" >&2; exit 2; }
  if [[ "$1" == pack ]]; then
    # The cpu partition's per-user cap is full: every cell in one multi-CPU job on a V100 node,
    # GPU hidden, CALIB_PACK_PAR cells at a time on 2 threads each (as launch_bands.sh pack, #4490).
    PAR="${CALIB_PACK_PAR:-30}"
    THREADS="OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2"
    submit pack --job-name="$JOB_NAME-pack" --partition=gpu --gres="${CALIB_PACK_GRES:-gpu:v100:1}" \
      --mem="$((PAR * 3))G" --cpus-per-task="$((PAR * 2))" --time="${CALIB_PACK_TIME:-8:00:00}" \
      --export=ALL --output="$LOGS/pack-%j.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && export CUDA_VISIBLE_DEVICES= $THREADS && cd $CALIB && \
seq 0 $((N - 1)) | xargs -P $PAR -I{} sh -c 'python run_cells.py --index {} > $LOGS/cell-{}.out 2>&1 || echo FAILED {}'; echo pack done"
  else
    submit cells --job-name="$JOB_NAME" --array="0-$((N-1))%$CONC" \
      --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
      --partition="$PARTITION" --export=ALL --output="$LOGS/cells-%A_%a.out" \
      --wrap="source $WT/gridenv.sh && $ENVX && cd $CALIB && python run_cells.py"
  fi
  ;;

redo)
  # A failed task leaves its previous output in place: delete the stale cell
  # files first, or a re-run mixes two runs' cells.
  IDXS="${2:?usage: launch.sh redo <comma-separated indices>}"
  submit redo --job-name="$JOB_NAME-redo" --array="$IDXS%$CONC" \
    --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" \
    --partition="$PARTITION" --export=ALL --output="$LOGS/redo-%A_%a.out" \
    --wrap="source $WT/gridenv.sh && $ENVX && cd $CALIB && python run_cells.py"
  ;;

status)
  echo "CALIB_EXP=$CALIB_EXP"
  squeue -u "$USER" -o "%.10i %.16j %.9T %.11M %.6D %R" | grep -E "fhibe|JOBID" || true
  echo "cells written: $(ls "$CALIB_RESULTS/cells" 2>/dev/null | grep -c '^task_[0-9]*\.csv' || true)"
  ;;
*)
  echo "usage: launch.sh {identities|prepare|size <idx>|cells|pack|redo <idx-list>|status}" >&2; exit 1 ;;
esac
