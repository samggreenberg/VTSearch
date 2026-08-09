#!/usr/bin/env bash
# Calibration study (#2781) full pipeline on the HLTCOE Grid.
#
# Reuses the Max-Patch prepare pickles + crops for the (dataset, embedder) pairs
# that coincide (visual_genome_m x {siglip, dinov3_patch}, caltech101_m x siglip)
# by pointing VTSEARCH_DATA_DIR at the Max-Patch datadir and symlinking its crops;
# prepare then only embeds the missing siglip_l pairs.  Chains:
#   prepare (GPU) -> cells array (afterok) -> analyze (afterany)
#
# Usage: bash launch_all.sh
set -uo pipefail

WT="${VTS_REPO:-/exp/$USER/projects/vts-calib}"
HERE="$WT/scripts/experiments/calibration"
MAXPATCH="/exp/$USER/max-patch"
export CALIB_EXP="${CALIB_EXP:-/exp/$USER/calibration}"
export CALIB_RESULTS="${CALIB_RESULTS:-$CALIB_EXP/results}"
# Reuse the Max-Patch datadir (demo data + embeddings pickles + models) directly;
# siglip_l pickles land alongside them, harmlessly.
export VTSEARCH_DATA_DIR="${VTSEARCH_DATA_DIR:-$MAXPATCH/datadir}"
export VTSEARCH_MODELS_DIR="${VTSEARCH_MODELS_DIR:-$MAXPATCH/models}"
export HF_HOME="${HF_HOME:-/exp/$USER/.cache/huggingface}"
LOGS="$CALIB_EXP/logs"
mkdir -p "$LOGS" "$CALIB_RESULTS/crops" "$CALIB_RESULTS/cells"

# --- Symlink the reusable Max-Patch crops (embeddings pickles are read in place
# from the shared datadir). ---
for base in visual_genome_m__siglip__crops visual_genome_m__dinov3_patch__crops; do
  for ext in npz json; do
    src="$MAXPATCH/results/crops/$base.$ext"
    dst="$CALIB_RESULTS/crops/$base.$ext"
    [[ -e "$src" && ! -e "$dst" ]] && ln -s "$src" "$dst" && echo "linked $base.$ext"
  done
done

ENVX="export CALIB_EXP=$CALIB_EXP CALIB_RESULTS=$CALIB_RESULTS VTSEARCH_DATA_DIR=$VTSEARCH_DATA_DIR VTSEARCH_MODELS_DIR=$VTSEARCH_MODELS_DIR HF_HOME=$HF_HOME"
GRES="${CALIB_GRES:-gpu:v100:1}"
MEM="${CALIB_MEM:-48G}"
CPUS="${CALIB_CPUS:-6}"
CONC="${CALIB_CONC:-8}"

# A submission is not a launch: an sbatch that is *refused* returns an empty id,
# and every dependent job then fails with "Job dependency problem" - a wall of
# noise whose real cause scrolled past.  Fail loudly at the first one instead.
require_jobid() {
  local id="$1" what="$2"
  if ! [[ "$id" =~ ^[0-9]+$ ]]; then
    echo "ERROR: $what was REFUSED by sbatch (no job id came back)." >&2
    echo "       Nothing downstream can run; fix the submission and re-launch." >&2
    exit 1
  fi
}

# --- Stage 0: prepare (embeds siglip_l; reuses the rest). GPU for the ViT-L. ---
# Honour CALIB_PARTITION / CALIB_GRES the same way launch_cells.sh does.  A run
# whose pickles are all cached needs no GPU, and passing `--gres=none` through
# literally is not a no-op: this cluster's submit filter rewrites it and then
# rejects the job ("--gpus-per-task option requires --tasks specification"), so
# the arm dies before it starts.  Drop the flag instead, and let the caller put
# prepare on the cpu partition.
PREP_PARTITION="${CALIB_PREP_PARTITION:-${CALIB_PARTITION:-gpu}}"
PREP_GRES_ARG=(--gres="$GRES")
[[ "$GRES" == "none" || -z "$GRES" ]] && PREP_GRES_ARG=()

P=$(sbatch --parsable --job-name=cal-prep "${PREP_GRES_ARG[@]}" --mem="$MEM" --cpus-per-task="$CPUS" \
  --time="${CALIB_PREP_TIME:-3:00:00}" --partition="$PREP_PARTITION" --export=ALL \
  --output="$LOGS/prepare-%j.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
echo "prepare job: $P"
require_jobid "$P" "prepare"

# --- Stage 1: cells array (sized after prepare writes prepare_info.json). ---
# A tiny launcher job computes the array size then submits the array + analyze,
# because the count isn't known until prepare finishes.
S=$(sbatch --parsable --dependency=afterok:$P --job-name=cal-launch --mem=4G --cpus-per-task=1 \
  --time=0:20:00 --partition=cpu --export=ALL --output="$LOGS/launch-%j.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && bash launch_cells.sh")
echo "cells launcher job (after prepare): $S"
require_jobid "$S" "cells launcher"
echo "Report -> $CALIB_RESULTS/REPORT.md   Logs -> $LOGS"
