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

# --- Stage 0: prepare (embeds siglip_l; reuses the rest). GPU for the ViT-L. ---
P=$(sbatch --parsable --job-name=cal-prep --gres="$GRES" --mem="$MEM" --cpus-per-task="$CPUS" \
  --time="${CALIB_PREP_TIME:-3:00:00}" --partition=gpu --export=ALL \
  --output="$LOGS/prepare-%j.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && python prepare_data.py")
echo "prepare job: $P"

# --- Stage 1: cells array (sized after prepare writes prepare_info.json). ---
# A tiny launcher job computes the array size then submits the array + analyze,
# because the count isn't known until prepare finishes.
S=$(sbatch --parsable --dependency=afterok:$P --job-name=cal-launch --mem=4G --cpus-per-task=1 \
  --time=0:20:00 --partition=cpu --export=ALL --output="$LOGS/launch-%j.out" \
  --wrap="source $WT/gridenv.sh && $ENVX && cd $HERE && bash launch_cells.sh")
echo "cells launcher job (after prepare): $S"
echo "Report -> $CALIB_RESULTS/REPORT.md   Logs -> $LOGS"
