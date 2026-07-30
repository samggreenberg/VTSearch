#!/usr/bin/env bash
# Submit the full Max-Patch pipeline to SLURM with dependencies.
#
#   prepare (embed every dataset x embedder, cache pickles + exemplar crops)
#     -> run_cells (array: one task per dataset x embedder x category x seed)
#       -> summarize (REPORT.md + figures)
#
# GPU nodes on this cluster are Exclusive_Process, so each job takes its own GPU.
# Sizing knobs are env vars read by experiment_config.py (MAXPATCH_N_SEEDS,
# MAXPATCH_N_CATEGORIES, MAXPATCH_MAX_STEPS, MAXPATCH_DATASETS,
# MAXPATCH_EMBEDDERS, MAXPATCH_PATCH_STYLES).
#
# DINOv3 is licence-gated on Hugging Face: export HF_TOKEN before submitting or
# the dinov3_patch arms are skipped by prepare_data.py.
#
# Usage:  bash queue_all.sh [N_ARRAY_CELLS]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT="${VTS_REPO:-/exp/$USER/projects/vts-maxpatch}"
LOGS="/exp/$USER/max-patch/logs"
mkdir -p "$LOGS"

GRES="${MAXPATCH_GRES:-gpu:a100:1}"
MEM="${MAXPATCH_MEM:-64G}"
CPUS="${MAXPATCH_CPUS:-8}"
TIME="${MAXPATCH_TIME:-8:00:00}"
NCELLS="${1:-240}"   # array size; pass the count printed by run_cells.py --print-cells

# Wrap a python stage in an srun-friendly one-liner sourced from gridenv.sh.
STAGE_WRAP="source $WT/gridenv.sh && cd $HERE"

PREP=$(sbatch --parsable --job-name=maxpatch-prep --gres="$GRES" --mem="$MEM" --cpus-per-task="$CPUS" \
  --time=12:00:00 --partition=gpu --output="$LOGS/prep-%j.out" \
  --export=ALL \
  --wrap="$STAGE_WRAP && python prepare_data.py")
echo "prepare: $PREP"

B=$(sbatch --parsable --dependency=afterok:$PREP --job-name=maxpatch-cells --array=0-$((NCELLS-1))%24 \
  --gres="$GRES" --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition=gpu \
  --export=ALL \
  --output="$LOGS/cells-%A_%a.out" \
  --wrap="$STAGE_WRAP && python run_cells.py")
echo "cells array: $B"

S=$(sbatch --parsable --dependency=afterany:$B --job-name=maxpatch-sum --mem=16G \
  --cpus-per-task=4 --time=0:30:00 --partition=gpu --output="$LOGS/summarize-%j.out" \
  --export=ALL \
  --wrap="$STAGE_WRAP && python summarize.py")
echo "summarize: $S"
echo "Report will land at /exp/$USER/max-patch/results/REPORT.md"
