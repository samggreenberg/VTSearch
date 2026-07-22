#!/usr/bin/env bash
# Submit the full MLP-vs-SVM pipeline to SLURM with dependencies.
#
#   prepare (embed once)  ->  stage_a (screen)   \
#                          ->  stage_b (array)    ->  summarize (report)
#                          ->  stage_c (timing)  /
#
# GPU nodes on this cluster are Exclusive_Process, so each job takes its own GPU.
# Sizing knobs are env vars read by experiment_config.py (MLPSVM_N_SEEDS,
# MLPSVM_N_CATEGORIES, MLPSVM_MAX_STEPS, MLPSVM_TRAINERS, MLPSVM_DATASETS).
#
# Usage:  bash queue_all.sh [N_ARRAY_CELLS]
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WT=/exp/sgreenberg/projects/vts-mlpsvm
LOGS=/exp/sgreenberg/mlp-svm/logs
mkdir -p "$LOGS"

GRES="${MLPSVM_GRES:-gpu:a100:1}"
MEM="${MLPSVM_MEM:-48G}"
CPUS="${MLPSVM_CPUS:-8}"
TIME="${MLPSVM_TIME:-8:00:00}"
NCELLS="${1:-288}"   # array size; pass the count printed by --print-cells

runner() {  # $1=jobname $2..=python argv ; prints the submitted job id
  local name="$1"; shift
  sbatch --parsable --job-name="mlpsvm-$name" --gres="$GRES" --mem="$MEM" \
    --cpus-per-task="$CPUS" --time="$TIME" --partition=gpu \
    --output="$LOGS/$name-%A_%a.out" "$@"
}

# Wrap a python stage in an srun-friendly one-liner sourced from gridenv.sh.
STAGE_WRAP="source $WT/gridenv.sh && cd $HERE"

PREP=$(sbatch --parsable --job-name=mlpsvm-prep --gres="$GRES" --mem="$MEM" --cpus-per-task="$CPUS" \
  --time=2:00:00 --partition=gpu --output="$LOGS/prep-%j.out" \
  --wrap="$STAGE_WRAP && python prepare_data.py \${MLPSVM_DATASETS_SPACE:-caltech101_m caltech256_a visual_genome_m}")
echo "prepare: $PREP"

A=$(sbatch --parsable --dependency=afterok:$PREP --job-name=mlpsvm-a --gres="$GRES" --mem="$MEM" \
  --cpus-per-task="$CPUS" --time=3:00:00 --partition=gpu --output="$LOGS/stagea-%j.out" \
  --wrap="$STAGE_WRAP && python stage_a_screen.py")
echo "stage_a: $A"

B=$(sbatch --parsable --dependency=afterok:$PREP --job-name=mlpsvm-b --array=0-$((NCELLS-1))%24 \
  --gres="$GRES" --mem="$MEM" --cpus-per-task="$CPUS" --time="$TIME" --partition=gpu \
  --output="$LOGS/stageb-%A_%a.out" \
  --wrap="$STAGE_WRAP && python stage_b_autopilot.py")
echo "stage_b array: $B"

C=$(sbatch --parsable --dependency=afterok:$PREP --job-name=mlpsvm-c --gres="$GRES" --mem="$MEM" \
  --cpus-per-task="$CPUS" --time=2:00:00 --partition=gpu --output="$LOGS/stagec-%j.out" \
  --wrap="$STAGE_WRAP && python stage_c_timing.py")
echo "stage_c: $C"

S=$(sbatch --parsable --dependency=afterany:$B:$A:$C --job-name=mlpsvm-sum --mem=16G \
  --cpus-per-task=4 --time=0:30:00 --partition=gpu --output="$LOGS/summarize-%j.out" \
  --wrap="$STAGE_WRAP && python summarize.py")
echo "summarize: $S"
echo "Report will land at /exp/sgreenberg/mlp-svm/results/REPORT.md"
