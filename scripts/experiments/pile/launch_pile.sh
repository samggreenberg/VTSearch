#!/usr/bin/env bash
# Build the missing pile cells on the GRID: prefetch weights (CPU), then one
# GPU job per dataset so the three run concurrently within the 4-GPU tier.
#
#   bash launch_pile.sh              # prefetch + submit all three dataset jobs
#   bash launch_pile.sh coco_val     # just one dataset's job
#
# Weights are prefetched in a separate CPU step because parallel GPU jobs would
# otherwise race to populate the same shared HF cache (see prefetch_models.py).
set -euo pipefail

USER="${USER:-sgreenberg}"
REPO="${VTS_REPO:-/exp/$USER/projects/vts-pile}"
PILE="${VTS_PILE:-/expscratch/$USER/vts-cache}"
HERE="$REPO/scripts/experiments/pile"
LOGS="${PILE}/logs"
GPU_TYPE="${VTS_GPU:-v100}"
# These sweeps peak well under 16G; a fatter request wedges the job off idle
# GPUs whose RAM is already reserved (GRID-PLAYBOOK section 1).
MEM="${VTS_MEM:-24G}"
TIME="${VTS_TIME:-8:00:00}"

mkdir -p "$LOGS"

ENVSET="module load python/3.12.3 && source /exp/$USER/projects/VTSearch/.venv/bin/activate"
ENVSET="$ENVSET && export VTS_REPO=$REPO VTS_PILE=$PILE"
# Keep HF off /exp: one model download there fills the 50G quota.
ENVSET="$ENVSET && export HF_HOME=$PILE/models VTSEARCH_MODELS_DIR=$PILE/models"
ENVSET="$ENVSET && export VTSEARCH_DATA_DIR=$PILE/datadir && cd $HERE"

# --- Stage 1: weights (CPU, blocking) -------------------------------------
echo "=== prefetching weights (CPU) ==="
srun --job-name=pile-prefetch --partition=cpu --cpus-per-task=4 --mem=8G --time=2:00:00 \
  bash -lc "$ENVSET && python prefetch_models.py" 2>&1 | tail -20

# --- Stage 2: one GPU job per dataset -------------------------------------
DATASETS=("${@:-visual_genome_m caltech101_m coco_val}")
read -r -a DATASETS <<< "${DATASETS[@]}"

for ds in "${DATASETS[@]}"; do
  jid=$(sbatch --parsable \
    --job-name="pile-$ds" \
    --partition=gpu \
    --gres="gpu:${GPU_TYPE}:1" \
    --cpus-per-task=8 \
    --mem="$MEM" \
    --time="$TIME" \
    --output="$LOGS/pile-$ds-%j.out" \
    --wrap "bash -lc '$ENVSET && python build_pile.py --datasets $ds'")
  # An empty job id means sbatch silently refused the request -- treat it as a
  # failure rather than reporting a launch that never happened (LESSONS.md).
  if [[ -z "$jid" ]]; then
    echo "FAILED to submit $ds (empty job id)" >&2
    exit 1
  fi
  echo "submitted $ds -> job $jid  (log: $LOGS/pile-$ds-$jid.out)"
done

echo
echo "watch:   squeue -u $USER"
echo "verify:  cd $HERE && python build_pile.py --verify"
