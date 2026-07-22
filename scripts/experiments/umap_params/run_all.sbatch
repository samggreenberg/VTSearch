#!/bin/bash
#SBATCH --job-name=umap-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=10:00:00
#SBATCH --output=/exp/scale26/datasets/external/vtsearch-demos/umap-sweep/run_all-%j.out
set -e
source /exp/sgreenberg/umap_env.sh
cd /exp/sgreenberg/projects/VTSearch
PY=.venv/bin/python
H=scripts/experiments/umap_params
echo "[$(date)] node $(hostname)"; nvidia-smi -L

echo "===== EMBED (idempotent) ====="
$PY $H/prepare_dataset.py places365_s places365_m places365_l caltech256_m inat_val
$PY $H/prepare_dataset.py fsd50k_eval

echo "===== SWEEP ====="
$PY $H/sweep.py --all

echo "===== SUMMARIZE ====="
$PY $H/summarize.py
echo "[$(date)] ALL DONE"
