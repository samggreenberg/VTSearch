#!/bin/bash
#SBATCH --job-name=fsd50k-fix
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/exp/scale26/datasets/external/vtsearch-demos/umap-sweep/fsd50k-fix-%j.out
set -e
source /exp/sgreenberg/umap_env.sh
cd /exp/sgreenberg/projects/VTSearch
PY=.venv/bin/python
H=scripts/experiments/umap_params
R=/exp/scale26/datasets/external/vtsearch-demos/umap-sweep
echo "[$(date)] $(hostname) — re-embedding FSD50K (4000 clips) + re-sweep"
rm -f $R/matrices/fsd50k_eval__clap.npz $R/rows/fsd50k_eval__clap.csv
$PY $H/prepare_fsd50k.py 4000
$PY $H/sweep.py fsd50k_eval__clap
$PY $H/summarize.py
echo "[$(date)] FSD50K fix + summarize DONE"
