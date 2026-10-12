#!/usr/bin/env bash
#SBATCH --job-name=fhibe-detect
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=detect-%A_%a.log
# MTCNN pass of the #4699 supply census: one shard per array task.
# usage, from this directory, after `fhibe_supply.py annotations`:
#   FHIBE_DATA=<dir with filepaths.csv> FHIBE_CENSUS=<census dir> \
#     sbatch --array=0-39%4 --output=<census dir>/logs/detect-%A_%a.log census_detect.sbatch 40
# (2026-10-09: 40 shards on 4 V100s took 78 min; ~2.3 s/image for all three resolutions.)
set -euo pipefail
umask 077
N=${1:?shard count}
W=$(git -C "$SLURM_SUBMIT_DIR" rev-parse --show-toplevel)
R=${FHIBE_DATA:?set to the dir holding filepaths.csv}
O=${FHIBE_CENSUS:?census output dir (beside the release)}
export LD_LIBRARY_PATH=/cluster/apps/python/3.12.3/lib OPENBLAS_CORETYPE=Haswell PYTHONPATH=$W
export HF_HOME=${TMPDIR:-/tmp}/hf VTSEARCH_DATA_DIR=${TMPDIR:-/tmp}/vts-data   # never the live app's data
echo "$(date '+%F %T') shard $SLURM_ARRAY_TASK_ID/$N on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) at $(git -C $W rev-parse --short HEAD)"
cd $W
${VTS_PYTHON:-/exp/$USER/projects/VTSearch/.venv/bin/python} scripts/experiments/pile/fhibe_supply.py detect \
  --root $R --out $O --shard "$SLURM_ARRAY_TASK_ID/$N" --long-side 1024 --long-side 640
