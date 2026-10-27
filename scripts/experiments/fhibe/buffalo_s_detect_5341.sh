#!/bin/bash
#SBATCH --partition=cpu
# #5341: det_500m on every one-person FHIBE photo at 1024/640, 16 shards (buffalo_s_detect_5341.py).
#SBATCH --cpus-per-task=7
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-15
set -euo pipefail
W=/expscratch/sgreenberg/fhibe/buffalo_s-5341 WT=/expscratch/sgreenberg/worktrees/vts-bs-run
export LD_LIBRARY_PATH=/cluster/apps/python/3.12.3/lib
mkdir -p $W/detect
$VTS_VENV/bin/python $WT/scripts/experiments/fhibe/buffalo_s_detect_5341.py $WT $SLURM_ARRAY_TASK_ID 16 $W/detect/shard-s-$SLURM_ARRAY_TASK_ID.csv 7 2>&1 | grep -v onnxruntime
