#!/bin/bash
#SBATCH --partition=cpu
# #5330: det_10g and det_2.5g on every one-person FHIBE photo at 1024/640, 16 shards (buffalo_m_detect_5330.py).
#SBATCH --cpus-per-task=7
#SBATCH --mem=8G
#SBATCH --time=3:00:00
#SBATCH --array=0-15
set -euo pipefail
W=/expscratch/sgreenberg/fhibe/ik-buffalo_m
export LD_LIBRARY_PATH=/cluster/apps/python/3.12.3/lib
mkdir -p $W/detect
$VTS_VENV/bin/python /expscratch/sgreenberg/worktrees/vts-5330-run/scripts/experiments/fhibe/buffalo_m_detect_5330.py /expscratch/sgreenberg/worktrees/vts-5330-run $SLURM_ARRAY_TASK_ID 16 $W/detect/shard-$SLURM_ARRAY_TASK_ID.csv 7 2>&1 | grep -v onnxruntime
