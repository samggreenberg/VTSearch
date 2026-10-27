#!/bin/bash
#SBATCH --partition=cpu
# #5330: per-face cost of the imagekit path, buffalo_l vs buffalo_m, two repeats, same node (buffalo_m_cost_5330.py).
#SBATCH --nodelist=rack7n11
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
set -euo pipefail
W=/expscratch/sgreenberg/fhibe/ik-buffalo_m
cd /expscratch/sgreenberg/worktrees/vts-5330-run
source gridenv.sh >/dev/null 2>&1
source scripts/experiments/pile/pile_env.sh
export IMAGEKIT_FACE_PROVIDER=oss VTSEARCH_DEVICE=cpu VTSEARCH_TORCH_THREADS=8 OMP_NUM_THREADS=8
hostname; grep -m1 'model name' /proc/cpuinfo
for rep in 1 2; do for size in 1024 640; do for v in l m; do
  IMAGEKIT_FACE_CONFIG=/expscratch/sgreenberg/imagekit-models/buffalo_$v.json python scripts/experiments/fhibe/buffalo_m_cost_5330.py 200 $size buffalo_$v 2>&1 | grep -v onnxruntime | tail -1
done; done; done
