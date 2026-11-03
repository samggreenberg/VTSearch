#!/bin/bash
#SBATCH --partition=cpu
# #5341: per-face cost of the imagekit path, buffalo_l / buffalo_m / buffalo_s, two repeats, same node, as #5330
# (buffalo_m_cost_5330.py). buffalo_s runs on the GRID default: IMAGEKIT_FACE_CONFIG unset, the pile's models dir.
#SBATCH --nodelist=rack7n11
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:30:00
set -euo pipefail
cd /expscratch/sgreenberg/worktrees/vts-bs-run
source scripts/grid/gridenv.sh >/dev/null 2>&1
source scripts/experiments/pile/pile_env.sh
unset IMAGEKIT_FACE_CONFIG IMAGEKIT_FACE_PROVIDER
export VTSEARCH_DEVICE=cpu VTSEARCH_TORCH_THREADS=8 OMP_NUM_THREADS=8
hostname; grep -m1 'model name' /proc/cpuinfo
for rep in 1 2; do for size in 1024 640; do for v in l m s; do
  if [[ $v == s ]]; then cfg=(); else cfg=(IMAGEKIT_FACE_CONFIG=/expscratch/sgreenberg/imagekit-models/buffalo_$v.json); fi
  env "${cfg[@]}" python scripts/experiments/fhibe/buffalo_m_cost_5330.py 200 $size buffalo_$v 2>&1 | grep -v onnxruntime | tail -1
done; done; done
