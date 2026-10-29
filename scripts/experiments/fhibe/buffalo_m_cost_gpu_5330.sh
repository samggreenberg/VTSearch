#!/bin/bash
#SBATCH --partition=gpu

#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
# #5330: per-face cost of the imagekit path on a GPU (sbatch --gres=gpu:v100:1; ORT_GPU_DIR=onnxruntime-gpu 1.22 for a V100), buffalo_l vs buffalo_m, two repeats.
set -uo pipefail
W=/expscratch/sgreenberg/fhibe/ik-buffalo_m
cd /expscratch/sgreenberg/worktrees/vts-5330-run
source scripts/grid/gridenv.sh >/dev/null 2>&1
source scripts/experiments/pile/pile_env.sh
export IMAGEKIT_FACE_PROVIDER=oss VTSEARCH_DEVICE=cuda VTSEARCH_TORCH_THREADS=8 OMP_NUM_THREADS=8
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; grep -m1 'model name' /proc/cpuinfo
for rep in 1 2; do for size in 1024 640; do for v in l m; do
  IMAGEKIT_FACE_CONFIG=/expscratch/sgreenberg/imagekit-models/buffalo_$v.json python scripts/experiments/fhibe/buffalo_m_cost_gpu_5330.py 200 $size buffalo_$v 2>&1 | grep -v -E 'pthread|VerifyOutput' | grep -E "^ort|^sessions|ms/face|Error|error" | sort -u
done; done; done
