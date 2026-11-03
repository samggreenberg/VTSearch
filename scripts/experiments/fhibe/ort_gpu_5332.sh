#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:30:00
# #5332: buffalo_s image2face on a V100 (onnxruntime-gpu 1.22 from ORT_GPU_DIR) vs the venv's CPU build, same node.
set -uo pipefail
W=/expscratch/sgreenberg/fhibe/ort-gpu-5332
G=/expscratch/sgreenberg/ort-gpu-5330-122
cd /expscratch/sgreenberg/worktrees/vts-5332
source scripts/grid/gridenv.sh >/dev/null 2>&1
source scripts/experiments/pile/pile_env.sh
export IMAGEKIT_FACE_PROVIDER=oss IMAGEKIT_FACE_CONFIG=/expscratch/sgreenberg/imagekit-models/grid-default/imagekit/face.json
export VTSEARCH_TORCH_THREADS=8 OMP_NUM_THREADS=8
S=scripts/experiments/fhibe/ort_gpu_5332.py
F='pthread|VerifyOutput'
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; grep -m1 'model name' /proc/cpuinfo
for size in 1024 640; do
  echo "== gpu, no preload, $size"
  ORT_GPU_DIR=$G VTSEARCH_DEVICE=cuda python $S run 200 $size gpu-nopreload $W/gpu-nopreload-$size.npz 2>&1 | grep -v -E "$F" | grep -E "^ort|^sessions|ms/face|Error|error" | sort -u
  for rep in 1 2; do
    echo "== gpu, preload, $size, rep $rep"
    ORT_GPU_DIR=$G ORT_PRELOAD=1 VTSEARCH_DEVICE=cuda python $S run 200 $size gpu $W/gpu-$size.npz 2>&1 | grep -v -E "$F" | grep -E "^ort|^sessions|ms/face|Error|error" | sort -u
  done
  echo "== cpu (venv ort), $size"
  VTSEARCH_DEVICE=cpu python $S run 200 $size cpu $W/cpu-$size.npz 2>&1 | grep -v -E "$F" | grep -E "^ort|^sessions|ms/face|Error|error" | sort -u
  echo "== cpu on the gpu build, $size"
  ORT_GPU_DIR=$G VTSEARCH_DEVICE=cpu python $S run 200 $size cpu-gpubuild $W/cpu-gpubuild-$size.npz 2>&1 | grep -v -E "$F" | grep -E "^ort|^sessions|ms/face|Error|error" | sort -u
  for pair in "gpu cpu" "gpu-nopreload gpu" "cpu-gpubuild cpu"; do set -- $pair
    echo "== compare $1 vs $2, $size"; python $S compare $W/$1-$size.npz $W/$2-$size.npz
  done
done
