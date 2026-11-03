#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1:00:00
# #5332: after the app venv's switch to onnxruntime-gpu 1.26.0: image2face from the venv on GPU and CPU, against the 1.30 CPU vectors.
set -uo pipefail
W=/expscratch/sgreenberg/fhibe/ort-gpu-5332
cd /expscratch/sgreenberg/worktrees/vts-5332
source scripts/grid/gridenv.sh >/dev/null 2>&1
source scripts/experiments/pile/pile_env.sh
export IMAGEKIT_FACE_PROVIDER=oss IMAGEKIT_FACE_CONFIG=/expscratch/sgreenberg/imagekit-models/grid-default/imagekit/face.json
export VTSEARCH_TORCH_THREADS=8 OMP_NUM_THREADS=8
S=scripts/experiments/fhibe/ort_gpu_5332.py
F='pthread|VerifyOutput'
hostname; nvidia-smi --query-gpu=name --format=csv,noheader; grep -m1 'model name' /proc/cpuinfo; echo "venv $VIRTUAL_ENV"
for size in 1024 640; do
  for rep in 1 2; do
    echo "== venv gpu, $size, rep $rep"
    VTSEARCH_DEVICE=cuda python $S run 200 $size venv-gpu $W/venv-gpu-$size.npz 2>&1 | grep -v -E "$F" | grep -E "^ort|^sessions|ms/face|Error|error" | sort -u
  done
  echo "== venv cpu, $size"
  VTSEARCH_DEVICE=cpu python $S run 200 $size venv-cpu $W/venv-cpu-$size.npz 2>&1 | grep -v -E "$F" | grep -E "^ort|^sessions|ms/face|Error|error" | sort -u
  for pair in "venv-gpu cpu" "venv-cpu cpu"; do set -- $pair
    echo "== compare $1 vs $2 (1.30 CPU), $size"; python $S compare $W/$1-$size.npz $W/$2-$size.npz
  done
done
