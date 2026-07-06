#!/bin/bash -l
# Build the DE-ViT conda env (github.com/mlzxy/devit), SEPARATE from the VTSearch venv.
#
# DE-ViT pins torch==1.13.1 / torchvision==0.14.1 (CUDA 11.7), which supports up to
# Ampere (sm_86) -> runs on V100/A100, NOT L40S/H100/H200 (Ada/Hopper).
#
# RUN ON A COMPUTE NODE (it compiles the vendored detectron2 CUDA ops; building on
# login is disallowed). Get an interactive V100 first, then run this:
#
#   srun --partition=gpu --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash -l
#   bash /exp/mlucio/projects/VTSearch/scripts/vg/devit_env_setup.sh
#
# If the detectron2 build fails on nvcc/CUDA mismatch, the usual fix is to install a
# matching toolkit into the env (conda install -c conda-forge cudatoolkit-dev=11.7)
# or `module load` an 11.7/11.8 cuda toolkit before `pip install -e`.
set -euo pipefail

DEVIT=${DEVIT:-/exp/$USER/projects/devit}
[ -d "$DEVIT" ] || git clone https://github.com/mlzxy/devit.git "$DEVIT"

module load anaconda3
# The cluster only ships CUDA 12.x/13.x modules, but torch here is cu117. detectron2's
# CUDA ops must be compiled with a matching 11.7 nvcc, so DON'T load a system cuda module;
# instead install cudatoolkit-dev=11.7 INTO the env (below) and point CUDA_HOME at it.

export CONDA_ENVS_PATH=/exp/$USER/.conda/envs
export CONDA_PKGS_DIRS=/exp/$USER/.conda/pkgs
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE "/devit$" || conda create -y -n devit python=3.9
conda activate devit

# Matching 11.7 nvcc for the detectron2 CUDA-op build (no 11.x module exists on the cluster).
conda install -y -c conda-forge cudatoolkit-dev=11.7
export CUDA_HOME=$CONDA_PREFIX
echo "nvcc: $(command -v nvcc || echo MISSING)"; nvcc --version | grep -i release || true

# Torch FIRST with the cu117 wheel (their pinned 1.13.1); then the rest of requirements
# WITHOUT torch/torchvision so pip can't clobber the CUDA build with a CPU wheel.
pip install torch==1.13.1+cu117 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
grep -viE '^(torch|torchvision)\b' "$DEVIT/requirements.txt" > /tmp/devit_reqs.txt
pip install -r /tmp/devit_reqs.txt

# Build DE-ViT's vendored (modified) detectron2 — compiles CUDA ops.
#   FORCE_CUDA=1         : build the CUDA ops even if no GPU is visible at build time
#   --no-build-isolation : its setup.py imports torch, so it must see the env's torch
FORCE_CUDA=1 pip install -e "$DEVIT" --no-build-isolation

echo "=== smoke test (expect: torch 1.13.1+cu117  cuda True  detectron2 0.6  _C OK) ==="
python -c "import torch, detectron2; from detectron2 import _C; print('torch', torch.__version__, ' cuda', torch.cuda.is_available(), ' detectron2', detectron2.__version__, ' _C OK')"
