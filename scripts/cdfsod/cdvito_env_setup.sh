#!/bin/bash -l
# Build the CD-ViTO conda env (github.com/lovelyqian/CDFSOD-benchmark), SEPARATE from
# the VTSearch venv. CD-ViTO is built ON DE-ViT (learnable instance features +
# instance-reweighting MLP on the frozen DINOv2 backbone), so it inherits DE-ViT's
# stack: torch==1.13.1 / torchvision==0.14.1 (CUDA 11.7) + a vendored (modified)
# detectron2 whose CUDA ops must be compiled.
#
# torch cu117 supports up to Ampere (sm_86) -> runs on V100/A100, NOT L40S/H100/H200
# (Ada/Hopper). This mirrors scripts/vg/devit_env_setup.sh; keep the two in sync.
#
# RUN ON A COMPUTE NODE (it compiles CUDA ops; building on a login node is disallowed).
# You are on a login node now (login1) -- grab an interactive V100 first, then run this:
#
#   srun --partition=gpu --gres=gpu:v100:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash -l
#   bash /exp/mlucio/projects/VTSearch/scripts/cdfsod/cdvito_env_setup.sh
#
# If the detectron2 build fails on an nvcc/CUDA mismatch, the fix is the same as DE-ViT:
# install a matching 11.7 toolkit INTO the env (done below via cudatoolkit-dev=11.7) and
# point CUDA_HOME at it; do NOT `module load` a system 12.x/13.x cuda.
set -euo pipefail

CDFSOD=${CDFSOD:-/exp/$USER/projects/cdfsod}
DEVIT=${DEVIT:-/exp/$USER/projects/devit}   # existing DE-ViT clone (weights + fallback detectron2)
[ -d "$CDFSOD" ] || git clone https://github.com/lovelyqian/CDFSOD-benchmark.git "$CDFSOD"

module load anaconda3
# The cluster only ships CUDA 12.x/13.x modules, but torch here is cu117. detectron2's
# CUDA ops must be compiled with a matching 11.7 nvcc, so DON'T load a system cuda module;
# install cudatoolkit-dev=11.7 INTO the env (below) and point CUDA_HOME at it.

export CONDA_ENVS_PATH=/exp/$USER/.conda/envs
export CONDA_PKGS_DIRS=/exp/$USER/.conda/pkgs
mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE "/cdfsod$" || conda create -y -n cdfsod python=3.9
conda activate cdfsod

# Matching 11.7 nvcc for the detectron2 CUDA-op build (no 11.x module exists on the cluster).
conda install -y -c conda-forge cudatoolkit-dev=11.7
export CUDA_HOME=$CONDA_PREFIX
export FORCE_CUDA=1   # build CUDA ops even if no GPU is visible at build time
echo "nvcc: $(command -v nvcc || echo MISSING)"; nvcc --version | grep -i release || true

# Torch FIRST with the cu117 wheel (DE-ViT's pinned 1.13.1); then the rest of
# requirements WITHOUT torch/torchvision so pip can't clobber the CUDA build with a
# CPU wheel.
pip install torch==1.13.1+cu117 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu117
if [ -f "$CDFSOD/requirements.txt" ]; then
  grep -viE '^(torch|torchvision)\b' "$CDFSOD/requirements.txt" > /tmp/cdfsod_reqs.txt
  pip install -r /tmp/cdfsod_reqs.txt
else
  echo "WARN: no requirements.txt at $CDFSOD -- check the repo's install instructions." >&2
fi

# Editable install of the CD-ViTO package. If the repo vendors its own (modified)
# detectron2 this compiles the CUDA ops (like DE-ViT). --no-build-isolation because
# its setup.py imports torch and must see the env's torch.
pip install -e "$CDFSOD" --no-build-isolation

# Contingency: CD-ViTO may not vendor detectron2 but import DE-ViT's build instead.
# If `import detectron2` fails after the step above, build DE-ViT's detectron2 into
# THIS env (the DE-ViT clone is already set up for it).
if ! python -c "import detectron2" 2>/dev/null; then
  echo "detectron2 not importable after installing $CDFSOD; building DE-ViT's vendored detectron2..."
  if [ -d "$DEVIT" ]; then
    pip install -e "$DEVIT" --no-build-isolation
  else
    echo "ERROR: no DE-ViT clone at $DEVIT to fall back to. Clone github.com/mlzxy/devit there," >&2
    echo "       or inspect $CDFSOD for its detectron2 install instructions." >&2
    exit 1
  fi
fi

echo "=== smoke test (expect: torch 1.13.1+cu117  cuda True  detectron2 <ver>  _C OK) ==="
python -c "import torch, detectron2; from detectron2 import _C; print('torch', torch.__version__, ' cuda', torch.cuda.is_available(), ' detectron2', detectron2.__version__, ' _C OK')"

cat <<NOTE

=== next steps (manual; see scripts/cdfsod/README.md) ===
1. Reuse the ONE reusable DE-ViT weight (background prototypes); everything else
   below is downloaded or already shipped:
     cd $CDFSOD
     mkdir -p weights/initial/background weights/trained/few-shot
     ln -s $DEVIT/weights/initial/background/background_prototypes.vitl14.pth \\
           weights/initial/background/background_prototypes.vitl14.pth
   - Class prototypes: already shipped in $CDFSOD/prototypes_init/ (nothing to do).
   - RPN R-50.pkl: detectron2 auto-downloads it (nothing to do).
2. Download the FEW-SHOT trained model (the OVD vitl_0069999.pth you have is the WRONG
   task) into weights/trained/few-shot/vitl_0089999.pth  (~1.2G):
     DE-ViT Box folder https://rutgers.box.com/s/2lco6ab66pn3ufq6rh4gmyfzg9vfkm23
     -> weights/trained/few-shot/vitl_0089999.pth   (browser download + scp is the
        reliable route; the folder has no clean anonymous CLI path).
3. Download ONE benchmark dataset (ArTaxOr smallest) from the repo README links into
   $CDFSOD/datasets/, then smoke-test on a single GPU (main_results.sh hardcodes 4):
     CUDA_VISIBLE_DEVICES=0 python tools/train_net.py --num-gpus 1 \\
       --config-file configs/artaxor/vitl_shot5_artaxor_finetune.yaml \\
       MODEL.WEIGHTS weights/trained/few-shot/vitl_0089999.pth \\
       DE.OFFLINE_RPN_CONFIG configs/RPN/mask_rcnn_R_50_C4_1x_ovd_FSD.yaml \\
       OUTPUT_DIR output/vitl/artaxor_5shot/
   A non-degenerate AP proves env+weights+build. (ViT-L may OOM a 16G V100 -> try vitb/vits.)
NOTE
