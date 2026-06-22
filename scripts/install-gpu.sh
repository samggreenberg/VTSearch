#!/bin/bash
set -euo pipefail

# GPU dependency installer for VTSearch.
#
# The PyTorch extra-index (download.pytorch.org/whl/cu*) sometimes serves
# source tarballs for packages like numpy and scipy.  This script
# pre-installs them as binary-only wheels before processing the full
# requirements, avoiding the need for a C++ compiler.
#
# The CUDA tag selects which prebuilt torch wheel you get, and each wheel
# only ships kernel images for a fixed set of GPU architectures. Pick a tag
# whose wheel covers your GPU's compute capability, or torch will import fine
# and then raise `cudaErrorNoKernelImageForDevice` on the first real op.
#
# There is a floor AND a ceiling. Newer GPUs need newer tags: Ampere/Ada work
# on cu118+, Hopper (H100) on cu121+, Blackwell (B100/B200, RTX 50xx) on
# cu128+. But the newest wheels also DROP the oldest architectures, so "just
# use the latest tag" is wrong for old hardware: cu128 dropped Volta (sm_70),
# so a Tesla V100 needs cu124 (or cu121/cu118), NOT cu128. Rule of thumb: use
# the oldest tag your driver supports that still covers your GPU. cu124 is a
# safe default that spans Volta through Hopper.
#
# (VTSearch also smoke-tests CUDA at runtime and falls back to CPU if the
# installed wheel can't run on the GPU, so a mismatch degrades instead of
# crashing - but you only get GPU acceleration with a matching wheel.)
#
# Usage:
#   bash scripts/install-gpu.sh              # defaults to cu124 (spans Volta..Hopper)
#   bash scripts/install-gpu.sh cu118        # for CUDA 11.8 (older drivers)
#   bash scripts/install-gpu.sh cu121        # for CUDA 12.1
#   bash scripts/install-gpu.sh cu124        # for CUDA 12.4 (V100/Volta, A100, H100)
#   bash scripts/install-gpu.sh cu128        # for CUDA 12.8 (Blackwell; drops Volta)

CUDA_TAG="${1:-cu124}"
EXTRA_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Shared progress-printing helpers.
# shellcheck source=_progress.sh
source "$SCRIPT_DIR/_progress.sh"

vts_progress_init 4 "Installing VTSearch GPU dependencies (CUDA tag: ${CUDA_TAG})"

vts_progress_step "Checking Python version (>= 3.10)"
source "$SCRIPT_DIR/_check-python.sh"

vts_progress_step "Upgrading pip / setuptools / wheel"
pip install --upgrade pip "setuptools<82" wheel --progress-bar on

vts_progress_step "Pre-installing binary-only wheels (numpy, scipy) from PyPI"
pip install --only-binary :all: \
  --index-url https://pypi.org/simple \
  "numpy" \
  "scipy" \
  --progress-bar on

vts_progress_step "Installing all dependencies via ${EXTRA_INDEX} (this may take several minutes)"
pip install --extra-index-url "$EXTRA_INDEX" \
  --prefer-binary \
  -r "$REPO_ROOT/requirements/gpu.txt" \
  --progress-bar on

vts_progress_done "GPU dependencies installed successfully"
