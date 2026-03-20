#!/bin/bash
set -euo pipefail

# GPU dependency installer for VTSearch.
#
# The PyTorch extra-index (download.pytorch.org/whl/cu*) sometimes serves
# source tarballs for packages like numpy and scipy.  Building those from
# source requires a C++ compiler (g++) that isn't always present on GPU
# machines.  This script works around the issue by pre-installing
# compilation-heavy packages as binary-only wheels from PyPI before
# processing the full requirements file.
#
# Usage:
#   bash install-gpu.sh              # defaults to cu118
#   bash install-gpu.sh cu121        # for CUDA 12.1
#   bash install-gpu.sh cu124        # for CUDA 12.4

CUDA_TAG="${1:-cu118}"
EXTRA_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "Installing VTSearch GPU dependencies (CUDA tag: ${CUDA_TAG})..."

# Step 1: Pre-install packages that the PyTorch index may serve as source
# tarballs.  Using --only-binary ensures we get pre-built wheels from PyPI,
# avoiding the need for a C++ compiler.
#
# Notes:
# - We use --only-binary :all: (not comma-separated package names) for
#   compatibility with pip >= 25, which deprecated the comma syntax.
# - We explicitly set --index-url to PyPI so that any extra-index-url
#   configured elsewhere (pip.conf, env vars, etc.) doesn't interfere
#   with finding the correct binary wheels.
echo "Pre-installing binary-only wheels (numpy, scipy)..."
pip install --only-binary :all: \
  --index-url https://pypi.org/simple \
  "numpy==1.26.4" \
  "scipy" \
  -q

# Step 2: Install the full GPU requirements.  numpy/scipy are already
# satisfied from step 1, so pip will skip them even if the extra index
# offers a source dist.
echo "Installing remaining GPU dependencies..."
pip install --extra-index-url "$EXTRA_INDEX" \
  --prefer-binary \
  -r requirements-gpu.txt \
  -q

echo "GPU dependencies installed successfully."
