#!/bin/bash
set -euo pipefail

# GPU dependency installer for VTSearch.
#
# The PyTorch extra-index (download.pytorch.org/whl/cu*) sometimes serves
# source tarballs for packages like numpy and scipy.  This script
# pre-installs them as binary-only wheels before processing the full
# requirements, avoiding the need for a C++ compiler.
#
# Usage:
#   bash install-gpu.sh              # defaults to cu118
#   bash install-gpu.sh cu121        # for CUDA 12.1
#   bash install-gpu.sh cu124        # for CUDA 12.4

CUDA_TAG="${1:-cu118}"
EXTRA_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

echo "Installing VTSearch GPU dependencies (CUDA tag: ${CUDA_TAG})..."

# Step 1: Pre-install packages that the PyTorch index may serve as source
# tarballs.  --only-binary ensures we get pre-built wheels from PyPI.
echo "Pre-installing binary-only wheels (numpy, scipy)..."
pip install --only-binary :all: \
  --index-url https://pypi.org/simple \
  "numpy" \
  "scipy" \
  -q

# Step 2: Install the project with GPU extras.  numpy/scipy are already
# satisfied from step 1, so pip will skip them.
echo "Installing remaining GPU dependencies..."
pip install --extra-index-url "$EXTRA_INDEX" \
  --prefer-binary \
  -e ".[gpu,dev]" \
  -q

# Step 3: Install plugin-specific dependencies (media types, importers, etc.)
echo "Installing plugin dependencies..."
bash "$(dirname "$0")/install-plugin-deps.sh"

echo "GPU dependencies installed successfully."
