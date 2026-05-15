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
#   bash scripts/install-gpu.sh              # defaults to cu118
#   bash scripts/install-gpu.sh cu121        # for CUDA 12.1
#   bash scripts/install-gpu.sh cu124        # for CUDA 12.4

CUDA_TAG="${1:-cu118}"
EXTRA_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing VTSearch GPU dependencies (CUDA tag: ${CUDA_TAG})..."

# Step 1: Pre-install packages that the PyTorch index may serve as source
# tarballs.  --only-binary ensures we get pre-built wheels from PyPI.
echo "Pre-installing binary-only wheels (numpy, scipy)..."
pip install --only-binary :all: \
  --index-url https://pypi.org/simple \
  "numpy" \
  "scipy" \
  -q

# Step 2: Regenerate requirements/plugins.txt from discovered plugin files.
bash "$SCRIPT_DIR/install-plugin-deps.sh" --dry-run

# Step 3: Install all dependencies (core + plugins) with GPU PyTorch.
echo "Installing all dependencies..."
pip install --extra-index-url "$EXTRA_INDEX" \
  --prefer-binary \
  -r "$REPO_ROOT/requirements/gpu.txt" \
  -q

# Step 4: Editable install so 'import vtsearch' works.
pip install --no-deps -e "$REPO_ROOT" -q

echo "GPU dependencies installed successfully."
