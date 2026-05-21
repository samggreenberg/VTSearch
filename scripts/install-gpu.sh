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

# Bail out early if the active Python is too old.  pyproject.toml requires
# >=3.10, but pip only enforces it partway through resolution, after a slow
# and confusing failure path.  Check up front instead.
source "$SCRIPT_DIR/_check-python.sh"

# Step 0: Ensure pip/setuptools/wheel are recent enough to resolve modern
# wheel-only packages.  Stale pips (e.g. 21.x) have a weaker resolver and
# will fall back to building ancient sdists from source, which then fails
# for lack of `bdist_wheel`.
echo "Upgrading pip / setuptools / wheel..."
pip install --upgrade pip setuptools wheel -q

# Step 1: Pre-install packages that the PyTorch index may serve as source
# tarballs.  --only-binary ensures we get pre-built wheels from PyPI.
echo "Pre-installing binary-only wheels (numpy, scipy)..."
pip install --only-binary :all: \
  --index-url https://pypi.org/simple \
  "numpy" \
  "scipy" \
  -q

# Step 2: Install runtime + dev deps and the vtsearch package (editable)
# via pyproject.toml. requirements/gpu.txt is just `-e .[dev]`.
echo "Installing all dependencies..."
pip install --extra-index-url "$EXTRA_INDEX" \
  --prefer-binary \
  -r "$REPO_ROOT/requirements/gpu.txt" \
  -q

echo "GPU dependencies installed successfully."
