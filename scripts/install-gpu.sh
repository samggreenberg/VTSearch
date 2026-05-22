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

# Shared progress-printing helpers.
# shellcheck source=_progress.sh
source "$SCRIPT_DIR/_progress.sh"

vts_progress_init 4 "Installing VTSearch GPU dependencies (CUDA tag: ${CUDA_TAG})"

vts_progress_step "Checking Python version (>= 3.10)"
source "$SCRIPT_DIR/_check-python.sh"

vts_progress_step "Upgrading pip / setuptools / wheel"
pip install --upgrade pip setuptools wheel --progress-bar on

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
