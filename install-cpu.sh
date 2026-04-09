#!/bin/bash
set -euo pipefail

# CPU dependency installer for VTSearch.
#
# Discovers plugin requirements, installs all dependencies with CPU PyTorch,
# and performs an editable install of the vtsearch package.
#
# Usage:
#   bash install-cpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing VTSearch CPU dependencies..."

# Step 1: Regenerate requirements-plugins.txt from discovered plugin files.
bash "$SCRIPT_DIR/install-plugin-deps.sh" --dry-run

# Step 2: Install all dependencies (core + plugins) with CPU PyTorch.
echo "Installing all dependencies..."
pip install -r requirements.txt -q

# Step 3: Editable install so 'import vtsearch' works.
pip install --no-deps -e . -q

echo "CPU dependencies installed successfully."
