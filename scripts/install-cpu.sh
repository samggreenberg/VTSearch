#!/bin/bash
set -euo pipefail

# CPU dependency installer for VTSearch.
#
# Discovers plugin requirements, installs all dependencies with CPU PyTorch,
# and performs an editable install of the vtsearch package.
#
# Usage:
#   bash scripts/install-cpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing VTSearch CPU dependencies..."

# Step 1: Regenerate requirements/plugins.txt from discovered plugin files.
bash "$SCRIPT_DIR/install-plugin-deps.sh" --dry-run

# Step 2: Install all dependencies (core + plugins) with CPU PyTorch.
echo "Installing all dependencies..."
pip install -r "$REPO_ROOT/requirements/base.txt" -q

# Step 3: Editable install so 'import vtsearch' works.
pip install --no-deps -e "$REPO_ROOT" -q

# Step 4: Wire up the pre-commit git hook (no-op if not in a git checkout
# or if .pre-commit-config.yaml is absent).
if [ -d "$REPO_ROOT/.git" ] && [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
    (cd "$REPO_ROOT" && pre-commit install --install-hooks) || \
        echo "warning: pre-commit install failed; run it manually to enable git hooks"
fi

echo "CPU dependencies installed successfully."
