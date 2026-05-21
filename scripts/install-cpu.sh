#!/bin/bash
set -euo pipefail

# CPU dependency installer for VTSearch.
#
# Installs runtime + dev dependencies and the vtsearch package itself
# (editable) by forwarding to pyproject.toml via `requirements/base.txt`,
# which is just `--extra-index-url <cpu wheel index>` + `-e .[dev]`.
#
# Usage:
#   bash scripts/install-cpu.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "Installing VTSearch CPU dependencies..."

# Bail out early if the active Python is too old.  pyproject.toml requires
# >=3.10, but pip only enforces it partway through resolution, after a slow
# and confusing failure path.  Check up front instead.
source "$SCRIPT_DIR/_check-python.sh"

# Ensure pip/setuptools/wheel are recent enough to resolve modern wheel-only
# packages.  Stale pips (e.g. 21.x) have a weaker resolver and will fall back
# to building ancient sdists from source, which then fails for lack of
# `bdist_wheel`.
pip install --upgrade pip setuptools wheel -q

pip install -r "$REPO_ROOT/requirements/base.txt" -q

# Wire up the pre-commit git hook (no-op if not in a git checkout
# or if .pre-commit-config.yaml is absent).
if [ -d "$REPO_ROOT/.git" ] && [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
    (cd "$REPO_ROOT" && pre-commit install --install-hooks) || \
        echo "warning: pre-commit install failed; run it manually to enable git hooks"
fi

echo "CPU dependencies installed successfully."
