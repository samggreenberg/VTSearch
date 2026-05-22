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

# Shared progress-printing helpers.
# shellcheck source=_progress.sh
source "$SCRIPT_DIR/_progress.sh"

vts_progress_init 4 "Installing VTSearch CPU dependencies"

vts_progress_step "Checking Python version (>= 3.10)"
source "$SCRIPT_DIR/_check-python.sh"

vts_progress_step "Upgrading pip / setuptools / wheel"
pip install --upgrade pip setuptools wheel --progress-bar on

vts_progress_step "Installing runtime + dev dependencies (this may take several minutes)"
pip install -r "$REPO_ROOT/requirements/base.txt" --progress-bar on

vts_progress_step "Wiring up pre-commit git hook"
if [ -d "$REPO_ROOT/.git" ] && [ -f "$REPO_ROOT/.pre-commit-config.yaml" ]; then
    (cd "$REPO_ROOT" && pre-commit install --install-hooks) || \
        echo "warning: pre-commit install failed; run it manually to enable git hooks"
else
    echo "  (skipped: no git checkout or .pre-commit-config.yaml)"
fi

vts_progress_done "CPU dependencies installed successfully"
