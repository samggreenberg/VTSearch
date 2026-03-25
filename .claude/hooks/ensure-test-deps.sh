#!/bin/bash
set -euo pipefail

# Lazy dependency installer — only runs pip install the first time.
# Called before test/app commands that actually need the full stack.
# Skips entirely outside remote (Claude Code on the web) environments.

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

MARKER="/tmp/.vtsearch-deps-installed"

if [ -f "$MARKER" ]; then
  exit 0
fi

echo "Installing project dependencies (first test run this session)..."

# Upgrade setuptools first — the system version (68.x) has a broken
# install_layout attribute that prevents building wheels for some packages
# (progressbar, wget) needed by laion_clap.
pip install --upgrade setuptools -q

# Work around debian-managed blinker (no RECORD file, so pip cannot
# uninstall it).  Force-installing a fresh copy lets Flask pick it up.
pip install --ignore-installed blinker -q

# Install the project with CPU PyTorch and dev tools.
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  --prefer-binary \
  -e ".[cpu,dev]" \
  -q

# Install frontend (Angular) dependencies.
# Re-run npm install whenever package-lock.json is newer than node_modules,
# so added/removed packages are always in sync.
if [ -f "frontend/package.json" ]; then
  if [ ! -d "frontend/node_modules" ] || \
     [ "frontend/package-lock.json" -nt "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    (cd frontend && npm install --no-audit --no-fund -q)
  fi
fi

touch "$MARKER"
echo "Dependencies installed."
