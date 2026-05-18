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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/../.."

# Upgrade setuptools first — the system version (68.x) has a broken
# install_layout attribute that prevents building wheels for some packages
# (progressbar, wget) needed by laion_clap.
pip install --upgrade setuptools -q

# Upgrade Ubuntu 24.04's pre-installed Python packages so pip-audit
# (run as part of ./run-tests.sh) doesn't flag stale baseline CVEs that
# have nothing to do with VTSearch itself:
#   - cryptography 41.0.7 / pyjwt 2.7.0 / wheel 0.42.0 / pip 24.0 ship
#     pre-installed; we upgrade them so they aren't flagged. Several are
#     debian-managed (no RECORD file → pip cannot uninstall them), so
#     we use --ignore-installed to drop fresh copies alongside.
#   - urllib3 2.6.3 is a real transitive dep (via requests); newer
#     versions patch CVE-2026-44431/44432.
pip install --upgrade --ignore-installed pip wheel cryptography pyjwt urllib3 -q

# Work around debian-managed blinker (no RECORD file, so pip cannot
# uninstall it).  Force-installing a fresh copy lets Flask pick it up.
pip install --ignore-installed blinker -q

# Install all dependencies + editable install via pyproject.toml
# (requirements/base.txt is just `-e .[dev]`).
pip install --prefer-binary \
  -r "$REPO_DIR/requirements/base.txt" \
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
