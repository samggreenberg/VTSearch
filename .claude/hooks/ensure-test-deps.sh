#!/bin/bash
set -euo pipefail

# Lazy dependency installer: only runs pip install the first time.
# Called before test/app commands that actually need the full stack.
# Skips entirely outside remote (Claude Code on the web) environments.

if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Hash the script content into the marker filename so that any edit to
# this hook (new upgrade step, new dep, etc.) invalidates the previous
# session's marker and forces the install block to re-run.  Without this,
# a long-lived container that ran an older version of the script would
# keep short-circuiting at the `-f $MARKER` check forever, even after the
# repo was updated to require new install steps, e.g. the pip-audit
# system-package upgrade block added in `e3eb87fd` was silently skipped
# on containers whose marker was created before that fix landed.
SCRIPT_HASH="$(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1 | cut -c1-12)"
MARKER="/tmp/.vtsearch-deps-installed-${SCRIPT_HASH}"

if [ -f "$MARKER" ]; then
  exit 0
fi

echo "Installing project dependencies (first test run this session)..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/../.."

# Upgrade setuptools first; the system version (68.x) has a broken
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
#   - httplib2 0.20.4 ships pre-installed (via launchpadlib, debian-managed);
#     0.32.0 patches PYSEC-2026-3444. Unused by VTSearch.
#   - pypdf 6.17.0 ships pre-installed in the Python 3.13 image, required by
#     nothing of ours (PyMuPDF reads our PDFs); 6.19.0 patches
#     PYSEC-2026-4153..4160.
pip install --upgrade --ignore-installed pip wheel cryptography pyjwt urllib3 httplib2 pypdf -q

# The container also carries a per-user site (/root/.local/lib/python3.11/
# site-packages, home of the preinstalled MCP tooling) that sits AHEAD of
# /usr/local on sys.path, with its own urllib3 / pyjwt / anyio. Those copies
# shadow the fresh ones installed above and are what `pip list` -- and so
# pip-audit -- sees, so they are upgraded in place, in that site (anyio is
# also a transitive VTSearch dep, via httpx). Skipped where no user site is
# enabled, e.g. inside a venv. The advisories that surfaced this:
# urllib3 < 2.8.0 (CVE-2026-97687..97689), pyjwt < 2.14.0
# (CVE-2026-101917..102274), anyio < 4.14.2 (CVE-2026-63349..64847).
if python -c 'import site, sys; sys.exit(0 if site.ENABLE_USER_SITE else 1)'; then
  pip install --user --upgrade --ignore-installed pyjwt urllib3 anyio -q
fi

# Work around debian-managed blinker (no RECORD file, so pip cannot
# uninstall it).  Force-installing a fresh copy lets Flask pick it up.
pip install --ignore-installed blinker -q

# VTSBrowse signpost naming deps (see docs/plans/vtsbrowse-toponymy.md):
# apricot-select's legacy setup.py needs the stdlib-distutils shim and must
# precede the main requirements pass (it's declared in pyproject.toml);
# toponymy goes in --no-deps because its transformers<5 pin would downgrade
# the app's transformers and is empirically unnecessary for our usage.
# Python 3.12 removed stdlib distutils, so there the shim makes setuptools'
# build backend unimportable and setuptools' own vendored copy is the one
# that builds it.
if python -c 'import sys; sys.exit(0 if sys.version_info < (3, 12) else 1)'; then
  SETUPTOOLS_USE_DISTUTILS=stdlib pip install apricot-select -q
else
  pip install apricot-select -q
fi
pip install --no-deps "toponymy==0.5.2" -q

# Install all dependencies + editable install via pyproject.toml
# (requirements/base.txt is just `-e .[dev,agpl]`; the `agpl` extra keeps
# ultralytics + PyMuPDF installed so the YOLO/PDF tests still exercise the
# real code paths rather than skipping).
pip install --prefer-binary \
  -r "$REPO_DIR/requirements/base.txt" \
  -q

# FaceNet face-identity embedder (vtscore.media.face). facenet-pytorch pins
# torch<2.3 / numpy<2 / Pillow<10.3, which a plain install would honor by
# downgrading the app's stack; those pins are empirically unnecessary (the
# model runs a correct forward pass on the modern stack), so install --no-deps
# and let the app's own torch/torchvision/numpy/Pillow satisfy it at runtime.
pip install --no-deps facenet-pytorch -q

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
