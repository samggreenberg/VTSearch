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
MARKER="${VTSEARCH_DEPS_MARKER_DIR:-/tmp}/.vtsearch-deps-installed-${SCRIPT_HASH}"

# Every install below goes through `python -m pip`, never a bare `pip`. The
# cloud image has shipped with the two on different interpreters (`python`
# 3.11 at /usr/local/bin, `pip` on 3.13), and a bare `pip install` then filled
# 3.13 while every documented `python ...` command ran on 3.11 and died on
# `No module named 'flask'` (issue #4662).
#
# For the same reason the marker alone is not trusted: one written by another
# interpreter, or by an older image, would short-circuit an install that never
# reached the `python` in use. So the stack must also import.
deps_importable() {
  python -c 'import flask, vulture' >/dev/null 2>&1
}

if [ -f "$MARKER" ]; then
  if deps_importable; then
    exit 0
  fi
  echo "Found $MARKER, but flask/vulture do not import under $(command -v python); reinstalling."
fi

echo "Installing project dependencies into $(command -v python) ($(python --version 2>&1))..."

# A `python` without its own pip would make every `python -m pip` below fail;
# bootstrap one rather than fall back to whatever `pip` is on PATH.
if ! python -m pip --version >/dev/null 2>&1; then
  python -m ensurepip --upgrade
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/../.."

# Upgrade setuptools first; the system version (68.x) has a broken
# install_layout attribute that prevents building wheels for some packages
# (progressbar, wget) needed by laion_clap.
python -m pip install --upgrade setuptools -q

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
python -m pip install --upgrade --ignore-installed pip wheel cryptography pyjwt urllib3 httplib2 -q

# pypdf ships pre-installed in the cloud container (in /usr/local, so a plain
# upgrade replaces it); VTSearch does not use it. 6.17.0 carries
# PYSEC-2026-4153..4160, patched in 6.19.0.
python -m pip install --upgrade "pypdf>=6.19.0" -q

# The container also carries a per-user site (/root/.local/lib/python3.X/
# site-packages, home of the preinstalled MCP tooling) that sits AHEAD of
# /usr/local on sys.path, with its own urllib3 / pyjwt / anyio. Those copies
# shadow the fresh ones installed above and are what `pip list` -- and so
# pip-audit -- sees, so they are upgraded in place, in that site (anyio is
# also a transitive VTSearch dep, via httpx). Skipped where no user site is
# enabled, e.g. inside a venv. The advisories that surfaced this:
# urllib3 < 2.8.0 (CVE-2026-97687..97689), pyjwt < 2.14.0
# (CVE-2026-101917..102274), anyio < 4.14.2 (CVE-2026-63349..64847).
if python -c 'import site, sys; sys.exit(0 if site.ENABLE_USER_SITE else 1)'; then
  python -m pip install --user --upgrade --ignore-installed pyjwt urllib3 anyio -q
fi

# Work around debian-managed blinker (no RECORD file, so pip cannot
# uninstall it).  Force-installing a fresh copy lets Flask pick it up.
python -m pip install --ignore-installed blinker -q

# VTSBrowse signpost naming deps (see docs/plans/vtsbrowse-toponymy.md):
# apricot-select ships a legacy setup.py sdist and must precede the main
# requirements pass (it's declared in pyproject.toml). Do NOT wrap it in
# SETUPTOOLS_USE_DISTUTILS=stdlib, as scripts/install.sh explains: setuptools
# >= 74 refuses to import with that value set, and Python >= 3.12 has no
# stdlib distutils, so the build dies with "BackendUnavailable: Cannot import
# 'setuptools.build_meta'" (it did on the Python 3.13 container).
# toponymy goes in --no-deps because its transformers<5 pin would downgrade
# the app's transformers and is empirically unnecessary for our usage.
python -m pip install apricot-select -q
python -m pip install --no-deps "toponymy==0.5.2" -q

# Install all dependencies + editable install via pyproject.toml
# (requirements/base.txt is just `-e .[dev,agpl]`; the `agpl` extra keeps
# ultralytics + PyMuPDF installed so the YOLO/PDF tests still exercise the
# real code paths rather than skipping).
python -m pip install --prefer-binary \
  -r "$REPO_DIR/requirements/base.txt" \
  -q

# FaceNet face-identity embedder (vtscore.media.face). facenet-pytorch pins
# torch<2.3 / numpy<2 / Pillow<10.3, which a plain install would honor by
# downgrading the app's stack; those pins are empirically unnecessary (the
# model runs a correct forward pass on the modern stack), so install --no-deps
# and let the app's own torch/torchvision/numpy/Pillow satisfy it at runtime.
python -m pip install --no-deps facenet-pytorch -q

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

# Never write a marker the next run would wrongly trust.
if ! deps_importable; then
  echo "Dependencies installed, but flask/vulture still do not import under $(command -v python)." >&2
  exit 1
fi

touch "$MARKER"
echo "Dependencies installed."
