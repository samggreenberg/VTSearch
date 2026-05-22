# Sourced helper: aborts the calling script if the active Python is < 3.10.
# Uses `command -v python || python3` so it works both inside and outside a venv.

_vts_py="$(command -v python || command -v python3 || true)"
if [ -z "$_vts_py" ]; then
    echo "ERROR: no python interpreter found on PATH." >&2
    exit 1
fi

if ! "$_vts_py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    _vts_ver="$("$_vts_py" --version 2>&1)"
    cat >&2 <<EOF
ERROR: VTSearch requires Python 3.10+, but '$_vts_py' is $_vts_ver.

Upgrade Python, then create a fresh venv before re-running this script:

  Amazon Linux 2023 / RHEL / Fedora:   sudo dnf install python3.11
  Ubuntu / Debian:                     sudo apt install python3.11
                                       (older Ubuntu may need the deadsnakes PPA)
  macOS (Homebrew):                    brew install python@3.11

Then:

  deactivate 2>/dev/null || true
  rm -rf venv
  python3.11 -m venv venv
  source venv/bin/activate

See docs/SETUP.md for full instructions.
EOF
    exit 1
fi

unset _vts_py _vts_ver
