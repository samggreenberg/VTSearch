#!/bin/bash
set -euo pipefail

# Auto-discover and install all plugin requirements.txt files.
#
# Each plugin (importer, exporter, media type, etc.) can declare its own
# dependencies in a local requirements.txt.  This script finds them all
# and installs them in a single pip invocation, giving you a cascading
# dependency tree without editing pyproject.toml.
#
# Plugin directories searched:
#   vtsearch/datasets/importers/*/requirements.txt
#   vtsearch/exporters/*/requirements.txt
#   vtsearch/labels/importers/*/requirements.txt
#   vtsearch/media/*/requirements.txt
#   vtsearch/processors/importers/*/requirements.txt
#   vtsearch/settings_io/importers/*/requirements.txt
#   vtsearch/settings_io/exporters/*/requirements.txt
#   vtsearch/settings_io/sources/*/requirements.txt
#   vtsearch/labels/sources/*/requirements.txt
#
# Usage:
#   bash install-plugin-deps.sh                  # install all plugin deps
#   bash install-plugin-deps.sh --dry-run        # just list what would be installed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
fi

# Collect all plugin requirements.txt files.
REQ_FILES=()
while IFS= read -r -d '' f; do
    REQ_FILES+=("$f")
done < <(find "$SCRIPT_DIR/vtsearch" -path "*/requirements.txt" -print0 | sort -z)

if [[ ${#REQ_FILES[@]} -eq 0 ]]; then
    echo "No plugin requirements.txt files found."
    exit 0
fi

echo "Found ${#REQ_FILES[@]} plugin requirements.txt files:"
for f in "${REQ_FILES[@]}"; do
    echo "  ${f#"$SCRIPT_DIR/"}"
done

if $DRY_RUN; then
    echo ""
    echo "(dry run — no packages installed)"
    exit 0
fi

# Build -r flags for pip.
PIP_ARGS=()
for f in "${REQ_FILES[@]}"; do
    PIP_ARGS+=("-r" "$f")
done

echo ""
echo "Installing plugin dependencies..."
pip install "${PIP_ARGS[@]}" -q
echo "Plugin dependencies installed successfully."
