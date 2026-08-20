#!/usr/bin/env bash
# Build one deck without make:  ./render.sh scale26-review [pdf|html|pptx]
set -euo pipefail
cd "$(dirname "$0")"

deck=${1:?usage: ./render.sh <deck-name> [pdf|html|pptx]}
fmt=${2:-pdf}

./build.py "$deck"
mkdir -p _out

# Marp warns but exits 0 when a figure path doesn't resolve, producing a deck
# with holes where the figures should be. Treat that warning as fatal.
log=$(mktemp)
trap 'rm -f "$log"' EXIT
npx --yes @marp-team/marp-cli@4 "_build/$deck.md" \
    --theme-set themes/ --allow-local-files -o "_out/$deck.$fmt" 2>&1 | tee "$log"

if grep -q "local files are missing" "$log"; then
    rm -f "_out/$deck.$fmt"
    echo "ERROR: Marp could not resolve some figures. Deck removed." >&2
    exit 1
fi
echo "-> _out/$deck.$fmt"
