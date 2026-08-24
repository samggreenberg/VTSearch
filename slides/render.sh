#!/usr/bin/env bash
# Build one deck without make:  ./render.sh hold-the-line [pdf|html|pptx] [--speaker]
# --speaker builds the speaker view -> _out/<deck>.speaker.<fmt>: each page is
# a miniature of the real rendered slide beside its presenter notes. It renders
# the audience deck to per-slide PNGs first, so it is a two-pass build.
set -euo pipefail
cd "$(dirname "$0")"

deck=${1:?usage: ./render.sh <deck-name> [pdf|html|pptx] [--speaker]}
shift
fmt=pdf
speaker=
for arg in "$@"; do
    case "$arg" in
        --speaker) speaker=1 ;;
        pdf|html|pptx) fmt=$arg ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done

MARP=(npx --yes @marp-team/marp-cli@4)
mkdir -p _out

# Marp warns but exits 0 when a figure path doesn't resolve, producing a deck
# with holes where the figures should be. Treat that warning as fatal.
run_marp() {
    local log
    log=$(mktemp)
    "${MARP[@]}" "$@" --theme-set themes/ --allow-local-files 2>&1 | tee "$log"
    if grep -q "local files are missing" "$log"; then
        rm -f "$log"
        echo "ERROR: Marp could not resolve some figures." >&2
        return 1
    fi
    rm -f "$log"
}

./build.py "$deck"

if [[ -n $speaker ]]; then
    mkdir -p _build/imgs
    rm -f "_build/imgs/$deck".*.png
    run_marp "_build/$deck.md" --images png -o "_build/imgs/$deck.png" \
        || { echo "ERROR: slide-image pass failed." >&2; exit 1; }
    ./build.py --speaker "$deck"
    out="_out/$deck.speaker.$fmt"
    run_marp "_build/$deck.speaker.md" -o "$out" \
        || { rm -f "$out"; echo "ERROR: speaker deck removed." >&2; exit 1; }
else
    out="_out/$deck.$fmt"
    run_marp "_build/$deck.md" -o "$out" \
        || { rm -f "$out"; echo "ERROR: deck removed." >&2; exit 1; }
fi
echo "-> $out"
