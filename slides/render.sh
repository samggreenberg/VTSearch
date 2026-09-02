#!/usr/bin/env bash
# Build one deck without make:  ./render.sh hold-the-line [pdf|html|pptx] [--speaker|--watch]
# --speaker builds the speaker view -> _out/<deck>.speaker.<fmt>: each page is
# a miniature of the real rendered slide beside its presenter notes. It renders
# the audience deck to per-slide PNGs first, so it is a two-pass build.
# --watch starts Marp's live-reloading browser preview instead of writing a file.
#
# This is the single Marp wrapper: slides/Makefile delegates every target here
# rather than invoking Marp itself, so the --no-stdin and PIPESTATUS fixes below
# cannot be bypassed by building through `make` (#3434).
set -euo pipefail
cd "$(dirname "$0")"

deck=${1:?usage: ./render.sh <deck-name> [pdf|html|pptx] [--speaker|--watch]}
shift
fmt=pdf
speaker=
watch=
for arg in "$@"; do
    case "$arg" in
        --speaker) speaker=1 ;;
        --watch) watch=1 ;;
        pdf|html|pptx) fmt=$arg ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done
if [[ -n $speaker && -n $watch ]]; then
    echo "--speaker and --watch are mutually exclusive" >&2
    exit 1
fi

MARP=(npx --yes @marp-team/marp-cli@4)
mkdir -p _out

# Marp warns but exits 0 when a figure path doesn't resolve, producing a deck
# with holes where the figures should be. Treat that warning as fatal.
run_marp() {
    local log status
    log=$(mktemp)
    # --no-stdin: without it Marp waits for EOF on stdin before converting, so
    # a render started from anything that does not close stdin (a script, a CI
    # step, an agent shell) hangs forever with no output rather than failing.
    "${MARP[@]}" "$@" --theme-set themes/ --allow-local-files --no-stdin 2>&1 | tee "$log"
    # `set -e` does not see a failure on the left of a pipe, and this script has
    # no `pipefail`, so without this the whole render reports success after Marp
    # has died — which is how a run that could not find a browser at all still
    # printed "-> _out/<deck>.pdf" and exited 0, with no PDF anywhere (#3301).
    status=${PIPESTATUS[0]}
    if [[ $status -ne 0 ]]; then
        rm -f "$log"
        echo "ERROR: Marp exited $status." >&2
        return "$status"
    fi
    if grep -q "local files are missing" "$log"; then
        rm -f "$log"
        echo "ERROR: Marp could not resolve some figures." >&2
        return 1
    fi
    rm -f "$log"
}

./build.py "$deck"

# Live preview: Marp stays resident and re-renders on save, so it never reaches
# the exit-status/figure checks in run_marp. --no-stdin still matters -- without
# it Marp blocks on stdin before the watcher ever starts.
if [[ -n $watch ]]; then
    exec "${MARP[@]}" "_build/$deck.md" --theme-set themes/ --allow-local-files \
        --no-stdin -w --preview
fi

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
