#!/usr/bin/env bash
# Build one deck without make:  ./render.sh hold-the-line [pdf|html|pptx|png] [--speaker|--watch]
# --speaker builds the speaker view -> _out/<deck>.speaker.<fmt>: each page is
# a miniature of the real rendered slide beside its presenter notes. It renders
# the audience deck to per-slide PNGs first, so it is a two-pass build.
# --watch starts Marp's live-reloading browser preview instead of writing a file.
# --no-pageno draws no page numbers -> _out/<deck>.unnumbered.<fmt>, for a deck
# being handed over rather than presented. It writes to its own file so the
# numbered deck — the one a question from the room can name a slide in — is
# still there beside it.
# --editable (pptx only) exports real PowerPoint shapes instead of one image per
# slide -> _out/<deck>.editable[.unnumbered].pptx. Needs LibreOffice on PATH:
# Marp renders the deck to PDF and has Impress import it. See README.md.
# The `png` format dumps one image per page and zips the pile
# -> _out/<deck>[.unnumbered]-pngs*.zip, for dropping the slides into somebody
# else's template as pictures. PNG_SCALE sets the resolution (default 2, i.e.
# 2560x1440); PNG_MAX_MB caps one zip. See pack_pngs.py.
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
pageno=
editable=
for arg in "$@"; do
    case "$arg" in
        --speaker) speaker=1 ;;
        --watch) watch=1 ;;
        --no-pageno) pageno=--no-pageno ;;
        --editable) editable=1 ;;
        pdf|html|pptx|png) fmt=$arg ;;
        *) echo "unknown argument: $arg" >&2; exit 1 ;;
    esac
done
if [[ -n $speaker && -n $watch ]]; then
    echo "--speaker and --watch are mutually exclusive" >&2
    exit 1
fi
# build.py refuses these pairs too; catching them here keeps the message next to
# the flags that caused it rather than under a stack of build output.
if [[ -n $speaker && -n $pageno ]]; then
    echo "--speaker and --no-pageno are mutually exclusive: the speaker view is navigated by those numbers" >&2
    exit 1
fi
if [[ -n $speaker && -n $editable ]]; then
    echo "--speaker and --editable are mutually exclusive: the speaker view is a PDF, not a deck to edit" >&2
    exit 1
fi
if [[ -n $editable && -n $watch ]]; then
    echo "--editable and --watch are mutually exclusive: --watch is a live preview, not an export" >&2
    exit 1
fi
if [[ -n $editable && $fmt != pptx ]]; then
    echo "--editable only applies to pptx (got $fmt): it is what makes PowerPoint shapes instead of slide images" >&2
    exit 1
fi
if [[ -n $editable ]] && ! command -v soffice >/dev/null 2>&1; then
    echo "--editable needs LibreOffice on PATH (soffice): Marp has Impress import the rendered PDF." >&2
    echo "  macOS: brew install --cask libreoffice     Debian/Ubuntu: apt-get install libreoffice-impress" >&2
    exit 1
fi

MARP=(npx --yes @marp-team/marp-cli@4)
mkdir -p _out

# How many times 1280x720 each page image is rendered at, and the ceiling on
# one zip. Two is presentation grade without being wasteful: 2560x1440 is over
# a 1080p projector's pixels and comfortably over what any deck displays a
# full-bleed picture at, and going to three doubles the bytes to buy resolution
# the room cannot resolve.
PNG_SCALE=${PNG_SCALE:-2}
PNG_MAX_MB=${PNG_MAX_MB:-25}

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

# Which assembled deck this render is of. build.py writes the unnumbered cut to
# its own file so the two never overwrite each other.
stem=$deck
build_args=("$deck")
marp_args=()
if [[ -n $editable ]]; then
    build_args+=(--editable)
    marp_args+=(--pptx --pptx-editable)
    stem="$stem.editable"
fi
if [[ -n $pageno ]]; then
    build_args+=(--no-pageno)
    stem="$stem.unnumbered"
fi

./build.py "${build_args[@]}"

# Live preview: Marp stays resident and re-renders on save, so it never reaches
# the exit-status/figure checks in run_marp. --no-stdin still matters -- without
# it Marp blocks on stdin before the watcher ever starts.
if [[ -n $watch ]]; then
    exec "${MARP[@]}" "_build/$stem.md" --theme-set themes/ --allow-local-files \
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
elif [[ $fmt == png ]]; then
    # One PNG per page into a directory of its own, then packed. Marp numbers
    # the files itself (`<stem>.001.png`), which is the order they have to be
    # dragged in, so the names are left exactly as it writes them.
    pile="_out/$stem-png"
    rm -rf "$pile"
    mkdir -p "$pile"
    run_marp "_build/$stem.md" --images png --image-scale "$PNG_SCALE" -o "$pile/$stem.png" \
        || { rm -rf "$pile"; echo "ERROR: page images removed." >&2; exit 1; }
    ./pack_pngs.py "$pile" "$stem" --max-mb "$PNG_MAX_MB"
    out="_out/$stem-pngs*.zip"
else
    out="_out/$stem.$fmt"
    run_marp "_build/$stem.md" ${marp_args[@]+"${marp_args[@]}"} -o "$out" \
        || { rm -f "$out"; echo "ERROR: deck removed." >&2; exit 1; }
fi
echo "-> $out"
