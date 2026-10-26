#!/usr/bin/env bash
# Re-render every screenshot to a temp dir and pixel-diff against the committed
# baselines under docs/user/assets/. Exits non-zero on drift. This flags
# screenshots that should have been refreshed (scripts/screenshots/refresh.sh)
# after a GUI change but weren't.
#
# This is a MANUAL pre-release / periodic chore, NOT a run-tests.sh gate: it
# renders through refresh.sh (so, like the baselines, on a fresh data dir with
# the seeded Browse map, when no app is already running), and it is far too slow
# and machine-sensitive (font hinting, see below) to gate every test run.
# (The cheap, browser-free docs⇄manifest gate is wiring-check.py.)
#
# Font hinting and anti-aliasing differ between rendering stacks, so a render
# from another one differs from the baselines even when the GUI has not moved:
# faint fringes along every edge, and the odd glyph drawn a sub-pixel along.
# With ImageMagick (7's `magick`, or 6's `compare`), two knobs absorb that:
#
#   SHOT_FUZZ    percent colour distance under which two pixels count as the
#                same (compare -fuzz). Default 5: it absorbs the fringes.
#   SHOT_MAX_PX  pixels that may still differ, past the fuzz, before a shot is
#                drift. Default 150: it absorbs the shifted glyphs.
#
# The defaults come from shots of an unchanged GUI rendered before and after the
# cloud container's Chromium upgrade of October 2026 (#5278). Past 5% fuzz the
# noisiest differed by 100 px, and the smallest real text change, one digit, by
# 241 px. A swapped tab icon (about 100 px) passes, which is why a pass whose
# pixels still differ is listed as "near", with its count: a near miss stays
# visible. Without ImageMagick this compares bytes, with no tolerance at all.
#
# The tolerance is for rasterisation, not typefaces. The app's font stack ends
# in the system sans-serif (Inter in the cloud container), so a machine that
# resolves it to another face (San Francisco on macOS, Segoe UI on Windows)
# moves every line of text, and no setting passes that and still catches a
# moved control.
set -euo pipefail
cd "$(dirname "$0")"

BASELINE="../../docs/user/assets"
SHOT_FUZZ="${SHOT_FUZZ:-5}"
SHOT_FUZZ="${SHOT_FUZZ%\%}"
SHOT_MAX_PX="${SHOT_MAX_PX:-150}"
if [[ ! "$SHOT_FUZZ" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "SHOT_FUZZ must be a percentage, like 5 or 2.5; got '$SHOT_FUZZ'" >&2
    exit 2
fi
if [[ ! "$SHOT_MAX_PX" =~ ^[0-9]+$ ]]; then
    echo "SHOT_MAX_PX must be a whole number of pixels; got '$SHOT_MAX_PX'" >&2
    exit 2
fi

# ImageMagick 7 runs every tool through `magick`; 6 installs `compare` and
# `identify` as commands of their own.
im=""
if command -v magick >/dev/null 2>&1; then
    im=magick
elif command -v compare >/dev/null 2>&1 && compare -version 2>/dev/null | grep -q ImageMagick; then
    im=legacy
fi
im_run() {
    if [[ "$im" == magick ]]; then magick "$@"; else "$@"; fi
}

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Rendering to $TMP …"
OUT_DIR="$TMP" ./refresh.sh "$@"

if [[ -n "$im" ]]; then
    echo "Comparing with ImageMagick: SHOT_FUZZ=$SHOT_FUZZ%, SHOT_MAX_PX=$SHOT_MAX_PX"
else
    echo "ImageMagick not found: comparing bytes, so SHOT_FUZZ and SHOT_MAX_PX do not apply."
fi

drift=0
checked=0
near=0
shopt -s nullglob
for new in "$TMP"/*.webp; do
    name="$(basename "$new")"
    base="$BASELINE/$name"
    checked=$((checked + 1))
    if [[ ! -f "$base" ]]; then
        echo "NEW (no baseline): $name"
        drift=1
        continue
    fi
    if [[ -z "$im" ]]; then
        if ! cmp -s "$base" "$new"; then
            echo "DRIFT (byte): $name"
            drift=1
        fi
        continue
    fi
    # ImageMagick 6 compares only the overlap of two sizes and reports it equal,
    # so a shot that grew is caught here instead.
    size_base=$(im_run identify -format '%wx%h' "$base" 2>&1) || true
    size_new=$(im_run identify -format '%wx%h' "$new" 2>&1) || true
    if [[ "$size_base" != "$size_new" ]]; then
        echo "DRIFT: $name (now $size_new, baseline $size_base)"
        drift=1
        continue
    fi
    # AE prints a pixel count: 6 may write a large one as 1.296e+06, and 7.1
    # follows it with a normalised share in parentheses. Anything else is an
    # error message, and a shot that could not be compared is not a match.
    out=$(im_run compare -metric AE -fuzz "$SHOT_FUZZ%" "$base" "$new" null: 2>&1) || true
    px="${out%% *}"
    if [[ ! "$px" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]]; then
        echo "DRIFT: $name (compare failed: $out)"
        drift=1
        continue
    fi
    px=$(LC_ALL=C awk -v n="$px" 'BEGIN { printf "%.0f", n }')
    if [[ "$px" -gt "$SHOT_MAX_PX" ]]; then
        echo "DRIFT: $name ($px px differ)"
        drift=1
    elif [[ "$px" -gt 0 ]]; then
        echo "near:  $name ($px px differ)"
        near=$((near + 1))
    fi
done

if [[ "$drift" -ne 0 ]]; then
    echo ""
    echo "Screenshots drifted from baselines. Review, then run refresh.sh to update."
    exit 1
fi
if [[ "$near" -gt 0 ]]; then
    echo "check.sh OK: all $checked screenshots match the committed baselines ($near within SHOT_MAX_PX)."
else
    echo "check.sh OK: all $checked screenshots match the committed baselines."
fi
