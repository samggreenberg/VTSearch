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
# The tolerance is for rasterisation, not typefaces. capture.ts draws every shot
# in faces it carries, unhinted (README.md, "One typeface on every machine"), so
# a render elsewhere draws the container's glyphs at the container's advances.
# A render on the GRID (Linux, AMD EPYC, fontconfig hinting `slight`) was the
# first off the container (#5302): every shot without callouts matched its
# baseline exactly past 5% fuzz, bar one whose numbers differed. So on another
# Linux machine these defaults have room to spare; a Mac's CoreText is not yet
# measured. What a render elsewhere cannot match is what the app computes there:
# the Browse map's layout and the odd score or percentage move with the CPU's
# floating point. Emoji, too: each platform draws its own.
#
# So the shots name what the app computes there (`computed` in the manifest:
# the Browse map, the training-domain percentages, the pictures a ranking
# serves), the render writes where it lies beside each shot, and those boxes
# are blanked in the render and the baseline alike before they are compared
# (#5378). A shot with a region masked says so ("masked"), and
#
#   SHOT_MASK=0  compares the masked regions too: on the container, whose
#                floating point made the baselines, they still match, and a
#                ranking or layout change inside them shows.
#
# Without ImageMagick nothing is masked, as nothing is tolerated.
set -euo pipefail
cd "$(dirname "$0")"

BASELINE="../../docs/user/assets"
SHOT_FUZZ="${SHOT_FUZZ:-5}"
SHOT_FUZZ="${SHOT_FUZZ%\%}"
SHOT_MAX_PX="${SHOT_MAX_PX:-150}"
SHOT_MASK="${SHOT_MASK:-1}"
if [[ ! "$SHOT_FUZZ" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "SHOT_FUZZ must be a percentage, like 5 or 2.5; got '$SHOT_FUZZ'" >&2
    exit 2
fi
if [[ ! "$SHOT_MAX_PX" =~ ^[0-9]+$ ]]; then
    echo "SHOT_MAX_PX must be a whole number of pixels; got '$SHOT_MAX_PX'" >&2
    exit 2
fi
if [[ "$SHOT_MASK" != 0 && "$SHOT_MASK" != 1 ]]; then
    echo "SHOT_MASK must be 1 (blank the computed regions) or 0; got '$SHOT_MASK'" >&2
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
    echo "Comparing with ImageMagick: SHOT_FUZZ=$SHOT_FUZZ%, SHOT_MAX_PX=$SHOT_MAX_PX, SHOT_MASK=$SHOT_MASK"
else
    echo "ImageMagick not found: comparing bytes, so SHOT_FUZZ and SHOT_MAX_PX do not apply."
fi

drift=0
checked=0
near=0
masked=0
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
    # Count the differing pixels from compare's mask (white where they differ
    # past the fuzz), not from `-metric AE`'s printout: what AE prints varies by
    # version. 7.1 follows the count with a normalised share in parentheses,
    # and 6.9.13 prints the count times the quantum range (6.5535e+06 for 100
    # px) with the true count in parentheses, which made every shot drift.
    # Anything other than a number is an error message, and a shot that could
    # not be compared is not a match.
    if [[ "$im" == magick ]]; then count=(magick); else count=(convert); fi
    # Blank the shot's computed regions, as the render boxed them, in both.
    # The draw goes in a lossless PNG, so the two blanks are the same pixels.
    a="$base" b="$new" note=""
    boxes="$TMP/${name%.webp}.computed"
    if [[ "$SHOT_MASK" == 1 && -s "$boxes" ]]; then
        draw=()
        while read -r box; do
            [[ "$box" =~ ^[0-9]+,[0-9]+\ [0-9]+,[0-9]+$ ]] || continue
            draw+=(-draw "rectangle $box")
        done <"$boxes"
        if [[ ${#draw[@]} -gt 0 ]]; then
            a="$TMP/mask-base.png" b="$TMP/mask-new.png"
            "${count[@]}" "$base" -alpha off -fill black "${draw[@]}" "$a"
            "${count[@]}" "$new" -alpha off -fill black "${draw[@]}" "$b"
            note=", computed regions masked: $((${#draw[@]} / 2))"
            masked=$((masked + 1))
        fi
    fi
    out=$(im_run compare -fuzz "$SHOT_FUZZ%" -alpha off -compose src \
        -highlight-color white -lowlight-color black "$a" "$b" png:- 2>/dev/null \
        | "${count[@]}" png:- -alpha off -format '%[fx:round(mean*w*h)]' info: 2>&1) || true
    px="${out%% *}"
    if [[ ! "$px" =~ ^[0-9]+$ ]]; then
        echo "DRIFT: $name (compare failed: $out)"
        drift=1
        continue
    fi
    px=$(LC_ALL=C awk -v n="$px" 'BEGIN { printf "%.0f", n }')
    if [[ "$px" -gt "$SHOT_MAX_PX" ]]; then
        echo "DRIFT: $name ($px px differ$note)"
        drift=1
    elif [[ "$px" -gt 0 ]]; then
        echo "near:  $name ($px px differ$note)"
        near=$((near + 1))
    elif [[ -n "$note" ]]; then
        echo "masked: $name (${note#, })"
    fi
done

if [[ "$drift" -ne 0 ]]; then
    echo ""
    echo "Screenshots drifted from baselines. Review, then run refresh.sh to update."
    exit 1
fi
summary="all $checked screenshots match the committed baselines"
if [[ "$near" -gt 0 ]]; then summary+=" ($near within SHOT_MAX_PX)"; fi
if [[ "$masked" -gt 0 ]]; then summary+=", $masked with computed regions masked (SHOT_MASK=0 compares them)"; fi
echo "check.sh OK: $summary."
