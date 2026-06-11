#!/usr/bin/env bash
# Re-render every screenshot to a temp dir and pixel-diff against the committed
# baselines under docs/user/assets/. Exits non-zero on drift. This flags
# screenshots that should have been refreshed (scripts/screenshots/refresh.sh)
# after a GUI change but weren't.
#
# This is a MANUAL pre-release / periodic chore, NOT a run-tests.sh gate: it
# needs chromium and a running app, neither of which the standard test
# container has. (The cheap, browser-free docs⇄manifest gate is wiring-check.py.)
#
# Note: font hinting / anti-aliasing can introduce sub-pixel noise across
# machines; treat a small number of near-zero-diff files as noise. A per-pixel
# tolerance is a tracked follow-up in docs/plans/user-docs-screenshots.md.
set -euo pipefail
cd "$(dirname "$0")"

BASELINE="../../docs/user/assets"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Rendering to $TMP …"
OUT_DIR="$TMP" node_modules/.bin/tsx capture.ts "$@"

drift=0
shopt -s nullglob
for new in "$TMP"/*.png; do
    name="$(basename "$new")"
    base="$BASELINE/$name"
    if [[ ! -f "$base" ]]; then
        echo "NEW (no baseline): $name"
        drift=1
        continue
    fi
    if command -v magick >/dev/null 2>&1; then
        diffpx=$(magick compare -metric AE "$base" "$new" null: 2>&1 || true)
        if [[ "${diffpx%%.*}" -gt 0 ]] 2>/dev/null; then
            echo "DRIFT: $name ($diffpx px differ)"
            drift=1
        fi
    else
        if ! cmp -s "$base" "$new"; then
            echo "DRIFT (byte): $name"
            drift=1
        fi
    fi
done

if [[ "$drift" -ne 0 ]]; then
    echo ""
    echo "Screenshots drifted from baselines. Review, then run refresh.sh to update."
    exit 1
fi
echo "check.sh OK: all screenshots match the committed baselines."
