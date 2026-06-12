#!/usr/bin/env bash
# Regenerate every user-docs screenshot from the manifest, in place.
# After it runs, `git diff --stat docs/user/assets/` is the precise list of
# shots the GUI change moved. See docs/plans/user-docs-screenshots.md.
#
# Usage:
#   scripts/screenshots/refresh.sh            # all shots, both themes
#   scripts/screenshots/refresh.sh <id>...    # only these shot ids
#
# This harness drives a SINGLE running app (it does not boot its own per run):
# the dev box is RAM-tight and a second app instance would load the image
# embedder twice. Determinism still holds because the synthetic generator uses
# a fixed seed. If no app is already serving on $APP, this script starts one,
# waits for readiness, captures, then stops it.
set -euo pipefail
cd "$(dirname "$0")"

APP="${APP:-http://localhost:5000}"
REPO_ROOT="$(cd ../.. && pwd)"

started_app=""
if ! curl -sf -o /dev/null "$APP/" 2>/dev/null; then
    echo "No app at $APP — starting one (empty dataset registry → SigLIP loads lazily, no CLAP)…"
    ( cd "$REPO_ROOT" && VTSEARCH_TORCH_THREADS=1 python app.py --local > /tmp/vtshots-refresh-app.log 2>&1 ) &
    started_app=$!
    for _ in $(seq 1 60); do
        curl -sf -o /dev/null "$APP/" 2>/dev/null && break
        sleep 2
    done
fi

# Create the deterministic fixtures the recipes need (idempotent).
node ensure-fixtures.mjs

# tsx is a dev dependency in this folder's package.json.
node_modules/.bin/tsx capture.ts "$@"

# Keep every PNG under the repo's 500 KB added-large-file pre-commit cap, the
# same way on every run (so check.sh stays drift-free): losslessly re-encode,
# and 256-colour quantise anything still over ~480 KB (screenshots quantise
# cleanly). Uses the project venv's Pillow.
python - "$REPO_ROOT/docs/user/assets" <<'PY'
import sys, os
from PIL import Image
CAP = 480 * 1024
for name in sorted(os.listdir(sys.argv[1])):
    if not name.endswith(".png"):
        continue
    p = os.path.join(sys.argv[1], name)
    Image.open(p).save(p, "PNG", optimize=True, compress_level=9)
    if os.path.getsize(p) > CAP:
        Image.open(p).convert("RGB").quantize(colors=256, dither=Image.NONE).save(p, "PNG", optimize=True)
        print(f"quantized {name} -> {os.path.getsize(p)//1024} KB")
PY

if [[ -n "$started_app" ]]; then
    echo "Stopping app started by refresh.sh (pid $started_app)…"
    kill "$started_app" 2>/dev/null || true
fi
