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
# embedder twice. Determinism still holds because the fixtures are the Smiley
# example's generated drawings (a pure function of the generator and its seeds)
# with a fixed vote baseline. If no app is already serving on $APP, this script
# starts one, waits for readiness, captures, then stops it.
#
# The Browse shots frame a UMAP map, which an unseeded fit lays out differently
# every time. The app this script starts fits under VTSEARCH_PROJECTION_SEED, so
# the map is the same on every refresh (#4296); an app you started yourself
# needs the same variable, or the Browse shots show some other map.
set -euo pipefail
cd "$(dirname "$0")"

APP="${APP:-http://localhost:5000}"
REPO_ROOT="$(cd ../.. && pwd)"
PROJECTION_SEED=0

started_app=""
if curl -sf -o /dev/null "$APP/" 2>/dev/null; then
    echo "Using the app already at $APP. Its Browse map matches the committed shots only if"
    echo "it was started with VTSEARCH_PROJECTION_SEED=$PROJECTION_SEED."
else
    echo "No app at $APP — starting one (empty dataset registry → SigLIP loads lazily, no CLAP)…"
    ( cd "$REPO_ROOT" && VTSEARCH_TORCH_THREADS=1 VTSEARCH_PROJECTION_SEED=$PROJECTION_SEED \
        python app.py --local > /tmp/vtshots-refresh-app.log 2>&1 ) &
    started_app=$!
    for _ in $(seq 1 60); do
        curl -sf -o /dev/null "$APP/" 2>/dev/null && break
        sleep 2
    done
fi

# Create the deterministic fixtures the recipes need (idempotent).
node ensure-fixtures.mjs

# tsx is a dev dependency in this folder's package.json. capture.ts writes
# WebP directly (see its encodeWebp), so there is no post-pass here.
node_modules/.bin/tsx capture.ts "$@"

if [[ -n "$started_app" ]]; then
    echo "Stopping app started by refresh.sh (pid $started_app)…"
    kill "$started_app" 2>/dev/null || true
fi
