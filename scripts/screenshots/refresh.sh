#!/usr/bin/env bash
# Regenerate every user-docs screenshot from the manifest, in place.
# After it runs, `git diff --stat docs/user/assets/` is the precise list of
# shots the GUI change moved. See docs/plans/user-docs-screenshots.md.
#
# Usage:
#   scripts/screenshots/refresh.sh            # all shots, both themes
#   scripts/screenshots/refresh.sh <id>...    # only these shot ids
#   OUT_DIR=/tmp/x scripts/screenshots/refresh.sh   # render elsewhere (check.sh)
#
# This harness drives a SINGLE app: the dev box is RAM-tight and a second
# instance would load the image embedder twice. If no app is serving on $APP,
# this script starts one, captures, then stops it; that is the reproducible
# path, and the one to use.
#
# The app it starts runs on a fresh data dir, emptied every run
# (data/.screenshots-app), so no refresh photographs state an earlier one left
# behind: example media a recipe uploaded, a dragged panel width, achievement
# counters, an import still winding down (#4299). The model cache is shared, so
# nothing is downloaded twice; the fixtures are imported again every run, which
# costs a few minutes of embedding on a CPU box.
#
# It also fits the Browse map under VTSEARCH_PROJECTION_SEED, because an
# unseeded UMAP fit lays the map out differently every time (#4296), and draws
# the spot check's picks under VTSEARCH_SPOT_CHECK_SEED, because an unseeded
# draw shows the floor-check shot a different pick every time (#4330).
#
# An app you started yourself is used as it is, with its own data and settings,
# so the shots it gives are not the committed ones.
set -euo pipefail
cd "$(dirname "$0")"

APP="${APP:-http://localhost:5000}"
REPO_ROOT="$(cd ../.. && pwd)"
PROJECTION_SEED=0
SPOT_CHECK_SEED=0
APP_DATA_DIR="$REPO_ROOT/data/.screenshots-app"
APP_LOG=/tmp/vtshots-refresh-app.log

started_app=""
stop_app() {
    if [[ -n "$started_app" ]]; then
        echo "Stopping the app refresh.sh started (pid $started_app)…"
        kill "$started_app" 2>/dev/null || true
        wait "$started_app" 2>/dev/null || true
    fi
}
trap stop_app EXIT

if curl -sf -o /dev/null "$APP/" 2>/dev/null; then
    echo "Using the app already at $APP, with its own data and settings. For the committed"
    echo "shots, stop it and let refresh.sh start one on a fresh data dir."
else
    echo "No app at $APP — starting one on a fresh data dir ($APP_DATA_DIR)…"
    rm -rf "$APP_DATA_DIR"
    mkdir -p "$APP_DATA_DIR"
    # `exec`, so $! is the app itself and stop_app's kill reaches it.
    (
        cd "$REPO_ROOT"
        VTSEARCH_DATA_DIR="$APP_DATA_DIR" \
            VTSEARCH_MODELS_DIR="${VTSEARCH_MODELS_DIR:-$REPO_ROOT/data/models}" \
            VTSEARCH_TORCH_THREADS=1 \
            VTSEARCH_PROJECTION_SEED=$PROJECTION_SEED \
            VTSEARCH_SPOT_CHECK_SEED=$SPOT_CHECK_SEED \
            exec python app.py --local > "$APP_LOG" 2>&1
    ) &
    started_app=$!
    # capture.ts shows this dir as the install's own data dir (maskVolatile).
    export SHOTS_APP_DATA_DIR="$APP_DATA_DIR"
    for _ in $(seq 1 90); do
        curl -sf -o /dev/null "$APP/" 2>/dev/null && break
        if ! kill -0 "$started_app" 2>/dev/null; then
            echo "The app exited before it was ready; the end of $APP_LOG:" >&2
            tail -20 "$APP_LOG" >&2
            exit 1
        fi
        sleep 2
    done
    curl -sf -o /dev/null "$APP/" 2>/dev/null || { echo "The app at $APP never came up; see $APP_LOG" >&2; exit 1; }
fi

# Create the deterministic fixtures the recipes need (idempotent).
node ensure-fixtures.mjs

# tsx is a dev dependency in this folder's package.json. capture.ts writes
# WebP directly (see its encodeWebp), so there is no post-pass here.
node_modules/.bin/tsx capture.ts "$@"
