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
# unseeded UMAP fit lays the map out differently every time (#4296).
#
# An app you started yourself is used as it is, with its own data and settings,
# so the shots it gives are not the committed ones. It must run this checkout's
# commit, though: one from any other commit is refused, and named (#4324).
set -euo pipefail
cd "$(dirname "$0")"

APP="${APP:-http://localhost:5000}"
REPO_ROOT="$(cd ../.. && pwd)"
PROJECTION_SEED=0
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

# An app already serving from another commit (one an older refresh.sh failed to
# stop, or one started before the last commit) runs that commit's code, and its
# shots look normal: capture.ts hides the stale-build toast. So compare its
# version with the checkout's and refuse on a mismatch. Both are HEAD's commit
# time (vtsearch/__init__.py), so uncommitted edits move neither, and an app
# started before them still passes.
refuse_stale_app() {
    local ours theirs hostport port="" listeners=""
    ours=$(cd "$REPO_ROOT" && python -c 'import vtsearch; print(vtsearch.__version__)' 2>/dev/null) || true
    theirs=$(curl -sf "$APP/api/version" 2>/dev/null \
        | python -c 'import json, sys; print(json.load(sys.stdin)["version"])' 2>/dev/null) || true
    # Either side unknown (no git, no baked stamp) is no evidence either way.
    if [[ -z "$ours" || -z "$theirs" || "$ours" == 0.0.0-unknown || "$theirs" == 0.0.0-unknown \
        || "$ours" == "$theirs" ]]; then
        return 0
    fi
    # Name the process when it is on this machine, as app.py's port preflight does.
    hostport="${APP#*://}"
    hostport="${hostport%%/*}"
    if [[ "$hostport" =~ ^(localhost|127\.0\.0\.1):([0-9]+)$ ]]; then
        port="${BASH_REMATCH[2]}"
        listeners=$(cd "$REPO_ROOT" && python -c '
import sys
from vtsearch.port_preflight import _describe_pid, _find_listener_pids
print(", ".join(_describe_pid(p) for p in _find_listener_pids(int(sys.argv[1]))))' "$port" 2>/dev/null) || true
    fi
    {
        echo "The app at $APP runs a different commit from this checkout, so its shots"
        echo "would not show this tree:"
        echo "  app:      $theirs${listeners:+, pid $listeners}"
        echo "  checkout: $ours"
        echo "Stop it${port:+ (\`fuser -k $port/tcp\`)} and rerun; with nothing serving,"
        echo "refresh.sh starts an app of its own on a fresh data dir."
    } >&2
    exit 1
}

if curl -sf -o /dev/null "$APP/" 2>/dev/null; then
    refuse_stale_app
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
