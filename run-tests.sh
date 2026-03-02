#!/usr/bin/env bash
# Run tests with minimal output and a clear PASS/FAIL summary.
#
# Usage:
#   ./run-tests.sh              # run all fast tests (default)
#   ./run-tests.sh core         # run only core group
#   ./run-tests.sh sorting      # run only sorting group
#   ./run-tests.sh core sorting # run core + sorting groups
#
# Available groups: core, api, sorting, datasets, io, models,
#                   downloads, integration, cli, converters
#
# Extra pytest args can follow a '--':
#   ./run-tests.sh core -- -x --tb=short

set -euo pipefail
cd "$(dirname "$0")"

# Install deps if needed
bash .claude/hooks/ensure-test-deps.sh

# Split arguments into groups and extra pytest args
TEST_GROUPS=()
EXTRA_ARGS=()
PAST_SEPARATOR=false

for arg in "$@"; do
    if [[ "$arg" == "--" ]]; then
        PAST_SEPARATOR=true
        continue
    fi
    if $PAST_SEPARATOR; then
        EXTRA_ARGS+=("$arg")
    else
        TEST_GROUPS+=("$arg")
    fi
done

# Build the pytest marker expression
if [[ ${#TEST_GROUPS[@]} -eq 0 ]]; then
    # Default: run all fast tests
    MARKER_EXPR=""
else
    # Combine groups with OR: -m "core or sorting"
    MARKER_EXPR=""
    for g in "${TEST_GROUPS[@]}"; do
        if [[ -n "$MARKER_EXPR" ]]; then
            MARKER_EXPR="$MARKER_EXPR or $g"
        else
            MARKER_EXPR="$g"
        fi
    done
fi

# Run pytest with:
#   --tb=short  — brief tracebacks (enough to diagnose, not overwhelming)
#   --no-header — skip the platform/plugin header noise
#   -q          — quiet mode (dots instead of full test names)
if [[ -n "$MARKER_EXPR" ]]; then
    python -m pytest tests/ -q --tb=short --no-header -m "$MARKER_EXPR" "${EXTRA_ARGS[@]}"
else
    python -m pytest tests/ -q --tb=short --no-header "${EXTRA_ARGS[@]}"
fi
