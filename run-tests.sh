#!/usr/bin/env bash
# Run tests with minimal output and a clear PASS/FAIL summary.
#
# Usage:
#   ./run-tests.sh              # run all fast tests (default) + frontend build check
#   ./run-tests.sh core         # run only core group + frontend build check
#   ./run-tests.sh sorting      # run only sorting group
#   ./run-tests.sh core sorting # run core + sorting groups + frontend build check
#
# Available groups: core, api, sorting, datasets, io, detectors,
#                   downloads, integration, cli, converters
#
# Each group is a folder under tests/. Marker assignment is automatic:
# any file at tests/<group>/test_*.py gets marked <group> by conftest.
#
# Extra pytest args can follow a '--':
#   ./run-tests.sh core -- -x --tb=short

set -euo pipefail
cd "$(dirname "$0")"

# Install deps if needed
bash .claude/hooks/ensure-test-deps.sh

# Ruff lint + format check (matches .github/workflows/lint.yml).
# Runs early because it's fast (~1s) and catches mistakes the pytest /
# frontend stages can't see, e.g. F401 unused-import on TYPE_CHECKING
# imports whose only "use" is inside a string-form forward reference.
echo "Running ruff check..."
if ! ruff check . ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: ruff check failed"
    echo "============================================================"
    exit 1
fi
echo "Running ruff format --check..."
if ! ruff format --check . ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: ruff format --check failed (run 'ruff format .')"
    echo "============================================================"
    exit 1
fi

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

# Run frontend TypeScript build check for full suite or when core/frontend groups are requested.
# Catches compilation errors without needing a browser (ng test requires Chrome, build:prod does not).
_run_frontend_check=false
if [[ ${#TEST_GROUPS[@]} -eq 0 ]]; then
    _run_frontend_check=true
else
    for _g in "${TEST_GROUPS[@]}"; do
        if [[ "$_g" == "core" || "$_g" == "frontend" ]]; then
            _run_frontend_check=true
            break
        fi
    done
fi

if $_run_frontend_check && [ -d "frontend/node_modules" ]; then
    echo "Checking frontend TypeScript build..."
    _fe_log=$(mktemp)
    if (cd frontend && npm run build:prod 2>&1) > "$_fe_log"; then
        # Treat Angular compiler warnings (e.g. NG8107) as errors
        if grep -q '▲ \[WARNING\]' "$_fe_log"; then
            echo ""
            echo "============================================================"
            echo "TESTS BLOCKED: Frontend build has warnings (treated as errors)"
            echo "============================================================"
            grep -A 10 '▲ \[WARNING\]' "$_fe_log"
            rm -f "$_fe_log"
            exit 1
        fi
        echo "Frontend build OK"
    else
        echo ""
        echo "============================================================"
        echo "TESTS BLOCKED: Frontend build failed"
        echo "============================================================"
        cat "$_fe_log"
        rm -f "$_fe_log"
        exit 1
    fi
    rm -f "$_fe_log"

    echo "Checking frontend dependencies for vulnerabilities..."
    _audit_log=$(mktemp)
    if (cd frontend && npm audit 2>&1) > "$_audit_log"; then
        echo "Frontend audit OK (0 vulnerabilities)"
    else
        echo ""
        echo "============================================================"
        echo "TESTS BLOCKED: npm audit found known vulnerabilities"
        echo "============================================================"
        cat "$_audit_log"
        rm -f "$_audit_log"
        exit 1
    fi
    rm -f "$_audit_log"
elif $_run_frontend_check && [ ! -d "frontend/node_modules" ]; then
    echo "Skipping frontend build check (node_modules not installed; run: cd frontend && npm install)"
fi

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
#   -n auto     — parallel execution via pytest-xdist (one worker per CPU)
if [[ -n "$MARKER_EXPR" ]]; then
    python -m pytest tests/ -q --tb=short --no-header -n auto -m "$MARKER_EXPR" "${EXTRA_ARGS[@]}"
else
    python -m pytest tests/ -q --tb=short --no-header -n auto "${EXTRA_ARGS[@]}"
fi
