#!/usr/bin/env bash
# Run tests with minimal output and a clear PASS/FAIL summary.
#
# Usage:
#   ./run-tests.sh              # run all fast tests (default) + frontend build check
#   ./run-tests.sh core         # run only core group + frontend build check
#   ./run-tests.sh sorting      # run only sorting group
#   ./run-tests.sh core sorting # run core + sorting groups + frontend build check
#   ./run-tests.sh vtscore-clean  # run tests_lib/ with Flask import blocked
#
# Available groups: core, api, sorting, datasets, io, detectors,
#                   downloads, integration, cli, converters
#
# Each group is a folder under tests/ AND tests_lib/. Marker assignment is
# automatic: any file at tests[_lib]/<group>/test_*.py gets marked <group>
# by the respective conftest.  tests_lib/ holds Flask-free library tests
# (Phase 7 of vtscore/docs/architecture.md).
#
# Extra pytest args can follow a '--':
#   ./run-tests.sh core -- -x --tb=short

set -euo pipefail
cd "$(dirname "$0")"

# Install deps if needed
bash .claude/hooks/ensure-test-deps.sh

# vtscore-clean: run only the library-tier tests with Flask blocked.
# Skips the linter / frontend stages because the goal of this mode is
# specifically to verify that the library tier can run independent of
# Flask - not to re-run the linting we already do in the main path.
if [[ "${1:-}" == "vtscore-clean" ]]; then
    shift
    exec python scripts/check-vtscore-clean.py "$@"
fi

# Ruff lint + format check.
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

echo "Running codespell..."
if ! codespell --toml pyproject.toml ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: codespell found typos"
    echo "============================================================"
    exit 1
fi
echo "Running deptry..."
if ! python -m deptry . ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: deptry found dependency issues"
    echo "============================================================"
    exit 1
fi

# pip-audit - scans installed Python packages against the PyPI advisory
# database. Auditing the resolved venv (not requirements files) catches
# transitive vulnerabilities and matches what production will actually run.
#
# `PIP_AUDIT_IGNORE` lists advisory IDs that pip-audit currently reports
# with no fix version available - pinning a "fixed" release isn't an
# option, so the gate would otherwise block indefinitely on upstream
# CVEs that have nothing to do with VTSearch code.  Re-audit the list
# whenever upstream ships a patched release; remove the entry and let
# `ensure-test-deps.sh` upgrade the dep instead of ignoring the CVE.
#   joblib 1.5.3       PYSEC-2024-277             (no upstream fix)
#   pyjwt  2.12.1      PYSEC-2025-183             (no upstream fix)
#   transformers 5.8.1 PYSEC-2025-211..218        (no upstream fix)
PIP_AUDIT_IGNORE=(
    --ignore-vuln PYSEC-2024-277
    --ignore-vuln PYSEC-2025-183
    --ignore-vuln PYSEC-2025-211
    --ignore-vuln PYSEC-2025-212
    --ignore-vuln PYSEC-2025-213
    --ignore-vuln PYSEC-2025-214
    --ignore-vuln PYSEC-2025-215
    --ignore-vuln PYSEC-2025-216
    --ignore-vuln PYSEC-2025-217
    --ignore-vuln PYSEC-2025-218
)
echo "Running pip-audit..."
if ! pip-audit "${PIP_AUDIT_IGNORE[@]}" ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: pip-audit found known vulnerabilities"
    echo "============================================================"
    exit 1
fi

# Pyright - full static type check across vtsearch/ and tests/
# (see `pyrightconfig.json` for the gated scope). The PYRIGHT_PYTHON_FORCE_VERSION
# pin keeps everyone on the same underlying pyright binary regardless of
# what the `pyright` PyPI wrapper would otherwise pull.
echo "Running pyright..."
if ! PYRIGHT_PYTHON_FORCE_VERSION=1.1.408 pyright ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: pyright found type errors"
    echo "============================================================"
    exit 1
fi

# OpenAPI snapshot drift check - regenerate the flask-smorest spec from
# the live app and diff against the checked-in snapshot at
# frontend/openapi.json. The frontend's generated TS client is built
# from this snapshot, so a stale file means the generated client lags
# the real API. Cheap (~2s) and runs every invocation.
echo "Checking OpenAPI snapshot drift..."
_openapi_regen=$(mktemp)
_openapi_dump_log=$(mktemp)
if ! python scripts/dump_openapi.py > "$_openapi_regen" 2> "$_openapi_dump_log"; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: OpenAPI spec dump failed"
    echo "============================================================"
    cat "$_openapi_dump_log"
    rm -f "$_openapi_regen" "$_openapi_dump_log"
    exit 1
fi
if ! diff -u frontend/openapi.json "$_openapi_regen" > /dev/null; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: OpenAPI snapshot is stale"
    echo "============================================================"
    echo "Run 'npm run regenerate-openapi-snapshot' (or"
    echo "'python scripts/dump_openapi.py > frontend/openapi.json') and"
    echo "commit the result."
    diff -u frontend/openapi.json "$_openapi_regen" | head -80
    rm -f "$_openapi_regen" "$_openapi_dump_log"
    exit 1
fi
rm -f "$_openapi_regen" "$_openapi_dump_log"

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
    # --omit=dev: only audit production deps. Dev-only deps (e.g.
    # @angular-devkit/build-angular → webpack-dev-server) regularly carry
    # advisories with "no fix available" upstream because Angular hasn't
    # cut a release yet. Those affect `ng serve` on a developer's machine,
    # not anything that ships to users. Auditing prod deps is the actual
    # security gate worth blocking tests on.
    _audit_log=$(mktemp)
    if (cd frontend && npm audit --omit=dev 2>&1) > "$_audit_log"; then
        echo "Frontend audit OK (0 vulnerabilities in production deps)"
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

# Coverage is opt-in via VTSEARCH_COVERAGE=1 (or `--cov` as first arg).
# Default off because tests already run in ~35s; coverage adds ~10-20%
# overhead and the coverage report is most useful when explicitly asked for.
COV_ARGS=()
if [[ "${VTSEARCH_COVERAGE:-}" == "1" ]]; then
    COV_ARGS=(--cov=vtsearch --cov-report=term-missing)
fi

# Run pytest with:
#   --tb=short  - brief tracebacks (enough to diagnose, not overwhelming)
#   --no-header - skip the platform/plugin header noise
#   -q          - quiet mode (dots instead of full test names)
#   -n auto     - parallel execution via pytest-xdist (one worker per CPU)
#
# Both tests/ (app tier) and tests_lib/ (library tier) are passed in.
# The two trees have independent conftests; pytest's auto-merge picks
# the right autouse fixtures per test based on file location.
if [[ -n "$MARKER_EXPR" ]]; then
    python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto -m "$MARKER_EXPR" "${COV_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto "${COV_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
