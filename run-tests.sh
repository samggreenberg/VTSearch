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
#                   downloads, integration, cli, converters, projection,
#                   frontend (build + audit + Vitest, no Python tests), gpu
#
# Each group is a folder under tests/ AND tests_lib/. Marker assignment is
# automatic: any file at tests[_lib]/<group>/test_*.py gets marked <group>
# by the respective conftest.  tests_lib/ holds Flask-free library tests
# (see vtscore/docs/architecture.md).
#
# NOTE: naming a group puts `-m <group>` on the pytest command line, which
# REPLACES the `-m 'not gpu and not slow'` default in pyproject.toml's
# addopts rather than combining with it. So a group run also picks up any
# slow/gpu tests in that folder (that is how `./run-tests.sh gpu` works).
# To keep the default exclusions, spell the filter out after `--`, e.g.
#   ./run-tests.sh cli -- -m 'cli and not slow'
#
# Extra pytest args can follow a '--':
#   ./run-tests.sh core -- -x --tb=short

set -euo pipefail
cd "$(dirname "$0")"
# Absolute, slash-containing path to this script. `$0` alone can be a bare
# filename (e.g. `run-tests.sh` when invoked as `bash run-tests.sh`), which
# makes `timeout`'s execvp do a PATH-only lookup below and fail with "No
# such file or directory". We've already cd'd into dirname("$0"), so pwd
# joined with basename("$0") is always absolute regardless of how we were
# invoked.
_self="$(pwd)/$(basename "$0")"

# Wall-clock cap on the whole run.
#
# A healthy full run is single-digit minutes (frontend gate ~3min, pytest ~35s,
# plus first-run dep install), so 30 minutes is ~5x the worst legitimate run and
# will not misfire — but it bounds the failure mode where the run wedges and
# nobody notices. That is not hypothetical: an xdist run once sat for 2h12m with
# three of its four workers `<defunct>` and the master idle, because *nothing*
# in this script had an upper bound. A per-test timeout does not cover that case
# (a dead worker can't fire its own timeout), which is why the cap lives out
# here, wrapping every stage — dep install, linters, npm, pytest alike.
#
# Implemented as a one-shot re-exec under `timeout`: the guard variable stops
# the child from re-wrapping itself. Set VTSEARCH_TEST_TIMEOUT=0 to opt out (for
# a deliberately long run, e.g. GPU tests or a full coverage sweep).
VTSEARCH_TEST_TIMEOUT=${VTSEARCH_TEST_TIMEOUT:-1800}
if [[ -z "${_VT_TIMEOUT_WRAPPED:-}" && "$VTSEARCH_TEST_TIMEOUT" != "0" ]] \
    && command -v timeout >/dev/null 2>&1; then
    export _VT_TIMEOUT_WRAPPED=1
    # TERM first so pytest can print what it was doing; KILL 30s later for a
    # process too wedged to answer (exactly the defunct-worker case).
    set +e
    timeout --signal=TERM --kill-after=30 "$VTSEARCH_TEST_TIMEOUT" "$_self" "$@"
    _timeout_status=$?
    set -e
    if [[ $_timeout_status -eq 124 || $_timeout_status -eq 137 ]]; then
        echo ""
        echo "============================================================"
        echo "TESTS TIMED OUT after ${VTSEARCH_TEST_TIMEOUT}s (wall-clock cap)"
        echo ""
        echo "The run wedged rather than failed. Check for dead xdist workers"
        echo "(ps aux | grep defunct) or a hung network/install step. Re-run a"
        echo "single group to narrow it down, or set VTSEARCH_TEST_TIMEOUT=0"
        echo "if this run is legitimately meant to take longer."
        echo "============================================================"
    fi
    exit $_timeout_status
fi

# Install deps if needed
bash .claude/hooks/ensure-test-deps.sh

# vtscore-clean: run only the library-tier tests with Flask blocked.
# Skips the linter / frontend stages because the goal of this mode is
# specifically to verify that the library tier can run independent of
# Flask; we do not re-run the linting we already do in the main path.
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
# Documentation drift: relative links, in-page anchors, backticked repo paths,
# leaked absolute machine paths, plan-file citations anywhere in the tree, and
# broken code fences. Pure invariants against the current tree — nothing to
# re-pin — and it imports nothing, so it costs ~0.4s and sits with the linters.
echo "Checking documentation..."
if ! python scripts/check-docs.py ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: documentation check found drift"
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

# pip-audit: scans installed Python packages against the PyPI advisory
# database. Auditing the resolved venv (not requirements files) catches
# transitive vulnerabilities and matches what production will actually run.
#
# `PIP_AUDIT_IGNORE` lists advisory IDs that pip-audit currently reports
# with no fix version available; pinning a "fixed" release isn't an
# option, so the gate would otherwise block indefinitely on upstream
# CVEs that have nothing to do with VTSearch code.  Re-audit the list
# whenever upstream ships a patched release; remove the entry and let
# `ensure-test-deps.sh` upgrade the dep instead of ignoring the CVE.
#   joblib 1.5.3       PYSEC-2024-277             (no upstream fix)
#   pyjwt  2.12.1      PYSEC-2025-183             (no upstream fix)
#   transformers 5.8.1 PYSEC-2025-211..218        (no upstream fix)
#   httplib2 0.20.4    PYSEC-2026-3444            (not a VTSearch dep; pulled in
#                                                  by launchpadlib in the Ubuntu
#                                                  base image, so requirements
#                                                  can't upgrade it)
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
    --ignore-vuln PYSEC-2026-3444
)
echo "Running pip-audit..."
if ! pip-audit "${PIP_AUDIT_IGNORE[@]}" ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: pip-audit found known vulnerabilities"
    echo "============================================================"
    exit 1
fi

# Pyright: full static type check across vtsearch/ and tests/
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

# OpenAPI snapshot drift check: regenerate the flask-smorest spec from
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

# Generated doc-inventory drift check: regenerate the registry-backed
# tables embedded in the docs (embedders, plugin families, demo datasets,
# ...) and fail if any committed region is stale. Same shape as the
# OpenAPI snapshot gate above; see scripts/gen-docs-inventories.py.
echo "Checking generated doc inventories..."
if ! python scripts/gen-docs-inventories.py --check ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: generated doc inventories are stale"
    echo "============================================================"
    echo "Run 'python scripts/gen-docs-inventories.py' and commit the"
    echo "result."
    exit 1
fi

echo "Checking Dockerfiles..."
if ! python scripts/check-dockerfiles.py ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: Dockerfile check failed"
    echo "============================================================"
    exit 1
fi

# User-docs screenshot wiring: every manifest shot id has both theme PNGs on
# disk, and every screenshot the user docs embed maps to a manifest id. Cheap,
# browser-free (see docs/plans/user-docs-screenshots.md); the pixel-diff
# (check.sh) needs chromium and stays a manual chore.
echo "Checking user-docs screenshot wiring..."
if ! python scripts/screenshots/wiring-check.py ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: user-docs screenshot wiring check failed"
    echo "============================================================"
    exit 1
fi

# vtscore package docs: every top-level module / sub-package of vtscore/ is
# covered by a packages/ doc, and no doc cites a file.py:NNN line anchor (they
# rot on the next edit; cite module-and-symbol instead). Regex sweep, imports
# nothing, ~0.1s. See scripts/check-vtscore-docs.py for the policy.
echo "Checking vtscore package docs..."
if ! python scripts/check-vtscore-docs.py ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: vtscore package docs check failed"
    echo "============================================================"
    exit 1
fi

# Eval/app sync: the eval framework reproduces a handful of app surfaces it
# cannot call (the TypeScript autopilot phase machine, the app's default
# resolution). This gate notices when one of those app surfaces changes, so the
# eval default arm can't quietly stop being the shipped algorithm. Parses
# source, imports nothing, ~0.3s.
echo "Checking eval/app sync..."
if ! python scripts/check-eval-app-sync.py ; then
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: eval framework is out of sync with the app"
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
# Catches compilation errors without needing a browser (build:prod is headless).
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

# Run the frontend Vitest unit suite for the full run or an explicit `frontend`
# group, but NOT for `core`. The unit run (a full app build + headless Vitest)
# is heavier than the compile-only build check, so it stays off the fast `core`
# path; the full `./run-tests.sh` (the real gate, since there's no CI) and
# `./run-tests.sh frontend` are where it runs.
_run_frontend_unit=false
if [[ ${#TEST_GROUPS[@]} -eq 0 ]]; then
    _run_frontend_unit=true
else
    for _g in "${TEST_GROUPS[@]}"; do
        if [[ "$_g" == "frontend" ]]; then
            _run_frontend_unit=true
            break
        fi
    done
fi

if $_run_frontend_check && [ -d "frontend/node_modules" ]; then
    echo "Checking frontend TypeScript build..."
    _fe_log=$(mktemp)
    if (cd frontend && npm run build:prod 2>&1) > "$_fe_log"; then
        # Treat Angular compiler warnings (e.g. NG8107) and budget warnings as
        # errors. Angular colourises its output even when stdout is a file, and
        # it interleaves the escapes *inside* the marker
        # (`ESC[33m▲ ESC[43;33m[ESC[43;30mWARNING…`), so the literal
        # `▲ [WARNING]` never matches the raw log — that blindness let an
        # over-budget initial bundle sail past this gate for months. Match
        # against an ANSI-stripped copy instead.
        _fe_plain=$(mktemp)
        sed -r 's/\x1b\[[0-9;]*m//g' "$_fe_log" > "$_fe_plain"
        if grep -q '▲ \[WARNING\]' "$_fe_plain"; then
            echo ""
            echo "============================================================"
            echo "TESTS BLOCKED: Frontend build has warnings (treated as errors)"
            echo "============================================================"
            grep -A 10 '▲ \[WARNING\]' "$_fe_plain"
            rm -f "$_fe_log" "$_fe_plain"
            exit 1
        fi
        rm -f "$_fe_plain"
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

# Frontend unit tests (Vitest, headless via jsdom). `npm run test:ci` regenerates
# the API client (pretest:ci) then runs `ng test --no-watch`, which exits
# non-zero on any spec failure.
if $_run_frontend_unit && [ -d "frontend/node_modules" ]; then
    echo "Running frontend unit tests (Vitest)..."
    _vt_log=$(mktemp)
    if (cd frontend && npm run test:ci 2>&1) > "$_vt_log"; then
        _vt_count=$(sed -r 's/\x1b\[[0-9;]*m//g' "$_vt_log" | grep -oE 'Tests +[0-9]+ passed' | tail -1)
        echo "Frontend unit tests OK${_vt_count:+ ($_vt_count)}"
    else
        echo ""
        echo "============================================================"
        echo "TESTS BLOCKED: Frontend unit tests failed"
        echo "============================================================"
        tail -80 "$_vt_log"
        rm -f "$_vt_log"
        exit 1
    fi
    rm -f "$_vt_log"
elif $_run_frontend_unit && [ ! -d "frontend/node_modules" ]; then
    echo "Skipping frontend unit tests (node_modules not installed; run: cd frontend && npm install)"
fi

# `frontend` is a frontend-only gate (build + audit + Vitest above); it has no
# Python tests. If it's the only requested group, skip pytest entirely so it
# doesn't error on an empty `-m frontend` selection.
if [[ ${#TEST_GROUPS[@]} -eq 1 && "${TEST_GROUPS[0]}" == "frontend" ]]; then
    echo "Frontend-only run complete (no Python tests in the 'frontend' group)."
    exit 0
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

# Coverage is opt-in via VTSEARCH_COVERAGE=1. Default off because tests
# already run in ~35s; coverage adds ~10-20% overhead and the coverage
# report is most useful when explicitly asked for.
COV_ARGS=()
if [[ "${VTSEARCH_COVERAGE:-}" == "1" ]]; then
    COV_ARGS=(--cov=vtsearch --cov-report=term-missing)
fi

# Run pytest with:
#   --tb=short: brief tracebacks (enough to diagnose, not overwhelming)
#   --no-header: skip the platform/plugin header noise
#   -q:         quiet mode (dots instead of full test names)
#   -n auto:    parallel execution via pytest-xdist (one worker per CPU)
#   --dist loadgroup: like the default load scheduling, except tests marked
#       @pytest.mark.xdist_group run together on one worker.  Used to pin all
#       real-UMAP-fit tests to a single worker so the ~30s numba JIT compile
#       of umap-learn's kernels is paid once per run, not once per worker
#       (see tests_lib/projection/test_umap_projection.py).
#
# Both tests/ (app tier) and tests_lib/ (library tier) are passed in.
# The two trees have independent conftests; pytest's auto-merge picks
# the right autouse fixtures per test based on file location.
if [[ -n "$MARKER_EXPR" ]]; then
    python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto --dist loadgroup -m "$MARKER_EXPR" "${COV_ARGS[@]}" "${EXTRA_ARGS[@]}"
else
    python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto --dist loadgroup "${COV_ARGS[@]}" "${EXTRA_ARGS[@]}"
fi
