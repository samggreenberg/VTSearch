#!/usr/bin/env bash
# Run tests with minimal output and a clear PASS/FAIL summary.
#
# Usage:
#   ./run-tests.sh              # full gate: every check + all fast tests
#   ./run-tests.sh core         # core group + the cheap gates (see below)
#   ./run-tests.sh sorting      # sorting group + the cheap gates
#   ./run-tests.sh core sorting # core + sorting groups
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
#
# ---------------------------------------------------------------------------
# How the run is staged
#
# There is no CI, so a *full* `./run-tests.sh` remains the only real gate and
# still runs every check. What changed is the shape of the run, because the
# stages are independent of each other and the box has more than one core:
#
#   1. Cheap gates, serial, fail-fast (~8s total). Linters, doc invariants,
#      snapshot drift. These are quick enough that stopping at the first
#      failure costs nothing and saves you reading past it.
#   2. The frontend production build, serial. It has to precede pytest because
#      a handful of tests serve the built bundle out of `static/`, and it is
#      also the frontend failure people hit most, so it is worth failing on
#      before spending minutes elsewhere.
#   3. The heavy, mutually independent stages *concurrently*: pyright,
#      pip-audit, the frontend unit suite, and pytest. Sequentially these were
#      ~338s on a 4-vCPU box; overlapped they land at ~250s, within a few
#      percent of what the machine's cores can physically do.
#
# Stage 3 deliberately does NOT stop at the first failure: every lane runs to
# completion and all failures are reported together. A serial chain that dies
# on the first bad gate makes you pay the whole runtime again per fix.
#
# Group runs (`./run-tests.sh <group>`) run stage 1 and their own tests, and
# skip the stage-3 gates that are not about the code you just changed —
# pyright, pip-audit, and the frontend suite. That keeps the edit/test loop in
# the seconds it should be instead of paying ~105s of whole-repo checks to run
# a five-second group. The skip is announced on every group run, because the
# full run is what actually gates a push. Set VTSEARCH_FULL_GATES=1 to force
# the complete chain on a group run.
# ---------------------------------------------------------------------------

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
# A healthy full run is a few minutes (see the staging note above), so 30
# minutes is many times the worst legitimate run and will not misfire — but it
# bounds the failure mode where the run wedges and nobody notices. That is not
# hypothetical: an xdist run once sat for 2h12m with three of its four workers
# `<defunct>` and the master idle, because *nothing* in this script had an
# upper bound. A per-test timeout does not cover that case (a dead worker can't
# fire its own timeout), which is why the cap lives out here, wrapping every
# stage — dep install, linters, npm, pytest alike.
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

# ---------------------------------------------------------------------------
# Argument parsing
#
# Done up front (it used to sit after the gate chain) because which gates run
# now depends on whether this is a full run or a group run.
# ---------------------------------------------------------------------------
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

_is_full_run=false
if [[ ${#TEST_GROUPS[@]} -eq 0 ]]; then
    _is_full_run=true
fi

# A group run can be promoted back to the complete chain on demand.
_run_whole_repo_gates=false
if $_is_full_run || [[ "${VTSEARCH_FULL_GATES:-}" == "1" ]]; then
    _run_whole_repo_gates=true
fi

_group_named() {
    local want="$1" g
    for g in ${TEST_GROUPS[@]+"${TEST_GROUPS[@]}"}; do
        [[ "$g" == "$want" ]] && return 0
    done
    return 1
}

# The frontend production build runs for the full suite and for the core /
# frontend groups. Catches compilation errors without needing a browser.
_run_frontend_check=false
if $_is_full_run || _group_named core || _group_named frontend; then
    _run_frontend_check=true
fi

# The frontend Vitest unit suite runs for the full run or an explicit
# `frontend` group, but NOT for `core`: it is heavier than the compile-only
# build check, so it stays off the fast `core` path.
_run_frontend_unit=false
if $_is_full_run || _group_named frontend; then
    _run_frontend_unit=true
fi

# `frontend` is a frontend-only gate; it has no Python tests. If it's the only
# requested group, skip pytest entirely so it doesn't error on an empty
# `-m frontend` selection.
_run_pytest=true
if [[ ${#TEST_GROUPS[@]} -eq 1 && "${TEST_GROUPS[0]}" == "frontend" ]]; then
    _run_pytest=false
fi

_blocked() {
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: $1"
    echo "============================================================"
}

# ---------------------------------------------------------------------------
# Stage 1: cheap gates. Serial and fail-fast — the whole set is ~8s, so
# stopping at the first failure costs nothing.
# ---------------------------------------------------------------------------

# Ruff lint + format check.
# Runs first because it's fast (~0.5s) and catches mistakes the pytest /
# frontend stages can't see, e.g. F401 unused-import on TYPE_CHECKING
# imports whose only "use" is inside a string-form forward reference.
echo "Running ruff check..."
if ! ruff check . ; then
    _blocked "ruff check failed"
    exit 1
fi
echo "Running ruff format --check..."
if ! ruff format --check . ; then
    _blocked "ruff format --check failed (run 'ruff format .')"
    exit 1
fi

echo "Running codespell..."
if ! codespell --toml pyproject.toml ; then
    _blocked "codespell found typos"
    exit 1
fi

# Documentation drift: relative links, in-page anchors, backticked repo paths,
# leaked absolute machine paths, plan-file citations anywhere in the tree, and
# broken code fences. Pure invariants against the current tree — nothing to
# re-pin — and it imports nothing, so it costs ~0.4s and sits with the linters.
echo "Checking documentation..."
if ! python scripts/check-docs.py ; then
    _blocked "documentation check found drift"
    exit 1
fi

echo "Running deptry..."
if ! python -m deptry . ; then
    _blocked "deptry found dependency issues"
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
    _blocked "OpenAPI spec dump failed"
    cat "$_openapi_dump_log"
    rm -f "$_openapi_regen" "$_openapi_dump_log"
    exit 1
fi
if ! diff -u frontend/openapi.json "$_openapi_regen" > /dev/null; then
    _blocked "OpenAPI snapshot is stale"
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
    _blocked "generated doc inventories are stale"
    echo "Run 'python scripts/gen-docs-inventories.py' and commit the"
    echo "result."
    exit 1
fi

echo "Checking Dockerfiles..."
if ! python scripts/check-dockerfiles.py ; then
    _blocked "Dockerfile check failed"
    exit 1
fi

# User-docs screenshot wiring: every manifest shot id has both theme PNGs on
# disk, and every screenshot the user docs embed maps to a manifest id. Cheap,
# browser-free (see docs/plans/user-docs-screenshots.md); the pixel-diff
# (check.sh) needs chromium and stays a manual chore.
echo "Checking user-docs screenshot wiring..."
if ! python scripts/screenshots/wiring-check.py ; then
    _blocked "user-docs screenshot wiring check failed"
    exit 1
fi

# vtscore package docs: every top-level module / sub-package of vtscore/ is
# covered by a packages/ doc, and no doc cites a file.py:NNN line anchor (they
# rot on the next edit; cite module-and-symbol instead). Regex sweep, imports
# nothing, ~0.1s. See scripts/check-vtscore-docs.py for the policy.
echo "Checking vtscore package docs..."
if ! python scripts/check-vtscore-docs.py ; then
    _blocked "vtscore package docs check failed"
    exit 1
fi

# Eval/app sync: the eval framework reproduces a handful of app surfaces it
# cannot call (the TypeScript autopilot phase machine, the app's default
# resolution). This gate notices when one of those app surfaces changes, so the
# eval default arm can't quietly stop being the shipped algorithm. Parses
# source, imports nothing, ~0.3s.
echo "Checking eval/app sync..."
if ! python scripts/check-eval-app-sync.py ; then
    _blocked "eval framework is out of sync with the app"
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 2: frontend production build (serial).
#
# Ahead of the parallel stage on purpose. `tests/core/test_frontend.py` and
# parts of `tests/api/test_dashboard.py` serve the built bundle out of
# `static/`, so building it *while* pytest reads it would race; and a broken
# build is the frontend failure that actually happens, so it is worth learning
# about before the long stage starts.
# ---------------------------------------------------------------------------
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
            _blocked "Frontend build has warnings (treated as errors)"
            grep -A 10 '▲ \[WARNING\]' "$_fe_plain"
            rm -f "$_fe_log" "$_fe_plain"
            exit 1
        fi
        rm -f "$_fe_plain"
        echo "Frontend build OK"
    else
        _blocked "Frontend build failed"
        cat "$_fe_log"
        rm -f "$_fe_log"
        exit 1
    fi
    rm -f "$_fe_log"
elif $_run_frontend_check && [ ! -d "frontend/node_modules" ]; then
    echo "Skipping frontend build check (node_modules not installed; run: cd frontend && npm install)"
fi

# ---------------------------------------------------------------------------
# Stage 3: the heavy, mutually independent stages, run concurrently.
#
# pytest runs in the foreground so its progress still streams live (it is the
# output people actually watch); everything else runs as a background lane
# writing to its own log, reported after pytest returns. Every lane runs to
# completion even if another fails, so one pass surfaces every problem.
# ---------------------------------------------------------------------------

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

_lane_names=()
_lane_pids=()
_lane_logs=()

_start_lane() {
    local name="$1"; shift
    local log
    log=$(mktemp)
    "$@" > "$log" 2>&1 &
    _lane_pids+=("$!")
    _lane_names+=("$name")
    _lane_logs+=("$log")
}

# Pyright: full static type check across vtsearch/ and tests/
# (see `pyrightconfig.json` for the gated scope). The PYRIGHT_PYTHON_FORCE_VERSION
# pin keeps everyone on the same underlying pyright binary regardless of
# what the `pyright` PyPI wrapper would otherwise pull.
_lane_pyright() { PYRIGHT_PYTHON_FORCE_VERSION=1.1.408 pyright; }
_lane_pip_audit() { pip-audit "${PIP_AUDIT_IGNORE[@]}"; }
# --omit=dev: only audit production deps. Dev-only deps (e.g.
# @angular-devkit/build-angular → webpack-dev-server) regularly carry
# advisories with "no fix available" upstream because Angular hasn't
# cut a release yet. Those affect `ng serve` on a developer's machine,
# not anything that ships to users. Auditing prod deps is the actual
# security gate worth blocking tests on.
_lane_npm_audit() { (cd frontend && npm audit --omit=dev); }
# `npm run test:ci` regenerates the API client (pretest:ci) then runs
# `ng test --no-watch`, which exits non-zero on any spec failure.
_lane_frontend_unit() { (cd frontend && npm run test:ci); }

if $_run_whole_repo_gates; then
    _start_lane "pyright" _lane_pyright
    _start_lane "pip-audit" _lane_pip_audit
fi
if $_run_frontend_check && [ -d "frontend/node_modules" ]; then
    _start_lane "npm audit" _lane_npm_audit
fi
if $_run_frontend_unit && [ -d "frontend/node_modules" ]; then
    _start_lane "frontend unit tests (Vitest)" _lane_frontend_unit
elif $_run_frontend_unit && [ ! -d "frontend/node_modules" ]; then
    echo "Skipping frontend unit tests (node_modules not installed; run: cd frontend && npm install)"
fi

if [[ ${#_lane_names[@]} -gt 0 ]]; then
    echo "Running in parallel with the tests: ${_lane_names[*]}"
fi
if ! $_run_whole_repo_gates; then
    echo "Group run: skipping pyright and pip-audit. A full './run-tests.sh' is"
    echo "the gate before pushing (or set VTSEARCH_FULL_GATES=1 to force them)."
fi

# --- pytest, in the foreground ---------------------------------------------
_pytest_status=0
if $_run_pytest; then
    # Build the pytest marker expression
    if $_is_full_run; then
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

    # Coverage is opt-in via VTSEARCH_COVERAGE=1. Default off because it adds
    # ~10-20% overhead and the report is most useful when explicitly asked for.
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
    #       @pytest.mark.xdist_group run together on one worker. Used to pin all
    #       real-UMAP-fit tests to a single worker so the ~30s numba JIT compile
    #       of umap-learn's kernels is paid once per run rather than once per
    #       worker (tests_lib/projection/test_umap_projection.py), and to keep
    #       each eval-sweep module together so its memoized sweeps hit the cache
    #       instead of being recomputed on every worker that draws one
    #       (tests_lib/detectors/sweep_cache.py).
    #
    # Both tests/ (app tier) and tests_lib/ (library tier) are passed in.
    # The two trees have independent conftests; pytest's auto-merge picks
    # the right autouse fixtures per test based on file location.
    set +e
    if [[ -n "$MARKER_EXPR" ]]; then
        python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto --dist loadgroup -m "$MARKER_EXPR" ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    else
        python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto --dist loadgroup ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    fi
    _pytest_status=$?
    set -e
fi

# --- collect the background lanes ------------------------------------------
_failed_lanes=()
if [[ ${#_lane_names[@]} -gt 0 ]]; then
    echo ""
    echo "Waiting for: ${_lane_names[*]}"
    for _i in "${!_lane_pids[@]}"; do
        set +e
        wait "${_lane_pids[$_i]}"
        _st=$?
        set -e
        if [[ $_st -eq 0 ]]; then
            echo "  ${_lane_names[$_i]}: OK"
        else
            echo "  ${_lane_names[$_i]}: FAILED"
            _failed_lanes+=("$_i")
        fi
    done
fi

# Report every failing lane, not just the first: the point of running them
# concurrently is that one pass tells you everything that is broken.
for _i in ${_failed_lanes[@]+"${_failed_lanes[@]}"}; do
    _blocked "${_lane_names[$_i]} failed"
    tail -80 "${_lane_logs[$_i]}"
done
for _log in ${_lane_logs[@]+"${_lane_logs[@]}"}; do
    rm -f "$_log"
done

# Final verdict banner. With lanes reporting after pytest, pytest's own
# summary is no longer the last thing printed — this banner is, so "read the
# last ====-bordered block" stays the way to read a run's outcome.
echo ""
echo "============================================================"
if [[ ${#_failed_lanes[@]} -gt 0 || $_pytest_status -ne 0 ]]; then
    _verdict_parts=()
    if [[ $_pytest_status -ne 0 ]]; then
        _verdict_parts+=("pytest")
    fi
    for _i in ${_failed_lanes[@]+"${_failed_lanes[@]}"}; do
        _verdict_parts+=("${_lane_names[$_i]}")
    done
    _joined=$(IFS=", "; echo "${_verdict_parts[*]}")
    echo "RUN FAILED: $_joined"
    echo "============================================================"
    if [[ $_pytest_status -ne 0 ]]; then
        exit $_pytest_status
    fi
    exit 1
fi
if $_run_pytest; then
    echo "RUN PASSED (all gates green; pytest summary above)"
else
    echo "RUN PASSED (frontend-only; no Python tests in the 'frontend' group)"
fi
echo "============================================================"
exit 0
