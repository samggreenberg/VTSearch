#!/usr/bin/env bash
# Run tests with minimal output and a clear PASS/FAIL summary.
#
# Usage:
#   ./run-tests.sh              # full gate: every check + all fast tests
#   ./run-tests.sh core         # core group + the cheap gates (see below)
#   ./run-tests.sh sorting      # sorting group + the cheap gates
#   ./run-tests.sh core sorting # core + sorting groups
#   ./run-tests.sh vtscore-clean  # run tests_lib/ with Flask import blocked
#   ./run-tests.sh slides       # slide-deck gates only (~8s, no Python tests)
#   ./run-tests.sh docs         # markdown-only gate (engages by itself; see below)
#
# Available groups: core, api, sorting, datasets, io, detectors,
#                   downloads, integration, cli, converters, projection,
#                   meta (repo/tooling tests: packaging, Dockerfiles, docs,
#                   scripts/, .claude/hooks/, the gates themselves, the test
#                   harness — nothing that ships as vtsearch/vtscore code),
#                   frontend (build + audit + Vitest, no Python tests),
#                   slides (the four gates a deck can trip, no Python
#                   tests — see the note below),
#                   docs (markdown-only: every cheap gate, plus the few tests
#                   that read a doc — see the note below), gpu
#
# Each group is a folder under tests/, tests_lib/, or both. Marker assignment is
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
# Nothing tests in CI, so a *full* `./run-tests.sh` is the only real gate and
# runs every check. The run is staged, because the stages are independent of
# each other and the box has more than one core. Measured on a warm 4-vCPU
# box with 17k tests (2026-10): the serial chain this replaced took 791s, of
# which 37s were the cheap gates one after another, 27s the frontend build,
# 239s pyright, and 488s pytest with the heavy lanes started beside it (411s
# with the box to itself).
#
#   1. The stale-tree check, alone (~0.2s): every other gate passes by
#      construction on the wrong tree, so whether it is the right tree is asked
#      first.
#   2. The cheap gates, CONCURRENTLY, every one to completion. Linters, doc
#      invariants, snapshot drift: fourteen single-threaded checks that read
#      the tree and write nothing, bounded by codespell (~9s warm). Serially
#      they were 37s; together they are the length of the slowest one, and one
#      pass reports every stage-2 failure instead of the first.
#   3. The heavy whole-repo gates, concurrently, in two kinds:
#        - GATING lanes, which must be green before pytest starts: pyright and
#          the frontend production build. pyright is the slowest gate after
#          pytest and the one a type error fails; the build is the frontend
#          failure people actually hit, and pytest serves the bundle it writes
#          to `static/`, so building while pytest reads would race. The first
#          gating lane to fail stops the run there, with its log: dying before
#          pytest skips most of the runtime, which is the cheap side of the
#          fail-fast trade.
#        - FREE lanes, which start now and are collected after pytest:
#          pip-audit, the vulture whitelist check, `npm audit`, and the
#          Vitest suite (started the moment the build is green, because both
#          regenerate the API client and must not do so at once). pyright
#          leaves two of four cores idle for its whole run; the free lanes
#          fill them, instead of starting beside pytest's four workers, where
#          they cost it 77s (startup went from 25s to 73s).
#      pyright itself is incremental: scripts/pyright-incremental.py checks
#      only the files changed since the last green check plus everything that
#      transitively imports them, which is exact rather than heuristic (a
#      file's diagnostics depend on its text, its transitive imports, the
#      installed packages and the config, and nothing else). A fresh clone, a
#      changed config or environment, or VTSEARCH_FRESH=1 gets the full check.
#      Measured: 239s full, ~3s for one changed test file.
#   4. pytest, alone on the box. Tests that already passed on this identical
#      working tree -- in a group run a minute ago, or in a full run nothing
#      has changed since -- are not re-run (tests_shared/pass_cache.py keys
#      the record on a hash of the tree and the environment); the rest run.
#      Then the free lanes are collected and every failure is reported
#      together.
#
# Group runs (`./run-tests.sh <group>`) run stages 1-2 and their own tests, and
# skip the whole-repo gates in stage 3 that are not about the code you just
# changed -- pyright, pip-audit and the vulture whitelist check. That keeps
# the edit/test loop in the seconds it should be. The skip is announced on
# every group run and named in the verdict, because the full run is what
# actually gates a push. Set VTSEARCH_FULL_GATES=1 to force those three back
# on for a group run. The frontend gates follow the group, not that variable:
# the build and npm audit run for `core` and `frontend`, Vitest for `frontend`
# alone.
#
# Three fast paths prune a full run further, each on a premise the run checks
# against the branch's diff rather than trusts, and each announcing itself:
#
# `slides` is the one group that also gates a push, because a change confined
# to slides/ cannot reach anything else in the repo: nothing imports
# slides/build.py (it is a standalone script), pyrightconfig.json does not
# include it, and no test in either tree touches a deck. So the group runs
# only the four gates that can actually observe a deck -- ruff (build.py),
# codespell (slide prose), check-docs.py (fragments are tracked markdown), and
# build.py --check -- and skips pytest and every whole-repo gate. The group
# refuses to run when the branch touches anything outside slides/, so the
# exemption cannot be taken by mistake.
#
# The markdown-only path is the same idea one step wider, and it does not wait
# to be asked. A change confined to tracked markdown cannot be seen by pyright
# (which excludes markdown), by pip-audit, by the frontend build or its unit
# suite, or by nearly all of the Python tests. So a *bare* `./run-tests.sh`
# looks at what the branch actually changed and, when the answer is "markdown
# and nothing else", keeps the whole of stage 2 -- the gates that genuinely
# read markdown live there -- and narrows pytest to the tests that can observe
# a doc (tests_shared/markdown_surface.py names them, and a meta test keeps
# that list honest). It auto-engages because the thing being fixed is the
# *default* path: a plan-file session pays the full run precisely by running
# the command everyone runs.
#
# The frontend-only path is its mirror: when the branch changes nothing
# outside frontend/, pyright (which excludes that tree), pip-audit and the
# vulture whitelist check cannot see the change and are skipped, the build,
# npm audit and Vitest run in full, and pytest is narrowed to the tests that
# can observe the frontend -- the ones that read a file under frontend/, serve
# the built bundle, or ask the Flask client for a route the SPA catch-all
# answers with it (tests_shared/frontend_surface.py names them; a meta test
# keeps that list honest too).
#
# The frontend gates also prune themselves on any full run: when the branch
# changes no frontend/ path they have nothing to judge (that is a claim about
# the diff, decided by scripts/check-frontend-gate.py), so the build, npm audit
# and Vitest are skipped and the verdict says so. The build still runs when
# `static/` holds no bundle, so the tests that serve one do not build it inside
# a pytest worker.
#
# VTSEARCH_FULL_GATES=1 turns the markdown-only and frontend-only narrowing off
# and forces the whole-repo gates on; VTSEARCH_FRESH=1 ignores every record of
# an earlier run (full pyright, every test re-run) without disabling the
# pruning that is proven against the diff.
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

# Memory headroom check.
#
# The gates in this script are the memory-hungry part of the repo, not pytest:
# `pyright` runs under node and peaks around 5.4G RSS on one thread (it takes
# a second thread only when 12G is free; see stage 3), and the Angular
# `build:prod` is node too and also memory-hungry. On a small machine the run does not
# fail cleanly — node aborts with "Ineffective mark-compacts near heap limit",
# or the kernel's OOM killer takes whatever it likes, including the shell that
# launched the run. That is not hypothetical either: a 3 GB box died this way
# mid-`pyright`, and because the wrapper below reads 137 as its kill signal, the
# corpse was labelled TESTS TIMED OUT rather than "out of memory".
#
# So say it up front, where it is still actionable. This warns rather than
# blocks: the number below is the observed requirement, not a measured cliff,
# and a machine under it may still finish.
if [[ -z "${_VT_TIMEOUT_WRAPPED:-}" && -r /proc/meminfo ]]; then
    _mem_avail_gb=$(awk '/^MemAvailable:/ {printf "%.1f", $2/1048576}' /proc/meminfo)
    if [[ -n "$_mem_avail_gb" ]] && awk "BEGIN{exit !($_mem_avail_gb < 6)}"; then
        echo "============================================================"
        echo "LOW MEMORY: ${_mem_avail_gb}G available; a full run wants ~6G"
        echo ""
        echo "pyright (node, ~5.4G peak) and the Angular build are what need"
        echo "it. Below this they tend to die as an OOM kill rather than a"
        echo "test failure, and can take the session with them."
        echo ""
        echo "Run the suite on a bigger machine instead — on the GRID:"
        echo "  sbatch --partition=cpu --cpus-per-task=8 --mem=32G \\"
        echo "         --time=02:00:00 --wrap 'source gridenv.sh && ./run-tests.sh'"
        echo "Or run one group at a time here (./run-tests.sh detectors),"
        echo "which skips the frontend gates unless the group is core/frontend."
        echo "============================================================"
        echo ""
    fi
fi

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

# `-n auto` does NOT mean "one worker per CPU I was given". pytest-xdist asks
# psutil for the MACHINE's physical cores and never looks at the cgroup, so a
# SLURM job with --cpus-per-task=8 on a 40-thread node starts ~20 workers, each
# importing torch. Measured (#3721): 90 oom_kill events in a 32G allocation, the
# workers dying one at a time so the run WEDGES rather than fails, and the only
# thing printed is the wall-clock cap above -- whose message then sends the
# reader hunting for a hung network step. Two runs lost 30 minutes each to it.
#
# The env var beats every detection path in xdist/plugin.py. On an unconstrained
# box the affinity mask is every core, so this is a no-op there; it only bites
# where the cgroup is narrower than the machine, which is exactly the GRID.
if [[ -z "${PYTEST_XDIST_AUTO_NUM_WORKERS:-}" ]]; then
    PYTEST_XDIST_AUTO_NUM_WORKERS=$(python -c 'import os; print(len(os.sched_getaffinity(0)))' 2>/dev/null || echo "")
    [[ -n "$PYTEST_XDIST_AUTO_NUM_WORKERS" ]] && export PYTEST_XDIST_AUTO_NUM_WORKERS
fi
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

# `slides` is the same shape: no Python tests, so pytest is skipped rather than
# asked for an empty `-m slides` selection. Unlike every other group this one
# is also a legitimate pre-push gate; see the staging note at the top and the
# guard immediately below, which is what makes that safe.
_is_slides_run=false
if [[ ${#TEST_GROUPS[@]} -eq 1 && "${TEST_GROUPS[0]}" == "slides" ]]; then
    _is_slides_run=true
    _run_pytest=false
fi

_blocked() {
    echo ""
    echo "============================================================"
    echo "TESTS BLOCKED: $1"
    echo "============================================================"
}

# Everything this branch changes relative to dev — committed, staged, unstaged
# and untracked alike. Untracked files count because a stray new .py would be
# linted by a full run and is exactly the kind of thing that should not ride
# along unchecked on a narrowed one.
#
# Every fast path below rests on this: `slides` verifies the diff is confined
# to slides/, and the markdown-only and frontend-only paths both detect and
# verify. Returns non-zero when there is no origin/dev to diff against, in
# which case no path can establish its premise and all of them decline to
# narrow anything.
_branch_changed_paths() {
    local base
    base=$(git merge-base HEAD origin/dev 2>/dev/null || true)
    [[ -z "$base" ]] && return 1
    {
        git diff --name-only "$base" HEAD
        git diff --name-only HEAD
        git ls-files --others --exclude-standard
    } | sort -u
}

# Guard on the `slides` fast path.
#
# The group's whole justification is that a change confined to slides/ cannot
# affect anything the skipped gates check. That premise is about the *diff*,
# not about intent, so it is worth verifying rather than trusting: the failure
# mode this prevents is someone believing their change is slides-only, taking
# the 8s path, and pushing an unchecked edit to real code.
#
# "Confined to slides/" means everything this branch changes relative to dev —
# committed, staged, unstaged, and untracked alike. Untracked files count
# because a stray new .py would be linted by a full run and is exactly the kind
# of thing that should not ride along unchecked.
#
# Strict on purpose: a docs/ or CLAUDE.md edit alongside the deck also fails
# this, because the doc-inventory gate can see markdown that this group skips.
# Run the full suite for a mixed change; it is the honest cost of one.
if $_is_slides_run; then
    if ! _slides_changed=$(_branch_changed_paths); then
        echo "Note: no origin/dev to diff against; skipping the slides-only guard."
    else
        _outside=$(printf '%s\n' "$_slides_changed" | grep -v '^slides/' || true)
        if [[ -n "$_slides_changed" && -n "$_outside" ]]; then
            _blocked "'slides' is a slides-only gate, but this branch changes other files"
            echo ""
            echo "The group skips pytest and every whole-repo gate, which is only"
            echo "sound when nothing outside slides/ has changed. Outside slides/:"
            echo ""
            echo "$_outside" | sed 's/^/  /'
            echo ""
            echo "Run the full suite instead:  ./run-tests.sh"
            exit 1
        fi
    fi
fi

# ---------------------------------------------------------------------------
# The markdown-only fast path.
#
# A change confined to tracked markdown is invisible to most of this script:
# pyright excludes markdown, pip-audit reads the venv, the Angular build only
# *copies* docs/user/*.md into the bundle (frontend/docs-assets is a symlink to
# it) and cannot fail on its contents, and of the Python tests only the ones
# named in tests_shared/markdown_surface.py ever open a doc. What *can* see a
# doc is the cheap-gate stage — check-docs, codespell, the doc-inventory and
# screenshot wiring snapshots, the deck preflight — so a markdown-only run
# keeps that stage whole and prunes the rest.
#
# Unlike `slides`, this engages on a bare `./run-tests.sh`. The complaint it
# answers is about the default: a session that writes one plan file pays the
# full run by running the command everybody runs, and a group name you have to
# remember would not have been typed. So the diff decides, the pruning is
# announced in full, and VTSEARCH_FULL_GATES=1 opts out.
#
# Three things keep the narrowing honest rather than optimistic:
#   - it needs a diff to reason about. No origin/dev, or no changes at all, and
#     nothing is pruned.
#   - the pytest selection is single-sourced in tests_shared/markdown_surface.py,
#     and tests_lib/meta/test_markdown_surface.py fails when a test outside that
#     selection learns to read a repo doc — the one way this could rot.
#   - naming the group (`./run-tests.sh docs`) asserts the premise instead of
#     detecting it, so it blocks on a mixed branch rather than quietly running
#     the full suite.
# ---------------------------------------------------------------------------
_is_docs_run=false
_docs_named=false
if [[ ${#TEST_GROUPS[@]} -eq 1 && "${TEST_GROUPS[0]}" == "docs" ]]; then
    _docs_named=true
fi

if $_docs_named || { $_is_full_run && [[ "${VTSEARCH_FULL_GATES:-}" != "1" ]]; }; then
    if ! _md_changed=$(_branch_changed_paths); then
        if $_docs_named; then
            echo "Note: no origin/dev to diff against; skipping the markdown-only guard."
            _is_docs_run=true
        fi
    else
        _md_outside=""
        if [[ -n "$_md_changed" ]]; then
            _md_outside=$(printf '%s\n' "$_md_changed" | grep -v '\.md$' || true)
        fi
        if $_docs_named; then
            if [[ -n "$_md_outside" ]]; then
                _blocked "'docs' is a markdown-only gate, but this branch changes other files"
                echo ""
                echo "The group narrows pytest and skips every whole-repo gate, which is"
                echo "only sound when nothing but tracked markdown has changed. Not markdown:"
                echo ""
                echo "$_md_outside" | sed 's/^/  /'
                echo ""
                echo "Run the full suite instead:  ./run-tests.sh"
                exit 1
            fi
            _is_docs_run=true
        elif [[ -n "$_md_changed" && -z "$_md_outside" ]]; then
            _is_docs_run=true
            echo "============================================================"
            echo "MARKDOWN-ONLY CHANGE: narrowing this run"
            echo ""
            echo "$_md_changed" | sed 's/^/  /'
            echo ""
            echo "Nothing but tracked markdown has changed, so this run keeps every"
            echo "cheap gate (check-docs, codespell, the doc-inventory and screenshot"
            echo "snapshots, the deck preflight) and runs the tests that read a doc,"
            echo "and skips pyright, pip-audit, the vulture whitelist check, the"
            echo "frontend build/audit/unit suite, and the rest of pytest — none of"
            echo "which can observe a markdown file."
            echo ""
            echo "Set VTSEARCH_FULL_GATES=1 to run everything anyway."
            echo "============================================================"
            echo ""
        fi
    fi
fi

_docs_targets=()
if $_is_docs_run; then
    # Nothing outside the cheap gates and the markdown surface can see a doc.
    _run_whole_repo_gates=false
    _run_frontend_check=false
    _run_frontend_unit=false
    _run_pytest=true
    # Single-sourced with the gate that proves the list is complete; see
    # tests_shared/markdown_surface.py.
    _md_surface=$(python -c 'from tests_shared.markdown_surface import MARKDOWN_TEST_SURFACE as m; print(" ".join(m))' 2>/dev/null || true)
    if [[ -z "$_md_surface" ]]; then
        _blocked "could not read MARKDOWN_TEST_SURFACE from tests_shared/markdown_surface.py"
        echo ""
        echo "The markdown-only path narrows pytest to that list, so an unreadable"
        echo "list would silently run no tests at all. Run './run-tests.sh' with"
        echo "VTSEARCH_FULL_GATES=1, and fix the import."
        exit 1
    fi
    read -r -a _docs_targets <<< "$_md_surface"
fi

# ---------------------------------------------------------------------------
# The frontend-only fast path: the markdown-only path's mirror.
#
# A change confined to frontend/ is invisible to pyright (pyrightconfig.json
# excludes that tree), to pip-audit (it reads the venv) and to the vulture
# whitelist check (it reads Python). What can see it runs in full: every cheap
# gate (codespell and check-docs read TypeScript and SCSS), the production
# build, npm audit and the Vitest suite. Of the Python tests, only the ones
# that read a file under frontend/, serve the built bundle, or ask the Flask
# client for a route the SPA catch-all answers with that bundle can observe
# the change; tests_shared/frontend_surface.py names them, and
# tests_lib/meta/test_frontend_surface.py fails when a test outside that list
# learns to reach the frontend.
#
# Same shape as the markdown path: engages on a bare `./run-tests.sh` when the
# diff says so, announces itself, and VTSEARCH_FULL_GATES=1 opts out. It needs
# a readable, non-empty diff; without one nothing is pruned. The explicit
# `frontend` group is a different, narrower thing — the build, audit and Vitest
# with no pytest — and stays the inner-loop gate for a mixed branch.
# ---------------------------------------------------------------------------
_is_frontend_only_run=false
if $_is_full_run && ! $_is_docs_run && [[ "${VTSEARCH_FULL_GATES:-}" != "1" ]]; then
    if _fe_changed=$(_branch_changed_paths) && [[ -n "$_fe_changed" ]]; then
        _fe_outside=$(printf '%s\n' "$_fe_changed" | grep -v '^frontend/' || true)
        if [[ -z "$_fe_outside" ]]; then
            _is_frontend_only_run=true
            echo "============================================================"
            echo "FRONTEND-ONLY CHANGE: narrowing this run"
            echo ""
            echo "$_fe_changed" | sed 's/^/  /'
            echo ""
            echo "Nothing outside frontend/ has changed, so this run keeps every cheap"
            echo "gate, runs the production build, npm audit and the Vitest suite in"
            echo "full, narrows pytest to the tests that can observe the frontend, and"
            echo "skips pyright, pip-audit and the vulture whitelist check — none of"
            echo "which can see a file under frontend/."
            echo ""
            echo "Set VTSEARCH_FULL_GATES=1 to run everything anyway."
            echo "============================================================"
            echo ""
        fi
    fi
fi

_frontend_targets=()
if $_is_frontend_only_run; then
    _run_whole_repo_gates=false
    _run_frontend_check=true
    _run_frontend_unit=true
    _run_pytest=true
    # Single-sourced with the gate that proves the list is complete; see
    # tests_shared/frontend_surface.py.
    _fe_surface=$(python -c 'from tests_shared.frontend_surface import FRONTEND_TEST_SURFACE as s; print(" ".join(s))' 2>/dev/null || true)
    if [[ -z "$_fe_surface" ]]; then
        _blocked "could not read FRONTEND_TEST_SURFACE from tests_shared/frontend_surface.py"
        echo ""
        echo "The frontend-only path narrows pytest to that list, so an unreadable"
        echo "list would silently run no tests at all. Run './run-tests.sh' with"
        echo "VTSEARCH_FULL_GATES=1, and fix the import."
        exit 1
    fi
    read -r -a _frontend_targets <<< "$_fe_surface"
fi

# ---------------------------------------------------------------------------
# The skip ledger.
#
# A gate that does not run must say so in the verdict, because the failure this
# answers (#4019) is not a gate that broke -- it is a gate that was never
# exercised while the banner still read `RUN PASSED (all gates green)`. A
# reviewer cannot tell that apart from a run that checked everything, which is
# the same shape as the globbing coverage gate in #4011 and the stale recipe
# docstring in #4007: a check that quietly judges less than it claims.
#
# The fast paths above do NOT record here, and neither does the frontend gates'
# diff-proven skip below. Each verifies against the diff that what it skips
# could not have seen the change, and each is named in its own verdict line --
# a proven-irrelevant gate is not an unjudged one, but it is still named.
# ---------------------------------------------------------------------------
_skipped_gates=()
_note_skipped() { _skipped_gates+=("$1"); }

# Which frontend gates run, decided up front against the branch diff by
# scripts/check-frontend-gate.py, so a frontend change in a worktree with no
# node_modules fails in the first seconds instead of after pytest, and a
# backend-only change does not spend the build and the Vitest suite re-checking
# origin/dev's frontend. An unreadable answer is a block: a gate that cannot
# say what it checked must not read as having checked everything.
#
# Naming the `frontend` group asks for the gates explicitly, so the diff does
# not decide there; only node_modules does.
_bundle_present() { [[ -f static/index.html && -f static/main.js ]]; }
_fe_proven_skip=false
_run_frontend_build=false
if $_run_frontend_check || $_run_frontend_unit; then
    if _group_named frontend; then
        if [ -d frontend/node_modules ]; then
            _run_frontend_build=true
        else
            _blocked "the 'frontend' group needs frontend/node_modules"
            echo ""
            echo "Install them:  cd frontend && npm install"
            exit 1
        fi
    else
        _fe_why=$(mktemp)
        _fe_gate_decision=$(python scripts/check-frontend-gate.py 2> "$_fe_why" || true)
        case "$_fe_gate_decision" in
            run)
                _run_frontend_build=true
                ;;
            skip)
                _fe_proven_skip=true
                _run_frontend_check=false
                _run_frontend_unit=false
                echo "Frontend gates skipped: $(tr '\n' ' ' < "$_fe_why" | sed 's/  */ /g')"
                if [ -d frontend/node_modules ] && ! _bundle_present; then
                    # A convenience, not a gate: the tests that serve the bundle
                    # would otherwise build it inside a pytest worker.
                    _run_frontend_build=true
                    echo "The production build still runs: static/ holds no bundle for the tests that serve one."
                fi
                ;;
            *)
                _blocked "the frontend gates cannot run in this worktree"
                echo ""
                cat "$_fe_why"
                echo ""
                echo "This run would otherwise have skipped the frontend build, npm audit"
                echo "and the Vitest suite and still reported every gate green (#4019)."
                rm -f "$_fe_why"
                exit 1
                ;;
        esac
        rm -f "$_fe_why"
    fi
fi

# ---------------------------------------------------------------------------
# Lanes: every gate from here on runs as a background job writing to its own
# log, in one of three kinds --
#   gate    a cheap stage-2 check; all run together and all are reported
#   gating  a heavy check pytest must wait for (pyright, the frontend build);
#           the first to fail stops the run
#   free    a heavy check pytest need not wait for; collected after pytest
# -- so that one pass of the box reports everything that is broken, and a run
# that dies takes its children with it.
# ---------------------------------------------------------------------------
_lane_log_dir=$(mktemp -d)
_ln_kind=(); _ln_name=(); _ln_pid=(); _ln_log=(); _ln_start=()
_ln_done=(); _ln_status=(); _ln_elapsed=()

_start_lane() {
    local kind="$1" name="$2"; shift 2
    local log
    log="$_lane_log_dir/$(printf '%s' "$name" | tr -c 'A-Za-z0-9' '_').log"
    # The subshell times itself into a sidecar, because a free lane is only
    # collected after pytest and the collection time says nothing about it.
    (
        set +e
        _t0=$SECONDS
        "$@"
        _st=$?
        echo "$(( SECONDS - _t0 ))" > "$log.elapsed"
        exit $_st
    ) > "$log" 2>&1 &
    _ln_kind+=("$kind"); _ln_name+=("$name"); _ln_pid+=("$!")
    _ln_log+=("$log"); _ln_start+=("$SECONDS")
}

_lane_wait_one() {
    local i="$1" st
    set +e
    wait "${_ln_pid[$i]}"
    st=$?
    set -e
    _ln_status[$i]=$st
    _ln_done[$i]=1
    if [[ -r "${_ln_log[$i]}.elapsed" ]]; then
        _ln_elapsed[$i]=$(< "${_ln_log[$i]}.elapsed")
    else
        _ln_elapsed[$i]=$(( SECONDS - _ln_start[$i] ))
    fi
}

_kill_tree() {
    local pid="$1" child
    for child in $(pgrep -P "$pid" 2>/dev/null || true); do
        _kill_tree "$child"
    done
    kill "$pid" 2>/dev/null || true
}

_kill_all_lanes() {
    local i
    for i in "${!_ln_pid[@]}"; do
        if [[ -z "${_ln_done[$i]:-}" ]] && kill -0 "${_ln_pid[$i]}" 2>/dev/null; then
            _kill_tree "${_ln_pid[$i]}"
        fi
    done
}

_cleanup_lanes() {
    _kill_all_lanes
    rm -rf "$_lane_log_dir"
}
trap _cleanup_lanes EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# Strip ANSI colour so a gate's last line reads cleanly in the per-gate report.
_last_line() { sed -r 's/\x1b\[[0-9;]*m//g' "$1" | grep -v '^[[:space:]]*$' | tail -n 1 | cut -c1-90; }

# ---------------------------------------------------------------------------
# Stage 1: the stale-tree ("phantom base") check, alone.
#
# Runs before every other gate, because a clobbered tree passes all of them by
# construction: when a branch reverts the PRs that merged just before it, the
# feature and its tests go together, so there is nothing left to fail. Five
# PRs have done exactly that (#2741, #2793, #2821, #3184 and the three
# restored in #3206); one reached `main`. Linting the wrong tree cleanly is
# not information, so this asks *whether it is the right tree* first.
#
# Compares against the working tree rather than HEAD: the damage exists
# before it is committed, which is often when tests run. Pure git, ~200ms.
#
# Two signals, because deletions are only the visible half: whole paths that
# vanished, and runs of consecutive commits the branch keeps nothing of (a
# clobber whose reverted hunks sit inside files that survive deletes nothing
# at all). See scripts/check-phantom-base.py for both and their measured
# false-positive rates.
# ---------------------------------------------------------------------------
echo "Checking for a stale tree..."
if ! python scripts/check-phantom-base.py ; then
    _blocked "branch reverts work it never touched (stale tree)"
    exit 1
fi

# ---------------------------------------------------------------------------
# Stage 2: the cheap gates, concurrently, every one to completion.
#
# Fourteen single-threaded checks that read the tree and write nothing but
# temp files, so they cannot interfere with each other. Serially they took
# 37s on a warm box (codespell ~9s, check-docs ~4s, the rest a second or two
# each, plus a cold page cache on the first run of a session); together they
# take as long as the slowest, and one pass reports every failure instead of
# the first one. Each gate is a function so its fix-it hint prints with its
# failure.
# ---------------------------------------------------------------------------
_gate_ruff_check() { ruff check . ; }
_gate_ruff_format() {
    ruff format --check . || { echo "Fix with: ruff format ."; return 1; }
}
_gate_codespell() { codespell --toml pyproject.toml && echo "codespell OK" ; }
# Documentation drift: relative links, in-page anchors, backticked repo paths,
# leaked absolute machine paths, plan-file citations anywhere in the tree, and
# broken code fences. Pure invariants against the current tree — nothing to
# re-pin.
_gate_check_docs() { python scripts/check-docs.py ; }
_gate_deptry() { python -m deptry . ; }
# OpenAPI snapshot drift: regenerate the flask-smorest spec from the live app
# and diff against the checked-in snapshot at frontend/openapi.json. The
# frontend's generated TS client is built from this snapshot, so a stale file
# means the generated client lags the real API.
_gate_openapi_drift() {
    local regen log
    regen=$(mktemp); log=$(mktemp)
    if ! python scripts/dump_openapi.py > "$regen" 2> "$log"; then
        echo "OpenAPI spec dump failed:"
        cat "$log"
        rm -f "$regen" "$log"
        return 1
    fi
    if ! diff -u frontend/openapi.json "$regen" > /dev/null; then
        echo "OpenAPI snapshot is stale. Run 'npm run regenerate-openapi-snapshot' (or"
        echo "'python scripts/dump_openapi.py > frontend/openapi.json') and commit the result."
        diff -u frontend/openapi.json "$regen" | head -80
        rm -f "$regen" "$log"
        return 1
    fi
    rm -f "$regen" "$log"
    echo "OpenAPI snapshot up to date"
}
# Generated doc-inventory drift: regenerate the registry-backed tables embedded
# in the docs (embedders, plugin families, demo datasets, ...) and fail if any
# committed region is stale. See scripts/gen-docs-inventories.py.
_gate_doc_inventories() {
    python scripts/gen-docs-inventories.py --check \
        || { echo "Run 'python scripts/gen-docs-inventories.py' and commit the result."; return 1; }
}
_gate_dockerfiles() { python scripts/check-dockerfiles.py ; }
# User-docs screenshot wiring: every manifest shot id has both theme PNGs on
# disk, and every screenshot the user docs embed maps to a manifest id. Cheap,
# browser-free (see docs/plans/user-docs-screenshots.md); the pixel-diff
# (check.sh) needs chromium and stays a manual chore.
_gate_screenshot_wiring() { python scripts/screenshots/wiring-check.py ; }
# vtscore package docs: every top-level module / sub-package of vtscore/ is
# covered by a packages/ doc, and no doc cites a file.py:NNN line anchor.
_gate_vtscore_docs() { python scripts/check-vtscore-docs.py ; }
# Extension docs: docs/EXTENDING-*.md and vtscore/docs/extending/ state the
# contract for the same plugin ABCs in their own words; this holds both to the
# class. See scripts/check-extension-docs.py.
_gate_extension_docs() { python scripts/check-extension-docs.py ; }
# Calibration script index: every .py/.sh in scripts/experiments/calibration/
# is filed under exactly one study in that directory's README (#3409).
_gate_calibration_index() { python scripts/check-calibration-index.py ; }
# Slide decks: every deck manifest names fragments that exist and figures that
# resolve. Marp only *warns* on a missing figure and still exits 0.
_gate_slide_decks() { python slides/build.py --check ; }
# Eval/app sync: the eval framework reproduces a handful of app surfaces it
# cannot call; this notices when one of them changes, so the eval default arm
# can't quietly stop being the shipped algorithm.
_gate_eval_app_sync() { python scripts/check-eval-app-sync.py ; }

echo "Running the cheap gates concurrently..."
_stage2_start=$SECONDS
_start_lane gate "ruff check" _gate_ruff_check
_start_lane gate "ruff format --check" _gate_ruff_format
_start_lane gate "codespell" _gate_codespell
_start_lane gate "documentation" _gate_check_docs
# Skipped on a `slides` run: none of these can see a deck. deptry reads
# imports, the OpenAPI and doc-inventory snapshots are generated from the
# app's registries, and the rest scan Dockerfiles / user-docs screenshots /
# vtscore. The guard above has already established that nothing outside
# slides/ changed, so there is nothing here for them to find.
if ! $_is_slides_run; then
    _start_lane gate "deptry" _gate_deptry
    _start_lane gate "OpenAPI snapshot drift" _gate_openapi_drift
    _start_lane gate "generated doc inventories" _gate_doc_inventories
    _start_lane gate "Dockerfiles" _gate_dockerfiles
    _start_lane gate "user-docs screenshot wiring" _gate_screenshot_wiring
    _start_lane gate "vtscore package docs" _gate_vtscore_docs
    _start_lane gate "extension docs" _gate_extension_docs
    _start_lane gate "calibration script index" _gate_calibration_index
    _start_lane gate "eval/app sync" _gate_eval_app_sync
fi
_start_lane gate "slide decks" _gate_slide_decks

_failed_gates=()
for _i in "${!_ln_kind[@]}"; do
    [[ "${_ln_kind[$_i]}" == gate ]] || continue
    _lane_wait_one "$_i"
    if [[ ${_ln_status[$_i]} -eq 0 ]]; then
        printf '  %-28s OK   %s\n' "${_ln_name[$_i]}" "$(_last_line "${_ln_log[$_i]}")"
    else
        printf '  %-28s FAILED\n' "${_ln_name[$_i]}"
        _failed_gates+=("$_i")
    fi
done
if [[ ${#_failed_gates[@]} -gt 0 ]]; then
    for _i in "${_failed_gates[@]}"; do
        _blocked "${_ln_name[$_i]} failed"
        tail -n 80 "${_ln_log[$_i]}"
    done
    exit 1
fi
echo "Cheap gates OK ($(( SECONDS - _stage2_start ))s)"

# ---------------------------------------------------------------------------
# Stage 3: the heavy whole-repo gates, concurrently.
#
# GATING lanes must be green before pytest starts, and the first to fail ends
# the run with its log; FREE lanes start now and are collected after pytest.
# The staging note at the top explains the split. pytest runs alone afterwards
# (stage 4) so its four workers have the box to themselves; a free lane still
# running when pytest starts simply finishes beside it, as every lane used to.
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
#   oauthlib 3.2.2     CVE-2026-49265             (same: an Ubuntu base-image
#                                                  package in dist-packages that
#                                                  nothing pip-installed requires)
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
    --ignore-vuln CVE-2026-49265
)

_lane_pip_audit() { pip-audit "${PIP_AUDIT_IGNORE[@]}"; }
# Vulture whitelist hygiene. NOT the dead-code audit itself: a vulture hit on a
# public vtscore name is not evidence of anything (out-of-tree extensions import
# names no in-repo grep can see), so the findings stay a human pre-release chore
# per docs/RELEASE.md step 1. What *is* mechanically checkable is the whitelist:
# an entry that suppresses no finding is unfalsifiable and will outlive the
# symbol it was written for. That is how the file rotted before -- 49 of 102
# entries were asserting nothing. Whole-repo scan (~14s), hence a lane.
_lane_vulture_whitelist() { python scripts/vulture-audit.py --check-whitelist; }
# Audits the whole tree, dev deps included. This used to be `--omit=dev`
# because `@angular-devkit/build-angular` dragged in the webpack toolchain
# (webpack-dev-server → sockjs/express/launch-editor), which regularly carried
# advisories with no upstream fix. That package was never used -- angular.json
# names only `@angular/build:*` builders -- so it was swapped for a direct
# `@angular/build` devDependency (#3439), taking ~400 packages and every
# unfixable advisory with it. Keep the gate wide: if a dev-only advisory ever
# has genuinely no fix, pin it in `overrides`, or -- only when no published
# version is patched -- waive that one advisory in `WAIVERS` in
# scripts/npm-audit-gate.py, rather than re-blinding the audit to the entire dev
# tree. Each waiver is pinned to the advisory's vulnerable range, so it lapses
# (and fails the gate) the moment upstream ships a fix (#4447).
_lane_npm_audit() { python scripts/npm-audit-gate.py; }
# `npm run test:ci` regenerates the API client (pretest:ci) then runs
# `ng test --no-watch`, which exits non-zero on any spec failure. Started only
# once the build lane is green: both regenerate the API client under
# frontend/src/app/generated/, and must not do so at the same time.
_lane_frontend_unit() { (cd frontend && npm run test:ci); }
# The frontend production build. Treats Angular compiler warnings (e.g.
# NG8107) and budget warnings as errors. Angular colourises its output even
# when stdout is a file, and it interleaves the escapes *inside* the marker
# (`ESC[33m▲ ESC[43;33m[ESC[43;30mWARNING…`), so the literal `▲ [WARNING]`
# never matches the raw log — that blindness let an over-budget initial bundle
# sail past this gate for months. Match against an ANSI-stripped copy instead.
_lane_frontend_build() {
    local log plain
    log=$(mktemp)
    if ! (cd frontend && npm run build:prod) > "$log" 2>&1; then
        echo "Frontend build failed:"
        cat "$log"
        rm -f "$log"
        return 1
    fi
    plain=$(mktemp)
    sed -r 's/\x1b\[[0-9;]*m//g' "$log" > "$plain"
    if grep -q '▲ \[WARNING\]' "$plain"; then
        echo "Frontend build has warnings (treated as errors):"
        grep -A 10 '▲ \[WARNING\]' "$plain"
        rm -f "$log" "$plain"
        return 1
    fi
    rm -f "$log" "$plain"
    echo "Frontend build OK"
}
# pyright, incrementally: scripts/pyright-incremental.py checks the files
# changed since the last green check plus everything that transitively imports
# them, and the whole tree when there is no record to work from (a fresh
# clone), when pyrightconfig.json, pyproject.toml or the installed packages
# changed, or when VTSEARCH_FRESH=1 asks for it. Scope is `pyrightconfig.json`.
#
# Two threads when there is memory for them, one otherwise. On a 4-vCPU box a
# full check took ~283s on one thread, ~239s on two and ~340s on four (each
# thread holds its own copy of the program, so more threads re-parse more), so
# two is the most worth having. Peak RSS was ~5.4G on one thread and ~9.6G on
# two. Below 12G available a second thread risks an OOM kill (see the memory
# check at the top), and a slow pass beats a dead one. Measured, not derived;
# re-measure before changing either number.
#
# The PYRIGHT_PYTHON_FORCE_VERSION pin keeps everyone on the same underlying
# pyright binary regardless of what the `pyright` PyPI wrapper would otherwise
# pull.
_pyright_threads=1
if [[ -r /proc/meminfo ]] \
    && awk '/^MemAvailable:/ {exit !($2 >= 12 * 1048576)}' /proc/meminfo; then
    _pyright_threads=2
fi
_lane_pyright() {
    PYRIGHT_PYTHON_FORCE_VERSION=1.1.414 python scripts/pyright-incremental.py --threads "$_pyright_threads"
}

_vitest_pending=false
if $_run_frontend_build && [ -d frontend/node_modules ]; then
    _start_lane gating "frontend build" _lane_frontend_build
    if $_run_frontend_unit; then
        _vitest_pending=true
    fi
elif $_run_frontend_unit && [ -d frontend/node_modules ]; then
    _start_lane free "frontend unit tests (Vitest)" _lane_frontend_unit
fi
if $_run_whole_repo_gates; then
    _start_lane gating "pyright ($_pyright_threads thread(s))" _lane_pyright
    _start_lane free "pip-audit" _lane_pip_audit
    _start_lane free "vulture whitelist" _lane_vulture_whitelist
fi
if $_run_frontend_check && [ -d frontend/node_modules ]; then
    _start_lane free "npm audit" _lane_npm_audit
fi

_gating_names=()
for _i in "${!_ln_kind[@]}"; do
    [[ "${_ln_kind[$_i]}" == gating ]] && _gating_names+=("${_ln_name[$_i]}")
done
if [[ ${#_gating_names[@]} -gt 0 ]]; then
    echo "Waiting for the gating lanes: $(IFS=,; echo "${_gating_names[*]}" | sed 's/,/, /g')"
fi
_pending=1
while [[ $_pending -eq 1 ]]; do
    _pending=0
    for _i in "${!_ln_kind[@]}"; do
        [[ "${_ln_kind[$_i]}" == gating && -z "${_ln_done[$_i]:-}" ]] || continue
        if kill -0 "${_ln_pid[$_i]}" 2>/dev/null; then
            _pending=1
            continue
        fi
        _lane_wait_one "$_i"
        if [[ ${_ln_status[$_i]} -ne 0 ]]; then
            _kill_all_lanes
            _blocked "${_ln_name[$_i]} failed"
            cat "${_ln_log[$_i]}"
            exit 1
        fi
        echo "  ${_ln_name[$_i]}: OK (${_ln_elapsed[$_i]}s) $(_last_line "${_ln_log[$_i]}")"
        if [[ "${_ln_name[$_i]}" == "frontend build" ]] && $_vitest_pending; then
            _start_lane free "frontend unit tests (Vitest)" _lane_frontend_unit
            _vitest_pending=false
        fi
    done
    if [[ $_pending -eq 1 ]]; then
        sleep 1
    fi
done

_free_running=()
for _i in "${!_ln_kind[@]}"; do
    if [[ "${_ln_kind[$_i]}" == free ]] && kill -0 "${_ln_pid[$_i]}" 2>/dev/null; then
        _free_running+=("${_ln_name[$_i]}")
    fi
done
if $_run_pytest && [[ ${#_free_running[@]} -gt 0 ]]; then
    echo "Still running beside the tests: $(IFS=,; echo "${_free_running[*]}" | sed 's/,/, /g')"
fi
if $_is_slides_run; then
    # Deliberately not the "a full run is the gate before pushing" notice below:
    # for a change confined to slides/ this *is* the gate. Say what was skipped
    # and why it is sound, so the claim stays auditable rather than folkloric.
    echo "Slides-only run: pytest and every whole-repo gate skipped — nothing"
    echo "outside slides/ changed, and no test or type-check reads a deck."
elif $_is_docs_run; then
    # Also not the "a full run is the gate" notice: for a change confined to
    # tracked markdown this *is* the gate. The cheap gates ran whole; what is
    # left is the tests that can actually open a doc.
    echo "Markdown-only run: every cheap gate ran; pytest is narrowed to the"
    echo "tests that read a repo doc (${_docs_targets[*]}), and pyright,"
    echo "pip-audit, the vulture whitelist check and the frontend lanes are"
    echo "skipped — none of them can see a markdown file."
elif $_is_frontend_only_run; then
    echo "Frontend-only run: every cheap gate ran, the frontend gates run in full,"
    echo "pytest is narrowed to the tests that can observe the frontend"
    echo "(${_frontend_targets[*]}), and pyright, pip-audit and the vulture"
    echo "whitelist check are skipped — none of them can see a file under frontend/."
elif ! $_run_whole_repo_gates; then
    _note_skipped "pyright"
    _note_skipped "pip-audit"
    _note_skipped "vulture whitelist"
    echo "Group run: skipping pyright, pip-audit and the vulture whitelist check."
    echo "A full './run-tests.sh' is the gate before pushing (or set"
    echo "VTSEARCH_FULL_GATES=1 to force them)."
fi

# ---------------------------------------------------------------------------
# Stage 4: pytest, in the foreground, with the box to itself.
# ---------------------------------------------------------------------------
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
    #
    # Tests that already passed on this identical working tree are deselected
    # by tests_shared/pass_cache.py (wired through both conftests) and named
    # in pytest's own summary; VTSEARCH_FRESH=1 re-runs them.
    set +e
    if $_is_docs_run; then
        # Paths, not a marker: the markdown surface is one folder plus one file
        # in another group, and no marker expression names that set. Leaving -m
        # off keeps pyproject.toml's `not gpu and not slow` default in force.
        python -m pytest "${_docs_targets[@]}" -q --tb=short --no-header -n auto --dist loadgroup ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    elif $_is_frontend_only_run; then
        # Same shape as the markdown surface.
        python -m pytest "${_frontend_targets[@]}" -q --tb=short --no-header -n auto --dist loadgroup ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    elif [[ -n "$MARKER_EXPR" ]]; then
        python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto --dist loadgroup -m "$MARKER_EXPR" ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    else
        python -m pytest tests/ tests_lib/ -q --tb=short --no-header -n auto --dist loadgroup ${COV_ARGS[@]+"${COV_ARGS[@]}"} ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
    fi
    _pytest_status=$?
    set -e
fi

# --- collect the free lanes ------------------------------------------------
_failed_lanes=()
_free_names=()
for _i in "${!_ln_kind[@]}"; do
    [[ "${_ln_kind[$_i]}" == free ]] && _free_names+=("${_ln_name[$_i]}")
done
if [[ ${#_free_names[@]} -gt 0 ]]; then
    echo ""
    echo "Collecting: $(IFS=,; echo "${_free_names[*]}" | sed 's/,/, /g')"
    for _i in "${!_ln_kind[@]}"; do
        [[ "${_ln_kind[$_i]}" == free ]] || continue
        _lane_wait_one "$_i"
        if [[ ${_ln_status[$_i]} -eq 0 ]]; then
            echo "  ${_ln_name[$_i]}: OK (${_ln_elapsed[$_i]}s)"
        else
            echo "  ${_ln_name[$_i]}: FAILED"
            _failed_lanes+=("$_i")
        fi
    done
fi

# Report every failing lane, not just the first: the point of running them
# concurrently is that one pass tells you everything that is broken.
for _i in ${_failed_lanes[@]+"${_failed_lanes[@]}"}; do
    _blocked "${_ln_name[$_i]} failed"
    tail -80 "${_ln_log[$_i]}"
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
        _verdict_parts+=("${_ln_name[$_i]}")
    done
    _joined=$(IFS=", "; echo "${_verdict_parts[*]}")
    echo "RUN FAILED: $_joined"
    echo "============================================================"
    if [[ $_pytest_status -ne 0 ]]; then
        exit $_pytest_status
    fi
    exit 1
fi
if $_is_docs_run; then
    echo "RUN PASSED (markdown-only; this is the full gate for a markdown-only change)"
elif $_is_frontend_only_run; then
    echo "RUN PASSED (frontend-only; this is the full gate for a frontend-only change)"
elif [[ ${#_skipped_gates[@]} -gt 0 ]]; then
    # Deliberately not "all gates green": everything that ran is green, and
    # these did not run. Naming them is the whole point of #4019.
    echo "RUN PASSED, EXCEPT ${#_skipped_gates[@]} GATE(S) THAT DID NOT RUN:"
    for _g in "${_skipped_gates[@]}"; do
        echo "  - $_g"
    done
    if $_fe_proven_skip; then
        echo "  (frontend build, npm audit and Vitest skipped: this branch changes no frontend/ path)"
    fi
    echo ""
    echo "Everything that ran is green. This is not the full gate."
elif $_run_pytest; then
    if $_fe_proven_skip; then
        echo "RUN PASSED (all gates green; frontend build, npm audit and Vitest skipped:"
        echo "            this branch changes no frontend/ path, so they had nothing to judge)"
    else
        echo "RUN PASSED (all gates green; pytest summary above)"
    fi
elif $_is_slides_run; then
    echo "RUN PASSED (slides-only; this is the full gate for a slides-only change)"
else
    echo "RUN PASSED (frontend-only; no Python tests in the 'frontend' group)"
fi
echo "============================================================"
exit 0
