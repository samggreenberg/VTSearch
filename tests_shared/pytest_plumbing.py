"""Pytest hook bodies shared by the two conftests.

The hooks themselves must be *defined* in a conftest for pytest to find them,
so each tier keeps a two-line ``pytest_collection_modifyitems`` /
``pytest_unconfigure`` that delegates here.
"""

from __future__ import annotations

import pytest

#: Directories under a test tree that are not test groups: no marker is added
#: for items collected from them.  Each tree's own root name is added by
#: :func:`add_group_markers`.
NON_GROUP_DIRS = frozenset({"fixtures", "__pycache__"})


def add_group_markers(items, root_dir_name: str) -> None:
    """Auto-assign group markers from each test file's parent directory.

    Layout: ``<root_dir_name>/<group>/test_*.py``; the folder name IS the group,
    so tests can be run by area (``pytest -m core``, ``pytest -m sorting``) and
    a new file automatically inherits its group from where it lives.  That
    kills the old registry-drift bug class, where a file added without an entry
    in a hand-maintained ``_TEST_GROUPS`` map was silently excluded from
    ``./run-tests.sh <group>``.
    """
    skip = NON_GROUP_DIRS | {root_dir_name}
    for item in items:
        parent = item.fspath.dirpath().basename
        if parent and parent not in skip and not parent.startswith("_"):
            item.add_marker(getattr(pytest.mark, parent))


def print_summary_and_exit(config, exitstatus: int) -> None:
    """Print the PASS/FAIL summary, then force-exit the interpreter.

    PyTorch, OpenMP, numba (via librosa) and other native libraries spin up C++
    thread pools that sometimes call ``std::terminate()`` during interpreter
    shutdown.  That produces "terminate called without an active exception" and
    exit code 134 even though every test passed.  ``os._exit()`` skips the
    normal teardown (atexit handlers, C++ static destructors) so the
    problematic cleanup never runs.

    Callers invoke this from ``pytest_unconfigure`` (the very last hook) rather
    than ``pytest_sessionfinish``, so ``pytest_terminal_summary`` still runs
    first and failure tracebacks are fully printed before the force-exit.
    """
    import os
    import sys

    reporter = config.pluginmanager.getplugin("terminalreporter")
    print("", flush=True)
    print("=" * 60, flush=True)
    if reporter:
        passed = len(reporter.stats.get("passed", []))
        failed = len(reporter.stats.get("failed", []))
        errors = len(reporter.stats.get("error", []))
        skipped = len(reporter.stats.get("skipped", []))
        xfailed = len(reporter.stats.get("xfailed", []))
        total = passed + failed + errors + skipped

        if failed or errors:
            parts = [f"{failed} failed", f"{errors} errors", f"{passed} passed", f"{skipped} skipped"]
            if xfailed:
                parts.append(f"{xfailed} xfailed")
            print(f"TESTS FAILED: {', '.join(parts)} (total: {total})", flush=True)
        else:
            extra = f"{skipped} skipped"
            if xfailed:
                extra += f", {xfailed} xfailed"
            print(f"ALL {passed} TESTS PASSED ({extra}, total: {total})", flush=True)
    else:
        status = "PASSED" if exitstatus == 0 else "FAILED"
        print(f"TESTS {status} (exit code {exitstatus}; reporter unavailable)", flush=True)
    print("=" * 60, flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exitstatus)
