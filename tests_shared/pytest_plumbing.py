"""Pytest hook bodies shared by the two conftests.

The hooks themselves must be *defined* in a conftest for pytest to find them,
so each tier keeps a two-line ``pytest_collection_modifyitems`` /
``pytest_unconfigure`` that delegates here.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

#: Directories under a test tree that are not test groups: no marker is added
#: for items collected from them.  Each tree's own root name is added by
#: :func:`add_group_markers`.
NON_GROUP_DIRS = frozenset({"fixtures", "__pycache__"})

#: ``tests_lib/`` mirrors ``vtscore``'s package layout (#5403), so there a
#: folder names the package its tests cover rather than a group.  This maps each
#: folder, as its path under ``tests_lib/`` (``""`` for the tree's root, which
#: holds the tests of ``vtscore``'s top-level modules), to the
#: ``./run-tests.sh`` group it runs under.  The folders that are not packages
#: (``meta``, ``gpu``, ``integration``) and the top-level ``imagekit`` package
#: are groups of their own.  ``tests_lib/meta/test_lib_layout.py`` checks the
#: map against the tree and against ``vtscore/``.
LIB_GROUPS: Mapping[str, str] = {
    "": "core",
    "cli": "cli",
    "concurrency": "core",
    "config": "core",
    "converters": "converters",
    "coverage": "sorting",
    "datasets": "datasets",
    "datasets/downloader": "downloads",
    "datasource_importers": "io",
    "detectors": "detectors",
    "embedding": "core",
    "eval": "detectors",
    "exporters": "io",
    "gpu": "gpu",
    "imagekit": "imagekit",
    "integration": "integration",
    "labels": "io",
    "media": "core",
    "meta": "meta",
    "plugins": "core",
    "projection": "projection",
    "security": "core",
    "seed_importers": "io",
    "state": "core",
    "sync": "io",
    "timing": "core",
    "training": "sorting",
    "utils": "core",
}


def _folder_in_tree(item, root_dir_name: str) -> tuple[str, ...] | None:
    """The parts of ``item``'s folder below ``<rootdir>/<root_dir_name>``, or None outside it."""
    try:
        return Path(item.path).parent.relative_to(item.config.rootpath / root_dir_name).parts
    except ValueError:
        return None


def add_group_markers(items, root_dir_name: str, groups: Mapping[str, str] | None = None) -> None:
    """Auto-assign group markers from where each test file lives.

    Without ``groups`` (``tests/``) the layout is
    ``<root_dir_name>/<group>/test_*.py``: the folder name IS the group, so
    tests can be run by area (``pytest -m core``, ``pytest -m sorting``) and a
    new file automatically inherits its group from where it lives.  That kills
    the old registry-drift bug class, where a file added without an entry in a
    hand-maintained ``_TEST_GROUPS`` map was silently excluded from
    ``./run-tests.sh <group>``.

    With ``groups`` (``tests_lib/``, see :data:`LIB_GROUPS`) the folder's path
    under the tree is looked up instead.  The map is keyed by folder, not by
    file, so a new file still inherits its group from where it lives, and a
    test in a folder the map lacks stops the run with a :class:`pytest.UsageError`
    instead of silently joining no group.

    Each conftest's hook is handed every collected item, both trees' alike, so
    items outside ``<rootdir>/<root_dir_name>`` are left to the other call.
    """
    for item in items:
        parts = _folder_in_tree(item, root_dir_name)
        if parts is None or (parts and (parts[0] in NON_GROUP_DIRS or parts[0].startswith("_"))):
            continue
        if groups is None:
            if parts:
                item.add_marker(getattr(pytest.mark, parts[-1]))
            continue
        folder = "/".join(parts)
        group = groups.get(folder)
        if group is None:
            raise pytest.UsageError(
                f"{root_dir_name}/{folder}/ holds tests but has no group in LIB_GROUPS "
                f"(tests_shared/pytest_plumbing.py). {root_dir_name}/ mirrors vtscore's packages: "
                f"move the test into the folder of the package it covers, or map the new folder "
                f"to the ./run-tests.sh group it belongs to."
            )
        item.add_marker(getattr(pytest.mark, group))


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

    from tests_shared.pass_cache import pass_cache_info

    reporter = config.pluginmanager.getplugin("terminalreporter")
    cache = pass_cache_info(config)
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
        elif passed == 0 and cache["reused"]:
            # Nothing passed *this* run because every test that could pass was
            # recorded as passing on this identical tree -- a pass, not an empty
            # run. Skips and xfails are never recorded, so they still ran.
            still_ran = [f"{n} {what}" for n, what in ((skipped, "skipped"), (xfailed, "xfailed")) if n]
            tail = f"; {', '.join(still_ran)}" if still_ran else ""
            print(
                f"ALL {cache['reused']} TESTS ALREADY PASSED ON THIS TREE (nothing re-run{tail})",
                flush=True,
            )
        else:
            extra = f"{skipped} skipped"
            if xfailed:
                extra += f", {xfailed} xfailed"
            print(f"ALL {passed} TESTS PASSED ({extra}, total: {total})", flush=True)
        if cache["reused"]:
            print(
                f"Pass cache: {cache['reused']} tests not re-run (passed on this identical tree "
                f"at {cache['previous_when']}); {cache['newly_recorded']} newly recorded.",
                flush=True,
            )
        elif cache["active"] and cache["newly_recorded"]:
            print(f"Pass cache: {cache['newly_recorded']} passes recorded for this tree (none reused).", flush=True)
    else:
        status = "PASSED" if exitstatus == 0 else "FAILED"
        print(f"TESTS {status} (exit code {exitstatus}; reporter unavailable)", flush=True)
    print("=" * 60, flush=True)

    # The one atexit job the force-exit below must not skip.
    from tests_shared.data_dir import release_data_dir

    release_data_dir()

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exitstatus)
