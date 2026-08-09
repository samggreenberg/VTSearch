#!/usr/bin/env python
"""Run ``tests_lib/`` with Flask import blocked.

The ``vtscore`` library (Phase 8 of ``../vtscore/docs/architecture.md``)
must be importable without Flask installed.  Until the physical
``git mv`` to ``vtscore/`` lands, the library code still lives at
``vtsearch.<subpackage>`` paths, so we can't simulate "Flask uninstalled"
by creating a Flask-less virtualenv - Flask is a dependency of the
top-level ``vtsearch`` package.

Instead we install a meta-path import hook that *blocks* ``flask`` (and
any other Flask-shaped module) before pytest collection starts.  If any
library-candidate module imports Flask, the test session crashes with a
clear error pointing at the offending import.

The hook itself lives in :mod:`tests_lib.flask_blocker`, because it has
to be installed in *every* process that imports library code: this
controller runs pytest with ``-n auto``, and the xdist workers doing the
actual importing are fresh subprocesses with their own ``sys.meta_path``.
We therefore export ``VTSEARCH_BLOCK_FLASK=1`` (workers inherit the
environment) and ``tests_lib/conftest.py`` re-installs the hook on the
way in.

Run this script with the same CLI arguments you would pass to pytest:

    python scripts/check-vtscore-clean.py
    python scripts/check-vtscore-clean.py -k diversity_tree

The script always restricts collection to ``tests_lib/`` and the
non-gpu non-slow markers - there's no point in running app-tier tests
in a mode that bans Flask.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    from tests_lib.flask_blocker import BLOCK_ENV_VAR, install  # noqa: PLC0415

    # Block Flask here (covers this controller process) *and* tell every
    # xdist worker to do the same on its way in - workers inherit the
    # environment but not ``sys.meta_path``, and they are what import the
    # test modules and the library code under test.
    os.environ[BLOCK_ENV_VAR] = "1"
    install()

    # Defer pytest import until after the blocker is in place - pytest
    # itself does NOT import Flask, but pulling it in early reduces the
    # window for accidental Flask imports.
    import pytest  # noqa: PLC0415

    repo_root = REPO_ROOT
    args = [
        str(repo_root / "tests_lib"),
        "-q",
        "--tb=short",
        "--no-header",
        "-n",
        "auto",
        "-m",
        "not gpu and not slow",
        *sys.argv[1:],
    ]
    return pytest.main(args)


if __name__ == "__main__":
    raise SystemExit(main())
