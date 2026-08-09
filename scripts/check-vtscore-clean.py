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

The blocker itself lives in ``tests_lib/flask_blocker.py`` and is armed
by the ``VTSCORE_BLOCK_FLASK`` environment variable this script sets.
That indirection matters: the run is parallelised with ``-n auto``, and
every test body executes in an **xdist worker subprocess**, not here.  A
blocker installed only in this controller process would let a Flask
import inside a test sail through (issue #2931).  ``tests_lib/conftest.py``
is imported by each worker and installs the blocker from the environment
variable, which the workers inherit.

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


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))

    from tests_lib.flask_blocker import BLOCK_ENV_VAR, install_flask_blocker  # noqa: PLC0415

    # Arm the blocker for every process in this run: this controller
    # (below) and each xdist worker (via tests_lib/conftest.py, which
    # inherits the environment).
    os.environ[BLOCK_ENV_VAR] = "1"
    install_flask_blocker()

    # Defer pytest import until after the blocker is in place - pytest
    # itself does NOT import Flask, but pulling it in early reduces the
    # window for accidental Flask imports.
    import pytest  # noqa: PLC0415

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
