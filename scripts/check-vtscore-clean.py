#!/usr/bin/env python
"""Run ``tests_lib/`` with Flask import blocked.

The ``vtscore`` library (Phase 8 of ``../vtscore/docs/architecture.md``)
must be importable without Flask installed.  Until the physical
``git mv`` to ``vtscore/`` lands, the library code still lives at
``vtsearch.<subpackage>`` paths, so we can't simulate "Flask uninstalled"
by creating a Flask-less virtualenv — Flask is a dependency of the
top-level ``vtsearch`` package.

Instead we install a meta-path import hook that *blocks* ``flask`` (and
any other Flask-shaped module) before pytest collection starts.  If any
library-candidate module imports Flask, the test session crashes with a
clear error pointing at the offending import.

Run this script with the same CLI arguments you would pass to pytest:

    python scripts/check-vtscore-clean.py
    python scripts/check-vtscore-clean.py -k diversity_tree

The script always restricts collection to ``tests_lib/`` and the
non-gpu non-slow markers — there's no point in running app-tier tests
in a mode that bans Flask.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path


_BLOCKED_TOP_LEVEL = {"flask", "werkzeug", "flask_smorest"}


class _FlaskBlocker(importlib.abc.MetaPathFinder):
    """Refuse to load Flask-shaped modules.

    Raises :class:`ImportError` with a directive pointing to the
    library-clean rule in ``../vtscore/docs/architecture.md``.
    """

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        root = fullname.partition(".")[0]
        if root in _BLOCKED_TOP_LEVEL:
            # Returning a spec whose loader raises makes the import look
            # like a real failure, which is what we want — pytest will
            # surface the importing module's path in the traceback.
            return importlib.machinery.ModuleSpec(fullname, _FlaskBlockerLoader(fullname))
        return None


class _FlaskBlockerLoader(importlib.abc.Loader):
    def __init__(self, fullname: str) -> None:
        self._fullname = fullname

    def create_module(self, spec):  # noqa: ARG002
        return None

    def exec_module(self, module):  # noqa: ARG002
        raise ImportError(
            f"Import of {self._fullname!r} is blocked in vtscore-clean test mode. "
            "Library-candidate code (tests_lib/ targets) must not import Flask. "
            "See ../vtscore/docs/architecture.md Phase 1/Phase 7 for the seam policy."
        )


def main() -> int:
    sys.meta_path.insert(0, _FlaskBlocker())

    # Defer pytest import until after the blocker is in place — pytest
    # itself does NOT import Flask, but pulling it in early reduces the
    # window for accidental Flask imports.
    import pytest  # noqa: PLC0415

    repo_root = Path(__file__).resolve().parent.parent
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
