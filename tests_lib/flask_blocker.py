"""Meta-path import hook that bans Flask from the library-tier test run.

The ``vtscore`` library (Phase 8 of ``../vtscore/docs/architecture.md``)
must be importable without Flask installed.  Because the library code
still ships inside the ``vtsearch`` distribution, we can't simulate
"Flask uninstalled" with a Flask-less virtualenv; instead
``scripts/check-vtscore-clean.py`` installs the hook below, which makes
``import flask`` (and friends) raise :class:`ImportError`.

**Why this lives here and not in the script.**  The gate runs pytest
with ``-n auto``: xdist workers are fresh ``execnet`` subprocesses that
inherit the parent's *environment* but not its ``sys.meta_path``, and
the workers are what import every test module and all the library code
underneath.  A blocker installed only in the controller process
therefore covers almost nothing.  ``tests_lib/conftest.py`` is imported
inside every worker before any test module, so it calls
:func:`install_if_requested` at the very top of the file and the
controller merely sets :data:`BLOCK_ENV_VAR` for the workers to see.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys


#: Environment variable the gate sets so xdist workers re-install the hook.
BLOCK_ENV_VAR = "VTSEARCH_BLOCK_FLASK"

BLOCKED_TOP_LEVEL = frozenset({"flask", "werkzeug", "flask_smorest"})


class FlaskBlocker(importlib.abc.MetaPathFinder):
    """Refuse to load Flask-shaped modules.

    Raises :class:`ImportError` with a directive pointing to the
    library-clean rule in ``../vtscore/docs/architecture.md``.
    """

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        root = fullname.partition(".")[0]
        if root in BLOCKED_TOP_LEVEL:
            # Returning a spec whose loader raises makes the import look
            # like a real failure, which is what we want - pytest will
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


def is_installed() -> bool:
    """True when a :class:`FlaskBlocker` is already on ``sys.meta_path``."""
    return any(isinstance(finder, FlaskBlocker) for finder in sys.meta_path)


def install() -> None:
    """Put the blocker at the front of ``sys.meta_path`` (idempotent).

    Any Flask-shaped module that slipped into ``sys.modules`` before the
    hook went up is evicted: a cached module is served straight out of
    ``sys.modules`` without consulting ``sys.meta_path``, so leaving it
    there would punch a silent hole in the gate.
    """
    for name in list(sys.modules):
        if name.partition(".")[0] in BLOCKED_TOP_LEVEL:
            del sys.modules[name]
    if not is_installed():
        sys.meta_path.insert(0, FlaskBlocker())


def install_if_requested() -> bool:
    """Install the blocker iff :data:`BLOCK_ENV_VAR` is set in the environment.

    Returns whether the blocker is active, so callers (conftest, tests)
    can branch on it.
    """
    if os.environ.get(BLOCK_ENV_VAR):
        install()
        return True
    return False
