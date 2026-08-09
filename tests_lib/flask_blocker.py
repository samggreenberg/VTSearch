"""Meta-path hook that refuses to import Flask-shaped modules.

Used by ``scripts/check-vtscore-clean.py`` (the ``./run-tests.sh
vtscore-clean`` gate) to prove the library tier runs without Flask
installed.  It lives here - next to the library-tier tests - rather than
inside the script because the blocker has to be installed in **every**
process that runs a test, and the gate runs pytest under ``xdist``:
the controller process never executes a test body, so a blocker
installed only there proves nothing.  ``tests_lib/conftest.py`` is
imported by every xdist worker, so it installs the blocker from here
whenever :data:`BLOCK_ENV_VAR` is set in the environment (which the
script does before handing off to pytest, and which subprocess workers
inherit).
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys

#: Environment variable that arms the blocker.  Set by
#: ``scripts/check-vtscore-clean.py``; inherited by xdist workers.
BLOCK_ENV_VAR = "VTSCORE_BLOCK_FLASK"

_BLOCKED_TOP_LEVEL = {"flask", "werkzeug", "flask_smorest"}


class FlaskBlocker(importlib.abc.MetaPathFinder):
    """Refuse to load Flask-shaped modules.

    Raises :class:`ImportError` with a directive pointing to the
    library-clean rule in ``vtscore/docs/architecture.md``.
    """

    def find_spec(self, fullname, path=None, target=None):  # noqa: ARG002
        root = fullname.partition(".")[0]
        if root in _BLOCKED_TOP_LEVEL:
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
            "See vtscore/docs/architecture.md Phase 1/Phase 7 for the seam policy."
        )


def install_flask_blocker() -> None:
    """Install the blocker on ``sys.meta_path`` (idempotent)."""
    if any(isinstance(finder, FlaskBlocker) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, FlaskBlocker())


def install_flask_blocker_if_armed() -> bool:
    """Install the blocker iff :data:`BLOCK_ENV_VAR` is set.

    Returns whether the blocker is now armed, so callers can report it.
    """
    if os.environ.get(BLOCK_ENV_VAR) != "1":
        return False
    install_flask_blocker()
    return True
