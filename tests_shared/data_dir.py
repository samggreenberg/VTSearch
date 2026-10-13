"""Give each pytest session a throwaway ``VTSEARCH_DATA_DIR`` (issue #4706).

``vtscore/config/paths.py`` resolves ``DATA_DIR`` from ``VTSEARCH_DATA_DIR`` once,
at import, and falls back to the checkout's ``data/``.  So a test that writes
under ``config.DATA_DIR`` without a redirect of its own writes wherever the
*environment* says the app's data lives.  On a laptop that is the developer's
own ``data/``.  On the HLTCOE GRID it is the live app's data: ``~/.bashrc``
exports the variable and ``sbatch`` passes it to every suite job.  Suite runs
there left 299 fixture ``autodetect_results_*.json``, 152
``local_uploads/local_folder_*`` dirs and ``staging/*.pkl`` pickles in the live
dir, and rewrote its ``user_settings.json``.

:func:`isolate_data_dir` closes that for every launcher at once.  Both conftests
call it before anything imports ``vtscore.config``.  It ignores what the
variable held and points it at a fresh temp dir, which every xdist worker and
every subprocess a test starts then inherits.  The suite needs nothing from
the old dir: a full GRID run against an empty one passed (job 933309).

The model cache is the one exception.  It is a download cache (~4 GB), not app
state, so when the dir the app would read models from exists,
``VTSEARCH_MODELS_DIR`` keeps naming it.  Otherwise ``./run-tests.sh gpu`` would
download every model again on each run.  When it does not exist, models go
under the session's dir like everything else, so the guard never creates
anything in the inherited dir.

Standard library only: this runs before ``vtscore`` may be imported.
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

#: Holds the isolated dir once a session has one.  When it matches
#: ``VTSEARCH_DATA_DIR``, this process inherited that session: it is an xdist
#: worker, the second conftest of a two-tree run, or a subprocess a test started.
#: Only the process that made the dir removes it.
MARKER = "VTSEARCH_TEST_DATA_DIR"

# ``paths.py``'s own fallback, restated because importing it is what this
# module has to precede.
_CHECKOUT_DATA = Path(__file__).resolve().parents[1] / "data"

# The dir this process made, if it made one.
_owned: Path | None = None


def isolate_data_dir() -> Path:
    """Point ``VTSEARCH_DATA_DIR`` at this session's temp dir and return it.

    Raises :class:`RuntimeError` when ``vtscore.config`` was already imported
    with a different ``DATA_DIR``: that is a conftest calling this too late, and
    the redirect would not reach the code under test.
    """
    global _owned
    current = os.environ.get("VTSEARCH_DATA_DIR")
    # A marker whose dir is gone was exported from a session that has ended.
    if current and os.environ.get(MARKER) == current and Path(current).is_dir():
        path = Path(current)
    else:
        models = Path(current) / "models" if current else _CHECKOUT_DATA / "models"
        if models.is_dir():
            os.environ.setdefault("VTSEARCH_MODELS_DIR", str(models))
        path = _owned = Path(tempfile.mkdtemp(prefix="vtsearch-test-data-"))
        os.environ["VTSEARCH_DATA_DIR"] = os.environ[MARKER] = str(path)
        atexit.register(release_data_dir)

    paths = sys.modules.get("vtscore.config.paths")
    if paths is not None and Path(paths.DATA_DIR) != path:
        raise RuntimeError(
            f"vtscore.config was imported before isolate_data_dir(), so DATA_DIR is "
            f"{paths.DATA_DIR} rather than the test session's {path}. Call it at the "
            f"top of the conftest, before any vtscore import."
        )
    return path


def release_data_dir() -> None:
    """Remove the session's data dir, if this process is the one that made it.

    Registered with :mod:`atexit`, and also called by
    :func:`tests_shared.pytest_plumbing.print_summary_and_exit`, because that
    ends a pytest run with ``os._exit()``, which skips atexit handlers.  A no-op
    in an xdist worker or a test's subprocess, which inherited the dir.
    """
    global _owned
    if _owned is not None:
        shutil.rmtree(_owned, ignore_errors=True)
        _owned = None
