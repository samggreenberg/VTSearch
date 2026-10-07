"""The viewer's and the curves' planted-answer self-tests, run as part of the suite (#4624).

``scripts/experiments/calibration/viewer.py`` builds the interactive page every
simulated-user study links from its report, and ``curves.py`` draws the pair of
PNGs beside it.  Each has a self-test with exact planted answers
(``selftest_viewer.py``, ``selftest_curves.py``), and until #4624 nothing ran
them: a run inside a spot check round dropped out of both means between rounds,
and a review's viewer dipped at the end of every session as the weak runs came
back.  The carry that fixes it is pinned by those self-tests, so they run here.

Meta-group: the subject is repo tooling under ``scripts/``, loaded by path with
``common`` stubbed inert, as ``test_calibration_stopping.py`` loads its self-test.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"

#: What the self-tests import by bare name; left as they were afterwards so
#: another test's by-path load is not shadowed.
_BARE = ("common", "viewer", "curves", "objective")


def _run_selftest(name: str, capsys) -> None:
    stub: Any = types.ModuleType("common")
    stub.setup_env = lambda: None
    stub.log = lambda _msg: None
    stub.RESULTS = Path(".")
    saved = {mod: sys.modules.get(mod) for mod in _BARE}
    saved_path = list(sys.path)
    sys.modules["common"] = stub
    sys.path.insert(0, str(_CALIB))
    try:
        spec = importlib.util.spec_from_file_location(f"_{name}", _CALIB / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rc = module.main()
    finally:
        # The self-tests put their own directory on sys.path too.
        sys.path[:] = saved_path
        for mod, was in saved.items():
            if was is None:
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = was
    out = capsys.readouterr().out
    failed = [line.strip() for line in out.splitlines() if line.strip().startswith("FAIL")]
    assert rc == 0, f"{name}.py failed:\n" + "\n".join(failed)


def test_the_viewer_selftest_passes(capsys) -> None:
    _run_selftest("selftest_viewer", capsys)


def test_the_curves_selftest_passes(capsys) -> None:
    pytest.importorskip("matplotlib")
    _run_selftest("selftest_curves", capsys)
