"""The stopping derivation's planted-answer self-test, run as part of the suite (#3560).

``scripts/experiments/calibration/stopping.py`` turns the ``phase`` column into
where the app's stopping rules fired and what the detector scored there, and
its traps (first fire vs sustained, censoring, the margins' sign convention,
the run's own best, a prompted spot check hiding the phase) are each pinned by
``selftest_stopping.py`` with an exact planted answer.  That script is the
module's test, but nothing ran it: the State of the App analyzer now reads the
module on every review, so a regression there would reach a report.  This runs
it in-process.

Meta-group: the subject is repo tooling under ``scripts/``, loaded by path with
``common`` stubbed inert, as ``test_analyzer_objective.py`` loads ``stopping``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"


def test_the_stopping_selftest_passes(capsys) -> None:
    stub: Any = types.ModuleType("common")
    stub.setup_env = lambda: None
    saved = {name: sys.modules.get(name) for name in ("common", "stopping", "objective")}
    saved_path = list(sys.path)
    sys.modules["common"] = stub
    sys.path.insert(0, str(_CALIB))
    try:
        spec = importlib.util.spec_from_file_location("_selftest_stopping", _CALIB / "selftest_stopping.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        rc = module.main()
    finally:
        # The self-test puts its own directory on sys.path too.
        sys.path[:] = saved_path
        # The self-test imports `stopping` and `objective` by bare name; leave
        # sys.modules as it was so another test's by-path load is not shadowed.
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    out = capsys.readouterr().out
    failed = [line.strip() for line in out.splitlines() if line.strip().startswith("FAIL")]
    assert rc == 0, "selftest_stopping.py failed:\n" + "\n".join(failed)
