"""The ladder's F-beta curves read the typed query at each preset's own line (#4625).

Since #4603 the app draws the typed query's line per preset, and ``text_baseline.py``
records it (``text_line_*_<beta>``).  ``analyze_progression_4184.py`` anchored every
F-beta curve's pre-hand-over clicks at the beta-blind cut instead, so Follow Suit
(slide 32) started near 0 at beta 1/4 and 1 while the app shows ~0.48 and ~0.37.
Pinned here: each curve reads its own preset's line, and a baseline from before
#4603 still reads the blind cut.

Meta-group: the subject is repo tooling under ``scripts/``, loaded by path with
``common`` stubbed inert, as ``test_analyzer_objective.py`` loads its analyzers.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"


@pytest.fixture(scope="module")
def prog():
    """``analyze_progression_4184`` imported by path, ``common`` stubbed while it loads."""
    stub: Any = types.ModuleType("common")
    stub.setup_env = lambda: None
    stub.log = lambda _msg: None
    stub.RESULTS = Path(".")
    saved = sys.modules.get("common")
    sys.modules["common"] = stub
    sys.path.insert(0, str(_CALIB))
    try:
        spec = importlib.util.spec_from_file_location("_prog_text_line_test", _CALIB / "analyze_progression_4184.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(_CALIB))
        if saved is None:
            sys.modules.pop("common", None)
        else:
            sys.modules["common"] = saved


def _baseline(**cols: float) -> pd.DataFrame:
    return pd.DataFrame([{"dataset": "d", "embedder": "e", "category": "c", "seed": 0, **cols}])


def test_each_curve_reads_its_own_presets_line(prog) -> None:
    lines = {}
    for tag, (p, r) in {"b025": (0.8, 0.2), "b1": (0.5, 0.5), "b4": (0.1, 0.9)}.items():
        lines[f"text_line_precision_{tag}"], lines[f"text_line_recall_{tag}"] = p, r
    base = _baseline(text_precision=0.01, text_recall=1.0, **lines)
    text_f, used = prog.text_fbeta_anchors(base)
    key = ("d", "e", "c", 0)
    assert text_f["f025"][key] == pytest.approx(float(prog.fbeta(0.8, 0.2, 0.25)))
    assert text_f["f1"][key] == pytest.approx(0.5)
    assert text_f["f4"][key] == pytest.approx(float(prog.fbeta(0.1, 0.9, 4.0)))
    assert all("preset's line" in how for how in used.values())


def test_a_baseline_from_before_4603_reads_the_blind_cut(prog) -> None:
    base = _baseline(text_precision=0.5, text_recall=0.5)
    text_f, used = prog.text_fbeta_anchors(base)
    assert text_f["f1"][("d", "e", "c", 0)] == pytest.approx(0.5)
    assert all("blind" in how for how in used.values())
