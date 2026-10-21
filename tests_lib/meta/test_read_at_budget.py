"""A run read at a smaller click budget (#5141): what ``read_at_budget.py`` keeps and what it drops.

The deck's Past the Exit slide reads #5134's 200-click arm at a review's 150 clicks, so the cut has to keep every
row, pick and rank frame at or before the budget, nothing after it, and the side frames the review's analyzer reads.

Meta-group: the subject is repo tooling under ``scripts/``, loaded by path.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "state_of_app" / "read_at_budget.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_read_at_budget", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tmp_path: Path) -> Path:
    src = tmp_path / "src" / "results"
    (src / "cells").mkdir(parents=True)
    (src / "crops").mkdir()
    (src / "prepare_info.json").write_text("{}")
    (src / "grid_shape.json").write_text(json.dumps({"n_seeds": 1, "max_steps": 200}))
    rows = pd.DataFrame({"t": [1, 150, 151, 200, 207], "phase": ["good", "hard", "hard", "done", "check"]})
    for name in ("task_0001.csv.gz", "task_0001__picks.csv.gz", "task_0001__rankframes.csv.gz"):
        rows.to_csv(src / "cells" / name, index=False, compression="gzip")
    rows.to_csv(src / "cells" / "task_0001__sweep.csv.gz", index=False, compression="gzip")
    return src


def test_cuts_rows_picks_and_frames_at_the_budget(mod, tmp_path, monkeypatch):
    src = _run(tmp_path)
    dst = tmp_path / "dst" / "results"
    monkeypatch.setattr(sys, "argv", ["read_at_budget.py", "--src", str(src), "--dst", str(dst), "--procs", "1"])
    mod.main()
    names = sorted(p.name for p in (dst / "cells").iterdir())
    # The side frames the review's analyzer reads, and no other.
    assert names == ["task_0001.csv.gz", "task_0001__picks.csv.gz", "task_0001__rankframes.csv.gz"]
    for name in names:
        assert pd.read_csv(dst / "cells" / name)["t"].tolist() == [1, 150]
    shape = json.loads((dst / "grid_shape.json").read_text())
    assert shape["max_steps"] == 150 and shape["n_seeds"] == 1
    assert (dst / "crops").is_symlink() and (dst / "prepare_info.json").is_symlink()
