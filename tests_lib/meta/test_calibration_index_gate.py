"""The calibration script index gate itself: ``scripts/check-calibration-index.py``.

The gate keeps ``scripts/experiments/calibration/README.md``'s study -> files
index total (#3409) and the directory closed to new study files (#5406).  The
second half is the one a session meets when it reaches for the old habit, so
pin that it fires on a new study file, stays quiet on a new shared one, and
keeps its grandfather list exact as files leave.

Meta-group: the subject is repo tooling under ``scripts/``, loaded by path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check-calibration-index.py"


def _load_gate():
    spec = importlib.util.spec_from_file_location("_calibration_index_gate", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gate = _load_gate()


def readme(shared: str, studies: str) -> str:
    return (
        "# Calibration\n\n"
        f"{gate.BEGIN}\n\n"
        "### The shared layer\n\n"
        "| Files | What they are |\n|---|---|\n"
        f"{shared}\n\n"
        f"{gate.STUDIES_HEADING}\n\n"
        "| Study | Files |\n|---|---|\n"
        f"{studies}\n"
        f"{gate.END}\n"
    )


SHARED = "| `common.py` | Env setup. |"
STUDY = "| **#2836 Cut rule** | `analyze_cut.py`, `launch_cut.sh` |"
ON_DISK = {"common.py", "analyze_cut.py", "launch_cut.sh"}


@pytest.fixture
def grandfathered(monkeypatch):
    monkeypatch.setattr(gate, "GRANDFATHERED", frozenset({"analyze_cut.py", "launch_cut.sh"}))


def test_the_repo_passes() -> None:
    assert gate.main() == 0


@pytest.mark.usefixtures("grandfathered")
class TestClosedToNewStudies:
    def test_the_grandfathered_index_passes(self) -> None:
        assert gate.check(readme(SHARED, STUDY), ON_DISK) == []

    def test_a_new_study_file_fails_by_name(self) -> None:
        studies = STUDY + "\n| **#9999 New study** | `analyze_new_9999.py` |"
        problems = gate.check(readme(SHARED, studies), ON_DISK | {"analyze_new_9999.py"})
        assert len(problems) == 1
        assert "#5406" in problems[0]
        assert "analyze_new_9999.py" in problems[0]

    def test_a_new_file_added_to_an_old_study_fails(self) -> None:
        studies = "| **#2836 Cut rule** | `analyze_cut.py`, `launch_cut.sh`, `figures_cut.py` |"
        problems = gate.check(readme(SHARED, studies), ON_DISK | {"figures_cut.py"})
        assert len(problems) == 1
        assert "figures_cut.py" in problems[0]

    def test_a_new_shared_file_passes(self) -> None:
        shared = SHARED + "\n| `new_loader.py` | A loader every study reads. |"
        assert gate.check(readme(shared, STUDY), ON_DISK | {"new_loader.py"}) == []

    def test_an_unlisted_file_names_both_homes(self) -> None:
        problems = gate.check(readme(SHARED, STUDY), ON_DISK | {"stray.py"})
        assert len(problems) == 1
        assert "stray.py" in problems[0]
        assert "shared layer" in problems[0] and "scripts/experiments/<study>/" in problems[0]

    def test_a_deleted_study_file_must_leave_the_list(self) -> None:
        studies = "| **#2836 Cut rule** | `analyze_cut.py` |"
        problems = gate.check(readme(SHARED, studies), ON_DISK - {"launch_cut.sh"})
        assert len(problems) == 1
        assert "GRANDFATHERED" in problems[0] and "launch_cut.sh" in problems[0]

    def test_a_promoted_study_file_must_leave_the_list(self) -> None:
        shared = SHARED + "\n| `analyze_cut.py` | The cut decomposition, now shared. |"
        studies = "| **#2836 Cut rule** | `launch_cut.sh` |"
        problems = gate.check(readme(shared, studies), ON_DISK)
        assert len(problems) == 1
        assert "GRANDFATHERED" in problems[0] and "analyze_cut.py" in problems[0]

    def test_a_parenthesised_cross_reference_is_not_a_filing(self) -> None:
        studies = STUDY + "\n| **#2883 Transfer** | (reuses `analyze_cut.py`'s decomposition) |"
        assert gate.check(readme(SHARED, studies), ON_DISK) == []


def test_an_index_without_the_studies_heading_fails() -> None:
    text = readme(SHARED, STUDY).replace(gate.STUDIES_HEADING, "### Studies")
    problems = gate.check(text, ON_DISK)
    assert len(problems) == 1
    assert gate.STUDIES_HEADING in problems[0]
