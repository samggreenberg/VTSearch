"""The `pile/` readers after the deletion: record-backed reads, and guarded ones (#4006).

#4004 fixed the `calibration/` half.  This pins the `pile/` half, whose defect is
the same but whose resolution is not uniform: a read of an artefact that IS in
#3729's committed record repoints at the record, a read of deleted study output
gets a guard, and a write is left alone because the writer creates its own
directory.  The tests below are the per-access split, made checkable.

The reason this is worth a test at all: nothing exercised these paths, so nobody
noticed they had gone stale.  A default that names a file which must exist is
exactly the kind of claim a suite can hold for you.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

PILE = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "pile"
CALIB = PILE.parent / "calibration"
RECORD = PILE / "human_record"

#: Every record file a `pile/` default now names, and the reader that names it.
#: A rename in the record breaks these before it breaks someone's run.
RECORD_DEFAULTS = {
    "check_review_coverage.py": "WORK__verdicts_20260820b.json",
    "shipped_pool_error.py": "WORK3588__slates__Table_Objects__manifest.csv",
}

#: #4012's third disposition, added after #4006's two: a reader whose input is
#: gone but whose ANSWER is committed elsewhere is restored against that copy,
#: not guarded. `negative_pass_strata.py` left RECORD_DEFAULTS at the same time:
#: it used to open the stratum manifest for a fact `verdicts.csv` already
#: carries, and one input beats two for the same thing.
CSV_BACKED = ("negative_pass_strata.py", "score_negative_pass.py")

#: Readers that must refuse to run when their input is gone, rather than dying on
#: whatever `open()` came first.
GUARDED = (
    "import_slates.py",
    "make_audit_pass.py",
    "silence_rate.py",
    "shipped_pool_error.py",
    "make_belowcut.py",
)

#: Writers from #4006's table. They `mkdir -p` their output, so the deletion
#: costs them nothing and a guard would be wrong -- the same reasoning that left
#: the launchers alone in #4004.
#: `make_contact_sheets.py` is NOT here, though #4006's table lists it as a
#: writer: it reads `{slates}/<class>/manifest.csv`. See this PR's description.
WRITERS = (
    "make_positive_slate.py",
    "make_negative_slate.py",
    "make_class_slate.py",
    "coco_class_sheets.py",
)


def _load_study_paths():
    spec = importlib.util.spec_from_file_location("_calib_study_paths_pile", CALIB / "study_paths.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CALIB))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CALIB))
    return module


def test_an_existing_file_passes_through_unchanged(tmp_path: Path) -> None:
    f = tmp_path / "queue.jsonl"
    f.write_text("{}\n")
    study_paths = _load_study_paths()
    assert study_paths.require_study_file(f, "--queue") == f
    assert study_paths.require_study_file(str(f), "--queue") == f


def test_a_missing_file_names_the_path_the_record_and_its_producer(tmp_path: Path, capsys) -> None:
    study_paths = _load_study_paths()
    missing = tmp_path / "annotation_queue.jsonl"
    with pytest.raises(SystemExit) as exc:
        study_paths.require_study_file(missing, "--queue", "annotation_queue.py")
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert str(missing) in err
    assert "2026-09-18" in err and "#4001" in err
    # The producer is the recovery path, so the message has to name it.
    assert "annotation_queue.py" in err
    assert "--queue" in err


def test_a_missing_file_without_a_producer_still_explains(tmp_path: Path, capsys) -> None:
    study_paths = _load_study_paths()
    with pytest.raises(SystemExit):
        study_paths.require_study_file(tmp_path / "gone.json", "--controls")
    err = capsys.readouterr().err
    assert "Re-run the study" in err and "--controls" in err


def test_a_directory_is_not_a_file(tmp_path: Path) -> None:
    study_paths = _load_study_paths()
    with pytest.raises(SystemExit):
        study_paths.require_study_file(tmp_path, "--queue")


@pytest.mark.parametrize(("script", "record_file"), sorted(RECORD_DEFAULTS.items()))
def test_every_record_backed_default_names_a_file_that_exists(script: str, record_file: str) -> None:
    """The default points into the repository, and the file is really there."""
    assert record_file in (PILE / script).read_text(), f"{script} no longer names {record_file}"
    assert (RECORD / record_file).is_file(), f"{record_file} is missing from the committed record"


@pytest.mark.parametrize("script", sorted(RECORD_DEFAULTS))
def test_a_record_backed_read_does_not_go_through_the_deleted_dir(script: str) -> None:
    """The point of the repoint: the deleted path is no longer on the read path."""
    text = (PILE / script).read_text()
    for line in text.splitlines():
        if line.lstrip().startswith("#") or "require_study" in line:
            continue
        # A read of the record file must not be spelled against the old base.
        assert not re.search(r"(vgscale-3156|classes-3588)[^\n]*verdicts_20260820b", line), line
        assert not re.search(r"(vgscale-3156|classes-3588)[^\n]*Table_Objects", line), line


@pytest.mark.parametrize("script", GUARDED)
def test_every_guarded_reader_imports_the_shared_guard(script: str) -> None:
    """One guard, reused -- not a second implementation per directory."""
    assert "from study_paths import require_study" in (PILE / script).read_text(), script


#: The third case, and the one the per-access rule does not cover by itself: a
#: read whose absence costs the script PART of its job. `check_review_coverage`
#: globs the triage sheet indexes, so a missing dir dropped one population and
#: the gate passed on the other two without saying it had judged less. Exiting 2
#: would be worse -- it would retire a gate that still works -- so these say so
#: in the output instead.
DEGRADES = ("check_review_coverage.py",)


@pytest.mark.parametrize("script", DEGRADES)
def test_a_partial_read_is_reported_rather_than_fatal_or_silent(script: str) -> None:
    text = (PILE / script).read_text()
    assert "require_study" not in text, f"{script} still works without the deleted dir; it must not exit"
    assert "NOTE:" in text and "#4001" in text, f"{script} must say which population it could not judge"


@pytest.mark.parametrize("script", CSV_BACKED)
def test_a_restored_reader_reads_the_committed_csv_and_is_not_guarded(script: str) -> None:
    """Restored, not retired: the judgements survive, so the script must run."""
    text = (PILE / script).read_text()
    assert "negative_pass_verdicts" in text, f"{script} must read the committed verdicts"
    assert "require_study" not in text, f"{script} works from the repository; it must not exit 2"
    # Naming the deleted dir in prose is fine and useful; opening it is not.
    assert "classes-3588" not in text, f"{script} still reads the deleted study dir"


@pytest.mark.parametrize("script", WRITERS)
def test_a_writer_is_left_alone(script: str) -> None:
    """A writer creates its own dir, so guarding it would break the re-run."""
    path = PILE / script
    if not path.exists():  # pragma: no cover - the table is from #4006, not a contract
        pytest.skip(f"{script} is not in this checkout")
    assert "require_study" not in path.read_text(), f"{script} is a writer and must not be guarded"
