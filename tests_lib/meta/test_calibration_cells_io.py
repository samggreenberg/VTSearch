"""The calibration analyzers must read their cells through one loader, and
that loader must describe what it dropped in one vocabulary.

Checks over ``scripts/experiments/calibration/`` — mostly static text, plus a
handful that load ``_cells_io`` itself by path and run both loaders over a
four-file fixture.  Nothing here imports an *analyzer* (they need matplotlib
and a real results tree), and nothing here tests shipped
``vtsearch``/``vtscore`` behaviour, which is why it lives in the ``meta``
group.

The failure this guards is the one that produced #3407 twice.  ``run_cells.py``
writes one **main** metric frame per cell (``task_NNNN.csv``) and five **side**
frames beside it (``task_NNNN__picks.csv`` and friends), which are separate
long-format tables with their own columns.  A bare ``glob("task_*.csv")`` matches
both, and concatenating a side frame into the main frame does not raise: the
shared identity columns line up, every metric column lands as NaN, and every
``groupby`` denominator moves.  The number still looks like a number.

Two independent guards, because the two ways in are independent:

* **Nothing globs the cells itself.**  ``bench_cells.py`` carried a private
  three-of-five copy of the exclusion list, so four bench analyzers read the
  #3267 pick log into their metric frames while ``_cells_io``'s docstring
  explained why that could not happen.
* **The side-frame registry matches the runner.**  ``SIDE_FRAME_SUFFIXES`` is no
  longer what excludes side frames (``main_frame_files`` excludes on the ``__``
  structurally), but ``side_frame_files`` still reads it, so a frame missing from
  it is a frame no analyzer can ask for by name.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

CALIB = Path(__file__).resolve().parents[2] / "scripts" / "experiments" / "calibration"

#: The modules allowed to glob the cells.  ``_cells_paths`` owns the rule and is
#: deliberately import-free (no pandas) so the csv-and-stdlib figure scripts can
#: share it; ``_cells_io`` re-exports it beside the pandas-side reader.
CELLS_IO = ("_cells_paths.py", "_cells_io.py")

#: Scripts that fabricate cells rather than read a run's: a selftest plants a
#: known answer in files it writes itself, so its globs are over its own fixture
#: and are part of what it is asserting.
_FIXTURE_WRITERS = re.compile(r"^selftest_")

#: A ``task_*`` glob of any spelling.  Matched on the *pattern literal* rather
#: than on the call, because the call has three shapes in this directory alone
#: (``d.glob("task_*.csv")``, ``glob.glob(str(d / "task_*.csv"))``, and an
#: f-string variant) and a check that only knows two of them is the same kind of
#: incomplete list as the one that caused #3407.
_TASK_GLOB = re.compile(r"""["']task_\*""")


def _scripts() -> list[Path]:
    return sorted(p for p in CALIB.glob("*.py") if p.name not in CELLS_IO and not _FIXTURE_WRITERS.match(p.name))


def _shell_scripts() -> list[Path]:
    return sorted(CALIB.glob("*.sh"))


@pytest.mark.parametrize("script", _scripts(), ids=lambda p: p.name)
def test_no_script_globs_cells_itself(script: Path) -> None:
    """Cell discovery goes through ``_cells_io``, never through a local glob."""
    hits = [
        f"{script.name}:{i}: {line.strip()}"
        for i, line in enumerate(script.read_text().splitlines(), 1)
        if _TASK_GLOB.search(line)
    ]
    assert not hits, (
        "these lines glob the cell CSVs directly instead of calling "
        "`_cells_io.main_frame_files` / `side_frame_files`:\n  " + "\n  ".join(hits) + "\n"
        "A bare `task_*` glob matches the side frames too, and concatenating one "
        "into the main frame is silent: the metric columns land as NaN and every "
        "groupby denominator moves.  This is exactly how #3407 happened."
    )


def test_side_frame_registry_matches_the_runner() -> None:
    """Every side frame ``run_cells.py`` writes is named in ``SIDE_FRAME_SUFFIXES``."""
    written = set(re.findall(r'f"task_\{idx:04d\}(__[a-z]+)\.csv"', (CALIB / "run_cells.py").read_text()))
    declared = re.search(r"SIDE_FRAME_SUFFIXES = \(([^)]*)\)", (CALIB / "_cells_paths.py").read_text())
    assert declared, "_cells_paths.py no longer declares SIDE_FRAME_SUFFIXES as a literal tuple"
    registry = set(declared.group(1).replace('"', "").split())
    registry = {s.rstrip(",") for s in registry if s.strip(",")}
    assert written, "run_cells.py writes no side frames — has the naming convention changed?"
    missing = written - registry
    assert not missing, (
        f"run_cells.py writes {sorted(missing)} but _cells_io.SIDE_FRAME_SUFFIXES does not list them. "
        "`main_frame_files` excludes them anyway (it filters on the `__` in the stem), so nothing is "
        "miscounted — but `side_frame_files` reads this tuple, so a frame missing from it is a frame "
        "no analyzer can ask for by name."
    )
    stale = registry - written
    assert not stale, (
        f"_cells_io.SIDE_FRAME_SUFFIXES lists {sorted(stale)}, which run_cells.py no longer writes. "
        "Drop the suffix, or point this test at whatever writes it now."
    )


def test_main_frames_are_the_ones_without_a_double_underscore() -> None:
    """The structural rule ``main_frame_files`` relies on, asserted on the runner.

    ``run_cells.py`` must write exactly one ``task_NNNN.csv`` and give every
    other frame a ``__suffix``.  If a second bare-named frame is ever added,
    ``main_frame_files`` starts returning two frames per cell and every analyzer
    double-counts — so the rule is checked where it is established, not only
    where it is used.
    """
    written = re.findall(r'f"task_\{idx:04d\}(.*?)\.csv"', (CALIB / "run_cells.py").read_text())
    bare = [w for w in written if not w.startswith("__")]
    assert bare == [""], f"run_cells.py writes non-main frames without a `__` prefix: {bare}"


#: A shell line that counts or iterates cell CSVs, in any of the five spellings
#: this directory uses (``find -name``, ``ls | grep -c``, a ``for f in`` glob,
#: an inline-Python ``glob``, a ``case`` on the filename).
_SHELL_CELLS = re.compile(r"task_[^\s'\"]*\.csv|task_\*")

#: The three ways a shell line legitimately keeps only main frames: exclude on
#: ``__``, or match the digits positively (``task_[0-9][0-9][0-9][0-9].csv`` /
#: ``^task_[0-9]*\.csv$``).  A by-name list of the side frames is **not** one of
#: them, which is the point: ``launch_horizon.sh`` carried
#: ``! -name '*sweep*' ! -name '*cutdiag*' ! -name '*cutincl*'`` and so counted
#: three times the cells that existed once ``__picks`` and ``__fitq`` arrived.
_SHELL_GUARDED = re.compile(r"__|task_\[0-9\]|\^task_\[0-9\]")


@pytest.mark.parametrize("script", _shell_scripts(), ids=lambda p: p.name)
def test_shell_cell_counts_exclude_side_frames(script: Path) -> None:
    """A launcher's cell count must not include the five side frames per cell.

    Shell cannot import ``_cells_paths``, so these carry the rule inline — and
    inline is exactly where the by-name list keeps reappearing.  A status line
    reporting six times the cells that exist is how a half-finished grid reads
    as finished, and a resume-gap computation built on the same glob resubmits
    nothing.
    """
    lines = script.read_text().splitlines()
    bad = []
    for i, line in enumerate(lines):
        if not _SHELL_CELLS.search(line) or line.lstrip().startswith("#"):
            continue
        # The guard may sit on the *next* line: `for f in "$d"/task_*.csv; do`
        # followed by `case "$f" in *__*) continue;; esac` is a correct loop, and
        # a line-at-a-time check would force it to be written worse.
        if any(_SHELL_GUARDED.search(x) for x in lines[i : i + 2]):
            continue
        bad.append(f"{script.name}:{i + 1}: {line.strip()}")
    assert not bad, (
        "these shell lines match every frame in a cells/ directory, not just the "
        "main frames:\n  " + "\n  ".join(bad) + "\n"
        "Exclude side frames on the `__` (`! -name '*__*'`, `case \"$f\" in *__*)`, "
        '`"__" not in p.stem`) or match the digits positively '
        "(`-name 'task_[0-9][0-9][0-9][0-9].csv'`).  Never by listing the side "
        "frames by name — that list is what went stale in #3407."
    )


# --- The coverage sentence ---------------------------------------------------
#
# Not static: these load real (tiny) cell trees through both loaders and read
# the sentence they produce.  The bug they exist for (#3808) was invisible to
# every static check, because both halves were individually correct -- the
# renamed keys and the sentence's key table just never agreed.


def _load_cells_io():
    """Import ``_cells_io`` by path, with its own directory importable.

    The calibration scripts import each other by bare name (``from
    _cells_paths import ...``) because they run as ``python analyze_foo.py``
    from their own directory.  Added and removed around the load rather than
    left in place: ``common``, ``curves`` and friends are generic enough that
    leaving the directory importable would shadow real modules.
    """
    spec = importlib.util.spec_from_file_location("_calib_cells_io_coverage", CALIB / "_cells_io.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(CALIB))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(CALIB))
    return module


#: The columns ``_base_rows`` filters on, so a fabricated cell survives it.
_ROW = "dataset,embedder,category,t,gmm_variant,schedule,pool_variant\nd,e,c,0.5,,,max\n"


@pytest.fixture
def arm(tmp_path: Path) -> Path:
    """One arm holding a cell of each kind the loaders count separately."""
    cells = tmp_path / "results" / "cells"
    cells.mkdir(parents=True)
    (cells / "task_0000.csv").write_text(_ROW)  # readable, one base row
    (cells / "task_0001.csv").write_bytes(b"")  # zero-byte: died mid-write
    (cells / "task_0002.csv").write_bytes(b"a,b\n\xff\xfe,2\n")  # unreadable: undecodable
    (cells / "task_0003.csv").write_text(_ROW.splitlines()[0] + "\n")  # header-only: starved
    return tmp_path / "results"


def test_both_loaders_count_the_same_four_cells(arm: Path) -> None:
    """The fixture is only worth reading if both loaders see it the same way."""
    cells_io = _load_cells_io()
    _frame, by_cells = cells_io.load_cells(arm / "cells")
    _arm_frame, by_arm = cells_io.load_arm(arm)
    for prov in (by_cells, by_arm):
        assert (prov["n_files"], prov["n_read"]) == (4, 1)
        assert len(prov["zero_byte"]) == 1
        assert len(prov["unreadable"]) == 1
    assert len(by_cells["header_only"]) == 1
    assert len(by_arm["no_positive_found"]) == 1, "load_arm renamed the starved count again?"


def test_the_coverage_sentence_is_the_same_for_either_loader(arm: Path) -> None:
    """#3808: ``describe_load`` read only ``load_cells``' spellings.

    ``load_arm`` renames ``header_only`` to ``no_positive_found`` on purpose --
    a starved cell is a result, not data loss -- but the shared sentence still
    looked for the old key, so ``prov.get("header_only")`` was ``None``,
    ``len(None or ())`` was 0, and every caller on ``load_arm``'s provenance
    reported *zero* starved cells however many there were.
    """
    cells_io = _load_cells_io()
    _f1, by_cells = cells_io.load_cells(arm / "cells")
    _f2, by_arm = cells_io.load_arm(arm)
    said_by_cells = cells_io.describe_load(by_cells)
    said_by_arm = cells_io.describe_load(by_arm)
    for said in (said_by_cells, said_by_arm):
        assert "1 zero-byte" in said
        assert "1 unreadable" in said
        assert "1 starved" in said, f"the starved cell vanished from the sentence: {said!r}"
    assert said_by_cells == said_by_arm, (
        "the two loaders describe the same four cells differently:\n"
        f"  load_cells: {said_by_cells}\n  load_arm:   {said_by_arm}\n"
        "`describe_load` is the sentence that makes 'N of M cells' mean the same thing "
        "in two reports; it has to read both loaders' key spellings."
    )


def test_the_starved_count_does_not_read_as_a_loss(arm: Path) -> None:
    """It is the extreme of the regime these studies measure, not a hole in them."""
    cells_io = _load_cells_io()
    _frame, prov = cells_io.load_arm(arm)
    said = cells_io.describe_load(prov)
    starved = said.split(", ")[-1]
    assert starved.startswith("1 starved"), said
    assert "not data loss" in starved, (
        f"the starved count reads like the loss counts beside it: {starved!r}.  "
        "That distinction is the whole reason `load_arm` renamed the key."
    )


@pytest.mark.parametrize("loader", ["load_cells", "load_arm"])
def test_every_key_a_loader_writes_is_one_the_sentence_knows(arm: Path, loader: str) -> None:
    """The guard against the *next* rename, in either direction.

    A loader that invents a key outside ``DESCRIBED_KEYS`` drops out of the
    coverage line silently -- the count does not go wrong, it goes *missing*,
    and the sentence still reads like a complete accounting.  Checked against
    what the loaders actually return rather than against a hand-kept list, so
    a third spelling fails here rather than in six studies' reports.
    """
    cells_io = _load_cells_io()
    prov = (cells_io.load_cells(arm / "cells") if loader == "load_cells" else cells_io.load_arm(arm))[1]
    #: The scalar bookkeeping the sentence names positionally, not by key.
    counted_elsewhere = {"cells_dir", "n_files", "n_read", "n_rows", "n_rows_all"}
    unnamed = sorted(
        key
        for key, value in prov.items()
        if key not in counted_elsewhere and isinstance(value, list) and key not in cells_io.DESCRIBED_KEYS
    )
    assert not unnamed, (
        f"`{loader}` reports {unnamed}, which `describe_load` does not look for, so those cells "
        "vanish from every study's coverage line.  Add the spelling to `LOSS_KEYS` or "
        "`STARVED_KEYS` in `_cells_io.py` in the same commit as the rename (#3808)."
    )
