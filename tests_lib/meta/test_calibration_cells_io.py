"""The calibration analyzers must read their cells through one loader.

Static text checks over ``scripts/experiments/calibration/`` — nothing here
imports the analyzers (they need pandas, matplotlib and a results tree), and
nothing here tests shipped ``vtsearch``/``vtscore`` behaviour, which is why it
lives in the ``meta`` group.

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

import re
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
